"""Message-level SFT trajectory dump.

Slime's :meth:`TrajectoryManager.get_trajectory` is token-only: with
``output_ids=[]`` (messages mode without a tokenizer) it emits no ``Sample``.
This module mirrors that leaf-walk but linearizes each leaf chain into a
``messages`` list instead of a token sequence, producing one ``Sample`` per
routing leaf whose ``prompt`` IS the full OpenAI/HF messages conversation.

The conversion from Anthropic content-blocks to OpenAI messages reuses the
pure-function logic from :mod:`anyharness.adapters.anthropic`
(:func:`_translate_messages` + :func:`_tools_to_chat_tools`), copied here to
avoid a circular import (``adapters`` imports ``trajectory``, which this module
augments). The tree itself is left untouched: ``record_turn`` already built it,
we only linearize.

Output schema (TRL / LLaMA-Factory / axolotl-compatible):
* ``prompt``  -> the messages list (system/user/assistant/tool).
* ``response`` -> the last generated assistant message's content text, or a
  readable summary when that turn is tool-call/thinking-only (never blank).
* ``response_length`` -> count of generated (trainable) assistant turns in
  this leaf's chain.
* ``metadata`` -> truncated / use_tool / ill_formed / session_id / tools
  (when the model saw tool definitions) + caller extra.
* ``tokens`` / ``loss_mask`` left empty/None; trainers derive the mask via
  ``assistant_only_loss``.

Tool-call ids: the persisted ``manager_message`` drops the wire-only tool-call
id (slime matches leaves by dict equality), so HF chat templates cannot pair a
tool result with its call. :func:`_pair_tool_call_ids` re-pairs them at dump
time (synthesized ids + matching ``tool_call_id``) without touching the tree.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .types import Sample


# ===========================================================================
# Anthropic wire -> OpenAI/HF messages  (pure functions, no adapter import)
# ===========================================================================


def _flatten_content(c: Any) -> str:
    """Flatten a wire content value (Anthropic or OpenAI blocks) to a string.

    Mirrors :func:`anyharness.adapters.common.flatten_content` for the
    assistant-text extraction path. Multiple blocks are joined with ``"\\n\\n"``
    (GAP B: matches the production GLM-5.2 converter's
    :func:`export_covered_claude_events_to_glm52_sft.content_part_to_text`, which
    is what the model saw at rollout for a multi-block system prompt such as
    Claude Code's ``[{type:text,...,cache_control:{...}}]``).
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t in {"text", "input_text", "output_text"}:
            parts.append(b.get("text", ""))
        elif t == "tool_result":
            parts.append(_flatten_content(b.get("content")))
        elif t in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in b:
            parts.append(_flatten_content(b.get("content")))
        elif "text" in b:
            parts.append(str(b.get("text") or ""))
    return "\n\n".join(p for p in parts if p)


def _looks_anthropic(msgs: list[dict]) -> bool:
    """True if any message carries Anthropic-shape content (a list of typed
    blocks, or a ``tool_result``/``thinking``/``tool_use`` block).

    OpenAI/HF messages use a plain ``content`` string (and ``tool_calls`` /
    ``tool_call_id`` keys); Anthropic uses ``content: [{"type": ...}]``.
    """
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") in {
                    "text",
                    "thinking",
                    "tool_use",
                    "tool_result",
                }:
                    return True
                # a list-of-blocks user message with non-typed dicts still reads
                # as Anthropic shape
            if c:
                return True
    return False


def anthropic_wire_to_sft(
    anthropic_messages: list[dict], tools: list[dict] | None = None
) -> tuple[list[dict], list[dict] | None]:
    """Convert Anthropic content-block messages (+ optional tools) to OpenAI/HF
    SFT messages (+ tools schema).

    Rules (faithful copy of
    :func:`anyharness.adapters.anthropic._translate_messages` +
    :func:`_tools_to_chat_tools`):

    * ``system`` (top-level or role) -> ``{"role":"system","content":str}``.
    * assistant ``text`` block -> message ``content`` string.
    * assistant ``thinking`` block -> ``reasoning_content``.
    * assistant ``tool_use`` block -> a ``tool_calls`` entry whose
      ``function.arguments`` is a **dict** (not a JSON string); the wire-only
      ``id`` is dropped.
    * user ``tool_result`` block -> a ``{"role":"tool", ...}`` message carrying
      ``tool_call_id`` (so trainers can pair it with the originating call) and,
      when the preceding assistant turn declared the tool, ``name`` (mirrors
      :func:`export_covered_claude_events_to_glm52_sft.anthropic_request_messages_to_glm52`).
    * top-level ``tools[].input_schema`` -> ``parameters`` wrapped in
      ``{type:function, function:{name,description,parameters}}``.

    Tool-name pairing (GAP A): a FIRST pass collects ``tool_name_by_id`` (tool_use
    ``id`` -> tool ``name``) from every assistant ``tool_use`` block; the emit
    pass then sets ``name`` on each tool message from that map. This keys on the
    ORIGINAL Anthropic ``toolu_`` id -- the same id the production converter
    pairs by. :func:`_pair_tool_call_ids` runs AFTER this and only back-fills
    MISSING ids; in the Anthropic path the tool_use-derived tool_calls have no id
    but the tool message carries the original ``tool_call_id``, which
    :func:`_pair_tool_call_ids` reuses on the matching call (its "reuse an
    existing id on the matching tool message" branch), so the original id (and
    thus the ``name`` we attached to it) is preserved end-to-end and stays
    correctly paired.
    """
    sft_messages: list[dict] = []

    # GAP A, pass 1: collect tool_use id -> name so tool-result messages can
    # carry the tool name (mirrors production anthropic_request_messages_to_glm52).
    tool_name_by_id: dict[str, str] = {}
    for m in anthropic_messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                call_id = b.get("id")
                name = b.get("name")
                if call_id and isinstance(name, str):
                    tool_name_by_id[call_id] = name

    for m in anthropic_messages:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "system":
            sft_messages.append({"role": "system", "content": _flatten_content(content)})
            continue
        if role == "user":
            blocks = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": _flatten_content(content)}]
            )
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tool_call_id = b.get("tool_use_id") or b.get("tool_call_id") or ""
                    tool_msg: dict[str, Any] = {
                        "role": "tool",
                        "content": _flatten_content(b.get("content")),
                        "tool_call_id": tool_call_id,
                    }
                    name = tool_name_by_id.get(tool_call_id)
                    if name:
                        tool_msg["name"] = name
                    sft_messages.append(tool_msg)
                elif isinstance(b, dict) and b.get("type") == "text":
                    sft_messages.append({"role": "user", "content": b.get("text", "")})
                else:
                    sft_messages.append({"role": "user", "content": _flatten_content(b)})
            continue
        if role == "assistant":
            texts, thinkings, tcs = [], [], []
            blocks = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": _flatten_content(content)}]
            )
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    # arguments stays a dict (not a JSON string); wire-only id dropped
                    tcs.append(
                        {
                            "type": "function",
                            "function": {
                                "name": b.get("name", "tool"),
                                "arguments": b.get("input") or {},
                            },
                        }
                    )
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            sft_messages.append(mo)
            continue
        if role == "tool":
            # already OpenAI-shape tool message; preserve tool_call_id if present
            tm = {"role": "tool", "content": _flatten_content(content)}
            if m.get("tool_call_id") is not None:
                tm["tool_call_id"] = m["tool_call_id"]
            sft_messages.append(tm)

    sft_tools = _tools_to_chat_tools(tools)
    return sft_messages, sft_tools


def _tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:
    """Anthropic tools -> OpenAI/HF function tool schema.

    ``input_schema`` is renamed to ``parameters`` and wrapped in
    ``{type:function, function:{name,description,parameters}}``. A missing
    schema defaults to ``{"type":"object","additionalProperties":True}``
    (GAP C: matches :func:`export_covered_claude_events_to_glm52_sft.anthropic_tool_to_glm52_def`,
    i.e. what the model saw at rollout). Faithful copy of
    :func:`anyharness.adapters.anthropic._tools_to_chat_tools` modulo the
    default-schema alignment.
    """
    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        params = t.get("input_schema") or t.get("parameters") or {
            "type": "object",
            "additionalProperties": True,
        }
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return ts or None


# ===========================================================================
# Message-level leaf-walk  (mirrors TrajectoryManager.get_trajectory)
# ===========================================================================


def _last_assistant_text(messages: list[dict]) -> str:
    """Content text of the last assistant message in ``messages``.

    FIX 4: a pure tool-call or thinking-only assistant turn has no text
    content (``content == ""``); the SFT target must not be blank, so we fall
    back to a readable summary of the turn's tool calls (or ``"<tool_call>"``
    when there are none) so the response is never empty.
    """
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else _flatten_content(content)
        if text:
            return text
        # No text content: synthesize a readable target from the tool calls so
        # the SFT record isn't blank (tool-call-only / thinking-only turns).
        tcs = m.get("tool_calls") or []
        if tcs:
            parts = []
            for tc in tcs:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = fn.get("name", "tool")
                args = fn.get("arguments", {})
                parts.append(f"{name}({json.dumps(args, ensure_ascii=False)})" if args else name)
            return " ".join(parts)
        # thinking-only turn with no tool calls and no text
        if m.get("reasoning_content"):
            return "<thinking>"
        return "<tool_call>"
    return ""


def _pair_tool_call_ids(messages: list[dict]) -> None:
    """Ensure every tool_calls entry has a unique ``id`` and every tool message
    has a matching ``tool_call_id`` (FIX 2, dump-time only).

    Slime deliberately drops the wire-only tool-call id from the persisted
    ``manager_message`` so a generated leaf compares equal (dict ``==``) to the
    same turn replayed as history (see ``tool_call_dict``). But an HF chat
    template pairs a tool result with its originating call by id, so the dumped
    record needs them. We synthesize ids here and re-pair tool messages to them,
    *without* touching the tree (which already matched without ids).

    Pairing order: the i-th tool message following an assistant turn is paired
    with the i-th tool_call of that turn (the order tools arrive matches the
    order they were requested, as Anthropic guarantees). If a tool message
    already carries a ``tool_call_id`` (e.g. from a caller-fed OpenAI-shaped
    chain), it is left in place and its id is re-used on the matching call.
    """
    counter = 0

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"call_{counter}"

    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if not isinstance(tcs, list) or not tcs:
            continue
        # collect tool messages that follow this assistant turn, up to the next
        # assistant turn (those belong to this call set)
        tool_msgs: list[dict] = []
        for j in range(i + 1, len(messages)):
            nm = messages[j]
            if not isinstance(nm, dict):
                continue
            if nm.get("role") == "assistant":
                break
            if nm.get("role") == "tool":
                tool_msgs.append(nm)

        for k, tc in enumerate(tcs):
            if not isinstance(tc, dict):
                continue
            existing = tc.get("id")
            # reuse an existing id on the matching tool message if present
            if existing is None and k < len(tool_msgs):
                existing = tool_msgs[k].get("tool_call_id")
            if existing is None:
                existing = _next_id()
            tc["id"] = existing
            # back-fill the matching tool message (by position)
            if k < len(tool_msgs):
                tool_msgs[k]["tool_call_id"] = existing
                # mirror the production converter's tool_name_by_id: set the tool
                # message's `name` from the originating tool_call (HF/LLaMA-Factory
                # templates that render the tool name on the result expect it).
                fn = tc.get("function") if isinstance(tc, dict) else None
                tname = fn.get("name") if isinstance(fn, dict) else None
                if tname and not tool_msgs[k].get("name"):
                    tool_msgs[k]["name"] = tname


def get_trajectory_messages(
    manager: Any,
    sid: str,
    *,
    base_sample: Sample,
    reward: float = 0.0,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Linearize ``manager``'s per-sid tree into message-level ``Sample`` objects
    and consume the session.

    Mirrors :meth:`TrajectoryManager.get_trajectory`: walk every routing leaf
    (``root.leaves()``, skipping the dummy root), follow ``path_from_root()``,
    and for each leaf chain collect ``node.message`` for every node whose
    message is set into an ordered ``messages`` list. One ``Sample`` is emitted
    per leaf. The sid is popped from ``manager._trees`` / ``manager._turn_count``
    afterwards (idempotent: a second call returns ``[]``).

    Parity with the token path (:meth:`TrajectoryManager.get_trajectory`):

    * ``response_length`` counts the chain's generated assistant turns (nodes
      with ``turn is not None``). Each leaf is linearized independently, so a
      turn shared by sibling leaves counts once per leaf-Sample -- correct for
      "response_length = turns in this conversation". The token path's
      first-wins ``response_trained`` avoids *training* a shared turn twice; it
      does not change the per-leaf turn count.
    * ``metadata["ill_formed"]`` is set when any generated turn was ill-formed.
    * a leaf with no generated assistant turn (all routing-only, or every
      generated assistant demoted by ``_try_merge_assistant_rewrite``) emits no
      Sample -- mirroring the token path's ``has_trained_response`` gate.

    ``metadata["tools"]`` is surfaced from the chain (first non-None
    ``node.metadata["tools"]``) or from ``extra_metadata["tools"]`` so the SFT
    record keeps the tool definitions the model saw at rollout.
    """
    root = manager._trees.get(sid)
    if root is None:
        return []

    # tools the model saw: prefer the caller-supplied schema (set by the
    # adapter from its per-sid capture), else the first non-None tools stashed
    # on a chain node by record_turn metadata.
    session_tools = None
    if isinstance(extra_metadata, dict) and extra_metadata.get("tools") is not None:
        session_tools = extra_metadata["tools"]

    samples: list[Sample] = []
    for routing_leaf in root.leaves():
        if routing_leaf.is_root:
            continue
        chain = routing_leaf.path_from_root()

        raw_messages: list[dict] = []
        for node in chain:
            if node.message is None:
                continue
            raw_messages.append(node.message)

        if not raw_messages:
            continue

        # The tree's manager_message is already OpenAI/HF shape (assistant
        # tool_calls carry dict args, thinking -> reasoning_content, tool
        # results live under role:tool). When a caller fed raw Anthropic
        # content-blocks instead, normalize them to the same OpenAI shape.
        if _looks_anthropic(raw_messages):
            sft_messages, _ = anthropic_wire_to_sft(raw_messages, tools=None)
        else:
            sft_messages = [copy.deepcopy(m) for m in raw_messages]

        # Parity FIX 2: pair every tool_calls entry with a unique id and every
        # tool message with a matching tool_call_id, so HF chat templates can
        # associate a tool result with its originating call. Slime deliberately
        # drops the id from the persisted manager_message (dict-equality tree
        # matching), so we re-pair at dump time only.
        _pair_tool_call_ids(sft_messages)

        generated_turns = [n for n in chain if n.turn is not None]

        # Parity FIX 3b: a leaf with no generated assistant turn emits no Sample
        # -- mirrors the token path's has_trained_response gate. This covers
        # both an all-routing-only leaf and one where every generated assistant
        # was demoted by _try_merge_assistant_rewrite (turn -> None). It also
        # drops a degenerate leaf whose response_message was None: such a node
        # keeps turn but has message=None, so it carries nothing to train on.
        generated_with_message = [n for n in generated_turns if n.message is not None]
        if not generated_with_message:
            continue

        last_finish = generated_with_message[-1].turn.finish_reason
        truncated = last_finish == "length"
        use_tool = any(
            bool((m.get("tool_calls"))) for m in sft_messages if isinstance(m, dict) and m.get("role") == "assistant"
        )
        ill_formed = any(n.turn.ill_formed for n in generated_with_message)

        # tools: caller-supplied schema wins; else first non-None on the chain.
        tools = session_tools
        if tools is None:
            for n in chain:
                t = n.metadata.get("tools") if isinstance(n.metadata, dict) else None
                if t is not None:
                    tools = t
                    break

        md = dict(extra_metadata or {})
        md["truncated"] = truncated
        md["use_tool"] = use_tool
        md["ill_formed"] = ill_formed
        md["session_id"] = sid
        if tools is not None:
            md["tools"] = tools

        s = Sample(
            index=base_sample.index,
            group_index=base_sample.group_index,
            rollout_id=base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index,
            prompt=sft_messages,
            response=_last_assistant_text(sft_messages),
            tokens=[],
            loss_mask=None,
            response_length=len(generated_with_message),
            reward=reward,
            status=Sample.Status.COMPLETED,
            metadata=md,
        )
        samples.append(s)

    # idempotent consume, mirroring get_trajectory
    manager._trees.pop(sid, None)
    manager._turn_count.pop(sid, None)
    return samples


__all__ = ["get_trajectory_messages", "anthropic_wire_to_sft"]
