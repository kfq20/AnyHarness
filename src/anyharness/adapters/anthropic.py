"""Anthropic Messages adapter for agent rollouts — the AnyHarness core.

AnyHarness is the layer between a harness (Claude Code)
and the sampling endpoint / API server. This adapter is its core: it exposes
``/v1/messages`` (and ``/v1/messages/count_tokens``) downstream to the harness,
forwards each turn upstream to a pluggable backend (sglang ``/generate`` with
per-token logprobs, or an arbitrary messages-API — see ``UPSTREAM_MODE``),
and feeds every turn into a shared :class:`~anyharness.trajectory.TrajectoryManager`
keyed by session id. ``finish_session(sid)`` drains a session's trajectory tree
into a list of :class:`~anyharness.types.Sample`.

"Harness in the loop" = the adapter sits *between* the harness and the model,
intercepting every ``/v1/messages`` call rather than fire-and-forget: it
captures the conversation as a tree, tolerates Claude Code's tool-result
trimming (``ToolMessage`` excludes ``content`` from ``==`` so a blanked
tool_result replay still matches the stored node), and linearizes the tree
into training samples at session end.

The per-sid tree inside TrajectoryManager handles sub-agent and compaction
patterns automatically: any divergence in the prompt prefix forks into a new
leaf, so we do not track explicit chains here.

This module mirrors slime.agent.adapters.openai; the section layout (adapter
class -> translation -> reply building -> request framing) is shared between
them. See BaseAdapter for the hooks to fill.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from typing import Any

from aiohttp import web

from anyharness.adapters.common import (
    BaseAdapter,
    Reply,
    flatten_content,
    manager_finish_reason,
    sid_from_bearer,
    tool_call_dict,
)
from anyharness.parsing import ParsedModelOutput

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter: wire translation and reply
    framing only; the turn machinery is inherited from BaseAdapter."""

    logger = logger
    log_prefix = "anthropic_adapter"
    max_token_keys = ("max_tokens",)
    stop_keys = ("stop_sequences",)

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._run_turn)
        app.router.add_post("/v1/messages/count_tokens", _count_tokens)
        # OpenAI chat-completions downstream: lets harnesses that speak
        # chat-completions (OpenAI Agents SDK, smolagents, ...) connect.
        app.router.add_post("/v1/chat/completions", self._run_chat_turn)
        # OpenAI Responses API downstream: lets harnesses that speak the
        # Responses API (Codex CLI, OpenAI Agents SDK default mode) connect.
        app.router.add_post("/v1/responses", self._run_responses_turn)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request)

    def _preprocess_body(self, body: dict) -> None:
        _fold_mid_list_system_into_user(body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        translated = _translate_messages(body.get("messages") or [], body.get("system"))
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    def _build_reply(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        blocks, stop_reason, manager_message = _build_reply_parts(parsed, raw_finish)
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=(blocks, stop_reason),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        blocks, stop_reason = reply.wire
        if stream:
            return await _render_stream(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_render_response(body, blocks, stop_reason, in_tok, out_tok))

    # -- chat-completions downstream (OpenAI wire format) --------------------

    async def _run_chat_turn(self, request: web.Request) -> web.StreamResponse:
        """Handle /v1/chat/completions: the OpenAI chat-completions wire format.

        Reuses _run_turn with downstream_format="chat" so translate + respond
        use the OpenAI shape (the tree's hub format) directly, without the
        Anthropic round-trip.
        """
        return await self._run_turn(request, downstream_format="chat")

    def _translate_chat(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Translate an OpenAI chat-completions request to the tree's hub format.

        chat-completions messages are ALREADY the OpenAI shape the tree stores
        ({role, content, tool_calls, tool_call_id}), so we pass them through
        with only ToolMessage wrapping on tool-role messages (so tool_result
        content trimming doesn't fork, same as the Anthropic path).
        """
        messages = body.get("messages") or []
        translated: list[dict] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "tool":
                translated.append(ToolMessage(role="tool", content=m.get("content", ""),
                                             tool_call_id=m.get("tool_call_id", "")))
            else:
                translated.append(m)
        tools_schema = body.get("tools")  # already OpenAI function-tool shape
        return translated, tools_schema

    async def _respond_chat(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        """Render the reply as an OpenAI chat-completions response.

        The manager_message IS already the OpenAI assistant-message shape
        ({role, content, tool_calls, reasoning_content}), so we wrap it in the
        choices envelope. finish_reason maps from the internal finish to the
        OpenAI vocabulary (tool_calls -> "tool_calls", length -> "length").
        """
        mm = reply.manager_message
        fr = reply.finish_reason
        openai_finish = {"tool_calls": "tool_calls", "length": "length"}.get(fr, "stop")
        msg = {"role": "assistant", "content": mm.get("content", "")}
        if mm.get("tool_calls"):
            # The tree stores `arguments` as a dict (see tool_call_dict), but the
            # OpenAI chat wire format requires a JSON *string* — a client that
            # json.loads() it (pi, the OpenAI SDK) otherwise sees no arguments at
            # all and rejects the call. `id` is likewise required for the client to
            # pair the following tool result. Matches what _respond_responses does.
            wire_tcs: list[dict] = []
            for i, tc in enumerate(mm["tool_calls"]):
                if not isinstance(tc, dict):
                    continue
                tc2 = dict(tc)
                fn = tc2.get("function")
                if isinstance(fn, dict):
                    fn = dict(fn)
                    if isinstance(fn.get("arguments"), dict):
                        fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                    tc2["function"] = fn
                tc2.setdefault("type", "function")
                if not tc2.get("id"):
                    tc2["id"] = f"call_{secrets.token_hex(8)}"
                wire_tcs.append(tc2)
            msg["tool_calls"] = wire_tcs
        if mm.get("reasoning_content"):
            msg["reasoning_content"] = mm["reasoning_content"]
        resp = {
            "id": f"chatcmpl_{secrets.token_hex(12)}",
            "object": "chat.completion",
            "model": body.get("model", "slime-actor"),
            "choices": [{"index": 0, "message": msg, "finish_reason": openai_finish}],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok, "total_tokens": in_tok + out_tok},
        }
        if stream:
            # minimal SSE: one chunk with the delta, then [DONE]
            chunk = {"id": resp["id"], "object": "chat.completion.chunk", "model": resp["model"],
                    "choices": [{"index": 0, "delta": msg, "finish_reason": None}]}
            return await _sse_stream(request, [chunk, {"choices": [{"index": 0, "delta": {}, "finish_reason": openai_finish}]}])
        return web.json_response(resp)

    # -- responses downstream (OpenAI Responses API wire format) ------------

    async def _run_responses_turn(self, request: web.Request) -> web.StreamResponse:
        """Handle /v1/responses: the OpenAI Responses API format (Codex CLI, Agents SDK)."""
        return await self._run_turn(request, downstream_format="responses")

    def _translate_responses(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Translate a Responses API request to the tree's hub format.

        Responses API ``input`` is either a string or a list of items (messages
        + function_call + function_call_output). We convert to the OpenAI
        chat-completions message shape the tree stores.

        Item types in input:
        - {type:"message", role, content:[{type:"input_text"/"output_text", text}]}
        - {type:"function_call", name, arguments(str), call_id}
        - {type:"function_call_output", call_id, output(str)}
        """
        raw_input = body.get("input")
        if isinstance(raw_input, str):
            return [{"role": "user", "content": raw_input}], body.get("tools")

        translated: list[dict] = []
        # Reasoning arrives as its own item just before the assistant turn it
        # belongs to; held here until that turn is built. See the `reasoning`
        # branch below.
        pending_reasoning = ""
        instructions = body.get("instructions")
        if instructions:
            translated.append({"role": "system", "content": instructions})

        for item in (raw_input or []):
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            # `type` is optional for message items: the Responses API accepts bare
            # {role, content} entries and that is what the OpenAI SDK and Codex CLI
            # actually send. Treat a typeless item carrying a role as a message --
            # without this the whole prompt is silently dropped and the model is
            # asked to answer nothing.
            if itype is None and item.get("role"):
                itype = "message"
            if itype == "reasoning":
                # Reasoning is replayed as its OWN top-level item, ahead of the
                # assistant message (or function_call) it belongs to. The generated
                # turn stores it as `reasoning_content` ON that assistant message,
                # so it has to be re-attached here or the replay will not compare
                # equal to the stored node and the tree forks on every thinking
                # turn. Codex also sends `encrypted_content` (opaque, not
                # round-trippable); the plaintext summary is what we can recover.
                texts = [
                    p.get("text", "") for p in (item.get("summary") or [])
                    if isinstance(p, dict)
                ]
                pending_reasoning = "".join(texts)
                continue
            if itype == "message":
                role = item.get("role", "user")
                content_parts = item.get("content") or []
                # content is a list of {type:"input_text"/"output_text", text}
                if isinstance(content_parts, str):
                    msg = {"role": role, "content": content_parts}
                else:
                    texts = [p.get("text", "") for p in content_parts if isinstance(p, dict)]
                    msg = {"role": role, "content": "".join(texts)}
                # OpenAI Responses uses "developer" for the system prompt; the hub
                # format (and chat templates) use "system", so normalize.
                if msg["role"] == "developer":
                    msg["role"] = "system"
                if pending_reasoning and role == "assistant":
                    msg["reasoning_content"] = pending_reasoning
                    pending_reasoning = ""
                translated.append(msg)
            elif itype == "function_call":
                # assistant tool call — arguments is a JSON string
                args = item.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except (ValueError, TypeError):
                        args = {"_raw": args}
                # The wire-only call id is dropped here on purpose, exactly as the
                # messages-mode replay does via tool_call_dict: the tree matches
                # history by dict equality, so a generated assistant turn (whose id
                # _build_reply_parts_from_blocks also drops) must compare equal to
                # this replayed echo. Keeping the id on one side only would fork the
                # tree on every tool call.
                tc = tool_call_dict(item.get("name", ""), args if isinstance(args, dict) else {})
                # The Responses API splits one assistant turn across items: the
                # text is a `message`, each tool call a sibling `function_call`.
                # The chat shape the tree stores puts text + tool_calls on a SINGLE
                # assistant message, which is what the generated node looks like --
                # so fold this call into the immediately preceding assistant
                # message instead of emitting a second turn (and fold sibling
                # function_calls together into one multi-call turn).
                prev = translated[-1] if translated else None
                if isinstance(prev, dict) and prev.get("role") == "assistant" and not isinstance(prev, ToolMessage):
                    prev.setdefault("tool_calls", []).append(tc)
                    if pending_reasoning and not prev.get("reasoning_content"):
                        prev["reasoning_content"] = pending_reasoning
                    pending_reasoning = ""
                else:
                    tc_msg: dict[str, Any] = {"role": "assistant", "content": "", "tool_calls": [tc]}
                    if pending_reasoning:
                        tc_msg["reasoning_content"] = pending_reasoning
                        pending_reasoning = ""
                    translated.append(tc_msg)
            elif itype == "function_call_output":
                # tool result — output is a string
                translated.append(ToolMessage(role="tool",
                    content=item.get("output", ""),
                    tool_call_id=item.get("call_id") or item.get("id") or ""))

        # tools in Responses API use the same function shape
        return translated, body.get("tools")

    async def _respond_responses(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        """Render the reply as an OpenAI Responses API response.

        The manager_message (chat shape) is converted to Responses output items:
        - text content -> ResponseOutputMessage with output_text parts
        - tool_calls -> ResponseFunctionToolCall items
        """
        mm = reply.manager_message
        fr = reply.finish_reason
        resp_id = f"resp_{secrets.token_hex(12)}"
        model = body.get("model", "slime-actor")

        output: list[dict] = []
        msg_item: dict[str, Any] = {"id": f"msg_{secrets.token_hex(8)}", "type": "message", "role": "assistant", "status": "completed"}
        content_parts = []
        if mm.get("reasoning_content"):
            # reasoning as a separate item (simplified — not full reasoning API)
            output.append({"id": f"rs_{secrets.token_hex(8)}", "type": "reasoning", "summary": [{"type": "summary_text", "text": mm["reasoning_content"]}]})
        if mm.get("content"):
            content_parts.append({"type": "output_text", "text": mm["content"]})
        msg_item["content"] = content_parts
        output.append(msg_item)

        # function_call items from tool_calls
        for tc in mm.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            output.append({"id": f"fc_{secrets.token_hex(8)}", "type": "function_call",
                "call_id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})

        status = "completed" if fr not in ("length",) else "incomplete"
        resp = {
            "id": resp_id, "object": "response", "model": model,
            "output": output, "status": status,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok},
        }
        if not stream:
            return web.json_response(resp)
        return await _responses_sse_stream(request, resp)


# --- Translation (Anthropic wire -> chat-template messages) ---


class ToolMessage(dict):
    """A ``role:tool`` message whose ``content`` is excluded from ``==``.

    Claude Code reclaims tokens by blanking a large ``tool_result`` (e.g. an
    image) to ``""`` on a later turn and restoring it later still. slime's tree
    matches history by full dict equality (``child.message == msg``), so a blank
    replay would NOT equal the stored real-content node -> spurious fork (one
    extra Sample carrying a near-duplicate conversation).

    This subclass compares equal regardless of ``content`` (role + tool_call_id
    + name still must match), while ``content`` is still read by subscript and
    ``json.dumps`` — so the SFT dump keeps the real tool result. Both the stored
    node and the replay go through ``_translate_messages``, so both are
    ``ToolMessage`` and compare equal. (Confirmed: only ~1/4118 real captures
    blank tool results; assistant rewrites are already handled by
    ``_try_merge_assistant_rewrite``.)
    """

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict):
            return NotImplemented
        return {k: v for k, v in self.items() if k != "content"} == {
            k: v for k, v in other.items() if k != "content"
        }

    __hash__ = None  # mutable; unhashable like dict is fine (dict already is)


def _translate_messages(msgs: list[dict], system: Any) -> list[dict]:
    """Anthropic messages + system -> chat-template messages. Pure function."""
    translated: list[dict] = []
    if system:
        translated.append({"role": "system", "content": flatten_content(system)})
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    # ToolMessage: content excluded from == so a blanked
                    # tool_result replay matches the stored real-content node.
                    translated.append(ToolMessage(role="tool", content=flatten_content(b.get("content"))))
                elif isinstance(b, dict) and b.get("type") == "text":
                    translated.append({"role": "user", "content": b.get("text", "")})
                else:
                    translated.append({"role": "user", "content": flatten_content(b)})
        elif role == "assistant":
            texts, thinkings, sigs, tcs = [], [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                    sig = b.get("signature")
                    if sig:
                        sigs.append(sig)
                elif b.get("type") == "tool_use":
                    # drop the wire-only id; tool_call_dict keeps arguments a dict
                    tcs.append(tool_call_dict(b.get("name", "tool"), b.get("input")))
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if sigs:
                # carry the signature so this replayed echo compares equal (dict
                # equality) to the generated leaf, which also stores it --
                # otherwise the tree forks on every thinking turn.
                mo["thinking_signature"] = "".join(sigs)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
        elif role == "system":
            translated.append({"role": "system", "content": flatten_content(content)})
    return translated


def hub_to_anthropic_messages(translated: list[dict]) -> tuple[list[dict], str | None]:
    """Inverse of :func:`_translate_messages`: hub (chat) shape -> Anthropic messages.

    Needed when the downstream wire format is NOT Anthropic (``/v1/chat/completions``
    or ``/v1/responses``) but ``UPSTREAM_MODE=messages``. The raw downstream body
    cannot be forwarded in that case: a Responses body carries ``input`` where
    ``/v1/messages`` requires ``messages`` (hard 400), and a chat body carries
    OpenAI-shaped ``tools``/``tool_calls`` that an Anthropic upstream does not read.

    Returns ``(messages, system)`` where ``system`` is the concatenated system text
    (Anthropic takes it as a top-level field, not a message).

    Two shape rules the Anthropic API enforces that the hub shape does not:

    * ``tool_result`` blocks must carry a ``tool_use_id`` matching a ``tool_use``
      in the preceding assistant turn. :func:`tool_call_dict` deliberately drops
      wire ids (they would fork the trajectory tree), so ids are re-synthesized
      positionally here — ``call_0``, ``call_1``, ... in emission order — and
      consumed by the following tool messages in the same order.
    * consecutive same-role messages must be merged into one message with
      multiple content blocks.
    """
    system_parts: list[str] = []
    msgs: list[dict] = []
    # tool_use ids emitted by the most recent assistant turn, awaiting pairing
    # with the tool messages that follow it.
    pending_ids: list[str] = []
    counter = 0

    def _append(role: str, blocks: list[dict]) -> None:
        """Append blocks, merging into the previous message when the role repeats."""
        if not blocks:
            return
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"].extend(blocks)
        else:
            msgs.append({"role": role, "content": blocks})

    for m in translated:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            text = flatten_content(m.get("content"))
            if text:
                system_parts.append(text)
        elif role == "tool":
            # Anthropic carries tool results as a user-role tool_result block.
            tool_use_id = pending_ids.pop(0) if pending_ids else f"call_{counter}"
            if not pending_ids:
                counter += 1
            _append("user", [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": flatten_content(m.get("content")),
            }])
        elif role == "assistant":
            blocks: list[dict] = []
            reasoning = m.get("reasoning_content")
            if reasoning:
                blocks.append({"type": "text", "text": flatten_content(reasoning)})
            text = flatten_content(m.get("content"))
            if text:
                blocks.append({"type": "text", "text": text})
            pending_ids = []
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw_arguments": args}
                tid = tc.get("id") or f"call_{counter}"
                counter += 1
                pending_ids.append(tid)
                blocks.append({
                    "type": "tool_use",
                    "id": tid,
                    "name": fn.get("name", "tool"),
                    "input": args if isinstance(args, dict) else {},
                })
            # An assistant turn must be non-empty to be a legal Anthropic message.
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            _append("assistant", blocks)
        else:
            _append("user", [{"type": "text", "text": flatten_content(m.get("content"))}])

    # Anthropic requires the first message to be user-role.
    if msgs and msgs[0]["role"] == "assistant":
        msgs.insert(0, {"role": "user", "content": [{"type": "text", "text": ""}]})
    return msgs, ("\n\n".join(system_parts) or None)


def chat_tools_to_anthropic_tools(tools_schema: list[dict] | None) -> list[dict] | None:
    """Inverse of :func:`_tools_to_chat_tools`: OpenAI function tools -> Anthropic tools."""
    if not tools_schema:
        return None
    out: list[dict] = []
    for t in tools_schema:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if "function" in t else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or fn.get("input_schema")
            or {"type": "object", "properties": {}},
        })
    return out or None


def _tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tools to tokenizer chat-template tool schema."""
    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return ts or None


# --- Reply building: parsed output -> Anthropic blocks + manager_message ---


def _build_reply_parts(
    parsed: ParsedModelOutput,
    finish: str,
) -> tuple[list[dict], str, dict[str, Any]]:
    """Return (anthropic blocks, wire stop_reason, manager_message).

    The tool_calls inside manager_message use canonical args (tool_call_dict) so
    this assistant turn compares equal (dict equality) to the same turn replayed
    as history on the next request.
    """
    blocks: list[dict] = []
    if parsed.reasoning:
        blocks.append({"type": "thinking", "thinking": parsed.reasoning})
    if parsed.text:
        blocks.append({"type": "text", "text": parsed.text})

    manager_tcs: list[dict] = []
    for tu in parsed.tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        # tu_id is wire-only; tool_call_dict drops it so the leaf matches its echo
        manager_tcs.append(tool_call_dict(tu["name"], tu.get("input")))

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if parsed.tool_uses:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    manager_message: dict[str, Any] = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        manager_message["reasoning_content"] = parsed.reasoning
    if manager_tcs:
        manager_message["tool_calls"] = manager_tcs

    return blocks, stop_reason, manager_message


def _build_reply_parts_from_blocks(
    blocks: list[dict], finish: str
) -> tuple[dict[str, Any], str]:
    """Build a manager_message directly from Anthropic-structured content blocks.

    Messages-mode counterpart to :func:`_build_reply_parts`: instead of parsing
    model-generation text, we read the upstream's canonical content blocks
    (text / thinking / tool_use) and assemble the OpenAI-shape assistant message
    that the trajectory tree stores and the SFT dump emits.

    tool_calls use :func:`tool_call_dict` (id dropped) so a generated assistant
    turn compares equal (dict equality) to its echo replayed as history on the
    next request — same invariant the sglang path relies on for tree matching.
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    signature_parts: list[str] = []   # extended-thinking provenance signatures
    manager_tcs: list[dict] = []
    has_tool_use = False
    for b in blocks:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            text_parts.append(b.get("text", ""))
        elif bt in ("thinking", "reasoning", "analysis"):
            # production converter treats reasoning/analysis as thinking too
            reasoning_parts.append(b.get("thinking") or b.get("reasoning") or b.get("analysis") or "")
            sig = b.get("signature")
            if sig:
                signature_parts.append(sig)
        elif bt == "tool_use":
            has_tool_use = True
            manager_tcs.append(tool_call_dict(b.get("name", "tool"), b.get("input")))
        elif bt == "code":
            # production converter appends code content as text
            text_parts.append(b.get("code", ""))
        elif bt in ("image", "image_url", "input_image", "input_audio", "audio", "video"):
            # multimodal: emit a reminder the model can't see it (matches production)
            text_parts.append(f"[{bt} omitted]")
        else:
            # redacted_thinking / server_tool_use / unknown: preserve as JSON text
            # (production content_part_to_text falls back to json.dumps) so content
            # is never silently dropped.
            import json as _json
            text_parts.append(_json.dumps(b, ensure_ascii=False))
    manager_message: dict[str, Any] = {"role": "assistant", "content": "".join(t for t in text_parts if t)}
    if reasoning_parts:
        manager_message["reasoning_content"] = "".join(reasoning_parts)
    if signature_parts:
        # thinking_signature: the extended-thinking provenance token. Joined
        # in block order to match how the trace presents thinking; downstream
        # (ShareGPT export / SFT) carries it as a provenance field.
        manager_message["thinking_signature"] = "".join(signature_parts)
    if manager_tcs:
        manager_message["tool_calls"] = manager_tcs
    if has_tool_use:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return manager_message, stop_reason


def chat_response_to_blocks(response: Any) -> tuple[list[dict], str]:
    """Convert a litellm/openai chat-completions response to Anthropic content blocks.

    chat mode routes through ``litellm.acompletion``, whose response is an
    OpenAI ``ModelResponse``. We re-shape it into the Anthropic content blocks
    that the messages-mode reply path already understands
    (:func:`_build_reply_parts_from_blocks` builds the manager_message from
    them, and ``_respond`` renders them back to Claude Code). This keeps chat
    mode on the *same* trajectory + dump path as messages mode.

    Tool-call arguments are normalized from the JSON **string** that litellm
    emits (live-API shape) to a **dict** (the HF/tree shape ``tool_call_dict``
    expects), so the stored turn matches its replay.

    Returns ``(blocks, stop_reason)`` where ``stop_reason`` is the Anthropic
    value (``tool_use`` / ``end_turn`` / ``max_tokens``).
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        return [{"type": "text", "text": ""}], "end_turn"
    msg = getattr(choices[0], "message", None)
    finish = getattr(choices[0], "finish_reason", None) or "stop"

    blocks: list[dict] = []
    # reasoning_content (if the chat endpoint returns it — e.g. via litellm's
    # reasoning extraction) -> a thinking block.
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    content = getattr(msg, "content", None)
    if content:
        blocks.append({"type": "text", "text": content})
    for tc in getattr(msg, "tool_calls", None) or []:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn else None
        args = getattr(fn, "arguments", None) if fn else None
        # litellm emits arguments as a JSON string; the tree stores a dict.
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except (ValueError, TypeError):
                args = {"_raw": args}
        if not isinstance(args, dict):
            args = args if isinstance(args, dict) else {}
        blocks.append({"type": "tool_use", "id": getattr(tc, "id", None) or "", "name": name or "tool", "input": args})

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if any(b.get("type") == "tool_use" for b in blocks):
        stop_reason = "tool_use"
    elif finish in ("length", "max_tokens"):
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return blocks, stop_reason


def responses_output_to_blocks(response: Any) -> tuple[list[dict], str]:
    """Convert an OpenAI Responses API response to Anthropic content blocks.

    Responses mode targets vLLM's /v1/responses (the only Responses impl with
    logprobs). The response has ``output`` (list of ResponseOutputItem), each with
    ``content`` (list of output parts). We extract text + tool calls into the
    Anthropic block shape the messages-mode reply path expects.

    Returns ``(blocks, stop_reason)``.
    """
    blocks: list[dict] = []
    has_tool_use = False
    status = getattr(response, "status", "completed")

    def _get(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

    for item in getattr(response, "output", []) or []:
        item_type = getattr(item, "type", None)

        # `function_call` and `reasoning` are TOP-LEVEL output items in the real
        # Responses API -- they are siblings of the message item, not parts inside
        # its `content`. Only scanning nested content silently drops every tool
        # call and all reasoning (an agent loop then sees an empty assistant turn
        # and stalls). Handle them here, then fall through for message items.
        if item_type == "function_call":
            has_tool_use = True
            args = _get(item, "arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except (ValueError, TypeError):
                    args = {"_raw": args}
            blocks.append({
                "type": "tool_use",
                "id": _get(item, "call_id") or _get(item, "id") or "",
                "name": _get(item, "name") or "tool",
                "input": args if isinstance(args, dict) else {},
            })
            continue
        if item_type == "reasoning":
            texts = []
            for part in _get(item, "summary") or []:
                text = _get(part, "text")
                if text:
                    texts.append(str(text))
            joined = "".join(texts)
            if joined:
                blocks.append({"type": "thinking", "thinking": joined})
            continue

        # message item: {role, content:[{type:output_text, text}, ...]}
        content = getattr(item, "content", []) or []
        for part in content:
            if not isinstance(part, dict) and not hasattr(part, "type"):
                continue
            pt = getattr(part, "type", None) if not isinstance(part, dict) else part.get("type")
            if pt == "output_text":
                text = getattr(part, "text", None) if not isinstance(part, dict) else part.get("text")
                if text:
                    blocks.append({"type": "text", "text": text})
            elif pt == "function_call":
                has_tool_use = True
                name = getattr(part, "name", None) if not isinstance(part, dict) else part.get("name")
                args = getattr(part, "arguments", None) if not isinstance(part, dict) else part.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except (ValueError, TypeError):
                        args = {"_raw": args}
                call_id = getattr(part, "call_id", None) or getattr(part, "id", None) if not isinstance(part, dict) else part.get("call_id") or part.get("id")
                blocks.append({"type": "tool_use", "id": call_id or "", "name": name or "tool", "input": args if isinstance(args, dict) else {}})
            elif pt == "reasoning":
                # reasoning content (vLLM may expose it)
                summary = getattr(part, "summary", None) if not isinstance(part, dict) else part.get("summary")
                if summary:
                    blocks.append({"type": "thinking", "thinking": str(summary)})

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if has_tool_use:
        stop_reason = "tool_use"
    elif status in ("incomplete", "failed"):
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    return blocks, stop_reason


# --- Request framing: session id + wire response/stream rendering ---


async def _responses_sse_stream(request: web.Request, resp: dict) -> web.StreamResponse:
    """Replay a finished Responses payload as the Responses SSE event sequence.

    We already have the complete reply, so this is a faithful re-emission rather
    than incremental generation: each output item is announced, its content sent
    as a single delta, then marked done. Clients that require the lifecycle
    events -- notably the Codex CLI, which hard-fails with "stream closed before
    response.completed" without them -- accept this.

    Event order per the Responses spec:
      response.created -> response.in_progress
        -> response.output_item.added
             (text)     response.output_text.delta / .done
             (reasoning) response.reasoning_summary_text.delta / .done
             (fn call)  response.function_call_arguments.delta / .done
           response.output_item.done
      -> response.completed
    Every event carries a monotonic ``sequence_number``.
    """
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await out.prepare(request)

    seq = 0

    async def send(event_type: str, payload: dict) -> None:
        nonlocal seq
        body = {"type": event_type, "sequence_number": seq, **payload}
        seq += 1
        await out.write(f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n".encode())

    items = resp.get("output") or []
    # response.created/in_progress advertise the response with output still empty.
    shell = {k: v for k, v in resp.items() if k != "output"}
    await send("response.created", {"response": {**shell, "status": "in_progress", "output": []}})
    await send("response.in_progress", {"response": {**shell, "status": "in_progress", "output": []}})

    for idx, item in enumerate(items):
        itype = item.get("type")
        await send("response.output_item.added", {"output_index": idx, "item": item})
        item_id = item.get("id", "")

        if itype == "message":
            for cidx, part in enumerate(item.get("content") or []):
                if part.get("type") != "output_text":
                    continue
                text = part.get("text") or ""
                await send("response.content_part.added", {
                    "item_id": item_id, "output_index": idx, "content_index": cidx,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                if text:
                    await send("response.output_text.delta", {
                        "item_id": item_id, "output_index": idx,
                        "content_index": cidx, "delta": text,
                    })
                await send("response.output_text.done", {
                    "item_id": item_id, "output_index": idx,
                    "content_index": cidx, "text": text,
                })
                await send("response.content_part.done", {
                    "item_id": item_id, "output_index": idx, "content_index": cidx,
                    "part": {"type": "output_text", "text": text, "annotations": []},
                })
        elif itype == "reasoning":
            for sidx, summary in enumerate(item.get("summary") or []):
                text = summary.get("text") or ""
                if text:
                    await send("response.reasoning_summary_text.delta", {
                        "item_id": item_id, "output_index": idx,
                        "summary_index": sidx, "delta": text,
                    })
                await send("response.reasoning_summary_text.done", {
                    "item_id": item_id, "output_index": idx,
                    "summary_index": sidx, "text": text,
                })
        elif itype == "function_call":
            args = item.get("arguments") or ""
            if args:
                await send("response.function_call_arguments.delta", {
                    "item_id": item_id, "output_index": idx, "delta": args,
                })
            await send("response.function_call_arguments.done", {
                "item_id": item_id, "output_index": idx, "arguments": args,
            })

        await send("response.output_item.done", {"output_index": idx, "item": item})

    terminal = "response.completed" if resp.get("status") == "completed" else "response.incomplete"
    await send(terminal, {"response": resp})
    await out.write_eof()
    return out


def _request_session_id(request: web.Request) -> str:
    """Resolve the session id for an inbound /v1/messages request.

    The sid routes the request to its trajectory tree; it is NOT the upstream
    API key (forwarding the sid upstream would 401). When the harness exports
    SLIME_SESSION_ID, every request in the run carries it as a header so the
    adapter can route without overloading the auth credential. Otherwise we
    fall back to the Bearer / X-Api-Key (legacy, sid == token).
    """
    sid = os.environ.get("SLIME_SESSION_ID")
    if sid:
        return sid
    return sid_from_bearer(request) or (request.headers.get("X-Api-Key") or "").strip() or "default"


async def _sse_stream(request: web.Request, chunks: list[dict]) -> web.StreamResponse:
    """Emit OpenAI-style SSE chunks (data: {json}\\n\\n) + [DONE]."""
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
    })
    await resp.prepare(request)
    for chunk in chunks:
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


def _render_response(body: dict, blocks: list[dict], stop_reason: str, in_tok: int, out_tok: int) -> dict:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "slime-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


async def _render_stream(request, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:
    """Stream blocks back as an Anthropic Messages SSE response: message_start,
    (content_block_start, content_block_delta, content_block_stop)*N,
    message_delta, message_stop."""
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)

    ms_data = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "role": "assistant",
            "model": "slime-actor",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    await out.write(f"event: message_start\ndata: {json.dumps(ms_data, ensure_ascii=False)}\n\n".encode())

    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:  # tool_use
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }

        cbs_data = {"type": "content_block_start", "index": idx, "content_block": start}
        await out.write(f"event: content_block_start\ndata: {json.dumps(cbs_data, ensure_ascii=False)}\n\n".encode())

        cbd_data = {"type": "content_block_delta", "index": idx, "delta": delta}
        await out.write(f"event: content_block_delta\ndata: {json.dumps(cbd_data, ensure_ascii=False)}\n\n".encode())

        cbe_data = {"type": "content_block_stop", "index": idx}
        await out.write(f"event: content_block_stop\ndata: {json.dumps(cbe_data, ensure_ascii=False)}\n\n".encode())

    md_data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    await out.write(f"event: message_delta\ndata: {json.dumps(md_data, ensure_ascii=False)}\n\n".encode())

    mst_data = {"type": "message_stop"}
    await out.write(f"event: message_stop\ndata: {json.dumps(mst_data, ensure_ascii=False)}\n\n".encode())

    return out


# count_tokens runs every turn but the client uses it only as a hint, not a
# hard budget, so returning 0 is fine.
async def _count_tokens(request: web.Request) -> web.Response:
    await request.read()
    return web.json_response({"input_tokens": 0})


# --- Anthropic-specific quirks: mid-list system folding ---


_MID_SYSTEM_WRAP_PREFIX = "<system-reminder>\n"
_MID_SYSTEM_WRAP_SUFFIX = "\n</system-reminder>\n"


def _fold_mid_list_system_into_user(body_obj: dict) -> bool:
    """Fold non-leading role:system messages into a neighbouring user message as
    a <system-reminder> text block. Mutates body_obj in place; returns True iff
    any fold happened.

    Some clients insert a system message in the middle of the message list, but
    many chat templates reject any system message past index 0. Attaching the
    wrapped reminder to the preceding user message (or the next one, if there is
    no prior user message) keeps the history acceptable to the template.
    """
    msgs = body_obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False

    system_idx = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system" and i > 0]
    if not system_idx:
        return False

    def _promote_to_list(msg: dict) -> list:
        c = msg.get("content")
        if isinstance(c, list):
            return c
        msg["content"] = [{"type": "text", "text": c if isinstance(c, str) else ""}]
        return msg["content"]

    def _wrap(text: str) -> dict:
        return {
            "type": "text",
            "text": _MID_SYSTEM_WRAP_PREFIX + text + _MID_SYSTEM_WRAP_SUFFIX,
        }

    changed = False
    TOMBSTONE: dict = {"__folded__": True}
    for i in system_idx:
        sys_msg = msgs[i]
        wrapped = _wrap(flatten_content(sys_msg.get("content")))
        target = None
        for j in range(i - 1, -1, -1):
            cand = msgs[j]
            if isinstance(cand, dict) and cand.get("role") == "user":
                target = cand
                _promote_to_list(target).append(wrapped)
                break
        if target is None:
            for j in range(i + 1, len(msgs)):
                cand = msgs[j]
                if isinstance(cand, dict) and cand.get("role") == "user":
                    target = cand
                    _promote_to_list(target).insert(0, wrapped)
                    break
        if target is None:
            msgs[i] = {"role": "user", "content": [wrapped]}
            changed = True
            continue
        msgs[i] = TOMBSTONE
        changed = True

    if changed:
        body_obj["messages"] = [m for m in msgs if m is not TOMBSTONE]
    return changed
