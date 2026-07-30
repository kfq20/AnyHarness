"""Dump trajectories from a TrajectoryManager as ShareGPT for distillation.

Companion to :mod:`anyharness.dump` (which emits token/loss_mask SFT samples).
This module walks the message *tree* a ``TrajectoryManager`` builds and folds
each session's leaf path into one ShareGPT conversation — the format the opb
swe-pro / deep-swe pipelines already consume:

  system    -> human   (the task instruction)
  user      -> human   (the prompt / tool_result the model saw)
  assistant -> gpt     (reasoning_content + thinking_signature + tool_calls)

The gpt turn carries the thinking summary wrapped in a think block and the
extended-thinking signature in a think_signature block, so a downstream SFT
trainer that reconstructs CoT sees both the reasoning text and its provenance
token. tool_calls are emitted as tool_call JSON blocks.

Usage:
    from anyharness.sharegpt_dump import dump_sharegpt
    dump_sharegpt(manager, out_path="trajectories_sharegpt.jsonl")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

# Block delimiters the model sees in a real agent transcript (plain strings so
# the ShareGPT content carries the exact markers a trainer/SFT loop expects).
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
SIG_OPEN, SIG_CLOSE = "<think_signature>", "</think_signature>"
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"


def _node_chain_to_sharegpt(root: Any) -> list[dict]:
    """Walk one session tree's first-leaf path, folding nodes into ShareGPT turns.

    The tree is built left-to-right as the agent runs; the first leaf is the
    canonical conversation (AnyHarness forks only on replay mismatch, which the
    signature-carrying echo now prevents). Each node's ``message`` is the
    hub-shape dict (role + content + reasoning_content? + thinking_signature?
    + tool_calls?).
    """
    # collect the root-to-first-leaf chain
    chain: list[Any] = []
    node = root
    while node is not None:
        chain.append(node)
        node = node.children[0] if node.children else None
    # the root is a dummy (message=None); skip it
    turns: list[dict] = []
    for node in chain[1:]:
        msg = node.message
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "assistant":
            parts: list[str] = []
            reasoning = (msg.get("reasoning_content") or "").strip()
            if reasoning:
                parts.append(f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}")
            sig = (msg.get("thinking_signature") or "").strip()
            if sig:
                parts.append(f"{SIG_OPEN}{sig}{SIG_CLOSE}")
            if content:
                parts.append(content)
            for tc in msg.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                call = {"name": fn.get("name"), "arguments": fn.get("input") or fn.get("arguments") or {}}
                parts.append(f"{CALL_OPEN}\n{json.dumps(call, ensure_ascii=False)}\n{CALL_CLOSE}")
            turns.append({"from": "gpt", "value": "\n".join(parts)})
        else:  # system / user / tool -> human
            turns.append({"from": "human", "value": content})
    return turns


def _iter_session_roots(manager: Any) -> Iterator[tuple[str, Any]]:
    """Yield (session_id, root_node) for every session the manager holds."""
    trees = getattr(manager, "_trees", None) or {}
    for sid, root in trees.items():
        if root is not None and root.children:
            yield sid, root


def manager_to_sharegpt_records(manager: Any, *, model: str | None = None) -> list[dict]:
    """Convert every session in a TrajectoryManager to a ShareGPT record."""
    records: list[dict] = []
    for sid, root in _iter_session_roots(manager):
        conv = _node_chain_to_sharegpt(root)
        if not conv:
            continue
        records.append({
            "id": sid,
            "source": "anyharness",
            "model": model or "",
            "conversations": conv,
        })
    return records


def dump_sharegpt(manager: Any, out_path: str | Path, *, model: str | None = None) -> Path:
    """Write one ShareGPT record per session to ``out_path`` (jsonl)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = manager_to_sharegpt_records(manager, model=model)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[sharegpt] wrote {len(records)} trajectories -> {out}")
    return out


if __name__ == "__main__":
    import sys
    import pickle
    if len(sys.argv) < 3:
        print("usage: python -m anyharness.sharegpt_dump <manager.pkl> <out.jsonl>")
        sys.exit(2)
    mgr = pickle.load(open(sys.argv[1], "rb"))
    dump_sharegpt(mgr, sys.argv[2])
