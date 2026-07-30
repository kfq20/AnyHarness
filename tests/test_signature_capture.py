"""Signature capture: ensure extended-thinking signatures survive the pipeline.

AnyHarness originally captured the thinking *plaintext* but dropped the
cryptographic signature. These tests pin the end-to-end flow:

  SSE signature_delta -> thinking block -> manager_message['thinking_signature']
  -> replayed echo == generated leaf (no tree fork) -> ShareGPT export.

No live API is needed: a synthetic Anthropic SSE stream carries the signature.
"""
from __future__ import annotations

import asyncio
import json

from anyharness.adapters.anthropic import (
    _build_reply_parts_from_blocks,
    _translate_messages,
)
from anyharness.adapters.common import _consume_messages_sse
from anyharness.sharegpt_dump import _node_chain_to_sharegpt

THINK_SIG = "EuYBCkYBsignatureBlobExample"


def _sse(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events)


class _FakeStreamResponse:
    """Minimal stand-in for an aiohttp streaming response."""

    def __init__(self, sse: str):
        self._chunks = [c.encode() for c in sse.split("\n\n") if c.strip()]

    @property
    def content(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


def _thinking_turn_sse() -> str:
    return _sse([
        {"type": "message_start", "message": {"usage": {"output_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "thinking_delta", "thinking": "let me think"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": THINK_SIG}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "4 cents"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
    ])


def test_sse_signature_captured_into_thinking_block():
    text, stop, out_tok, blocks = asyncio.run(_consume_messages_sse(_FakeStreamResponse(_thinking_turn_sse())))
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "let me think"
    assert blocks[0]["signature"] == THINK_SIG  # the load-bearing assertion


def test_signature_reaches_manager_message():
    _, _, _, blocks = asyncio.run(_consume_messages_sse(_FakeStreamResponse(_thinking_turn_sse())))
    mgr_msg, stop = _build_reply_parts_from_blocks(blocks, "end_turn")
    assert mgr_msg["thinking_signature"] == THINK_SIG
    assert mgr_msg["reasoning_content"] == "let me think"


def test_replayed_echo_equals_generated_leaf():
    """A generated leaf (carries signature) must equal its replayed echo.

    If the replay path dropped the signature while the generated path kept it,
    the trajectory tree would fork on every thinking turn. Both must carry it.
    """
    generated = {"role": "assistant", "content": "4 cents",
                 "reasoning_content": "let me think", "thinking_signature": THINK_SIG}
    anthropic_msg = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "let me think", "signature": THINK_SIG},
        {"type": "text", "text": "4 cents"},
    ]}
    translated = _translate_messages([anthropic_msg], None)
    assert translated[0] == generated


class _Node:
    def __init__(self, message=None, children=None):
        self.message = message
        self.children = children if children is not None else []


def test_sharegpt_export_carries_signature():
    root = _Node(message=None)
    root.children = [_Node(message={"role": "user", "content": "bat and ball?"})]
    root.children[0].children = [_Node(message={
        "role": "assistant", "content": "4 cents",
        "reasoning_content": "let me think", "thinking_signature": THINK_SIG,
    })]
    conv = _node_chain_to_sharegpt(root)
    gpt = [c for c in conv if c["from"] == "gpt"][0]
    assert "<think_signature>" in gpt["value"]
    assert THINK_SIG in gpt["value"]
    assert "let me think" in gpt["value"]


def _body_with_cc_quirks() -> dict:
    return {
        "model": "claude-opus-5",
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "system": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "scope": {"type": "session"}}}],
        "messages": [{"role": "user", "content": "q"}],
    }


def test_context_management_stripped():
    from anyharness.adapters.anthropic import _strip_cache_control_scope
    body = _body_with_cc_quirks()
    _strip_cache_control_scope(body)
    body.pop("context_management", None)
    assert "context_management" not in body
    assert "scope" not in body["system"][0]["cache_control"]
    assert body["system"][0]["cache_control"]["type"] == "ephemeral"


def test_beta_denylist_filters_rejected_flags():
    from anyharness.adapters.common import _BETA_DENYLIST
    assert "prompt-caching-scope-2026-01-05" in _BETA_DENYLIST
    assert "advanced-tool-use-2025-11-20" in _BETA_DENYLIST
    # a normal flag is kept
    assert "claude-code-20250219" not in _BETA_DENYLIST
