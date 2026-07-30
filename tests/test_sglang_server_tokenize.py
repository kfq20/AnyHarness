"""sglang server-side tokenize: skip the local tokenizer when the server can render.

Phase 2: SglangUpstream prefers the server's /v1/tokenize (messages) over local
_render_token_ids, so sglang mode no longer requires MODEL_PATH. Verified live
against sglang 0.5.12 (POST /v1/tokenize with messages -> input_ids). The output
ids + logprobs come back paired from /generate, unchanged.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ENV = ("UPSTREAM_MODE", "SLIME_SGLANG_URL", "SLIME_SESSION_ID", "MODEL_PATH", "SLIME_TOP_LOGPROBS")


def _env(**kw):
    saved = {k: os.environ.get(k) for k in ENV}
    os.environ.update(UPSTREAM_MODE="sglang", SLIME_SGLANG_URL="http://127.0.0.1:1",
                      SLIME_SESSION_ID="srv-tok", **kw)
    return saved


def _restore(saved):
    for k, v in saved.items():
        (os.environ.pop if v is None else os.environ.__setitem__)(k, v)


class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._p = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
        return False

    async def json(self, content_type=None):
        return self._p

    async def text(self):
        return str(self._p)


class _FakeSession:
    def __init__(self, handler):
        self._h = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
        return False

    def post(self, url, **kw):
        return self._h(url, kw)


def _run(monkeypatch, tok, fake_session, **kw):
    from anyharness.adapters.anthropic import AnthropicAdapter
    monkeypatch.setattr("anyharness.adapters.common.aiohttp.ClientSession", lambda *a, **k: fake_session)
    saved = _env(**kw)
    try:
        a = AnthropicAdapter(tokenizer=tok, sglang_url="http://sglang.invalid:30000")

        class S:
            sampling_defaults = {}
            max_context_tokens = 32768

        return asyncio.run(a.upstream.sample(
            [{"role": "user", "content": "Say OK"}], None,
            {"max_tokens": 8, "temperature": 0.0}, S(), "srv-tok",
        ))
    finally:
        _restore(saved)


def test_server_tokenize_used_when_no_local_tokenizer(monkeypatch):
    """No MODEL_PATH, no local tokenizer — server /v1/tokenize provides the ids."""
    seen = {}

    def handler(url, kw):
        if "/v1/tokenize" in url or "/tokenize" in url:
            seen["tok_url"] = url
            seen["tok_payload"] = kw.get("json")
            return _FakeResp(200, {"input_ids": [1, 2, 3, 4]})
        if "/generate" in url:
            seen["gen_input_ids"] = kw.get("json", {}).get("input_ids")
            return _FakeResp(200, {"meta_info": {"output_token_logprobs": [[-0.1, 5], [-0.2, 6]],
                                                 "finish_reason": {"type": "stop"}}})
        return _FakeResp(404, {})

    r = _run(monkeypatch, None, _FakeSession(handler))
    assert seen["tok_payload"]["messages"] == [{"role": "user", "content": "Say OK"}]
    assert seen["tok_payload"]["add_generation_prompt"] is True
    # the generated ids flow straight from /generate, server-tokenize ids only feed it
    assert seen["gen_input_ids"] == [1, 2, 3, 4]
    assert r.turn.output_ids == [5, 6]
    assert r.turn.output_log_probs == pytest.approx([-0.1, -0.2])


def test_falls_back_to_local_tokenizer_when_server_unavailable(monkeypatch):
    """A 5xx/404 from the server falls back to the local tokenizer render."""

    class FakeTok:
        def apply_chat_template(self, msgs, **kw):
            return {"input_ids": [99, 100]}

    def handler(url, kw):
        if "/tokenize" in url:
            return _FakeResp(500, {"detail": "down"})
        if "/generate" in url:
            assert kw["json"]["input_ids"] == [99, 100]  # local render used
            return _FakeResp(200, {"meta_info": {"output_token_logprobs": [[-0.3, 7]], "finish_reason": {"type": "stop"}}})
        return _FakeResp(404, {})

    r = _run(monkeypatch, FakeTok(), _FakeSession(handler))
    assert r.turn.output_ids == [7]
    assert r.turn.output_log_probs == pytest.approx([-0.3])


def test_falls_back_to_empty_when_no_server_and_no_tokenizer(monkeypatch):
    """Server down + no tokenizer -> empty ids (graceful, not a crash)."""

    def handler(url, kw):
        if "/tokenize" in url:
            return _FakeResp(500, {})
        if "/generate" in url:
            # empty prompt ids from the local-render fallback (no tokenizer)
            assert kw["json"]["input_ids"] == []
            return _FakeResp(200, {"meta_info": {"output_token_logprobs": [], "finish_reason": {"type": "stop"}}})
        return _FakeResp(404, {})

    r = _run(monkeypatch, None, _FakeSession(handler))
    assert r.turn.output_ids == []
    assert r.turn.output_log_probs == []


def test_tools_forwarded_to_server_tokenize(monkeypatch):
    """tools_schema is passed through to the server tokenize call."""
    seen = {}

    def handler(url, kw):
        if "/tokenize" in url:
            seen["tools"] = kw.get("json", {}).get("tools")
            return _FakeResp(200, {"input_ids": [1]})
        if "/generate" in url:
            return _FakeResp(200, {"meta_info": {"output_token_logprobs": [[-0.1, 2]], "finish_reason": {"type": "stop"}}})
        return _FakeResp(404, {})

    from anyharness.adapters.anthropic import AnthropicAdapter
    monkeypatch.setattr("anyharness.adapters.common.aiohttp.ClientSession", lambda *a, **k: _FakeSession(handler))
    saved = _env()
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url="http://x")

        class S:
            sampling_defaults = {}
            max_context_tokens = 0

        asyncio.run(a.upstream.sample(
            [{"role": "user", "content": "go"}],
            [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            {"max_tokens": 8}, S(), "t"))
    finally:
        _restore(saved)
    assert seen["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {}}}]
