"""ChatUpstream tool-call bypass: tool-call replays take raw HTTP, tool-free stays litellm.

litellm 1.88 rejects the standard tool_calls[].type="function" on vLLM 0.26
(expects ChatCompletionMessageCustomToolCallParam). ChatUpstream.sample detects
tool_calls in the request and routes to _call_chat_upstream_raw (plain HTTP).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ENV = ("UPSTREAM_MODE", "SLIME_CHAT_BASE_URL", "SLIME_CHAT_API_KEY",
       "SLIME_CHAT_MODEL", "SLIME_CHAT_LOGPROBS", "SLIME_CHAT_TOP_LOGPROBS",
       "SLIME_CHAT_ARGS_AS_DICT", "SLIME_SESSION_ID")


def _env(**kw):
    saved = {k: os.environ.get(k) for k in ENV}
    os.environ.update(UPSTREAM_MODE="chat", SLIME_CHAT_BASE_URL="http://127.0.0.1:1/v1",
                      SLIME_CHAT_API_KEY="t", SLIME_CHAT_MODEL="m", SLIME_SESSION_ID="b", **kw)
    return saved


def _restore(saved):
    for k, v in saved.items():
        (os.environ.pop if v is None else os.environ.__setitem__)(k, v)


class _S:
    sampling_defaults = {}
    max_context_tokens = 0


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
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
        return False

    def post(self, url, **kw):
        self.posts.append((url, kw))
        return self._h(url, kw)


def _vllm_resp(content, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    choice = {"message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}
    # vLLM puts token_ids on the choice (with return_token_ids); include when logprobs
    choice["token_ids"] = [1, 2, 3]
    choice["logprobs"] = {"content": [{"logprob": -0.1}, {"logprob": -0.2}, {"logprob": -0.3}]}
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_tool_call_replay_uses_raw_http(monkeypatch):
    import litellm
    from anyharness.adapters.anthropic import AnthropicAdapter
    from anyharness.adapters import common as C

    litellm_called = []
    async def fake_acompletion(**kw):
        litellm_called.append(kw)
        return _vllm_resp("ok")
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    session = _FakeSession(lambda url, kw: _FakeResp(200, _vllm_resp("done", tool_calls=[{"id":"c1","type":"function","function":{"name":"read","arguments":'{"path":"x"}'}}])))
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)

    saved = _env()
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url=None)
        a._inbound_auth = {}
        msgs = [
            {"role": "user", "content": "read x"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"path": "x"}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "X"},
        ]
        r = asyncio.run(a.upstream.sample(msgs, None, {"max_tokens": 8}, _S(), "b"))
        assert not litellm_called, "litellm must NOT be called for tool-call replays"
        assert len(session.posts) == 1
        assert session.posts[0][0].endswith("/chat/completions")
        assert r.content_blocks is not None
        assert r.usage == {"input_tokens": 10, "output_tokens": 5}
    finally:
        _restore(saved)


def test_tool_free_turn_uses_litellm(monkeypatch):
    import litellm
    from anyharness.adapters.anthropic import AnthropicAdapter
    from anyharness.adapters import common as C

    litellm_called = []
    async def fake_acompletion(**kw):
        litellm_called.append(kw)
        # litellm path: response is an object with .choices/.usage
        class _M:
            content = "hi"; role = "assistant"; tool_calls = None
        class _Ch:
            message = _M(); finish_reason = "stop"; logprobs = None
        class _R:
            choices = [_Ch()]
            usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 3})()
        return _R()
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    session = _FakeSession(lambda url, kw: _FakeResp(500, {}))
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)

    saved = _env()
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url=None)
        a._inbound_auth = {}
        r = asyncio.run(a.upstream.sample([{"role": "user", "content": "hi"}], None, {"max_tokens": 8}, _S(), "b"))
        assert len(litellm_called) == 1, "tool-free turn must use litellm"
        assert len(session.posts) == 0, "no raw HTTP for tool-free turn"
        assert r.usage == {"input_tokens": 7, "output_tokens": 3}
    finally:
        _restore(saved)


def test_raw_path_returns_paired_ids_logprobs(monkeypatch):
    from anyharness.adapters.anthropic import AnthropicAdapter
    from anyharness.adapters import common as C

    session = _FakeSession(lambda url, kw: _FakeResp(200, _vllm_resp("done", tool_calls=[{"id":"c1","type":"function","function":{"name":"read","arguments":"{}"}}])))
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)

    saved = _env(SLIME_CHAT_LOGPROBS="1")
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url=None)
        a._inbound_auth = {}
        msgs = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": {"a": 1}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "r"}]
        r = asyncio.run(a.upstream.sample(msgs, None, {"max_tokens": 8}, _S(), "b"))
        assert r.turn.output_ids == [1, 2, 3]
        assert r.turn.output_log_probs == pytest.approx([-0.1, -0.2, -0.3])
        assert len(r.turn.output_ids) == len(r.turn.output_log_probs)
    finally:
        _restore(saved)
