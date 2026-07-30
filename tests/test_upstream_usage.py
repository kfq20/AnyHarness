"""Upstream usage truthfulness (Phase 3).

Message-level backends (messages/chat/responses) surface the upstream server's
own token counts onto SamplingResult.usage; _run_turn reports those as
prompt_tokens/completion_tokens instead of the locally-rendered len(prompt_ids)
and the (zero) len(turn.output_ids).

Token-level backends (sglang/tinker) leave usage None and keep the local-count
fallback, so they are unchanged.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CHAT_ENV = ("UPSTREAM_MODE", "SLIME_CHAT_BASE_URL", "SLIME_CHAT_API_KEY",
            "SLIME_CHAT_MODEL", "SLIME_CHAT_LOGPROBS", "SLIME_CHAT_TOP_LOGPROBS",
            "SLIME_MESSAGES_UPSTREAM_URL", "ANTHROPIC_API_KEY",
            "SLIME_RESPONSES_BASE_URL", "SLIME_RESPONSES_API_KEY",
            "SLIME_RESPONSES_MODEL", "SLIME_RESPONSES_LOGPROBS",
            "SLIME_SGLANG_URL", "SLIME_SESSION_ID")


def _env(mode, **kw):
    saved = {k: os.environ.get(k) for k in CHAT_ENV}
    base = {"UPSTREAM_MODE": mode, "SLIME_SESSION_ID": "u"}
    if mode == "chat":
        base.update(SLIME_CHAT_BASE_URL="http://127.0.0.1:1/v1", SLIME_CHAT_API_KEY="t",
                    SLIME_CHAT_MODEL="m")
    elif mode == "messages":
        base.update(SLIME_MESSAGES_UPSTREAM_URL="http://127.0.0.1:1",
                    ANTHROPIC_API_KEY="t")
    elif mode == "responses":
        base.update(SLIME_RESPONSES_BASE_URL="http://127.0.0.1:1",
                    SLIME_RESPONSES_API_KEY="t", SLIME_RESPONSES_MODEL="m")
    elif mode == "sglang":
        base.update(SLIME_SGLANG_URL="http://127.0.0.1:1")
    base.update(kw)
    os.environ.update(**base)
    return saved


def _restore(saved):
    for k, v in saved.items():
        (os.environ.pop if v is None else os.environ.__setitem__)(k, v)


class _U:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class _Msg:
    def __init__(self, content):
        self.content = content
        self.role = "assistant"
        self.tool_calls = None
        self.reasoning_content = None

    def get(self, k, d=None):
        return getattr(self, k, d)


class _Choice:
    def __init__(self, content, usage):
        self.message = _Msg(content)
        self.finish_reason = "stop"
        self.logprobs = None
        self.usage = usage


class _Resp:
    def __init__(self, content, usage):
        self.choices = [_Choice(content, usage)]
        self.usage = usage


class _S:
    sampling_defaults = {}
    max_context_tokens = 0


async def _fake_acompletion_with_usage(**kw):
    return _Resp("ok", _U(19, 64))


async def _fake_acompletion_no_usage(**kw):
    return _Resp("ok", None)


def test_chat_upstream_surfaces_usage(monkeypatch):
    import litellm
    from anyharness.adapters.anthropic import AnthropicAdapter

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion_with_usage)
    saved = _env("chat")
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url=None)
        r = asyncio.run(a.upstream.sample(
            [{"role": "user", "content": "hi"}], None,
            {"max_tokens": 8}, _S(), "u"))
        assert r.usage == {"input_tokens": 19, "output_tokens": 64}
    finally:
        _restore(saved)


def test_result_usage_drives_token_counts():
    """The _run_turn computation: result.usage -> prompt/completion tokens."""
    from anyharness.adapters.common import SamplingResult
    from anyharness.trajectory import TurnRecord

    # mimic _run_turn's branch
    r = SamplingResult(turn=TurnRecord(prompt_ids=[], output_ids=[], finish_reason="stop"),
                      usage={"input_tokens": 19, "output_tokens": 64})
    prompt_ids = []  # _run_turn still renders this
    if r.usage:
        in_tok = int(r.usage.get("input_tokens") or len(prompt_ids))
        out_tok = int(r.usage.get("output_tokens") or len(r.turn.output_ids))
    else:
        in_tok, out_tok = len(prompt_ids), len(r.turn.output_ids)
    assert (in_tok, out_tok) == (19, 64)


def test_token_level_message_level_false():
    """sglang/tinker declare message_level False (token-level, no usage surfacing)."""
    from anyharness.adapters.anthropic import AnthropicAdapter

    saved = _env("sglang")
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url="http://127.0.0.1:1")
        assert a.upstream.message_level is False
    finally:
        _restore(saved)
    saved = _env("tinker", TINKER_BASE_URL="http://127.0.0.1:1",
                 TINKER_BASE_MODEL="m", MODEL_PATH="m")
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url=None)
        assert a.upstream.message_level is False
    finally:
        _restore(saved)


def test_usage_falls_back_when_upstream_omits(monkeypatch):
    """No usage on the response -> SamplingResult.usage None -> local counts."""
    import litellm
    from anyharness.adapters.anthropic import AnthropicAdapter

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion_no_usage)
    saved = _env("chat")
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url=None)
        r = asyncio.run(a.upstream.sample(
            [{"role": "user", "content": "hi"}], None,
            {"max_tokens": 8}, _S(), "u"))
        assert r.usage is None
    finally:
        _restore(saved)
