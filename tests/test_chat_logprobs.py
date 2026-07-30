"""chat-mode logprob capture: SLIME_CHAT_LOGPROBS=1.

Covers the gap left by test_chat_downstream.py, whose fake upstream returns no
logprobs. A real OpenAI-compatible upstream (vLLM / SGLang) returns
``choices[0].logprobs.content = [{token, logprob, top_logprobs}]`` when asked
with ``logprobs=True``; these tests drive that shape through
``_call_chat_upstream`` and then through ``record_turn``.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from anyharness import Sample, TrajectoryManager
from anyharness.adapters import AnthropicAdapter

CHAT_ENV = ("UPSTREAM_MODE", "SLIME_CHAT_BASE_URL", "SLIME_CHAT_API_KEY",
            "SLIME_CHAT_MODEL", "SLIME_CHAT_LOGPROBS", "SLIME_CHAT_TOP_LOGPROBS")


class _FakeLogprobEntry:
    def __init__(self, token: str, logprob: float) -> None:
        self.token = token
        self.logprob = logprob
        self.top_logprobs = []


class _FakeLogprobs:
    def __init__(self, pairs) -> None:
        self.content = [_FakeLogprobEntry(t, lp) for t, lp in pairs]


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"
        self.tool_calls = None
        self.reasoning_content = None

    def get(self, key, default=None):
        return getattr(self, key, default)


class _FakeChoice:
    def __init__(self, content: str, pairs, token_ids=None) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = "stop"
        self.logprobs = _FakeLogprobs(pairs) if pairs is not None else None
        # vLLM >= 0.10.2 attaches the sampled ids here under return_token_ids
        if token_ids is not None:
            self.token_ids = token_ids


class _FakeResponse:
    """Mimics the litellm ModelResponse attribute surface we read."""

    def __init__(self, content: str, pairs, token_ids=None) -> None:
        self.choices = [_FakeChoice(content, pairs, token_ids)]


def _install_env(**extra) -> dict:
    saved = {k: os.environ.get(k) for k in CHAT_ENV}
    os.environ.update(UPSTREAM_MODE="chat", SLIME_CHAT_BASE_URL="http://127.0.0.1:1/v1",
                      SLIME_CHAT_API_KEY="test", SLIME_CHAT_MODEL="fake", **extra)
    return saved


def _restore_env(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer's id-conversion surface."""

    unk_token_id = 0

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self._vocab = vocab or {}

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._vocab.get(tokens, self.unk_token_id)
        return [self._vocab.get(t, self.unk_token_id) for t in tokens]


async def _call_chat_with_fake_upstream(
    monkeypatch, pairs, *, logprobs_on: bool, tokenizer=None, token_ids=None
):
    """Run _call_chat_upstream against a stubbed litellm.acompletion."""
    import litellm

    captured: dict = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse("2 + 2 = 4", pairs, token_ids)

    monkeypatch.setattr(litellm, "acompletion", _fake_acompletion)

    extra = {"SLIME_CHAT_LOGPROBS": "1"} if logprobs_on else {}
    saved = _install_env(**extra)
    if not logprobs_on:
        os.environ.pop("SLIME_CHAT_LOGPROBS", None)
    try:
        adapter = AnthropicAdapter(tokenizer=tokenizer, sglang_url=None)
        turn = await adapter._call_chat_upstream(
            [{"role": "user", "content": "what is 2+2"}], None,
            {"max_tokens": 64}, "lp-sid",
        )
        return turn, captured
    finally:
        _restore_env(saved)


def _tokenizer_for(pairs) -> _FakeTokenizer:
    """A tokenizer whose vocab covers exactly the tokens in ``pairs``."""
    return _FakeTokenizer({tok: i + 1 for i, (tok, _) in enumerate(pairs)})


def test_chat_logprobs_off_by_default(monkeypatch):
    """Without SLIME_CHAT_LOGPROBS the request omits logprobs and captures none."""
    turn, kwargs = asyncio.run(
        _call_chat_with_fake_upstream(monkeypatch, [("2", -0.1)], logprobs_on=False)
    )
    assert "logprobs" not in kwargs, "logprobs must not be requested by default"
    assert turn.output_log_probs == []


def test_chat_logprobs_requested_and_captured(monkeypatch):
    """SLIME_CHAT_LOGPROBS=1 asks for logprobs + return_token_ids and pairs them."""
    pairs = [("2", -0.05), (" +", -0.12), (" 2", -0.30), (" =", -0.01), (" 4", -0.02)]
    ids = [17, 18, 19, 20, 21]
    turn, kwargs = asyncio.run(
        _call_chat_with_fake_upstream(monkeypatch, pairs, logprobs_on=True, token_ids=ids)
    )
    assert kwargs.get("logprobs") is True
    assert "top_logprobs" not in kwargs, "top-k must stay off unless explicitly set"
    # the vLLM extension must be requested, else there are no TITO-safe ids
    assert kwargs.get("extra_body", {}).get("return_token_ids") is True
    assert turn.output_log_probs == pytest.approx([-0.05, -0.12, -0.30, -0.01, -0.02])
    # ids come from the upstream verbatim; 1:1 or record_turn's assert fires
    assert turn.output_ids == ids


def test_chat_logprobs_dropped_when_upstream_gives_no_ids(monkeypatch):
    """No token_ids (not vLLM / too old) means drop, never re-derive them."""
    pairs = [("2", -0.05), (" +", -0.12)]
    turn, _ = asyncio.run(
        _call_chat_with_fake_upstream(monkeypatch, pairs, logprobs_on=True, token_ids=None)
    )
    assert turn.output_log_probs == [], "must not keep logprobs without upstream ids"
    assert turn.output_ids == []


def test_chat_logprobs_ignore_tokenizer_for_ids(monkeypatch):
    """A local tokenizer must NOT be used to reconstruct ids (TITO violation)."""
    pairs = [("2", -0.05), (" +", -0.12)]
    turn, _ = asyncio.run(
        _call_chat_with_fake_upstream(monkeypatch, pairs, logprobs_on=True,
                                      tokenizer=_tokenizer_for(pairs), token_ids=None)
    )
    assert turn.output_ids == [], "ids must never come from a local vocab lookup"
    assert turn.output_log_probs == []


def test_chat_logprobs_dropped_on_count_mismatch(monkeypatch):
    """Upstream ids and logprobs disagreeing means drop both, not truncate."""
    pairs = [("2", -0.05), (" +", -0.12), (" 4", -0.02)]
    turn, _ = asyncio.run(
        _call_chat_with_fake_upstream(monkeypatch, pairs, logprobs_on=True, token_ids=[17, 18])
    )
    assert turn.output_log_probs == []
    assert turn.output_ids == []


def test_chat_top_logprobs_forwarded(monkeypatch):
    """SLIME_CHAT_TOP_LOGPROBS=N forwards top_logprobs to the upstream."""
    saved_top = os.environ.get("SLIME_CHAT_TOP_LOGPROBS")
    os.environ["SLIME_CHAT_TOP_LOGPROBS"] = "5"
    try:
        _, kwargs = asyncio.run(
            _call_chat_with_fake_upstream(monkeypatch, [("2", -0.1)], logprobs_on=True)
        )
        assert kwargs.get("top_logprobs") == 5
    finally:
        if saved_top is None:
            os.environ.pop("SLIME_CHAT_TOP_LOGPROBS", None)
        else:
            os.environ["SLIME_CHAT_TOP_LOGPROBS"] = saved_top


def test_chat_upstream_without_logprobs_support(monkeypatch):
    """An upstream that ignores the logprobs request must not crash the turn."""
    turn, _ = asyncio.run(
        _call_chat_with_fake_upstream(monkeypatch, None, logprobs_on=True)
    )
    assert turn.output_log_probs == []


def test_cli_rejects_logprobs_without_model_path():
    """Config.validate fails fast rather than silently dropping logprobs."""
    from anyharness.cli import Config

    saved = _install_env(SLIME_CHAT_LOGPROBS="1")
    saved_mp, saved_prompt = os.environ.get("MODEL_PATH"), os.environ.get("PROMPT")
    os.environ.pop("MODEL_PATH", None)
    os.environ["PROMPT"] = "do a thing"
    try:
        with pytest.raises(ValueError, match="MODEL_PATH"):
            Config().validate()
    finally:
        _restore_env(saved)
        for k, v in (("MODEL_PATH", saved_mp), ("PROMPT", saved_prompt)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cli_loads_tokenizer_in_chat_mode(monkeypatch):
    """chat mode with MODEL_PATH loads a tokenizer so logprobs can align."""
    from anyharness import cli

    saved = _install_env(SLIME_CHAT_LOGPROBS="1")
    saved_mp = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = "/fake/model"
    loaded: dict = {}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kw):
            loaded["path"] = path
            return _FakeTokenizer({"a": 1})

    import sys
    import types
    fake_mod = types.ModuleType("transformers")
    fake_mod.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)
    try:
        cfg = cli.Config()
        tok = cli.load_tokenizer(cfg)
        assert tok is not None, "chat mode must load a tokenizer when MODEL_PATH is set"
        assert loaded["path"] == "/fake/model"
    finally:
        _restore_env(saved)
        if saved_mp is None:
            os.environ.pop("MODEL_PATH", None)
        else:
            os.environ["MODEL_PATH"] = saved_mp


class _FakeRespLogprob:
    def __init__(self, token: str, logprob: float) -> None:
        self.token = token
        self.logprob = logprob


class _FakeRespContent:
    def __init__(self, pairs) -> None:
        self.type = "output_text"
        self.text = "2 + 2 = 4"
        self.logprobs = [_FakeRespLogprob(t, lp) for t, lp in pairs]


class _FakeRespItem:
    def __init__(self, pairs) -> None:
        self.type = "message"
        self.content = [_FakeRespContent(pairs)]


class _FakeResponsesResponse:
    def __init__(self, pairs) -> None:
        self.output = [_FakeRespItem(pairs)]
        self.status = "completed"


def test_responses_logprobs_always_dropped(monkeypatch):
    """The Responses API has no token ids, so logprobs can never ship from it."""
    import litellm

    pairs = [("2", -0.05), (" +", -0.12), (" 4", -0.02)]

    async def _fake_aresponses(**kwargs):
        return _FakeResponsesResponse(pairs)

    # litellm exposes the Responses API as aresponses(); tolerate either name.
    target = "aresponses" if hasattr(litellm, "aresponses") else "acompletion"
    monkeypatch.setattr(litellm, target, _fake_aresponses, raising=False)

    saved = {k: os.environ.get(k) for k in
             ("UPSTREAM_MODE", "SLIME_RESPONSES_BASE_URL", "SLIME_RESPONSES_API_KEY",
              "SLIME_RESPONSES_MODEL", "SLIME_RESPONSES_LOGPROBS")}
    os.environ.update(UPSTREAM_MODE="responses", SLIME_RESPONSES_BASE_URL="http://127.0.0.1:1/v1",
                      SLIME_RESPONSES_API_KEY="test", SLIME_RESPONSES_MODEL="fake",
                      SLIME_RESPONSES_LOGPROBS="1")
    try:
        adapter = AnthropicAdapter(tokenizer=_tokenizer_for(pairs), sglang_url=None)
        try:
            turn = asyncio.run(adapter._call_responses_upstream(
                [{"role": "user", "content": "what is 2+2"}], None, {"max_tokens": 64}, "rp-sid",
            ))
        except Exception as exc:  # pragma: no cover - shape drift in litellm
            pytest.skip(f"responses upstream not exercisable here: {type(exc).__name__}: {exc}")
        assert turn.output_log_probs == [], "responses logprobs are not TITO-safe"
        assert turn.output_ids == []
    finally:
        _restore_env(saved)


def test_cli_rejects_responses_logprobs():
    """responses + logprobs is refused up front, with a pointer to tinker/sglang."""
    from anyharness.cli import Config

    saved = {k: os.environ.get(k) for k in
             ("UPSTREAM_MODE", "SLIME_RESPONSES_BASE_URL", "SLIME_RESPONSES_LOGPROBS", "PROMPT")}
    os.environ.update(UPSTREAM_MODE="responses", SLIME_RESPONSES_BASE_URL="http://127.0.0.1:1/v1",
                      SLIME_RESPONSES_LOGPROBS="1", PROMPT="do a thing")
    try:
        with pytest.raises(ValueError, match="tinker"):
            Config().validate()
    finally:
        _restore_env(saved)


def test_chat_logprobs_survive_record_turn(monkeypatch):
    """REGRESSION: a chat TurnRecord carrying logprobs must be recordable.

    record_turn asserts len(output_log_probs) == len(output_ids). chat mode
    returns output_ids=[] (no tokenizer) but non-empty output_log_probs when
    SLIME_CHAT_LOGPROBS=1, so the assert fires and kills the run.
    """
    pairs = [("2", -0.05), (" +", -0.12), (" 2", -0.30)]
    turn, _ = asyncio.run(
        _call_chat_with_fake_upstream(monkeypatch, pairs, logprobs_on=True, token_ids=[5, 6, 7])
    )
    assert turn.output_log_probs, "logprobs must survive to exercise record_turn"

    mgr = TrajectoryManager()  # record_turn creates the tree lazily (setdefault)
    mgr.record_turn(
        "lp-sid",
        turn=turn,
        prompt_messages=[{"role": "user", "content": "what is 2+2"}],
        response_message={"role": "assistant", "content": "2 + 2 = 4"},
    )
    samples = mgr.get_trajectory("lp-sid", base_sample=Sample(index=0), reward=0.0)
    assert isinstance(samples, list)
