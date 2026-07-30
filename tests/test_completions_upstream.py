"""CompletionsUpstream: vLLM /v1/completions as a token-level backend.

Shapes here mirror a live probe of vLLM 0.26 on the 8-card box (GPU 0, port 30001):

    POST /v1/completions  {"prompt": [ids], "return_token_ids": true, "logprobs": N}
    -> {"choices": [{"text": ..., "token_ids": [...], "finish_reason": "stop",
                     "logprobs": {"token_logprobs": [...], "top_logprobs":
                                  [{token_str: logprob}, ...]}}],
        "usage": {"prompt_tokens": P, "completion_tokens": C}}

The whole point: the prompt goes out as a raw token-id array (server uses our
ids verbatim — input-side TITO) and token_ids + token_logprobs come back paired
from ONE response (output-side TITO, no separate re-issue like chat needs).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anyharness.adapters import common as C  # noqa: E402
from anyharness.adapters.anthropic import AnthropicAdapter  # noqa: E402
from anyharness.trajectory import MASK_LOGPROB  # noqa: E402

ENV = ("UPSTREAM_MODE", "SLIME_COMPLETIONS_BASE_URL", "SLIME_COMPLETIONS_API_KEY",
       "SLIME_COMPLETIONS_MODEL", "SLIME_COMPLETIONS_LOGPROBS",
       "SLIME_COMPLETIONS_TOP_LOGPROBS", "SLIME_SESSION_ID")


def _env(**kw) -> dict:
    saved = {k: os.environ.get(k) for k in ENV}
    os.environ.update(
        UPSTREAM_MODE="completions",
        SLIME_COMPLETIONS_BASE_URL="http://127.0.0.1:1/v1",
        SLIME_COMPLETIONS_MODEL="qwen3-9b",
        SLIME_SESSION_ID="c-sid",
        **kw,
    )
    return saved


def _restore(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _Session:
    sampling_defaults: dict = {}
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
        self.posts: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
        return False

    def post(self, url, **kw):
        self.posts.append((url, kw))
        return self._h(url, kw)


def _vllm_completion(ids, logprobs=None, finish="stop", topk=None):
    """Build a vLLM /v1/completions response body."""
    choice = {
        "text": "decoded",
        "token_ids": ids,
        "finish_reason": finish,
        "logprobs": {},
    }
    if logprobs is not None:
        choice["logprobs"]["token_logprobs"] = logprobs
    if topk is not None:
        choice["logprobs"]["top_logprobs"] = topk
    return {"choices": [choice], "usage": {"prompt_tokens": 5, "completion_tokens": len(ids)}}


def _run(monkeypatch, handler, *, body=None, prompt_ids=(1, 2, 3), env=None):
    session = _FakeSession(handler)
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)
    saved = _env(**(env or {}))
    try:
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        adapter._inbound_auth = {}
        turn, top_raw = asyncio.run(C.call_completions(
            list(prompt_ids), _Session(), body or {"max_tokens": 4},
            adapter=adapter, session_id="c-sid",
        ))
        return turn, top_raw, session
    finally:
        _restore(saved)


# --------------------------------------------------------------------------- #
# call_completions: wire shape + TITO
# --------------------------------------------------------------------------- #


def test_prompt_sent_as_token_id_array(monkeypatch):
    """Input-side TITO: prompt is a raw int list, not text; return_token_ids set."""
    turn, _, sess = _run(monkeypatch, lambda url, kw: _FakeResp(200, _vllm_completion([10, 11])))
    body = sess.posts[0][1]["json"]
    assert body["prompt"] == [1, 2, 3], "prompt must be the raw token-id array"
    assert isinstance(body["prompt"], list) and all(isinstance(i, int) for i in body["prompt"])
    assert body["return_token_ids"] is True


def test_paired_ids_and_logprobs_from_one_response(monkeypatch):
    """Output-side TITO: ids + logprobs from the same response, aligned."""
    turn, _, _ = _run(monkeypatch, lambda url, kw: _FakeResp(
        200, _vllm_completion([10, 11, 12], logprobs=[-0.1, -0.2, -0.3])),
        env={"SLIME_COMPLETIONS_LOGPROBS": "1"})
    assert turn.output_ids == [10, 11, 12]
    assert turn.output_log_probs == pytest.approx([-0.1, -0.2, -0.3])
    assert len(turn.output_ids) == len(turn.output_log_probs)
    assert turn.prompt_ids == [1, 2, 3]


def test_no_logprobs_requested_leaves_ids_only(monkeypatch):
    """Without SLIME_COMPLETIONS_LOGPROBS, ids still arrive; logprobs empty (vacuous)."""
    saved = _env()
    os.environ.pop("SLIME_COMPLETIONS_LOGPROBS", None)
    try:
        session = _FakeSession(lambda url, kw: _FakeResp(200, _vllm_completion([5, 6])))
        monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        adapter._inbound_auth = {}
        turn, top_raw = asyncio.run(C.call_completions(
            [1, 2], _Session(), {"max_tokens": 4}, adapter=adapter, session_id="c-sid"))
    finally:
        _restore(saved)
    assert turn.output_ids == [5, 6]
    assert turn.output_log_probs == [], "no logprobs requested -> empty (invariant vacuous)"
    assert top_raw is None


def test_stop_reason_mapping(monkeypatch):
    """length -> length, stop -> stop, at the capture boundary."""
    t_len, _, _ = _run(monkeypatch, lambda url, kw: _FakeResp(
        200, _vllm_completion([1], finish="length")))
    assert t_len.finish_reason == "length"
    assert t_len.sequence.stop_reason == "length"

    t_stop, _, _ = _run(monkeypatch, lambda url, kw: _FakeResp(
        200, _vllm_completion([1], finish="stop")))
    assert t_stop.finish_reason == "stop"
    assert t_stop.sequence.stop_reason == "stop"

    # tool_calls is a normal stop from the sampling perspective
    t_tool, _, _ = _run(monkeypatch, lambda url, kw: _FakeResp(
        200, _vllm_completion([1], finish="tool_calls")))
    assert t_tool.sequence.stop_reason == "stop"


def test_logprobs_without_token_ids_dropped(monkeypatch):
    """Server gave logprobs but no ids -> drop logprobs (TITO: no safe id source)."""
    saved = _env(SLIME_COMPLETIONS_LOGPROBS="1")
    try:
        # response has token_logprobs but no token_ids field
        session = _FakeSession(lambda url, kw: _FakeResp(200, {
            "choices": [{"text": "x", "finish_reason": "stop",
                         "logprobs": {"token_logprobs": [-0.1, -0.2]}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 2}}))
        monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        adapter._inbound_auth = {}
        turn, _ = asyncio.run(C.call_completions(
            [1], _Session(), {"max_tokens": 4}, adapter=adapter, session_id="c-sid"))
    finally:
        _restore(saved)
    assert turn.output_ids == [], "no ids returned"
    assert turn.output_log_probs == [], "logprobs must be dropped without paired ids"


def test_mismatched_lengths_raise_at_construction(monkeypatch):
    """Server returns unequal id/logprob counts -> fail at the boundary, not later."""
    saved = _env(SLIME_COMPLETIONS_LOGPROBS="1")
    try:
        session = _FakeSession(lambda url, kw: _FakeResp(200, _vllm_completion(
            [1, 2, 3], logprobs=[-0.1, -0.2])))  # 3 ids, 2 logprobs
        monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        adapter._inbound_auth = {}
        with pytest.raises(ValueError, match="logprobs length"):
            asyncio.run(C.call_completions(
                [1], _Session(), {"max_tokens": 4}, adapter=adapter, session_id="c-sid"))
    finally:
        _restore(saved)


def test_context_cap_short_circuits(monkeypatch):
    """Prompt past max_context_tokens returns length without hitting the network."""
    class _Capped(_Session):
        max_context_tokens = 4

    saved = _env()
    session = _FakeSession(lambda url, kw: _FakeResp(200, _vllm_completion([1])))
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)
    try:
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        adapter._inbound_auth = {}
        turn, _ = asyncio.run(C.call_completions(
            [1, 2, 3, 4, 5], _Capped(), {"max_tokens": 16}, adapter=adapter, session_id="c-sid"))
    finally:
        _restore(saved)
    assert turn.finish_reason == "length"
    assert turn.output_ids == []
    assert session.posts == [], "must not hit the network when prompt overflows"


def test_upstream_error_raises(monkeypatch):
    """A 4xx/5xx surfaces instead of yielding an empty turn."""
    with pytest.raises(RuntimeError, match="completions upstream 503"):
        _run(monkeypatch, lambda url, kw: _FakeResp(503, {"error": "down"}))


# --------------------------------------------------------------------------- #
# CompletionsUpstream.sample: dispatch + message_level
# --------------------------------------------------------------------------- #


def test_completions_is_token_level_and_routed(monkeypatch):
    """message_level=False; sample() builds a SamplingResult through call_completions."""
    saved = _env()
    session = _FakeSession(lambda url, kw: _FakeResp(200, _vllm_completion([7, 8])))
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)
    try:
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        adapter._inbound_auth = {}
        assert adapter.upstream.message_level is False
        r = asyncio.run(adapter.upstream.sample(
            [{"role": "user", "content": "hi"}], None, {"max_tokens": 4}, _Session(), "c-sid"))
    finally:
        _restore(saved)
    assert r.content_blocks is None, "token-level: no content blocks"
    assert r.turn.output_ids == [7, 8]
    assert r.top_logprobs is None  # no top-k requested


def test_sample_topk_logprobs_wired_end_to_end(monkeypatch):
    """Env flags -> vLLM top_logprobs shape -> tokenizer str->id mapping ->
    SamplingResult.top_logprobs (TopkLogprobs), end to end through sample()."""
    k = 3
    # vLLM completions top_logprobs: list[dict[token_str -> logprob]] per position.
    payload = _vllm_completion(
        [10, 13],
        logprobs=[-0.1, -0.2],
        topk=[{"foo": -0.30, "bar": -0.40, "baz": -0.50},
              {"qux": -0.60, "quux": -0.70, "corge": -0.80}])
    session = _FakeSession(lambda url, kw: _FakeResp(200, payload))
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: session)
    saved = _env(SLIME_COMPLETIONS_LOGPROBS="1",
                 SLIME_COMPLETIONS_TOP_LOGPROBS=str(k))
    try:
        adapter = AnthropicAdapter(tokenizer=_FakeTok(), sglang_url=None)
        adapter._inbound_auth = {}
        r = asyncio.run(adapter.upstream.sample(
            [{"role": "user", "content": "hi"}], None, {"max_tokens": 4}, _Session(), "c-sid"))
    finally:
        _restore(saved)

    tp = r.top_logprobs
    assert tp is not None, "top-k requested -> SamplingResult.top_logprobs set"
    assert len(tp.token_ids) == 2           # two output positions
    assert len(tp.logprobs) == 2
    for row in tp.token_ids:
        assert len(row) == k                 # padded to width k
    # _FakeTok maps foo->11751, bar->13, qux->760; unknown strings -> unk(0)
    assert tp.token_ids[0] == [11751, 13, 0]   # foo, bar, baz(unmapped)
    assert tp.logprobs[0] == pytest.approx([-0.30, -0.40, -0.50])
    assert tp.token_ids[1][0] == 760           # qux
    assert tp.logprobs[1][0] == pytest.approx(-0.60)


# --------------------------------------------------------------------------- #
# _completions_topk_to_typed: sentinel padding + string->id mapping
# --------------------------------------------------------------------------- #


class _FakeTok:
    """convert_tokens_to_ids: known strings map to ints; unknown -> unk (0)."""
    unk_id = 0

    def convert_tokens_to_ids(self, tok_str):
        return {" Paris": 11751, ".": 13, "\n": 198,
                "foo": 11751, "bar": 13, "baz": 0,
                "qux": 760, "quux": 1, "corge": 2}.get(tok_str, 0)

    def apply_chat_template(self, messages, *, tools=None, tokenize=True,
                            add_generation_prompt=True):
        # minimal: return a fixed prompt-id list so _render_token_ids works
        return [1, 2, 3]


def test_topk_k_zero_returns_none():
    assert C._completions_topk_to_typed([{"a": -1.0}], 0, _FakeTok()) is None
    assert C._completions_topk_to_typed(None, 3, _FakeTok()) is None


def test_topk_sentinel_pads_short_positions():
    raw = [{" Paris": -0.5, ".": -2.0}]  # 2 alternatives, k=5
    out = C._completions_topk_to_typed(raw, 5, _FakeTok())
    assert out is not None
    assert len(out.token_ids) == 1 and len(out.logprobs) == 1
    assert len(out.token_ids[0]) == 5  # padded to width k
    # known tokens keep their ids; the rest are sentinel-padded
    assert out.token_ids[0][:2] == [11751, 13]
    assert out.token_ids[0][2:] == [0, 0, 0]
    # real logprobs kept for known; sentinel for the padding
    assert out.logprobs[0][:2] == pytest.approx([-0.5, -2.0])
    assert out.logprobs[0][2:] == [MASK_LOGPROB, MASK_LOGPROB, MASK_LOGPROB]


def test_topk_unmapped_string_takes_sentinel_id():
    raw = [{"☃": -0.7}]  # snowman, not in the fake vocab
    out = C._completions_topk_to_typed(raw, 1, _FakeTok())
    assert out.token_ids[0] == [0]  # unmapped -> sentinel id
    assert out.logprobs[0] == pytest.approx([-0.7])  # real logprob kept


def test_topk_empty_position_padded():
    raw = [{}]  # position with no alternatives
    out = C._completions_topk_to_typed(raw, 3, _FakeTok())
    assert out.token_ids[0] == [0, 0, 0]
    assert out.logprobs[0] == [MASK_LOGPROB, MASK_LOGPROB, MASK_LOGPROB]


def test_topk_no_tokenizer_all_sentinel_ids():
    """Without a tokenizer we can't map strings; every id becomes sentinel."""
    raw = [{" Paris": -0.5}]
    out = C._completions_topk_to_typed(raw, 1, None)
    assert out.token_ids[0] == [0]
    assert out.logprobs[0] == pytest.approx([-0.5])


# --------------------------------------------------------------------------- #
# CLI config
# --------------------------------------------------------------------------- #


def test_cli_completions_mode_requires_base_url():
    from anyharness.cli import Config

    saved = {k: os.environ.get(k) for k in ENV}
    os.environ.update(UPSTREAM_MODE="completions", PROMPT="do a thing")
    os.environ.pop("SLIME_COMPLETIONS_BASE_URL", None)
    try:
        with pytest.raises(ValueError, match="SLIME_COMPLETIONS_BASE_URL"):
            Config().validate()
        os.environ["SLIME_COMPLETIONS_BASE_URL"] = "http://localhost:30001/v1"
        Config().validate()  # now OK
    finally:
        _restore(saved)


def test_cli_completions_accepts_no_model_path():
    """No MODEL_PATH required: server /tokenize can render."""
    from anyharness.cli import Config

    saved = {k: os.environ.get(k) for k in (*ENV, "MODEL_PATH")}
    os.environ.update(UPSTREAM_MODE="completions", PROMPT="do a thing",
                      SLIME_COMPLETIONS_BASE_URL="http://localhost:30001/v1")
    os.environ.pop("MODEL_PATH", None)
    try:
        Config().validate()  # no MODEL_PATH, no error
    finally:
        _restore(saved)
