"""Tinker/Mint ``asample`` upstream: native token-in token-out.

Shapes here mirror a live probe of a Mint gateway (``areal-mint``, Qwen3.6-35B-A3B):

    POST /api/v1/asample   -> {"request_id": "..."}
    POST /api/v1/retrieve_future
        -> {"sequences": [{"tokens": [...], "logprobs": [...],
                           "routed_experts": null, "stop_reason": "length"}],
            "prompt_logprobs": [null, -5.95, ...], "topk_prompt_logprobs": null,
            "type": "sample"}

The point of this mode is that ``tokens`` and ``logprobs`` come back paired from
the server and the prompt goes out as ids, so nothing is ever re-tokenized.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("UPSTREAM_MODE", "sglang")

from anyharness.adapters import common as C  # noqa: E402
from anyharness.adapters.anthropic import AnthropicAdapter  # noqa: E402

# one real capture: 16 sampled tokens with 16 matching logprobs
LIVE_TOKENS = [198, 71093, 12305, 198, 2, 257, 653, 257, 653, 257, 653, 257, 653, 257, 653, 257]
LIVE_LOGPROBS = [-1.9855, -3.0521, -1.1766, -0.0031, -2.1309, -3.0902, -2.7960, -2.5778,
                 -1.3289, -1.0616, -1.0152, -0.6392, -0.9024, -0.4210, -0.6943, -0.3957]

TINKER_ENV = ("UPSTREAM_MODE", "TINKER_BASE_URL", "TINKER_API_KEY", "TINKER_BASE_MODEL",
              "TINKER_MODEL_ID", "TINKER_PROMPT_LOGPROBS", "MODEL_PATH", "PROMPT")


class _Session:
    """Stands in for the adapter's per-sid session state."""

    sampling_defaults: dict = {}
    max_context_tokens = 0


def _install_env(**extra) -> dict:
    saved = {k: os.environ.get(k) for k in TINKER_ENV}
    os.environ.update(UPSTREAM_MODE="tinker", TINKER_BASE_URL="http://mint.invalid:28000",
                      TINKER_API_KEY="test-key", TINKER_BASE_MODEL="Qwen/Qwen3.6-35B-A3B", **extra)
    return saved


def _restore_env(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _FakePost:
    """Async-context-manager stand-in for ``ClientSession.post``."""

    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self, content_type=None):
        return self._payload

    async def text(self):
        return str(self._payload)


class _FakeSession:
    """Records posted bodies and replays scripted replies per endpoint."""

    def __init__(self, script: dict) -> None:
        self.script = script
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json or {}))
        for suffix, (status, payload) in self.script.items():
            if url.endswith(suffix):
                if callable(payload):
                    payload = payload(len([c for c in self.calls if c[0].endswith(suffix)]))
                return _FakePost(status, payload)
        raise AssertionError(f"unexpected POST {url}")


def _run_tinker(monkeypatch, script, *, body=None, session=None, prompt_ids=(1, 2, 3, 4)):
    fake = _FakeSession(script)
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: fake)
    adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
    turn, top_k = asyncio.run(C.call_tinker_sample(
        list(prompt_ids), session or _Session(), body or {"max_tokens": 16},
        adapter=adapter, session_id="tk-sid",
    ))
    return turn, top_k, fake


def test_tinker_sends_token_ids_and_returns_paired_logprobs(monkeypatch):
    """Prompt goes out as ids; tokens+logprobs come back paired 1:1."""
    saved = _install_env()
    try:
        turn, top_k, fake = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-1"}),
            "/api/v1/retrieve_future": (200, {
                "sequences": [{"tokens": LIVE_TOKENS, "logprobs": LIVE_LOGPROBS,
                               "routed_experts": None, "stop_reason": "length"}],
                "prompt_logprobs": None, "type": "sample",
            }),
        }, prompt_ids=[9707, 11, 1879, 0])
    finally:
        _restore_env(saved)

    submit_url, submit_body = fake.calls[0]
    assert submit_url.endswith("/api/v1/asample")
    # the whole point: raw ids on the wire, no text
    assert submit_body["prompt"]["chunks"][0]["tokens"] == [9707, 11, 1879, 0]
    assert submit_body["prompt"]["chunks"][0]["type"] == "encoded_text"
    assert submit_body["num_samples"] == 1
    assert submit_body["base_model"] == "Qwen/Qwen3.6-35B-A3B"
    assert fake.calls[1][1] == {"request_id": "rid-1"}

    assert turn.output_ids == LIVE_TOKENS
    assert turn.output_log_probs == pytest.approx(LIVE_LOGPROBS)
    # record_turn asserts on this
    assert len(turn.output_ids) == len(turn.output_log_probs)
    assert turn.prompt_ids == [9707, 11, 1879, 0]
    assert turn.finish_reason == "length"
    assert top_k is None


def test_tinker_survives_record_turn(monkeypatch):
    """The pairing must satisfy TrajectoryManager.record_turn's length assert."""
    from anyharness.trajectory import TrajectoryManager

    saved = _install_env()
    try:
        turn, _, _ = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-2"}),
            "/api/v1/retrieve_future": (200, {"sequences": [
                {"tokens": LIVE_TOKENS, "logprobs": LIVE_LOGPROBS, "stop_reason": "stop"}
            ]}),
        })
    finally:
        _restore_env(saved)

    mgr = TrajectoryManager()
    # would raise AssertionError if tokens/logprobs were misaligned
    mgr.record_turn(
        "tk-sid",
        turn=turn,
        prompt_messages=[{"role": "user", "content": "hi"}],
        response_message={"role": "assistant", "content": "ok"},
    )


def test_tinker_model_id_takes_precedence(monkeypatch):
    """TINKER_MODEL_ID selects a training step instead of the base model."""
    saved = _install_env(TINKER_MODEL_ID="sess-21e8b59b_0", TINKER_PROMPT_LOGPROBS="1")
    try:
        _, _, fake = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-3"}),
            "/api/v1/retrieve_future": (200, {"sequences": [
                {"tokens": [1, 2], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}
            ]}),
        })
    finally:
        _restore_env(saved)

    body = fake.calls[0][1]
    assert body["model_id"] == "sess-21e8b59b_0"
    assert "base_model" not in body, "model_id and base_model are mutually exclusive"
    assert body["prompt_logprobs"] is True
    assert body["include_prompt_logprobs"] is True


def test_tinker_requests_logprobs_by_default(monkeypatch):
    """Verified live: prompt_logprobs also gates the sampled logprobs, so it must
    default on or token-level data comes back silently empty."""
    saved = _install_env()
    os.environ.pop("TINKER_LOGPROBS", None)
    try:
        _, _, fake = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-lp"}),
            "/api/v1/retrieve_future": (200, {"sequences": [
                {"tokens": [1, 2], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}
            ]}),
        })
    finally:
        _restore_env(saved)

    body = fake.calls[0][1]
    assert body["prompt_logprobs"] is True
    assert body["include_prompt_logprobs"] is True


def test_tinker_logprobs_can_be_disabled(monkeypatch):
    """TINKER_LOGPROBS=0 opts out for message-level-only runs."""
    saved = _install_env(TINKER_LOGPROBS="0")
    try:
        _, _, fake = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-nolp"}),
            "/api/v1/retrieve_future": (200, {"sequences": [
                {"tokens": [1, 2], "logprobs": [], "stop_reason": "stop"}
            ]}),
        })
    finally:
        _restore_env(saved)

    assert "prompt_logprobs" not in fake.calls[0][1]


def test_tinker_warns_on_tokenizer_family_mismatch(monkeypatch, caplog):
    """A tokenizer from another model family silently corrupts the rendered prompt."""
    class _Tok:
        name_or_path = "meta-llama/Llama-3-8B"

    C._tinker_mismatch_warned.clear()
    saved = _install_env()
    fake = _FakeSession({
        "/api/v1/asample": (200, {"request_id": "rid-mm"}),
        "/api/v1/retrieve_future": (200, {"sequences": [
            {"tokens": [1], "logprobs": [-0.1], "stop_reason": "stop"}
        ]}),
    })
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: fake)
    try:
        adapter = AnthropicAdapter(tokenizer=_Tok(), sglang_url=None)
        with caplog.at_level("WARNING"):
            asyncio.run(C.call_tinker_sample(
                [1, 2], _Session(), {"max_tokens": 8}, adapter=adapter, session_id="mm"))
    finally:
        _restore_env(saved)
        C._tinker_mismatch_warned.clear()

    assert "may not match served model" in caplog.text


def test_tinker_no_warning_on_matching_family(monkeypatch, caplog):
    """The same family must not trip the heuristic."""
    class _Tok:
        name_or_path = "/models/Qwen3.6-35B-A3B"

    C._tinker_mismatch_warned.clear()
    saved = _install_env()
    fake = _FakeSession({
        "/api/v1/asample": (200, {"request_id": "rid-ok"}),
        "/api/v1/retrieve_future": (200, {"sequences": [
            {"tokens": [1], "logprobs": [-0.1], "stop_reason": "stop"}
        ]}),
    })
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: fake)
    try:
        adapter = AnthropicAdapter(tokenizer=_Tok(), sglang_url=None)
        with caplog.at_level("WARNING"):
            asyncio.run(C.call_tinker_sample(
                [1, 2], _Session(), {"max_tokens": 8}, adapter=adapter, session_id="ok"))
    finally:
        _restore_env(saved)
        C._tinker_mismatch_warned.clear()

    assert "may not match served model" not in caplog.text


def test_tinker_polls_until_ready(monkeypatch):
    """A future without `sequences` means not-ready; keep polling."""
    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(C.asyncio, "sleep", _no_sleep)
    saved = _install_env()

    def _future(nth_call):
        if nth_call < 3:
            return {"type": "sample"}  # still pending
        return {"sequences": [{"tokens": [5, 6], "logprobs": [-0.5, -0.6], "stop_reason": "stop"}]}

    try:
        turn, _, fake = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-4"}),
            "/api/v1/retrieve_future": (200, _future),
        })
    finally:
        _restore_env(saved)

    polls = [c for c in fake.calls if c[0].endswith("/retrieve_future")]
    assert len(polls) == 3, "must poll until the result materialises"
    assert turn.output_ids == [5, 6]


def test_tinker_drops_logprobs_on_count_mismatch(monkeypatch):
    """If the server ever returns unequal lengths, drop rather than emit bad pairs."""
    saved = _install_env()
    try:
        turn, _, _ = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-5"}),
            "/api/v1/retrieve_future": (200, {"sequences": [
                {"tokens": [1, 2, 3], "logprobs": [-0.1, -0.2], "stop_reason": "stop"}
            ]}),
        })
    finally:
        _restore_env(saved)

    assert turn.output_ids == [1, 2, 3], "ids are still the truth of what was sampled"
    assert turn.output_log_probs == [], "misaligned logprobs must not ship"


def test_tinker_upstream_error_raises(monkeypatch):
    """A 4xx/5xx from asample surfaces instead of yielding an empty turn."""
    saved = _install_env()
    try:
        with pytest.raises(RuntimeError, match="tinker asample 503"):
            _run_tinker(monkeypatch, {
                "/api/v1/asample": (503, {"detail": "No inference backend for model"}),
            })
    finally:
        _restore_env(saved)


def test_tinker_context_cap_short_circuits(monkeypatch):
    """A prompt past max_context_tokens returns length without calling upstream."""
    class _Capped(_Session):
        max_context_tokens = 4

    saved = _install_env()
    fake = _FakeSession({})
    monkeypatch.setattr(C.aiohttp, "ClientSession", lambda *a, **k: fake)
    try:
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        turn, top_k = asyncio.run(C.call_tinker_sample(
            [1, 2, 3, 4, 5], _Capped(), {"max_tokens": 16}, adapter=adapter, session_id="tk-sid",
        ))
    finally:
        _restore_env(saved)

    assert turn.finish_reason == "length"
    assert turn.output_ids == []
    assert fake.calls == [], "must not hit the network when the prompt already overflows"


def test_tinker_sampling_params_translated(monkeypatch):
    """Tinker takes max_tokens, not sglang's max_new_tokens + detokenize knobs."""
    saved = _install_env()
    try:
        _, _, fake = _run_tinker(monkeypatch, {
            "/api/v1/asample": (200, {"request_id": "rid-6"}),
            "/api/v1/retrieve_future": (200, {"sequences": [
                {"tokens": [1], "logprobs": [-0.1], "stop_reason": "stop"}
            ]}),
        }, body={"max_tokens": 32, "temperature": 0.7, "top_p": 0.9})
    finally:
        _restore_env(saved)

    sp = fake.calls[0][1]["sampling_params"]
    assert sp["max_tokens"] == 32
    assert sp["temperature"] == 0.7
    assert sp["top_p"] == 0.9
    for dead in ("max_new_tokens", "skip_special_tokens", "spaces_between_special_tokens", "no_stop_trim"):
        assert dead not in sp, f"{dead} is sglang-only and meaningless to tinker"


def test_cli_tinker_mode_requires_model_path_and_url():
    """tinker renders locally, so MODEL_PATH and TINKER_BASE_URL are both required."""
    from anyharness.cli import Config

    saved = {k: os.environ.get(k) for k in TINKER_ENV}
    os.environ.update(UPSTREAM_MODE="tinker", PROMPT="do a thing",
                      TINKER_BASE_URL="http://mint.invalid:28000",
                      TINKER_BASE_MODEL="Qwen/Qwen3.6-35B-A3B")
    os.environ.pop("MODEL_PATH", None)
    try:
        with pytest.raises(ValueError, match="MODEL_PATH"):
            Config().validate()
        os.environ["MODEL_PATH"] = "/fake/model"
        os.environ.pop("TINKER_BASE_URL", None)
        with pytest.raises(ValueError, match="TINKER_BASE_URL"):
            Config().validate()
        os.environ["TINKER_BASE_URL"] = "http://mint.invalid:28000"
        os.environ.pop("TINKER_BASE_MODEL", None)
        with pytest.raises(ValueError, match="TINKER_MODEL_ID"):
            Config().validate()
    finally:
        _restore_env(saved)
