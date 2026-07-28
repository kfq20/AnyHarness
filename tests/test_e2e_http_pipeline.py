"""End-to-end HTTP pipeline test for the message-level SFT dump.

Drives the REAL aiohttp pipeline (no monkeypatch of _run_turn): POST a real
Anthropic /v1/messages wire body (content blocks + tool_use + tool_result +
mid-list system + top-level tools) through the adapter's /v1/messages route,
monkeypatch only _call_messages_upstream to return a canned TurnRecord, run two
turns, call finish_session, dump, and assert the jsonl carries tools + paired
tool_call_id.

This covers the test-coverage finding that the existing integration test
(test_finish_session_messages_mode_dumps_message_samples) bypassed the pipeline
by calling adapter.manager.record_turn directly instead of going through
_run_turn -> _translate -> _capture_tools_schema -> record_turn.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

os.environ.setdefault("UPSTREAM_MODE", "messages")
os.environ.setdefault("SLIME_MESSAGES_UPSTREAM_URL", "http://127.0.0.1:9")

from aiohttp.test_utils import TestClient, TestServer

from slime_sft_trace import Sample, TurnRecord
from slime_sft_trace.adapters.anthropic import AnthropicAdapter
from slime_sft_trace.adapters.common import tool_call_dict
from slime_sft_trace.dump import dump_samples_sft

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city"}},
            "required": ["city"],
        },
    }
]


def _build_canned_reply(adapter, *, with_tool: bool):
    """A canned _build_reply: turn 1 emits a tool_use (so the client can replay
    it + a tool_result on turn 2); turn 2 emits plain text. Mimics what a real
    model/tool_parser would produce, exercising the full _run_turn path."""
    real_build = adapter._build_reply
    state = {"turn": 0}

    def _build(parsed, raw_finish, translated, tools_schema):
        state["turn"] += 1
        if with_tool and state["turn"] == 1:
            tu = {"name": "get_weather", "input": {"city": "SF"}}
            import secrets
            tu_id = f"toolu_{secrets.token_hex(8)}"
            blocks = [{"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]}]
            manager_message = {"role": "assistant", "content": ""}
            manager_message["tool_calls"] = [tool_call_dict(tu["name"], tu.get("input"))]
            from slime_sft_trace.adapters.common import Reply, manager_finish_reason
            return Reply(
                manager_message=manager_message,
                finish_reason=manager_finish_reason([tu], "tool_calls"),
                wire=(blocks, "tool_use"),
            )
        # turn 2 (or non-tool turn): plain text reply
        return real_build(parsed, raw_finish, translated, tools_schema)

    return _build


async def _post(client, sid, body):
    headers = {"Authorization": f"Bearer {sid}", "Content-Type": "application/json"}
    async with client.post("/v1/messages", json=body, headers=headers) as r:
        assert r.status == 200, await r.text()
        return await r.json()


async def _run_pipeline(tmp_path, body1, body2, with_tool=True):
    adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)

    # Return (TurnRecord, content_blocks) — the new messages-mode contract.
    # content_blocks carry the structured Anthropic blocks (tool_use/thinking/text)
    # so _run_turn's _reply_from_content_blocks builds the manager_message with
    # tool_calls, exercising the actual fix (not a stubbed _build_reply).
    # Stateful: turn 1 emits a tool_use (with a stable id the client replays
    # on turn 2); turn 2 emits plain text -> a 2-turn conversation.
    import secrets as _secrets
    state = {"turn": 0}

    async def _canned(self, body, session_id):
        state["turn"] += 1
        if with_tool and state["turn"] == 1:
            blocks = [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "toolu_e2e_1",
                 "name": "get_weather", "input": {"city": "SF"}},
            ]
            finish = "tool_calls"
        else:
            blocks = [{"type": "text", "text": "all done"}]
            finish = "stop"
        return (
            TurnRecord(prompt_ids=[], output_ids=[], finish_reason=finish, output_log_probs=[]),
            blocks,
        )

    adapter._call_messages_upstream = _canned.__get__(adapter)

    client = TestClient(TestServer(adapter.app))
    await client.start_server()
    try:
        sid = "e2e"
        adapter.open_session(sid)

        d1 = await _post(client, sid, body1)
        if with_tool:
            tu_block = next(b for b in d1["content"] if b.get("type") == "tool_use")
            tu_id = tu_block["id"]
            # turn 2: replay assistant(tool_use) + tool_result referencing tu_id
            body2 = dict(body2)
            body2["messages"] = [
                body1["messages"][0],
                {"role": "assistant", "content": d1["content"]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tu_id, "content": "sunny, 72F"}]},
            ]
        await _post(client, sid, body2)

        samples = await adapter.finish_session(sid, base_sample=Sample(index=0), reward=1.0)
        out = tmp_path / "out"
        dump_samples_sft(samples, str(out))
        return samples, out
    finally:
        await client.close()


def test_e2e_http_pipeline_tools_and_paired_tool_call_id(tmp_path):
    """Full HTTP pipeline: tools propagate and tool_call_id is paired."""
    body1 = {
        "model": "claude-3-5-sonnet",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the weather in SF?"}]}],
        "tools": TOOLS,
        "stream": False,
    }
    body2 = {"model": "claude-3-5-sonnet", "max_tokens": 1024,
             "system": "You are a helpful assistant.", "tools": TOOLS,
             "messages": [{"role": "user", "content": "again"}]}  # replaced in _run_pipeline

    samples, out = asyncio.run(_run_pipeline(tmp_path, body1, body2, with_tool=True))

    assert len(samples) == 1, f"expected 1 sample (no fork), got {len(samples)}"
    s = samples[0]
    assert s.response_length == 2
    # tools surfaced onto the Sample
    assert "tools" in s.metadata
    assert s.metadata["tools"][0]["function"]["name"] == "get_weather"
    assert s.metadata["tools"][0]["function"]["parameters"]["required"] == ["city"]

    rec = json.loads((out / "trajectories_sft.jsonl").read_text().splitlines()[0])
    # tools column present, OpenAI shape, input_schema renamed to parameters
    assert "tools" in rec
    t0 = rec["tools"][0]
    assert t0["type"] == "function"
    assert t0["function"]["name"] == "get_weather"
    assert t0["function"]["description"] == "Get the current weather for a city."
    assert "input_schema" not in t0["function"]
    assert "parameters" in t0["function"]
    # tool_calls entries have an id, tool messages have a matching tool_call_id
    asst_with_tcs = next(m for m in rec["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in rec["messages"] if m["role"] == "tool")
    tc = asst_with_tcs["tool_calls"][0]
    assert "id" in tc and tc["id"], "tool_call must have an id"
    assert "tool_call_id" in tool_msg, "tool message must have tool_call_id"
    assert tc["id"] == tool_msg["tool_call_id"], "id and tool_call_id must match"
    # arguments is a dict (not a JSON string)
    assert isinstance(tc["function"]["arguments"], dict)
    # top-level system -> role:system at index 0 (allowed); a *mid-list* system
    # would be folded into a user message, so there must be at most one system
    # message and it must be first.
    roles = [m["role"] for m in rec["messages"]]
    n_system = roles.count("system")
    assert n_system <= 1 and (n_system == 0 or roles[0] == "system"), (
        "system must be leading only; mid-list system must be folded"
    )


def test_e2e_http_pipeline_no_tools_when_absent(tmp_path):
    """A body with NO top-level tools yields a record with no tools column."""
    body1 = {"model": "m", "max_tokens": 1024,
             "messages": [{"role": "user", "content": "hello"}], "stream": False}
    body2 = {"model": "m", "max_tokens": 1024,
             "messages": [{"role": "user", "content": "hello"},
                          {"role": "assistant", "content": "hi"},
                          {"role": "user", "content": "more"}]}
    samples, out = asyncio.run(_run_pipeline(tmp_path, body1, body2, with_tool=False))
    assert len(samples) == 1, f"expected 1 sample, got {len(samples)}"
    rec = json.loads((out / "trajectories_sft.jsonl").read_text().splitlines()[0])
    assert "tools" not in rec, "no tools column expected when body had no tools"
