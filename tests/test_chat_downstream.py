"""E2E: OpenAI chat-completions harness → HiLR → trajectory dump.

Proves the /v1/chat/completions downstream works: an OpenAI-format harness
(like OpenAI Agents SDK / smolagents) drives a multi-turn tool loop through
HiLR, and the trajectory is captured correctly. Both downstream and upstream
use chat-completions wire format.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket

import aiohttp
import pytest
from aiohttp import web

from anyharness import Sample
from anyharness.adapters import AnthropicAdapter


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def _fake_chat_upstream(request: web.Request) -> web.Response:
    """Turn 1: tool_call. Turn 2 (has tool message): text answer."""
    body = await request.json()
    has_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
    if not has_tool:
        return web.json_response({"choices": [{"message": {
            "role": "assistant", "content": "checking",
            "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "Read", "arguments": '{"file_path":"/tmp/x.py"}'}}],
            "reasoning_content": "I should read it"},
            "finish_reason": "tool_calls"}]})
    return web.json_response({"choices": [{"message": {
        "role": "assistant", "content": "it defines add()"}, "finish_reason": "stop"}]})


async def _run_chat_downstream() -> list:
    up_p, ad_p = _free_port(), _free_port()
    up_app = web.Application()
    up_app.router.add_post("/v1/chat/completions", _fake_chat_upstream)
    up_runner = web.AppRunner(up_app)
    await up_runner.setup()
    await web.TCPSite(up_runner, "127.0.0.1", up_p).start()

    # Save/restore env so we don't pollute other tests (UPSTREAM_MODE is global).
    _saved = {k: os.environ.get(k) for k in
              ("UPSTREAM_MODE", "SLIME_CHAT_BASE_URL", "SLIME_CHAT_API_KEY", "SLIME_CHAT_MODEL", "SLIME_SESSION_ID")}
    os.environ.update(UPSTREAM_MODE="chat",
                      SLIME_CHAT_BASE_URL=f"http://127.0.0.1:{up_p}",
                      SLIME_CHAT_API_KEY="test", SLIME_CHAT_MODEL="fake")

    adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
    ad_runner = web.AppRunner(adapter.app)
    await ad_runner.setup()
    await web.TCPSite(ad_runner, "127.0.0.1", ad_p).start()
    adapter.open_session("chat-e2e")
    os.environ["SLIME_SESSION_ID"] = "chat-e2e"

    try:
        async with aiohttp.ClientSession() as sess:
            # turn 1: tool call
            b1 = {"model": "fake", "max_tokens": 100,
                  "messages": [{"role": "user", "content": "read /tmp/x.py"}],
                  "tools": [{"type": "function", "function": {"name": "Read", "parameters": {"type": "object"}}}]}
            r1 = await (await sess.post(f"http://127.0.0.1:{ad_p}/v1/chat/completions", json=b1)).json()
            m1 = r1["choices"][0]["message"]
            assert m1.get("tool_calls"), "turn 1 should return tool_calls"
            assert m1.get("reasoning_content"), "turn 1 should return reasoning_content"
            assert r1["choices"][0]["finish_reason"] == "tool_calls"

            # turn 2: replay + tool_result
            b2 = {"model": "fake", "max_tokens": 100, "messages": [
                {"role": "user", "content": "read /tmp/x.py"},
                {"role": "assistant", "content": "checking", "tool_calls": m1["tool_calls"]},
                {"role": "tool", "tool_call_id": "call_1", "content": "def add(a,b): return a+b"}]}
            r2 = await (await sess.post(f"http://127.0.0.1:{ad_p}/v1/chat/completions", json=b2)).json()
            assert r2["choices"][0]["finish_reason"] == "stop"
            assert "add" in r2["choices"][0]["message"]["content"]

        samples = await adapter.finish_session("chat-e2e", base_sample=Sample(index=0), reward=0.0)
        return samples
    finally:
        await ad_runner.cleanup()
        await up_runner.cleanup()
        # Restore env so other tests aren't polluted by UPSTREAM_MODE=chat.
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_chat_downstream_multi_turn_tool_loop():
    """OpenAI-format harness drives a 2-turn tool loop via /v1/chat/completions."""
    samples = asyncio.run(_run_chat_downstream())
    assert len(samples) == 1, f"expected 1 sample, got {len(samples)}"
    roles = [m["role"] for m in samples[0].prompt]
    assert roles == ["user", "assistant", "tool", "assistant"], f"roles: {roles}"
    assert samples[0].metadata["use_tool"] is True
    # assistant turn 1 has tool_calls; the chat wire format serializes
    # arguments to a JSON string (the OpenAI live-API shape), so a client
    # that json.loads() it recovers the dict.
    asst1 = samples[0].prompt[1]
    assert asst1.get("tool_calls")
    args = asst1["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)  # wire shape
    assert json.loads(args) == {"file_path": "/tmp/x.py"}
    assert asst1["tool_calls"][0]["id"]  # id required to pair the tool result
    # tool message has tool_call_id + name; the id is synthesized by the adapter
    # (the upstream's wire id is dropped, per tool_call_dict's tree-matching
    # invariant) and must match the assistant turn's tool_call id above.
    tool_msg = samples[0].prompt[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg.get("tool_call_id")
    assert tool_msg.get("tool_call_id") == asst1["tool_calls"][0]["id"]
    assert tool_msg.get("name") == "Read"
