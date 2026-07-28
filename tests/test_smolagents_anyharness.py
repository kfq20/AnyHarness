"""smolagents → AnyHarness chat downstream → mintcn upstream → SFT trace.

Runs aiohttp server in a background thread so smolagents' sync HTTP calls
don't deadlock with the asyncio event loop.
"""
import asyncio
import os
import sys
import socket
import threading
import time

PROJ = "/root/.claude/jobs/203771d4/tmp/slime-sft-trace"
sys.path.insert(0, PROJ + "/src")

MINTCN_KEY = "REDACTED_MINTCN_KEY"
MINTCN_URL = "https://mintcn.macaron.xin"
MODEL = "macaron-v1-coding-venti"


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def start_adapter_in_thread(port, sid):
    """Start the AnyHarness aiohttp server in a background event loop thread."""
    from aiohttp import web
    from slime_sft_trace.adapters import AnthropicAdapter

    os.environ.update(
        UPSTREAM_MODE="chat",
        SLIME_CHAT_BASE_URL=MINTCN_URL,
        SLIME_CHAT_API_KEY=MINTCN_KEY,
        SLIME_CHAT_MODEL=MODEL,
        SLIME_SESSION_ID=sid,
    )
    adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
    adapter.open_session(sid)

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    stop_event = asyncio.Event()

    async def serve():
        runner = web.AppRunner(adapter.app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        ready.set()
        await stop_event.wait()
        await runner.cleanup()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve())

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    ready.wait(timeout=5)

    def stop():
        loop.call_soon_threadsafe(stop_event.set)

    async def finish(base_sample, reward):
        fut = asyncio.run_coroutine_threadsafe(
            adapter.finish_session(sid, base_sample=base_sample, reward=reward), loop
        )
        return fut.result(timeout=10)

    return stop, adapter, loop


if __name__ == "__main__":
    port = free_port()
    sid = "smolagents-anyharness"
    stop_server, adapter, loop = start_adapter_in_thread(port, sid)

    try:
        from smolagents import OpenAIModel, ToolCallingAgent

        model = OpenAIModel(
            model_id=MODEL,
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="dummy",
        )
        agent = ToolCallingAgent(tools=[], model=model, max_steps=3)

        t0 = time.time()
        result = agent.run("What is 2+2? Reply with just the number.")
        print(f"AGENT RESULT: {result}  time={time.time()-t0:.1f}s")

        from slime_sft_trace import Sample
        fut = asyncio.run_coroutine_threadsafe(
            adapter.finish_session(sid, base_sample=Sample(index=0), reward=0.0),
            loop
        )
        samples = fut.result(timeout=10)
        print(f"SAMPLES: {len(samples)}")
        if samples:
            roles = [m["role"] for m in samples[0].prompt]
            print(f"ROLES: {roles}")
            print(f"TURNS: {len([r for r in roles if r == 'assistant'])}")
    finally:
        stop_server()
