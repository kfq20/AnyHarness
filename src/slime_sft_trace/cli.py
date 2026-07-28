"""CLI entrypoint: run one Claude Code rollout through the trace-capturing adapter.

Mirrors Slime's ``examples/coding_agent_rl/generate.py`` run loop, simplified to
a single process:

1. Load config from env (MODEL_PATH, upstream URL + mode, ports, prompt, output).
2. Build the :class:`AnthropicAdapter` (the other agent's vendored module).
3. Serve its aiohttp ``.app`` on ``ADAPTER_PORT`` in a background thread via
   :class:`aiohttp.web.AppRunner` (so the harness can talk to it from the main
   thread).
4. ``open_session`` -> run the :class:`ClaudeCodeHarness` -> ``finish_session``
   -> :func:`dump.dump_samples``.

Upstream mode is chosen by env ``UPSTREAM_MODE``:
* ``sglang`` (default): adapter talks to an sglang ``/generate`` with logprobs.
  Needs ``MODEL_PATH`` (HF tokenizer) + ``SLIME_SGLANG_URL``.
* ``messages``: adapter forwards to an arbitrary ``/v1/messages`` API upstream.
  Needs ``SLIME_MESSAGES_UPSTREAM_URL``; tokenizer is None.

Sandbox: ``LocalSandbox`` by default; env ``SANDBOX=e2b`` switches to
:class:`E2BSandbox` for real (networked) tasks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
from typing import Any

from .dump import dump_samples
from .harness import ClaudeCodeHarness, LocalSandbox
from .harness.claude_code import E2BSandbox

logger = logging.getLogger("slime_sft_trace.cli")


# ===========================================================================
# Config
# ===========================================================================


class Config:
    """Run configuration, all sourced from environment variables."""

    def __init__(self) -> None:
        self.upstream_mode = os.environ.get("UPSTREAM_MODE", "sglang").strip().lower()
        if self.upstream_mode not in ("sglang", "messages", "chat"):
            raise ValueError(f"UPSTREAM_MODE must be 'sglang', 'messages', or 'chat', got {self.upstream_mode!r}")

        self.model_path = os.environ.get("MODEL_PATH") or None
        self.sglang_url = os.environ.get("SLIME_SGLANG_URL") or None
        self.messages_upstream_url = os.environ.get("SLIME_MESSAGES_UPSTREAM_URL") or None
        self.adapter_port = int(os.environ.get("ADAPTER_PORT", "18080"))
        self.claude_model = os.environ.get("CLAUDE_MODEL", "slime-actor")
        self.prompt = os.environ.get("PROMPT", "")
        self.output_dir = os.environ.get("OUTPUT_DIR", "./trajectories")
        self.sandbox = os.environ.get("SANDBOX", "local").strip().lower()
        self.time_budget_sec = int(os.environ.get("TIME_BUDGET_SEC", "600"))
        self.tool_parser = os.environ.get("TOOL_PARSER") or None
        self.reasoning_parser = os.environ.get("REASONING_PARSER") or None
        self.fork_threshold_tokens = int(os.environ["FORK_THRESHOLD_TOKENS"]) if os.environ.get(
            "FORK_THRESHOLD_TOKENS"
        ) else None

    def validate(self) -> None:
        if self.upstream_mode == "sglang":
            if not self.model_path:
                raise ValueError("MODEL_PATH is required in sglang mode (HF tokenizer path)")
            if not self.sglang_url:
                raise ValueError("SLIME_SGLANG_URL is required in sglang mode")
        elif self.upstream_mode == "messages":
            if not self.messages_upstream_url:
                raise ValueError("SLIME_MESSAGES_UPSTREAM_URL is required in messages mode")
        else:  # chat
            if not os.environ.get("SLIME_CHAT_BASE_URL"):
                raise ValueError("SLIME_CHAT_BASE_URL is required in chat mode")
        if not self.prompt:
            raise ValueError("PROMPT env var is required")


# ===========================================================================
# Adapter server (background thread)
# ===========================================================================


class AdapterServer:
    """Run an aiohttp web.Application in a background thread.

    ``start()`` blocks until the server is listening (so the caller can hit the
    URL); ``stop()`` shuts it down. The adapter object is accessible via
    ``.adapter`` for ``open_session`` / ``finish_session``.
    """

    def __init__(self, adapter: Any, port: int, host: str = "127.0.0.1") -> None:
        self.adapter = adapter
        self.port = port
        self.host = host
        self._runner: Any = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._start_exc: BaseException | None = None
        self._stop: Any = None  # asyncio.Event, set once the loop is running

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, timeout: float = 30.0) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"adapter server did not become ready within {timeout}s")
        if self._start_exc is not None:
            raise self._start_exc

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:  # pragma: no cover - server crash
            self._start_exc = exc
            self._ready.set()
        finally:
            loop.close()

    async def _serve(self) -> None:
        from aiohttp import web

        self._stop = asyncio.Event()  # created on the server loop
        self._runner = web.AppRunner(self.adapter.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self._ready.set()
        # Block until stop() sets the event.
        await self._stop.wait()

    def stop(self) -> None:
        if self._loop is None:
            return

        async def _shutdown() -> None:
            if self._stop is not None:
                self._stop.set()
            if self._runner is not None:
                await self._runner.cleanup()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        try:
            fut.result(timeout=15)
        except Exception:
            logger.exception("adapter server shutdown failed")
        if self._thread is not None:
            self._thread.join(timeout=15)


# ===========================================================================
# Wiring
# ===========================================================================


def load_tokenizer(cfg: Config):
    """Load the HF tokenizer for sglang mode; None in messages mode."""
    if cfg.upstream_mode != "sglang":
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)


def build_adapter(cfg: Config, tokenizer: Any) -> Any:
    """Construct the vendored AnthropicAdapter.

    Adapter API (vendored by the parallel agent):
        AnthropicAdapter(tokenizer=, sglang_url=, tool_parser=,
                         reasoning_parser=, fork_threshold_tokens=)
    The adapter reads ``UPSTREAM_MODE`` and ``SLIME_MESSAGES_UPSTREAM_URL`` from
    the environment itself (inside BaseAdapter.__init__ / _call_messages_upstream),
    so we pass the same kwargs in both modes — ``sglang_url`` is required by the
    signature but unused in messages mode (None is tolerated).
    """
    from slime_sft_trace.adapters import AnthropicAdapter

    return AnthropicAdapter(
        tokenizer=tokenizer,
        sglang_url=cfg.sglang_url,
        tool_parser=cfg.tool_parser,
        reasoning_parser=cfg.reasoning_parser,
        fork_threshold_tokens=cfg.fork_threshold_tokens,
    )


def make_sandbox(cfg: Config):
    if cfg.sandbox == "e2b":
        return E2BSandbox()
    return LocalSandbox()


async def run_once(cfg: Config, *, session_id: str | None = None) -> list:
    """One end-to-end rollout: adapter up -> harness -> finish -> samples.

    A deterministic ``session_id`` may be passed for testing/reproducibility;
    otherwise a random one is generated. The id is also exported to the harness
    environment as ``ADAPTER_AUTH`` so a custom claude binary can echo it back as
    the Authorization Bearer token.
    """
    from slime_sft_trace import Sample  # vendored by the parallel agent

    tokenizer = load_tokenizer(cfg)
    adapter = build_adapter(cfg, tokenizer)

    server = AdapterServer(adapter, cfg.adapter_port)
    server.start()
    logger.info("adapter serving at %s", server.url)

    try:
        session_id = session_id or secrets.token_hex(8)
        adapter.open_session(session_id)
        # Export to the ambient env so a custom claude binary can reach the
        # adapter (the real claude CLI uses ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN
        # passed via the harness command env; ADAPTER_URL/AUTH are for test
        # stand-ins and reproducibility).
        os.environ["ADAPTER_AUTH"] = session_id
        os.environ["ADAPTER_URL"] = server.url
        # The adapter (background thread) routes requests to the trajectory tree
        # by session id; expose it so _request_session_id reads it instead of
        # overloading the auth credential (which must be the real upstream key).
        os.environ["SLIME_SESSION_ID"] = session_id

        harness = ClaudeCodeHarness(model=cfg.claude_model)
        sb = make_sandbox(cfg)
        exit_code = await harness.run(
            sb,
            workdir=os.getcwd(),
            session_id=session_id,
            adapter_url=server.url,
            prompt=cfg.prompt,
            time_budget_sec=cfg.time_budget_sec,
        )
        logger.info("harness exit_code=%d", exit_code)

        samples = await adapter.finish_session(
            session_id, base_sample=Sample(index=0), reward=0.0
        )
        # In messages mode the token-level dump is empty (no logprobs/tokens);
        # also write the message-level SFT trajectory so the conversation is
        # captured. Both dumps are written when samples are available. We write
        # BOTH flavors (glm52 dict-arguments + openai_wire stringified) so users
        # can pick based on their datasets version: glm52 needs datasets>=4.7
        # (or an explicit Json() Features schema); openai_wire loads on
        # datasets<4.7 which infers tool_calls.function.arguments as a string.
        if cfg.upstream_mode in ("messages", "chat"):
            from .dump import dump_samples_sft

            dump_samples_sft(samples, cfg.output_dir, flavor="glm52")
            dump_samples_sft(samples, cfg.output_dir, flavor="openai_wire")
        return samples
    finally:
        server.stop()


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = Config()
    cfg.validate()

    samples = asyncio.run(run_once(cfg))
    out = dump_samples(samples, cfg.output_dir)
    print(f"wrote {len(samples)} sample(s) to {out}")


if __name__ == "__main__":
    main()
