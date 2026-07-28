"""Shared adapter primitives for token-capturing agent rollouts.

A protocol adapter (Anthropic / OpenAI) subclasses BaseAdapter and fills in the
wire-specific hooks (_register_routes, _session_id, _translate, _build_reply,
_respond, and optionally _preprocess_body) plus a few class attributes (logger,
log_prefix, max_token_keys, stop_keys). The session lifecycle, per-sid turn cap,
inflight-task bookkeeping and the one-turn _run_turn pipeline are inherited.

flatten_content, tool_call_dict and manager_finish_reason cover the parts both
protocols handle identically.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from typing import Any

import aiohttp
from aiohttp import web

from slime_sft_trace.parsing import parse_model_output
from slime_sft_trace.trajectory import TrajectoryManager, TurnRecord


__all__ = ["TurnRecord"]


@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults and context budget.

    Trajectory state lives in the shared TrajectoryManager (BaseAdapter.manager),
    not here.
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0


@dataclasses.dataclass
class Reply:
    """Output of an adapter's _build_reply, consumed by _run_turn.

    manager_message and finish_reason feed record_turn and the debug callback;
    wire is opaque to the pipeline and only the adapter's own _respond reads it.
    """

    manager_message: dict
    finish_reason: str
    wire: Any


def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Render a chat-message list to token ids with the served chat template."""
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


def flatten_content(c: Any) -> str:
    """Flatten a wire content value into a chat-template string.

    Handles both Anthropic and OpenAI block shapes. A non-list value (str /
    dict / other) is returned via str() unchanged.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t in {"text", "input_text", "output_text"}:
            parts.append(b.get("text", ""))
        elif t == "tool_result":
            parts.append(flatten_content(b.get("content")))
        elif t in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in b:
            parts.append(flatten_content(b.get("content")))
        elif "text" in b:
            parts.append(str(b.get("text") or ""))
    # join with "\n\n" to match the production converter's content_part_to_text
    # (export_covered_claude_events_to_glm52_sft.py) so dumped multi-block system
    # prompts tokenize identically to production SFT data.
    return "\n\n".join(p for p in parts if p)


def _stringify_tool_call_args(messages: list[dict]) -> list[dict]:
    """Return a copy of ``messages`` with every assistant ``tool_calls[].function.arguments``
    serialized to a JSON string (the OpenAI live-API shape).

    The tree stores arguments as a dict (HF shape), but some OpenAI-compatible
    upstreams reject dict arguments in a replayed assistant message. This is a
    wire-only transform for the chat upstream; the tree is untouched (dict).
    """
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant" or not isinstance(m.get("tool_calls"), list):
            out.append(m)
            continue
        m2 = dict(m)
        tcs = []
        for tc in m["tool_calls"]:
            if isinstance(tc, dict):
                tc2 = dict(tc)
                fn = tc2.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), dict):
                    fn = dict(fn)
                    fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                    tc2["function"] = fn
                tcs.append(tc2)
            else:
                tcs.append(tc)
        m2["tool_calls"] = tcs
        out.append(m2)
    return out


def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call stored on manager_message.

    arguments stays a dict (not a JSON string): the chat template needs a
    mapping, and the trajectory manager matches history by dict equality, so a
    sampled leaf and its replayed echo compare equal regardless of key order.
    The wire-only tool-call id is dropped for the same reason.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}


def manager_finish_reason(tool_uses: list[dict], raw_finish: str) -> str:
    """Finish reason stored on the manager turn: tool_calls if the turn called a
    tool, else the raw sglang finish."""
    return "tool_calls" if tool_uses else (raw_finish or "stop")


class BaseAdapter:
    """Base HTTP adapter: session lifecycle plus the shared one-turn pipeline.

    See the module docstring for the class attributes and hooks a subclass must
    supply; everything else is inherited.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    log_prefix: str = "adapter"
    # body keys that cap max_new_tokens and carry stop sequences, in priority order
    max_token_keys: tuple[str, ...] = ()
    stop_keys: tuple[str, ...] = ()
    manager: Any

    def __init__(
        self,
        *,
        tokenizer,
        sglang_url,
        tool_parser=None,
        reasoning_parser=None,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        debug_callback: Callable[..., None] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        # upstream backend: "sglang" (default) POSTs /generate with return_logprob;
        # "messages" forwards the raw /v1/messages body to an arbitrary messages API.
        mode = os.environ.get("UPSTREAM_MODE", "sglang").strip().lower() or "sglang"
        if mode not in ("sglang", "messages", "chat", "responses"):
            raise ValueError(f"UPSTREAM_MODE={mode!r} must be 'sglang', 'messages', 'chat', or 'responses'")
        self.upstream_mode = mode
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        # tools_schema the model saw at rollout, keyed by sid (first-wins: tools
        # don't change within a session). Surfaced onto message-level SFT samples
        # by get_trajectory_messages so the dumped record keeps the tool defs.
        self._sid_tools: dict[str, list[dict] | None] = {}
        self.app = web.Application(client_max_size=64 * 1024 * 1024)

        # one manager shared across all sids; per-sid trees live inside it.
        # fork_threshold_tokens left None means the manager uses its own default.
        mgr_kwargs: dict[str, int] = {}
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)

        self.debug_callback: Callable[..., None] | None = debug_callback
        # per-sid turn cap: return 429 to kill the run once exceeded
        self.max_turns_per_sid: int | None = max_turns_per_sid
        self._sid_turn_count: dict[str, int] = {}
        # inbound auth headers (Authorization / x-api-key) captured per-turn in
        # _run_turn and forwarded to the messages upstream so it can authenticate.
        self._inbound_auth: dict[str, str] = {}
        # messages-mode only: the structured content blocks from the last upstream
        # response, so _run_turn can build a manager_message carrying tool_calls /
        # reasoning_content directly from Anthropic wire (parse_model_output can't
        # recover tool_use blocks from decoded text — it expects model-gen syntax).
        self._last_content_blocks: list[dict] = []
        # sglang-only: per-turn top-k alternative logprobs (from output_top_logprobs)
        self._last_top_logprobs: list | None = None

        self.app.router.add_get("/healthz", _health)
        self.app.router.add_get("/v1/models", _health)
        self._register_routes(self.app)

    # -- upstream dispatch ------------------------------------------------------

    async def _call_upstream(
        self,
        prompt_ids: list[int],
        session: Any,
        body: dict,
        session_id: str | None,
        *,
        translated: list[dict] | None = None,
        tools_schema: list[dict] | None = None,
    ) -> TurnRecord:
        """Forward one turn to the configured upstream backend and pack the reply.

        Dispatches on ``self.upstream_mode`` (read from ``UPSTREAM_MODE``):

        * ``sglang`` (default): the original path — render the prompt to
          token ids and POST them to ``{sglang_url}/generate`` with
          ``return_logprob: True``, then parse ``meta_info.output_token_logprobs``
          into token-level ids and logprobs (delegated to the module-level
          :func:`call_sglang_generate`).
        * ``messages``: forward the *raw* ``/v1/messages`` request body as-is to
          ``SLIME_MESSAGES_UPSTREAM_URL`` (any messages-API upstream: real
          Anthropic, lsai, mintcn, ...), decode the assistant text out of the
          stream/non-stream response, and re-tokenize it so the downstream
          tree/parser see real output tokens. No logprobs are available in this
          mode, so ``output_log_probs`` is ``[]`` (message-level SFT only).
        * ``chat``: route the chat-completions shape (the tree's hub format) to
          ``litellm.acompletion`` against ``SLIME_CHAT_BASE_URL`` (any
          OpenAI-compatible endpoint). litellm picks the wire format; the
          response is re-shaped to Anthropic content blocks and replayed through
          the messages-mode reply path. No logprobs (message-level SFT only).
        """
        if self.upstream_mode == "messages":
            turn, blocks = await self._call_messages_upstream(body, session_id)
            self._last_content_blocks = blocks
            return turn
        if self.upstream_mode == "chat":
            return await self._call_chat_upstream(translated, tools_schema, body, session_id)
        if self.upstream_mode == "responses":
            return await self._call_responses_upstream(translated, tools_schema, body, session_id)
        # sglang: call_sglang_generate returns (TurnRecord, top_k_logprobs|None)
        turn, top_k = await call_sglang_generate(
            prompt_ids, session, body, adapter=self, session_id=session_id
        )
        # stash per-turn top-k logprobs for _run_turn to attach to the record.
        self._last_top_logprobs = top_k
        return turn

    async def _call_messages_upstream(
        self,
        body: dict,
        session_id: str | None,
    ) -> TurnRecord:
        """Forward the raw /v1/messages body to an arbitrary messages-API upstream.

        No tokenizer round-trip is performed on the prompt (we forward ``body``
        verbatim), so ``prompt_ids`` is empty and only the assistant text is
        recovered from the upstream response. The text is re-tokenized (when a
        tokenizer is available) so :func:`parse_model_output` and the trajectory
        tree see real output tokens; ``output_log_probs`` is empty (no upstream
        logprobs). When no tokenizer is configured, ``output_ids`` is empty too
        and the tree is built purely from ``prompt_messages``/``response_message``
        (acceptable for message-level SFT).
        """
        logger = self.logger
        upstream = os.environ.get("SLIME_MESSAGES_UPSTREAM_URL", "").strip()
        if not upstream:
            raise RuntimeError(
                "UPSTREAM_MODE=messages requires SLIME_MESSAGES_UPSTREAM_URL to be set"
            )
        upstream = upstream.rstrip("/")

        # Forward the body verbatim. Streaming is honoured only to *consume* the
        # upstream response; downstream we always feed decoded text into
        # parse_model_output, so we collapse any stream to its final assistant text.
        want_stream = bool(body.get("stream"))
        fwd_body = dict(body)
        fwd_body["stream"] = want_stream
        timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
        finish_reason = "stop"
        text = ""
        usage_out = 0
        content_blocks: list[dict] = []
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(
                    f"{upstream}/v1/messages",
                    json=fwd_body,
                    headers={"Content-Type": "application/json", **self._inbound_auth},
                ) as r:
                    if r.status >= 400:
                        detail = await r.text()
                        logger.warning(
                            "[%s] sid=%s messages upstream %d: %.200s",
                            self.log_prefix,
                            session_id,
                            r.status,
                            detail,
                        )
                        raise RuntimeError(f"messages upstream {r.status}: {detail[:400]}")
                    if want_stream and r.headers.get("Content-Type", "").startswith("text/event-stream"):
                        text, finish_reason, usage_out, content_blocks = await _consume_messages_sse(r)
                    else:
                        data = await r.json(content_type=None)
                        text, finish_reason, usage_out, content_blocks = _parse_messages_json(data)
        except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(
                "[%s] sid=%s messages upstream aborted: %s",
                self.log_prefix,
                session_id,
                type(e).__name__,
            )
            raise

        # Map Anthropic stop_reason -> sglang finish_reason shape.
        finish = {
            "max_tokens": "length",
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(finish_reason, finish_reason or "stop")

        tok = self.tokenizer
        output_ids: list[int] = []
        if tok is not None and text:
            try:
                output_ids = list(tok.encode(text, add_special_tokens=False))
            except Exception:
                logger.debug("[%s] sid=%s tokenizer.encode failed; output_ids empty", self.log_prefix, session_id)
                output_ids = []

        return TurnRecord(
            prompt_ids=[],
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=[],
        ), content_blocks

    async def _call_chat_upstream(
        self,
        translated: list[dict],
        tools_schema: list[dict] | None,
        body: dict,
        session_id: str | None,
    ) -> TurnRecord:
        """Route one turn through litellm.acompletion (OpenAI-compatible upstream).

        chat mode hands the *chat-completions* shape (the tree's internal hub
        format — already produced by ``_translate_messages``) to
        ``litellm.acompletion``, letting litellm pick the provider/wire format.
        The response is re-shaped to Anthropic content blocks via
        :func:`chat_response_to_blocks`, stashed on ``self._last_content_blocks``
        so ``_run_turn``'s messages-mode reply path builds the manager_message
        (with tool_calls / reasoning_content) and renders the Anthropic response
        back to Claude Code. No logprobs (message-level SFT only).

        Env: ``SLIME_CHAT_BASE_URL``, ``SLIME_CHAT_API_KEY``, ``SLIME_CHAT_MODEL``
        (e.g. an OpenAI-compatible gateway). Auth from the inbound request
        (Authorization/x-api-key) is forwarded via ``api_key`` when no explicit
        key is set, so a client's own credential reaches the upstream.
        """
        import litellm

        from slime_sft_trace.adapters.anthropic import chat_response_to_blocks

        base_url = os.environ.get("SLIME_CHAT_BASE_URL") or ""
        # litellm appends /chat/completions to api_base, so it must end with /v1.
        # If the caller gave a bare origin, append /v1; if already /v1, keep it.
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        model = os.environ.get("SLIME_CHAT_MODEL") or "gpt-4o"
        api_key = os.environ.get("SLIME_CHAT_API_KEY") or self._inbound_auth.get("authorization") or ""
        # strip "Bearer " prefix if we forwarded the raw Authorization header
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:]
        if not base_url:
            raise RuntimeError("UPSTREAM_MODE=chat requires SLIME_CHAT_BASE_URL")

        # Some OpenAI-compatible upstreams (e.g. mintcn) reject `tool_calls`
        # whose `function.arguments` is a *dict* in a replayed assistant message
        # (502), while accepting a JSON *string* (the OpenAI live-API shape). The
        # tree stores dict args (HF shape), so normalize to a JSON string for the
        # wire unless the caller opts out via SLIME_CHAT_ARGS_AS_DICT=1. litellm
        # does NOT re-serialize pre-existing tool_calls on the request, so we must.
        args_as_dict = os.environ.get("SLIME_CHAT_ARGS_AS_DICT", "") == "1"
        wire_messages = translated if args_as_dict else _stringify_tool_call_args(translated)

        # litellm routes to an OpenAI-compatible endpoint via "openai/<model>".
        # tools_schema is already OpenAI function-tool shape ({type,function:{...}}).
        kwargs: dict[str, Any] = {
            "model": f"openai/{model}",
            "messages": wire_messages,
            "stream": False,
            "api_base": base_url,
            "api_key": api_key,
        }
        if tools_schema:
            kwargs["tools"] = tools_schema
        if body.get("max_tokens"):
            kwargs["max_tokens"] = int(body["max_tokens"])
        # forward anthropic-version etc. as extra headers where supported
        extra = {k: v for k, v in self._inbound_auth.items() if k.lower() not in ("authorization",)}
        if extra:
            kwargs["extra_headers"] = extra

        # Request logprobs from the chat upstream (vLLM + SGLang /v1/chat/completions
        # both support logprobs=True + top_logprobs). Disabled by default; enable via
        # SLIME_CHAT_LOGPROBS=1 (sets logprobs=True) + SLIME_CHAT_TOP_LOGPROBS=N (top-k).
        if os.environ.get("SLIME_CHAT_LOGPROBS") == "1":
            kwargs["logprobs"] = True
            top_k = int(os.environ.get("SLIME_CHAT_TOP_LOGPROBS", "0") or "0")
            if top_k > 0:
                kwargs["top_logprobs"] = top_k

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            self.logger.warning("[%s] sid=%s chat upstream failed: %s", self.log_prefix, session_id, exc)
            raise

        blocks, stop_reason = chat_response_to_blocks(response)
        self._last_content_blocks = blocks

        # Extract per-token logprobs from the chat response (OpenAI shape:
        # choices[0].logprobs.content = [{token, logprob, top_logprobs:[...]}]).
        # Pack into output_log_probs for the TurnRecord. top_logprobs_num on the
        # TurnRecord is left to the sglang path; chat stores the sampled-token
        # logprob only (sufficient for GRPO; top-k is in the response object if needed).
        output_log_probs: list[float] = []
        if os.environ.get("SLIME_CHAT_LOGPROBS") == "1":
            try:
                choice = response.choices[0]
                lp = getattr(choice, "logprobs", None)
                if lp and getattr(lp, "content", None):
                    for entry in lp.content:
                        lp_val = getattr(entry, "logprob", None)
                        if lp_val is not None:
                            output_log_probs.append(float(lp_val))
            except (AttributeError, IndexError, TypeError):
                pass  # upstream didn't return logprobs despite the request

        # map Anthropic stop_reason -> sglang finish_reason shape (tool_use->tool_calls etc.)
        finish = {
            "max_tokens": "length",
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason or "stop")
        return TurnRecord(prompt_ids=[], output_ids=[], finish_reason=finish, output_log_probs=output_log_probs)

    async def _call_responses_upstream(
        self,
        translated: list[dict],
        tools_schema: list[dict] | None,
        body: dict,
        session_id: str | None,
    ) -> TurnRecord:
        """Route one turn through litellm.aresponses (OpenAI Responses API).

        responses mode targets vLLM's /v1/responses endpoint (the only one that
        implements logprobs on the Responses API — SGLang's is still TODO). The
        chat-completions hub shape is adapted to the Responses API input format,
        the response is re-shaped to Anthropic content blocks, and logprobs are
        extracted from output_text.logprobs.

        Env: ``SLIME_RESPONSES_BASE_URL``, ``SLIME_RESPONSES_API_KEY``,
        ``SLIME_RESPONSES_MODEL``. ``SLIME_RESPONSES_LOGPROBS=1`` enables
        ``include=["message.output_text.logprobs"]``;
        ``SLIME_RESPONSES_TOP_LOGPROBS=N`` sets top-k.
        """
        import litellm

        from slime_sft_trace.adapters.anthropic import responses_output_to_blocks

        base_url = os.environ.get("SLIME_RESPONSES_BASE_URL") or ""
        model = os.environ.get("SLIME_RESPONSES_MODEL") or "gpt-4o"
        api_key = os.environ.get("SLIME_RESPONSES_API_KEY") or self._inbound_auth.get("authorization") or ""
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:]
        if not base_url:
            raise RuntimeError("UPSTREAM_MODE=responses requires SLIME_RESPONSES_BASE_URL")

        # Responses API uses `input` (messages or string) instead of `messages`.
        include = []
        want_logprobs = os.environ.get("SLIME_RESPONSES_LOGPROBS") == "1"
        if want_logprobs:
            include.append("message.output_text.logprobs")
        top_k = int(os.environ.get("SLIME_RESPONSES_TOP_LOGPROBS", "0") or "0")

        kwargs: dict[str, Any] = {
            "model": f"openai/{model}",
            "input": translated,  # Responses API accepts chat messages as input
            "api_base": base_url,
            "api_key": api_key,
            "stream": False,
        }
        if tools_schema:
            kwargs["tools"] = tools_schema
        if body.get("max_tokens"):
            kwargs["max_output_tokens"] = int(body["max_tokens"])
        if include:
            kwargs["include"] = include
        if top_k > 0:
            kwargs["top_logprobs"] = top_k
        extra = {k: v for k, v in self._inbound_auth.items() if k.lower() not in ("authorization",)}
        if extra:
            kwargs["extra_headers"] = extra

        try:
            response = await litellm.aresponses(**kwargs)
        except Exception as exc:
            self.logger.warning("[%s] sid=%s responses upstream failed: %s", self.log_prefix, session_id, exc)
            raise

        blocks, stop_reason = responses_output_to_blocks(response)
        self._last_content_blocks = blocks

        # extract logprobs from output_text.logprobs (if requested)
        output_log_probs: list[float] = []
        if want_logprobs:
            try:
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        lp = getattr(content, "logprobs", None)
                        if lp:
                            for entry in lp:
                                val = getattr(entry, "logprob", None)
                                if val is not None:
                                    output_log_probs.append(float(val))
            except (AttributeError, TypeError):
                pass

        finish = {
            "max_tokens": "length",
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason or "stop")
        return TurnRecord(prompt_ids=[], output_ids=[], finish_reason=finish, output_log_probs=output_log_probs)

    def _reply_from_content_blocks(
        self, blocks: list[dict], finish: str, tools_schema: list[dict] | None
    ) -> "Reply":
        """Build a Reply directly from Anthropic-structured content blocks.

        Messages-mode only: the upstream (an Anthropic-compatible API) returns
        canonical content blocks (text/thinking/tool_use), not model-generation
        text. We convert them to a manager_message in OpenAI chat shape — the same
        shape the tree stores and the SFT dump emits — so tool_calls and
        reasoning_content are preserved. The wire reply (Anthropic blocks + SSE
        rendering) is left to the adapter's _respond, which re-renders from the
        original body; here we only populate what the trajectory records.
        """
        from .anthropic import _build_reply_parts_from_blocks  # local to avoid cycle

        manager_message, stop_reason = _build_reply_parts_from_blocks(blocks, finish)
        # manager_finish_reason: tool_calls if any tool_use present, else the raw finish
        has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks)
        fr = "tool_calls" if has_tool_use else (finish or "stop")
        return Reply(manager_message=manager_message, finish_reason=fr, wire=(blocks, stop_reason))


    # -- wire hooks (subclass overrides) -------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register the protocol's POST route(s) and bind self._run_turn."""
        raise NotImplementedError

    def _session_id(self, request: web.Request, body: dict) -> str:
        raise NotImplementedError

    def _preprocess_body(self, body: dict) -> None:
        """Mutate the parsed body in place before sid resolution (default no-op)."""

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Return (chat_messages, tools_schema) from the wire body."""
        raise NotImplementedError

    def _build_reply(self, parsed, raw_finish: str, translated: list[dict], tools_schema: list[dict] | None) -> Reply:
        """Pack parsed model output into a Reply."""
        raise NotImplementedError

    async def _respond(
        self,
        request: web.Request,
        body: dict,
        reply: Reply,
        in_tok: int,
        out_tok: int,
        stream: bool,
    ) -> web.StreamResponse:
        raise NotImplementedError

    # -- session lifecycle ---------------------------------------------------

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        """Register a fresh per-sid Session; sids must be unique."""
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        self.store[sid] = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
        )

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        """Mark a sid closed and drain its in-flight turn tasks."""
        self.closed.add(sid)
        tasks = [t for t in self.inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return

        async def _drain() -> None:
            _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        loop = tasks[0].get_loop()
        try:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_drain(), loop))
        except Exception:
            self.logger.exception("[%s] sid=%s shutdown drain failed", self.log_prefix, sid)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list:
        """Drain a session's trajectory into fully-formed Sample objects.

        Waits out in-flight requests for the sid, linearises the per-sid tree,
        then decodes each sample's trained tail into .response (the manager is
        tokenizer-free, so the adapter that owns the tokenizer fills this in).
        Idempotent: a second call for an already-popped sid returns [].
        """
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        session = self.store.pop(sid, None)
        max_sample_tokens = int(getattr(session, "max_context_tokens", 0) or 0) if session is not None else 0

        # messages mode carries no tokens (output_ids == []): the token-level
        # get_trajectory would return [] -- and it pops the per-sid tree while
        # doing so, leaving nothing for a message-level fallback to read. So in
        # messages mode we branch to the message-level path BEFORE touching
        # get_trajectory; the token path below is unchanged for sglang mode.
        if self.upstream_mode in ("messages", "chat", "responses"):
            return self._finish_messages_session(
                sid, base_sample=base_sample, reward=reward, extra_metadata=extra_metadata
            )

        samples = self.manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=extra_metadata,
            max_sample_tokens=max_sample_tokens,
        )
        for s in samples:
            rlen = int(s.response_length or 0)
            s.response = (
                self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen and s.tokens else ""
            )
            # attach per-turn top-k logprobs from the last sglang turn (if collected)
            if self._last_top_logprobs is not None:
                s.output_top_logprobs = self._last_top_logprobs
        self._last_top_logprobs = None
        return samples

    def _finish_messages_session(
        self,
        sid: str,
        *,
        base_sample,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
    ) -> list:
        """Message-level SFT fallback: linearize the per-sid tree into message
        ``Sample`` objects (one per routing leaf). Lazily imported to avoid a
        circular import (``message_dump`` imports ``types`` only, but keeping
        the import local documents the fallback boundary)."""
        from slime_sft_trace.message_dump import get_trajectory_messages

        # Surface the tools_schema the model saw this session onto the dumped
        # Sample. tools_schema is keyed by sid on the adapter; node.metadata
        # carries it too (per-turn), so the dump can fall back to the chain.
        tools_schema = self._sid_tools.get(sid)
        md = dict(extra_metadata or {})
        if tools_schema is not None and "tools" not in md:
            md["tools"] = tools_schema
        samples = get_trajectory_messages(
            self.manager,
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=md,
        )
        # consumed: drop the sid-keyed tools cache too
        self._sid_tools.pop(sid, None)
        return samples

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        self.store.pop(sid, None)
        self._sid_tools.pop(sid, None)
        self.manager.drop_session(sid)

    # -- shared request pipeline ---------------------------------------------

    def _check_turn_cap(self, sid: str) -> web.Response | None:
        """Enforce max_turns_per_sid, returning a 429 response once exceeded.

        Increments the per-sid counter as a side effect when under the cap.
        """
        cap = self.max_turns_per_sid
        if cap is None:
            return None
        prior = self._sid_turn_count.get(sid, 0)
        if prior >= cap:
            self.logger.warning("[%s] sid=%s exceeded max_turns_per_sid=%d; killing run", self.log_prefix, sid, cap)
            return web.json_response(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "message": (f"adapter: sid {sid!r} exceeded max_turns_per_sid={cap}; killing run"),
                    }
                },
                status=429,
            )
        self._sid_turn_count[sid] = prior + 1
        return None

    def _run_debug_callback(self, sid, translated, tools_schema, manager_message, turn) -> None:
        """Run the optional debug-only data dump callback; unset in production."""
        callback = self.debug_callback
        if callback is None:
            return
        try:
            callback(sid, translated, tools_schema, manager_message, turn)
        except Exception:
            self.logger.exception("debug_callback failed (sid=%s)", sid)

    async def _run_turn(self, request: web.Request, *, downstream_format: str = "messages") -> web.StreamResponse:
        """One full agent turn: translate -> sglang -> parse -> append -> respond.

        The wire-specific steps are delegated to the subclass hooks; the rest
        (sid resolution, closed/cap guards, inflight tracking, record_turn) is
        shared across protocols.

        ``downstream_format`` selects the request/response wire shape: "messages"
        (Anthropic /v1/messages, the default) or "chat" (OpenAI /v1/chat/completions).
        """
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:  # session drained; refuse stragglers
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        tok = self.tokenizer
        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            if downstream_format == "chat":
                translated, tools_schema = self._translate_chat(body)
            elif downstream_format == "responses":
                translated, tools_schema = self._translate_responses(body)
            else:
                translated, tools_schema = self._translate(body)
            prompt_ids = (
                _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)
                if tok is not None else []
            )

            # Pass the inbound auth headers (Authorization / x-api-key) through to
            # the messages upstream so the upstream can authenticate the request.
            # In messages mode we forward the body verbatim, so the client's own
            # credentials are the right ones to present upstream.
            self._inbound_auth = {
                k: v
                for k, v in request.headers.items()
                if k.lower() in ("authorization", "x-api-key", "x-goog-api-key", "anthropic-version")
            }
            turn = await self._call_upstream(prompt_ids, s, body, sid, translated=translated, tools_schema=tools_schema)

            raw_output = (
                tok.decode(turn.output_ids, skip_special_tokens=False)
                if tok is not None and turn.output_ids else ""
            )
            parsed = parse_model_output(
                raw_output,
                tools_schema=tools_schema,
                tool_parser_name=self.tool_parser,
                reasoning_parser_name=self.reasoning_parser,
            )
            # messages-mode / chat-mode: the upstream returned Anthropic-structured
            # content blocks (tool_use/thinking/text), which parse_model_output
            # CANNOT recover from decoded text. Build the manager_message directly
            # from those blocks so tool_calls + reasoning_content survive into the
            # trajectory (and thus the SFT dump). sglang mode has no blocks and
            # falls through to the parsed reply (unchanged).
            if self.upstream_mode in ("messages", "chat", "responses") and self._last_content_blocks:
                reply = self._reply_from_content_blocks(self._last_content_blocks, turn.finish_reason, tools_schema)
                self._last_content_blocks = []
            else:
                reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)

            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")

            # Flush the response before recording the trajectory: a client that
            # disconnected during generation makes _respond raise here, and we
            # must not record a turn the client never received.
            try:
                if downstream_format == "chat":
                    response = await self._respond_chat(request, body, reply, in_tok, out_tok, stream)
                elif downstream_format == "responses":
                    response = await self._respond_responses(request, body, reply, in_tok, out_tok, stream)
                else:
                    response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                self.logger.warning(
                    "[%s] sid=%s client disconnected before response flush: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")

            self._run_debug_callback(
                sid,
                translated,
                tools_schema,
                reply.manager_message,
                turn,
            )

            self._capture_tools_schema(sid, tools_schema)
            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": sid, "tools": tools_schema},
            )
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)

    def _capture_tools_schema(self, sid: str, tools_schema: list[dict] | None) -> None:
        """Remember the tools_schema the model saw this turn (first-wins).

        Tools don't change within a session, so the first turn's schema is the
        canonical one. Kept on the adapter so the message-level dump path can
        surface it onto the emitted Sample (the tree node also carries it in
        record_turn metadata as a second, adapter-independent source).
        """
        if sid not in self._sid_tools:
            self._sid_tools[sid] = tools_schema


def sid_from_bearer(request: web.Request) -> str | None:
    """sid from the Authorization: Bearer <sid> header, or None if absent."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def sid_from_body(body: dict | None) -> str | None:
    """sid from the OpenAI-shape body (metadata.session_id / user), or None."""
    if not body:
        return None
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return str(metadata["session_id"])
    if body.get("user"):
        return str(body["user"])
    return None


def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(session.sampling_defaults or {}),
    }

    for key in max_token_keys:
        if body.get(key) is not None:
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp



async def call_sglang_generate(
    prompt_ids: list[int],
    session: Any,
    body: dict,
    *,
    adapter: BaseAdapter,
    session_id: str | None = None,
) -> TurnRecord:
    """POST one turn to sglang /generate and pack the reply into a TurnRecord.

    Module-level (not a method) so tests can monkeypatch it.
    """
    logger = adapter.logger
    sp = _sampling_params(session, body, max_token_keys=adapter.max_token_keys, stop_keys=adapter.stop_keys)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[%s] sid=%s prompt exceeds max_context_tokens (%d >= %d)",
                adapter.log_prefix,
                session_id,
                len(prompt_ids),
                session.max_context_tokens,
            )
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length")
        sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", remaining_context)), remaining_context)

    sglang_url = adapter.sglang_url
    rid = uuid.uuid4().hex
    headers = {"X-SMG-Routing-Key": session_id} if session_id and session_id != "default" else None
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    # top-k logprobs: env SLIME_TOP_LOGPROBS=N requests the top-N alternative
    # tokens per position from sglang (output_top_logprobs in meta_info).
    top_logprobs_num = int(os.environ.get("SLIME_TOP_LOGPROBS", "0") or "0")
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess, sess.post(
            f"{sglang_url}/generate",
            json={
                "rid": rid,
                "input_ids": prompt_ids,
                "sampling_params": sp,
                "return_logprob": True,
                **({"top_logprobs_num": top_logprobs_num} if top_logprobs_num > 0 else {}),
            },
            headers=headers,
        ) as r:
            if r.status >= 400:
                text = await r.text()
                logger.warning(
                    "[%s] sid=%s rid=%s sglang upstream %d: %.200s",
                    adapter.log_prefix,
                    session_id,
                    rid,
                    r.status,
                    text,
                )
                raise RuntimeError(f"sglang upstream {r.status}: {text[:400]}")
            data = await r.json(content_type=None)
        meta = data.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        output_ids = [x[1] for x in output_token_logprobs]
        output_log_probs = [float(x[0]) for x in output_token_logprobs]
        # top-k alternative logprobs per position: list of [(logprob, token_id), ...]
        # (None when SLIME_TOP_LOGPROBS not set / upstream didn't return them).
        output_top_logprobs = meta.get("output_top_logprobs") or None
        finish = (meta.get("finish_reason") or {}).get("type", "stop") or "stop"
    except (asyncio.CancelledError, aiohttp.ClientError, asyncio.TimeoutError) as e:
        # free the sglang slot eagerly on client cancel/timeout, else the
        # orphaned generation keeps occupying KV until its own length cap
        logger.debug("[%s] sid=%s rid=%s turn aborted: %s", adapter.log_prefix, session_id, rid, type(e).__name__)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s2:
                await s2.post(f"{sglang_url}/abort_request", json={"rid": rid})
        except Exception:
            pass
        raise

    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
    ), output_top_logprobs


def _parse_messages_json(data: dict) -> tuple[str, str, int, list[dict]]:
    """Pull (text, stop_reason, output_tokens, content_blocks) from a non-stream JSON.

    ``content_blocks`` is the raw Anthropic content list (text/thinking/tool_use)
    so the messages-mode path can build a structured manager_message directly
    from the upstream's canonical blocks — without it, tool_calls would be lost
    (parse_model_output expects model-generation syntax, not Anthropic wire).
    """
    content = data.get("content") if isinstance(data, dict) else None
    blocks: list[dict] = []
    if isinstance(content, list):
        blocks = [b for b in content if isinstance(b, dict)]
    elif isinstance(content, str):
        blocks = [{"type": "text", "text": content}]
    texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
    text = "".join(texts)
    stop_reason = ""
    if isinstance(data, dict):
        stop_reason = data.get("stop_reason") or ""
        usage = data.get("usage") or {}
        out_tok = usage.get("output_tokens") if isinstance(usage, dict) else None
    else:
        out_tok = None
    out_tok = int(out_tok) if isinstance(out_tok, (int, float)) and out_tok is not None else 0
    return text, stop_reason or "", out_tok, blocks


async def _consume_messages_sse(response: aiohttp.ClientResponse) -> tuple[str, str, int, list[dict]]:
    """Consume an Anthropic Messages SSE stream.

    Returns (text, stop_reason, out_tokens, content_blocks). ``content_blocks``
    reassembles every content block from its deltas so the messages-mode path
    can recover tool_use/thinking blocks the model emitted.
    """
    # Accumulate per-block-index content. Each block has a type (text/thinking/
    # tool_use) set at content_block_start; text deltas append to text blocks,
    # input_json_delta appends to tool_use 'input' (as a JSON fragment string).
    blocks: dict[int, dict] = {}
    block_order: list[int] = []
    text_parts: list[str] = []  # legacy: concatenated text for the text return
    stop_reason = ""
    out_tokens = 0
    async for raw in response.content:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            evt = json.loads(payload)
        except Exception:
            continue
        etype = evt.get("type")
        if etype == "content_block_start":
            idx = evt.get("index", 0)
            block = evt.get("content_block") or {}
            blocks[idx] = dict(block)  # {type, text?, name?, id?, input?}
            if "input" not in blocks[idx] and block.get("type") == "tool_use":
                blocks[idx]["input"] = {}  # filled by input_json_delta
            block_order.append(idx)
        elif etype == "content_block_delta":
            delta = evt.get("delta") or {}
            idx = evt.get("index", 0)
            dt = delta.get("type")
            if dt == "text_delta":
                txt = delta.get("text", "")
                text_parts.append(txt)
                if idx in blocks:
                    blocks[idx]["text"] = blocks[idx].get("text", "") + txt
            elif dt == "input_json_delta":
                frag = delta.get("partial_json", "")
                if idx in blocks:
                    blocks[idx]["_input_json"] = blocks[idx].get("_input_json", "") + frag
            elif dt == "thinking_delta":
                if idx in blocks:
                    blocks[idx]["thinking"] = blocks[idx].get("thinking", "") + delta.get("thinking", "")
        elif etype == "content_block_stop":
            idx = evt.get("index", 0)
            if idx in blocks and "_input_json" in blocks[idx]:
                raw_input = blocks[idx].pop("_input_json")
                try:
                    blocks[idx]["input"] = json.loads(raw_input) if raw_input else {}
                except (ValueError, TypeError):
                    blocks[idx]["input"] = {"_raw": raw_input}
        elif etype == "message_delta":
            delta = evt.get("delta") or {}
            if delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
            usage = evt.get("usage") or {}
            if isinstance(usage, dict) and usage.get("output_tokens") is not None:
                out_tokens = int(usage["output_tokens"])
        elif etype == "message_start":
            msg = evt.get("message") or {}
            usage = msg.get("usage") or {}
            if isinstance(usage, dict) and usage.get("output_tokens") is not None:
                out_tokens = int(usage["output_tokens"])
    ordered_blocks = [blocks[i] for i in block_order if i in blocks]
    return "".join(text_parts), stop_reason, out_tokens, ordered_blocks


async def _health(request: web.Request) -> web.Response:
    """Handler for /healthz and /v1/models readiness probes."""
    return web.json_response({"ok": True})
