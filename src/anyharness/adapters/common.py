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
import itertools
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import aiohttp
from aiohttp import web

from anyharness.parsing import parse_model_output
from anyharness.trajectory import MASK_LOGPROB, TopkLogprobs, TrajectoryManager, TurnRecord


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


@dataclasses.dataclass(frozen=True)
class SamplingResult:
    """One upstream turn. Exactly one of (content_blocks | token-level ids) is
    meaningful; the consumer branches on ``content_blocks``, not on a mode string.

    Block-level backends (messages/chat/responses) set ``content_blocks`` and leave
    ``turn.output_ids`` empty; token-level backends (sglang/tinker) leave
    ``content_blocks`` None and populate ``turn.output_ids``/``output_log_probs``.
    ``top_logprobs`` carries the per-turn top-k (sglang only) that used to live on
    the ``_last_top_logprobs`` side-effect channel.
    """

    turn: TurnRecord
    content_blocks: list[dict] | None = None
    top_logprobs: list | None = None
    # Upstream-reported token counts (None = token-level backend, or upstream
    # omitted usage; caller falls back to len(prompt_ids)/len(turn.output_ids)).
    # Keys: input_tokens, output_tokens.
    usage: dict | None = None


@runtime_checkable
class SamplingUpstream(Protocol):
    """A sampling backend behind the messages->token-ids abstraction.

    Each implementation encapsulates its own rendering: server-backed backends
    (messages/chat/responses) delegate render to the upstream; token-native
    backends (sglang/tinker) render locally. ``message_level`` selects the
    trajectory-dump path (block-level vs token-level) without a mode-string check.
    """

    message_level: bool

    async def sample(
        self,
        messages: list[dict],
        tools_schema: list[dict] | None,
        body: dict,
        session: "Session",
        session_id: str | None,
    ) -> SamplingResult: ...


def _sglang_topk_to_typed(
    raw: list | None, k: int
) -> TopkLogprobs | None:
    """Convert sglang ``output_top_logprobs`` to a sentinel-padded TopkLogprobs.

    sglang returns ``list[list[(logprob, token_id)]]`` — per position, up to k
    alternatives (fewer when the position had fewer candidates). We split into
    paired dense arrays and pad short positions with ``(0, MASK_LOGPROB)`` so the
    pair is rectangular ``(num_tokens, k)`` — the Tinker SDK TopkPromptLogprobs
    convention. None when the upstream returned no top-k.
    """
    if not raw:
        return None
    token_ids: list[list[int]] = []
    logprobs: list[list[float]] = []
    for position in raw:
        if not position:
            token_ids.append([0] * k)
            logprobs.append([MASK_LOGPROB] * k)
            continue
        ids = [int(tid) for _, tid in position]
        lps = [float(lp) for lp, _ in position]
        if len(ids) < k:
            pad = k - len(ids)
            ids += [0] * pad
            lps += [MASK_LOGPROB] * pad
        token_ids.append(ids)
        logprobs.append(lps)
    return TopkLogprobs(token_ids=token_ids, logprobs=logprobs)


class _DelegatingUpstream:
    """Base for the Phase-1 implementations: holds the adapter and delegates to
    its existing ``_call_*_upstream`` methods / module functions. The delegation
    keeps behaviour byte-identical (87-test contract) while the dispatch and
    consumption sites move to the unified interface. Subclasses set ``message_level``
    and implement :meth:`sample`.
    """

    message_level = False

    def __init__(self, adapter: "BaseAdapter") -> None:
        self._a = adapter


class MessagesUpstream(_DelegatingUpstream):
    """``UPSTREAM_MODE=messages``: forward a /v1/messages body verbatim (or rebuilt
    from hub messages when the downstream wire is chat/responses). Block-level."""

    message_level = True

    async def sample(self, messages, tools_schema, body, session, session_id):
        self._a._last_usage = None  # reset per turn
        turn, blocks = await self._a._call_messages_upstream(
            body, session_id, translated=messages, tools_schema=tools_schema
        )
        return SamplingResult(turn=turn, content_blocks=blocks, usage=self._a._last_usage)


class ChatUpstream(_DelegatingUpstream):
    """``UPSTREAM_MODE=chat``: litellm.acompletion against an OpenAI-compatible
    endpoint. Block-level; logprobs (with vLLM return_token_ids) ship on the turn."""

    message_level = True

    async def sample(self, messages, tools_schema, body, session, session_id):
        self._a._last_usage = None
        # The Responses API uses FLAT tool shape ({type:function, name, parameters});
        # the chat upstream (litellm/vLLM) needs NESTED ({type:function,
        # function:{name,parameters}}). Normalize once here so both paths match.
        tools_schema = _nest_tools(tools_schema)
        # litellm 1.88 rejects the standard tool_calls[].type="function" shape on
        # vLLM 0.26 (expects a non-standard CustomToolCallParam). Route any request
        # that replays a tool_call through plain HTTP, bypassing the validator.
        has_tool_replay = any(
            isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
            for m in messages
        )
        if has_tool_replay:
            raw = await self._a._call_chat_upstream_raw(messages, tools_schema, body, session_id)
            from anyharness.trajectory import SampledSequence, TurnRecord
            turn = TurnRecord(
                prompt_ids=[],
                output_ids=raw["output_ids"],
                finish_reason=raw["stop_reason"],
                output_log_probs=raw["output_log_probs"],
            )
            return SamplingResult(turn=turn, content_blocks=raw["blocks"], usage=raw["usage"])
        turn = await self._a._call_chat_upstream(messages, tools_schema, body, session_id)
        return SamplingResult(turn=turn, content_blocks=self._a._last_content_blocks,
                               usage=self._a._last_usage)


class ResponsesUpstream(_DelegatingUpstream):
    """``UPSTREAM_MODE=responses``: litellm.aresponses against the Responses API.
    Block-level; no TITO token ids on this API."""

    message_level = True

    async def sample(self, messages, tools_schema, body, session, session_id):
        self._a._last_usage = None
        turn = await self._a._call_responses_upstream(messages, tools_schema, body, session_id)
        return SamplingResult(turn=turn, content_blocks=self._a._last_content_blocks,
                               usage=self._a._last_usage)


class SglangUpstream(_DelegatingUpstream):
    """``UPSTREAM_MODE=sglang``: native ``/generate`` with return_logprob.
    Token-level (real output_ids + logprobs); carries per-turn top-k."""

    message_level = False

    async def sample(self, messages, tools_schema, body, session, session_id):
        # sglang >= 0.5.12 exposes /v1/tokenize with messages, so the server can
        # render the chat template itself and we skip the local tokenizer entirely
        # (TITO-safe: the ids come from the server, not from re-encoding text).
        # Fall back to local render when the server can't (old sglang / no URL).
        prompt_ids = await self._a._tokenize_messages_server(
            self._a.sglang_url, messages, tools_schema, session_id
        )
        if not prompt_ids:
            prompt_ids = self._a._render_prompt_for(messages, tools_schema)
        turn, top_k = await call_sglang_generate(
            prompt_ids, session, body, adapter=self._a, session_id=session_id
        )
        k = int(os.environ.get("SLIME_TOP_LOGPROBS", "0") or "0") or None
        return SamplingResult(turn=turn, top_logprobs=_sglang_topk_to_typed(top_k, k or 0))


class TinkerUpstream(_DelegatingUpstream):
    """``UPSTREAM_MODE=tinker``: native ``/api/v1/asample`` + retrieve_future.
    Token-in token-out (real output_ids + logprobs)."""

    message_level = False

    async def sample(self, messages, tools_schema, body, session, session_id):
        prompt_ids = self._a._render_prompt_for(messages, tools_schema)
        turn, _ = await call_tinker_sample(
            prompt_ids, session, body, adapter=self._a, session_id=session_id
        )
        return SamplingResult(turn=turn)


def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Render a chat-message list to token ids with the served chat template."""
    # Chat templates call .items() on tool_call arguments, so a JSON *string*
    # (the OpenAI wire shape some harnesses replay) breaks rendering. Normalize
    # string arguments to dicts for the template; the tree is untouched.
    renderable = messages
    if any(isinstance(m, dict) and m.get("tool_calls") for m in messages):
        renderable = []
        for m in messages:
            if not (isinstance(m, dict) and isinstance(m.get("tool_calls"), list)):
                renderable.append(m)
                continue
            m2 = dict(m)
            m2["tool_calls"] = []
            for tc in m["tool_calls"]:
                tc2 = tc if isinstance(tc, dict) else tc
                if isinstance(tc2, dict):
                    tc2 = dict(tc2)
                    fn = tc2.get("function")
                    if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                        fn = dict(fn)
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except json.JSONDecodeError:
                            fn["arguments"] = {"_raw": fn["arguments"]}
                        tc2["function"] = fn
                m2["tool_calls"].append(tc2)
            renderable.append(m2)
    enc = tokenizer.apply_chat_template(
        renderable,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


class _AttrView:
    """Read-only attribute view over a parsed JSON dict.

    ``responses_output_to_blocks`` reads the upstream payload with ``getattr``
    (it normally receives a pydantic model). When we parse a Responses body
    ourselves we hand it this instead, so nested items expose ``.type`` /
    ``.content`` while leaf parts stay plain dicts -- which that function already
    handles via its ``isinstance(part, dict)`` branches.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict) -> None:
        self._d = d

    def __getattr__(self, name: str) -> Any:
        try:
            val = self._d[name]
        except KeyError:
            raise AttributeError(name) from None
        if isinstance(val, dict):
            return _AttrView(val)
        if isinstance(val, list):
            return [_AttrView(v) if isinstance(v, dict) else v for v in val]
        return val

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_AttrView({self._d!r})"


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


def _nest_tools(tools: list[dict] | None) -> list[dict] | None:
    """Wrap flat Responses-API tools ({type,name,parameters}) into the nested
    chat-completions shape ({type,function:{name,parameters}}) the chat upstream
    (litellm/vLLM) requires. Already-nested tools pass through unchanged."""
    if not tools:
        return tools
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "function" in t:
            out.append(t)
            continue
        if t.get("type") == "function" and t.get("name"):
            out.append({"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or t.get("input_schema") or {"type": "object", "properties": {}},
            }})
        else:
            out.append(t)
    return out or None


def _ensure_tool_call_ids(messages: list[dict]) -> list[dict]:
    """Re-synthesize missing tool_call ids and pair them onto following tool msgs.

    The tree stores tool_calls via ``tool_call_dict`` which deliberately drops the
    wire id (so a generated turn compares dict-equal to its replayed echo). But
    vLLM 0.26 requires every tool_call to carry an ``id`` and the tool message a
    matching ``tool_call_id``. We emit one positional id per assistant tool_call
    (``call_0``, ``call_1``, ...) and assign it to the next ``len(tool_calls)``
    tool messages in order — mirroring hub_to_anthropic. Tree is untouched.
    """
    out: list[dict] = []
    counter = 0
    pending: list[str] = []  # ids emitted by the last assistant turn, awaiting tool msgs
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            m2 = dict(m)
            tcs = []
            pending = []
            for tc in m["tool_calls"]:
                if not isinstance(tc, dict):
                    tcs.append(tc)
                    continue
                tc2 = dict(tc)
                tid = tc2.get("id") or f"call_{counter}"
                if not tc2.get("id"):
                    counter += 1
                tc2["id"] = tid
                tc2.setdefault("type", "function")
                pending.append(tid)
                tcs.append(tc2)
            m2["tool_calls"] = tcs
            out.append(m2)
        elif role == "tool":
            m2 = dict(m)
            if not m2.get("tool_call_id") and pending:
                m2["tool_call_id"] = pending.pop(0)
            out.append(m2)
        else:
            out.append(m)
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


def _extract_upstream_token_ids(choice: Any) -> list[int] | None:
    """Pull the upstream's own generated token ids off a chat-completions choice.

    vLLM >= 0.10.2 returns them on the OpenAI-compatible endpoint when the request
    sets ``return_token_ids: true`` (field ``token_ids`` on the choice; the prompt
    side arrives as ``prompt_token_ids``). These are the ids the server actually
    sampled, which is the only TITO-safe source: token ids must never be
    reconstructed by re-encoding decoded text or by looking token *strings* back
    up in a vocab, because tokenization is not injective and the recovered
    sequence may differ from what the policy produced.

    Returns ``None`` when the upstream did not supply ids (not vLLM, too old, or
    the flag was rejected). Callers must drop logprobs in that case rather than
    fall back to reconstruction.
    """
    for holder in (choice, getattr(choice, "model_extra", None) or {}):
        ids = holder.get("token_ids") if isinstance(holder, dict) else getattr(holder, "token_ids", None)
        if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
            return list(ids)
    return None


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
        if mode not in ("sglang", "messages", "chat", "responses", "tinker"):
            raise ValueError(
                f"UPSTREAM_MODE={mode!r} must be 'sglang', 'messages', 'chat', "
                "'responses', or 'tinker'"
            )
        self.upstream_mode = mode
        # The unified sampling backend (Protocol). Built once at construction: this is
        # the single place a mode string is read to pick an implementation — every
        # later dispatch is through self.upstream.sample(...) / .message_level.
        self.upstream = self._build_upstream(mode)
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
        # message-level upstreams: the upstream's own usage, surfaced onto
        # SamplingResult.usage so _run_turn reports truthful token counts.
        self._last_usage: dict | None = None

        self.app.router.add_get("/healthz", _health)
        self.app.router.add_get("/v1/models", _health)
        self._register_routes(self.app)

    # -- upstream dispatch ------------------------------------------------------

    def _build_upstream(self, mode: str) -> SamplingUpstream:
        """Pick the sampling backend implementation. The single mode-string read;
        every later dispatch goes through the returned object."""
        if mode == "messages":
            return MessagesUpstream(self)
        if mode == "chat":
            return ChatUpstream(self)
        if mode == "responses":
            return ResponsesUpstream(self)
        if mode == "tinker":
            return TinkerUpstream(self)
        return SglangUpstream(self)  # sglang (default)

    def _render_prompt_for(
        self, translated: list[dict], tools_schema: list[dict] | None
    ) -> list[int]:
        """Render hub-format messages to token ids (sglang/tinker). No-op when no
        tokenizer is configured (returns []); server-backed backends don't call this."""
        tok = self.tokenizer
        if tok is None:
            return []
        return _render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)

    async def _tokenize_messages_server(
        self,
        base_url: str,
        translated: list[dict],
        tools_schema: list[dict] | None,
        session_id: str | None,
    ) -> list[int] | None:
        """Render messages to token ids via the upstream server's tokenize endpoint.

        Avoids the local tokenizer entirely when the server can render the chat
        template itself. Verified against vLLM 0.26 (``POST /tokenize`` with
        ``messages``) and sglang >= 0.5.12 (``POST /v1/tokenize`` with ``messages``).
        Returns ``None`` (caller falls back to local render) on any failure — a
        missing/old/errored endpoint is a silent degradation, not a hard error.
        """
        if not base_url:
            return None
        url = base_url.rstrip("/")
        # vLLM exposes /tokenize; sglang exposes /v1/tokenize. Try both shapes; the
        # server that doesn't recognise the path 404s and we fall back.
        for path in ("/tokenize", "/v1/tokenize"):
            payload: dict[str, Any] = {
                "messages": translated,
                "add_generation_prompt": True,
            }
            if tools_schema:
                payload["tools"] = tools_schema
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(
                        f"{url}{path}", json=payload,
                        headers={"Content-Type": "application/json", **self._inbound_auth},
                    ) as r:
                        if r.status >= 400:
                            continue  # try the other path / fall back
                        data = await r.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return None
            ids = data.get("input_ids") or data.get("tokens") or []
            if isinstance(ids, list) and ids:
                self.logger.debug(
                    "[%s] sid=%s server-side tokenize -> %d ids", self.log_prefix, session_id, len(ids),
                )
                return [int(i) for i in ids]
        return None

    async def _call_upstream(
        self,
        prompt_ids: list[int],
        session: Any,
        body: dict,
        session_id: str | None,
        *,
        translated: list[dict] | None = None,
        tools_schema: list[dict] | None = None,
    ) -> SamplingResult:
        """Forward one turn to the configured upstream backend.

        Dispatch is a single call to ``self.upstream.sample``; the implementation
        (Messages/Chat/Responses/Sglang/Tinker) encapsulates its own rendering and
        returns a :class:`SamplingResult`. Block-level backends set
        ``content_blocks``; token-level backends set ``turn.output_ids`` +
        ``top_logprobs``. The side-effect channels (``_last_content_blocks`` /
        ``_last_top_logprobs``) are absorbed into the result here for the
        legacy consumers that still read them.
        """
        result = await self.upstream.sample(
            translated or [], tools_schema, body, session, session_id
        )
        # Keep the legacy side-effect channels populated for any caller still
        # reading them (finish_session reads _last_top_logprobs; tests assert on
        # _last_content_blocks). _run_turn now reads result.* directly.
        if result.content_blocks is not None:
            self._last_content_blocks = result.content_blocks
        if result.top_logprobs is not None:
            self._last_top_logprobs = result.top_logprobs
        return result

    def _anthropic_body_from_hub(
        self,
        body: dict,
        translated: list[dict] | None,
        tools_schema: list[dict] | None,
    ) -> dict:
        """Build a legal Anthropic /v1/messages body from hub-format messages.

        Used when the downstream wire format is chat/responses but the upstream
        speaks the messages API. ``max_tokens`` is required by the Anthropic API,
        so a default is supplied when the downstream body omits it (the Responses
        API spells it ``max_output_tokens``).
        """
        from anyharness.adapters.anthropic import (
            chat_tools_to_anthropic_tools,
            hub_to_anthropic_messages,
        )

        msgs, system = hub_to_anthropic_messages(translated or [])
        out: dict[str, Any] = {
            "model": os.environ.get("SLIME_MESSAGES_MODEL") or body.get("model") or "slime-actor",
            "messages": msgs,
            "max_tokens": int(
                body.get("max_tokens") or body.get("max_output_tokens") or 4096
            ),
        }
        if system:
            out["system"] = system
        anth_tools = chat_tools_to_anthropic_tools(tools_schema)
        if anth_tools:
            out["tools"] = anth_tools
        for k in ("temperature", "top_p", "stop_sequences"):
            if body.get(k) is not None:
                out[k] = body[k]
        return out

    async def _call_messages_upstream(
        self,
        body: dict,
        session_id: str | None,
        translated: list[dict] | None = None,
        tools_schema: list[dict] | None = None,
    ) -> TurnRecord:
        """Forward a /v1/messages body to an arbitrary messages-API upstream.

        When the downstream wire format is already Anthropic (``/v1/messages``),
        ``body`` is forwarded verbatim as before. When it is ``chat`` or
        ``responses``, the body is rebuilt from the hub-format ``translated``
        messages via :func:`hub_to_anthropic_messages`, because the raw body is
        not a legal ``/v1/messages`` payload: a Responses body carries ``input``
        instead of ``messages`` (hard 400 from the upstream), and a chat body
        carries OpenAI-shaped ``tools`` an Anthropic upstream will not read.

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
        if getattr(self, "_downstream_format", "messages") == "messages":
            fwd_body = dict(body)
        else:
            fwd_body = self._anthropic_body_from_hub(body, translated, tools_schema)
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

        # Stash the upstream's own usage for MessagesUpstream.sample to surface onto
        # SamplingResult (input_tokens isn't parsed by _parse_messages_json yet, so
        # only output_tokens is truthful here; input falls back to local render).
        if usage_out:
            self._last_usage = {"output_tokens": int(usage_out), "input_tokens": None}
        return TurnRecord(
            prompt_ids=[],
            output_ids=output_ids,
            finish_reason=finish,
            output_log_probs=[],
        ), content_blocks

    def _pair_logprobs_with_upstream_ids(
        self,
        token_ids: list[int] | None,
        log_probs: list[float],
        *,
        session_id: str | None,
        mode: str,
    ) -> tuple[list[int], list[float]]:
        """Pair captured logprobs with the upstream's own token ids, or drop both.

        ``record_turn`` asserts ``len(output_log_probs) == len(output_ids)``, so
        logprobs are only usable alongside exactly one id per value — and TITO
        requires those ids be the ones the server sampled, not ids recovered from
        decoded text or from vocab lookups of token strings (tokenization is not
        injective, so a recovered sequence can differ from what the policy
        produced, silently training on tokens it never generated).

        Returns ``([], [])`` when the upstream supplied no ids or the counts
        disagree. Dropping is deliberate: there is no safe reconstruction, and
        message-level SFT is strictly better than token-level data that is subtly
        wrong. Enable ids by pointing ``chat`` mode at vLLM >= 0.10.2, or use
        ``UPSTREAM_MODE=sglang`` whose ``/generate`` path is natively token-in
        token-out.
        """
        if not log_probs:
            return [], []
        if not token_ids:
            self.logger.warning(
                "[%s] sid=%s %s upstream returned %d logprobs but no token_ids "
                "(needs vLLM >= 0.10.2 honouring return_token_ids); dropping "
                "logprobs — re-deriving ids would violate token-in-token-out",
                self.log_prefix, session_id, mode, len(log_probs),
            )
            return [], []
        if len(token_ids) != len(log_probs):
            self.logger.warning(
                "[%s] sid=%s %s upstream token_ids (%d) and logprobs (%d) disagree; "
                "dropping both rather than emit a misaligned pair",
                self.log_prefix, session_id, mode, len(token_ids), len(log_probs),
            )
            return [], []
        return token_ids, log_probs

    async def _chat_upstream_token_ids_and_logprobs(
        self, base_url: str, api_key: str, kwargs: dict, session_id: str | None
    ) -> tuple[list[int] | None, list[float] | None]:
        """Re-issue the chat call over plain HTTP to recover ids AND logprobs together.

        litellm <= 1.88 keeps ``extra_body`` nested, so vLLM (>=0.10.2) returns
        ``token_ids: None`` even with the flag set. We reach here only when the
        litellm path already returned logprobs but no ids. Re-issuing over plain
        HTTP with ``return_token_ids`` at the top level returns BOTH token_ids and
        logprobs from the SAME response — critical for alignment, because logprobs
        (from the litellm call) and ids (from a separate call) can differ in length
        when generation is non-deterministic (reasoning tokens). Using the raw
        response's own paired logprobs guarantees ``len(ids) == len(logprobs)``.
        """
        if not base_url:
            return None, None
        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        payload = {
            k: v for k, v in kwargs.items()
            if k not in ("api_base", "api_key", "extra_headers", "extra_body", "model")
        }
        payload["model"] = str(kwargs.get("model", "")).split("/", 1)[-1]
        payload["return_token_ids"] = True  # top-level, the only place vLLM reads it
        if kwargs.get("logprobs"):
            payload["logprobs"] = True
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(kwargs.get("extra_headers") or {})
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=payload, headers=headers) as resp:
                    if resp.status >= 400:
                        return None, None
                    data = json.loads(await resp.text())
        except Exception:
            return None, None
        choice = (data.get("choices") or [{}])[0]
        ids = choice.get("token_ids")
        ids = list(ids) if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids) else None
        # logprobs from the SAME response (OpenAI shape: choices[0].logprobs.content)
        lp_entries = (choice.get("logprobs") or {}).get("content") or []
        lps = [float(e["logprob"]) for e in lp_entries if isinstance(e, dict) and e.get("logprob") is not None] or None
        if ids is not None:
            self.logger.info(
                "[%s] sid=%s recovered %d token ids + %d logprobs via raw HTTP "
                "(litellm extra_body not flattened); ids now TITO-safe",
                self.log_prefix, session_id, len(ids), len(lps or []),
            )
        return ids, lps

    async def _call_chat_upstream_raw(
        self,
        translated: list[dict],
        tools_schema: list[dict] | None,
        body: dict,
        session_id: str | None,
    ) -> dict:
        """POST /chat/completions over plain HTTP, bypassing litellm's request validator.

        litellm 1.88's request-body validator rejects the OpenAI-standard
        ``tool_calls[].type == "function"`` shape on vLLM 0.26 (expects a
        non-standard ChatCompletionMessageCustomToolCallParam). So any chat
        request that replays an assistant tool_call — every post-tool agentic
        turn — must bypass litellm. This issues the call directly and parses the
        same response shape litellm would have returned, in ONE request, so
        token_ids and logprobs are paired (same-response, TITO-safe).

        Returns a dict with: blocks, stop_reason, usage, output_ids, output_log_probs.
        """
        from anyharness.adapters.anthropic import chat_response_to_blocks

        base_url = os.environ.get("SLIME_CHAT_BASE_URL") or ""
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = base_url + "/v1"
        api_key = os.environ.get("SLIME_CHAT_API_KEY") or self._inbound_auth.get("authorization") or ""
        if api_key.lower().startswith("bearer "):
            api_key = api_key[7:]
        model = os.environ.get("SLIME_CHAT_MODEL") or "gpt-4o"
        args_as_dict = os.environ.get("SLIME_CHAT_ARGS_AS_DICT", "") == "1"
        wire_messages = translated if args_as_dict else _stringify_tool_call_args(translated)
        # vLLM 0.26 requires every tool_call to carry an `id` (and the following
        # tool message a matching `tool_call_id`). The tree's tool_call_dict drops
        # the wire id on purpose (for dict-equality matching), so re-synthesize one
        # per assistant tool_call and pair it onto the following tool messages —
        # positionally, like hub_to_anthropic does for the messages upstream.
        wire_messages = _ensure_tool_call_ids(wire_messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": wire_messages,
            "stream": False,
        }
        if tools_schema:
            payload["tools"] = tools_schema
        if body.get("max_tokens"):
            payload["max_tokens"] = int(body["max_tokens"])
        want_logprobs = os.environ.get("SLIME_CHAT_LOGPROBS") == "1"
        if want_logprobs:
            payload["logprobs"] = True
            top_k = int(os.environ.get("SLIME_CHAT_TOP_LOGPROBS", "0") or "0")
            if top_k > 0:
                payload["top_logprobs"] = top_k
        payload["return_token_ids"] = True  # top-level, the only place vLLM reads it

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        extra = {k: v for k, v in self._inbound_auth.items() if k.lower() not in ("authorization",)}
        headers.update(extra)
        url = f"{base_url}/chat/completions"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=900)) as sess:
            async with sess.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"chat raw upstream {resp.status}: {text[:400]}")
                data = await resp.json(content_type=None)

        choice = (data.get("choices") or [{}])[0]
        # build a litellm-shaped object so chat_response_to_blocks works
        blocks, stop_reason = chat_response_to_blocks(_AttrView(data))

        u = data.get("usage")
        usage = None
        if isinstance(u, dict):
            usage = {
                "input_tokens": u.get("prompt_tokens"),
                "output_tokens": u.get("completion_tokens"),
            }

        output_ids: list[int] = []
        output_log_probs: list[float] = []
        if want_logprobs:
            ids = choice.get("token_ids")
            if isinstance(ids, list) and ids:
                output_ids = [int(i) for i in ids]
            lp_entries = (choice.get("logprobs") or {}).get("content") or []
            output_log_probs = [float(e["logprob"]) for e in lp_entries
                                 if isinstance(e, dict) and e.get("logprob") is not None]
            if output_log_probs and len(output_log_probs) != len(output_ids):
                self.logger.warning(
                    "[%s] sid=%s chat raw ids (%d) != logprobs (%d); dropping logprobs",
                    self.log_prefix, session_id, len(output_ids), len(output_log_probs))
                output_log_probs = []

        return {
            "blocks": blocks,
            "stop_reason": stop_reason,
            "usage": usage,
            "output_ids": output_ids,
            "output_log_probs": output_log_probs,
        }

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
        back to Claude Code. Message-level SFT by default; ``SLIME_CHAT_LOGPROBS=1``
        adds per-token logprobs when a tokenizer can align them to ids.

        Env: ``SLIME_CHAT_BASE_URL``, ``SLIME_CHAT_API_KEY``, ``SLIME_CHAT_MODEL``
        (e.g. an OpenAI-compatible gateway). Auth from the inbound request
        (Authorization/x-api-key) is forwarded via ``api_key`` when no explicit
        key is set, so a client's own credential reaches the upstream.
        """
        import litellm

        from anyharness.adapters.anthropic import chat_response_to_blocks

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
            # Ask the server for the token ids it sampled (vLLM >= 0.10.2). This is
            # the only TITO-safe way to align logprobs to ids; without it the
            # logprobs get dropped below. Passed via extra_body since it is a vLLM
            # extension, not an OpenAI field. Note vllm#27482: return_token_ids can
            # drop tokens on *streaming* tool calls — this path is non-streaming.
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "return_token_ids": True}

        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            self.logger.warning("[%s] sid=%s chat upstream failed: %s", self.log_prefix, session_id, exc)
            raise

        blocks, stop_reason = chat_response_to_blocks(response)
        self._last_content_blocks = blocks
        # Surface the upstream's own usage (litellm ModelResponse.usage) so
        # SamplingResult.usage reports truthful prompt/completion token counts
        # instead of len(local render)/0.
        u = getattr(response, "usage", None)
        if u is not None:
            self._last_usage = {
                "input_tokens": getattr(u, "prompt_tokens", None),
                "output_tokens": getattr(u, "completion_tokens", None),
            }

        # Extract per-token logprobs from the chat response (OpenAI shape:
        # choices[0].logprobs.content = [{token, logprob, top_logprobs:[...]}]).
        # Pack into output_log_probs for the TurnRecord. top_logprobs_num on the
        # TurnRecord is left to the sglang path; chat stores the sampled-token
        # logprob only (sufficient for GRPO; top-k is in the response object if needed).
        output_log_probs: list[float] = []
        upstream_ids: list[int] | None = None
        if os.environ.get("SLIME_CHAT_LOGPROBS") == "1":
            try:
                choice = response.choices[0]
                lp = getattr(choice, "logprobs", None)
                if lp and getattr(lp, "content", None):
                    for entry in lp.content:
                        lp_val = getattr(entry, "logprob", None)
                        if lp_val is not None:
                            output_log_probs.append(float(lp_val))
                upstream_ids = _extract_upstream_token_ids(choice)
            except (AttributeError, IndexError, TypeError):
                pass  # upstream didn't return logprobs despite the request

        # litellm <= 1.88 keeps ``extra_body`` nested instead of flattened to the
        # top level, so vLLM (>=0.10.2) silently returns token_ids=None even with
        # the flag set. Re-issue over plain HTTP with return_token_ids at the top
        # level — and take the logprobs from the SAME response, so ids and
        # logprobs are paired (the litellm call's logprobs can differ in length
        # from a separately-sampled raw call when generation is non-deterministic).
        if output_log_probs and not upstream_ids:
            raw_ids, raw_lps = await self._chat_upstream_token_ids_and_logprobs(
                base_url, api_key, kwargs, session_id
            )
            if raw_ids is not None:
                upstream_ids = raw_ids
                if raw_lps is not None:
                    output_log_probs = raw_lps  # use the same-response logprobs

        # record_turn asserts len(output_log_probs) == len(output_ids); ids must come
        # from the upstream (TITO), never be re-derived here.
        output_ids, output_log_probs = self._pair_logprobs_with_upstream_ids(
            upstream_ids, output_log_probs, session_id=session_id, mode="chat"
        )

        # map Anthropic stop_reason -> sglang finish_reason shape (tool_use->tool_calls etc.)
        finish = {
            "max_tokens": "length",
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason or "stop")
        return TurnRecord(
            prompt_ids=[], output_ids=output_ids, finish_reason=finish, output_log_probs=output_log_probs
        )

    async def _call_responses_upstream_raw(
        self,
        kwargs: dict[str, Any],
        base_url: str,
        api_key: str,
        session_id: str | None,
        *,
        litellm_error: Exception,
    ) -> Any | None:
        """Re-issue a Responses call as plain HTTP, bypassing litellm's validation.

        Returns an object exposing ``.output`` / ``.status`` (what
        ``responses_output_to_blocks`` reads via ``getattr``), or ``None`` when the
        upstream genuinely failed and the original litellm error should surface.

        This exists because litellm's ``ResponsesAPIResponse`` requires
        ``created_at``; an upstream that omits it gets its 200 turned into an
        ``APIError`` even though the body carries complete output. We only reach
        here after litellm already raised, so the cost is one extra request on a
        path that was otherwise about to fail the whole turn.
        """
        url = base_url.rstrip("/")
        if not url.endswith("/responses"):
            url = f"{url}/responses"
        payload = {
            k: v for k, v in kwargs.items()
            if k not in ("api_base", "api_key", "extra_headers", "model")
        }
        # litellm takes "openai/<model>"; the wire wants the bare model name.
        payload["model"] = str(kwargs.get("model", "")).split("/", 1)[-1]
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(kwargs.get("extra_headers") or {})

        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=payload, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        self.logger.warning(
                            "[%s] sid=%s responses raw fallback got HTTP %s: %s",
                            self.log_prefix, session_id, resp.status, text[:200],
                        )
                        return None
                    data = json.loads(text)
        except Exception as exc:  # network / JSON failure: surface the original
            self.logger.warning(
                "[%s] sid=%s responses raw fallback failed: %s", self.log_prefix, session_id, exc
            )
            return None

        if not isinstance(data, dict) or "output" not in data:
            return None

        missing = sorted(
            f for f in ("created_at",) if f not in data
        )
        self.logger.warning(
            "[%s] sid=%s Responses upstream is missing %s, which litellm requires "
            "(%s: %s); parsed the 200 directly instead. Output is intact; this only "
            "bypasses litellm's schema validation.",
            self.log_prefix, session_id, missing or "required field(s)",
            type(litellm_error).__name__, str(litellm_error)[:120],
        )
        return _AttrView(data)

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

        from anyharness.adapters.anthropic import responses_output_to_blocks

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
            # Tools arrive in the OpenAI chat-completions shape
            # ({type:function, function:{name,description,parameters}}) because the
            # hub format is chat-shaped. The Responses API wants them FLAT
            # ({type:function, name, description, parameters}) -- send the nested
            # shape and the upstream silently ignores the tools, so the model
            # answers in prose instead of emitting a function_call.
            kwargs["tools"] = [
                {
                    "type": "function",
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}},
                }
                for t in tools_schema
                if isinstance(t, dict) and t.get("function")
            ]
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
            # litellm validates the payload against its own ResponsesAPIResponse
            # model, which requires `created_at` -- a field several Responses-API
            # implementations omit. Such an upstream returns a perfectly usable 200
            # that litellm still raises on, so retry once over plain HTTP and parse
            # the body ourselves rather than failing the turn.
            response = await self._call_responses_upstream_raw(
                kwargs, base_url, api_key, session_id, litellm_error=exc
            )
            if response is None:
                self.logger.warning(
                    "[%s] sid=%s responses upstream failed: %s", self.log_prefix, session_id, exc
                )
                raise

        blocks, stop_reason = responses_output_to_blocks(response)
        self._last_content_blocks = blocks
        # Surface the upstream's own usage (Responses API usage: input_tokens/
        # output_tokens). Works for both the litellm pydantic response and the
        # _AttrView fallback (getattr falls through to the parsed dict).
        u = getattr(response, "usage", None)
        if u is not None:
            self._last_usage = {
                "input_tokens": getattr(u, "input_tokens", None)
                if not isinstance(u, dict) else u.get("input_tokens"),
                "output_tokens": getattr(u, "output_tokens", None)
                if not isinstance(u, dict) else u.get("output_tokens"),
            }

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

        # Same TITO requirement as the chat path: the Responses API exposes no
        # token-id field, so captured logprobs are always dropped here rather than
        # paired with re-derived ids. Passing None keeps that explicit (and logged).
        output_ids, output_log_probs = self._pair_logprobs_with_upstream_ids(
            None, output_log_probs, session_id=session_id, mode="responses"
        )

        finish = {
            "max_tokens": "length",
            "end_turn": "stop",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }.get(stop_reason, stop_reason or "stop")
        return TurnRecord(
            prompt_ids=[], output_ids=output_ids, finish_reason=finish, output_log_probs=output_log_probs
        )

    def _recover_text_tool_calls(
        self, blocks: list[dict], tools_schema: list[dict] | None
    ) -> list[dict]:
        """Promote tool calls the upstream left as raw text into ``tool_use`` blocks.

        An upstream that has no tool parser configured (or that does not recognise
        the model's tool syntax) returns the call as literal text such as
        ``<tool_call>read<arg_key>file_path</arg_key>...</tool_call>``. Downstream
        that reads as an assistant message with no tool call at all, so the agent
        loops asking for a file it never receives.

        Only applied when the upstream returned NO structured ``tool_use`` block and
        the text carries the marker, so a well-behaved upstream is untouched.
        """
        if not tools_schema or not blocks:
            return blocks
        if any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks):
            return blocks
        if not any(
            isinstance(b, dict) and b.get("type") == "text" and "<tool_call>" in (b.get("text") or "")
            for b in blocks
        ):
            return blocks

        from anyharness.parsing import parse_xml_tool_uses

        out: list[dict] = []
        recovered = 0
        for b in blocks:
            if not (isinstance(b, dict) and b.get("type") == "text"):
                out.append(b)
                continue
            cleaned, tool_uses = parse_xml_tool_uses(b.get("text") or "", tools_schema)
            if not tool_uses:
                out.append(b)
                continue
            if cleaned.strip():
                out.append({"type": "text", "text": cleaned})
            for tu in tool_uses:
                recovered += 1
                out.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:16]}",
                    "name": tu["name"],
                    "input": tu.get("input") or {},
                })
        if recovered:
            self.logger.info(
                "[%s] recovered %d tool call(s) from raw upstream text "
                "(upstream returned no structured tool_use)",
                self.log_prefix,
                recovered,
            )
        return out

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

        blocks = self._recover_text_tool_calls(blocks, tools_schema)
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
        # message_level backends (messages/chat/responses) carry no tokens
        # (output_ids == []): the token-level get_trajectory would return [] and
        # pop the per-sid tree while doing so, leaving nothing for a message-level
        # fallback. Branch to the message-level path BEFORE touching get_trajectory.
        if self.upstream.message_level:
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
        from anyharness.message_dump import get_trajectory_messages

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
            # messages-mode forwards the body verbatim, which is only valid when the
            # downstream body is already Anthropic-shaped. Record the inbound wire
            # format so _call_messages_upstream can translate when it is not.
            self._downstream_format = downstream_format
            result = await self._call_upstream(prompt_ids, s, body, sid, translated=translated, tools_schema=tools_schema)
            turn = result.turn

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
            # Block-level backends (messages/chat/responses) returned Anthropic-
            # structured content blocks, which parse_model_output CANNOT recover
            # from decoded text. Build the manager_message directly from those
            # blocks so tool_calls + reasoning_content survive into the trajectory
            # (and thus the SFT dump). Token-level backends (sglang/tinker) have no
            # blocks and fall through to the parsed reply (unchanged). The branch is
            # on the result shape, not on the upstream_mode string.
            if result.content_blocks:
                reply = self._reply_from_content_blocks(result.content_blocks, turn.finish_reason, tools_schema)
            else:
                reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)

            # Truthful usage: message-level backends report the upstream's own token counts
            # (server-rendered). Token-level backends (sglang/tinker) leave usage None
            # and fall back to the locally-rendered prompt_ids / output_ids length.
            if result.usage:
                in_tok = int(result.usage.get("input_tokens") or len(prompt_ids))
                out_tok = int(result.usage.get("output_tokens") or len(turn.output_ids))
            else:
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


_tinker_mismatch_warned: set[str] = set()


def _warn_if_tokenizer_mismatched(adapter: BaseAdapter, session_id: str | None) -> None:
    """Warn once when the local tokenizer looks unrelated to the served model.

    tinker mode renders the chat template locally, so a tokenizer from a different
    model family silently produces a prompt the server never expects — the symptom
    is the model echoing the prompt back rather than answering it, which is easy to
    mistake for a bad checkpoint. Only a heuristic (the served name is free-form),
    so this warns rather than raises.
    """
    served = os.environ.get("TINKER_BASE_MODEL") or ""
    tok = getattr(adapter, "tokenizer", None)
    local = str(getattr(tok, "name_or_path", "") or "")
    if not served or not local:
        return
    key = f"{local}->{served}"
    if key in _tinker_mismatch_warned:
        return

    def _family(name: str) -> str:
        """Leading alphabetic run of the model name: Qwen3.6-35B-A3B -> "qwen"."""
        leaf = name.rstrip("/").split("/")[-1].lower()
        return "".join(itertools.takewhile(str.isalpha, leaf))

    if _family(local) != _family(served):
        _tinker_mismatch_warned.add(key)
        adapter.logger.warning(
            "[%s] sid=%s local tokenizer %r may not match served model %r; tinker "
            "mode renders the chat template locally, so a mismatch yields prompts "
            "the server misreads (often echoing the prompt back). Point MODEL_PATH "
            "at the served model's tokenizer.",
            adapter.log_prefix, session_id, local, served,
        )


def _tinker_sampling_params(session: Any, body: dict, *, adapter: BaseAdapter) -> dict:
    """Translate our sampling dict to Tinker's SamplingParams.

    Tinker accepts only ``max_tokens``/``temperature``/``top_k``/``top_p``/``stop``/
    ``seed``; the sglang-specific detokenization knobs (``skip_special_tokens`` &c.)
    are meaningless here because nothing is detokenized on the wire.
    """
    sp = _sampling_params(session, body, max_token_keys=adapter.max_token_keys, stop_keys=adapter.stop_keys)
    out: dict[str, Any] = {"max_tokens": int(sp.get("max_new_tokens", 4096))}
    for k in ("temperature", "top_p", "top_k", "stop", "seed"):
        if sp.get(k) is not None:
            out[k] = sp[k]
    return out


async def call_tinker_sample(
    prompt_ids: list[int],
    session: Any,
    body: dict,
    *,
    adapter: BaseAdapter,
    session_id: str | None = None,
) -> TurnRecord:
    """POST one turn to a Tinker/Mint ``/api/v1/asample`` and await its future.

    Natively token-in token-out: ``prompt`` carries raw token ids and the reply's
    ``sequences[0]`` returns ``tokens`` alongside an equal-length ``logprobs``, so
    ids never round-trip through text and the TITO invariant holds by construction
    (no ``return_token_ids``-style opt-in, no vocab lookups).

    Two-step by design — ``asample`` returns ``{"request_id": ...}`` and the result
    is collected from ``/api/v1/retrieve_future``. Module-level (not a method) so
    tests can monkeypatch it, matching :func:`call_sglang_generate`.
    """
    logger = adapter.logger
    base = (os.environ.get("TINKER_BASE_URL") or "").rstrip("/")
    api_key = os.environ.get("TINKER_API_KEY") or ""
    sp = _tinker_sampling_params(session, body, adapter=adapter)
    _warn_if_tokenizer_mismatched(adapter, session_id)

    if session.max_context_tokens > 0:
        remaining_context = session.max_context_tokens - len(prompt_ids)
        if remaining_context <= 0:
            logger.warning(
                "[%s] sid=%s prompt exceeds max_context_tokens (%d >= %d)",
                adapter.log_prefix, session_id, len(prompt_ids), session.max_context_tokens,
            )
            return TurnRecord(prompt_ids=list(prompt_ids), output_ids=[], finish_reason="length"), None
        sp["max_tokens"] = min(int(sp.get("max_tokens", remaining_context)), remaining_context)

    payload: dict[str, Any] = {
        "num_samples": 1,
        "prompt": {"chunks": [{"type": "encoded_text", "tokens": list(prompt_ids)}]},
        "sampling_params": sp,
    }
    # model_id selects a specific training step; base_model uses the served base.
    if os.environ.get("TINKER_MODEL_ID"):
        payload["model_id"] = os.environ["TINKER_MODEL_ID"]
    else:
        payload["base_model"] = os.environ.get("TINKER_BASE_MODEL") or ""
    # Verified against a live Mint gateway: prompt_logprobs also gates the
    # *sampled* logprobs — with it false, sequences[].logprobs comes back empty.
    # Since token-level logprobs are the whole reason to use this mode, default
    # it on; TINKER_LOGPROBS=0 opts out for message-level-only runs.
    if os.environ.get("TINKER_LOGPROBS", "1") != "0":
        payload.update(prompt_logprobs=True, include_prompt_logprobs=True)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)

    async with aiohttp.ClientSession(timeout=timeout) as sess:
        async with sess.post(f"{base}/api/v1/asample", json=payload, headers=headers) as r:
            if r.status >= 400:
                text = await r.text()
                logger.warning(
                    "[%s] sid=%s tinker asample %d: %.200s", adapter.log_prefix, session_id, r.status, text
                )
                raise RuntimeError(f"tinker asample {r.status}: {text[:400]}")
            submit = await r.json(content_type=None)
        request_id = submit.get("request_id")
        if not request_id:
            raise RuntimeError(f"tinker asample returned no request_id: {str(submit)[:200]}")

        data = await _tinker_retrieve_future(
            sess, base, headers, request_id, adapter=adapter, session_id=session_id
        )

    seqs = data.get("sequences") or []
    if not seqs:
        raise RuntimeError(f"tinker future had no sequences: {str(data)[:200]}")
    seq = seqs[0]
    output_ids = [int(t) for t in (seq.get("tokens") or [])]
    output_log_probs = [float(x) for x in (seq.get("logprobs") or [])]
    # The server pairs these itself; a mismatch means a shape change we should not
    # paper over, since record_turn asserts on it and training would consume it.
    if output_log_probs and len(output_log_probs) != len(output_ids):
        logger.warning(
            "[%s] sid=%s tinker tokens (%d) != logprobs (%d); dropping logprobs",
            adapter.log_prefix, session_id, len(output_ids), len(output_log_probs),
        )
        output_log_probs = []
    finish = {"length": "length", "stop": "stop", "abort": "abort"}.get(
        seq.get("stop_reason") or "stop", seq.get("stop_reason") or "stop"
    )
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=finish,
        output_log_probs=output_log_probs,
    ), None


async def _tinker_retrieve_future(
    sess: aiohttp.ClientSession,
    base: str,
    headers: dict,
    request_id: str,
    *,
    adapter: BaseAdapter,
    session_id: str | None,
) -> dict:
    """Poll ``/api/v1/retrieve_future`` until the sample result materialises.

    The endpoint returns the completed payload directly; a reply without
    ``sequences`` means "not ready yet", so back off and retry until the deadline.
    """
    poll_timeout = float(os.environ.get("TINKER_FUTURE_TIMEOUT", "900") or "900")
    delay, waited = 0.5, 0.0
    while True:
        async with sess.post(
            f"{base}/api/v1/retrieve_future", json={"request_id": request_id}, headers=headers
        ) as r:
            if r.status >= 400:
                text = await r.text()
                raise RuntimeError(f"tinker retrieve_future {r.status}: {text[:400]}")
            data = await r.json(content_type=None)
        if data.get("sequences"):
            return data
        if waited >= poll_timeout:
            raise RuntimeError(f"tinker future {request_id} not ready after {poll_timeout}s")
        await asyncio.sleep(delay)
        waited += delay
        delay = min(delay * 1.5, 5.0)  # ramp down polling pressure on long generations


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
