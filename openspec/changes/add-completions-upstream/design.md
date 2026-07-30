## Context

AnyHarness unifies sampling backends behind a `SamplingUpstream` Protocol (`common.py`). Five modes exist: `messages`/`chat`/`responses` (block-level, `message_level=True`) and `sglang`/`tinker` (token-level, `message_level=False`). The token-level path is the clean one: `_run_turn` decodes `turn.output_ids`, runs `parse_model_output`, and builds the reply — no chat-shape negotiation.

vLLM is token-id-capable but today is reached only via the `chat` upstream, whose vLLM path is a stack of workarounds (`_call_chat_upstream_raw`, `_nest_tools`, `_ensure_tool_call_ids`, `_stringify_tool_call_args`, and a same-response re-issue for ids). These exist solely because `/v1/chat/completions` forces a messages+tools shape that litellm 1.88 and vLLM 0.26 disagree on. vLLM also serves `/v1/completions` — a text-completion endpoint that is natively token-in/token-out (verified live against vLLM 0.26 on GPU 0, port 30001):

- `prompt` accepts a **token-id array**; the server echoes `prompt_token_ids` byte-identical to the input (no re-tokenization).
- `choice.token_ids` + `choice.logprobs.token_logprobs` are returned **1:1-aligned** from a single response.
- `choice.logprobs.top_logprobs` carries per-position top-k (dict of `{token_str: logprob}`).
- Tool calls are returned as raw text (the server's tool parser does not run at the completions layer), which is exactly what `parse_model_output` is built to parse.

## Goals / Non-Goals

**Goals:**
- A `completions` token-level upstream that gives vLLM a clean path reusing the sglang-style pipeline, with TITO-safe ids+logprobs from one request.
- Conformance to the existing token-level requirements (structural pairing, StopReason, top-k sentinel padding, usage fallback) without new generic machinery.

**Non-Goals:**
- Removing or altering the `chat` upstream. It stays for text-only OpenAI-compatible endpoints (mintcn) and for any caller already depending on it.
- Server-side tool parsing. `/v1/completions` returns raw text; tool calls are parsed client-side by `parse_model_output`, as sglang already does.
- A second local-tokenizer render path. Prompt rendering reuses `_tokenize_messages_server` (server `/tokenize`) with the existing local-render fallback.

## Decisions

### Decision: `completions` is a new `UPSTREAM_MODE`, not a replacement of `chat`
Two different jobs. `chat` speaks messages+tools to an OpenAI-compatible endpoint that may return only text. `completions` speaks raw token-ids to an endpoint that returns token-ids+logprobs. Collapsing them would push the "is this endpoint token-id-capable?" decision onto the user via fragile auto-detection. A distinct mode is one branch in `_build_upstream` and an explicit env contract.

### Decision: prompt passed as a token-id array, not as text
Verified: vLLM `/v1/completions` accepts `{"prompt": [ids]}` and uses them verbatim. This is input-side TITO — the server never re-tokenizes, so prompt ids are exactly what the adapter rendered (and what the tree forks on). Sending text would let the server re-tokenize and silently drift from the local render used for the think-block fork logic. `prompt_ids` comes from `_tokenize_messages_server` (server `/tokenize`) with local-render fallback — identical to `SglangUpstream`.

### Decision: mirror `SglangUpstream` + `call_sglang_generate`, not `ChatUpstream`
`SglangUpstream.sample` is the template: render prompt → call module function → pack `SamplingResult(turn=..., top_logprobs=...)`. A new `call_completions` module function (module-level so tests can monkeypatch it, matching `call_sglang_generate`) does the POST and parses. `message_level=False` routes it through the token-level `_run_turn` branch unchanged.

### Decision: parse `top_logprobs` with a completions-specific converter
vLLM completions returns `choice.logprobs.top_logprobs` as `list[dict[str, float]]` (token string → logprob), unlike sglang's `list[list[(logprob, token_id)]]`. The converter must map token strings to ids. **Caveat**: token-string→id lookup is exactly the kind of vocab round-trip TITO forbids for *sampled* tokens — but it is only used to fill the *top-k alternatives* (auxiliary training signal), never the sampled `output_ids` (which come from `choice.token_ids`, the server's own ids). We map via the local tokenizer's `convert_tokens_to_ids`; any unmapped alternative gets the sentinel `(0, MASK_LOGPROB)`. This keeps top-k best-effort while the sampled sequence stays strictly server-sourced.

### Decision: reuse `_sampling_params` + adapter's `max_token_keys`/`stop_keys`
Sampling knobs (`temperature`/`top_p`/`max_tokens`/`stop`) flow through the same `_sampling_params` helper sglang uses, mapped onto the OpenAI completions field names. `max_context_tokens` budget enforcement mirrors `call_sglang_generate`.

## Risks / Trade-offs

- [Token-string→id mapping for top-k alternatives can be lossy on special/multibyte tokens] → Mitigation: unmapped alternatives fall back to the sentinel pair; the sampled sequence (`output_ids`) is never affected, so training correctness holds; top-k is auxiliary.
- [Endpoint returns ids+logprobs of mismatched length on some vLLM config] → Mitigation: assert at construction (the structural-pairing requirement already enforces this); a mismatch raises rather than emitting a misaligned pair.
- [Server `/tokenize` unavailable (old vLLM / endpoint without it)] → Mitigation: `_tokenize_messages_server` returns None → local-render fallback, same as sglang. If no tokenizer is configured, prompt_ids is empty and the completions request degrades — same failure mode as sglang without a tokenizer.
- [Tool-call text the server's parser would have normalized differs from raw] → Mitigation: `parse_model_output` with the adapter's `tool_parser`/`reasoning_parser` (qwen3_xml etc.) parses the raw text identically to how sglang parses its raw output — no new parsing surface.
