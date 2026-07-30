## Why

vLLM is the only token-id-bearing backend that currently reaches the adapter through the `chat` upstream — and that path is a pile of vLLM/litellm-specific workarounds (`_call_chat_upstream_raw`, `_nest_tools`, `_ensure_tool_call_ids`, `_stringify_tool_call_args`) needed because chat-completions forces a messages+tools shape that vLLM 0.26 + litellm 1.88 fight over. vLLM also exposes `/v1/completions`, which is structurally identical to sglang's `/generate`: a text-completion endpoint that is natively token-in/token-out (accepts `prompt` as a token-id array; returns `choice.token_ids` + `choice.logprobs.token_logprobs` 1:1-aligned, verified). A `completions` upstream gives vLLM the clean token-level path that reuses the existing `parse_model_output` → `_build_reply` pipeline, with no chat-shape adaptation. This was verified live against vLLM 0.26 before proposing.

## What Changes

- Add `UPSTREAM_MODE=completions`: a new token-level upstream (`message_level=False`) targeting an OpenAI-compatible `/v1/completions` endpoint (vLLM, and sglang's OpenAI-serve layer).
- New `CompletionsUpstream` class, structurally parallel to `SglangUpstream`: render `prompt_ids` via `_tokenize_messages_server` (server `/tokenize`, local-render fallback), POST `{prompt: prompt_ids, return_token_ids: true, logprobs, max_tokens, sampling}`, parse `choice.token_ids` + `logprobs.token_logprobs` + `top_logprobs` into a `SamplingResult`.
- **Input-side TITO**: the prompt is passed as a raw token-id array, so the server uses the adapter's ids verbatim — it never re-tokenizes text. Verified: vLLM echoes `prompt_token_ids` identical to the input array.
- Output ids and logprobs come from the **same** response (single request), guaranteeing the length invariant — no separate raw-HTTP re-issue as the chat path needs.
- New env: `SLIME_COMPLETIONS_BASE_URL`, `SLIME_COMPLETIONS_API_KEY`, `SLIME_COMPLETIONS_MODEL`, `SLIME_COMPLETIONS_LOGPROBS=1`, `SLIME_COMPLETIONS_TOP_LOGPROBS=N`.
- The `chat` upstream is **unchanged** and **not replaced**: it remains the path for OpenAI-compatible endpoints that only return text (e.g. mintcn) and for which the messages/tools shape is the only option.

## Capabilities

### New Capabilities

(none — `completions` is a new backend within the existing `sampling-upstream` capability, not a new capability)

### Modified Capabilities

- `sampling-upstream`: extend the token-level backend set to include `completions` and pin its input-side TITO contract (prompt passed as a token-id array; output ids+logprobs from the same response). The existing token-level requirements (structural pairing, StopReason mapping, top-k sentinel padding, truthful usage fallback) already frame their scenarios generically as "a token-level upstream"; `completions` conforms to them and is enumerated alongside sglang/tinker.

## Impact

- `src/anyharness/adapters/common.py` — new `CompletionsUpstream` class + `call_completions` module function; `_build_upstream` gains a `completions` branch; `UPSTREAM_MODE` validation accepts `completions`.
- `src/anyharness/cli.py` — `load_tokenizer`/`Config.validate` handle the `completions` mode + its env vars (local tokenizer optional like sglang, since server `/tokenize` can render).
- `tests/` — new test(s) for `CompletionsUpstream` (token-id pairing, StopReason mapping, top-k padding) mirroring `test_tinker_upstream.py`/`test_sglang_*`.
- No changes to `chat` upstream code or its tests; no breaking changes to existing modes.
