# Tasks

## 1. Raw-HTTP chat call (bypass litellm request validator)

- [x] 1.1 Add `_call_chat_upstream_raw(self, kwargs, base_url, api_key, session_id) -> dict` in `common.py`: POST `/chat/completions` with `return_token_ids`+`logprobs` top-level; parse and return `{blocks, stop_reason, usage, output_ids, output_log_probs}`. Reuse `chat_response_to_blocks`. (Folds the existing `_chat_upstream_token_ids_and_logprobs` into one call.)
- [x] 1.2 `ChatUpstream.sample`: detect `tool_calls` in `translated`; if present, call `_call_chat_upstream_raw` and build `SamplingResult(turn=..., content_blocks=blocks, usage=usage)`. If `SLIME_CHAT_LOGPROBS=1`, the turn carries real ids+logprobs (paired from the one response).

## 2. Tool-free path unchanged

- [x] 2.1 Verify `_call_chat_upstream` (litellm) is still used when no `tool_calls` in the request; no change to its body.

## 3. Tests

- [x] 3.1 Tool-call-bearing request → raw-HTTP branch (stub aiohttp, assert POST happens, litellm.acompletion NOT called).
- [x] 3.2 Tool-free request → litellm branch (litellm.acompletion called, no aiohttp POST).
- [x] 3.3 Raw-HTTP path with logprobs returns paired ids+logprobs (same response).

## 4. Verify

- [x] 4.1 `pytest tests/ -q` all green.
- [x] 4.2 Live: vLLM 0.26 chat upstream, pi messages/chat/responses downstream — multi-turn tool loop completes, answer correct.
