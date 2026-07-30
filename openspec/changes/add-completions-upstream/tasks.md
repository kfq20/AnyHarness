## 1. Core: CompletionsUpstream + call_completions

- [x] 1.1 Add module-level `call_completions(prompt_ids, session, body, *, adapter, session_id=None) -> (TurnRecord, topk_raw)` to `common.py`: POST `{prompt: prompt_ids, return_token_ids: True, logprobs, max_tokens, sampling}` to `SLIME_COMPLETIONS_BASE_URL/v1/completions`; parse `choice.token_ids` → `output_ids`, `choice.logprobs.token_logprobs` → `output_log_probs`, `choice.finish_reason` → StopReason; mirror `call_sglang_generate`'s `max_context_tokens` budget + `_sampling_params` + abort-on-cancel. Module-level so tests can monkeypatch (matches `call_sglang_generate`).
- [x] 1.2 Add `_completions_topk_to_typed(raw, k, tokenizer) -> TopkLogprobs | None`: convert vLLM `top_logprobs` (`list[dict[str,float]]`) to sentinel-padded pair; map token strings → ids via `tokenizer.convert_tokens_to_ids`; unmapped → `(0, MASK_LOGPROB)`; short positions padded to width k. Returns None when k==0 (mirrors `_sglang_topk_to_typed`).
- [x] 1.3 Add `CompletionsUpstream(_DelegatingUpstream)` with `message_level=False` and a `sample()` that renders `prompt_ids` via `_tokenize_messages_server` (local fallback), calls `call_completions`, packs `SamplingResult(turn=..., top_logprobs=..., usage=None)`. Mirror `SglangUpstream.sample` exactly.
- [x] 1.4 Add `completions` branch to `_build_upstream` and accept `completions` in `UPSTREAM_MODE` validation (`__init__` ValueError message).

## 2. CLI / config

- [x] 2.1 `cli.py`: handle `completions` in `load_tokenizer` (local tokenizer optional — required only when server `/tokenize` unavailable, same as sglang) and `Config.validate` (check `SLIME_COMPLETIONS_BASE_URL`; `_COMPLETIONS_*` env vars).

## 3. Tests

- [x] 3.1 `tests/test_completions_upstream.py`: monkeypatch `call_completions` (aiohttp fake session) — assert `prompt` sent as token-id array, `return_token_ids: true`, `message_level=False`, `output_ids`/`output_log_probs` paired from same response, StopReason mapping (`length`/`stop`).
- [x] 3.2 Test `_completions_topk_to_typed`: short-position sentinel padding, unmapped-token-string → sentinel, k==0 → None.
- [x] 3.3 Test mismatched ids/logprobs lengths raise at construction (structural-pairing invariant).

## 4. Verify

- [x] 4.1 Full suite green: `PYTHONPATH=src python -m pytest tests/ -q` (125 + new tests).
- [x] 4.2 Live cell: pi × completions upstream (vLLM 0.26, GPU 0 port 30001) — confirm answer + token_ids/logprobs aligned via the matrix harness.
