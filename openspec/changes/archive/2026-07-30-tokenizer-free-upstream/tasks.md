# Tasks

## 1. Add `usage` to SamplingResult and extract it per backend

- [x] 1.1 Add `usage: dict | None = None` field to `SamplingResult` in `common.py` (keys: `input_tokens`, `output_tokens`).
- [x] 1.2 `MessagesUpstream.sample`: populate `usage` from the `usage_out` value already returned by `_call_messages_upstream` (it returns `(turn, blocks)`; extend the tuple or read the inbound usage the method already computes). Inspect `_call_messages_upstream`'s `usage_out` variable.
- [x] 1.3 `ChatUpstream.sample`: populate `usage` from `response.usage` (litellm ModelResponse) — `prompt_tokens`→`input_tokens`, `completion_tokens`→`output_tokens`. Guard for None.
- [x] 1.4 `ResponsesUpstream.sample`: populate `usage` from the parsed response's `usage` (`input_tokens`/`output_tokens`). The litellm path and `_call_responses_upstream_raw`/`_AttrView` both carry it — extract once where blocks are built.
- [x] 1.5 `SglangUpstream` / `TinkerUpstream`: leave `usage=None` (token-level).

## 2. Wire truthful usage into _run_turn

- [x] 2.1 In `_run_turn`, replace `in_tok, out_tok = len(prompt_ids), len(turn.output_ids)` with: `in_tok = result.usage.get("input_tokens") if result.usage else len(prompt_ids)` and `out_tok` likewise from `output_tokens` / `len(turn.output_ids)`.
- [x] 2.2 Verify no test asserts on a specific `usage` value for message-level backends; update any that do (e.g. e2e tests that check `prompt_tokens`).

## 3. Tests

- [x] 3.1 Add a test: message-level upstream (chat, with a fake litellm response carrying `usage`) → `SamplingResult.usage` populated and `_run_turn` reports it as `prompt_tokens`/`completion_tokens`.
- [x] 3.2 Add a test: token-level upstream (sglang/tinker) → `usage` stays None, `in_tok`/`out_tok` from ids (no regression).
- [x] 3.3 Add a test: upstream returns no usage → falls back to `len(prompt_ids)`/`len(turn.output_ids)`.

## 4. Verify

- [x] 4.1 `pytest tests/ -q` — all green.
- [x] 4.2 Optional live: point at vLLM 0.26 chat upstream, confirm downstream `usage.completion_tokens` is nonzero (truthful) instead of 0.
