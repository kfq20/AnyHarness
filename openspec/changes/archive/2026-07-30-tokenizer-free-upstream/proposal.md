## Why

The message-level upstreams (messages / chat / responses) are *server-backed*: the upstream server renders the chat template and returns real token usage. Yet `_run_turn` (`common.py`) still locally renders `prompt_ids` for every turn and computes `in_tok = len(prompt_ids)` / `out_tok = len(turn.output_ids)` — both wrong for these backends:

- `in_tok` comes from the **local** tokenizer render, which can diverge from what the server actually tokenized (and is `0` when no tokenizer is configured — silent, not flagged).
- `out_tok` is `len(turn.output_ids)`, which is `0` for message-level backends (they carry content blocks, not output ids). So `usage.completion_tokens` is reported as `0` to the harness even though the model generated tokens.

This means the trajectory's reported token counts are wrong for three of the five upstreams, and the local render in `_run_turn` is dead work that also blocks the "cloud run without a tokenizer" goal for these backends.

The sglang backend already got tokenizer-free render in Phase 2 (`_tokenize_messages_server`); this change extends the same freedom to chat / responses / messages and makes usage counts truthful.

## What Changes

- `SamplingResult` gains an optional `usage` field (input/output token counts) populated from the upstream's own response when available.
- `_run_turn` no longer locally renders `prompt_ids` for message-level upstreams; it uses `result.usage` for `in_tok`/`out_tok` and falls back to `len(prompt_ids)`/`len(turn.output_ids)` only for token-level upstreams (sglang/tinker).
- `MessagesUpstream` / `ChatUpstream` / `ResponsesUpstream` populate `usage` from the upstream response (Anthropic `usage`, OpenAI `usage`, Responses `usage`). When the upstream omits usage, the counts fall back to the existing local-render values (no regression).
- The `prompt_ids` parameter of `_call_upstream` becomes unused by message-level backends but is kept (passed through) so token-level backends that still need it aren't disturbed. Phase 4 (out of scope here) may drop it entirely.

## Impact

- **Affected files:** `src/anyharness/adapters/common.py` (`SamplingResult`, `_run_turn`, the three message-level upstream classes), `src/anyharness/adapters/anthropic.py` (nothing — `_respond_*` already take `in_tok`/`out_tok`).
- **No behavior change for sglang/tinker** (token-level: `usage` stays None, `in_tok`/`out_tok` from ids as today).
- **Behavior change for messages/chat/responses:** `usage.prompt_tokens`/`completion_tokens` become truthful (from upstream) instead of locally-rendered/zero. Tests that asserted on usage values for these modes must be updated.
- **Out of scope:** removing the `prompt_ids` parameter entirely (Phase 4), server-side detokenize for sglang output decoding.
