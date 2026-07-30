## Context

Phase 1 unified the five upstreams behind `SamplingUpstream` / `SamplingResult`. Phase 2 made sglang tokenizer-free via server `/v1/tokenize`. But `_run_turn` still locally renders `prompt_ids` for every turn and derives `in_tok`/`out_tok` from it — wrong for the three message-level backends, where the server already tokenized and reported usage. This change makes usage truthful and drops the dead local render for message-level backends.

## Decisions

### `SamplingResult.usage` — a small optional field, not a new type

Add `usage: dict | None = None` to `SamplingResult` (keys: `input_tokens`, `output_tokens`). Use a plain dict, matching what the upstreams already parse. `None` = "no upstream usage; caller falls back to local counts". This avoids a new dataclass for a 2-key structure.

### Per-backend usage extraction (verbatim from existing code paths)

- **MessagesUpstream** (`_call_messages_upstream`): already parses `usage_out` from the upstream SSE/JSON response (`_consume_messages_sse` / `_parse_messages_json` return it). Populate `usage = {"input_tokens":..., "output_tokens": usage_out}`.
- **ChatUpstream** (`_call_chat_upstream`): litellm `response.usage` exposes `prompt_tokens`/`completion_tokens`. Populate from there.
- **ResponsesUpstream** (`_call_responses_upstream`): the raw-HTTP fallback (`_AttrView`) and litellm path both carry `usage.input_tokens`/`output_tokens`. Populate from the parsed response.
- **SglangUpstream / TinkerUpstream**: leave `usage = None` (token-level; counts come from ids).

### `_run_turn` consumption

```python
result = await self._call_upstream(...)
turn = result.turn
...
in_tok = result.usage.get("input_tokens") if result.usage else len(prompt_ids)
out_tok = result.usage.get("output_tokens") if result.usage else len(turn.output_ids)
```

The local `prompt_ids` render (lines ~1381-1384) stays for token-level backends (sglang/tinker need it). It becomes dead work only for message-level backends — but it's cheap (one `apply_chat_template` or `[]` when no tokenizer) and removing it would change the `prompt_ids` argument contract to `_call_upstream`. Keep it; the win is truthful usage, not avoiding one render.

### Fallback semantics

`result.usage is None` → existing `len(prompt_ids)`/`len(turn.output_ids)`. This is the exact current behavior for sglang/tinker, so they're untouched.

## Alternatives considered

- **Drop `prompt_ids` from `_call_upstream` entirely**: bigger blast radius (the two module-level functions take it positionally). Deferred to a Phase 4.
- **Server-side detokenize for sglang output**: sglang `/generate` already returns `output_ids` natively (no detokenize needed for the ids; only `_run_turn`'s `raw_output = tok.decode(...)` for parse_model_output would need server detokenize, but sglang block-level reply path doesn't use `raw_output`). Out of scope.
