## Why

The trace-capture layer (`TurnRecord` + `SamplingResult`) grew organically across five upstreams. Its token-id/logprob representation is inconsistent and diverges from the well-typed contract the Tinker SDK established — the same contract AnyHarness's tinker backend already speaks on the wire. Concretely:

- `TurnRecord.output_log_probs` is `list[float]`; `output_ids` is `list[int]`. They are paired only by an ad-hoc `record_turn` length assertion (`trajectory.py:295`), not by a type that makes the pairing structural. A backend that returns mismatched lengths trips the assert at runtime rather than failing to typecheck.
- `SamplingResult.top_logprobs` is `list | None` — untyped; the sglang path puts `list[list[(logprob, token_id)]]` there, the others leave None. The Tinker SDK models this as a typed `TopkPromptLogprobs` (paired `token_ids` + `logprobs` matrices of shape `(length, k)`).
- `stop_reason` / `finish_reason` is a free-form `str` everywhere. Tinker's `StopReason` is `Literal["length", "stop"]`; AnyHarness's five backends emit a sprawl of values (`stop`/`length`/`tool_calls`/`end_turn`/`max_tokens`/`abort`/`stop_sequence`) mapped by ad-hoc dicts per backend.
- Logprob "not computed" is represented inconsistently: None (tinker), absent (sglang), 0.0 padding (`_append_tokens` in `trajectory.py:224` fills logprobs with `0.0` when none). Tinker uses `NaN` for prompt positions where a logprob wasn't computed and `-99999.0` (`MASK_LOGPROB`) as the top-k sentinel.

Strengthening the OOP design means: make the token-id/logprob pairing structural in the types, and adopt the Tinker SDK's representation conventions (typed `StopReason`, sentinel-valued masks, equal-length array pairing) so the capture layer is self-describing and a misaligned backend fails loudly at construction, not silently in training data.

## What Changes

- Introduce typed result types that make token-id ↔ logprob pairing structural: a `SampledSequence`-shaped object carrying `tokens: list[int]` + `logprobs: list[float] | None` where `len(logprobs) == len(tokens)` is an invariant asserted at construction (when logprobs present). `TurnRecord` composes this rather than carrying parallel flat lists.
- Adopt `StopReason = Literal["length", "stop"]` as the canonical finish-reason type on the result; map each backend's wire vocabulary to it at the upstream boundary (the wire-side `_respond_*` keep their format-specific vocab, mapping is at capture time only). `tool_calls`/`end_turn`/`max_tokens` collapse to `stop` or `length` per the Tinker semantics (max→length, everything else→stop).
- Standardize the "no logprob" representation: `None` for an entire missing logprob array (as Tinker does for `logprobs` when not requested); within top-k, use a typed sentinel (`MASK_LOGPROB = -99999.0`, padded token_id `0`) matching Tinker's `TopkPromptLogprobs`, so top-k is always a dense pair of equal-length arrays.
- `SamplingResult.top_logprobs` becomes a typed `TopkLogprobs | None` (paired `token_ids: list[list[int]]` + `logprobs: list[list[float]]`, shape `(num_tokens, k)`, sentinel-padded) rather than raw `list`.

## Impact

- **Affected:** `trajectory.py` (`TurnRecord`, `_SampleBuilder` consumers), `common.py` (`SamplingResult`, the five upstream `sample()` impls, `finish_session`), every place that reads `turn.output_ids`/`output_log_probs`/`finish_reason`.
- **Not affected:** the wire rendering (`_respond_chat`/`_respond_responses`/`_render_response`) keeps producing format-specific `finish_reason` strings for the downstream harness — the typed `StopReason` is capture-layer only, mapped at the upstream→TurnRecord boundary.
- **Migration:** mechanical — the flat `output_ids`/`output_log_probs` lists become attributes of a nested sequence object; `finish_reason` narrows to the literal type with a mapping shim. Existing tests that assert on these fields update.
- **Out of scope:** switching to numpy arrays (Tinker uses np; AnyHarness stays list-of-int/float for zero-dep), changing the trajectory tree linearization, adding new upstreams.
