## Context

`TurnRecord` and `SamplingResult` carry token ids and logprobs as parallel flat lists with only a runtime length assert linking them. The Tinker SDK — whose wire contract AnyHarness's tinker backend already speaks — models these as typed, structurally-paired objects (`SampledSequence`, `TopkPromptLogprobs`) with a narrow `StopReason` literal and sentinel-valued masks. This change brings the capture layer up to that same representation so misalignment fails at construction and the types are self-describing.

## Decisions

### `SampledSequence` — paired tokens+logprobs, list-based (not numpy)

A frozen dataclass mirroring tinker's `SampledSequence` shape but list-based (AnyHarness stays zero-numpy):

```python
@dataclass(frozen=True)
class SampledSequence:
    tokens: list[int]
    logprobs: list[float] | None = None
    stop_reason: StopReason = "stop"
    # __post_init__ asserts len(logprobs)==len(tokens) when logprobs is not None
```

`TurnRecord` composes one of these for the generated turn instead of `output_ids` + `output_log_probs`. `prompt_ids` stays a flat `list[int]` (it has no paired logprobs by default). Compat properties (`output_ids`, `output_log_probs`) delegate to the sequence so existing call sites migrate incrementally — but new code reads `turn.sequence.tokens`.

### `StopReason = Literal["length", "stop"]`

Canonical on `SampledSequence.stop_reason`. Each upstream maps its wire vocab to it at the `sample()` boundary via a shared `_to_stop_reason(wire: str) -> StopReason` helper: `{"length","max_tokens","abort"}` → `"length"`, everything else → `"stop"`. The wire `_respond_*` keep emitting format strings (chat `tool_calls`/`length`/`stop`, etc.) — narrowing is capture-only, as the spec requires.

### `TopkLogprobs` — typed pair, sentinel-padded

```python
MASK_LOGPROB = -99999.0
@dataclass(frozen=True)
class TopkLogprobs:
    token_ids: list[list[int]]   # shape (num_tokens, k)
    logprobs: list[list[float]]  # shape (num_tokens, k)
```

`SamplingResult.top_logprobs` narrows from `list | None` to `TopkLogprobs | None`. The sglang backend pads its `[[ (lp, tid), ... ]]` to width k with `(0, MASK_LOGPROB)`; backends without top-k leave None.

### Sentinel for "no logprob" vs "not requested"

- Entire logprob array absent → `SampledSequence.logprobs = None` (as Tinker: None when not requested).
- Within a present top-k, missing alternatives → `MASK_LOGPROB`/`0` sentinels (dense rectangular), never None (a list cell can't be None in a dense array).
- The trajectory builder's current `0.0` logprob padding (`_append_tokens`) is unrelated (it pads the *trained* sequence's missing-logprob tokens with 0.0 for shape); that stays — it's about linearization, not the upstream result.

### Migration shape

Mechanical: every `turn.output_ids` → `turn.sequence.tokens`; `turn.output_log_probs` → `turn.sequence.logprobs or []`; `turn.finish_reason` → `turn.sequence.stop_reason`. Compat properties on `TurnRecord` paper over the transition so tests migrate in one pass. `_SampleBuilder.append_turn` reads `turn.sequence`.

### What stays

- numpy: not adopted (zero-dep stance holds).
- trajectory tree/linearization: unchanged.
- wire rendering finish_reason strings: unchanged.
- `prompt_ids`: flat list, no paired logprobs (prompt logprobs are a separate future concern).

## Alternatives considered

- **Full tinker types (numpy)**: adds a numpy dependency and converts everywhere; the list-of-int/float already carries the same invariant. Rejected for zero-dep.
- **Subclass per backend of SampledSequence**: no per-backend variation in the captured shape — one type suffices. Rejected (YAGNI).
- **Keep flat lists + stronger assert**: doesn't make the pairing visible to readers/type-checkers; the assert still fires at `record_turn`, far from the cause. Rejected.
