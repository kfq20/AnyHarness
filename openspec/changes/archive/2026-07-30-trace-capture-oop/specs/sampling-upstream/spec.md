## ADDED Requirements

### Requirement: Structural token-id ↔ logprob pairing

The system SHALL represent sampled tokens and their logprobs as a single sequence object whose `tokens` (list[int]) and `logprobs` (list[float] | None) are structurally paired: when `logprobs` is not None, `len(logprobs) == len(tokens)` SHALL be an invariant enforced at construction. Backends returning mismatched lengths SHALL fail at `SamplingResult` construction rather than tripping a downstream assertion in `record_turn`.

#### Scenario: aligned ids and logprobs

- **WHEN** a token-level upstream (sglang or tinker) returns equal-length `output_ids` and `output_log_probs`
- **THEN** the `SamplingResult.turn` SHALL carry them as the `tokens`/`logprobs` of one sequence object with the length-invariant holding

#### Scenario: mismatched lengths fail at construction

- **WHEN** a backend returns `output_ids` of length N but `output_log_probs` of length M != N (M > 0)
- **THEN** constructing the result SHALL raise (not silently truncate or pad), surfacing the bug at the upstream boundary

#### Scenario: no logprobs requested

- **WHEN** a backend sampled without logprobs (e.g. `TINKER_LOGPROBS=0`)
- **THEN** the sequence's `logprobs` SHALL be None, and `tokens` SHALL still carry the sampled ids — the invariant is vacuous, not violated

### Requirement: Typed StopReason at the capture boundary

The system SHALL represent stop reason as `Literal["length", "stop"]` (`StopReason`) on the capture-layer result (`SamplingResult.turn`), mapping each upstream's wire vocabulary to exactly one of these at the upstream→result boundary. The wire-rendering layer (`_respond_*`) MAY continue to emit format-specific finish-reason strings for the downstream harness; the narrowing happens at capture time only.

#### Scenario: max_tokens maps to length

- **WHEN** an upstream reports `max_tokens` / `length` / `abort` (sglang finish "length", vLLM "length", tinker "length"/"abort")
- **THEN** the capture-layer `StopReason` SHALL be `"length"`

#### Scenario: terminal turns map to stop

- **WHEN** an upstream reports `stop` / `end_turn` / `tool_calls` / `stop_sequence` (any non-length terminal reason)
- **THEN** the capture-layer `StopReason` SHALL be `"stop"` — a tool-calling turn is a normal stop from the sampling perspective

### Requirement: Sentinel-valued logprob masks for top-k

The system SHALL represent per-position top-k alternative logprobs as a typed pair of equal-length dense arrays (`token_ids: list[list[int]]`, `logprobs: list[list[float]]`, shape `(num_tokens, k)`), padded with sentinel values `token_id=0` and `logprob=MASK_LOGPROB` (`-99999.0`) at positions where the upstream returned fewer than k alternatives. This matches the Tinker SDK `TopkPromptLogprobs` convention.

#### Scenario: dense top-k with fewer alternatives than k

- **WHEN** a position's upstream returned only 2 alternatives but k=5 was requested
- **THEN** the arrays SHALL be padded to width 5 with `(token_id=0, logprob=-99999.0)` sentinels, keeping the pair rectangular

#### Scenario: no top-k requested

- **WHEN** no top-k was requested (sglang `SLIME_TOP_LOGPROBS` unset, or other backends)
- **THEN** `SamplingResult.top_logprobs` SHALL be None (not an empty list), matching the Tinker `None`-when-not-requested convention
