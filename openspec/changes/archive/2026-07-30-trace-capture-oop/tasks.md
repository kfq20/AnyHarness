# Tasks

## 1. Typed result primitives

- [x] 1.1 Add `StopReason = Literal["length", "stop"]` and `_to_stop_reason(wire: str) -> StopReason` (maps length/max_tokens/abort→"length", else "stop") in `trajectory.py` (or a new `types`-adjacent spot).
- [x] 1.2 Add `MASK_LOGPROB = -99999.0` and `TopkLogprobs` frozen dataclass (`token_ids: list[list[int]]`, `logprobs: list[list[float]]`).
- [x] 1.3 Add `SampledSequence` frozen dataclass (`tokens`, `logprobs|None`, `stop_reason`) with `__post_init__` asserting `len(logprobs)==len(tokens)` when logprobs not None.

## 2. TurnRecord composes SampledSequence

- [x] 2.1 `TurnRecord` gains `sequence: SampledSequence` (prompt_ids stays flat). Add compat properties `output_ids`/`output_log_probs`/`finish_reason` delegating to `sequence` (and `prompt_ids`).
- [x] 2.2 Keep a backward-compat constructor path: accept the old flat `output_ids`/`output_log_probs`/`finish_reason` kwargs and build the sequence internally (so existing call sites in the five upstreams migrate without churn in this step).

## 3. SamplingResult narrows top_logprobs

- [x] 3.1 `SamplingResult.top_logprobs` type changes from `list | None` to `TopkLogprobs | None`.
- [x] 3.2 `SglangUpstream.sample`: convert sglang's `output_top_logprobs` (`list[list[(logprob,token_id)]]`) to a sentinel-padded `TopkLogprobs` (pad to width k with `(0, MASK_LOGPROB)`). None when absent.

## 4. Upstream boundaries map StopReason

- [x] 4.1 sglang/tinker `sample()`: build `SampledSequence` with `stop_reason=_to_stop_reason(wire_finish)` instead of raw string.
- [x] 4.2 messages/chat/responses `sample()`: map their finish (already normalized to sglang vocab) through `_to_stop_reason` when building the sequence.

## 5. Consumers

- [x] 5.1 `_SampleBuilder.append_turn` (`trajectory.py`): read `turn.sequence.tokens`/`.logprobs`; keep the existing length assert as a redundant guard (now also enforced at construction).
- [x] 5.2 `_run_turn` / `finish_session` in `common.py`: `turn.output_ids`/`output_log_probs` calls go through compat properties (no change needed if properties delegate) — verify.

## 6. Tests

- [x] 6.1 `SampledSequence.__post_init__` raises on mismatched lengths; accepts None logprobs; accepts aligned.
- [x] 6.2 `TopkLogprobs` sglang conversion pads to k with sentinels; None when absent.
- [x] 6.3 `_to_stop_reason` maps length/max_tokens/abort→length, stop/end_turn/tool_calls/stop_sequence→stop.
- [x] 6.4 Existing suite green after migration (TurnRecord compat props keep e2e/tinker tests passing).

## 7. Verify

- [x] 7.1 `pytest tests/ -q` all green.
- [x] 7.2 Optional live: tinker upstream still returns aligned tokens/logprobs through the new `SampledSequence`.
