# sampling-upstream Specification

## Purpose
TBD - created by archiving change tokenizer-free-upstream. Update Purpose after archive.
## Requirements
### Requirement: Upstream truthfulness of token usage

The system SHALL report token usage (`prompt_tokens` / `completion_tokens`) in the downstream wire response from the upstream server's own usage when the upstream provides it, rather than from a local tokenizer render or a zero fallback.

#### Scenario: message-level upstream returns usage

- **WHEN** a message-level upstream (messages, chat, or responses) returns a response carrying a `usage` field with input/output token counts
- **THEN** the adapter's downstream `usage.prompt_tokens` / `usage.completion_tokens` SHALL equal those upstream counts, not the local `len(prompt_ids)` / `len(turn.output_ids)`

#### Scenario: upstream omits usage

- **WHEN** an upstream response carries no usable `usage`
- **THEN** the adapter SHALL fall back to the existing local counts (`len(prompt_ids)` for input, `len(turn.output_ids)` for output) so token-level backends (sglang/tinker) are unaffected

#### Scenario: token-level upstream unchanged

- **WHEN** the upstream is token-level (sglang or tinker) whose `SamplingResult.usage` is None
- **THEN** `in_tok`/`out_tok` SHALL remain `len(prompt_ids)` / `len(turn.output_ids)` exactly as before this change

### Requirement: Message-level upstreams render on the server

The system SHALL NOT require a local tokenizer for message-level upstreams (messages, chat, responses): the upstream server renders the chat template, so the adapter forwards hub-format messages directly and computes no local `prompt_ids` for these backends.

#### Scenario: no MODEL_PATH set for chat upstream

- **WHEN** `UPSTREAM_MODE=chat` (or `responses`/`messages`) and no `MODEL_PATH`/tokenizer is configured
- **THEN** the adapter SHALL still serve a full agent turn without error, because no local render is attempted for that backend

#### Scenario: local render retained for token-level upstreams

- **WHEN** the upstream is sglang or tinker
- **THEN** local rendering (tokenizer or server `/v1/tokenize` per Phase 2) SHALL remain the source of `prompt_ids`, unchanged

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

### Requirement: Chat upstream bypasses litellm for tool-call replays

When the chat upstream request replays an assistant message carrying `tool_calls`, the system SHALL issue the upstream call over plain HTTP rather than through `litellm.acompletion`, because litellm's request-body validator rejects the OpenAI-standard `tool_calls[].type == "function"` shape on vLLM 0.26 (expecting a non-standard `ChatCompletionMessageCustomToolCallParam`). Tool-free turns (no `tool_calls` in the request) SHALL continue through litellm unchanged.

#### Scenario: tool-call replay takes the raw-HTTP path

- **WHEN** the chat request's messages contain an assistant message with `tool_calls`
- **THEN** the upstream call SHALL be issued via plain HTTP (no litellm request validation), and the response parsed by the same `chat_response_to_blocks` the litellm path uses

#### Scenario: tool-free turn stays on litellm

- **WHEN** the chat request's messages contain no assistant `tool_calls`
- **THEN** the upstream call SHALL go through `litellm.acompletion` as before (litellm adds no tool-call fields for plain turns)

#### Scenario: raw-HTTP tool-call path returns paired ids+logprobs

- **WHEN** `SLIME_CHAT_LOGPROBS=1` and the request carries tool_calls
- **THEN** the raw-HTTP call SHALL request `return_token_ids` + `logprobs` and the `SamplingResult.turn` SHALL carry token ids and logprobs from the SAME response (length-aligned), never from a separate call

