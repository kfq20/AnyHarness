## ADDED Requirements

### Requirement: Completions upstream is a token-level backend

The system SHALL provide an `UPSTREAM_MODE=completions` upstream that targets an OpenAI-compatible `/v1/completions` endpoint (vLLM, sglang's OpenAI-serve layer) and is token-level: `message_level=False`, routing through the same `parse_model_output` → `_build_reply` pipeline as sglang/tinker. It SHALL NOT use the chat-completions messages+tools shape or any of the chat upstream's litellm/vLLM workarounds (`_call_chat_upstream_raw`, `_nest_tools`, `_ensure_tool_call_ids`, `_stringify_tool_call_args`).

#### Scenario: completions routes through the token-level pipeline

- **WHEN** `UPSTREAM_MODE=completions` serves a turn
- **THEN** the `SamplingResult` SHALL have `message_level=False` with `content_blocks=None`, and `_run_turn` SHALL decode `turn.output_ids` and build the reply via `parse_model_output` + `_build_reply` (the sglang/tinker path), not via `_reply_from_content_blocks`

#### Scenario: chat upstream remains available and unchanged

- **WHEN** `UPSTREAM_MODE=chat` serves a turn (tool-free or tool-call replay)
- **THEN** the chat upstream SHALL behave exactly as before this change — `completions` adds a mode, it does not alter or replace `chat`

### Requirement: Completions prompt is passed as a token-id array

The system SHALL pass the prompt to `/v1/completions` as a raw token-id array (`{"prompt": prompt_ids}`), so the upstream server uses the adapter's rendered ids verbatim and never re-tokenizes text. Input-side token-in-token-out SHALL hold: the prompt ids the server samples against are exactly those the adapter rendered (the ones the trajectory tree forks on). `prompt_ids` SHALL come from `_tokenize_messages_server` (server `/tokenize` or `/v1/tokenize`) with a local-render fallback, identical to the sglang upstream.

#### Scenario: prompt sent as token-id array

- **WHEN** `CompletionsUpstream.sample` issues the upstream request
- **THEN** the request body's `prompt` field SHALL be a list of integers (the rendered `prompt_ids`), not a string

#### Scenario: server tokenizes on the adapter's behalf

- **WHEN** the completions server exposes `/tokenize` or `/v1/tokenize` with messages
- **THEN** `prompt_ids` SHALL be obtained from that endpoint (no local tokenizer required), falling back to local render only when the server cannot

### Requirement: Completions output ids and logprobs come from one response

The system SHALL request `return_token_ids: true` and `logprobs` from `/v1/completions` and take both `output_ids` (`choice.token_ids`) and `output_log_probs` (`choice.logprobs.token_logprobs`) from the **same** response, so the structural-pairing invariant (`len(ids) == len(logprobs)`) holds by construction — never from a separate re-issue. A mismatch (server returned differing counts) SHALL fail at `SamplingResult` construction rather than emit a misaligned pair.

#### Scenario: paired ids and logprobs

- **WHEN** the completions response carries `choice.token_ids` of length N and `choice.logprobs.token_logprobs` of length N
- **THEN** the `SamplingResult.turn` SHALL carry both as one sequence with the length invariant holding, sourced from that single response

#### Scenario: no logprobs requested

- **WHEN** `SLIME_COMPLETIONS_LOGPROBS` is not `1`
- **THEN** `output_log_probs` SHALL be None and `output_ids` SHALL still carry `choice.token_ids` — the invariant is vacuous, not violated

### Requirement: Completions top-k logprobs are sentinel-padded

The system SHALL convert `choice.logprobs.top_logprobs` (a `list[dict[str, float]]` of token-string → logprob per position) into the sentinel-padded `TopkLogprobs` pair `(token_ids, logprobs)` of shape `(num_tokens, k)`, mapping each alternative token string to its id via the local tokenizer and padding short positions with `(token_id=0, logprob=MASK_LOGPROB)`. Token strings that fail to map SHALL take the sentinel pair rather than be dropped, keeping the array rectangular. The sampled `output_ids` SHALL remain the server's own `choice.token_ids` — the string→id mapping is used only for the auxiliary top-k alternatives, never for the sampled sequence.

#### Scenario: top-k padded to width k

- **WHEN** `SLIME_COMPLETIONS_TOP_LOGPROBS=N` is set and a position returned fewer than N alternatives
- **THEN** the arrays SHALL be padded to width N with `(0, MASK_LOGPROB)` sentinels

#### Scenario: no top-k requested

- **WHEN** `SLIME_COMPLETIONS_TOP_LOGPROBS` is unset
- **THEN** `SamplingResult.top_logprobs` SHALL be None, matching the not-requested convention

### Requirement: Completions StopReason mapped at the capture boundary

The system SHALL map the completions `choice.finish_reason` to a `StopReason` (`"length"` for `length`/`max_tokens`, `"stop"` for any other terminal reason including `stop`/`tool_calls`) at the upstream→result boundary, conforming to the typed-StopReason requirement.

#### Scenario: length finish maps to length

- **WHEN** the completions response reports `finish_reason: "length"`
- **THEN** the `SamplingResult.turn.finish_reason` SHALL be `"length"`

#### Scenario: stop finish maps to stop

- **WHEN** the completions response reports `finish_reason: "stop"`
- **THEN** the `SamplingResult.turn.finish_reason` SHALL be `"stop"`

### Requirement: Completions mode accepts no local tokenizer

The system SHALL allow `UPSTREAM_MODE=completions` with no `MODEL_PATH`/local tokenizer when the completions server exposes `/tokenize` (server renders the prompt), mirroring the no-local-tokenizer guarantee of the message-level upstreams. When the server cannot tokenize and no tokenizer is configured, prompt_ids SHALL be empty and the turn SHALL still complete (degraded, matching sglang without a tokenizer) rather than raise at startup.

#### Scenario: no tokenizer, server tokenizes

- **WHEN** `UPSTREAM_MODE=completions` and no `MODEL_PATH` is set but the server exposes `/tokenize`
- **THEN** the adapter SHALL serve a full turn, obtaining `prompt_ids` from the server

#### Scenario: no tokenizer, no server tokenize

- **WHEN** `UPSTREAM_MODE=completions`, no `MODEL_PATH`, and the server exposes no tokenize endpoint
- **THEN** `prompt_ids` SHALL be empty and the turn SHALL complete without raising (the request uses an empty prompt, same degradation as sglang without a tokenizer)
