## MODIFIED Requirements

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

## ADDED Requirements

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
