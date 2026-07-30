## ADDED Requirements

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
