## Why

`ChatUpstream` (UPSTREAM_MODE=chat) routes requests through `litellm.acompletion`. litellm 1.88's request-body validator chokes on the OpenAI-standard `tool_calls[].type == "function"` shape that harnesses (pi, OpenAI SDK) replay on multi-turn: it expects vLLM's `ChatCompletionMessageCustomToolCallParam` (`id` + `custom` fields) and raises `BadRequestError` on the standard shape. This breaks any vLLM-chat agentic turn that replays a tool_call — i.e. every turn after the first tool call. mintcn (its own OpenAI impl) accepted the shape; vLLM 0.26 + litellm 1.88 does not.

This is an upstream litellm/vLLM schema mismatch, not an AnyHarness bug — but AnyHarness must route around it so vLLM chat stays usable for tool-using rollouts.

## What Changes

- When the chat request's messages carry an assistant `tool_calls` (i.e. a replayed tool-calling turn), `ChatUpstream` SHALL route the call through plain HTTP (`_call_chat_upstream_raw_chat`), bypassing litellm's request-body validator entirely. The response is parsed the same way as the litellm path (`chat_response_to_blocks`).
- Tool-free turns (no `tool_calls` in the request) keep the litellm path unchanged.
- The raw HTTP path also returns `return_token_ids` + logprobs from the same response (it already does), so the TITO-safe token-id/logprob pairing established for vLLM is preserved on the tool-call path too.
- No new env var: the bypass is automatic (triggered by the request shape), not opt-in.

## Impact

- **Affected:** `common.py` — `ChatUpstream.sample` (detect tool_calls in `translated`, branch to raw HTTP) + a new `_call_chat_upstream_raw_chat` helper (sibling of the existing `_chat_upstream_token_ids_and_logprobs`).
- **Not affected:** sglang/tinker/messages/responses (only chat uses litellm); the litellm path for tool-free turns.
- **Tests:** add a unit test that a tool-call-bearing request takes the raw-HTTP branch (stub aiohttp) and a tool-free request takes litellm.
