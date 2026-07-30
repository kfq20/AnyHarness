## Context

ChatUpstream routes through `litellm.acompletion`. On vLLM 0.26 + litellm 1.88, a request that replays an assistant `tool_calls` (every post-tool agentic turn) fails litellm's request-body validator: it expects vLLM's non-standard `ChatCompletionMessageCustomToolCallParam` (`id` + `custom`), rejecting the OpenAI-standard `type:"function"` shape with `BadRequestError`. mintcn accepted the shape; vLLM 0.26 via litellm does not.

We already have a raw-HTTP helper (`_chat_upstream_token_ids_and_logprobs`) that POSTs to the chat endpoint with top-level `return_token_ids`+`logprobs` and parses the response. The fix extends that approach to carry the *whole* call (blocks + usage, not just ids) when tool_calls are present.

## Decisions

### Detect tool-call replays in the translated messages

In `ChatUpstream.sample` (or `_call_chat_upstream`), check whether any message in `translated` has `tool_calls`. If yes → raw-HTTP path; if no → litellm path (unchanged). This is the only branch; no env flag.

### Unify the raw-HTTP chat call

Promote the existing `_chat_upstream_token_ids_and_logprobs` into a single `_call_chat_upstream_raw` that returns everything the chat path needs from one response:

```python
async def _call_chat_upstream_raw(self, kwargs, base_url, api_key, session_id) -> dict:
    """POST /chat/completions over plain HTTP (bypasses litellm's request validator).
    Returns {blocks, stop_reason, usage, output_ids, output_log_probs}."""
```

It POSTs `kwargs` messages + tools + `return_token_ids` + `logprobs`, then extracts: blocks via `chat_response_to_blocks`, usage via `response.usage`, token_ids + logprobs from the same response (paired). `ChatUpstream.sample` calls this when tool_calls present and builds the `SamplingResult` (turn + usage + content_blocks) from it. When `SLIME_CHAT_LOGPROBS=1` the turn carries real ids+logprobs; otherwise ids/logprobs empty (block-level).

### `_call_chat_upstream` (litellm path) stays the primary for tool-free turns

No change to its body. The branch lives in `ChatUpstream.sample`: decide litellm-vs-raw by the tool_calls presence in `translated`, call the right one, both produce the same `SamplingResult` shape.

### What stays

- litellm path for tool-free turns (no validator issue there).
- The existing `_chat_upstream_token_ids_and_logprobs` is folded into `_call_chat_upstream_raw` (one raw call, not two).
- sglang/tinker/messages/responses untouched.

## Alternatives

- **Monkeypatch litellm's validator**: fragile, breaks on litellm upgrade. Rejected.
- **Always raw-HTTP for chat**: drops litellm's provider routing/retries for all chat calls, not just the broken ones. Rejected — keep litellm for what works.
