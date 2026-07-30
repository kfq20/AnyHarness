# AnyHarness

Build SFT training trajectories from a coding agent's `/v1/messages` traffic.

**AnyHarness** is the architectural name for the layer
between a harness (Claude Code) and the sampling endpoint / API server. It
manages the harness↔upstream contract: the adapter exposes `/v1/messages`
downstream to Claude Code, forwards upstream to a pluggable backend
(sglang, vLLM `/v1/completions`, or any messages/chat/responses API), and
captures each turn into a trajectory tree that an offline aggregator dumps as
SFT data. The "harness in the loop" is that the adapter sits *between* the
harness and the model — it is not fire-and-forget: every `/v1/messages` call is
intercepted, matched into a per-session tree (tolerant of Claude Code's
tool-result trimming), and linearized into training samples at `finish_session`.

A standalone adapter + sandbox-pluggable Claude Code harness + CLI that drives
the agent through the adapter and dumps loss-masked SFT samples.

## Layout

```
src/anyharness/
  trajectory.py        # TrajectoryManager, TurnRecord, SampledSequence, TopkLogprobs
  types.py             # Sample
  parsing.py            # parse_model_output (reasoning + tool-call parsers)
  adapters/             # AnthropicAdapter: /v1/messages downstream + 5 upstreams
                         #   (sglang /generate, vLLM /v1/completions, vLLM chat,
                         #    Responses, Tinker asample)
  harness/
    claude_code.py      # ClaudeCodeHarness + Sandbox/LocalSandbox/E2BSandbox
  cli.py                # main(): adapter-in-thread -> harness -> finish_session -> dump
  dump.py               # dump_samples -> trajectories.jsonl + .md (+ .pt)
  sharegpt_dump.py     # fold a tree into ShareGPT (reasoning + thinking_signature)
```

## How it works

1. The CLI builds an `AnthropicAdapter` and serves its aiohttp app on
   `ADAPTER_PORT` (a background thread via `aiohttp.web.AppRunner`).
2. `ClaudeCodeHarness` launches `claude -p <PROMPT> --output-format stream-json`
   with `ANTHROPIC_BASE_URL=<adapter>` and `ANTHROPIC_AUTH_TOKEN=<session_id>`.
   Each `/v1/messages` call is rendered to tokens, sent upstream (sglang
   `/generate` with logprobs, or forwarded to another `/v1/messages` API), and
   recorded into a per-session trajectory tree.
3. `finish_session` linearizes the tree into `Sample`s: `tokens` = full
   prompt+response sequence, `loss_mask` marks prompt=0 / response=1.
4. `dump_samples` writes `trajectories.jsonl`, `trajectories.md`, and (if
   torch is installed) `trajectories.pt`.

## Install

```bash
pip install -e ".[sglang,test]"   # sglang mode + tests
# or messages mode (no tokenizer needed):
pip install -e ".[test]"
```

The `claude` CLI must already be on `PATH` (Slime installs it inside E2B; this
harness skips that step). Override with `CLAUDE_BIN=/path/to/claude`.

## Usage

sglang mode (capture logprobs from a served model):

```bash
export UPSTREAM_MODE=sglang
export MODEL_PATH=Qwen/Qwen3.5-9B
export SLIME_SGLANG_URL=http://localhost:30000
export PROMPT="Fix the failing test in tests/test_foo.py"
export OUTPUT_DIR=./out
anyharness
```

completions mode (vLLM `/v1/completions`, native token-in token-out — the
cleanest path for vLLM, no chat-shape workarounds):

```bash
export UPSTREAM_MODE=completions
export SLIME_COMPLETIONS_BASE_URL=http://localhost:30001   # vLLM OpenAI server
export SLIME_COMPLETIONS_MODEL=qwen3-9b
export SLIME_COMPLETIONS_LOGPROBS=1                        # capture per-token logprobs
export SLIME_COMPLETIONS_TOP_LOGPROBS=5                    # top-k alternatives
export MODEL_PATH=Qwen/Qwen3.5-9B                          # tokenizer (optional if
                                                            #  the server exposes /tokenize)
export PROMPT="..."
anyharness
```

messages mode (forward to an existing `/v1/messages` endpoint):

```bash
export UPSTREAM_MODE=messages
export SLIME_MESSAGES_UPSTREAM_URL=https://api.example.com
export PROMPT="..."
anyharness
```

tinker mode (token-in token-out against a Tinker/Mint gateway):

```bash
export UPSTREAM_MODE=tinker
export TINKER_BASE_URL=http://your-mint-gateway:28000
export TINKER_API_KEY=...                       # sent as Bearer
export TINKER_BASE_MODEL=Qwen/Qwen3.6-35B-A3B   # or TINKER_MODEL_ID=sess-xxx_0
export MODEL_PATH=/path/to/matching/tokenizer
export PROMPT="..."
anyharness
```

### Token-level data and TITO

Token ids must come from whoever sampled them. Re-encoding assistant text — or
looking a logprob's token *string* back up in a vocab — breaks the
[token-in-token-out invariant](https://huggingface.co/blog/huggingface/tito):
tokenization is not injective, so the recovered ids can differ from what the
policy actually produced, and training then optimizes tokens the model never
emitted. Where each mode stands:

| Mode | Token ids from | Logprobs |
|------|----------------|----------|
| `tinker` | ids in, ids out (`/api/v1/asample`) | yes, paired by the server |
| `completions` | vLLM `choice.token_ids` (`/v1/completions`) | yes — ids + logprobs from one response |
| `sglang` | native `/generate` (`meta_info`) | yes |
| `chat` | upstream `return_token_ids` (vLLM >= 0.10.2) | only if the upstream supplies ids |
| `responses` | not available | no — the Responses API exposes no ids |
| `messages` | n/a (message-level SFT) | no |

When ids aren't available the adapter **drops** the logprobs and logs why, rather
than reconstruct them. Message-level SFT is better than token-level data that is
subtly wrong.

In `tinker` and `sglang` mode the chat template is rendered locally, so
`MODEL_PATH` must point at the *served* model's tokenizer. A mismatched
tokenizer produces a prompt the server misreads — usually visible as the model
echoing your prompt back — and the adapter warns when it detects one.

### Multi-turn token drift

Each turn's prompt is rendered from the full message list, so it will not always
reproduce the ids the policy sampled last turn. For a thinking model it *cannot*:
Qwen3's template emits `<think>...</think>` while an assistant message is the
final message and strips it once anything follows, so the re-render diverges
across the whole response, every turn.

The trajectory builder classifies that divergence:

- **CLEAN** — the prompt extends the held tokens; append the tail.
- **REALIGN** — divergence covers only untrained tokens; heal in place and stay
  contiguous.
- **FORK** — divergence would overwrite *trained* tokens; close the sample and
  open a new one instead.

Trained tokens are never overwritten to preserve contiguity. Forking costs only
contiguity — every sampled token still trains, spread across several `Sample`s
that each re-emit the shared prefix as `loss_mask=0` context. That re-emission is
not free: a 16-turn thinking rollout yields 16 samples and ~10x the tokens of one
contiguous sample. The alternative — carrying sampled ids forward verbatim and
appending only the new messages — keeps one sample but changes what the model
conditions on (its own `<think>` blocks stay in context), and only works in modes
where we send token ids rather than messages. It is not implemented.

### Environment

| Var | Default | Meaning |
|-----|---------|---------|
| `UPSTREAM_MODE` | `sglang` | `sglang`, `completions`, `tinker`, `chat`, `responses`, or `messages` |
| `MODEL_PATH` | — | HF tokenizer path (required in sglang/tinker; optional elsewhere) |
| `SLIME_SGLANG_URL` | — | sglang base URL (sglang mode) |
| `SLIME_COMPLETIONS_BASE_URL` | — | vLLM OpenAI server base URL (completions mode) |
| `SLIME_COMPLETIONS_MODEL` | — | served model name (completions mode) |
| `SLIME_COMPLETIONS_API_KEY` | — | Bearer token for the completions upstream |
| `SLIME_COMPLETIONS_LOGPROBS` | `0` | `1` captures per-token logprobs |
| `SLIME_COMPLETIONS_TOP_LOGPROBS` | `0` | top-k alternatives per position |
| `SLIME_CHAT_BASE_URL` | — | OpenAI-compatible base URL (chat mode) |
| `SLIME_CHAT_MODEL` | — | served model name (chat mode) |
| `SLIME_CHAT_LOGPROBS` | `0` | `1` captures logprobs (needs vLLM `return_token_ids`) |
| `SLIME_RESPONSES_BASE_URL` | — | Responses API base URL (responses mode) |
| `SLIME_MESSAGES_UPSTREAM_URL` | — | upstream `/v1/messages` URL (messages mode) |
| `TINKER_BASE_URL` | — | Tinker/Mint gateway base URL (tinker mode) |
| `TINKER_API_KEY` | — | Bearer token for the gateway |
| `TINKER_BASE_MODEL` | — | served base model, e.g. `Qwen/Qwen3.6-35B-A3B` |
| `TINKER_MODEL_ID` | — | a specific training step instead of the base model |
| `TINKER_LOGPROBS` | `1` | `0` disables logprob capture (it also gates prompt logprobs) |
| `TINKER_FUTURE_TIMEOUT` | `900` | seconds to await `/retrieve_future` |
| `ADAPTER_PORT` | `18080` | port the adapter listens on |
| `CLAUDE_MODEL` | `slime-actor` | model name advertised to the CLI |
| `PROMPT` | — | the task prompt (required) |
| `OUTPUT_DIR` | `./trajectories` | where samples are written |
| `SANDBOX` | `local` | `local` (LocalSandbox) or `e2b` (E2BSandbox) |
| `TIME_BUDGET_SEC` | `600` | harness time budget |
| `CLAUDE_BIN` | `claude` | path to the claude CLI |
| `TOOL_PARSER` / `REASONING_PARSER` | — | sglang parser names |
| `FORK_THRESHOLD_TOKENS` | — | trajectory fork threshold |

## Tests

```bash
pytest tests/test_harness_smoke.py
```

The smoke test exercises the trajectory layer end to end (no model, sglang, or
claude binary): it builds a `TrajectoryManager`, feeds hand-built turns, and
asserts loss masks (prompt=0, response=1).
