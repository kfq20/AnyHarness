# slime-sft-trace — AnyHarness

Build SFT training trajectories from a coding agent's `/v1/messages` traffic.

**AnyHarness** is the architectural name for the layer
between a harness (Claude Code) and the sampling endpoint / API server. It
manages the harness↔upstream contract: the adapter exposes `/v1/messages`
downstream to Claude Code, forwards upstream to a pluggable backend
(sglang with per-token logprobs, or any messages-API), and captures each turn
into a trajectory tree that an offline aggregator dumps as SFT data. The
"harness in the loop" is that the adapter sits *between* the harness and the
model — it is not fire-and-forget: every `/v1/messages` call is intercepted,
matched into a per-session tree (tolerant of Claude Code's tool-result
trimming), and linearized into training samples at `finish_session`.

This standalone project reuses Slime's agent code (trajectory manager, types,
and the Anthropic Messages adapter) and adds a sandbox-pluggable Claude Code
harness + CLI that drives the agent through the adapter and dumps loss-masked
SFT samples.

## Layout

```
src/slime_sft_trace/
  trajectory.py        # TrajectoryManager, TurnRecord (vendored)
  types.py             # Sample (vendored)
  parsing.py            # parse_model_output (vendored)
  adapters/             # AnthropicAdapter: /v1/messages, sglang|messages upstream (vendored)
  harness/
    claude_code.py      # ClaudeCodeHarness + Sandbox/LocalSandbox/E2BSandbox
  cli.py                # main(): adapter-in-thread -> harness -> finish_session -> dump
  dump.py               # dump_samples -> trajectories.jsonl + .md (+ .pt)
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
export MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct
export SLIME_SGLANG_URL=http://localhost:30000
export PROMPT="Fix the failing test in tests/test_foo.py"
export OUTPUT_DIR=./out
slime-sft-trace
```

messages mode (forward to an existing `/v1/messages` endpoint):

```bash
export UPSTREAM_MODE=messages
export SLIME_MESSAGES_UPSTREAM_URL=https://api.example.com
export PROMPT="..."
slime-sft-trace
```

### Environment

| Var | Default | Meaning |
|-----|---------|---------|
| `UPSTREAM_MODE` | `sglang` | `sglang` (logprob capture) or `messages` (forward) |
| `MODEL_PATH` | — | HF tokenizer path (sglang mode only, required there) |
| `SLIME_SGLANG_URL` | — | sglang base URL (sglang mode) |
| `SLIME_MESSAGES_UPSTREAM_URL` | — | upstream `/v1/messages` URL (messages mode) |
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

The smoke test exercises the vendored trajectory layer end to end (no model,
sglang, or claude binary): it builds a `TrajectoryManager`, feeds hand-built
turns, and asserts loss masks (prompt=0, response=1).
