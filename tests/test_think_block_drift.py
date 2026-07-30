"""Regression: thinking-model token drift silently discards the training signal.

These tests use a REAL tokenizer, unlike test_harness_smoke.py which hand-builds
``p2 = p1 + o1 + tail`` and is therefore CLEAN by construction. That is why this
never surfaced: REALIGN has no coverage against a live chat template.

The mechanism (measured, not theorised):

Qwen3's template renders an assistant message *differently* depending on whether
it is the final message. As the last message it carries the reasoning block::

    <|im_start|>assistant\\n<think>\\nlet me think\\n</think>\\n\\nok<|im_end|>

Once anything follows it, the template strips that block from history::

    <|im_start|>assistant\\nok<|im_end|>

Since the adapter re-renders the whole message list every turn
(``common.py`` ``_render_token_ids(translated, ...)``), turn N+1's prompt cannot
reproduce the tokens the policy actually sampled at turn N -- the reasoning block
is gone by construction. That is not cosmetic drift; it is a systematic
divergence exactly the size of each response's think block.

``_SampleBuilder.classify_token_drift`` sees the divergence land inside the most
recent response span and returns REALIGN, whose healing step overwrites that span
from the prompt with ``loss_mask=0``. Result: every assistant turn but the last is
zeroed. In an agentic tool loop the zeroed turns are the tool calls -- the
decisions RL is supposed to reinforce.

Marked xfail: these encode the intended behaviour (sampled tokens keep their
signal) and currently fail. They should flip to passing when the builder stops
re-rendering history and instead extends the held token sequence per turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TOKENIZER = "Qwen/Qwen3-8B"


def _tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:  # pragma: no cover
        pytest.skip("transformers not installed")
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - offline / no HF cache
        pytest.skip(f"tokenizer {TOKENIZER} unavailable: {type(exc).__name__}")


def test_template_strips_think_block_from_history():
    """The root cause, asserted directly: history rendering is not stable."""
    tok = _tokenizer()
    assistant = {"role": "assistant", "content": "ok", "reasoning_content": "let me think"}

    as_last = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, assistant],
        tokenize=False, add_generation_prompt=False,
    )
    in_history = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, assistant, {"role": "user", "content": "next"}],
        tokenize=False, add_generation_prompt=True,
    )

    assert "<think>" in as_last, "reasoning block present while the message is last"
    assert "<think>" not in in_history, (
        "template drops the reasoning block once the message is history — so a "
        "re-render can never reproduce what the policy sampled"
    )


def _run_tool_loop(n_turns: int, *, think: bool, fork_threshold: int = 1024):
    """Drive TrajectoryManager over a tool loop; return (sampled, trained)."""
    from anyharness import Sample, TrajectoryManager, TurnRecord
    from anyharness.adapters.common import _render_token_ids

    tok = _tokenizer()
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    tools = [{
        "type": "function",
        "function": {"name": "read_file", "description": "read",
                     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    }]

    mgr = TrajectoryManager(fork_threshold_tokens=fork_threshold)
    sid = "drift-sid"
    convo: list[dict] = [{"role": "user", "content": "Audit the repo."}]
    sampled = 0

    for i in range(n_turns):
        prompt_ids = _render_token_ids(convo, tok, tools=tools)
        think_block = "<think>\n\nchecking f%d\n</think>\n\n" % i if think else ""
        call = '<tool_call>\n{"name": "read_file", "arguments": {"path":"f%d.py"}}\n</tool_call><|im_end|>\n' % i
        output_ids = encode(think_block + call)
        mgr.record_turn(
            sid,
            turn=TurnRecord(prompt_ids=prompt_ids, output_ids=output_ids,
                            finish_reason="tool_calls", output_log_probs=[-0.1] * len(output_ids)),
            prompt_messages=list(convo),
            response_message={"role": "assistant", "content": "",
                              "tool_calls": [{"id": f"c{i}", "type": "function",
                                              "function": {"name": "read_file",
                                                           "arguments": '{"path":"f%d.py"}' % i}}]},
        )
        sampled += len(output_ids)
        convo = convo + [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": f"c{i}", "type": "function",
                             "function": {"name": "read_file", "arguments": '{"path":"f%d.py"}' % i}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x = 1\n"},
        ]

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0), reward=1.0)
    return sampled, sum(sum(s.loss_mask) for s in samples)


def test_no_think_block_keeps_all_signal():
    """Control: without a reasoning block the re-render is stable and nothing is lost."""
    sampled, trained = _run_tool_loop(4, think=False)
    assert sampled > 0
    assert trained == sampled, "a stable template must retain every sampled token"


def test_think_block_loop_keeps_all_signal():
    """Every sampled token must be trained, drift or no drift.

    Before the fix this was 8/34 -- REALIGN zeroed all but the final turn.
    """
    sampled, trained = _run_tool_loop(4, think=True)
    assert sampled > 0
    assert trained == sampled, f"only {trained}/{sampled} sampled tokens survive drift healing"


def test_think_block_signal_does_not_decay_with_depth():
    """The trained fraction must not fall as the loop gets longer.

    Before the fix only the final turn survived, so the ratio decayed as 1/n:
    2 turns -> 23.5%, 3 -> 13.3%, 5 -> 7.1%, 9 -> 3.7%.
    """
    ratios = []
    for n in (2, 4, 8, 16):
        sampled, trained = _run_tool_loop(n, think=True)
        ratios.append(trained / sampled)
    assert min(ratios) == 1.0, f"training signal decays with depth: {ratios}"


def test_drifted_turns_fork_rather_than_zeroing():
    """The fix routes think-block drift to FORK, so signal lands in several Samples.

    Contiguity is what forking costs; every sampled token still trains. The
    fork_threshold no longer suppresses this, because the decision is made on the
    trained tokens a REALIGN would destroy, not on the incoming response length.
    """
    from anyharness import Sample, TrajectoryManager, TurnRecord
    from anyharness.adapters.common import _render_token_ids

    tok = _tokenizer()
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    mgr = TrajectoryManager()
    sid = "fork-sid"
    convo: list[dict] = [{"role": "user", "content": "Go."}]
    sampled = 0

    for i in range(3):
        out = encode("<think>\n\nreasoning %d\n</think>\n\nStep %d.<|im_end|>\n" % (i, i))
        mgr.record_turn(
            sid,
            turn=TurnRecord(prompt_ids=_render_token_ids(convo, tok, tools=None),
                            output_ids=out, finish_reason="stop",
                            output_log_probs=[-0.1] * len(out)),
            prompt_messages=list(convo),
            response_message={"role": "assistant", "content": "Step %d." % i},
        )
        sampled += len(out)
        convo = convo + [{"role": "assistant", "content": "Step %d." % i},
                         {"role": "user", "content": "continue"}]

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0), reward=1.0)
    assert len(samples) > 1, "drifted turns fork into separate Samples"
    assert sum(sum(s.loss_mask) for s in samples) == sampled
    for s in samples:
        assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length


def test_cosmetic_drift_still_realigns():
    """REALIGN must survive for what it was built for: drift over untrained tokens.

    A prompt tail that gets re-rendered (whitespace, punctuation) carries no signal,
    so healing it in place is free and keeps the trajectory contiguous. Only drift
    that would destroy *trained* tokens forks.
    """
    from anyharness.trajectory import DriftKind, TurnRecord, _SampleBuilder

    b = _SampleBuilder(fork_threshold=1024)
    t1 = TurnRecord(prompt_ids=[1, 2, 3], output_ids=[7, 8], finish_reason="stop",
                    output_log_probs=[-0.1, -0.2])
    b.append_turn(t1, b.classify_token_drift(t1))

    # A second turn whose prompt re-renders the *untrained* tail: held tokens are
    # p1+o1 = [1,2,3,7,8]; this turn re-emits o1 as context then drifts after it.
    t2 = TurnRecord(prompt_ids=[1, 2, 3, 7, 8, 9], output_ids=[20], finish_reason="stop",
                    output_log_probs=[-0.3])
    b.append_turn(t2, b.classify_token_drift(t2))
    # Now drift falls strictly inside t2's *untrained* prompt tail region.
    b.loss_mask[-1] = 0  # demote t2's response so the span below is untrained
    t3 = TurnRecord(prompt_ids=[1, 2, 3, 7, 8, 9, 21], output_ids=[30], finish_reason="stop",
                    output_log_probs=[-0.4])
    assert b.classify_token_drift(t3) is DriftKind.REALIGN, (
        "drift over untrained tokens should still heal in place"
    )
