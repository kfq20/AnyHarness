"""Smoke test for the vendored trajectory layer.

Does NOT need a real model, sglang, or the claude binary. It builds a
:class:`~slime_sft_trace.TrajectoryManager` directly, feeds it 2-3 hand-built
turns (via :meth:`record_turn` with explicit token ids), and asserts that
:meth:`get_trajectory` produces :class:`Sample` objects whose ``loss_mask``
marks prompt tokens as 0 and model-response tokens as 1.

This validates the trajectory layer (the other agent's vendored code) end to
end. The adapter + harness layers (also vendored by other agents) are not
exercised here — they need network and a tokenizer.

Trajectory contract being asserted
-----------------------------------
``to_sample`` keeps the FULL token sequence (leading prompt + every appended
prompt-tail + every response) on ``Sample.tokens``. ``loss_mask`` is aligned to
``tokens[leading_prompt_len:]``: the first turn's prompt is stripped, response
tokens are ``1`` (trained), and any *prompt-tail* appended on later turns (the
part of a later prompt beyond what the builder already holds) is ``0``. So
``len(loss_mask) == len(tokens) - leading_prompt_len`` and
``response_length == len(loss_mask)``.
"""

from __future__ import annotations

import pytest

# The trajectory layer is vendored by a parallel agent. If their files are not
# ready at collection time, skip with a clear message rather than erroring.
try:
    from slime_sft_trace import Sample, TrajectoryManager, TurnRecord
except Exception as exc:  # pragma: no cover - depends on other agents' files
    pytest.skip(
        f"slime_sft_trace trajectory layer not importable yet (parallel agent "
        f"not finished): {exc!r}",
        allow_module_level=True,
    )


def _turn(
    prompt_ids: list[int],
    output_ids: list[int],
    *,
    finish: str = "stop",
    logprobs: list[float] | None = None,
) -> TurnRecord:
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=list(output_ids),
        finish_reason=finish,
        output_log_probs=logprobs if logprobs is not None else [-0.5] * len(output_ids),
    )


def _user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def test_single_turn_loss_mask_prompt_zero_response_one():
    """One user->assistant turn: tokens hold prompt+response; loss_mask marks
    only the response (the leading prompt is stripped off the mask)."""
    mgr = TrajectoryManager()
    sid = "sess-1"

    prompt_ids = [1, 2, 3, 4, 5]  # the rendered prompt
    output_ids = [10, 11, 12]  # the model's response
    mgr.record_turn(
        sid,
        turn=_turn(prompt_ids, output_ids),
        prompt_messages=[_user_msg("hello")],
        response_message=_assistant_msg("hi there"),
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0), reward=1.0)

    assert len(samples) == 1
    s = samples[0]
    # tokens keep the FULL sequence (prompt + response).
    assert s.tokens == prompt_ids + output_ids
    # loss_mask is aligned to tokens[leading_prompt_len:], i.e. response only.
    assert s.loss_mask == [1, 1, 1]
    assert len(s.loss_mask) == len(s.tokens) - len(prompt_ids)
    assert s.response_length == 3
    assert s.reward == 1.0
    assert s.status == Sample.Status.COMPLETED
    # logprobs survived the round trip, aligned to the response region.
    assert s.rollout_log_probs == [-0.5, -0.5, -0.5]
    assert s.metadata["truncated"] is False


def test_two_turn_chained_prompt_zero_response_one():
    """A second turn whose prompt extends the first: both responses trained,
    but the prompt-tail appended on turn 2 is loss_mask=0 (context, not trained)."""
    mgr = TrajectoryManager()
    sid = "sess-2"

    p1 = [1, 2, 3]
    o1 = [7, 8]
    mgr.record_turn(
        sid,
        turn=_turn(p1, o1),
        prompt_messages=[_user_msg("q1")],
        response_message=_assistant_msg("a1"),
    )

    # Second turn prompt = prior context (p1 + o1) + a new tail [9, 10]. CLEAN:
    # the builder holds p1+o1, so only [9, 10] is appended (as loss_mask=0),
    # then o2 is appended as loss_mask=1.
    p2 = p1 + o1 + [9, 10]
    o2 = [20, 21, 22]
    mgr.record_turn(
        sid,
        turn=_turn(p2, o2),
        prompt_messages=[_user_msg("q1"), _assistant_msg("a1"), _user_msg("q2")],
        response_message=_assistant_msg("a2"),
    )

    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0), reward=0.5)
    assert len(samples) == 1
    s = samples[0]
    # Full tokens: p1 + o1 + (prompt-tail [9,10]) + o2
    assert s.tokens == [1, 2, 3, 7, 8, 9, 10, 20, 21, 22]
    # loss_mask aligned to tokens[len(p1):]: o1 trained, tail 0, o2 trained.
    assert s.loss_mask == [1, 1, 0, 0, 1, 1, 1]
    assert s.response_length == len(s.loss_mask) == 7
    assert s.reward == 0.5
    # logprobs aligned to the response region: o1 had values, tail padded 0.0.
    assert s.rollout_log_probs == [-0.5, -0.5, 0.0, 0.0, -0.5, -0.5, -0.5]


def test_finish_reason_length_marks_truncated():
    """A turn that ended on 'length' is flagged truncated in metadata."""
    mgr = TrajectoryManager()
    sid = "sess-3"
    mgr.record_turn(
        sid,
        turn=_turn([1, 2], [3, 4], finish="length"),
        prompt_messages=[_user_msg("go")],
        response_message=_assistant_msg("..."),
    )
    samples = mgr.get_trajectory(sid, base_sample=Sample(index=0), reward=0.0)
    assert len(samples) == 1
    assert samples[0].metadata["truncated"] is True


def test_empty_session_returns_no_samples():
    """get_trajectory on a never-seen sid returns [] (and is idempotent)."""
    mgr = TrajectoryManager()
    assert mgr.get_trajectory("nope", base_sample=Sample(index=0)) == []
