"""Trace-capture typed primitives (OOP strengthening).

SampledSequence enforces the token↔logprob length invariant at construction;
TopkLogprobs is the sentinel-padded dense pair; to_stop_reason narrows wire
finish reasons to the Tinker SDK Literal["length","stop"].
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anyharness.trajectory import (  # noqa: E402
    MASK_LOGPROB,
    SampledSequence,
    TopkLogprobs,
    TurnRecord,
    to_stop_reason,
)
from anyharness.adapters.common import _sglang_topk_to_typed  # noqa: E402


# --- SampledSequence invariant ---

def test_sampled_sequence_aligned_ok():
    s = SampledSequence(tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3])
    assert len(s.tokens) == len(s.logprobs) == 3


def test_sampled_sequence_no_logprobs_ok():
    s = SampledSequence(tokens=[1, 2, 3])
    assert s.logprobs is None
    assert s.tokens == [1, 2, 3]


def test_sampled_sequence_mismatch_raises():
    with pytest.raises(ValueError, match="logprobs length"):
        SampledSequence(tokens=[1, 2, 3], logprobs=[-0.1, -0.2])


def test_sampled_sequence_default_stop_reason():
    assert SampledSequence(tokens=[]).stop_reason == "stop"
    assert SampledSequence(tokens=[], stop_reason="length").stop_reason == "length"


# --- TurnRecord derives sequence ---

def test_turn_record_sequence_built():
    t = TurnRecord(prompt_ids=[1, 2], output_ids=[5, 6], finish_reason="length",
                  output_log_probs=[-0.1, -0.2])
    assert isinstance(t.sequence, SampledSequence)
    assert t.sequence.tokens == [5, 6]
    assert t.sequence.logprobs == [-0.1, -0.2]
    assert t.sequence.stop_reason == "length"
    # flat fields still readable (compat)
    assert t.output_ids == [5, 6]
    assert t.output_log_probs == [-0.1, -0.2]
    assert t.finish_reason == "length"


def test_turn_record_mismatch_propagates():
    with pytest.raises(ValueError):
        TurnRecord(prompt_ids=[], output_ids=[1, 2], finish_reason="stop",
                   output_log_probs=[-0.1])


def test_turn_record_from_sequence():
    seq = SampledSequence(tokens=[7, 8], logprobs=[-0.3, -0.4], stop_reason="stop")
    t = TurnRecord.from_sequence(prompt_ids=[1], sequence=seq)
    # sequence is rebuilt by __post_init__ (value-equal, not same object)
    assert t.sequence.tokens == seq.tokens
    assert t.sequence.logprobs == seq.logprobs
    assert t.sequence.stop_reason == seq.stop_reason
    assert t.output_ids == [7, 8]
    assert t.finish_reason == "stop"


# --- to_stop_reason mapping ---

@pytest.mark.parametrize("wire", ["length", "max_tokens", "abort"])
def test_to_stop_reason_length(wire):
    assert to_stop_reason(wire) == "length"


@pytest.mark.parametrize("wire", ["stop", "end_turn", "tool_calls", "stop_sequence", ""])
def test_to_stop_reason_stop(wire):
    assert to_stop_reason(wire) == "stop"


# --- TopkLogprobs conversion (sglang) ---

def test_sglang_topk_none_when_absent():
    assert _sglang_topk_to_typed(None, k=3) is None
    assert _sglang_topk_to_typed([], k=3) is None


def test_sglang_topk_pads_to_k():
    raw = [
        [(-0.1, 5), (-0.2, 6)],          # 2 alts, k=3 -> pad 1
        [(-0.3, 7)],                       # 1 alt -> pad 2
    ]
    t = _sglang_topk_to_typed(raw, k=3)
    assert isinstance(t, TopkLogprobs)
    assert len(t.token_ids) == 2 and len(t.logprobs) == 2
    # rectangular: every position width 3
    assert all(len(row) == 3 for row in t.token_ids)
    assert all(len(row) == 3 for row in t.logprobs)
    # position 0: [5,6,0] / [-0.1,-0.2,MASK]
    assert t.token_ids[0] == [5, 6, 0]
    assert t.logprobs[0] == pytest.approx([-0.1, -0.2, MASK_LOGPROB])
    # position 1: [7,0,0] / [-0.3,MASK,MASK]
    assert t.token_ids[1] == [7, 0, 0]
    assert t.logprobs[1][0] == pytest.approx(-0.3)
    assert t.logprobs[1][1:] == [MASK_LOGPROB, MASK_LOGPROB]


def test_sglang_topk_exact_k_no_pad():
    raw = [[(-0.1, 5), (-0.2, 6), (-0.3, 7)]]
    t = _sglang_topk_to_typed(raw, k=3)
    assert t.token_ids == [[5, 6, 7]]
    assert t.logprobs == [pytest.approx([-0.1, -0.2, -0.3])]


def test_sglang_topk_empty_position():
    raw = [[]]  # position with no alternatives
    t = _sglang_topk_to_typed(raw, k=2)
    assert t.token_ids == [[0, 0]]
    assert t.logprobs == [[MASK_LOGPROB, MASK_LOGPROB]]
