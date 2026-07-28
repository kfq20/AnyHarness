"""Adversarial verification tests for the message-level SFT dump.

Exercises edge cases the shipped tests don't cover: tools propagation,
ill_formed absence, multi-block thinking join, tool-role passthrough, the
raw-Anthropic-content-block branch via get_trajectory_messages, and the
finish_session ordering regression directly.
"""

from __future__ import annotations

import pytest

from slime_sft_trace import Sample, TrajectoryManager, TurnRecord
from slime_sft_trace.message_dump import (
    anthropic_wire_to_sft,
    get_trajectory_messages,
)
from slime_sft_trace.dump import _sample_to_sft_record, dump_samples_sft


def _msg_turn(*, finish="stop"):
    return TurnRecord(prompt_ids=[], output_ids=[], finish_reason=finish, output_log_probs=[])


def _user(t):
    return {"role": "user", "content": t}


def _asst(t, **kw):
    m = {"role": "assistant", "content": t}
    m.update(kw)
    return m


# ---------------------------------------------------------------------------
# CLAIM 1: messages-mode non-zero where token path is zero
# ---------------------------------------------------------------------------


def test_token_path_zero_but_message_path_nonzero():
    """With output_ids=[] the token-level get_trajectory emits 0 samples (no
    trainable response), while get_trajectory_messages emits >=1."""
    mgr = TrajectoryManager()
    sid = "s1"
    mgr.record_turn(
        sid, turn=_msg_turn(), prompt_messages=[_user("hi")], response_message=_asst("hello")
    )
    # token path: no output_ids -> builders have empty response -> 0 samples
    tok = mgr.get_trajectory(sid, base_sample=Sample(index=0))
    assert tok == []
    assert mgr.has_session(sid) is False  # token path consumed it already

    # rebuild fresh tree (token path popped it) and prove message path is nonzero
    mgr2 = TrajectoryManager()
    sid2 = "s1b"
    mgr2.record_turn(sid2, turn=_msg_turn(), prompt_messages=[_user("hi")], response_message=_asst("hello"))
    msg = get_trajectory_messages(mgr2, sid2, base_sample=Sample(index=0))
    assert len(msg) >= 1


def test_chain_of_assistant_turns_response_length_counts_generated_only():
    mgr = TrajectoryManager()
    sid = "len"
    # user, assistant(gen1), user, assistant(gen2), user, assistant(gen3)
    mgr.record_turn(sid, turn=_msg_turn(), prompt_messages=[_user("a")], response_message=_asst("a1"))
    mgr.record_turn(
        sid, turn=_msg_turn(), prompt_messages=[_user("a"), _asst("a1"), _user("b")], response_message=_asst("a2")
    )
    mgr.record_turn(
        sid, turn=_msg_turn(),
        prompt_messages=[_user("a"), _asst("a1"), _user("b"), _asst("a2"), _user("c")],
        response_message=_asst("a3"),
    )
    s = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))[0]
    assert s.response_length == 3
    assert [m["role"] for m in s.prompt] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert s.response == "a3"


# ---------------------------------------------------------------------------
# CLAIM 3: format correctness beyond the shipped block test
# ---------------------------------------------------------------------------


def test_multiple_thinking_blocks_are_joined():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "part1"},
            {"type": "thinking", "thinking": "part2"},
            {"type": "text", "text": "ans"},
        ]},
    ]
    out, _ = anthropic_wire_to_sft(msgs, tools=None)
    assert out[1]["reasoning_content"] == "part1part2"
    assert out[1]["content"] == "ans"


def test_tool_use_with_no_input_yields_empty_dict_args():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": "ping"}]},  # no input
    ]
    out, _ = anthropic_wire_to_sft(msgs, tools=None)
    tc = out[1]["tool_calls"][0]
    assert tc["function"]["arguments"] == {}
    assert isinstance(tc["function"]["arguments"], dict)


def test_openai_shape_passthrough_already_role_tool():
    """A tool message already in OpenAI shape should keep its tool_call_id."""
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]},
        {"role": "tool", "content": "result", "tool_call_id": "call_9"},
    ]
    out, _ = anthropic_wire_to_sft(msgs, tools=None)
    # the tool message must survive and keep tool_call_id
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[-1]["tool_call_id"] == "call_9"
    assert tool_msgs[-1]["content"] == "result"


def test_system_string_top_level():
    out, _ = anthropic_wire_to_sft(
        [{"role": "system", "content": [{"type": "text", "text": "be brief"}]}, {"role": "user", "content": "hi"}],
        tools=None,
    )
    assert out[0] == {"role": "system", "content": "be brief"}


# ---------------------------------------------------------------------------
# CLAIM 2b: _try_merge_assistant_rewrite demote -> node NOT counted as generated
# ---------------------------------------------------------------------------


def test_merge_assistant_rewrite_demotes_turn_and_drops_from_response_length():
    """A short generated assistant that is rewritten by a later prompt (merge)
    is demoted to turn=None and must NOT count in response_length."""
    mgr = TrajectoryManager(fork_threshold_tokens=1024)
    sid = "merge"
    # turn 1: user -> assistant(gen) with a short response
    mgr.record_turn(sid, turn=_msg_turn(), prompt_messages=[_user("q")], response_message=_asst("first answer"))
    # turn 2: replay the SAME prompt but a REWRITTEN assistant then a new user
    # -> _try_merge_assistant_rewrite demotes the generated node to turn=None
    mgr.record_turn(
        sid, turn=_msg_turn(),
        prompt_messages=[_user("q"), _asst("first answer!"), _user("more")],  # note trailing "!"
        response_message=_asst("second answer"),
    )
    s = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))[0]
    # the demoted turn must not be counted: only "second answer" is generated
    assert s.response_length == 1
    assert s.response == "second answer"


# ---------------------------------------------------------------------------
# CLAIM 1 (negation): token path still works in sglang mode (output_ids set)
# ---------------------------------------------------------------------------


def test_token_path_with_output_ids_yields_sample():
    mgr = TrajectoryManager()
    sid = "tok"
    mgr.record_turn(
        sid, turn=TurnRecord(prompt_ids=[1, 2], output_ids=[9, 9], finish_reason="stop"),
        prompt_messages=[_user("hi")], response_message=_asst("x"),
    )
    tok = mgr.get_trajectory(sid, base_sample=Sample(index=0))
    assert len(tok) == 1
    assert tok[0].tokens == [1, 2, 9, 9]


# ---------------------------------------------------------------------------
# CLAIM 3 (FORMAT) + GAP: tools never reach the SFT record in the live path
# ---------------------------------------------------------------------------


def test_tools_propagation_live_path_gap():
    """The tools SCHEMA (function definitions) is surfaced onto the SFT record
    only when it was captured (record_turn metadata={"tools":...}, as _run_turn
    now sets, or via extra_metadata). A tree built WITHOUT capturing tools still
    has no tools column -- the schema isn't synthesized from the tool_calls."""
    mgr = TrajectoryManager()
    sid = "tools-live"
    # simulate an adapter-built manager_message (OpenAI shape) with tool_calls,
    # but NO tools schema captured on the node (mirrors the pre-fix tree shape)
    tc = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}}
    mgr.record_turn(
        sid, turn=_msg_turn(), prompt_messages=[_user("weather?")],
        response_message=_asst("ok", tool_calls=[tc]),
    )
    s = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))[0]
    rec = _sample_to_sft_record(s)
    # the manager_message carries tool_calls (assistant side)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in rec["messages"])
    # no tools schema was captured for this tree -> no tools column
    assert "tools" not in rec


# ---------------------------------------------------------------------------
# CLAIM 3: ill_formed present in token path metadata but ABSENT in message path
# ---------------------------------------------------------------------------


def test_message_path_metadata_includes_ill_formed():
    """FIX 3a: get_trajectory sets metadata['ill_formed']; the message path now
    matches it (parity) instead of omitting it."""
    mgr = TrajectoryManager()
    sid = "ill"
    mgr.record_turn(
        sid, turn=TurnRecord(prompt_ids=[], output_ids=[], finish_reason="stop", ill_formed=True),
        prompt_messages=[_user("hi")], response_message=_asst("x"),
    )
    s = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))[0]
    assert s.metadata["ill_formed"] is True
    # token path matches:
    mgr2 = TrajectoryManager()
    mgr2.record_turn(
        sid, turn=TurnRecord(prompt_ids=[1], output_ids=[9], finish_reason="stop", ill_formed=True),
        prompt_messages=[_user("hi")], response_message=_asst("x"),
    )
    ts = mgr2.get_trajectory(sid, base_sample=Sample(index=0))[0]
    assert ts.metadata["ill_formed"] is True


# ---------------------------------------------------------------------------
# CLAIM 3: dump_samples_sft end-to-end produces valid OpenAI tool-call JSONL
# ---------------------------------------------------------------------------


def test_dump_samples_sft_writes_valid_jsonl_with_dict_arguments(tmp_path):
    tc = {"type": "function", "function": {"name": "f", "arguments": {"k": "v"}}}
    s = Sample(
        index=0,
        prompt=[{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [tc]}],
        response="done",
        metadata={"session_id": "s", "use_tool": True, "truncated": False},
    )
    dump_samples_sft([s], tmp_path)
    import json
    lines = (tmp_path / "trajectories_sft.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"k": "v"}
    assert rec["meta"]["session_id"] == "s"
