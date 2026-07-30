"""Tests for the message-level SFT trajectory dump.

NO network, NO model, NO tokenizer. Builds a :class:`TrajectoryManager`
directly, feeds it messages-mode turns (``output_ids=[]``), and asserts
:func:`get_trajectory_messages` linearizes the tree into message-level
``Sample`` objects.

These cover the gap that token-only ``get_trajectory`` leaves open: in
messages mode without a tokenizer, ``output_ids`` is empty so the token path
emits 0 samples, but the tree's ``MessageNode.message`` dicts still carry the
full conversation.
"""

from __future__ import annotations

import copy
import json

import pytest

try:
    from anyharness import Sample, TrajectoryManager, TurnRecord
    from anyharness.message_dump import (
        anthropic_wire_to_sft,
        get_trajectory_messages,
    )
except Exception as exc:  # pragma: no cover - depends on vendored layer
    pytest.skip(
        f"anyharness message_dump not importable: {exc!r}",
        allow_module_level=True,
    )


def _msg_turn(*, finish: str = "stop") -> TurnRecord:
    """A messages-mode-no-tokenizer TurnRecord: empty ids, no logprobs."""
    return TurnRecord(
        prompt_ids=[],
        output_ids=[],
        finish_reason=finish,
        output_log_probs=[],
    )


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _asst(text: str, **extra) -> dict:
    m = {"role": "assistant", "content": text}
    m.update(extra)
    return m


def _tool(content: str, tool_call_id: str = "t1") -> dict:
    return {"role": "tool", "content": content, "tool_call_id": tool_call_id}


# ---------------------------------------------------------------------------
# (a) basic 2-turn conversation: returns a sample with the full message list,
#     correct response_length, idempotent consumption.
# ---------------------------------------------------------------------------


def test_basic_two_turn_returns_message_sample_and_consumes_sid():
    mgr = TrajectoryManager()
    sid = "sess-a"

    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("hello")],
        response_message=_asst("hi there"),
    )
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("hello"), _asst("hi there"), _user("how are you?")],
        response_message=_asst("good, thanks"),
    )

    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0), reward=0.5)

    assert len(samples) >= 1
    s = samples[0]
    # prompt is a messages list including the assistant + tool turns
    assert isinstance(s.prompt, list)
    roles = [m["role"] for m in s.prompt]
    assert roles == ["user", "assistant", "user", "assistant"]
    # two generated assistant turns -> response_length == 2
    assert s.response_length == 2
    # last assistant text is the response
    assert s.response == "good, thanks"
    assert s.reward == 0.5
    assert s.status == Sample.Status.COMPLETED
    assert s.metadata["session_id"] == sid
    assert s.metadata["truncated"] is False
    assert s.metadata["use_tool"] is False
    assert s.tokens == []
    assert s.loss_mask is None

    # sid consumed -> second call returns []
    assert get_trajectory_messages(mgr, sid, base_sample=Sample(index=0)) == []


# ---------------------------------------------------------------------------
# (b) 2-turn tool-call conversation: tool_calls arguments is a DICT.
# ---------------------------------------------------------------------------


def test_tool_call_conversation_arguments_are_dict():
    mgr = TrajectoryManager()
    sid = "sess-b"

    tool_call = {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "SF"}},
    }
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("weather?")],
        response_message=_asst("", tool_calls=[tool_call]),
    )
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[
            _user("weather?"),
            _asst("", tool_calls=[tool_call]),
            _tool("sunny, 72F", tool_call_id="t1"),
        ],
        response_message=_asst("It is sunny and 72F in SF."),
    )

    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    msgs = samples[0].prompt
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]

    asst1 = msgs[1]
    assert asst1["tool_calls"][0]["function"]["arguments"] == {"city": "SF"}
    # arguments must be a dict, not a JSON string
    assert isinstance(asst1["tool_calls"][0]["function"]["arguments"], dict)

    assert samples[0].metadata["use_tool"] is True
    assert samples[0].response_length == 2


# ---------------------------------------------------------------------------
# (c) anthropic_wire_to_sft: Anthropic content-blocks -> OpenAI messages.
# ---------------------------------------------------------------------------


def test_anthropic_wire_to_sft_block_conversion():
    anthropic_messages = [
        {"role": "user", "content": "plan a trip"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me consider the route"},
                {"type": "text", "text": "I will use a tool"},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "search_flights",
                    "input": {"origin": "SFO", "dest": "JFK"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc",
                    "content": "UA 200, 9am",
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Booked UA 200 at 9am."}]},
    ]
    tools = [
        {
            "name": "search_flights",
            "description": "Search for flights",
            "input_schema": {"type": "object", "properties": {"origin": {"type": "string"}}},
        }
    ]

    sft_messages, sft_tools = anthropic_wire_to_sft(anthropic_messages, tools)

    # reasoning_content set from thinking
    assert sft_messages[1]["reasoning_content"] == "let me consider the route"
    # tool_calls arguments is a DICT (not a JSON string); wire-only id dropped
    tc = sft_messages[1]["tool_calls"][0]
    assert tc["function"]["name"] == "search_flights"
    assert tc["function"]["arguments"] == {"origin": "SFO", "dest": "JFK"}
    assert isinstance(tc["function"]["arguments"], dict)
    assert "id" not in tc
    # tool role with tool_call_id
    assert sft_messages[2]["role"] == "tool"
    assert sft_messages[2]["tool_call_id"] == "toolu_abc"
    assert sft_messages[2]["content"] == "UA 200, 9am"
    # final assistant text
    assert sft_messages[3]["content"] == "Booked UA 200 at 9am."

    # tools[] uses parameters (renamed from input_schema), wrapped in function
    assert sft_tools is not None
    t0 = sft_tools[0]
    assert t0["type"] == "function"
    assert t0["function"]["name"] == "search_flights"
    assert t0["function"]["parameters"] == {
        "type": "object",
        "properties": {"origin": {"type": "string"}},
    }
    assert "input_schema" not in t0["function"]


def test_anthropic_wire_to_sft_no_tools_returns_none():
    msgs = [{"role": "user", "content": "hi"}]
    sft_messages, sft_tools = anthropic_wire_to_sft(msgs, tools=None)
    assert sft_messages == [{"role": "user", "content": "hi"}]
    assert sft_tools is None


# ---------------------------------------------------------------------------
# (d) subagent fork: a divergent prompt prefix forks the tree into 2 leaves;
#     each leaf yields its own Sample.
# ---------------------------------------------------------------------------


def test_subagent_fork_yields_two_samples():
    mgr = TrajectoryManager()
    sid = "sess-fork"
    turn = _msg_turn()

    # shared first turn
    mgr.record_turn(
        sid,
        turn=turn,
        prompt_messages=[_user("start")],
        response_message=_asst("branch0"),
    )
    # divergent 2nd user message -> fork into two leaves
    mgr.record_turn(
        sid,
        turn=turn,
        prompt_messages=[_user("start"), _asst("branch0"), _user("go-A")],
        response_message=_asst("A"),
    )
    mgr.record_turn(
        sid,
        turn=turn,
        prompt_messages=[_user("start"), _asst("branch0"), _user("go-B")],
        response_message=_asst("B"),
    )

    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 2

    # each leaf chain has its own divergent final assistant message
    last_contents = sorted(s.prompt[-1]["content"] for s in samples)
    assert last_contents == ["A", "B"]
    # both share the prefix
    for s in samples:
        assert s.prompt[0] == _user("start")
        assert s.prompt[1]["content"] == "branch0"
        assert s.response_length == 2  # branch0 + the fork's generated turn

    # consumed
    assert mgr.has_session(sid) is False


# ---------------------------------------------------------------------------
# truncation flag from finish_reason == "length"
# ---------------------------------------------------------------------------


def test_length_finish_marks_truncated():
    mgr = TrajectoryManager()
    sid = "sess-trunc"
    mgr.record_turn(
        sid,
        turn=_msg_turn(finish="length"),
        prompt_messages=[_user("go")],
        response_message=_asst("..."),
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    assert samples[0].metadata["truncated"] is True


# ---------------------------------------------------------------------------
# finish_session integration: messages mode must produce message-level samples
# even though the token-level get_trajectory returns [] (and pops the tree).
# Regression for the branch-before-get_trajectory ordering in
# BaseAdapter.finish_session.
# ---------------------------------------------------------------------------


def test_finish_session_messages_mode_dumps_message_samples():
    import asyncio
    import os

    from anyharness.adapters.anthropic import AnthropicAdapter

    os.environ["UPSTREAM_MODE"] = "messages"
    os.environ.setdefault("SLIME_MESSAGES_UPSTREAM_URL", "http://127.0.0.1:9")
    try:
        adapter = AnthropicAdapter(tokenizer=None, sglang_url=None)
        assert adapter.upstream_mode == "messages"
        sid = "finish-sess"
        adapter.open_session(sid)
        adapter.manager.record_turn(
            sid,
            turn=_msg_turn(),
            prompt_messages=[_user("hi")],
            response_message=_asst("hello"),
        )
        adapter.manager.record_turn(
            sid,
            turn=_msg_turn(),
            prompt_messages=[_user("hi"), _asst("hello"), _user("more")],
            response_message=_asst("here"),
        )

        samples = asyncio.run(
            adapter.finish_session(sid, base_sample=Sample(index=0), reward=0.0)
        )
        # token path returns [] in messages mode -> message-level path must yield
        assert len(samples) == 1
        s = samples[0]
        assert isinstance(s.prompt, list)
        assert [m["role"] for m in s.prompt] == ["user", "assistant", "user", "assistant"]
        assert s.response_length == 2
        assert s.status == Sample.Status.COMPLETED
        # idempotent: sid consumed
        assert adapter.manager.has_session(sid) is False
    finally:
        os.environ.pop("UPSTREAM_MODE", None)


# ---------------------------------------------------------------------------
# FIX 1: tools propagation -- the tools_schema the model saw at rollout must
# surface onto the emitted Sample (md["tools"]) and the dump must write a
# tools column.
# ---------------------------------------------------------------------------


def test_tools_propagate_from_record_turn_metadata():
    """record_turn metadata={"sid":..,"tools":[...]} (as _run_turn now sets)
    surfaces onto the Sample and into the dumped SFT record's tools column."""
    import json
    import tempfile

    from anyharness.dump import dump_samples_sft

    mgr = TrajectoryManager()
    sid = "tools-1"
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    # simulate the _run_turn capture: tools_schema stashed in record_turn metadata
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("weather?")],
        response_message=_asst("sunny"),
        metadata={"sid": sid, "tools": tools_schema},
    )

    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    s = samples[0]
    assert s.metadata["tools"] == tools_schema
    # OpenAI/HF shape preserved
    assert s.metadata["tools"][0]["function"]["name"] == "get_weather"
    assert "parameters" in s.metadata["tools"][0]["function"]

    # dump_samples_sft writes the tools column
    out = tempfile.mkdtemp()
    dump_samples_sft(samples, out)
    import os

    rec = json.loads(open(os.path.join(out, "trajectories_sft.jsonl")).readline())
    assert "tools" in rec
    assert rec["tools"] == tools_schema


def test_tools_propagate_via_extra_metadata():
    """extra_metadata["tools"] (set by the adapter from its per-sid capture)
    wins and is surfaced even when chain nodes carry no tools metadata."""
    mgr = TrajectoryManager()
    sid = "tools-2"
    tools_schema = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("hi")],
        response_message=_asst("hello"),
    )
    samples = get_trajectory_messages(
        mgr, sid, base_sample=Sample(index=0), extra_metadata={"tools": tools_schema}
    )
    assert len(samples) == 1
    assert samples[0].metadata["tools"] == tools_schema


def test_no_tools_when_never_set():
    """A session that never saw tools emits no tools key (column omitted)."""
    mgr = TrajectoryManager()
    sid = "no-tools"
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("hi")],
        response_message=_asst("hello"),
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    assert "tools" not in samples[0].metadata


# ---------------------------------------------------------------------------
# FIX 2: tool_call_id pairing -- dumped tool_calls have an id and tool
# messages have a matching tool_call_id (dump-time re-pairing; the tree is
# left untouched, so slime's dict-equality matching is unaffected).
# ---------------------------------------------------------------------------


def test_tool_call_id_paired_in_live_path():
    """In the LIVE (OpenAI-shape) path the persisted manager_message has no id;
    the dump must re-pair a unique id onto tool_calls and a matching
    tool_call_id onto the tool message."""
    mgr = TrajectoryManager()
    sid = "pair-1"
    tool_call = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}}
    mgr.record_turn(
        sid,
        turn=_msg_turn(finish="tool_calls"),
        prompt_messages=[_user("weather?")],
        response_message=_asst("", tool_calls=[tool_call]),
    )
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[
            _user("weather?"),
            _asst("", tool_calls=[tool_call]),
            _tool("sunny, 72F", tool_call_id="t1"),
        ],
        response_message=_asst("It is sunny and 72F in SF."),
    )

    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    msgs = samples[0].prompt
    asst1 = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in msgs if m["role"] == "tool")

    tc = asst1["tool_calls"][0]
    assert "id" in tc and tc["id"], "tool_call must gain a unique id"
    assert "tool_call_id" in tool_msg, "tool message must gain a tool_call_id"
    assert tc["id"] == tool_msg["tool_call_id"], "id and tool_call_id must match"
    # arguments stays a dict (not a JSON string)
    assert isinstance(tc["function"]["arguments"], dict)


def test_tool_message_name_set_from_call_in_live_path():
    """LIVE path: the tool message's ``name`` is set from the originating
    tool_call's function name (mirrors the production converter's
    tool_name_by_id), so HF/LLaMA-Factory templates that render the tool name
    on a tool result see it."""
    mgr = TrajectoryManager()
    sid = "name-1"
    tool_call = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}}
    mgr.record_turn(
        sid,
        turn=_msg_turn(finish="tool_calls"),
        prompt_messages=[_user("weather?")],
        response_message=_asst("", tool_calls=[tool_call]),
    )
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[
            _user("weather?"),
            _asst("", tool_calls=[copy.deepcopy(tool_call)]),
            _tool("sunny, 72F"),
        ],
        response_message=_asst("It is sunny and 72F in SF."),
    )

    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    tool_msg = next(m for m in samples[0].prompt if m["role"] == "tool")
    assert tool_msg.get("name") == "get_weather", "tool message must carry the tool name"


def test_tool_call_id_reuses_existing_tool_call_id():
    """When the tool message already carries a tool_call_id (caller-fed), the
    dump reuses it on the matching tool_call instead of synthesizing a new id."""
    mgr = TrajectoryManager()
    sid = "pair-2"
    tool_call = {"type": "function", "function": {"name": "calc", "arguments": {"x": 1}}}
    mgr.record_turn(
        sid,
        turn=_msg_turn(finish="tool_calls"),
        prompt_messages=[_user("calc")],
        response_message=_asst("", tool_calls=[tool_call]),
    )
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[
            _user("calc"),
            _asst("", tool_calls=[tool_call]),
            {"role": "tool", "content": "2", "tool_call_id": "existing-id"},
        ],
        response_message=_asst("done"),
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    msgs = samples[0].prompt
    asst1 = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msg = next(m for m in msgs if m["role"] == "tool")
    assert asst1["tool_calls"][0]["id"] == "existing-id"
    assert tool_msg["tool_call_id"] == "existing-id"


def test_multiple_tool_calls_get_unique_pairing():
    """Multiple tool calls in one turn each get a distinct id, each matched to
    its tool message in order."""
    mgr = TrajectoryManager()
    sid = "pair-3"
    tc1 = {"type": "function", "function": {"name": "a", "arguments": {}}}
    tc2 = {"type": "function", "function": {"name": "b", "arguments": {}}}
    mgr.record_turn(
        sid,
        turn=_msg_turn(finish="tool_calls"),
        prompt_messages=[_user("run both")],
        response_message=_asst("", tool_calls=[tc1, tc2]),
    )
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[
            _user("run both"),
            _asst("", tool_calls=[tc1, tc2]),
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
        ],
        response_message=_asst("done"),
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    msgs = samples[0].prompt
    asst1 = next(m for m in msgs if m["role"] == "assistant" and m.get("tool_calls"))
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    ids = [tc["id"] for tc in asst1["tool_calls"]]
    assert len(set(ids)) == 2, "ids must be unique"
    assert [m["tool_call_id"] for m in tool_msgs] == ids, "tool messages pair in order"


# ---------------------------------------------------------------------------
# FIX 3: token-path parity -- ill_formed metadata, empty-leaf skip.
# ---------------------------------------------------------------------------


def test_ill_formed_set_when_turn_is_ill_formed():
    mgr = TrajectoryManager()
    sid = "ill-1"
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("go")],
        response_message=_asst("ok"),
    )
    mgr.record_turn(
        sid,
        turn=TurnRecord(
            prompt_ids=[], output_ids=[], finish_reason="stop", output_log_probs=[], ill_formed=True
        ),
        prompt_messages=[_user("go"), _asst("ok"), _user("again")],
        response_message=_asst("again-ok"),
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    assert samples[0].metadata["ill_formed"] is True


def test_ill_formed_false_when_all_clean():
    mgr = TrajectoryManager()
    sid = "ill-2"
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("go")],
        response_message=_asst("ok"),
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert samples[0].metadata["ill_formed"] is False


def test_empty_leaf_response_message_none_yields_zero_samples():
    """A leaf whose response_message was None (turn set, message None) carries
    nothing to train on -> 0 samples, mirroring the token path."""
    mgr = TrajectoryManager()
    sid = "empty-leaf"
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("hi")],
        response_message=None,
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert samples == []


def test_demoted_only_leaf_yields_zero_samples():
    """When _try_merge_assistant_rewrite demotes the only generated assistant
    to routing-only (turn -> None) and no new response is generated, the leaf
    emits 0 samples (no generated assistant turn)."""
    mgr = TrajectoryManager()
    sid = "demoted"
    # generated assistant, short output
    mgr.record_turn(
        sid,
        turn=TurnRecord(prompt_ids=[1], output_ids=[2], finish_reason="stop"),
        prompt_messages=[_user("start")],
        response_message=_asst("branch0"),
    )
    # replay with a whitespace tweak -> demotes branch0 (turn -> None); no new
    # generated response_message, so the leaf has no generated assistant turn.
    mgr.record_turn(
        sid,
        turn=TurnRecord(prompt_ids=[1], output_ids=[3], finish_reason="stop"),
        prompt_messages=[_user("start"), _asst("branch0 "), _user("go-A")],
        response_message=None,
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert samples == []


# ---------------------------------------------------------------------------
# FIX 4: empty response -- a tool-call-only / thinking-only assistant turn
# must yield a non-empty response (the SFT target must not be blank).
# ---------------------------------------------------------------------------


def test_tool_call_only_turn_yields_nonempty_response():
    mgr = TrajectoryManager()
    sid = "toolonly-1"
    tool_call = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}}
    mgr.record_turn(
        sid,
        turn=_msg_turn(finish="tool_calls"),
        prompt_messages=[_user("weather?")],
        response_message=_asst("", tool_calls=[tool_call]),
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    s = samples[0]
    assert s.response, "response must be non-empty for a tool-call-only turn"
    assert "get_weather" in s.response  # readable summary includes the tool name
    assert s.response_length == 1  # still one generated turn


def test_thinking_only_turn_yields_nonempty_response():
    mgr = TrajectoryManager()
    sid = "thinkonly-1"
    mgr.record_turn(
        sid,
        turn=_msg_turn(),
        prompt_messages=[_user("ponder")],
        response_message={"role": "assistant", "content": "", "reasoning_content": "hmm"},
    )
    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1
    assert samples[0].response, "response must be non-empty for a thinking-only turn"


# ---------------------------------------------------------------------------
# GAP A (parity with production GLM52 converter): tool messages emitted from
# Anthropic tool_result blocks carry the originating tool name, paired by the
# ORIGINAL tool_use id -- and the name must survive _pair_tool_call_ids re-id.
# ---------------------------------------------------------------------------


def test_anthropic_tool_result_carries_name_paired_by_original_id():
    """A user tool_result block becomes a tool message whose `name` is the tool
    that produced it (from the preceding assistant tool_use), paired by the
    original Anthropic toolu_ id -- mirrors
    export_covered_claude_events_to_glm52.anthropic_request_messages_to_glm52."""
    from anyharness.message_dump import _pair_tool_call_ids

    msgs = [
        {"role": "user", "content": "plan a trip"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will use a tool"},
                {"type": "tool_use", "id": "toolu_abc", "name": "search_flights", "input": {"origin": "SFO"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "UA 200, 9am"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Booked."}]},
    ]
    sft, _ = anthropic_wire_to_sft(msgs, tools=None)
    tool_msg = next(m for m in sft if m["role"] == "tool")
    assert tool_msg["name"] == "search_flights"
    assert tool_msg["tool_call_id"] == "toolu_abc"
    assert tool_msg["content"] == "UA 200, 9am"

    # name must survive the dump-time re-pairing that get_trajectory_messages runs
    _pair_tool_call_ids(sft)
    asst = next(m for m in sft if m["role"] == "assistant" and m.get("tool_calls"))
    tc = asst["tool_calls"][0]
    tool_msg2 = next(m for m in sft if m["role"] == "tool")
    # original id preserved (reused, not replaced) so name stays correctly paired
    assert tc["id"] == tool_msg2["tool_call_id"] == "toolu_abc"
    assert tool_msg2["name"] == "search_flights"


def test_anthropic_parallel_tool_results_each_get_own_name():
    """Multiple tool_use blocks in one turn -> each tool message carries its own
    name, paired by its own tool_use id, in order."""
    msgs = [
        {"role": "user", "content": "run both"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "u1", "name": "alpha", "input": {"x": 1}},
                {"type": "tool_use", "id": "u2", "name": "beta", "input": {"y": 2}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "u1", "content": "r1"},
                {"type": "tool_result", "tool_use_id": "u2", "content": "r2"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    sft, _ = anthropic_wire_to_sft(msgs, tools=None)
    tool_msgs = [m for m in sft if m["role"] == "tool"]
    assert [(t["name"], t["tool_call_id"]) for t in tool_msgs] == [("alpha", "u1"), ("beta", "u2")]


def test_tool_result_without_matching_tool_use_has_no_name():
    """A tool_result whose id has no preceding assistant tool_use (orphan) gets
    no name -- it's only ever set when the map resolves (never fabricated)."""
    msgs = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "ghost", "content": "r"}]},
    ]
    sft, _ = anthropic_wire_to_sft(msgs, tools=None)
    tool_msg = next(m for m in sft if m["role"] == "tool")
    assert "name" not in tool_msg
    assert tool_msg["tool_call_id"] == "ghost"


# ---------------------------------------------------------------------------
# GAP B: system prompt as a list of text blocks with cache_control (real Claude
# Code shape) is flattened to a single string, cache_control stripped, blocks
# joined; bare-string system still works.
# ---------------------------------------------------------------------------


def test_system_list_with_cache_control_stripped_and_joined():
    msgs = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Claude Code.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral"}},
            ],
        },
        {"role": "user", "content": "hi"},
    ]
    sft, _ = anthropic_wire_to_sft(msgs, tools=None)
    assert sft[0]["role"] == "system"
    assert "cache_control" not in json.dumps(sft[0])
    assert sft[0]["content"] == "You are Claude Code.\n\nBe terse."


def test_system_bare_string_passes_through():
    sft, _ = anthropic_wire_to_sft(
        [{"role": "system", "content": "bare system"}, {"role": "user", "content": "hi"}], tools=None
    )
    assert sft[0] == {"role": "system", "content": "bare system"}


# ---------------------------------------------------------------------------
# GAP C: a tool with no input_schema defaults to additionalProperties:True
# (matches the production converter / what the model saw at rollout).
# ---------------------------------------------------------------------------


def test_tool_without_schema_defaults_to_additional_properties_true():
    tools = [{"name": "mystery", "description": "no schema given"}]
    _, sft_tools = anthropic_wire_to_sft([], tools=tools)
    assert sft_tools is not None
    assert sft_tools[0]["function"]["parameters"] == {"type": "object", "additionalProperties": True}


def test_tool_with_full_schema_preserved_verbatim():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    tools = [{"name": "f", "description": "d", "input_schema": schema}]
    _, sft_tools = anthropic_wire_to_sft([], tools=tools)
    assert sft_tools[0]["function"]["parameters"] == schema
    assert "input_schema" not in sft_tools[0]["function"]


# ---------------------------------------------------------------------------
# SFT flavors: glm52 keeps tool_calls.function.arguments as a DICT;
# openai_wire JSON-stringifies them and sets meta.format, so datasets<4.7.0
# (which infers arguments/tools as a string) can load the record without a
# Features schema. Mirrors the production converter's stringify_tool_arguments.
# ---------------------------------------------------------------------------


def _tool_call_sample():
    """A Sample whose assistant turn carries a dict-arguments tool call."""
    tc = {
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "SF", "n": 3}},
    }
    return Sample(
        index=0,
        prompt=[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [tc]},
            {"role": "tool", "content": "sunny, 72F", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "It is sunny and 72F in SF."},
        ],
        response="It is sunny and 72F in SF.",
        metadata={"session_id": "flavor-1", "use_tool": True, "truncated": False},
    )


def test_glm52_flavor_keeps_dict_arguments(tmp_path):
    from anyharness.dump import dump_samples_sft

    s = _tool_call_sample()
    dump_samples_sft([s], tmp_path, flavor="glm52")

    path = tmp_path / "trajectories_sft.jsonl"
    assert path.exists(), "glm52 flavor must write the default filename"
    rec = json.loads(path.read_text().splitlines()[0])

    asst = next(m for m in rec["messages"] if m.get("tool_calls"))
    args = asst["tool_calls"][0]["function"]["arguments"]
    # glm52: arguments stay a dict (needs datasets>=4.7 / Json() Features schema)
    assert isinstance(args, dict)
    assert args == {"city": "SF", "n": 3}
    assert rec["meta"]["format"] == "glm52_chat_messages_with_tools_jsonl"


def test_openai_wire_flavor_stringifies_arguments_and_sets_meta_format(tmp_path):
    from anyharness.dump import dump_samples_sft

    s = _tool_call_sample()
    dump_samples_sft([s], tmp_path, flavor="openai_wire")

    path = tmp_path / "trajectories_sft_openai.jsonl"
    assert path.exists(), "openai_wire flavor must write the companion filename"
    rec = json.loads(path.read_text().splitlines()[0])

    asst = next(m for m in rec["messages"] if m.get("tool_calls"))
    args = asst["tool_calls"][0]["function"]["arguments"]
    # openai_wire: arguments JSON-stringified so datasets<4.7 infers str
    assert isinstance(args, str)
    assert json.loads(args) == {"city": "SF", "n": 3}
    # meta.format marker mirrors the production converter's stringify_tool_arguments
    assert rec["meta"]["format"] == "openai_chat_messages_with_tools_jsonl"


def test_glm52_and_openai_wire_flavors_both_load_via_json(tmp_path):
    """Both flavors round-trip through json.loads and the openai_wire arguments
    decode back to the original dict (datasets<4.7 would json.loads it)."""
    from anyharness.dump import dump_samples_sft

    s = _tool_call_sample()
    dump_samples_sft([s], tmp_path, flavor="glm52")
    dump_samples_sft([s], tmp_path, flavor="openai_wire")

    glm52_rec = json.loads(
        (tmp_path / "trajectories_sft.jsonl").read_text().splitlines()[0]
    )
    openai_rec = json.loads(
        (tmp_path / "trajectories_sft_openai.jsonl").read_text().splitlines()[0]
    )

    glm52_args = next(
        m for m in glm52_rec["messages"] if m.get("tool_calls")
    )["tool_calls"][0]["function"]["arguments"]
    openai_args = next(
        m for m in openai_rec["messages"] if m.get("tool_calls")
    )["tool_calls"][0]["function"]["arguments"]

    assert isinstance(glm52_args, dict)
    assert isinstance(openai_args, str)
    # the stringified form decodes to the same dict the glm52 flavor kept
    assert json.loads(openai_args) == glm52_args


def test_openai_wire_stringify_mirrors_production_converter(tmp_path):
    """The openai_wire flavor must match the production converter's
    stringify_tool_arguments output byte-for-byte (format marker + JSON
    stringification with ensure_ascii=False). Regression for drift."""
    from anyharness.dump import _stringify_tool_arguments, _sample_to_sft_record

    s = _tool_call_sample()
    base = _sample_to_sft_record(s)

    # Production converter (export_covered_claude_events_to_glm52_sft) logic:
    converted = copy.deepcopy(base)
    converted["meta"]["format"] = "openai_chat_messages_with_tools_jsonl"
    for message in converted.get("messages") or []:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                fn["arguments"] = json.dumps(args or {}, ensure_ascii=False)

    ours = _stringify_tool_arguments(base)
    assert ours == converted


def test_default_flavor_is_glm52(tmp_path):
    """dump_samples_sft without an explicit flavor writes the glm52 file with
    dict arguments (backward-compatible default for existing callers)."""
    from anyharness.dump import dump_samples_sft

    s = _tool_call_sample()
    dump_samples_sft([s], tmp_path)  # no flavor kwarg

    assert (tmp_path / "trajectories_sft.jsonl").exists()
    rec = json.loads((tmp_path / "trajectories_sft.jsonl").read_text().splitlines()[0])
    args = next(m for m in rec["messages"] if m.get("tool_calls"))["tool_calls"][0][
        "function"
    ]["arguments"]
    assert isinstance(args, dict)
    assert rec["meta"]["format"] == "glm52_chat_messages_with_tools_jsonl"


def test_build_reply_preserves_non_standard_block_types():
    """_build_reply_parts_from_blocks must not silently drop code / reasoning /
    redacted_thinking / unknown blocks — they fall back to text (matching the
    production converter's content_part_to_text)."""
    from anyharness.adapters.anthropic import _build_reply_parts_from_blocks

    mm, stop = _build_reply_parts_from_blocks(
        [
            {"type": "thinking", "thinking": "plan it"},
            {"type": "text", "text": "doing it"},
            {"type": "code", "code": "print(1)"},
            {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}},
            {"type": "redacted_thinking", "data": "abc"},
        ],
        "stop",
    )
    assert "doing it" in mm["content"]
    assert "print(1)" in mm["content"], "code block content must be preserved as text"
    assert "redacted_thinking" in mm["content"], "unknown block must be JSON-preserved"
    assert mm["reasoning_content"] == "plan it"
    assert mm["tool_calls"] and mm["tool_calls"][0]["function"]["name"] == "f"
    assert stop == "tool_use"


def test_multiblock_prompt_joined_with_double_newline():
    """flatten_content joins multi-block text with \\n\\n (production parity),
    so a multi-block system prompt tokenizes like production SFT data."""
    from anyharness.adapters.common import flatten_content

    joined = flatten_content([{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}])
    assert joined == "line one\n\nline two"


def test_tool_result_blanked_does_not_fork():
    """Claude Code trims large tool_results to "" on a later turn. The tree must
    NOT fork: the blanked replay must match the stored real-content tool node,
    and the real content must survive into the SFT dump (ToolMessage excludes
    content from == but keeps it for subscript/json)."""
    from anyharness.adapters.anthropic import _translate_messages

    mgr = TrajectoryManager()
    sid = "blank"
    mgr.record_turn(
        sid, turn=_msg_turn(),
        prompt_messages=_translate_messages([_user("list")], system=None),
        response_message=_asst("", tool_calls=[{"type": "function", "function": {"name": "run", "arguments": {"cmd": "ls"}}}]),
    )
    real = [
        _user("list"),
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "run", "input": {"cmd": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.txt\nb.txt"}]},
    ]
    mgr.record_turn(sid, turn=_msg_turn(), prompt_messages=_translate_messages(real, system=None),
                    response_message=_asst("found 2 files"))
    # same branch, but tool_result content blanked to "" (ctx trimming)
    blanked = [
        _user("list"),
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "run", "input": {"cmd": "ls"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": ""}]},
        {"role": "assistant", "content": "found 2 files"},
    ]
    mgr.record_turn(sid, turn=_msg_turn(), prompt_messages=_translate_messages(blanked, system=None),
                    response_message=_asst("done"))

    samples = get_trajectory_messages(mgr, sid, base_sample=Sample(index=0))
    assert len(samples) == 1, f"blanked tool_result must not fork; got {len(samples)}"
    # real tool content survives into the dump (not "")
    tool_content = [m["content"] for m in samples[0].prompt if m["role"] == "tool"]
    assert tool_content == ["a.txt\nb.txt"], f"real tool result lost: {tool_content}"


# ---------------------------------------------------------------------------
# chat upstream mode (litellm.acompletion) — pure-function transform tests
# ---------------------------------------------------------------------------


def test_chat_response_to_blocks_tool_call_dict_args():
    """litellm emits tool_calls.arguments as a JSON STRING; chat_response_to_blocks
    must normalize to a dict (the tree/dump shape) and emit tool_use blocks."""
    from anyharness.adapters.anthropic import chat_response_to_blocks

    class _Fn:
        name = "Read"
        arguments = '{"file_path": "/tmp/x.py"}'  # string, as litellm emits

    class _TC:
        id = "call_1"
        type = "function"
        function = _Fn()

    class _Msg:
        content = "reading"
        tool_calls = [_TC()]
        reasoning_content = "plan"

    class _Choice:
        finish_reason = "tool_calls"
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    blocks, stop = chat_response_to_blocks(_Resp())
    types = [b["type"] for b in blocks]
    assert "thinking" in types and "text" in types and "tool_use" in types
    tu = next(b for b in blocks if b["type"] == "tool_use")
    assert tu["name"] == "Read"
    assert tu["input"] == {"file_path": "/tmp/x.py"}  # dict, not str
    assert tu["id"] == "call_1"
    assert stop == "tool_use"


def test_stringify_tool_call_args_serializes_dict_for_wire():
    """_stringify_tool_call_args turns dict arguments into a JSON string for the
    chat upstream (some endpoints reject dict args in replayed history), without
    touching the original messages."""
    from anyharness.adapters.common import _stringify_tool_call_args

    original = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": {"file_path": "/a"}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"},
    ]
    wire = _stringify_tool_call_args(original)
    # original untouched (dict)
    assert isinstance(original[2]["tool_calls"][0]["function"]["arguments"], dict)
    # wire has string args
    assert wire[2]["tool_calls"][0]["function"]["arguments"] == '{"file_path": "/a"}'
    # non-assistant messages pass through unchanged
    assert wire[3] == original[3]
