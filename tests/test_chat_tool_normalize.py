"""Tool-shape normalization for the chat upstream: _nest_tools + _ensure_tool_call_ids.

The Responses API uses flat tools ({type,function,name,parameters}) and the tree
stores tool_calls without wire ids (tool_call_dict drops them for dict-equality
matching). vLLM 0.26 + litellm require nested tools and id-bearing tool_calls,
so ChatUpstream normalizes both at its boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anyharness.adapters.common import _ensure_tool_call_ids, _nest_tools  # noqa: E402


def test_nest_flat_to_nested():
    flat = [{"type": "function", "name": "read", "description": "d",
             "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    n = _nest_tools(flat)
    assert n == [{"type": "function", "function": {
        "name": "read", "description": "d",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]


def test_nest_already_nested_passthrough():
    nested = [{"type": "function", "function": {"name": "r", "parameters": {}}}]
    assert _nest_tools(nested) == nested


def test_nest_none_passthrough():
    assert _nest_tools(None) is None
    assert _nest_tools([]) is None or _nest_tools([]) == []


def test_nest_uses_input_schema_fallback():
    flat = [{"type": "function", "name": "t", "description": "",
             "input_schema": {"type": "object", "properties": {}}}]
    n = _nest_tools(flat)
    assert n[0]["function"]["parameters"] == {"type": "object", "properties": {}}


def test_ensure_ids_synthesized_positionally():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "read", "arguments": {}}}]},
        {"role": "tool", "content": "r"},
    ]
    out = _ensure_tool_call_ids(msgs)
    assert out[1]["tool_calls"][0]["id"] == "call_0"
    assert out[1]["tool_calls"][0]["type"] == "function"
    assert out[2]["tool_call_id"] == "call_0"  # paired onto the tool message


def test_ensure_ids_parallel_distinct():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "a", "arguments": {}}},
        {"type": "function", "function": {"name": "b", "arguments": {}}}]}]
    out = _ensure_tool_call_ids(msgs)
    ids = [t["id"] for t in out[0]["tool_calls"]]
    assert ids == ["call_0", "call_1"]


def test_ensure_ids_preserves_existing():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "keep_me", "type": "function", "function": {"name": "a", "arguments": {}}}]}]
    out = _ensure_tool_call_ids(msgs)
    assert out[0]["tool_calls"][0]["id"] == "keep_me"


def test_ensure_ids_pairs_multiple_tool_msgs():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "a", "arguments": {}}},
            {"type": "function", "function": {"name": "b", "arguments": {}}}]},
        {"role": "tool", "content": "ra"},
        {"role": "tool", "content": "rb"},
    ]
    out = _ensure_tool_call_ids(msgs)
    assert out[1]["tool_call_id"] == "call_0"
    assert out[2]["tool_call_id"] == "call_1"
