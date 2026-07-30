"""responses mode must flatten the chat-shaped tool schema before sending upstream."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import os

import pytest

from anyharness.adapters.anthropic import AnthropicAdapter  # noqa: E402

SAVED = ("UPSTREAM_MODE", "SLIME_RESPONSES_BASE_URL", "SLIME_RESPONSES_MODEL",
         "SLIME_RESPONSES_API_KEY")


def _env(**kw):
    saved = {k: os.environ.get(k) for k in SAVED}
    os.environ.update(UPSTREAM_MODE="responses", SLIME_RESPONSES_BASE_URL="http://x",
                      SLIME_RESPONSES_MODEL="m", SLIME_RESPONSES_API_KEY="k", **kw)
    return saved


def _restore(saved):
    for k, v in saved.items():
        (os.environ.pop if v is None else os.environ.__setitem__)(k, v)


def test_responses_flattens_nested_tools(monkeypatch):
    """nested {type,function:{...}} -> flat {type,name,description,parameters}."""
    import litellm
    captured = {}

    async def fake(**kwargs):
        captured.update(kwargs)
        return object()  # _call_responses_upstream will fail parsing, that's fine

    monkeypatch.setattr(litellm, "aresponses", fake)
    saved = _env()
    try:
        a = AnthropicAdapter(tokenizer=None, sglang_url=None)
        nested = [{"type": "function", "function": {"name": "read", "description": "d",
                   "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
        try:
            import asyncio
            asyncio.run(a._call_responses_upstream(nested, nested, {"max_tokens": 64}, "s"))
        except Exception:
            pass  # we only care about what was sent
        tools = captured.get("tools")
        assert tools is not None
        assert tools == [{"type": "function", "name": "read", "description": "d",
                          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}], tools
    finally:
        _restore(saved)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
