"""Sandbox-pluggable harness for launching coding-agent CLIs.

Exports the :class:`Sandbox` protocol, the :class:`LocalSandbox` and
:class:`E2BSandbox` implementations, and :class:`ClaudeCodeHarness` which drives
the Claude Code CLI against the trajectory-capturing adapter.
"""

from __future__ import annotations

from .claude_code import ClaudeCodeHarness, E2BSandbox, LocalSandbox, Sandbox

__all__ = ["Sandbox", "LocalSandbox", "E2BSandbox", "ClaudeCodeHarness"]
