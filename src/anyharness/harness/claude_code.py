"""Claude Code harness — a simplified, sandbox-pluggable version of Slime's.

Slime's harness (``vendor_slime_ref/agent/harness/``) runs the Claude Code CLI
*inside an E2B sandbox*: it installs Node + the npm CLI, pre-acks
bypass-permissions, then launches ``claude -p ... --output-format stream-json``
against the local AnthropicAdapter.

This module keeps the launch-and-wait shape but drops the E2B coupling: the
sandbox is a pluggable :class:`Sandbox` protocol with two implementations
(:class:`LocalSandbox` for dev/testing, :class:`E2BSandbox` for real tasks). The
npm-install step is removed — **the ``claude`` CLI is assumed to already be on
PATH** (see :meth:`ClaudeCodeHarness.run`). ``write_config`` is kept as an
optional pre-ack step.

The adapter the CLI talks to is served out-of-process by ``cli.py`` (an
aiohttp app exposing ``/v1/messages``); the harness only needs its URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ===========================================================================
# Sandbox protocol + implementations
# ===========================================================================


@runtime_checkable
class Sandbox(Protocol):
    """Minimal command-execution surface a harness needs.

    Mirrors the subset of Slime's ``slime/agent/sandbox.py`` the harness calls:
    ``exec`` (run a shell command, capture stdout) and ``write_file`` (seed a
    path with bytes/text). ``exec`` returns ``(exit_code, stdout)``.
    """

    async def exec(
        self,
        cmd: str,
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str]:
        ...

    async def write_file(self, path: str, content: str | bytes) -> None:
        ...


class LocalSandbox:
    """Run commands on the local host via ``asyncio.create_subprocess_exec``.

    For dev/testing: no sandboxing. Commands run through ``bash -c`` so shell
    features (pipes, ``&&``, redirects) work, matching how Slime's sandbox
    treats ``exec(cmd)`` as a shell string.
    """

    def __init__(self) -> None:
        self._env: dict[str, str] = dict(os.environ)

    async def exec(
        self,
        cmd: str,
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str]:
        full_env = {**self._env, **(env or {})}
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            cmd,
            cwd=workdir,
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return (-1, f"<command timed out after {timeout}s>")
        text = out.decode(errors="replace") if out else ""
        return (proc.returncode if proc.returncode is not None else -1, text)

    async def write_file(self, path: str, content: str | bytes) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            p.write_text(content, encoding="utf-8")
        else:
            p.write_bytes(content)


class E2BSandbox:
    """Thin async wrapper over ``e2b_code_interpreter.AsyncSandbox``.

    Used for real tasks; the e2b SDK is imported lazily so the module loads
    without it. Mirrors Slime's ``sandbox.exec`` (``sbx.commands.run``) and
    ``sandbox.write_file`` (``sbx.files.write``).
    """

    def __init__(self, sbx: Any | None = None, *, api_key: str | None = None) -> None:
        if sbx is not None:
            self._sbx = sbx
        else:
            from e2b_code_interpreter import AsyncSandbox

            self._sbx = AsyncSandbox.create(api_key=api_key) if api_key else AsyncSandbox.create()

    async def exec(
        self,
        cmd: str,
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str]:
        result = await self._sbx.commands.run(
            cmd,
            envs=env or None,
            cwd=workdir,
            timeout=timeout,
        )
        return (getattr(result, "exit_code", 0), getattr(result, "stdout", "") or "")

    async def write_file(self, path: str, content: str | bytes) -> None:
        data = content.encode() if isinstance(content, str) else content
        await self._sbx.files.write(path, data)


# ===========================================================================
# Harness
# ===========================================================================


class ClaudeCodeHarness:
    """Launch the Claude Code CLI against the trajectory-capturing adapter.

    Unlike Slime's E2B-coupled harness, this one is sandbox-agnostic: pass any
    :class:`Sandbox` (LocalSandbox / E2BSandbox) to :meth:`run`. The npm CLI
    install is gone — ``claude`` must already be on PATH (documented below).
    """

    name = "claude_code"

    # Flags mirror Slime's launch_flags, minus --include-hook-events (not
    # needed for trajectory capture) to keep the stream lean.
    launch_flags = (
        "--permission-mode bypassPermissions "
        "--output-format stream-json --include-partial-messages --verbose"
    )

    static_env = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }

    # Assumption: the `claude` CLI is installed and on PATH before run() is
    # called. Slime installs it inside the sandbox via install_npm_cli; that
    # step is intentionally removed here. Override the binary with CLAUDE_BIN
    # (default "claude") if it lives elsewhere.
    claude_bin_env = "CLAUDE_BIN"

    def __init__(self, *, model: str | None = None,
                 launch_flags: str | None = None,
                 preack_bypass: bool = True) -> None:
        """Configure the claude-code launch.

        Parameters
        ----------
        model:
            Model id passed to claude via ``ANTHROPIC_MODEL``. Defaults to the
            ``CLAUDE_MODEL`` env var or ``"slime-actor"``.
        launch_flags:
            Override the flags appended to ``claude -p <prompt>``. The default
            uses ``--permission-mode bypassPermissions`` (headless, no prompts),
            which requires a sandbox where disarming permissions is safe. In
            environments where bypass is unavailable or undesirable (e.g. a
            locked root host, local dev, CI that forbids bypass), pass an
            alternative such as ``"--output-format stream-json --verbose"``
            combined with a ``--settings <file>`` (acceptEdits) path. The flags
            must still emit ``stream-json`` for trajectory capture. The string
            is parsed with :func:`shlex.split` and each token is shell-quoted
            before being joined into the command (so values containing spaces
            must be quoted in the string, e.g.
            ``--settings '/path with space/settings.json'``); this prevents
            shell-injection and quoting breaks since the command runs through
            ``bash -c``.
        preack_bypass:
            Whether to pre-ack bypass-permissions via :meth:`write_config`
            before launching. Leave ``True`` when using the default
            ``launch_flags`` (which enables bypass) — otherwise claude-code may
            prompt on startup and block the run. Set ``False`` only with a
            non-bypass ``launch_flags`` (the bypass pre-ack is then irrelevant).
            ``ValueError`` is raised for the likely-incorrect combination of
            the default (bypass) ``launch_flags`` with ``preack_bypass=False``.
        """
        self.model = model or os.environ.get("CLAUDE_MODEL", "slime-actor")
        self._launch_flags = launch_flags if launch_flags is not None else self.launch_flags
        self._preack_bypass = preack_bypass
        # Guard the likely-incorrect combination: default (bypass) launch_flags
        # but the caller disabled the bypass pre-ack. A non-bypass launch_flags
        # override is the only case where preack_bypass=False makes sense.
        if not preack_bypass and "bypassPermissions" in self._launch_flags:
            raise ValueError(
                "preack_bypass=False is inconsistent with a launch_flags that "
                "enables bypassPermissions (the default). Either keep "
                "preack_bypass=True (default) with the default launch_flags, or "
                "pass a non-bypass launch_flags (e.g. '--output-format "
                "stream-json --verbose --settings <acceptEdits file>')."
            )

    async def write_config(self, sb: Sandbox, workdir: str) -> None:
        """Pre-ack bypass-permissions so claude-code starts headless.

        Writes ~/.claude.json + ~/.claude/settings.json with
        hasCompletedOnboarding + bypassPermissionsModeAccepted. Optional: if it
        fails (e.g. read-only home) the run continues — claude can still run
        with --permission-mode bypassPermissions in most setups.
        """
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        home = os.environ.get("HOME", "/root")
        try:
            await sb.exec(
                f"mkdir -p {shlex.quote(home + '/.claude')} && "
                f"echo {shlex.quote(settings)} "
                f"| tee {shlex.quote(home + '/.claude.json')} "
                f"{shlex.quote(home + '/.claude/settings.json')} > /dev/null",
                timeout=30,
            )
        except Exception:
            logger.warning("write_config failed (continuing); bypass-permissions may prompt")

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        prompt: str,
        time_budget_sec: int,
    ) -> int:
        """Run claude-code to completion and return its exit code.

        Steps: (optional) write_config -> launch ``claude -p <prompt>`` with the
        adapter as ANTHROPIC_BASE_URL -> wait within ``time_budget_sec``.

        The adapter routes requests to their trajectory tree by session id, but
        the id is NOT the upstream API key (forwarding it upstream would 401).
        We pass the session id via SLIME_SESSION_ID (read by the adapter), and let
        claude-code carry the real upstream credential from ANTHROPIC_AUTH_TOKEN /
        ANTHROPIC_API_KEY so the adapter can forward it verbatim to the upstream.

        When ``launch_flags`` omits ``--permission-mode bypassPermissions`` (or
        ``preack_bypass=False``), the bypass pre-ack is skipped. The caller is
        then responsible for headless execution — typically by including a
        ``--settings <file>`` with ``defaultMode: acceptEdits`` in
        ``launch_flags`` (and, if a global ``~/.claude/settings.json`` overrides
        ``env`` keys like ``ANTHROPIC_BASE_URL``, embedding the adapter URL + auth
        token in that settings file's ``env`` block, since claude-code's
        ``--settings`` wins over the ambient environment and global settings).
        """
        if self._preack_bypass:
            await self.write_config(sb, workdir)

        claude_bin = os.environ.get(self.claude_bin_env, "claude")
        # launch_flags is user-supplied and the command runs through `bash -c`
        # (LocalSandbox.exec), so quote each token to prevent shell injection
        # and quoting breaks on spaces (e.g. "--settings /path with space").
        flag_tokens = shlex.split(self._launch_flags)
        quoted_flags = " ".join(shlex.quote(t) for t in flag_tokens)
        cmd = f"{shlex.quote(claude_bin)} -p {shlex.quote(prompt)} {quoted_flags}"
        # Carry the real upstream credential through to claude-code (it sends it
        # as the Authorization header, which the adapter forwards to the upstream).
        upstream_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        env = {
            "ANTHROPIC_BASE_URL": adapter_url,
            "ANTHROPIC_MODEL": self.model,
            "SLIME_SESSION_ID": session_id,
            **self.static_env,
        }
        if upstream_token:
            # claude-code sends ANTHROPIC_AUTH_TOKEN as the Authorization: Bearer
            # header; the adapter forwards it verbatim to the messages upstream.
            env["ANTHROPIC_AUTH_TOKEN"] = upstream_token

        exit_code, out = await sb.exec(
            cmd, env=env, workdir=workdir, timeout=time_budget_sec
        )
        if exit_code != 0:
            logger.warning(
                "claude_code run exited %d; tail:\n%s", exit_code, out[-1500:]
            )
        return exit_code


__all__ = ["Sandbox", "LocalSandbox", "E2BSandbox", "ClaudeCodeHarness"]
