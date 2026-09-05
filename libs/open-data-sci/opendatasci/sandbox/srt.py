"""SRT-backed sandbox for OpenDataSci sessions.

Platform support: the underlying Sandbox Runtime only sandboxes on macOS and
Linux. Windows is unsupported: ``sandbox_runtime`` imports the Unix-only
``resource`` module at import time, so this module fails to import at all on
Windows; this class is therefore exercised only under mocks on such hosts.

Accelerator (GPU/NPU) passthrough for the ``[deep-learning]`` extra is Linux
only and opt-in — see ``_inject_accelerator_devices`` below for the mechanism
and the security tradeoff it logs a warning about. Tested on native Linux and
WSL2 with an NVIDIA GPU; macOS has no accelerator passthrough, so deep
learning there runs on CPU only.
"""

import asyncio
import base64
import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from sandbox_runtime import SandboxManager, SandboxRuntimeConfig
from sandbox_runtime.utils.platform import get_platform

from opendatasci._utils.accelerator_utils import discover_accelerator_devices
from opendatasci._utils.fs_utils import find_maybe_sensitive_paths
from opendatasci._utils.package_extras_utils import is_deep_learning_extra_active
from opendatasci._utils.system_dependency_utils import (
    build_linux_install_command,
    build_macos_brew_install_command,
)
from opendatasci.sandbox.base import (
    PAYLOAD_SENTINEL,
    BaseSandbox,
    BaseSandboxFactory,
    SandboxExecResult,
    validate_cli_command,
)

logger = logging.getLogger(__name__)

_DEFAULT_COMMAND_TIMEOUT = 43200  # 12 hours

# asyncio.StreamReader.readline()'s default 64KB-per-line buffer is fine for
# communicate()-style whole-output reads (no line splitting involved) but
# becomes a real constraint once execute() reads line-by-line to stream
# progress -- a single print() of an unformatted large object with no
# embedded newline could exceed it. Raised generously here; a pathological
# single line still longer than this degrades to an SRTError via execute()'s
# broad except-Exception fallback rather than hanging, which is an accepted
# (documented, not silently swallowed) edge case rather than something this
# handles specially.
_STDOUT_STREAM_LIMIT = 10 * 1024 * 1024  # 10 MB

# Install commands per platform, used to build an actionable error message when
# the native sandbox binaries (bwrap/socat/ripgrep) are missing. ``pip install``
# cannot provide these — they must come from the OS package manager.
_INSTALL_HINTS: dict[str, str] = {
    "macos": "brew install ripgrep",
    "linux": (
        "sudo apt-get install -y bubblewrap socat ripgrep  # Debian/Ubuntu\n"
        "  sudo dnf install -y bubblewrap socat ripgrep      # Fedora\n"
        "  sudo pacman -S --noconfirm bubblewrap socat ripgrep  # Arch"
    ),
}


def check_sandbox_dependencies() -> None:
    """Raise ``RuntimeError`` with actionable guidance if the sandbox cannot run.

    Call this as early as possible (e.g. at agent construction) so a missing
    system dependency surfaces immediately, rather than on the first sandboxed
    code execution deep into a session.
    """
    status = get_system_dependency_status()
    if status.satisfied:
        return

    if not status.supported:
        raise RuntimeError(
            f"OpenDataSci's sandbox is not supported on platform '{status.platform}'. "
            "Only macOS and Linux are supported."
        )

    message = f"Missing required sandbox dependencies for {status.platform}: {status.description}."
    if status.manual_install_hint:
        message += f" Install with:\n  {status.manual_install_hint}"
    raise RuntimeError(message)


@dataclass(frozen=True)
class SystemDependencyStatus:
    """Non-raising snapshot of whether the sandbox's native OS dependencies are present.

    Used by the TUI's startup wizard to offer installing missing dependencies
    before the sandbox is ever constructed, rather than surfacing a
    ``RuntimeError`` mid-session. See :func:`check_sandbox_dependencies` for
    the raising counterpart used deeper in the stack.
    """

    satisfied: bool
    platform: str
    supported: bool
    description: str
    manual_install_hint: str | None


def get_system_dependency_status() -> SystemDependencyStatus:
    """Return whether the host has everything the sandbox needs, without raising."""
    platform = get_platform()
    return SystemDependencyStatus(
        satisfied=SandboxManager.check_dependencies(),
        platform=platform,
        supported=SandboxManager.is_supported_platform(platform),
        description=(
            "ripgrep (rg)" if platform == "macos" else "ripgrep (rg), bubblewrap (bwrap), and socat"
        ),
        manual_install_hint=_INSTALL_HINTS.get(platform),
    )


def build_auto_install_command(platform: str) -> list[str] | None:
    """argv to auto-install the sandbox's missing dependencies on *platform*.

    Returns ``None`` when nothing can be driven automatically: the platform
    isn't macOS/Linux, Homebrew isn't installed on macOS, or no known package
    manager is on PATH on Linux — in every such case the caller should fall
    back to :class:`SystemDependencyStatus`'s ``manual_install_hint``.

    The actual package-manager detection is generic (see
    ``_utils.system_dependency_utils``); only the sandbox's specific package
    list per platform lives here. Homebrew is invoked without ``sudo`` (it
    refuses to run as root); the Linux managers all require it, so the caller
    must run this in a real terminal (not with output captured) so ``sudo``
    can prompt for a password the normal, secure way.
    """
    if platform == "macos":
        return build_macos_brew_install_command(["ripgrep"])
    if platform == "linux":
        return build_linux_install_command(["bubblewrap", "socat", "ripgrep"])
    return None


_RUNNER_SRC = Path(__file__).parent / "_runner.py"


# Domains the sandboxed ``gh`` CLI (see ``execute_cli_command``) is permitted to
# reach. Scoped narrowly to GitHub's own hosts rather than opening network
# access generally; every other sandboxed command remains fully offline.
_CLI_ALLOWED_NETWORK_DOMAINS: tuple[str, ...] = (
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "objects.githubusercontent.com",
    "raw.githubusercontent.com",
)

# Allowlist of host environment variables propagated into the sandboxed
# subprocess. The host env carries secrets (API keys, DB URLs, cloud creds);
# model-generated code runs via ``exec`` inside the runner and could otherwise
# read and exfiltrate them through the workspace. Only variables needed for the
# interpreter, the sandbox wrapper (bwrap/sandbox-exec), and locale/temp
# resolution are forwarded.
_ENV_PASSTHROUGH: tuple[str, ...] = (
    # POSIX essentials
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    # Windows essentials (mock/dev hosts only)
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "PATHEXT",
    "COMSPEC",
    "USERPROFILE",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
)


# ---------------------------------------------------------------------------
# Accelerator (GPU/NPU) passthrough (Linux only; opt-in via ``[deep-learning]``)
# ---------------------------------------------------------------------------
#
# ``sandbox_runtime`` (0.2.x) unconditionally runs ``bwrap --dev /dev`` on
# Linux, which mounts only standard nodes (null, zero, random) and drops real
# accelerator devices (/dev/nvidia*, /dev/dri/*, /dev/accel/*) with no config
# knob to restore them. We work around that below by activating only when a
# ``[deep-learning]`` package (torch/jax/transformers/sentence_transformers)
# is importable, since it hands sandboxed code direct ioctl access to the
# host kernel's accelerator driver (a real CVE surface, e.g.
# nvidia-container-toolkit) with no resource limiting the way CPU/memory
# have — hence the one-time warning below. Only compute-capable nodes are
# ever bound, never /dev/dri/card* (display/KMS).
#
# Detection lives in ``opendatasci._utils`` (pure, reusable); the
# bwrap-specific injection and its one-time warning stay here. The warning
# flag is module-scoped so it fires once per process, not once per sandbox
# instance.

_ACCELERATOR_WARNING_EMITTED: bool = False


def _inject_accelerator_devices(wrapped_command: str) -> str:
    """Append ``--dev-bind`` for discovered accelerator devices to a wrapped bwrap command.

    ``sandbox_runtime`` always emits a literal ``--dev /dev`` token on Linux
    (see module docstring), which this appends immediately after — bwrap
    applies bind mounts in argument order, so a later ``--dev-bind`` for a
    specific node inside ``/dev`` takes effect over the synthetic devtmpfs
    ``--dev`` created for that same path. If the sandbox isn't restricting
    anything (no ``--dev /dev`` in the command — e.g. no filesystem/network
    config was passed), the command is returned unchanged: an unrestricted
    bwrap already sees the real ``/dev``, and a raw (non-bwrap) command has
    nothing to inject into.
    """
    devices = discover_accelerator_devices()
    if not devices:
        return wrapped_command

    marker = "--dev /dev"
    if marker not in wrapped_command:
        return wrapped_command

    global _ACCELERATOR_WARNING_EMITTED
    if not _ACCELERATOR_WARNING_EMITTED:
        _ACCELERATOR_WARNING_EMITTED = True
        warnings.warn(
            "opendatasci: the sandbox now has access to this machine's accelerator "
            f"device(s) ({', '.join(devices)}), needed for the [deep-learning] extra to run "
            "deep learning on your GPU/NPU. This expands what sandboxed code can "
            "reach on your machine. Reinstall without the [deep-learning] extra to disable it.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning("Accelerator passthrough active: bind-mounting %s into the sandbox", devices)

    bind_args = " ".join(f"--dev-bind {shlex.quote(d)} {shlex.quote(d)}" for d in devices)
    return wrapped_command.replace(marker, f"{marker} {bind_args}", 1)


# ---------------------------------------------------------------------------
# Process-global manager lifecycle
# ---------------------------------------------------------------------------
#
# ``SandboxManager`` is module-level singleton state inside ``sandbox_runtime``:
# the network proxies and (on Linux) the network bridge are shared by every
# sandbox in the process. It must therefore be initialized exactly once per
# process and torn down only at process exit — a job the library already
# performs via its own ``atexit``/signal handlers. An individual session-scoped
# sandbox must never call ``SandboxManager.reset()``, or it would rip the shared
# infrastructure out from under its concurrently running siblings.
_manager_lock = asyncio.Lock()
_manager_initialized = False


async def _ensure_manager_initialized(config: SandboxRuntimeConfig) -> None:
    """Initialize the global ``SandboxManager`` once per process (idempotent)."""
    global _manager_initialized
    if _manager_initialized:
        return
    async with _manager_lock:
        if _manager_initialized:
            return
        await SandboxManager.initialize(config)
        _manager_initialized = True


def _base_sandbox_env() -> dict[str, str]:
    """Return a minimal, allowlisted copy of the host environment."""
    return {key: os.environ[key] for key in _ENV_PASSTHROUGH if key in os.environ}


class SRTSandbox(BaseSandbox):
    """Session-scoped sandbox powered by Anthropic's Sandbox Runtime (SRT).

    Executes Python snippets and allowlisted TUI commands in an OS-level
    sandbox — no Docker or remote container is required. Every :meth:`execute`
    spins up a fresh interpreter with an empty namespace: no Python-level
    state (variables, results) carries over from one call to the next, even
    within the same instance. Only the workspace filesystem persists across
    calls. This keeps executions fully independent, so concurrent calls on
    the same instance (e.g. a foreground call racing a background one) run
    genuinely in parallel rather than serializing behind each other.

    ``reset()`` and construction-time bookkeeping (copying the runner script
    once, initializing the process-global ``SandboxManager``) are the only
    per-instance state that could otherwise race; :meth:`_ensure_initialized`
    guards that under its own small lock.
    """

    def __init__(
        self,
        workspace_path: Path | None = None,
        *,
        command_timeout: int | None = None,
    ) -> None:
        self._workspace_path = workspace_path
        self._command_timeout = (
            command_timeout if command_timeout is not None else _DEFAULT_COMMAND_TIMEOUT
        )

        self._session_dir = Path(tempfile.mkdtemp(prefix="opendatasci_srt_"))
        self._runner_path = self._session_dir / "runner.py"

        self._history: list[SandboxExecResult] = []
        self._sandbox_config: SandboxRuntimeConfig | None = None
        self._cli_sandbox_config: SandboxRuntimeConfig | None = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Sandbox protocol
    # ------------------------------------------------------------------

    async def execute(
        self,
        code: str,
        on_stdout_line: Callable[[str], Awaitable[None]] | None = None,
    ) -> SandboxExecResult:
        try:
            await self._ensure_initialized()

            workspace = str(self._workspace_path or self._session_dir)
            env = {
                **_base_sandbox_env(),
                "OPENDATASCI_CODE_B64": base64.b64encode(code.encode("utf-8")).decode("ascii"),
                "OPENDATASCI_WORKSPACE": workspace,
            }

            command = f"{shlex.quote(sys.executable)} {shlex.quote(str(self._runner_path))}"
            wrapped = await SandboxManager.wrap_with_sandbox(
                command, custom_config=self._make_config()
            )
            if is_deep_learning_extra_active():
                wrapped = _inject_accelerator_devices(wrapped)
            stdout_str, stderr_str, _ = await self._run_subprocess(
                wrapped, env=env, cwd=workspace, on_stdout_line=on_stdout_line
            )

            payload = self._parse_runner_payload(stdout_str, stderr_str)
            stdout = payload.get("stdout", "")

            if payload.get("success"):
                result = SandboxExecResult(
                    success=True,
                    output=payload.get("result"),
                    stdout=stdout,
                    code=code,
                )
            else:
                result = SandboxExecResult(
                    success=False,
                    error=payload.get("error", "Unknown execution error"),
                    stdout=stdout,
                    code=code,
                )
        except TimeoutError:
            result = self._fail(
                code,
                f"TimeoutError: execution timed out after {self._command_timeout}s",
            )
        except Exception as exc:
            result = self._fail(code, f"SRTError: {exc}\n{traceback.format_exc()}")

        self._history.append(result)
        return result

    async def execute_cli(self, command: str) -> SandboxExecResult:
        error = validate_cli_command(command)
        if error:
            result = self._fail(command, f"Error: {error}")
            self._history.append(result)
            return result

        try:
            await self._ensure_initialized()

            workspace = str(self._workspace_path or self._session_dir)
            wrapped = await SandboxManager.wrap_with_sandbox(
                command, custom_config=self._make_cli_config()
            )
            stdout_str, stderr_str, exit_code = await self._run_subprocess(
                wrapped, env=_base_sandbox_env(), cwd=workspace
            )

            combined = "\n".join(filter(None, [stdout_str, stderr_str]))
            success = exit_code == 0
            result = SandboxExecResult(
                success=success,
                stdout=combined,
                error=None if success else f"Command failed (exit {exit_code})",
                code=command,
            )
        except TimeoutError:
            result = self._fail(
                command,
                f"TimeoutError: command timed out after {self._command_timeout}s",
            )
        except Exception as exc:
            result = self._fail(command, f"SRTCLIError: {exc}")

        self._history.append(result)
        return result

    def get_history(self) -> list[SandboxExecResult]:
        return list(self._history)

    def reset(self) -> None:
        """Clear the execution history. No Python-level state to wipe: every
        ``execute()`` call already starts from an empty namespace."""
        self._history.clear()

    async def close(self) -> None:
        # The SandboxManager is a process-global singleton shared with every
        # concurrent sibling sandbox; tearing it down here would break them.
        # Its own atexit/signal handlers perform the single process-level
        # teardown. We own only our session directory.
        self._initialized = False
        shutil.rmtree(self._session_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        # Concurrent execute()/execute_cli() calls no longer serialize behind
        # a shared lock, so this one-time bookkeeping (runner copy + flag)
        # needs its own guard against a race on the very first call.
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await _ensure_manager_initialized(self._make_config())
            shutil.copy2(_RUNNER_SRC, self._runner_path)
            self._initialized = True

    def _make_config(self) -> SandboxRuntimeConfig:
        if self._sandbox_config is None:
            workspace = str(self._workspace_path or self._session_dir)
            self._sandbox_config = SandboxRuntimeConfig(
                network={"allowed_domains": [], "denied_domains": []},
                filesystem={
                    "deny_read": find_maybe_sensitive_paths(),
                    "allow_write": [workspace, str(self._session_dir)],
                    "deny_write": [],
                },
            )
        return self._sandbox_config

    def _make_cli_config(self) -> SandboxRuntimeConfig:
        """Config for :meth:`execute_cli`: identical to :meth:`_make_config` except
        network access is scoped to GitHub's hosts, for the ``gh`` CLI.

        Kept separate from ``_make_config`` (used by :meth:`execute`) so Python
        code execution remains fully network-isolated; only CLI commands (which
        pass through the ``ALLOWED_CLI_COMMANDS`` allowlist) get this narrower
        network exception.
        """
        if self._cli_sandbox_config is None:
            workspace = str(self._workspace_path or self._session_dir)
            self._cli_sandbox_config = SandboxRuntimeConfig(
                network={
                    "allowed_domains": list(_CLI_ALLOWED_NETWORK_DOMAINS),
                    "denied_domains": [],
                },
                filesystem={
                    "deny_read": find_maybe_sensitive_paths(),
                    "allow_write": [workspace, str(self._session_dir)],
                    "deny_write": [],
                },
            )
        return self._cli_sandbox_config

    async def _run_subprocess(
        self,
        command: str,
        env: dict[str, str],
        cwd: str,
        on_stdout_line: Callable[[str], Awaitable[None]] | None = None,
    ) -> tuple[str, str, int]:
        # Launch the wrapped command in its own process group/session so a
        # timeout can signal the *entire* tree (shell → bwrap/sandbox-exec →
        # python), not just the top-level shell, which would otherwise leak the
        # sandbox and its python child as orphans.
        spawn_kwargs: dict[str, Any]
        if sys.platform != "win32":
            spawn_kwargs = {"start_new_session": True}
        else:
            spawn_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
                limit=_STDOUT_STREAM_LIMIT,
                **spawn_kwargs,
            )
            # Never None: both stdout and stderr are always requested as PIPE
            # above, so asyncio always attaches a StreamReader to each. Bound
            # to locals (rather than narrowed via `assert proc.stdout is not
            # None`) so the type-checker can see the non-None type inside the
            # nested closure below, which it can't infer through `proc.stdout`.
            assert proc.stdout is not None
            assert proc.stderr is not None
            proc_stdout, proc_stderr = proc.stdout, proc.stderr
            try:
                # stdout is drained line-by-line (forwarding each line to
                # on_stdout_line as it arrives, when given) while stderr is
                # drained in full in parallel via asyncio.gather -- reading
                # only one of the two pipes here would risk the classic
                # subprocess deadlock (the unread pipe fills and blocks the
                # child), which is exactly what proc.communicate() avoids
                # internally. The trailing proc.wait() ensures returncode is
                # populated, matching communicate()'s own guarantee; it's part
                # of the same timeout budget below, not an extra one, so total
                # time bound stays exactly self._command_timeout as before.
                async def _drain_and_wait() -> tuple[str, bytes]:
                    stdout_str, stderr_bytes = await asyncio.gather(
                        self._drain_stdout(proc_stdout, on_stdout_line),
                        proc_stderr.read(),
                    )
                    await proc.wait()
                    return stdout_str, stderr_bytes

                stdout_str, stderr_bytes = await asyncio.wait_for(
                    _drain_and_wait(), timeout=self._command_timeout
                )
            except asyncio.CancelledError:
                # Cancellation (e.g. via stop_task) throws in here without ever
                # raising TimeoutError, so without this handler the wrapped
                # subprocess (shell -> bwrap/sandbox-exec -> python) would be
                # orphaned rather than killed.
                await self._terminate_process_tree(proc)
                raise
            returncode = proc.returncode
            if returncode is None:
                # Should not happen after proc.wait(); surface it as a failure
                # rather than masking it as success (exit 0).
                logger.warning(
                    "Subprocess returncode is None after proc.wait(); treating as failure"
                )
                returncode = -1
            return (
                stdout_str.strip(),
                stderr_bytes.decode("utf-8", errors="replace").strip(),
                returncode,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                await self._terminate_process_tree(proc)
            raise TimeoutError(f"Command timed out after {self._command_timeout}s: {command!r}")

    async def _drain_stdout(
        self,
        stream: asyncio.StreamReader,
        on_stdout_line: Callable[[str], Awaitable[None]] | None,
    ) -> str:
        """Read *stream* to EOF, forwarding each line to *on_stdout_line* as it
        arrives, and return the full decoded text (equivalent to what
        ``communicate()`` would have returned for stdout).

        Forwarding stops at the runner's PAYLOAD_SENTINEL-prefixed line (the
        terminal JSON payload, see ``_parse_runner_payload``) -- that line is
        the result, not progress, so a caller streaming into an activity log
        never sees it as one more line of "output". For non-runner commands
        (``execute_cli``) the sentinel never appears, so every line is
        forwarded when *on_stdout_line* is given.
        """
        chunks: list[str] = []
        payload_seen = False
        while True:
            raw_line = await stream.readline()
            if not raw_line:
                break
            text = raw_line.decode("utf-8", errors="replace")
            chunks.append(text)
            if payload_seen:
                continue
            if PAYLOAD_SENTINEL in text:
                payload_seen = True
            elif on_stdout_line is not None:
                stripped = text[:-1] if text.endswith("\n") else text
                if stripped:
                    await on_stdout_line(stripped)
        return "".join(chunks)

    async def _terminate_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Kill the subprocess's whole group and reap it, so no orphans or
        unreaped transports remain."""
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await proc.wait()
        except Exception:
            logger.exception("Failed to reap timed-out subprocess")

    def _parse_runner_payload(self, raw_stdout: str, raw_stderr: str = "") -> dict[str, Any]:
        # The runner prefixes its final JSON payload with PAYLOAD_SENTINEL, so
        # look for the *last* occurrence (rfind, not find) -- if user code
        # itself ever printed that literal string, only the runner's own
        # trailing write is the real payload. This also lets a streaming
        # reader (_drain_stdout) recognise the payload line the moment it
        # arrives, without waiting for EOF to try "is this JSON?" on every line.
        if not raw_stdout:
            detail = f"stderr: {raw_stderr}" if raw_stderr else "no output"
            raise ValueError(f"SRT runner returned no stdout payload ({detail}).")

        idx = raw_stdout.rfind(PAYLOAD_SENTINEL)
        if idx == -1:
            raise ValueError(f"SRT runner output carried no payload sentinel: {raw_stdout}")

        payload_text = raw_stdout[idx + len(PAYLOAD_SENTINEL) :].strip()
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SRT runner payload was not valid JSON: {payload_text}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"SRT runner payload was not a JSON object: {payload_text}")
        return parsed

    def _fail(self, code: str, error: str) -> SandboxExecResult:
        return SandboxExecResult(success=False, error=error, stdout="", code=code)

    def __del__(self) -> None:
        # Warn on un-closed sandboxes, but always reclaim the temp dir — it is
        # allocated unconditionally in __init__, so even a sandbox that was never
        # run (or never closed) must not leak it.
        if getattr(self, "_initialized", False):
            warnings.warn(
                f"{self.__class__.__name__} was not properly closed; always use SRTSandboxFactory as a context manager",
                ResourceWarning,
                source=self,
            )
        session_dir = getattr(self, "_session_dir", None)
        if session_dir is not None:
            shutil.rmtree(session_dir, ignore_errors=True)


class SRTSandboxFactory(BaseSandboxFactory):
    """Factory that creates :class:`SRTSandbox` instances as async context managers.

    Usage::

        factory = SRTSandboxFactory()
        async with factory.create(workspace_path=path) as sandbox:
            result = await sandbox.execute(code)
        # sandbox is closed here

    Args:
        command_timeout: Maximum seconds a single sandbox command may run
            before being killed.  Forwarded verbatim to every
            :class:`SRTSandbox` created by :meth:`create`.

    Raises:
        RuntimeError: From :meth:`create`, if the host is missing a required
            native sandbox dependency (e.g. bubblewrap/socat on Linux,
            ripgrep on macOS) or the platform is unsupported (e.g. Windows).
    """

    def __init__(self, *, command_timeout: int | None = None) -> None:
        self._command_timeout = command_timeout

    @asynccontextmanager
    async def create(self, workspace_path: Path | None = None) -> AsyncIterator[SRTSandbox]:
        check_sandbox_dependencies()
        sandbox = SRTSandbox(
            workspace_path=workspace_path,
            command_timeout=self._command_timeout,
        )
        try:
            yield sandbox
        finally:
            await sandbox.close()
