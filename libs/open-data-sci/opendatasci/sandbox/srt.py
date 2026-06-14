"""SRT-backed sandbox for OpenDataSci sessions.

Platform support: the underlying Sandbox Runtime only sandboxes on macOS and
Linux. On other platforms (e.g. Windows) ``SandboxManager.initialize`` raises a
``RuntimeError`` at first use; this class is therefore exercised only under mocks
on such hosts.
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
from pathlib import Path
from typing import Any, AsyncIterator

from sandbox_runtime import SandboxManager, SandboxRuntimeConfig

from opendatasci.sandbox.base import (
    BaseSandbox,
    BaseSandboxFactory,
    SandboxExecResult,
    validate_cli_command,
)

logger = logging.getLogger(__name__)

_RUNNER_SRC = Path(__file__).parent / "_runner.py"

# Sensitive host locations the sandbox must never expose to model-generated
# code. These are expanded to absolute, symlink-resolved paths before being
# handed to SRT, which resolves deny rules relative to the workspace cwd and
# does *not* expand ``~`` itself.
_SENSITIVE_READ_PATHS: tuple[str, ...] = (
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config/gcloud",
    "~/.kube",
    "~/.docker",
    "~/.netrc",
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
    sandbox — no Docker or remote container is required.  Python state
    (variables, results) is preserved across calls within the same instance.

    Each :meth:`execute`/:meth:`execute_cli`/:meth:`reset` call is serialized by
    a per-instance lock so overlapping calls cannot interleave their
    read-modify-write of the on-disk ``state.pkl``.

    Note: every :meth:`execute` spins up a fresh interpreter that unpickles the
    entire namespace, runs, and re-pickles it. This keeps executions hermetic
    but makes cost O(state) per call; a persistent-kernel runner would remove
    that overhead and is the natural next step for long interactive sessions.
    """

    def __init__(
        self,
        workspace_path: Path | None = None,
        *,
        command_timeout: int | None = None,
    ) -> None:
        self._workspace_path = workspace_path
        self._command_timeout = command_timeout if command_timeout is not None else 1800

        self._session_dir = Path(tempfile.mkdtemp(prefix="opendatasci_srt_"))
        self._state_path = self._session_dir / "state.pkl"
        self._runner_path = self._session_dir / "runner.py"

        self._history: list[SandboxExecResult] = []
        self._results: dict[str, str] = {}
        self._var_info: dict[str, str] = {}
        self._sandbox_config: SandboxRuntimeConfig | None = None
        self._initialized = False
        # Set by reset(); consumed under _lock at the start of the next execute
        # so the on-disk wipe happens inside the serialized critical section.
        self._reset_pending = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Sandbox protocol
    # ------------------------------------------------------------------

    async def execute(self, code: str) -> SandboxExecResult:
        async with self._lock:
            try:
                await self._ensure_initialized()

                if self._reset_pending:
                    # Deleting the state file clears both variables and saved
                    # results (the latter live inside the pickle under
                    # RESULTS_KEY), so a single unlink is a complete wipe.
                    self._state_path.unlink(missing_ok=True)
                    self._reset_pending = False

                workspace = str(self._workspace_path or self._session_dir)
                env = {
                    **_base_sandbox_env(),
                    "OPENDATASCI_CODE_B64": base64.b64encode(code.encode("utf-8")).decode("ascii"),
                    "OPENDATASCI_STATE_PATH": str(self._state_path),
                    "OPENDATASCI_WORKSPACE": workspace,
                }

                command = f"{shlex.quote(sys.executable)} {shlex.quote(str(self._runner_path))}"
                wrapped = await SandboxManager.wrap_with_sandbox(
                    command, custom_config=self._make_config()
                )
                stdout_str, stderr_str, _ = await self._run_subprocess(
                    wrapped, env=env, cwd=workspace
                )

                payload = self._parse_runner_payload(stdout_str, stderr_str)
                self._var_info.update(payload.get("var_info", {}))
                self._results.update(payload.get("saved_results", {}))

                stdout = payload.get("stdout", "")
                dropped_vars = payload.get("dropped_vars", [])
                if dropped_vars:
                    warning = f"Warning: variable(s) not persisted (not picklable): {', '.join(dropped_vars)}"
                    stdout = f"{stdout}\n{warning}" if stdout else warning

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

        async with self._lock:
            try:
                await self._ensure_initialized()

                workspace = str(self._workspace_path or self._session_dir)
                wrapped = await SandboxManager.wrap_with_sandbox(
                    command, custom_config=self._make_config()
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
        # Clear the in-memory views eagerly; defer the on-disk state wipe to the
        # next execute so it runs inside the serialized critical section (and
        # cannot clobber an in-flight execution's pickle write). The two views
        # only diverge in the window before the next execute, which itself
        # reconciles them — no public read path observes the difference.
        self._history.clear()
        self._var_info.clear()
        self._results.clear()
        self._reset_pending = True

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
        # Caller holds ``self._lock``, so the per-instance bookkeeping below
        # (runner copy + flag) cannot race a concurrent first call.
        if self._initialized:
            return
        await _ensure_manager_initialized(self._make_config())
        shutil.copy2(_RUNNER_SRC, self._runner_path)
        self._initialized = True

    def _make_config(self) -> SandboxRuntimeConfig:
        if self._sandbox_config is None:
            workspace = str(self._workspace_path or self._session_dir)
            deny_read = [
                os.path.realpath(os.path.expanduser(path)) for path in _SENSITIVE_READ_PATHS
            ]
            self._sandbox_config = SandboxRuntimeConfig(
                network={"allowed_domains": [], "denied_domains": []},
                filesystem={
                    "deny_read": deny_read,
                    "allow_write": [workspace, str(self._session_dir)],
                    "deny_write": [],
                },
            )
        return self._sandbox_config

    async def _run_subprocess(
        self, command: str, env: dict[str, str], cwd: str
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
                **spawn_kwargs,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._command_timeout,
            )
            returncode = proc.returncode
            if returncode is None:
                # Should not happen after communicate(); surface it as a failure
                # rather than masking it as success (exit 0).
                logger.warning(
                    "Subprocess returncode is None after communicate(); treating as failure"
                )
                returncode = -1
            return (
                stdout_bytes.decode("utf-8", errors="replace").strip(),
                stderr_bytes.decode("utf-8", errors="replace").strip(),
                returncode,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                await self._terminate_process_tree(proc)
            raise TimeoutError(f"Command timed out after {self._command_timeout}s: {command!r}")

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
        # The runner emits its result as a single trailing JSON line, so we scan
        # bottom-up: this survives arbitrary user ``print()`` output captured
        # above it. (A user subprocess writing raw bytes directly to fd 1 *after*
        # the payload line could still corrupt parsing — an accepted edge case.)
        if not raw_stdout:
            detail = f"stderr: {raw_stderr}" if raw_stderr else "no output"
            raise ValueError(f"SRT runner returned no stdout payload ({detail}).")

        for line in reversed(raw_stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        raise ValueError(f"SRT runner output was not JSON: {raw_stdout}")

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
    """

    def __init__(self, *, command_timeout: int | None = None) -> None:
        self._command_timeout = command_timeout

    @asynccontextmanager
    async def create(self, workspace_path: Path | None = None) -> AsyncIterator[SRTSandbox]:
        sandbox = SRTSandbox(
            workspace_path=workspace_path,
            command_timeout=self._command_timeout,
        )
        try:
            yield sandbox
        finally:
            await sandbox.close()
