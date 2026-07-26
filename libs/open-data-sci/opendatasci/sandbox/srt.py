"""SRT-backed sandbox for OpenDataSci sessions.

Platform support: the underlying Sandbox Runtime only sandboxes on macOS and
Linux. Windows is unsupported: ``sandbox_runtime`` imports the Unix-only
``resource`` module at import time, so this module fails to import at all on
Windows; this class is therefore exercised only under mocks on such hosts.

GPU passthrough (Linux only, opt-in via the ``[deep-learning]`` extra):
``sandbox_runtime`` (0.2.x) unconditionally runs ``bwrap --dev /dev`` on
Linux, which mounts a fresh devtmpfs containing only the standard nodes
(``null``, ``zero``, ``random``, etc.) — no ``/dev/nvidia*`` or
``/dev/dri/*``. It exposes no device-related config to work around this
(only ``network`` and ``filesystem``), and any ``/dev/*`` path handed to it
via ``filesystem.allow_write`` is silently dropped (its own comment assumes
``--dev /dev`` already covers device access, which it doesn't for real
devices). Verified experimentally (bwrap 0.8, WSL2 + RTX 5070 Ti): appending
an explicit ``--dev-bind`` for the GPU's device node after ``--dev /dev`` is
sufficient to restore full accelerator access with no other changes needed.

``SRTSandbox`` does this itself, gated on whether a deep-learning package
(``torch``, ``jax``, ``transformers``, ``sentence_transformers``) is
importable — see ``_is_host_dl_extra_active`` and
``_inject_gpu_devices``. It is off unless one of those is installed, and it
logs a one-time warning when it activates, because it is a materially
different risk than the rest of this sandbox's isolation:

- It hands sandboxed code direct ``ioctl`` access to the **host kernel's**
  GPU driver, not just a contained resource. NVIDIA's driver ioctl surface
  has a real CVE history for exactly this class of issue (e.g. the
  ``nvidia-container-toolkit`` CVEs), so a driver bug reachable from
  sandboxed code is a host-kernel bug, not a sandbox-contained one.
- There is no GPU equivalent of ``ResourceLimitsConfig`` (which caps
  CPU/memory/file-size) — nothing stops sandboxed code from exhausting GPU
  memory or compute, a DoS against anything else using that GPU.
- Only compute-capable nodes are ever bound: ``/dev/nvidia*`` and
  ``/dev/dri/renderD*``. ``/dev/dri/card*`` (display/KMS ioctls) is never
  exposed, deliberately.

macOS/Metal passthrough is not implemented here: it needs Seatbelt profile
changes (allowing the ``AGXDeviceUserClient`` IOKit class and the
``com.apple.MTLCompilerService`` mach-lookup), matching the shape of
upstream's unmerged ``allowGPU`` proposal
(anthropic-experimental/sandbox-runtime#181), but that's unverified here (no
macOS hardware available) — deep learning code on macOS still runs on CPU.
"""

import asyncio
import base64
import glob
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
from sandbox_runtime.utils.platform import get_platform

from opendatasci._utils.package_extras_utils import is_host_dl_extra_active
from opendatasci._utils.gpu_utils import discover_gpu_devices
from opendatasci.sandbox.base import (
    BaseSandbox,
    BaseSandboxFactory,
    SandboxExecResult,
    validate_cli_command,
)

logger = logging.getLogger(__name__)

_DEFAULT_COMMAND_TIMEOUT = 43200  # 12 hours

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
    if SandboxManager.check_dependencies():
        return

    platform = get_platform()
    if not SandboxManager.is_supported_platform(platform):
        raise RuntimeError(
            f"OpenDataSci's sandbox is not supported on platform '{platform}'. "
            "Only macOS and Linux are supported."
        )

    install_hint = _INSTALL_HINTS.get(platform)
    required = (
        "ripgrep (rg)" if platform == "macos" else "ripgrep (rg), bubblewrap (bwrap), and socat"
    )
    message = f"Missing required sandbox dependencies for {platform}: {required}."
    if install_hint:
        message += f" Install with:\n  {install_hint}"
    raise RuntimeError(message)


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
# GPU passthrough (Linux only; opt-in via the ``[deep-learning]`` extra)
# ---------------------------------------------------------------------------
#
# See the module docstring for what this does and why it's gated. Detection
# (``is_host_dl_extra_active``/``discover_gpu_devices``, in
# ``opendatasci._utils.package_extras_utils``) is pure and reusable; injecting the bwrap
# args and emitting the one-time warning stay here since they're tied to this
# module's specific wrapped-command format. The warning is cached at module
# scope so it fires once per process, not once per sandbox instance.


_GPU_WARNING_EMITTED: bool = False


def _inject_gpu_devices(wrapped_command: str) -> str:
    """Append ``--dev-bind`` for discovered GPU devices to a wrapped bwrap command.

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
    devices = discover_gpu_devices()
    if not devices:
        return wrapped_command

    marker = "--dev /dev"
    if marker not in wrapped_command:
        return wrapped_command

    global _GPU_WARNING_EMITTED
    if not _GPU_WARNING_EMITTED:
        _GPU_WARNING_EMITTED = True
        warnings.warn(
            "opendatasci: a deep-learning package is installed, so GPU compute "
            f"device(s) are being bind-mounted into the sandbox: {', '.join(devices)}. "
            "This gives sandboxed code direct ioctl access to the host kernel's GPU "
            "driver — a materially different risk than the sandbox's filesystem/"
            "network isolation (real CVE history for GPU driver ioctl surfaces; no "
            "GPU-side resource limiting). See opendatasci/sandbox/srt.py module "
            "docstring for details. Uninstall torch/jax/transformers/"
            "sentence-transformers to disable this.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            "GPU passthrough active: bind-mounting %s into the sandbox "
            "(deep-learning package detected)",
            devices,
        )

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
        self._command_timeout = (
            command_timeout if command_timeout is not None else _DEFAULT_COMMAND_TIMEOUT
        )

        self._session_dir = Path(tempfile.mkdtemp(prefix="opendatasci_srt_"))
        self._state_path = self._session_dir / "state.pkl"
        self._runner_path = self._session_dir / "runner.py"

        self._history: list[SandboxExecResult] = []
        self._results: dict[str, str] = {}
        self._var_info: dict[str, str] = {}
        self._sandbox_config: SandboxRuntimeConfig | None = None
        self._cli_sandbox_config: SandboxRuntimeConfig | None = None
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
                if is_host_dl_extra_active():
                    wrapped = _inject_gpu_devices(wrapped)
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

    def _credential_deny_paths(self) -> list[str]:
        """Builds paths to deny IO operations on common credential paths. Not comprehensive."""
        home = os.path.expanduser("~")
        candidates: list[str] = []

        # Every top-level dotfile/dotdir directly under the home directory:
        # ``~/.ssh``, ``~/.aws``, ``~/.netrc``, ``~/.config`` (and everything
        # under it), etc. Blanket-denies by naming convention rather than an
        # explicit allowlist of known tools, so it also catches ones we haven't
        # enumerated. Only reaches one level deep (``~/.foo``, not
        # ``~/projects/.env``).
        candidates.extend(glob.glob(os.path.join(home, ".*")))

        # macOS Keychain: the platform's primary credential store. Not a dotfile
        # (lives under ``~/Library``), so the glob above never reaches it.
        candidates.append(os.path.join(home, "Library", "Keychains"))

        # macOS browser/WebKit cookie jars, which commonly hold live session
        # tokens and likewise live outside ``~/Library``'s non-dot prefix.
        candidates.append(os.path.join(home, "Library", "Cookies"))
        candidates.append(os.path.join(home, "Library", "HTTPStorages"))

        return [os.path.realpath(path) for path in candidates if os.path.exists(path)]

    def _make_config(self) -> SandboxRuntimeConfig:
        if self._sandbox_config is None:
            workspace = str(self._workspace_path or self._session_dir)
            self._sandbox_config = SandboxRuntimeConfig(
                network={"allowed_domains": [], "denied_domains": []},
                filesystem={
                    "deny_read": self._credential_deny_paths(),
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
                    "deny_read": self._credential_deny_paths(),
                    "allow_write": [workspace, str(self._session_dir)],
                    "deny_write": [],
                },
            )
        return self._cli_sandbox_config

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
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._command_timeout,
                )
            except asyncio.CancelledError:
                # Cancellation (e.g. via cancel_task) throws in here without ever
                # raising TimeoutError, so without this handler the wrapped
                # subprocess (shell -> bwrap/sandbox-exec -> python) would be
                # orphaned rather than killed.
                await self._terminate_process_tree(proc)
                raise
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
