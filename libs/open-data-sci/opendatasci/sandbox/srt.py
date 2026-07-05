"""SRT-backed sandbox for OpenDataSci sessions.

Platform support: the underlying Sandbox Runtime only sandboxes on macOS and
Linux. Windows is unsupported: ``sandbox_runtime`` imports the Unix-only
``resource`` module at import time, so this module fails to import at all on
Windows; this class is therefore exercised only under mocks on such hosts.
"""

import asyncio
import base64
import json
import logging
import os
import platform
import re
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
from typing import Any, AsyncIterator

from sandbox_runtime import SandboxManager, SandboxRuntimeConfig
from sandbox_runtime.utils.platform import get_platform

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


@dataclass
class _HardwareResources:
    """One free-form description per hardware aspect of the probed machine."""

    platform: str
    cpu: str
    ram: str
    disk: str
    gpu: str
    accelerators: str


class _HardwareResourcesProbe:
    """Best-effort discovery of the hardware available to SRT-sandboxed code.

    SRT executes code on the local machine, so probing the host directly
    describes exactly the hardware that sandboxed code will see. All probes
    are read-only, run fixed command lines (never model input), and degrade
    to an explanatory message when a command is missing or fails.
    """

    _PROBE_TIMEOUT = 15  # seconds per external command

    # lspci device-class keywords that indicate a GPU.
    _GPU_PCI_PATTERN = re.compile(r"vga|3d controller|display controller", re.IGNORECASE)
    # lspci keywords that indicate an NPU or another dedicated accelerator.
    _ACCEL_PCI_PATTERN = re.compile(
        r"neural|npu|vpu|tpu|habana|gaudi|coral|fpga|processing accelerator", re.IGNORECASE
    )

    def __init__(self, workspace_path: Path | None = None) -> None:
        self._workspace_path = workspace_path

    async def collect(self) -> _HardwareResources:
        """Probe every hardware aspect concurrently and return the results."""
        cpu, ram, disk, gpu, accel = await asyncio.gather(
            self._cpu_section(),
            self._ram_section(),
            self._disk_section(),
            self._gpu_section(),
            self._accelerator_section(),
        )
        return _HardwareResources(
            platform=platform.platform(),
            cpu=cpu,
            ram=ram,
            disk=disk,
            gpu=gpu,
            accelerators=accel,
        )

    # -- probing primitives -------------------------------------------------

    async def _run_command(self, *argv: str) -> str | None:
        """Run *argv* and return its stripped stdout, or ``None`` on any failure."""
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self._PROBE_TIMEOUT
            )
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            return None
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        text = stdout_bytes.decode("utf-8", errors="replace").strip()
        return text or None

    @staticmethod
    def _read_text(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip() or None
        except OSError:
            return None

    @staticmethod
    def _format_bytes(num_bytes: int) -> str:
        return f"{num_bytes / 1024**3:.1f} GiB"

    async def _lspci_lines(self, pattern: "re.Pattern[str]") -> list[str]:
        output = await self._run_command("lspci")
        if not output:
            return []
        return [line for line in output.splitlines() if pattern.search(line)]

    # -- sections ------------------------------------------------------------

    async def _cpu_section(self) -> str:
        lines = [
            f"Architecture: {platform.machine() or 'unknown'}",
            f"Logical cores: {os.cpu_count() or 'unknown'}",
        ]
        if hasattr(os, "getloadavg"):
            try:
                load_1, load_5, load_15 = os.getloadavg()
                lines.append(
                    f"Load average (1/5/15 min): {load_1:.2f}, {load_5:.2f}, {load_15:.2f}"
                )
            except OSError:
                pass
        if sys.platform == "darwin":
            # Dumps the whole machdep.cpu subtree: brand string, core counts,
            # and capability flags (features/leaf7_features on Intel;
            # hw.optional.* covers arm64 features).
            for probe in (
                ("sysctl", "machdep.cpu"),
                ("sysctl", "hw.physicalcpu", "hw.logicalcpu"),
                ("sysctl", "hw.optional"),
            ):
                output = await self._run_command(*probe)
                if output:
                    lines.append(output)
        elif sys.platform.startswith("linux"):
            # lscpu includes the model name, core/socket topology, frequencies,
            # caches, and the full capability flag list (AVX/AMX/SVE, …).
            lscpu = await self._run_command("lscpu")
            if lscpu:
                lines.append(lscpu)
            else:
                cpuinfo = self._read_text(Path("/proc/cpuinfo"))
                if cpuinfo:
                    # First processor block is enough: the flags repeat per core.
                    lines.append(cpuinfo.split("\n\n", 1)[0])
        elif platform.processor():
            lines.append(f"Processor: {platform.processor()}")
        return "\n".join(lines)

    async def _ram_section(self) -> str:
        if sys.platform.startswith("linux"):
            meminfo = self._read_text(Path("/proc/meminfo"))
            if meminfo:
                wanted = ("MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree")
                lines = [
                    line for line in meminfo.splitlines() if line.split(":")[0].strip() in wanted
                ]
                if lines:
                    return "\n".join(lines)
        elif sys.platform == "darwin":
            lines = []
            memsize = await self._run_command("sysctl", "-n", "hw.memsize")
            if memsize and memsize.isdigit():
                lines.append(f"Total RAM: {self._format_bytes(int(memsize))}")
            vm_stat = await self._run_command("vm_stat")
            if vm_stat:
                free = self._parse_vm_stat_free_bytes(vm_stat)
                if free is not None:
                    lines.append(f"Free RAM (free + inactive pages): {self._format_bytes(free)}")
                lines.append(vm_stat)
            if lines:
                return "\n".join(lines)
        elif sys.platform == "win32":
            win = self._windows_memory_status()
            if win:
                return win
        return "RAM information unavailable on this platform (best-effort probe)."

    async def _disk_section(self) -> str:
        if self._workspace_path is None:
            return "No workspace path configured; disk space not probed."
        try:
            usage = shutil.disk_usage(self._workspace_path)
        except OSError:
            return f"Disk usage unavailable for workspace path {self._workspace_path}."
        return (
            f"Workspace: {self._workspace_path}\n"
            f"Total: {self._format_bytes(usage.total)}, "
            f"free: {self._format_bytes(usage.free)}"
        )

    @staticmethod
    def _parse_vm_stat_free_bytes(vm_stat_output: str) -> int | None:
        """Compute free bytes from ``vm_stat``: (free + inactive pages) × page size."""
        page_size_match = re.search(r"page size of (\d+) bytes", vm_stat_output)
        if not page_size_match:
            return None
        page_size = int(page_size_match.group(1))
        total_pages = 0
        found = False
        for label in ("Pages free", "Pages inactive"):
            match = re.search(rf"{label}:\s+(\d+)", vm_stat_output)
            if match:
                total_pages += int(match.group(1))
                found = True
        return total_pages * page_size if found else None

    @classmethod
    def _windows_memory_status(cls) -> str | None:
        """Total/available RAM via GlobalMemoryStatusEx (dev/mock hosts only)."""
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        try:
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
        except (AttributeError, OSError):
            return None
        return (
            f"Total RAM: {cls._format_bytes(status.ullTotalPhys)}\n"
            f"Available RAM: {cls._format_bytes(status.ullAvailPhys)}"
        )

    async def _gpu_section(self) -> str:
        parts: list[str] = []

        nvidia = await self._run_command(
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv",
        )
        if nvidia is None:
            # Older drivers may not support compute_cap; fall back to the listing.
            nvidia = await self._run_command("nvidia-smi", "-L")
        if nvidia:
            parts.append(f"NVIDIA (nvidia-smi):\n{nvidia}")
            tensor_notes = self._nvidia_tensor_core_notes(nvidia)
            parts.append(
                tensor_notes
                or "Note: tensor cores are present on compute capability >= 7.0 (Volta "
                "and newer); their count scales with the GPU's SM count."
            )

        rocm = await self._run_command("rocm-smi", "--showproductname", "--showmeminfo", "vram")
        if rocm:
            parts.append(f"AMD (rocm-smi):\n{rocm}")

        if sys.platform == "darwin":
            displays = await self._run_command("system_profiler", "SPDisplaysDataType")
            if displays:
                parts.append(f"macOS (system_profiler SPDisplaysDataType):\n{displays}")

        if sys.platform.startswith("linux") and not parts:
            pci_gpus = await self._lspci_lines(self._GPU_PCI_PATTERN)
            if pci_gpus:
                parts.append(
                    "PCI display devices (lspci; no vendor tool available for details):\n"
                    + "\n".join(pci_gpus)
                )

        if not parts:
            return (
                "No GPU detected (best-effort probe via nvidia-smi, rocm-smi, "
                "lspci, system_profiler)."
            )
        return "\n\n".join(parts)

    @classmethod
    def _nvidia_tensor_core_notes(cls, nvidia_csv: str) -> str | None:
        """Per-GPU architecture/tensor-core notes from nvidia-smi CSV output.

        Returns ``None`` when nothing can be parsed (e.g. ``nvidia-smi -L``
        fallback output), letting the caller fall back to a generic note.
        """
        notes = []
        for line in nvidia_csv.splitlines()[1:]:  # first line is the CSV header
            fields = [field.strip() for field in line.split(",")]
            if len(fields) > 2:
                described = cls._describe_compute_cap(fields[2])
                if described:
                    notes.append(f"{fields[0]}: {described}")
        return "\n".join(notes) or None

    @staticmethod
    def _describe_compute_cap(compute_cap: str) -> str | None:
        """Map an NVIDIA compute capability to architecture and tensor-core support."""
        try:
            major, minor = (int(part) for part in compute_cap.split("."))
        except ValueError:
            return None
        if major >= 10:
            arch = "Blackwell (5th-gen tensor cores; FP4/FP8/BF16/TF32)"
        elif major == 9:
            arch = "Hopper (4th-gen tensor cores; FP8/BF16/TF32)"
        elif (major, minor) == (8, 9):
            arch = "Ada Lovelace (4th-gen tensor cores; FP8/BF16/TF32)"
        elif major == 8:
            arch = "Ampere (3rd-gen tensor cores; BF16/TF32)"
        elif (major, minor) == (7, 5):
            arch = "Turing (2nd-gen tensor cores; FP16/INT8)"
        elif major == 7:
            arch = "Volta (1st-gen tensor cores; FP16)"
        else:
            arch = "pre-Volta (no tensor cores)"
        return f"compute capability {major}.{minor} — {arch}"

    async def _accelerator_section(self) -> str:
        parts: list[str] = []

        if sys.platform == "darwin" and platform.machine() == "arm64":
            parts.append(
                "Apple Neural Engine: present (integrated in the Apple Silicon SoC; "
                "reachable via Core ML; the GPU is reachable via Metal/MPS)."
            )

        if sys.platform.startswith("linux"):
            # Intel NPUs, Habana Gaudi, and similar devices register under the
            # kernel's accel subsystem.
            accel_devices = sorted(str(p) for p in Path("/dev").glob("accel*"))
            if accel_devices:
                parts.append(f"Kernel accel devices: {', '.join(accel_devices)}")
            # Coral Edge TPUs appear as /dev/apex_*.
            apex_devices = sorted(str(p) for p in Path("/dev").glob("apex*"))
            if apex_devices:
                parts.append(f"Coral Edge TPU devices: {', '.join(apex_devices)}")
            pci_accels = await self._lspci_lines(self._ACCEL_PCI_PATTERN)
            if pci_accels:
                parts.append("PCI accelerator devices (lspci):\n" + "\n".join(pci_accels))

        tpu_env = {
            var: os.environ[var] for var in ("TPU_NAME", "COLAB_TPU_ADDR") if var in os.environ
        }
        if tpu_env:
            parts.append(
                "Cloud TPU environment: " + ", ".join(f"{k}={v}" for k, v in tpu_env.items())
            )

        if not parts:
            return "No NPU or other dedicated accelerator detected (best-effort probe)."
        return "\n\n".join(parts)


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

    async def get_available_hardware_resources(self) -> str:
        probe = _HardwareResourcesProbe(self._workspace_path or self._session_dir)
        hw = await probe.collect()
        return "\n\n".join(
            [
                f"# Available hardware resources — {hw.platform} (best-effort probe)",
                f"## CPU\n{hw.cpu}",
                f"## RAM\n{hw.ram}",
                f"## Disk\n{hw.disk}",
                f"## GPU\n{hw.gpu}",
                f"## NPU / other accelerators\n{hw.accelerators}",
            ]
        )

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
