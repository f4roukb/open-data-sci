"""Unit tests for opendatasci.sandbox.srt — the native sandbox dependency check
and its wiring into SRTSandboxFactory.create()."""

import asyncio
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.sandbox.base import PAYLOAD_SENTINEL
from opendatasci.sandbox.srt import SRTSandbox, SRTSandboxFactory, check_sandbox_dependencies


def _mock_proc(stdout_lines: list[bytes], stderr: bytes = b"", returncode: int = 0) -> MagicMock:
    """Build a MagicMock standing in for asyncio.subprocess.Process, wired for
    the streaming read path _run_subprocess now uses (readline()/read()/wait()
    rather than communicate())."""
    proc = MagicMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(side_effect=[*stdout_lines, b""])
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=stderr)
    proc.wait = AsyncMock()
    proc.returncode = returncode
    return proc

# ---------------------------------------------------------------------------
# check_sandbox_dependencies
# ---------------------------------------------------------------------------


class TestCheckSandboxDependencies:
    def test_passes_when_dependencies_available(self) -> None:
        with patch("opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=True):
            check_sandbox_dependencies()  # must not raise

    def test_raises_on_unsupported_platform(self) -> None:
        with (
            patch("opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=False),
            patch(
                "opendatasci.sandbox.srt.SandboxManager.is_supported_platform", return_value=False
            ),
            patch("opendatasci.sandbox.srt.get_platform", return_value="windows"),
        ):
            with pytest.raises(RuntimeError, match="not supported on platform 'windows'"):
                check_sandbox_dependencies()

    def test_raises_with_macos_install_hint(self) -> None:
        with (
            patch("opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=False),
            patch(
                "opendatasci.sandbox.srt.SandboxManager.is_supported_platform", return_value=True
            ),
            patch("opendatasci.sandbox.srt.get_platform", return_value="macos"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                check_sandbox_dependencies()
            message = str(exc_info.value)
            assert "ripgrep (rg)" in message
            assert "brew install ripgrep" in message

    def test_raises_with_linux_install_hint(self) -> None:
        with (
            patch("opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=False),
            patch(
                "opendatasci.sandbox.srt.SandboxManager.is_supported_platform", return_value=True
            ),
            patch("opendatasci.sandbox.srt.get_platform", return_value="linux"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                check_sandbox_dependencies()
            message = str(exc_info.value)
            assert "bubblewrap (bwrap)" in message
            assert "socat" in message
            assert "apt-get install" in message


# ---------------------------------------------------------------------------
# SRTSandboxFactory.create() surfaces the check before yielding a sandbox
# ---------------------------------------------------------------------------


class TestSRTSandboxFactoryCreate:
    async def test_create_raises_before_constructing_sandbox_when_deps_missing(self) -> None:
        factory = SRTSandboxFactory()
        with patch(
            "opendatasci.sandbox.srt.check_sandbox_dependencies",
            side_effect=RuntimeError("missing deps"),
        ):
            with pytest.raises(RuntimeError, match="missing deps"):
                async with factory.create(workspace_path=None):
                    pytest.fail("sandbox should never be yielded when dependencies are missing")

    async def test_create_yields_sandbox_when_deps_available(self) -> None:
        factory = SRTSandboxFactory()
        with (
            patch("opendatasci.sandbox.srt.check_sandbox_dependencies"),
            patch("opendatasci.sandbox.srt.SRTSandbox") as sandbox_cls,
        ):
            sandbox_instance = MagicMock()
            sandbox_instance.close = AsyncMock()
            sandbox_cls.return_value = sandbox_instance

            async with factory.create(workspace_path=None) as sandbox:
                assert sandbox is sandbox_instance


# ---------------------------------------------------------------------------
# SRTSandbox._run_subprocess: cancellation must not orphan the spawned process
# ---------------------------------------------------------------------------


class TestRunSubprocessCancellation:
    async def test_cancellation_terminates_process_tree_before_propagating(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            proc = MagicMock()
            proc.communicate = AsyncMock(side_effect=asyncio.CancelledError())

            with (
                patch(
                    "opendatasci.sandbox.srt.asyncio.create_subprocess_shell",
                    AsyncMock(return_value=proc),
                ),
                patch.object(sandbox, "_terminate_process_tree", AsyncMock()) as terminate_mock,
            ):
                with pytest.raises(asyncio.CancelledError):
                    await sandbox._run_subprocess("echo hi", env={}, cwd=".")

                terminate_mock.assert_awaited_once_with(proc)
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    async def test_timeout_still_terminates_process_tree(self) -> None:
        """Regression guard: the pre-existing timeout path must keep working
        alongside the new cancellation handling."""
        sandbox = SRTSandbox(workspace_path=None, command_timeout=0.01)
        try:
            proc = MagicMock()

            async def _hang(*args: object, **kwargs: object) -> tuple[bytes, bytes]:
                await asyncio.sleep(10)
                return b"", b""

            proc.communicate = _hang

            with (
                patch(
                    "opendatasci.sandbox.srt.asyncio.create_subprocess_shell",
                    AsyncMock(return_value=proc),
                ),
                patch.object(sandbox, "_terminate_process_tree", AsyncMock()) as terminate_mock,
            ):
                with pytest.raises(TimeoutError):
                    await sandbox._run_subprocess("echo hi", env={}, cwd=".")

                terminate_mock.assert_awaited_once_with(proc)
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)
