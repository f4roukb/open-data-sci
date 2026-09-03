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
            proc = _mock_proc(stdout_lines=[])
            proc.stdout.readline = AsyncMock(side_effect=asyncio.CancelledError())

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
            proc = _mock_proc(stdout_lines=[])

            async def _hang(*args: object, **kwargs: object) -> bytes:
                await asyncio.sleep(10)
                return b""

            proc.stdout.readline = _hang

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


# ---------------------------------------------------------------------------
# SRTSandbox._drain_stdout / _run_subprocess: streaming stdout into a
# per-line callback (the mechanism a background task's push_activity hooks
# into) while still returning the full transcript.
# ---------------------------------------------------------------------------


class TestDrainStdout:
    async def test_forwards_each_line_to_callback_in_order(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            stream = MagicMock()
            stream.readline = AsyncMock(side_effect=[b"first\n", b"second\n", b""])
            seen: list[str] = []

            async def _on_line(line: str) -> None:
                seen.append(line)

            text = await sandbox._drain_stdout(stream, _on_line)

            assert seen == ["first", "second"]
            assert text == "first\nsecond\n"
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    async def test_payload_sentinel_line_is_not_forwarded(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            payload_line = f'{PAYLOAD_SENTINEL}{{"success": true}}\n'.encode()
            stream = MagicMock()
            stream.readline = AsyncMock(side_effect=[b"progress\n", payload_line, b""])
            seen: list[str] = []

            async def _on_line(line: str) -> None:
                seen.append(line)

            text = await sandbox._drain_stdout(stream, _on_line)

            assert seen == ["progress"]
            assert PAYLOAD_SENTINEL in text
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    async def test_lines_after_sentinel_are_still_captured_but_not_forwarded(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            payload_line = f'{PAYLOAD_SENTINEL}{{"success": true}}\n'.encode()
            stream = MagicMock()
            stream.readline = AsyncMock(side_effect=[payload_line, b"stray\n", b""])
            seen: list[str] = []

            async def _on_line(line: str) -> None:
                seen.append(line)

            text = await sandbox._drain_stdout(stream, _on_line)

            assert seen == []
            assert "stray" in text
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    async def test_no_callback_still_returns_full_text(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            stream = MagicMock()
            stream.readline = AsyncMock(side_effect=[b"a\n", b"b\n", b""])

            text = await sandbox._drain_stdout(stream, None)

            assert text == "a\nb\n"
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    async def test_blank_lines_are_not_forwarded(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            stream = MagicMock()
            stream.readline = AsyncMock(side_effect=[b"\n", b"real\n", b""])
            seen: list[str] = []

            async def _on_line(line: str) -> None:
                seen.append(line)

            await sandbox._drain_stdout(stream, _on_line)

            assert seen == ["real"]
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)


class TestRunSubprocessStreaming:
    async def test_streams_lines_and_returns_full_stdout_and_returncode(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            proc = _mock_proc(stdout_lines=[b"one\n", b"two\n"], returncode=0)
            seen: list[str] = []

            async def _on_line(line: str) -> None:
                seen.append(line)

            with patch(
                "opendatasci.sandbox.srt.asyncio.create_subprocess_shell",
                AsyncMock(return_value=proc),
            ):
                stdout_str, stderr_str, returncode = await sandbox._run_subprocess(
                    "echo hi", env={}, cwd=".", on_stdout_line=_on_line
                )

            assert seen == ["one", "two"]
            assert stdout_str == "one\ntwo\n".strip()
            assert stderr_str == ""
            assert returncode == 0
            proc.wait.assert_awaited_once()
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    async def test_stderr_is_drained_independently_of_stdout(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            proc = _mock_proc(stdout_lines=[b"out\n"], stderr=b"warn", returncode=1)

            with patch(
                "opendatasci.sandbox.srt.asyncio.create_subprocess_shell",
                AsyncMock(return_value=proc),
            ):
                stdout_str, stderr_str, returncode = await sandbox._run_subprocess(
                    "echo hi", env={}, cwd="."
                )

            assert stdout_str == "out"
            assert stderr_str == "warn"
            assert returncode == 1
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    async def test_no_callback_omits_forwarding_but_keeps_full_stdout(self) -> None:
        sandbox = SRTSandbox(workspace_path=None)
        try:
            proc = _mock_proc(stdout_lines=[b"quiet output\n"])

            with patch(
                "opendatasci.sandbox.srt.asyncio.create_subprocess_shell",
                AsyncMock(return_value=proc),
            ):
                stdout_str, _, _ = await sandbox._run_subprocess("echo hi", env={}, cwd=".")

            assert stdout_str == "quiet output"
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# SRTSandbox._parse_runner_payload: sentinel-based parsing
# ---------------------------------------------------------------------------


class TestParseRunnerPayload:
    def _sandbox(self) -> SRTSandbox:
        return SRTSandbox(workspace_path=None)

    def test_parses_payload_after_sentinel(self) -> None:
        sandbox = self._sandbox()
        try:
            raw = f'{PAYLOAD_SENTINEL}{{"success": true, "result": "42"}}'
            payload = sandbox._parse_runner_payload(raw)
            assert payload == {"success": True, "result": "42"}
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    def test_ignores_progress_output_before_sentinel(self) -> None:
        sandbox = self._sandbox()
        try:
            raw = f'progress line 1\nprogress line 2\n{PAYLOAD_SENTINEL}{{"success": true}}'
            payload = sandbox._parse_runner_payload(raw)
            assert payload == {"success": True}
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    def test_uses_last_sentinel_occurrence_if_user_output_contains_one(self) -> None:
        sandbox = self._sandbox()
        try:
            raw = f'print("{PAYLOAD_SENTINEL}fake")\n{PAYLOAD_SENTINEL}{{"success": true}}'
            payload = sandbox._parse_runner_payload(raw)
            assert payload == {"success": True}
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    def test_raises_on_missing_sentinel(self) -> None:
        sandbox = self._sandbox()
        try:
            with pytest.raises(ValueError, match="no payload sentinel"):
                sandbox._parse_runner_payload("just some text, no sentinel here")
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    def test_raises_on_empty_stdout(self) -> None:
        sandbox = self._sandbox()
        try:
            with pytest.raises(ValueError, match="no stdout payload"):
                sandbox._parse_runner_payload("")
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    def test_raises_on_non_json_payload(self) -> None:
        sandbox = self._sandbox()
        try:
            with pytest.raises(ValueError, match="not valid JSON"):
                sandbox._parse_runner_payload(f"{PAYLOAD_SENTINEL}not json")
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)

    def test_raises_on_non_object_json_payload(self) -> None:
        sandbox = self._sandbox()
        try:
            with pytest.raises(ValueError, match="not a JSON object"):
                sandbox._parse_runner_payload(f"{PAYLOAD_SENTINEL}[1, 2, 3]")
        finally:
            shutil.rmtree(sandbox._session_dir, ignore_errors=True)
