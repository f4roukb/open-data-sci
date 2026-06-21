"""Unit tests for opendatasci.sandbox.srt — the native sandbox dependency check
and its wiring into SRTSandboxFactory.create()."""


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.sandbox.srt import SRTSandboxFactory, check_sandbox_dependencies

# ---------------------------------------------------------------------------
# check_sandbox_dependencies
# ---------------------------------------------------------------------------


class TestCheckSandboxDependencies:
    def test_passes_when_dependencies_available(self) -> None:
        with patch(
            "opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=True
        ):
            check_sandbox_dependencies()  # must not raise

    def test_raises_on_unsupported_platform(self) -> None:
        with (
            patch(
                "opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=False
            ),
            patch(
                "opendatasci.sandbox.srt.SandboxManager.is_supported_platform", return_value=False
            ),
            patch("opendatasci.sandbox.srt.get_platform", return_value="windows"),
        ):
            with pytest.raises(RuntimeError, match="not supported on platform 'windows'"):
                check_sandbox_dependencies()

    def test_raises_with_macos_install_hint(self) -> None:
        with (
            patch(
                "opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=False
            ),
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
            patch(
                "opendatasci.sandbox.srt.SandboxManager.check_dependencies", return_value=False
            ),
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
