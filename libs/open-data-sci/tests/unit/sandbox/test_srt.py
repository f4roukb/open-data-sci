"""Unit tests for opendatasci.sandbox.srt — the native sandbox dependency check,
its wiring into SRTSandboxFactory.create(), and the hardware resources probe."""


import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.sandbox.base import BaseSandbox, SandboxExecResult
from opendatasci.sandbox.srt import (
    SRTSandbox,
    SRTSandboxFactory,
    _HardwareResources,
    _HardwareResourcesProbe,
    check_sandbox_dependencies,
)

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


# ---------------------------------------------------------------------------
# _HardwareResourcesProbe — probing primitives
# ---------------------------------------------------------------------------


class TestProbeRunCommand:
    @pytest.mark.asyncio
    async def test_missing_binary_returns_none(self) -> None:
        result = await _HardwareResourcesProbe()._run_command("definitely-not-a-real-binary-xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_none(self) -> None:
        result = await _HardwareResourcesProbe()._run_command(
            sys.executable, "-c", "raise SystemExit(3)"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_success_returns_stripped_stdout(self) -> None:
        result = await _HardwareResourcesProbe()._run_command(
            sys.executable, "-c", "print('  hello  ')"
        )
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_empty_stdout_returns_none(self) -> None:
        result = await _HardwareResourcesProbe()._run_command(sys.executable, "-c", "pass")
        assert result is None


class TestProbeVmStatParsing:
    _VM_STAT = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                              1000.\n"
        "Pages active:                            5000.\n"
        "Pages inactive:                          2000.\n"
    )

    def test_free_plus_inactive_times_page_size(self) -> None:
        assert (
            _HardwareResourcesProbe._parse_vm_stat_free_bytes(self._VM_STAT)
            == (1000 + 2000) * 16384
        )

    def test_missing_page_size_returns_none(self) -> None:
        assert _HardwareResourcesProbe._parse_vm_stat_free_bytes("Pages free: 10.") is None

    def test_no_page_counts_returns_none(self) -> None:
        assert _HardwareResourcesProbe._parse_vm_stat_free_bytes("(page size of 4096 bytes)") is None


class TestProbeFormatBytes:
    def test_formats_gib_with_one_decimal(self) -> None:
        assert _HardwareResourcesProbe._format_bytes(16 * 1024**3) == "16.0 GiB"


# ---------------------------------------------------------------------------
# _HardwareResourcesProbe.collect — one populated field per aspect
# ---------------------------------------------------------------------------


class TestProbeCollect:
    def _make_probe(self) -> _HardwareResourcesProbe:
        # Neutralise external probes so tests are fast and host-independent.
        probe = _HardwareResourcesProbe()
        probe._run_command = AsyncMock(return_value=None)  # type: ignore[method-assign]
        return probe

    @pytest.mark.asyncio
    async def test_returns_dataclass_instance(self) -> None:
        hw = await self._make_probe().collect()
        assert isinstance(hw, _HardwareResources)

    @pytest.mark.asyncio
    async def test_every_aspect_is_populated(self) -> None:
        hw = await self._make_probe().collect()
        assert hw.platform
        assert hw.cpu
        assert hw.ram
        assert hw.disk
        assert hw.gpu
        assert hw.accelerators

    @pytest.mark.asyncio
    async def test_cpu_always_reports_core_count(self) -> None:
        hw = await self._make_probe().collect()
        assert "Logical cores:" in hw.cpu

    @pytest.mark.asyncio
    async def test_disk_reports_workspace_free_space(self, tmp_path) -> None:
        probe = _HardwareResourcesProbe(workspace_path=tmp_path)
        probe._run_command = AsyncMock(return_value=None)  # type: ignore[method-assign]
        hw = await probe.collect()
        assert str(tmp_path) in hw.disk
        assert "free:" in hw.disk

    @pytest.mark.asyncio
    async def test_disk_without_workspace_path_reports_not_probed(self) -> None:
        hw = await self._make_probe().collect()
        assert "not probed" in hw.disk


class TestDescribeComputeCap:
    def test_blackwell(self) -> None:
        assert "Blackwell" in _HardwareResourcesProbe._describe_compute_cap("12.0")

    def test_ada_lovelace(self) -> None:
        assert "Ada Lovelace" in _HardwareResourcesProbe._describe_compute_cap("8.9")

    def test_ampere(self) -> None:
        assert "Ampere" in _HardwareResourcesProbe._describe_compute_cap("8.6")

    def test_turing(self) -> None:
        assert "Turing" in _HardwareResourcesProbe._describe_compute_cap("7.5")

    def test_volta(self) -> None:
        assert "Volta" in _HardwareResourcesProbe._describe_compute_cap("7.0")

    def test_pre_volta_has_no_tensor_cores(self) -> None:
        assert "no tensor cores" in _HardwareResourcesProbe._describe_compute_cap("6.1")

    def test_unparseable_returns_none(self) -> None:
        assert _HardwareResourcesProbe._describe_compute_cap("[N/A]") is None

    def test_per_gpu_notes_from_csv(self) -> None:
        csv = (
            "name, driver_version, compute_cap, memory.total [MiB]\n"
            "NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0, 16303 MiB\n"
            "NVIDIA T4, 591.86, 7.5, 16384 MiB"
        )
        notes = _HardwareResourcesProbe._nvidia_tensor_core_notes(csv)
        assert "NVIDIA GeForce RTX 5070 Ti: compute capability 12.0" in notes
        assert "NVIDIA T4: compute capability 7.5" in notes

    def test_unparseable_listing_returns_none(self) -> None:
        assert (
            _HardwareResourcesProbe._nvidia_tensor_core_notes(
                "GPU 0: NVIDIA T4 (UUID: GPU-abc)"
            )
            is None
        )


# ---------------------------------------------------------------------------
# get_available_hardware_resources — base default / SRT override
# ---------------------------------------------------------------------------


class _MinimalSandbox(BaseSandbox):
    async def execute(self, code: str) -> SandboxExecResult:
        raise NotImplementedError

    async def execute_cli(self, command: str) -> SandboxExecResult:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class TestSandboxHardwareResources:
    @pytest.mark.asyncio
    async def test_base_default_reports_no_information(self) -> None:
        result = await _MinimalSandbox().get_available_hardware_resources()
        assert "No hardware resource information" in result

    @pytest.mark.asyncio
    async def test_srt_sandbox_formats_probe_result_into_sections(self) -> None:
        hw = _HardwareResources(
            platform="TestOS-1.0",
            cpu="16 cores",
            ram="32 GiB total",
            disk="512 GiB free",
            gpu="RTX 4090, 24 GiB VRAM",
            accelerators="none",
        )
        sandbox = SRTSandbox()
        try:
            with patch("opendatasci.sandbox.srt._HardwareResourcesProbe") as probe_cls:
                probe_cls.return_value.collect = AsyncMock(return_value=hw)
                result = await sandbox.get_available_hardware_resources()
        finally:
            await sandbox.close()
        probe_cls.return_value.collect.assert_awaited_once_with()
        assert "TestOS-1.0" in result
        assert "## CPU\n16 cores" in result
        assert "## RAM\n32 GiB total" in result
        assert "## Disk\n512 GiB free" in result
        assert "## GPU\nRTX 4090, 24 GiB VRAM" in result
        assert "## NPU / other accelerators\nnone" in result
