"""Unit tests for SandboxExecResult (opendatasci.sandbox.base)."""


from datetime import datetime

from opendatasci.sandbox.base import SandboxExecResult


class TestSandboxExecResultDefaults:
    def test_required_field_success(self) -> None:
        r = SandboxExecResult(success=True)
        assert r.success is True

    def test_output_defaults_to_none(self) -> None:
        assert SandboxExecResult(success=True).output is None

    def test_stdout_defaults_to_empty_string(self) -> None:
        assert SandboxExecResult(success=True).stdout == ""

    def test_error_defaults_to_none(self) -> None:
        assert SandboxExecResult(success=True).error is None

    def test_code_defaults_to_empty_string(self) -> None:
        assert SandboxExecResult(success=True).code == ""

