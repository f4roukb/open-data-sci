"""Unit tests for opendatasci.sandbox._runner.

The runner is a standalone script executed as a subprocess; tests exercise it
by invoking it directly via the current interpreter and inspecting the JSON
payload it emits to stdout.
"""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from opendatasci.sandbox.base import PAYLOAD_SENTINEL

_RUNNER = str(Path(__file__).parents[3] / "opendatasci" / "sandbox" / "_runner.py")


def _run(code: str, workspace: str) -> dict:
    """Execute the runner script and return the parsed JSON payload."""
    env = {
        **os.environ,
        "OPENDATASCI_CODE_B64": base64.b64encode(code.encode()).decode("ascii"),
        "OPENDATASCI_WORKSPACE": workspace,
    }
    result = subprocess.run(
        [sys.executable, _RUNNER],
        capture_output=True,
        text=True,
        env=env,
    )
    # The runner now tees progress prints straight to real stdout as they
    # happen (see _runner.py's _TeeStdout), so the payload is no longer
    # necessarily the only content on stdout -- find it by its sentinel
    # prefix instead of assuming it's the last line.
    idx = result.stdout.rfind(PAYLOAD_SENTINEL)
    if idx == -1:
        pytest.fail(
            f"Runner produced no payload sentinel.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return json.loads(result.stdout[idx + len(PAYLOAD_SENTINEL) :].strip())


@pytest.fixture()
def workspace(tmp_path):
    return str(tmp_path / "workspace")


class TestRunnerSuccess:
    def test_simple_assignment(self, workspace) -> None:
        payload = _run("x = 1 + 1", workspace)
        assert payload["success"] is True

    def test_result_variable_is_captured(self, workspace) -> None:
        payload = _run("result = 42", workspace)
        assert payload["success"] is True
        assert payload["result"] == "42"

    def test_stdout_is_captured(self, workspace) -> None:
        payload = _run("print('hello')", workspace)
        assert payload["success"] is True
        assert "hello" in payload["stdout"]

    def test_var_info_populated(self, workspace) -> None:
        payload = _run("x = [1, 2, 3]", workspace)
        assert "x" in payload["var_info"]
        assert "list" in payload["var_info"]["x"]

    def test_private_vars_excluded_from_var_info(self, workspace) -> None:
        payload = _run("_private = 99", workspace)
        assert "_private" not in payload["var_info"]

    def test_save_result_helper(self, workspace) -> None:
        payload = _run("save_result('answer', 42)", workspace)
        assert payload["success"] is True
        assert "answer" in payload["saved_results"]


class TestRunnerNoStatePersistence:
    """Each call executes in a fresh namespace; nothing carries over."""

    def test_variable_does_not_survive_across_calls(self, workspace) -> None:
        _run("x = 10", workspace)
        payload = _run("result = x * 2", workspace)
        assert payload["success"] is False
        assert "NameError" in payload["error"]

    def test_saved_results_do_not_survive_across_calls(self, workspace) -> None:
        _run("save_result('val', 7)", workspace)
        payload = _run("x = 1", workspace)
        assert "val" not in payload["saved_results"]


class TestRunnerError:
    def test_syntax_error_returns_failure(self, workspace) -> None:
        payload = _run("def f(:\n    pass", workspace)
        assert payload["success"] is False
        assert payload["error"]

    def test_runtime_error_returns_failure(self, workspace) -> None:
        payload = _run("raise ValueError('boom')", workspace)
        assert payload["success"] is False
        assert "ValueError" in payload["error"]
        assert "boom" in payload["error"]

    def test_error_includes_traceback(self, workspace) -> None:
        payload = _run("1 / 0", workspace)
        assert "Traceback" in payload["error"]

    def test_stdout_before_error_is_preserved(self, workspace) -> None:
        payload = _run("print('before'); raise RuntimeError('after')", workspace)
        assert payload["success"] is False
        assert "before" in payload["stdout"]
