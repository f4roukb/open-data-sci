"""Unit tests for opendatasci.sandbox._runner.

The runner is a standalone script executed as a subprocess; tests exercise it
by invoking it directly via the current interpreter and inspecting the JSON
payload it emits to stdout.
"""

import base64
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_RUNNER = str(Path(__file__).parents[3] / "opendatasci" / "sandbox" / "_runner.py")


def _run(code: str, state_path: str, workspace: str) -> dict:
    """Execute the runner script and return the parsed JSON payload."""
    env = {
        **os.environ,
        "OPENDATASCI_CODE_B64": base64.b64encode(code.encode()).decode("ascii"),
        "OPENDATASCI_STATE_PATH": state_path,
        "OPENDATASCI_WORKSPACE": workspace,
    }
    result = subprocess.run(
        [sys.executable, _RUNNER],
        capture_output=True,
        text=True,
        env=env,
    )
    # The last non-empty line of stdout is the JSON payload.
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line:
            return json.loads(line)
    pytest.fail(f"Runner produced no JSON payload.\nstdout: {result.stdout}\nstderr: {result.stderr}")


@pytest.fixture()
def session(tmp_path):
    return str(tmp_path / "state.pkl"), str(tmp_path / "workspace")


class TestRunnerSuccess:
    def test_simple_assignment(self, session) -> None:
        state, ws = session
        payload = _run("x = 1 + 1", state, ws)
        assert payload["success"] is True

    def test_result_variable_is_captured(self, session) -> None:
        state, ws = session
        payload = _run("result = 42", state, ws)
        assert payload["success"] is True
        assert payload["result"] == "42"

    def test_stdout_is_captured(self, session) -> None:
        state, ws = session
        payload = _run("print('hello')", state, ws)
        assert payload["success"] is True
        assert "hello" in payload["stdout"]

    def test_var_info_populated(self, session) -> None:
        state, ws = session
        payload = _run("x = [1, 2, 3]", state, ws)
        assert "x" in payload["var_info"]
        assert "list" in payload["var_info"]["x"]

    def test_private_vars_excluded_from_var_info(self, session) -> None:
        state, ws = session
        payload = _run("_private = 99", state, ws)
        assert "_private" not in payload["var_info"]

    def test_non_picklable_var_is_dropped(self, session) -> None:
        state, ws = session
        payload = _run("import threading; lock = threading.Lock()", state, ws)
        assert payload["success"] is True
        assert "lock" in payload["dropped_vars"]
        assert "lock" not in payload["var_info"]


class TestRunnerStatePersistence:
    def test_variable_survives_across_calls(self, session) -> None:
        state, ws = session
        _run("x = 10", state, ws)
        payload = _run("result = x * 2", state, ws)
        assert payload["success"] is True
        assert payload["result"] == "20"

    def test_save_result_helper(self, session) -> None:
        state, ws = session
        payload = _run("save_result('answer', 42)", state, ws)
        assert payload["success"] is True
        assert "answer" in payload["saved_results"]

    def test_saved_results_persist_across_calls(self, session) -> None:
        state, ws = session
        _run("save_result('val', 7)", state, ws)
        payload = _run("x = 1", state, ws)
        assert "val" in payload["saved_results"]

    def test_deleting_state_wipes_saved_results(self, session) -> None:
        # Saved results live inside state.pkl, so removing the file (how
        # SRTSandbox.reset() performs a wipe) clears them along with variables.
        state, ws = session
        _run("save_result('val', 7)", state, ws)
        Path(state).unlink()
        payload = _run("x = 1", state, ws)
        assert "val" not in payload["saved_results"]


class TestRunnerError:
    def test_syntax_error_returns_failure(self, session) -> None:
        state, ws = session
        payload = _run("def f(:\n    pass", state, ws)
        assert payload["success"] is False
        assert payload["error"]

    def test_runtime_error_returns_failure(self, session) -> None:
        state, ws = session
        payload = _run("raise ValueError('boom')", state, ws)
        assert payload["success"] is False
        assert "ValueError" in payload["error"]
        assert "boom" in payload["error"]

    def test_error_includes_traceback(self, session) -> None:
        state, ws = session
        payload = _run("1 / 0", state, ws)
        assert "Traceback" in payload["error"]

    def test_stdout_before_error_is_preserved(self, session) -> None:
        state, ws = session
        payload = _run("print('before'); raise RuntimeError('after')", state, ws)
        assert payload["success"] is False
        assert "before" in payload["stdout"]
