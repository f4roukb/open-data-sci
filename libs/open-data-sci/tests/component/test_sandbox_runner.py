"""Component tests: the sandbox runner script (``opendatasci/sandbox/_runner.py``).

In production the runner is injected into the SRT sandbox session directory and
executed as a subprocess: it reads user code from ``OPENDATASCI_CODE_B64``,
executes it inside a fresh namespace (nothing carries over between calls), and
emits one JSON payload on stdout.  Here we execute the *real* script in-process
via ``runpy`` with the same environment contract (code/workspace env-vars),
asserting on the JSON payload — the exact observable surface the sandbox host
sees.
"""

import base64
import contextlib
import io
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

import opendatasci.sandbox as _sandbox_pkg
from opendatasci.sandbox.base import PAYLOAD_SENTINEL

RUNNER_PATH = Path(_sandbox_pkg.__file__).parent / "_runner.py"


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    """Configure the runner's env-var contract and return an executor.

    The executor runs the real ``_runner.py`` in-process (so coverage is
    measured) while preserving the process-level state the script mutates:
    the working directory and ``sys.stdin``.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENDATASCI_WORKSPACE", str(workspace))

    def run(code: str) -> dict:
        monkeypatch.setenv(
            "OPENDATASCI_CODE_B64", base64.b64encode(code.encode("utf-8")).decode("ascii")
        )
        old_cwd = os.getcwd()
        old_stdin = sys.stdin
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                runpy.run_path(str(RUNNER_PATH))
        finally:
            os.chdir(old_cwd)
            sys.stdin = old_stdin
        # The runner now tees progress prints straight through as they
        # happen (see _runner.py's _TeeStdout), so captured stdout is no
        # longer necessarily just the JSON payload -- pull it out by its
        # PAYLOAD_SENTINEL prefix instead of parsing the whole buffer.
        stdout = buf.getvalue()
        idx = stdout.rfind(PAYLOAD_SENTINEL)
        assert idx != -1, f"Runner produced no payload sentinel.\nstdout: {stdout}"
        return json.loads(stdout[idx + len(PAYLOAD_SENTINEL) :].strip())

    run.workspace = workspace  # type: ignore[attr-defined]
    return run


class TestSuccessPayload:
    def test_simple_execution_reports_stdout_result_and_vars(self, runner_env) -> None:
        payload = runner_env("x = 1\nprint('hello')\nresult = x + 1\n")
        assert payload["success"] is True
        assert payload["stdout"] == "hello\n"
        assert payload["result"] == "2"
        assert payload["var_info"] == {"x": "int"}
        assert payload["saved_results"] == {}

    def test_no_result_variable_yields_null_result(self, runner_env) -> None:
        payload = runner_env("x = 1\n")
        assert payload["success"] is True
        assert payload["result"] is None

    def test_workspace_scaffolding_created_and_exposed(self, runner_env) -> None:
        payload = runner_env("in_ws = str(workspacedir)\nods = opendatasci_directory.name\n")
        assert payload["success"] is True
        assert (runner_env.workspace / ".opendatasci").is_dir()
        assert payload["var_info"] == {"in_ws": "str", "ods": "str"}

    def test_code_runs_with_workspace_as_cwd(self, runner_env) -> None:
        payload = runner_env("import os\nprint(os.path.basename(os.getcwd()))\n")
        assert payload["success"] is True
        assert payload["stdout"] == "workspace\n"


class TestVariableDescriptions:
    def test_dataframe_list_dict_and_scalar_descriptions(self, runner_env) -> None:
        payload = runner_env(
            "import pandas as pd\n"
            "df = pd.DataFrame({'a': [1, 2]})\n"
            "items = [1, 2, 3]\n"
            "mapping = {'k': 1}\n"
            "flag = True\n"
        )
        assert payload["success"] is True
        assert payload["var_info"]["df"] == "DataFrame (2, 1)"
        assert payload["var_info"]["items"] == "list (len=3)"
        assert payload["var_info"]["mapping"] == "dict (len=1)"
        assert payload["var_info"]["flag"] == "bool"

    def test_underscore_variables_are_skipped(self, runner_env) -> None:
        payload = runner_env("_private = 3\npublic = 4\n")
        assert payload["success"] is True
        assert "_private" not in payload["var_info"]
        assert "public" in payload["var_info"]

    def test_non_picklable_values_are_reported_like_any_other_var(self, runner_env) -> None:
        payload = runner_env("f = lambda v: v\nkeep = 7\n")
        assert payload["success"] is True
        assert payload["var_info"]["f"] == "function"
        assert payload["var_info"]["keep"] == "int"


class TestNoStatePersistence:
    """Every call runs in a fresh namespace; nothing carries over to the next."""

    def test_variables_do_not_survive_across_runs(self, runner_env) -> None:
        assert runner_env("x = 5\n")["success"] is True
        payload = runner_env("y = x * 2\nresult = y\n")
        assert payload["success"] is False
        assert "NameError" in payload["error"]

    def test_result_variable_is_not_carried_over(self, runner_env) -> None:
        assert runner_env("result = 10\n")["success"] is True
        payload = runner_env("z = result\n")
        assert payload["success"] is False
        assert "NameError" in payload["error"]


class TestSaveResult:
    def test_save_result_reported_within_the_run_it_was_called_in(self, runner_env) -> None:
        payload = runner_env("save_result('metric', 42)\nsave_result('other', 'ok')\n")
        assert payload["success"] is True
        assert payload["saved_results"] == {"metric": "42", "other": "'ok'"}

    def test_saved_results_do_not_survive_across_runs(self, runner_env) -> None:
        runner_env("save_result('metric', 42)\n")
        payload = runner_env("x = 1\n")
        assert payload["saved_results"] == {}


class TestErrorPayload:
    def test_exception_reports_type_message_and_traceback(self, runner_env) -> None:
        payload = runner_env("print('before')\nraise RuntimeError('boom')\n")
        assert payload["success"] is False
        assert payload["error"].startswith("RuntimeError: boom")
        assert "Traceback" in payload["error"]
        # stdout produced before the crash is still surfaced
        assert payload["stdout"] == "before\n"
        assert payload["var_info"] == {}
        assert payload["saved_results"] == {}

    def test_stdin_reads_fail_instead_of_hanging(self, runner_env) -> None:
        payload = runner_env("value = input()\n")
        assert payload["success"] is False
        assert "EOFError" in payload["error"]

    def test_syntax_error_is_reported(self, runner_env) -> None:
        payload = runner_env("def broken(:\n")
        assert payload["success"] is False
        assert "SyntaxError" in payload["error"]
