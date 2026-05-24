"""Unit tests for opendatasci.tools.coding."""


from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from opendatasci.sandbox.base import SandboxExecResult
from opendatasci.tools.coding import (
    create_cli_tools,
    create_code_verification_tools,
    create_coding_tools,
    list_python_libs,
)

# ---------------------------------------------------------------------------
# execute_python_code — error formatting (tested indirectly via the tool)
# ---------------------------------------------------------------------------


class TestExecutePythonCodeErrorFormatting:
    async def _run(self, code: str, error: str) -> str:
        sandbox = MagicMock()
        sandbox.execute = AsyncMock(return_value=SandboxExecResult(success=False, error=error))
        tools = create_coding_tools(sandbox)
        execute_python_code = next(t for t in tools if t.name == "execute_python_code")
        return await execute_python_code.ainvoke(
            {"code": code, "summary": "s", "communication": "c"}
        )

    @pytest.mark.asyncio
    async def test_extracts_error_type_and_message(self) -> None:
        error = 'Traceback (most recent call last):\n  File "<opendatasci>", line 1, in <module>\nValueError: bad input'
        result = await self._run("x = 1", error)
        assert "Error [ValueError]" in result
        assert "bad input" in result

    @pytest.mark.asyncio
    async def test_extracts_failing_line_number(self) -> None:
        code = "a = 1\nb = bad_call()\nc = 3"
        error = 'Traceback (most recent call last):\n  File "<opendatasci>", line 2, in <module>\nNameError: bad_call'
        result = await self._run(code, error)
        assert "on line 2" in result

    @pytest.mark.asyncio
    async def test_includes_code_snippet_for_failing_line(self) -> None:
        code = "a = 1\nb = bad_call()\nc = 3"
        error = 'Traceback (most recent call last):\n  File "<opendatasci>", line 2, in <module>\nNameError: bad_call'
        result = await self._run(code, error)
        assert "b = bad_call()" in result

    @pytest.mark.asyncio
    async def test_no_line_reference_when_file_not_opendatasci(self) -> None:
        error = 'Traceback (most recent call last):\n  File "other.py", line 5, in foo\nRuntimeError: boom'
        result = await self._run("x = 1", error)
        assert "on line" not in result

    @pytest.mark.asyncio
    async def test_error_type_only_when_no_colon_separator(self) -> None:
        result = await self._run("x = 1", "SyntaxError")
        assert "Error [SyntaxError]" in result

    @pytest.mark.asyncio
    async def test_always_ends_with_retry_guidance(self) -> None:
        result = await self._run("x = 1", "ValueError: oops")
        assert "Address this specific error before retrying." in result

    @pytest.mark.asyncio
    async def test_snippet_stripped_of_leading_whitespace(self) -> None:
        code = "def foo():\n    raise ValueError('x')"
        error = 'Traceback (most recent call last):\n  File "<opendatasci>", line 2, in foo\nValueError: x'
        result = await self._run(code, error)
        assert "raise ValueError('x')" in result

    @pytest.mark.asyncio
    async def test_out_of_range_line_number_produces_no_snippet(self) -> None:
        error = 'Traceback (most recent call last):\n  File "<opendatasci>", line 99, in <module>\nIndexError: out'
        result = await self._run("x = 1", error)
        assert "Code:" not in result

    @pytest.mark.asyncio
    async def test_last_error_line_takes_precedence(self) -> None:
        error = (
            "Traceback (most recent call last):\n"
            '  File "<opendatasci>", line 1, in <module>\n'
            "ValueError: first\n"
            "TypeError: second"
        )
        result = await self._run("x = 1", error)
        assert "TypeError" in result

    @pytest.mark.asyncio
    async def test_empty_error_string_returns_generic_header(self) -> None:
        result = await self._run("x = 1", "")
        assert "Error [Error]" in result


# ---------------------------------------------------------------------------
# execute_cli_command — result formatting (tested indirectly via the tool)
# ---------------------------------------------------------------------------


class TestExecuteCliCommandResultFormatting:
    async def _run(self, exec_result: SandboxExecResult) -> str:
        sandbox = MagicMock()
        sandbox.execute_cli = AsyncMock(return_value=exec_result)
        tool = create_cli_tools(sandbox)[0]
        return await tool.ainvoke({"command": "ls", "summary": "s", "communication": "c"})

    @pytest.mark.asyncio
    async def test_success_with_stdout_returns_stdout(self) -> None:
        result = await self._run(SandboxExecResult(success=True, stdout="hello"))
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_success_with_no_stdout_returns_default_message(self) -> None:
        result = await self._run(SandboxExecResult(success=True, stdout=""))
        assert result == "Command succeeded (no output)."

    @pytest.mark.asyncio
    async def test_failure_with_stdout_and_error_combines_both(self) -> None:
        result = await self._run(
            SandboxExecResult(success=False, stdout="partial", error="something failed")
        )
        assert "partial" in result
        assert "something failed" in result

    @pytest.mark.asyncio
    async def test_failure_with_only_error(self) -> None:
        result = await self._run(SandboxExecResult(success=False, error="permission denied"))
        assert "permission denied" in result

    @pytest.mark.asyncio
    async def test_failure_with_only_stdout(self) -> None:
        result = await self._run(SandboxExecResult(success=False, stdout="output only"))
        assert "output only" in result

    @pytest.mark.asyncio
    async def test_failure_with_no_output_returns_fallback(self) -> None:
        result = await self._run(SandboxExecResult(success=False))
        assert result == "Command failed."


# ---------------------------------------------------------------------------
# list_python_libs
# ---------------------------------------------------------------------------


@contextmanager
def _mock_pyproject(data: dict):
    """Patch PYPROJECT_TOML and tomllib.load so no real file is read."""
    mock_fh = MagicMock()
    mock_path = MagicMock()
    mock_path.open.return_value.__enter__.return_value = mock_fh
    mock_path.open.return_value.__exit__.return_value = False
    with (
        patch("opendatasci.tools.coding.PYPROJECT_TOML", mock_path),
        patch("opendatasci.tools.coding.tomllib.load", return_value=data),
    ):
        yield


class TestListPythonLibs:
    def test_returns_comma_separated_libs(self) -> None:
        data = {"tool": {"opendatasci": {"opendatasci_agent_libs": ["pandas>=2.0", "numpy", "scikit-learn"]}}}
        with _mock_pyproject(data):
            result = list_python_libs.invoke({})
        assert result == "pandas>=2.0,numpy,scikit-learn"

    def test_single_lib_has_no_comma(self) -> None:
        data = {"tool": {"opendatasci": {"opendatasci_agent_libs": ["requests"]}}}
        with _mock_pyproject(data):
            result = list_python_libs.invoke({})
        assert result == "requests"

    def test_empty_opendatasci_agent_libs_returns_no_libs_message(self) -> None:
        data = {"tool": {"opendatasci": {"opendatasci_agent_libs": []}}}
        with _mock_pyproject(data):
            result = list_python_libs.invoke({})
        assert result == "No agent libraries configured."

    def test_missing_opendatasci_agent_libs_key_returns_no_libs_message(self) -> None:
        data = {"tool": {"opendatasci": {}}}
        with _mock_pyproject(data):
            result = list_python_libs.invoke({})
        assert result == "No agent libraries configured."

    def test_missing_opendatasci_section_returns_no_libs_message(self) -> None:
        data = {"tool": {}}
        with _mock_pyproject(data):
            result = list_python_libs.invoke({})
        assert result == "No agent libraries configured."

    def test_missing_tool_section_returns_no_libs_message(self) -> None:
        data = {}
        with _mock_pyproject(data):
            result = list_python_libs.invoke({})
        assert result == "No agent libraries configured."

    def test_opens_pyproject_toml_in_binary_mode(self) -> None:
        mock_fh = MagicMock()
        mock_path = MagicMock()
        mock_path.open.return_value.__enter__.return_value = mock_fh
        mock_path.open.return_value.__exit__.return_value = False
        data = {"tool": {"opendatasci": {"opendatasci_agent_libs": ["pandas"]}}}
        with (
            patch("opendatasci.tools.coding.PYPROJECT_TOML", mock_path),
            patch("opendatasci.tools.coding.tomllib.load", return_value=data),
        ):
            list_python_libs.invoke({})
        mock_path.open.assert_called_once_with("rb")

    def test_libs_preserve_version_constraints(self) -> None:
        libs = ["pandas>=2.0,<3.0", "numpy~=1.26"]
        data = {"tool": {"opendatasci": {"opendatasci_agent_libs": libs}}}
        with _mock_pyproject(data):
            result = list_python_libs.invoke({})
        for lib in libs:
            assert lib in result


# ---------------------------------------------------------------------------
# create_coding_tools
# ---------------------------------------------------------------------------


class TestGetCodingTools:
    def _make_sandbox(self) -> MagicMock:
        sandbox = MagicMock()
        sandbox.execute = AsyncMock()
        return sandbox

    def test_returns_two_tools(self) -> None:
        tools = create_coding_tools(self._make_sandbox())
        assert len(tools) == 2

    def test_tool_names(self) -> None:
        tools = create_coding_tools(self._make_sandbox())
        names = {t.name for t in tools}
        assert "execute_python_code" in names
        assert "list_python_libs" in names

    @pytest.mark.asyncio
    async def test_execute_python_success_with_output(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute.return_value = SandboxExecResult(success=True, output="42", stdout="")
        tools = create_coding_tools(sandbox)
        execute_python_code = next(t for t in tools if t.name == "execute_python_code")
        result = await execute_python_code.ainvoke(
            {"code": "result = 42", "summary": "test", "communication": "testing"}
        )
        assert "42" in result

    @pytest.mark.asyncio
    async def test_execute_python_success_with_stdout(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute.return_value = SandboxExecResult(success=True, stdout="hello", output=None)
        tools = create_coding_tools(sandbox)
        execute_python_code = next(t for t in tools if t.name == "execute_python_code")
        result = await execute_python_code.ainvoke(
            {"code": "print('hello')", "summary": "test", "communication": "testing"}
        )
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_python_success_no_output_returns_default(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute.return_value = SandboxExecResult(success=True, stdout="", output=None)
        tools = create_coding_tools(sandbox)
        execute_python_code = next(t for t in tools if t.name == "execute_python_code")
        result = await execute_python_code.ainvoke(
            {"code": "x = 1", "summary": "test", "communication": "testing"}
        )
        assert "no output" in result.lower()

    @pytest.mark.asyncio
    async def test_execute_python_failure_returns_formatted_error(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute.return_value = SandboxExecResult(
            success=False,
            error='File "<opendatasci>", line 1\nSyntaxError: invalid syntax',
        )
        tools = create_coding_tools(sandbox)
        execute_python_code = next(t for t in tools if t.name == "execute_python_code")
        result = await execute_python_code.ainvoke(
            {"code": "def bad(", "summary": "test", "communication": "testing"}
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_execute_python_delegates_to_sandbox_execute(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute.return_value = SandboxExecResult(success=True, stdout="", output=None)
        tools = create_coding_tools(sandbox)
        execute_python_code = next(t for t in tools if t.name == "execute_python_code")
        await execute_python_code.ainvoke({"code": "x = 1", "summary": "s", "communication": "c"})
        sandbox.execute.assert_awaited_once_with("x = 1")


# ---------------------------------------------------------------------------
# get_cli_tool
# ---------------------------------------------------------------------------


class TestGetCliTool:
    def _make_sandbox(self) -> MagicMock:
        sandbox = MagicMock()
        sandbox.execute_cli = AsyncMock()
        return sandbox

    def test_returns_single_tool(self) -> None:
        tool = create_cli_tools(self._make_sandbox())[0]
        assert tool.name == "execute_cli_command"

    @pytest.mark.asyncio
    async def test_cli_success_returns_stdout(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute_cli.return_value = SandboxExecResult(success=True, stdout="file.csv")
        tool = create_cli_tools(sandbox)[0]
        result = await tool.ainvoke(
            {"command": "ls", "summary": "list", "communication": "listing files"}
        )
        assert "file.csv" in result

    @pytest.mark.asyncio
    async def test_cli_failure_returns_error(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute_cli.return_value = SandboxExecResult(
            success=False, error="command not found"
        )
        tool = create_cli_tools(sandbox)[0]
        result = await tool.ainvoke(
            {"command": "badcmd", "summary": "bad", "communication": "running"}
        )
        assert "command not found" in result

    @pytest.mark.asyncio
    async def test_cli_delegates_command_to_sandbox(self) -> None:
        sandbox = self._make_sandbox()
        sandbox.execute_cli.return_value = SandboxExecResult(success=True, stdout="")
        tool = create_cli_tools(sandbox)[0]
        await tool.ainvoke({"command": "ls -la", "summary": "s", "communication": "c"})
        sandbox.execute_cli.assert_awaited_once_with("ls -la")


# ---------------------------------------------------------------------------
# get_verify_python_code_tool / verify_python_code
# ---------------------------------------------------------------------------


def _make_review_tool(
    verdict: str = "LGTM",
    correctness: str = "No issues found.",
    optimality: str = "No issues found.",
):
    """Build a verify_python_code tool whose LLM chain is replaced by a mock.

    ``create_model(cfg).with_structured_output(...)`` is the chain wired at
    construction time.  We mock ``create_model`` to return a base LLM mock
    whose ``.with_structured_output()`` returns a structured-LLM mock whose
    ``ainvoke`` yields a mock review object with the expected attributes.
    """
    mock_review = MagicMock()
    mock_review.verdict = verdict
    mock_review.correctness = correctness
    mock_review.optimality = optimality
    mock_structured_llm = MagicMock()
    mock_structured_llm.ainvoke = AsyncMock(return_value=mock_review)
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_structured_llm
    with patch("opendatasci.tools.coding.create_model", return_value=mock_base_llm):
        tool = create_code_verification_tools(MagicMock())[0]
    return tool, mock_structured_llm


class TestGetReviewTool:
    def test_returns_tool_named_review_my_code(self) -> None:
        with patch("opendatasci.tools.coding.create_model"):
            tool = create_code_verification_tools(MagicMock())[0]
        assert tool.name == "verify_python_code"

    def test_creates_model_from_agent_config(self) -> None:
        datasci_config = MagicMock()
        with patch("opendatasci.tools.coding.create_model") as mock_create:
            create_code_verification_tools(datasci_config)[0]
        mock_create.assert_called_once_with(datasci_config)

    def test_with_structured_output_called_with_code_review_schema(self) -> None:
        mock_base_llm = MagicMock()
        with patch("opendatasci.tools.coding.create_model", return_value=mock_base_llm):
            create_code_verification_tools(MagicMock()[0])
        mock_base_llm.with_structured_output.assert_called_once()
        schema_cls = mock_base_llm.with_structured_output.call_args[0][0]
        assert issubclass(schema_cls, pydantic.BaseModel)
        assert {"verdict", "correctness", "optimality"} <= schema_cls.model_fields.keys()

    def test_model_created_once_not_per_invocation(self) -> None:
        """create_model must be called at tool-creation time, not per call."""
        with patch("opendatasci.tools.coding.create_model") as mock_create:
            tool = create_code_verification_tools(MagicMock()[0])
        assert mock_create.call_count == 1
        _ = tool


class TestReviewMyCodeStaticChecks:
    @pytest.mark.asyncio
    async def test_syntax_error_returns_static_check_failure(self) -> None:
        tool, _ = _make_review_tool()
        result = await tool.ainvoke({"code": "def bad("})
        assert "Static check failed [SyntaxError]" in result

    @pytest.mark.asyncio
    async def test_syntax_error_includes_line_number(self) -> None:
        tool, _ = _make_review_tool()
        result = await tool.ainvoke({"code": "x = 1\ndef bad("})
        assert "line" in result.lower()

    @pytest.mark.asyncio
    async def test_syntax_error_does_not_call_llm(self) -> None:
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": "def bad("})
        mock_structured_llm.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_syntax_error_instructs_to_fix(self) -> None:
        tool, _ = _make_review_tool()
        result = await tool.ainvoke({"code": "def bad("})
        assert "Fix" in result or "fix" in result

    @pytest.mark.asyncio
    async def test_valid_syntax_passes_static_check(self) -> None:
        tool, _ = _make_review_tool()
        result = await tool.ainvoke({"code": "x = 1 + 2"})
        assert "Static check failed" not in result


class TestReviewMyCodeLlmCall:
    @pytest.mark.asyncio
    async def test_valid_code_calls_llm(self) -> None:
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": "x = [i**2 for i in range(100)]"})
        mock_structured_llm.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_receives_system_and_human_messages(self) -> None:
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": "x = 1"})
        messages = mock_structured_llm.ainvoke.call_args[0][0]
        assert any(isinstance(m, SystemMessage) for m in messages)
        assert any(isinstance(m, HumanMessage) for m in messages)

    @pytest.mark.asyncio
    async def test_system_prompt_mentions_correctness(self) -> None:
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": "x = 1"})
        messages = mock_structured_llm.ainvoke.call_args[0][0]
        sys_msg = next(m for m in messages if isinstance(m, SystemMessage))
        assert "Correctness" in sys_msg.content or "correctness" in sys_msg.content.lower()

    @pytest.mark.asyncio
    async def test_system_prompt_mentions_optimality(self) -> None:
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": "x = 1"})
        messages = mock_structured_llm.ainvoke.call_args[0][0]
        sys_msg = next(m for m in messages if isinstance(m, SystemMessage))
        assert "Optimality" in sys_msg.content or "optimality" in sys_msg.content.lower()

    @pytest.mark.asyncio
    async def test_code_is_embedded_in_human_message(self) -> None:
        snippet = "result = df.groupby('id').sum()"
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": snippet})
        messages = mock_structured_llm.ainvoke.call_args[0][0]
        human_msg = next(m for m in messages if isinstance(m, HumanMessage))
        assert snippet in human_msg.content

    @pytest.mark.asyncio
    async def test_context_is_embedded_in_human_message(self) -> None:
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": "x = 1", "context": "Must run in under 5 s"})
        messages = mock_structured_llm.ainvoke.call_args[0][0]
        human_msg = next(m for m in messages if isinstance(m, HumanMessage))
        assert "Must run in under 5 s" in human_msg.content

    @pytest.mark.asyncio
    async def test_no_context_omits_context_prefix(self) -> None:
        tool, mock_structured_llm = _make_review_tool()
        await tool.ainvoke({"code": "x = 1"})
        messages = mock_structured_llm.ainvoke.call_args[0][0]
        human_msg = next(m for m in messages if isinstance(m, HumanMessage))
        assert "Context:" not in human_msg.content


class TestReviewMyCodeOutput:
    @pytest.mark.asyncio
    async def test_lgtm_verdict_appears_in_output(self) -> None:
        tool, _ = _make_review_tool(verdict="LGTM")
        result = await tool.ainvoke({"code": "x = 1"})
        assert "VERDICT: LGTM" in result

    @pytest.mark.asyncio
    async def test_needs_changes_verdict_appears_in_output(self) -> None:
        tool, _ = _make_review_tool(verdict="NEEDS CHANGES")
        result = await tool.ainvoke({"code": "x = 1"})
        assert "VERDICT: NEEDS CHANGES" in result

    @pytest.mark.asyncio
    async def test_correctness_findings_appear_under_heading(self) -> None:
        tool, _ = _make_review_tool(correctness="Off-by-one in loop at line 3.")
        result = await tool.ainvoke({"code": "x = 1"})
        assert "### Correctness" in result
        assert "Off-by-one in loop at line 3." in result

    @pytest.mark.asyncio
    async def test_optimality_findings_appear_under_heading(self) -> None:
        tool, _ = _make_review_tool(optimality="Use vectorised ops instead of the for loop.")
        result = await tool.ainvoke({"code": "x = 1"})
        assert "### Optimality" in result
        assert "Use vectorised ops instead of the for loop." in result

    @pytest.mark.asyncio
    async def test_output_format_is_stable(self) -> None:
        tool, _ = _make_review_tool(
            verdict="LGTM",
            correctness="No issues found.",
            optimality="No issues found.",
        )
        result = await tool.ainvoke({"code": "x = 1"})
        assert result == (
            "VERDICT: LGTM\n\n"
            "### Correctness\nNo issues found.\n\n"
            "### Optimality\nNo issues found."
        )
