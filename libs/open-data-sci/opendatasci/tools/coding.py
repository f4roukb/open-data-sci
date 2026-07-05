"""Execution-related tools: Python, TUI, and library listing."""

import ast
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from opendatasci.models.factory import create_model
from opendatasci.sandbox.base import BaseSandbox, SandboxExecResult

if TYPE_CHECKING:
    from opendatasci.configs import OpenDataSciConfig

PYPROJECT_TOML: Path = Path(__file__).parent.parent / "pyproject.toml"


@tool
def list_python_libs() -> str:
    """Check which Python libraries are available before writing code that imports them.

    Stdlib modules are always present; only non-standard imports need checking.
    """
    with PYPROJECT_TOML.open("rb") as fh:
        data = tomllib.load(fh)
    libs = data.get("tool", {}).get("opendatasci", {}).get("opendatasci_agent_libs", [])
    if not libs:
        return "No agent libraries configured."
    return ",".join(libs)


class _CodeReview(BaseModel):
    verdict: Literal["LGTM", "NEEDS CHANGES"] = Field(
        description="Overall verdict: LGTM if the code is correct and optimal, NEEDS CHANGES otherwise."
    )
    correctness: str = Field(
        description=(
            "Concise findings on correctness: bugs, logical errors, off-by-one errors, "
            "incorrect API usage, unhandled edge cases, type mismatches. "
            'Use "No issues found." if none.'
        )
    )
    optimality: str = Field(
        description=(
            "Concise findings on optimality: unnecessary latency, excessive memory allocation, "
            "redundant computation, missed vectorisation, suboptimal data structures. "
            'Use "No issues found." if none.'
        )
    )


_REVIEW_SYSTEM_PROMPT = """\
You are an expert Python code reviewer. Your role is to critically evaluate code \
before it runs in an expensive or high-latency pipeline stage, where a bug or \
inefficiency could be very costly to recover from.

Review the provided code on exactly two dimensions:

**Correctness** — bugs, logical errors, off-by-one errors, incorrect API usage, \
unhandled edge cases, wrong variable names, type mismatches, or any issue that would \
cause the code to raise an exception or produce incorrect results at runtime.

**Optimality** — unnecessary latency (e.g. redundant passes over large datasets, \
serial loops that should be vectorised, blocking I/O inside loops), excessive memory \
allocation, redundant computation, or suboptimal algorithm/data-structure choices \
that inflate wall-clock time or peak memory usage.

Be terse. Reference specific lines or variable names. Do not explain what the code does.\
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_exec_error(code: str, error: str) -> str:
    """Format a Python execution error as a structured message for the agent.

    Parses the traceback to extract the error type, the failing line number,
    and the relevant code snippet so the agent addresses the specific problem
    rather than retrying blindly.
    """
    lines = error.splitlines()

    error_type = "Error"
    error_msg = ""
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if ": " in line:
            error_type, error_msg = line.split(": ", 1)
        else:
            error_type = line
        break

    failing_line: int | None = None
    for line in lines:
        m = re.search(r'File "<opendatasci>", line (\d+)', line)
        if m:
            failing_line = int(m.group(1))

    snippet = ""
    if failing_line is not None:
        code_lines = code.splitlines()
        if 1 <= failing_line <= len(code_lines):
            snippet = code_lines[failing_line - 1].strip()

    header = f"Error [{error_type}]"
    if failing_line is not None:
        header += f" on line {failing_line}"

    parts = [header]
    if snippet:
        parts.append(f"Code:    {snippet}")
    if error_msg:
        parts.append(f"Message: {error_msg}")
    parts.append("")
    parts.append("Address this specific error before retrying.")
    return "\n".join(parts)


def _format_cli_result(result: SandboxExecResult) -> str:
    """Format a TUI SandboxExecResult as a string for the agent."""
    if result.success:
        return result.stdout or "Command succeeded (no output)."
    parts = []
    if result.stdout:
        parts.append(f"stdout:\n{result.stdout}")
    if result.error:
        parts.append(result.error)
    return "\n".join(parts) if parts else "Command failed."


def create_code_verification_tools(datasci_config: "OpenDataSciConfig") -> list[BaseTool]:
    """Return the ``verify_python_code`` tool pre-wired to *datasci_config*'s LLM."""
    _llm = create_model(datasci_config).with_structured_output(_CodeReview)

    @tool
    async def verify_python_code(code: str, context: str = "") -> str:
        """Gate-check Python code for correctness and optimality before a costly execution.

        Returns a LGTM / NEEDS CHANGES verdict with per-dimension findings.

        # When to use this tool
        - Before executing code whose failure mid-pipeline would be expensive to recover from:
          model training, distributed jobs, multi-step preprocessing pipelines.
        - When the code is non-trivial and bugs would be hard to diagnose post-hoc.

        # When NOT to use this tool
        - When the code is cheap to run — just execute it and fix errors from the output.
        - As a substitute for running code: verification reduces obvious risk but does not
          prove correctness.

        Args:
            code:    Python code to review.
            context: Optional description of what the code does and any relevant
                     constraints (e.g. "Trains a gradient-boosting classifier on a
                     10 M-row DataFrame; must finish in under 30 s and use < 8 GB RAM").
        """
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return (
                f"Static check failed [SyntaxError] on line {exc.lineno}: {exc.msg}\n"
                "Fix the syntax error and try again."
            )

        user_content = f"```python\n{code}\n```"
        if context:
            user_content = f"Context: {context}\n\n{user_content}"

        messages = [
            SystemMessage(content=_REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        review: _CodeReview = await _llm.ainvoke(messages)  # type: ignore[assignment]

        return (
            f"VERDICT: {review.verdict}\n\n"
            f"### Correctness\n{review.correctness}\n\n"
            f"### Optimality\n{review.optimality}"
        )

    return [verify_python_code]


def create_coding_tools(sandbox: BaseSandbox) -> list[BaseTool]:
    """Return execution tools bound to *sandbox*: execute_python_code."""

    @tool
    async def execute_python_code(code: str, summary: str, communication: str) -> str:
        """Execute Python code in the active workspace environment.

        # Pre-bound variables
        - ``wb``: workspace data files.
        - ``sheets``: ``{"sheet_name": DataFrame, ...}``
        - ``text_files``: ``{"filename": content, ...}``
        - ``opendatasci_directory``: ``Path`` for saving output files to the workspace.
        - ``save_result(name, value)``: persist a named result for export.

        # How to use this tool
        - Assign ``result = ...`` to return a value.
        - Any library can be imported; check ``list_python_libs`` first for non-standard ones.
        - Prefer vectorised operations over row-wise loops on large DataFrames.

        # How NOT to use this tool
        - Don't retry the same failing code verbatim — address the structured error before retrying.

        Args:
            code:          Python code to execute.
            summary:       3-4 word status label (e.g. "Calculating monthly totals").
            communication: Brief message to the user about what you're doing
                           (e.g. "Let me load the sales data and check for missing values.").
        """
        exec_result = await sandbox.execute(code)
        if exec_result.success:
            parts = []
            if exec_result.stdout:
                parts.append(f"stdout:\n{exec_result.stdout}")
            if exec_result.output is not None:
                parts.append(f"result:\n{exec_result.output}")
            return "\n".join(parts) if parts else "Code executed successfully (no output)"
        return _format_exec_error(code, exec_result.error or "")

    return [execute_python_code, list_python_libs]


def create_cli_tools(sandbox: BaseSandbox) -> list[BaseTool]:
    """Return the execute_cli_command tool bound to *sandbox* (main agent only)."""

    @tool
    async def execute_cli_command(command: str, summary: str, communication: str) -> str:
        """Run a read-oriented TUI command inside the active workspace directory.

        Useful for inspecting the workspace without Python: listing files,
        searching for patterns, counting lines, or diffing outputs.

        # Permitted commands
        ``ls``, ``cat``, ``grep``, ``wc``, ``find``, ``head``, ``tail``, ``cut``,
        ``diff``, and others in the safe set. ``|`` and ``&&`` are allowed.

        # When NOT to use this tool
        - For write operations (file creation, deletion, or modification) — not permitted.
        - When ``list_workspace_files`` already covers the need.

        Args:
            command:       TUI command to run (e.g. ``"ls -la"``, ``"grep -r 'keyword' ."``).
            summary:       3-4 word status label (e.g. "Listing workspace files").
            communication: Brief message to the user about what you're doing
                           (e.g. "Let me see what files are available.").
        """
        cli_result = await sandbox.execute_cli(command)
        return _format_cli_result(cli_result)

    return [execute_cli_command]
