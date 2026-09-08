"""Execution-related tools: Python, TUI, and library listing."""

import ast
import re
import tomllib
from pathlib import Path
from typing import Any, ClassVar, Literal, override
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from opendatasci.configs import OpenDataSciConfig
from opendatasci.human_inputs.human_approval import HumanApprovalBaseManager
from opendatasci.models.factory import bind_structured_output, create_model
from opendatasci.sandbox.base import BaseSandbox, SandboxExecResult
from opendatasci.tasks.base import BackgroundTaskManagerBase, RunMode
from opendatasci.tools.base import OpenDataSciBaseTool

PYPROJECT_TOML: Path = Path(__file__).parent.parent / "pyproject.toml"


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


def _format_exec_result(code: str, exec_result: SandboxExecResult) -> str:
    """Format a completed Python execution as the string returned to the agent."""
    if exec_result.success:
        parts = []
        if exec_result.stdout:
            parts.append(f"stdout:\n{exec_result.stdout}")
        if exec_result.output is not None:
            parts.append(f"result:\n{exec_result.output}")
        return "\n".join(parts) if parts else "Code executed successfully (no output)"
    return _format_exec_error(code, exec_result.error or "")


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


class ListPythonLibsTool(OpenDataSciBaseTool):
    """Check available Python libraries."""

    class CallArgs(BaseModel):
        summary: str
        communication: str

    name: str = "list_python_libs"
    description: str = (
        "Check which Python libraries are available before writing code that imports them.\n\n"
        "Stdlib modules are always present; only non-standard imports need checking."
    )
    args_schema: type[BaseModel] = CallArgs

    @override
    async def _arun(self, **kwargs: Any) -> str:
        with PYPROJECT_TOML.open("rb") as fh:
            data = tomllib.load(fh)
        libs = data.get("tool", {}).get("opendatasci", {}).get("opendatasci_agent_libs", [])
        if not libs:
            return "No agent libraries configured."
        return ",".join(libs)


class ExecutePythonCodeForegroundOnlyTool(OpenDataSciBaseTool):
    """Execute Python code inside the active sandbox, foreground only.

    Used where there's no way to monitor a background run (worker agents have
    no ``check_task``/``monitor_task`` tools) -- see ``ExecutePythonCodeTool``
    for the background-capable variant given to the main agent.
    """

    class CallArgs(BaseModel):
        code: str
        summary: str
        communication: str

    name: str = "execute_python_code"
    description: str = """\
Execute Python code in the active workspace environment.

# Pre-bound variables
- ``wb``: workspace data files.
- ``sheets``: ``{"sheet_name": DataFrame, ...}``
- ``text_files``: ``{"filename": content, ...}``
- ``opendatasci_directory``: ``Path`` for saving output files to the workspace.
- ``save_result(name, value)``: record an additional named result from this run
  (alongside, or instead of, ``result``); scoped to this single execution.

# How to use this tool
- Assign ``result = ...`` to return a value.
- Any library can be imported; check ``list_python_libs`` first for non-standard ones.
- Prefer vectorised operations over row-wise loops on large DataFrames.
- For long-running code, print or log periodic progress rather than staying silent —
  it can then be tracked with `monitor_task` when running in the background.

# How NOT to use this tool
- Don't retry the same failing code verbatim — address the structured error before retrying.

Args:
    code:          Python code to execute.
    summary:       3-4 word status label (e.g. "Calculating monthly totals").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me load the sales data and check for missing values.").\
"""
    args_schema: type[BaseModel] = CallArgs

    sandbox: BaseSandbox

    @override
    async def _arun(self, code: str, summary: str, communication: str, **kwargs: Any) -> str:
        return _format_exec_result(code, await self.sandbox.execute(code))


class ExecutePythonCodeTool(OpenDataSciBaseTool):
    """Execute Python code inside the active sandbox, with an optional background run mode."""

    class CallArgs(BaseModel):
        code: str
        summary: str
        communication: str
        run_mode: RunMode = RunMode.FOREGROUND

    name: str = "execute_python_code"
    description: str = """\
Execute Python code in the active workspace environment.

# Pre-bound variables
- ``wb``: workspace data files.
- ``sheets``: ``{"sheet_name": DataFrame, ...}``
- ``text_files``: ``{"filename": content, ...}``
- ``opendatasci_directory``: ``Path`` for saving output files to the workspace.
- ``save_result(name, value)``: record an additional named result from this run
  (alongside, or instead of, ``result``); scoped to this single execution.

# How to use this tool
- Assign ``result = ...`` to return a value.
- Any library can be imported; check ``list_python_libs`` first for non-standard ones.
- Prefer vectorised operations over row-wise loops on large DataFrames.
- Assign a ``run_mode``:

  # When to use "background"
  - You don't need the result right away and can keep helping the user with
    something else in the meantime.
  - The code is expected to take a while (heavy training runs, large-scale
    data processing, anything that would otherwise stall the conversation).
  - Print or log periodic progress rather than staying silent — each line
    is streamed into the task's activity log as it's printed, so a
    `monitor_task` registered against it can catch a marker (e.g. "epoch 5
    done", "ERROR") the moment it appears, not just once the whole run ends.

  # When to use "foreground" (default)
  - You need the result before you can proceed with anything else.
  - The code is quick — scheduling overhead outweighs the benefit.

  ``"background"`` schedules the code in the background and returns a task ID
  immediately instead of blocking on completion. Rather than polling, register
  a `monitor_task` for a pattern you expect to see (e.g. a completion marker
  or error) to be notified the moment it shows up; use `check_task`/`list_tasks`
  to inspect the run on demand, and `stop_task` to stop it.

# How NOT to use this tool
- Don't retry the same failing code verbatim — address the structured error before retrying.

Args:
    code:          Python code to execute.
    summary:       3-4 word status label (e.g. "Calculating monthly totals").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me load the sales data and check for missing values.").
    run_mode:      "foreground" to wait for and return the result; "background" to
                   schedule it in the background and return its task ID immediately.
                   Prefer "background" for long-running code.\
"""
    args_schema: type[BaseModel] = CallArgs

    sandbox: BaseSandbox
    background_task_manager: BackgroundTaskManagerBase

    @override
    async def _arun(
        self,
        code: str,
        summary: str,
        communication: str,
        run_mode: RunMode = RunMode.FOREGROUND,
        **kwargs: Any,
    ) -> str:
        if run_mode == RunMode.BACKGROUND:

            async def _work(task_id: UUID) -> str:
                async def _on_stdout_line(line: str) -> None:
                    await self.background_task_manager.push_activity(task_id, line)

                exec_result = await self.sandbox.execute(code, on_stdout_line=_on_stdout_line)
                return _format_exec_result(code, exec_result)

            task_id = await self.background_task_manager.submit_task(_work, summary=summary)
            return (
                f"Scheduled background task_id={task_id} — {summary}\n"
                "Use `check_task`/`list_tasks` to monitor, `stop_task` to stop it, "
                "or `monitor_task` to watch its activity log for a pattern."
            )

        return _format_exec_result(code, await self.sandbox.execute(code))


class ExecuteCliCommandTool(OpenDataSciBaseTool):
    """Run a CLI command in the workspace (no approval gate)."""

    class CallArgs(BaseModel):
        command: str
        summary: str
        communication: str

    name: str = "execute_cli_command"
    description: str = """\
Run a read-oriented TUI command inside the active workspace directory.

Useful for inspecting the workspace without Python: listing files,
searching for patterns, counting lines, or diffing outputs. ``gh`` is also
available for read-oriented GitHub lookups (``gh repo view``, ``gh issue
list``, ``gh pr diff``, ``gh search ...``, ``gh api``) — see the
``github.com`` skill domain; it is the only command with network access,
and that access is scoped to GitHub's own hosts.

# Permitted commands
``ls``, ``cat``, ``grep``, ``wc``, ``find``, ``head``, ``tail``, ``cut``,
``diff``, ``gh``, and others in the safe set. ``|`` and ``&&`` are allowed.

# How to use this tool
- For a long-running command, prefer a form that prints progress as it goes —
  it can then be tracked with `monitor_task` when running in the background.

# When NOT to use this tool
- For write operations (file creation, deletion, or modification) — not permitted,
  including ``gh`` write subcommands (``create``, ``merge``, ``delete``, etc.).
- When ``list_workspace_files`` already covers the need.

Args:
    command:       TUI command to run (e.g. ``"ls -la"``, ``"grep -r 'keyword' ."``).
    summary:       3-4 word status label (e.g. "Listing workspace files").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me see what files are available.").\
"""
    args_schema: type[BaseModel] = CallArgs

    sandbox: BaseSandbox

    @override
    async def _arun(self, command: str, summary: str, communication: str, **kwargs: Any) -> str:
        return _format_cli_result(await self.sandbox.execute_cli(command))


class ExecuteCliCommandWithApprovalTool(OpenDataSciBaseTool):
    """Run a CLI command in the workspace with an optional human-approval gate."""

    class CallArgs(BaseModel):
        command: str
        summary: str
        communication: str
        request_approval: bool = False

    _COMMAND_DECLINED_MESSAGE: ClassVar[str] = (
        "The user declined to run this command. If the command had a potential security "
        "risk, maybe you need to try a safer approach that the user may be more inclined "
        "to accept. Never attempt to execute harmful commands."
    )

    name: str = "execute_cli_command"
    description: str = """\
Run a read-oriented TUI command inside the active workspace directory.

Useful for inspecting the workspace without Python: listing files,
searching for patterns, counting lines, or diffing outputs. ``gh`` is also
available for read-oriented GitHub lookups (``gh repo view``, ``gh issue
list``, ``gh pr diff``, ``gh search ...``, ``gh api``) — see the
``github.com`` skill domain; it is the only command with network access,
and that access is scoped to GitHub's own hosts.

# Permitted commands
``ls``, ``cat``, ``grep``, ``wc``, ``find``, ``head``, ``tail``, ``cut``,
``diff``, ``gh``, and others in the safe set. ``|`` and ``&&`` are allowed.

# How to use this tool
- For a long-running command, prefer a form that prints progress as it goes —
  it can then be tracked with `monitor_task` when running in the background.

# When NOT to use this tool
- For write operations (file creation, deletion, or modification) — not permitted,
  including ``gh`` write subcommands (``create``, ``merge``, ``delete``, etc.).
- When ``list_workspace_files`` already covers the need.

Args:
    command:          TUI command to run (e.g. ``"ls -la"``, ``"grep -r 'keyword' ."``).
    summary:          3-4 word status label (e.g. "Listing workspace files").
    communication:    Brief message to the user about what you're doing
                      (e.g. "Let me see what files are available.").
    request_approval: Set to True when the command could disrupt the user's
                      device or active work in any way; execution then pauses
                      until the user explicitly approves it. Leave False for
                      trivially safe read-only commands.\
"""
    args_schema: type[BaseModel] = CallArgs

    sandbox: BaseSandbox
    approval_manager: HumanApprovalBaseManager

    @override
    async def _arun(
        self,
        command: str,
        summary: str,
        communication: str,
        request_approval: bool = False,
        **kwargs: Any,
    ) -> str:
        if request_approval:
            approved = await self.approval_manager.ask_for_command_approval(command)
            if not approved:
                return self._COMMAND_DECLINED_MESSAGE
        return _format_cli_result(await self.sandbox.execute_cli(command))


class VerifyPythonCodeTool(OpenDataSciBaseTool):
    """Gate-check Python code for correctness and optimality."""

    class CallArgs(BaseModel):
        code: str
        context: str = ""
        summary: str
        communication: str

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

    _REVIEW_SYSTEM_PROMPT: ClassVar[str] = """\
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

    name: str = "verify_python_code"
    description: str = """\
Gate-check Python code for correctness and optimality before a costly execution.

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
    code:          Python code to review.
    context:       Optional description of what the code does and any relevant
                   constraints (e.g. "Trains a gradient-boosting classifier on a
                   10 M-row DataFrame; must finish in under 30 s and use < 8 GB RAM").
    summary:       3-4 word status label (e.g. "Reviewing pipeline code").
    communication: Brief message to the user about what you're doing
                   (e.g. "Let me review this before we run it.").\
"""
    args_schema: type[BaseModel] = CallArgs

    datasci_config: "OpenDataSciConfig"
    _llm: Any = PrivateAttr()

    @model_validator(mode="after")
    def _build_llm(self) -> "VerifyPythonCodeTool":
        self._llm = bind_structured_output(create_model(self.datasci_config), self._CodeReview)
        return self

    @override
    async def _arun(
        self,
        code: str,
        summary: str,
        communication: str,
        context: str = "",
        **kwargs: Any,
    ) -> str:
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
            SystemMessage(content=self._REVIEW_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        review: VerifyPythonCodeTool._CodeReview = await self._llm.ainvoke(messages)

        return (
            f"VERDICT: {review.verdict}\n\n"
            f"### Correctness\n{review.correctness}\n\n"
            f"### Optimality\n{review.optimality}"
        )


def create_coding_tools(
    sandbox: BaseSandbox,
    background_task_manager: BackgroundTaskManagerBase | None = None,
) -> list[BaseTool]:
    """Return execution tools bound to *sandbox*: execute_python_code and list_python_libs.

    When *background_task_manager* is provided (main agent only — background
    tasks are only reachable via ``check_task``/``list_tasks``/``monitor_task``,
    which worker agents don't have), ``execute_python_code`` exposes a
    ``run_mode`` argument that schedules the code in the background instead of
    blocking on it, mirroring the ``task`` tool's own background mode. Worker
    agents omit it and get the plain, foreground-only tool — same pattern as
    ``create_cli_tools``'s ``approval_manager``.
    """
    if background_task_manager is None:
        execute_python_code: BaseTool = ExecutePythonCodeForegroundOnlyTool(sandbox=sandbox)
    else:
        execute_python_code = ExecutePythonCodeTool(
            sandbox=sandbox, background_task_manager=background_task_manager
        )
    return [execute_python_code, ListPythonLibsTool()]


def create_cli_tools(
    sandbox: BaseSandbox,
    approval_manager: HumanApprovalBaseManager | None = None,
) -> list[BaseTool]:
    """Return the execute_cli_command tool bound to *sandbox*.

    When *approval_manager* is provided (main agent only — approval interrupts
    require a checkpointed graph), the tool exposes a ``request_approval``
    argument that pauses the agent for the user's explicit yes/no consent
    before the command runs. Worker agents omit it and get the plain tool.
    """
    if approval_manager is None:
        return [ExecuteCliCommandTool(sandbox=sandbox)]
    return [ExecuteCliCommandWithApprovalTool(sandbox=sandbox, approval_manager=approval_manager)]


def create_code_verification_tools(datasci_config: "OpenDataSciConfig") -> list[BaseTool]:
    """Return the ``verify_python_code`` tool pre-wired to *datasci_config*'s LLM."""
    return [VerifyPythonCodeTool(datasci_config=datasci_config)]
