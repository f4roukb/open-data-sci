"""Abstract sandbox interface for Python and TUI code execution."""

import re
import shlex
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SandboxExecResult:
    """Result of a single Python or TUI execution in the sandbox."""

    success: bool
    output: Any = None
    stdout: str = ""
    error: str | None = None
    code: str = ""


ALLOWED_CLI_COMMANDS: frozenset[str] = frozenset(
    {
        # Directory listing & navigation
        "ls",
        "dir",
        "pwd",
        "tree",
        # File viewing
        "cat",
        "head",
        "tail",
        "file",
        "stat",
        # Text search & discovery
        "grep",
        "find",
        "which",
        # Text processing (read-oriented)
        "cut",
        "sort",
        "uniq",
        "wc",
        "awk",
        "sed",
        "tr",
        "strings",
        # File comparison & checksums
        "diff",
        "cmp",
        "md5sum",
        "sha256sum",
        "shasum",
        # Structured data / binary inspection
        "jq",
        "xxd",
        "od",
        # General output & info
        "echo",
        "printf",
        "date",
        "uname",
        "printenv",
        "env",
        # Archive inspection (listing only)
        "unzip",
        "tar",
        "zip",
    }
)

_FORBIDDEN_CLI_OPERATORS: tuple[str, ...] = ("||", ";", "`", "$(", ">", "<")
_ALLOWED_CLI_SHELL_OPERATORS: frozenset[str] = frozenset({"|", "&&"})


def validate_cli_command(command: str) -> str | None:
    """Validate *command* against the TUI allowlist.

    Returns an error string describing the first violation found, or ``None``
    if the command is valid. This function is pure (no side effects) and can
    be called from tests directly.
    """
    command = command.strip()
    if not command:
        return "Empty command."

    for op in _FORBIDDEN_CLI_OPERATORS:
        if op in command:
            return f"Shell operator '{op}' is not allowed."

    use_shell = any(op in command for op in _ALLOWED_CLI_SHELL_OPERATORS)
    segments = re.split(r"&&|\|(?!\|)", command) if use_shell else [command]

    for seg in segments:
        seg = seg.strip()
        if not seg:
            return "Empty pipeline segment."
        try:
            parts = shlex.split(seg)
        except ValueError as exc:
            return f"Invalid command syntax: {exc}"
        cmd_name = Path(parts[0]).name.lower()
        if cmd_name not in ALLOWED_CLI_COMMANDS:
            sample = sorted(ALLOWED_CLI_COMMANDS)
            return (
                f"Command '{cmd_name}' is not allowed. "
                f"Permitted commands include: {', '.join(sample)}, … "
                f"({len(ALLOWED_CLI_COMMANDS)} total)"
            )

    return None


class BaseSandbox(ABC):
    """Stateful code execution sandbox scoped to a single agent session.

    Responsible for running Python code and TUI commands, capturing output,
    and preserving state (variables, results) across turns within the same
    conversation.
    """

    @abstractmethod
    async def execute(self, code: str) -> SandboxExecResult:
        """Execute Python *code* and return the result."""

    @abstractmethod
    async def execute_cli(self, command: str) -> SandboxExecResult:
        """Execute a TUI *command* and return the result.

        Implementations must validate *command* against ``validate_cli_command``
        before running it and return a failed ``SandboxExecResult`` on violation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all session state (history, variables, results)."""

    async def close(self) -> None:
        """Release external resources held by this sandbox.

        The default implementation is a no-op. Override in sandboxes that
        manage external processes or connections (e.g. Docker containers).
        """


class BaseSandboxFactory(ABC):
    """Abstract factory that creates :class:`BaseSandbox` instances via async context managers.

    Callers must always acquire a sandbox through :meth:`create` so that
    teardown is guaranteed regardless of how the calling code exits::

        async with factory.create(workspace_path=path) as sandbox:
            result = await sandbox.execute(code)
        # sandbox.close() has been called here
    """

    @abstractmethod
    def create(
        self, workspace_path: Path | None = None
    ) -> AbstractAsyncContextManager["BaseSandbox"]:
        """Return an async context manager that yields a fresh :class:`BaseSandbox`.

        The sandbox is closed automatically when the context manager exits,
        whether by normal return, exception, or cancellation.

        Args:
            workspace_path: Optional filesystem root that the sandbox should
                treat as its working directory.
        """
