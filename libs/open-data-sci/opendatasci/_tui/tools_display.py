"""Tool display metadata for the TUI renderer.

Centralises ToolDisplay declarations and the global REGISTRY so the TUI
and streaming layer can look up labels and summary arguments without
knowing anything about tool internals.

Usage::

    from opendatasci._tui.tools_display import REGISTRY, ToolDisplay

    display = REGISTRY.get("execute_python_code", ToolDisplay(label="execute_python_code"))
    print(display.label)  # Code
"""

import types
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDisplay:
    """Display metadata for a single tool.

    Attributes:
        label:          Human-readable tool name shown in the TUI
                        (e.g. ``"Python"``, ``"Web Search"``).
        summary_arg:    Name of the tool argument whose value is used as the
                        short summary line in the ephemeral block.  ``None``
                        means no summary (the block shows only the label and
                        communication).
        display_status: When ``True`` (default) a tool-status line is shown
                        while the tool runs and kept as a breadcrumb afterwards.
                        Set to ``False`` for internal tools: only their
                        streamed ``communication`` (if any) is surfaced, never
                        the tool identity/status line.
    """

    label: str
    summary_arg: str | None = None
    display_status: bool = True


# Global registry: canonical tool name → ToolDisplay.
# _registry is the mutable backing dict; REGISTRY is the public read-only view.
_registry: dict[str, ToolDisplay] = {}
REGISTRY: types.MappingProxyType[str, ToolDisplay] = types.MappingProxyType(_registry)


def register(tool_name: str, display: ToolDisplay) -> None:
    """Associate *display* with *tool_name* in the global registry."""
    _registry[tool_name] = display


# ── Registrations ─────────────────────────────────────────────────────────────
# String literals are used for tool names so this module stays free of imports
# from opendatasci.tools.factory (which creates a circular dependency).

register("execute_python_code", ToolDisplay(label="Code", summary_arg="summary"))
register("execute_cli_command", ToolDisplay(label="Command", summary_arg="summary"))
register(
    "list_python_libs",
    ToolDisplay(label="Checking available libraries", summary_arg="summary"),
)
register(
    "enter_plan_mode",
    ToolDisplay(label="Planning the next steps", summary_arg="summary"),
)
register("exit_plan_mode", ToolDisplay(label="Planning complete", summary_arg="summary"))
register("load_skill", ToolDisplay(label="Loading skill", summary_arg="summary"))
register(
    "list_skills",
    ToolDisplay(label="Checking available skills", summary_arg="summary"),
)
register("spawn_workers", ToolDisplay(label="Spawning workers", summary_arg="summary"))
register(
    "read_dataset_info",
    ToolDisplay(
        label="Reading dataset information",
        summary_arg="summary",
    ),
)
register(
    "update_dataset_info",
    ToolDisplay(label="Updating dataset notes", summary_arg="summary"),
)
register(
    "profile_dataset",
    ToolDisplay(label="Profiling dataset", summary_arg="summary"),
)
register(
    "list_workspace_files",
    ToolDisplay(label="Listing workspace files", summary_arg="summary"),
)
register(
    "web_search",
    ToolDisplay(label="Searching the web", summary_arg="summary"),
)
register(
    "fetch_url",
    ToolDisplay(label="Fetching web content", summary_arg="summary"),
)
register(
    "ask_user_mcq",
    ToolDisplay(label="Question", summary_arg="summary", display_status=False),
)
register(
    "enter_self_review_mode",
    ToolDisplay(label="Reviewing progress so far", summary_arg="summary"),
)
register(
    "exit_self_review_mode",
    ToolDisplay(label="Done reviewing progress", summary_arg="summary"),
)
register(
    "verify_python_code",
    ToolDisplay(label="Reviewing code", summary_arg="summary"),
)
