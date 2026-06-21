"""Tool display metadata for the TUI renderer.

Centralises ToolDisplay declarations and the global REGISTRY so the TUI
and streaming layer can look up icons, labels, and summary arguments without
knowing anything about tool internals.

Usage::

    from opendatasci._tui.tools_display import REGISTRY, ToolDisplay

    display = REGISTRY.get("execute_python_code", ToolDisplay(label="execute_python_code"))
    print(display.icon, display.label)  # 🐍 Code
"""

import types
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDisplay:
    """Display metadata for a single tool.

    Attributes:
        label:          Human-readable tool name shown in the TUI
                        (e.g. ``"Python"``, ``"Web Search"``).
        icon:           Optional single emoji rendered before the label
                        (e.g. ``"🐍"``).
        summary_arg:    Name of the tool argument whose value is used as the
                        short summary line in the ephemeral block.  ``None``
                        means no summary (the block shows only the label and
                        communication).
        display:        When ``True`` (default) an ephemeral block is shown
                        while the tool runs and kept as a breadcrumb afterwards.
                        Set to ``False`` for silent internal tools.
    """

    label: str
    icon: str = ""
    summary_arg: str | None = None
    display: bool = True


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

register("execute_python_code", ToolDisplay(label="Code", icon="🐍", summary_arg="summary"))
register("execute_cli_command", ToolDisplay(label="Command", icon="⌨️", summary_arg="summary"))
register("list_python_libs", ToolDisplay(label="Checking available libraries", icon="📦"))
register("enter_plan_mode", ToolDisplay(label="Planning", icon="🎯"))
register("exit_plan_mode", ToolDisplay(label="Done planning", icon="✅"))
register("load_skill", ToolDisplay(label="Loading skill", icon="🧠", summary_arg="summary"))
register("spawn_workers", ToolDisplay(label="Spawning workers", icon="⚙️"))
register(
    "read_dataset_info",
    ToolDisplay(label="Reading dataset info", icon="📚", summary_arg="summary"),
)
register(
    "update_dataset_info",
    ToolDisplay(label="Updating dataset notes", icon="📝", display=False),
)
register("profile_dataset", ToolDisplay(label="Profiling dataset", icon="📊"))
register("list_workspace_files", ToolDisplay(label="Listing workspace files", icon="📁"))
register(
    "web_search",
    ToolDisplay(label="Searching the web", icon="🌐", summary_arg="summary"),
)
register(
    "fetch_url",
    ToolDisplay(label="Fetching content", icon="🔗", summary_arg="summary"),
)
register("ask_user_mcq", ToolDisplay(label="Question", icon="💬", display=False))
register("enter_self_review_mode", ToolDisplay(label="Reviewing progress so far", icon="🔍"))
register("exit_self_review_mode", ToolDisplay(label="Done reviewing progress", icon="✅"))
register("verify_python_code", ToolDisplay(label="Reviewing code", icon="🔎"))
