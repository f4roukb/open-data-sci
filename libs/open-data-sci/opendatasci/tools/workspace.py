"""Workspace navigation tool: inspect workspace files."""

from pathlib import Path

from langchain_core.tools import BaseTool, tool


def create_workspace_tools(workspace_path: Path | None) -> list[BaseTool]:
    """Return workspace tools bound to *workspace_path*."""

    @tool
    def list_workspace_files(summary: str, communication: str) -> str:
        """Map the active workspace: list all files and directories with sizes.

        Call when you need to know what files are available in the workspace,
        and before referencing workspace paths in code.
        Hidden directories (dot-prefixed) are excluded.

        Args:
            summary:       3-4 word status label (e.g. "Listing workspace files").
            communication: Brief message to the user about what you're doing
                           (e.g. "Let me check what files are available.").
        """
        path = workspace_path
        if path is None:
            return "No active workspace."
        try:
            entries = sorted(
                (
                    f
                    for f in path.rglob("*")
                    if not any(part.startswith(".") for part in f.relative_to(path).parts)
                ),
                key=lambda f: (f.is_dir(), str(f).lower()),
            )
            if not entries:
                return f"Workspace '{path.name}' is empty."
            lines = [f"Files in workspace '{path.name}':"]
            for entry in entries:
                if entry.is_dir():
                    lines.append(f"  {entry}/")
                else:
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024**2:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / 1024**2:.1f} MB"
                    lines.append(f"  {entry}  ({size_str})")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error listing workspace files: {type(exc).__name__}: {exc}"

    return [list_workspace_files]
