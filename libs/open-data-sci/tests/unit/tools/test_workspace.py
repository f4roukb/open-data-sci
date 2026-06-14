"""Unit tests for opendatasci.tools.workspace."""


from pathlib import Path
from opendatasci.tools.workspace import create_workspace_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tool(workspace_path: Path | None):
    tools = create_workspace_tools(workspace_path)
    return tools[0]


# ---------------------------------------------------------------------------
# create_workspace_tools – structure
# ---------------------------------------------------------------------------


class TestGetWorkspaceToolsStructure:
    def test_returns_one_tool(self) -> None:
        tools = create_workspace_tools(None)
        assert len(tools) == 1

    def test_tool_name_is_list_workspace_files(self) -> None:
        tools = create_workspace_tools(None)
        assert tools[0].name == "list_workspace_files"


# ---------------------------------------------------------------------------
# list_workspace_files
# ---------------------------------------------------------------------------


class TestListWorkspaceFiles:
    def test_no_workspace_path_returns_no_active_workspace(self) -> None:
        tool = _get_tool(None)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert "No active workspace" in result

    def test_empty_directory_returns_is_empty_message(self, tmp_path: Path) -> None:
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert "empty" in result.lower()

    def test_lists_files_with_absolute_paths(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b,c\n1,2,3\n")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert str(f) in result

    def test_does_not_list_files_with_relative_paths(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_bytes(b"a,b,c\n1,2,3\n")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        lines_with_csv = [line for line in result.splitlines() if "data.csv" in line]
        assert lines_with_csv, "data.csv not found in output"
        assert all(str(tmp_path) in line for line in lines_with_csv)

    def test_includes_workspace_name_in_output(self, tmp_path: Path) -> None:
        (tmp_path / "file.txt").write_text("hello")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert tmp_path.name in result

    def test_shows_directory_with_absolute_path_and_trailing_slash(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("x")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert f"{subdir}/" in result

    def test_shows_nested_file_with_absolute_path(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        nested = subdir / "nested.txt"
        nested.write_text("x")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert str(nested) in result

    def test_excludes_hidden_directories(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".opendatasci"
        hidden.mkdir()
        (hidden / "secret.json").write_text("{}")
        (tmp_path / "visible.csv").write_text("data")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert ".opendatasci" not in result
        assert "secret.json" not in result

    def test_excludes_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden_file").write_text("hidden")
        (tmp_path / "visible.txt").write_text("visible")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert ".hidden_file" not in result

    def test_shows_file_size_in_bytes_for_small_file(self, tmp_path: Path) -> None:
        (tmp_path / "small.txt").write_bytes(b"hello")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert " B" in result

    def test_shows_file_size_in_kb_for_medium_file(self, tmp_path: Path) -> None:
        (tmp_path / "medium.dat").write_bytes(b"x" * 2048)
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert "KB" in result

    def test_shows_file_size_in_mb_for_large_file(self, tmp_path: Path) -> None:
        (tmp_path / "large.dat").write_bytes(b"x" * (2 * 1024 * 1024))
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert "MB" in result

    def test_multiple_files_all_listed_with_absolute_paths(self, tmp_path: Path) -> None:
        files = [tmp_path / name for name in ("a.csv", "b.txt", "c.json")]
        for f in files:
            f.write_text("data")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        for f in files:
            assert str(f) in result

    def test_nested_files_listed_with_absolute_paths(self, tmp_path: Path) -> None:
        subdir = tmp_path / "models"
        subdir.mkdir()
        model = subdir / "model.pkl"
        model.write_bytes(b"\x80")
        tool = _get_tool(tmp_path)
        result = tool.invoke({"summary": "Listing files", "communication": "listing"})
        assert str(model) in result
