"""Unit tests for opendatasci.tools.dataset_info."""


import ast
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from opendatasci.sandbox.base import SandboxExecResult
from opendatasci.tools.dataset_info import (
    build_profile_code,
    create_data_context_tools,
    create_profile_dataset_tools,
    create_read_dataset_info_tools,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> MagicMock:
    return MagicMock()


def _make_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.execute = AsyncMock()
    return sandbox


# ---------------------------------------------------------------------------
# build_profile_code
# ---------------------------------------------------------------------------


class TestBuildProfileCode:
    def test_returns_string(self) -> None:
        result = build_profile_code("/data/file.csv")
        assert isinstance(result, str)

    def test_injects_path_repr(self) -> None:
        result = build_profile_code("/data/sales.csv")
        assert repr("/data/sales.csv") in result

    def test_path_with_spaces_quoted_correctly(self) -> None:
        result = build_profile_code("/my data/file name.csv")
        assert repr("/my data/file name.csv") in result

    def test_path_with_single_quotes_escaped(self) -> None:
        result = build_profile_code("/it's/a/path.csv")
        assert "/it's/a/path.csv" in result

    def test_imports_pandas(self) -> None:
        result = build_profile_code("/data.csv")
        assert "import pandas as pd" in result

    def test_sets_result_variable(self) -> None:
        result = build_profile_code("/data.csv")
        assert "result" in result

    def test_contains_profile_skip_sentinel_handling(self) -> None:
        result = build_profile_code("/data.csv")
        assert "__PROFILE_SKIP__" in result

    def test_contains_shape_statistics(self) -> None:
        result = build_profile_code("/data.csv")
        assert "shape" in result or "_rows" in result

    def test_contains_memory_calculation(self) -> None:
        result = build_profile_code("/data.csv")
        assert "memory_usage" in result

    def test_contains_duplicate_row_check(self) -> None:
        result = build_profile_code("/data.csv")
        assert "duplicated" in result

    def test_contains_numeric_summary_section(self) -> None:
        result = build_profile_code("/data.csv")
        assert "Numeric Summary" in result

    def test_contains_top_categoricals_section(self) -> None:
        result = build_profile_code("/data.csv")
        assert "Top Categoricals" in result or "value_counts" in result

    def test_handles_csv_extension_explicitly(self) -> None:
        result = build_profile_code("/data.csv")
        assert ".csv" in result

    def test_handles_parquet_extension(self) -> None:
        result = build_profile_code("/data.parquet")
        assert "parquet" in result

    def test_different_paths_produce_different_code(self) -> None:
        code_a = build_profile_code("/path/a.csv")
        code_b = build_profile_code("/path/b.csv")
        assert code_a != code_b

    def test_same_path_produces_identical_code(self) -> None:
        code_a = build_profile_code("/same/path.csv")
        code_b = build_profile_code("/same/path.csv")
        assert code_a == code_b

    def test_generated_code_is_valid_python_syntax(self) -> None:
        code = build_profile_code("/data/test.csv")
        ast.parse(code)


# ---------------------------------------------------------------------------
# read_dataset_info
# ---------------------------------------------------------------------------


class TestReadDatasetInfoTool:
    @pytest.mark.asyncio
    async def test_no_context_returns_error(self) -> None:
        tool = create_read_dataset_info_tools(None)[0]
        result = await tool.ainvoke(
            {
                "path": "/some/path.csv",
                "summary": "Reading dataset info",
                "communication": "Let me read notes.",
            }
        )
        assert "Error" in result
        assert "workspace" in result.lower()

    @pytest.mark.asyncio
    async def test_success_returns_content(self) -> None:
        context = _make_context()
        context.read_dataset_info = AsyncMock(return_value="# DATASET NOTES\n\nsome notes")
        tool = create_read_dataset_info_tools(context)[0]
        result = await tool.ainvoke(
            {
                "path": "/data/file.csv",
                "summary": "Reading file.csv info",
                "communication": "Let me read notes.",
            }
        )
        assert "# DATASET NOTES" in result

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error_message(self) -> None:
        context = _make_context()
        context.read_dataset_info = AsyncMock(side_effect=FileNotFoundError("not found"))
        tool = create_read_dataset_info_tools(context)[0]
        result = await tool.ainvoke(
            {
                "path": "/missing.csv",
                "summary": "Reading missing.csv",
                "communication": "Let me read notes.",
            }
        )
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error_with_type(self) -> None:
        context = _make_context()
        context.read_dataset_info = AsyncMock(side_effect=ValueError("bad value"))
        tool = create_read_dataset_info_tools(context)[0]
        result = await tool.ainvoke(
            {
                "path": "/data.csv",
                "summary": "Reading data.csv",
                "communication": "Let me read notes.",
            }
        )
        assert "Error loading dataset info" in result
        assert "ValueError" in result

    @pytest.mark.asyncio
    async def test_delegates_path_to_context(self) -> None:
        context = _make_context()
        context.read_dataset_info = AsyncMock(return_value="content")
        tool = create_read_dataset_info_tools(context)[0]
        await tool.ainvoke(
            {
                "path": "/specific/data.csv",
                "summary": "Reading specific data",
                "communication": "Let me read notes.",
            }
        )
        context.read_dataset_info.assert_called_once_with("/specific/data.csv")


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------


class TestProfileDatasetTool:
    def test_no_context_returns_error(self) -> None:
        tool = create_profile_dataset_tools(None, _make_sandbox())[0]

        async def _run():
            return await tool.ainvoke(
                {
                    "path": "/data.csv",
                    "summary": "Profiling data.csv",
                    "communication": "Let me profile the dataset.",
                }
            )

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert "Error" in result
        assert "workspace" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_existing_profile_without_executing(self) -> None:
        context = _make_context()
        sandbox = _make_sandbox()
        context.get_profile_info = AsyncMock(
            return_value=(
                MagicMock(),
                "abc123",
                "# Cached Profile",
            )
        )
        tool = create_profile_dataset_tools(context, sandbox)[0]
        result = await tool.ainvoke(
            {
                "path": "/data.csv",
                "summary": "Profiling data.csv",
                "communication": "Let me profile the dataset.",
            }
        )
        assert "# Cached Profile" in result
        sandbox.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generates_and_saves_new_profile(self) -> None:
        context = _make_context()
        sandbox = _make_sandbox()
        context.get_profile_info = AsyncMock(
            return_value=(
                MagicMock(),
                "hash123",
                None,
            )
        )
        sandbox.execute.return_value = SandboxExecResult(success=True, output="# New Profile")
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "opendatasci.tools.dataset_info.build_profile_code", return_value="profile_code"
        ):
            tool = create_profile_dataset_tools(context, sandbox)[0]
            result = await tool.ainvoke(
                {
                    "path": "/data.csv",
                    "summary": "Profiling data.csv",
                    "communication": "Let me profile the dataset.",
                }
            )
        assert "# New Profile" in result
        context.save_dataset_profile.assert_called_once_with("hash123", "# New Profile")

    @pytest.mark.asyncio
    async def test_execution_failure_returns_error(self) -> None:
        context = _make_context()
        sandbox = _make_sandbox()
        context.get_profile_info = AsyncMock(return_value=(MagicMock(), "h", None))
        sandbox.execute.return_value = SandboxExecResult(success=False, error="sandbox error")
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "opendatasci.tools.dataset_info.build_profile_code", return_value="code"
        ):
            tool = create_profile_dataset_tools(context, sandbox)[0]
            result = await tool.ainvoke(
                {
                    "path": "/data.csv",
                    "summary": "Profiling data.csv",
                    "communication": "Let me profile the dataset.",
                }
            )
        assert "Profiling failed" in result
        assert "sandbox error" in result

    @pytest.mark.asyncio
    async def test_none_output_returns_empty_message(self) -> None:
        context = _make_context()
        sandbox = _make_sandbox()
        context.get_profile_info = AsyncMock(return_value=(MagicMock(), "h", None))
        sandbox.execute.return_value = SandboxExecResult(success=True, output=None)
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "opendatasci.tools.dataset_info.build_profile_code", return_value="code"
        ):
            tool = create_profile_dataset_tools(context, sandbox)[0]
            result = await tool.ainvoke(
                {
                    "path": "/data.csv",
                    "summary": "Profiling data.csv",
                    "communication": "Let me profile the dataset.",
                }
            )
        assert "no output" in result.lower()

    @pytest.mark.asyncio
    async def test_profile_skip_sentinel_returns_message_without_saving(self) -> None:
        context = _make_context()
        sandbox = _make_sandbox()
        context.get_profile_info = AsyncMock(return_value=(MagicMock(), "h", None))
        sandbox.execute.return_value = SandboxExecResult(
            success=True, output="__PROFILE_SKIP__Cannot load file (UnsupportedFormat)"
        )
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "opendatasci.tools.dataset_info.build_profile_code", return_value="code"
        ):
            tool = create_profile_dataset_tools(context, sandbox)[0]
            result = await tool.ainvoke(
                {
                    "path": "/data.csv",
                    "summary": "Profiling data.csv",
                    "communication": "Let me profile the dataset.",
                }
            )
        assert "Cannot load file" in result
        context.save_dataset_profile.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self) -> None:
        context = _make_context()
        context.get_profile_info = AsyncMock(side_effect=FileNotFoundError("missing"))
        tool = create_profile_dataset_tools(context, _make_sandbox())[0]
        result = await tool.ainvoke(
            {
                "path": "/missing.csv",
                "summary": "Profiling missing.csv",
                "communication": "Let me profile the dataset.",
            }
        )
        assert "Error" in result
        assert "missing" in result


# ---------------------------------------------------------------------------
# update_dataset_info (via create_data_context_tools)
# ---------------------------------------------------------------------------


class TestUpdateDatasetInfoTool:
    def _get_update_tool(self, context, sandbox=None):
        tools = create_data_context_tools(context, sandbox or _make_sandbox())
        return next(t for t in tools if t.name == "update_dataset_info")

    @pytest.mark.asyncio
    async def test_no_context_returns_error(self) -> None:
        tool = self._get_update_tool(None)
        result = await tool.ainvoke({"path": "/data.csv", "update": "notes"})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_success_returns_notes_file_path(self) -> None:
        context = _make_context()
        context.update_dataset_info = AsyncMock(
            return_value="/ws/.opendatasci/dataset_notes/2026/05/04/abc.md"
        )
        tool = self._get_update_tool(context)
        result = await tool.ainvoke({"path": "/data.csv", "update": "some notes"})
        assert "/ws/.opendatasci/dataset_notes" in result

    @pytest.mark.asyncio
    async def test_merge_true_by_default(self) -> None:
        context = _make_context()
        context.update_dataset_info = AsyncMock(return_value="/path.md")
        tool = self._get_update_tool(context)
        await tool.ainvoke({"path": "/data.csv", "update": "notes"})
        context.update_dataset_info.assert_called_once_with("/data.csv", "notes", merge=False)

    @pytest.mark.asyncio
    async def test_merge_false_passed_through(self) -> None:
        context = _make_context()
        context.update_dataset_info = AsyncMock(return_value="/path.md")
        tool = self._get_update_tool(context)
        await tool.ainvoke({"path": "/data.csv", "update": "replace", "merge": False})
        context.update_dataset_info.assert_called_once_with(
            "/data.csv", "replace", merge=False
        )

    @pytest.mark.asyncio
    async def test_file_not_found_returns_error(self) -> None:
        context = _make_context()
        context.update_dataset_info = AsyncMock(side_effect=FileNotFoundError("gone"))
        tool = self._get_update_tool(context)
        result = await tool.ainvoke({"path": "/gone.csv", "update": "x"})
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error_with_type(self) -> None:
        context = _make_context()
        context.update_dataset_info = AsyncMock(side_effect=OSError("disk full"))
        tool = self._get_update_tool(context)
        result = await tool.ainvoke({"path": "/data.csv", "update": "x"})
        assert "Error updating dataset info" in result
        assert "OSError" in result


# ---------------------------------------------------------------------------
# create_data_context_tools – structure
# ---------------------------------------------------------------------------


class TestGetContextTools:
    def test_returns_three_tools(self) -> None:
        tools = create_data_context_tools(_make_context(), _make_sandbox())
        assert len(tools) == 3

    def test_tool_names(self) -> None:
        names = {t.name for t in create_data_context_tools(_make_context(), _make_sandbox())}
        assert names == {"read_dataset_info", "profile_dataset", "update_dataset_info"}
