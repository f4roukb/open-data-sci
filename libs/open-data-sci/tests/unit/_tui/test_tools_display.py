"""Unit tests for opendatasci._tui.tools_display."""

import types

import pytest

from opendatasci._tui.tools_display import REGISTRY, ToolDisplay, _registry, register


# ---------------------------------------------------------------------------
# ToolDisplay dataclass
# ---------------------------------------------------------------------------


class TestToolDisplay:
    def test_label_required(self) -> None:
        display = ToolDisplay(label="My Tool")
        assert display.label == "My Tool"

    def test_icon_defaults_to_empty_string(self) -> None:
        display = ToolDisplay(label="Tool")
        assert display.icon == ""

    def test_summary_arg_defaults_to_none(self) -> None:
        display = ToolDisplay(label="Tool")
        assert display.summary_arg is None

    def test_display_defaults_to_true(self) -> None:
        display = ToolDisplay(label="Tool")
        assert display.display is True

    def test_custom_icon(self) -> None:
        display = ToolDisplay(label="Python", icon="🐍")
        assert display.icon == "🐍"

    def test_custom_summary_arg(self) -> None:
        display = ToolDisplay(label="Tool", summary_arg="summary")
        assert display.summary_arg == "summary"

    def test_display_false(self) -> None:
        display = ToolDisplay(label="Quiet Tool", display=False)
        assert display.display is False

    def test_is_frozen_immutable(self) -> None:
        display = ToolDisplay(label="Tool")
        with pytest.raises(Exception):
            display.label = "changed"  # type: ignore[misc]

    def test_equality_same_fields(self) -> None:
        assert ToolDisplay(label="A", icon="X") == ToolDisplay(label="A", icon="X")

    def test_inequality_different_fields(self) -> None:
        assert ToolDisplay(label="A") != ToolDisplay(label="B")


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_adds_to_registry(self) -> None:
        display = ToolDisplay(label="Test Tool")
        register("__test_tool__", display)
        assert "__test_tool__" in REGISTRY
        assert REGISTRY["__test_tool__"] == display

    def test_register_overwrites_existing(self) -> None:
        register("__overwrite_test__", ToolDisplay(label="First"))
        register("__overwrite_test__", ToolDisplay(label="Second"))
        assert REGISTRY["__overwrite_test__"].label == "Second"

    def test_register_does_not_affect_other_keys(self) -> None:
        before = set(REGISTRY.keys())
        register("__isolated_test__", ToolDisplay(label="Isolated"))
        after = set(REGISTRY.keys())
        assert after - before == {"__isolated_test__"}


# ---------------------------------------------------------------------------
# REGISTRY type and mutability contract
# ---------------------------------------------------------------------------


class TestRegistryType:
    def test_registry_is_mapping_proxy(self) -> None:
        assert isinstance(REGISTRY, types.MappingProxyType)

    def test_registry_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            REGISTRY["__direct_write__"] = ToolDisplay(label="X")  # type: ignore[index]

    def test_registry_is_live_view_of_backing_dict(self) -> None:
        key = "__live_view_test__"
        _registry.pop(key, None)
        assert key not in REGISTRY
        register(key, ToolDisplay(label="Live"))
        assert key in REGISTRY
        _registry.pop(key)

    def test_registry_reflects_backing_dict_deletion(self) -> None:
        key = "__deletion_test__"
        register(key, ToolDisplay(label="Temp"))
        assert key in REGISTRY
        del _registry[key]
        assert key not in REGISTRY


# ---------------------------------------------------------------------------
# Registry contents — spot-check every registered tool
# ---------------------------------------------------------------------------


class TestRegistryContents:
    # ── Code execution ────────────────────────────────────────────────────────

    def test_execute_python_label(self) -> None:
        assert REGISTRY["execute_python_code"].label == "Code"

    def test_execute_python_icon(self) -> None:
        assert REGISTRY["execute_python_code"].icon == "🐍"

    def test_execute_python_summary_arg(self) -> None:
        assert REGISTRY["execute_python_code"].summary_arg == "summary"

    def test_execute_python_display_true(self) -> None:
        assert REGISTRY["execute_python_code"].display is True

    def test_execute_cli_label(self) -> None:
        assert REGISTRY["execute_cli_command"].label == "Command"

    def test_execute_cli_icon(self) -> None:
        assert REGISTRY["execute_cli_command"].icon == "⌨️"

    def test_execute_cli_summary_arg(self) -> None:
        assert REGISTRY["execute_cli_command"].summary_arg == "summary"

    def test_list_python_libs_label(self) -> None:
        assert REGISTRY["list_python_libs"].label == "Checking available libraries"

    def test_list_python_libs_icon(self) -> None:
        assert REGISTRY["list_python_libs"].icon == "📦"

    def test_list_python_libs_no_summary_arg(self) -> None:
        assert REGISTRY["list_python_libs"].summary_arg is None

    # ── Planning ──────────────────────────────────────────────────────────────

    def test_enter_plan_mode_label(self) -> None:
        assert REGISTRY["enter_plan_mode"].label == "Planning"

    def test_enter_plan_mode_icon(self) -> None:
        assert REGISTRY["enter_plan_mode"].icon == "🎯"

    def test_exit_plan_mode_label(self) -> None:
        assert REGISTRY["exit_plan_mode"].label == "Done planning"

    def test_exit_plan_mode_icon(self) -> None:
        assert REGISTRY["exit_plan_mode"].icon == "✅"

    # ── Skills ────────────────────────────────────────────────────────────────

    def test_load_skill_label(self) -> None:
        assert REGISTRY["load_skill"].label == "Loading skill"

    def test_load_skill_icon(self) -> None:
        assert REGISTRY["load_skill"].icon == "🧠"

    def test_load_skill_summary_arg(self) -> None:
        assert REGISTRY["load_skill"].summary_arg == "summary"

    # ── Workers ───────────────────────────────────────────────────────────────

    def test_spawn_workers_label(self) -> None:
        assert REGISTRY["spawn_workers"].label == "Spawning workers"

    def test_spawn_workers_icon(self) -> None:
        assert REGISTRY["spawn_workers"].icon == "⚙️"

    def test_spawn_workers_no_summary_arg(self) -> None:
        assert REGISTRY["spawn_workers"].summary_arg is None

    # ── Dataset ───────────────────────────────────────────────────────────────

    def test_read_dataset_info_label(self) -> None:
        assert REGISTRY["read_dataset_info"].label == "Reading dataset info"

    def test_read_dataset_info_icon(self) -> None:
        assert REGISTRY["read_dataset_info"].icon == "📚"

    def test_read_dataset_info_summary_arg(self) -> None:
        assert REGISTRY["read_dataset_info"].summary_arg == "summary"

    def test_update_dataset_info_label(self) -> None:
        assert REGISTRY["update_dataset_info"].label == "Updating dataset notes"

    def test_update_dataset_info_display_false(self) -> None:
        assert REGISTRY["update_dataset_info"].display is False

    def test_profile_dataset_label(self) -> None:
        assert REGISTRY["profile_dataset"].label == "Profiling dataset"

    def test_profile_dataset_icon(self) -> None:
        assert REGISTRY["profile_dataset"].icon == "📊"

    # ── Workspace ─────────────────────────────────────────────────────────────

    def test_list_workspace_files_label(self) -> None:
        assert REGISTRY["list_workspace_files"].label == "Listing workspace files"

    def test_list_workspace_files_icon(self) -> None:
        assert REGISTRY["list_workspace_files"].icon == "📁"

    # ── Web ───────────────────────────────────────────────────────────────────

    def test_web_search_label(self) -> None:
        assert REGISTRY["web_search"].label == "Searching the web"

    def test_web_search_icon(self) -> None:
        assert REGISTRY["web_search"].icon == "🌐"

    def test_web_search_summary_arg(self) -> None:
        assert REGISTRY["web_search"].summary_arg == "summary"

    def test_fetch_url_label(self) -> None:
        assert REGISTRY["fetch_url"].label == "Fetching content"

    def test_fetch_url_icon(self) -> None:
        assert REGISTRY["fetch_url"].icon == "🔗"

    def test_fetch_url_summary_arg(self) -> None:
        assert REGISTRY["fetch_url"].summary_arg == "summary"

    # ── User interaction ──────────────────────────────────────────────────────

    def test_ask_user_mcq_label(self) -> None:
        assert REGISTRY["ask_user_mcq"].label == "Question"

    def test_ask_user_mcq_display_false(self) -> None:
        assert REGISTRY["ask_user_mcq"].display is False

    # ── Self-review ───────────────────────────────────────────────────────────

    def test_enter_self_review_mode_label(self) -> None:
        assert REGISTRY["enter_self_review_mode"].label == "Reviewing progress so far"

    def test_enter_self_review_mode_icon(self) -> None:
        assert REGISTRY["enter_self_review_mode"].icon == "🔍"

    def test_exit_self_review_mode_label(self) -> None:
        assert REGISTRY["exit_self_review_mode"].label == "Done reviewing progress"

    def test_verify_python_code_label(self) -> None:
        assert REGISTRY["verify_python_code"].label == "Reviewing code"

    def test_verify_python_code_icon(self) -> None:
        assert REGISTRY["verify_python_code"].icon == "🔎"


# ---------------------------------------------------------------------------
# Cross-module coverage: every ToolName must appear in REGISTRY
# ---------------------------------------------------------------------------


class TestAllToolNamesRegistered:
    def test_all_tool_names_have_display_entry(self) -> None:
        from opendatasci.tools.factory import ToolName

        missing = [name.value for name in ToolName if name.value not in REGISTRY]
        assert missing == [], f"Tools missing from REGISTRY: {missing}"

    def test_no_extra_entries_beyond_known_tools(self) -> None:
        """REGISTRY should not silently accumulate unrecognised entries.

        This acts as a guard against typos in tool name strings — a misspelled
        registration produces an orphan key that never matches a real tool.
        """
        from opendatasci.tools.factory import ToolName

        known = {name.value for name in ToolName}
        # Filter out test-only keys injected by other tests in this session.
        orphans = {
            k for k in REGISTRY if k not in known and not k.startswith("__")
        }
        assert orphans == set(), f"Unexpected orphan keys in REGISTRY: {orphans}"
