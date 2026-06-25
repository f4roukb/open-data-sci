"""Unit tests for opendatasci._tui.controller."""


from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.streaming import (
    AgentStreamEvent,
    ErrorEvent,
    ReasoningEvent,
    ResponseEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
    WorkerDoneEvent,
)
from opendatasci._tui.controller import (
    CLIController,
    PasteAttachment,
    _build_agent_query,
    _build_user_display,
    _discover_files,
    _FileRef,
    _find_at_fragment,
    _find_slash_fragment,
    _parse_file_refs,
    _split_existing_file_refs,
)
from opendatasci.configs import OpenDataSciConfig

# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------


class TestParseFileRefs:
    def test_plain_text_returns_no_refs(self) -> None:
        clean, refs = _parse_file_refs("hello world")
        assert clean == "hello world"
        assert refs == []

    def test_single_at_ref_stripped_from_text(self) -> None:
        clean, refs = _parse_file_refs("look at @README.md")
        assert "@README.md" not in clean
        assert len(refs) == 1
        assert refs[0]._path == "README.md"

    def test_at_ref_no_line_range_syntax(self) -> None:
        # The :L1-L2 syntax is no longer supported; the colon becomes part
        # of the path token and is parsed as a single ref with a colon in its name.
        clean, refs = _parse_file_refs("see @src/foo.py")
        assert len(refs) == 1
        assert refs[0]._path == "src/foo.py"

    def test_multiple_refs_all_captured(self) -> None:
        clean, refs = _parse_file_refs("@a.py and @b.py")
        assert len(refs) == 2
        assert {r._path for r in refs} == {"a.py", "b.py"}

    def test_only_ref_no_text_gives_empty_clean(self) -> None:
        clean, refs = _parse_file_refs("@file.csv")
        assert clean == ""
        assert len(refs) == 1


class TestFileRefProperties:
    def test_display_name_returns_basename(self) -> None:
        ref = _FileRef("path/to/file.py")
        assert ref.display_name == "file.py"

    def test_display_name_simple_filename(self) -> None:
        ref = _FileRef("data.csv")
        assert ref.display_name == "data.csv"


class TestBuildUserDisplay:
    def test_plain_text_escaped(self) -> None:
        result = _build_user_display("hello [world]", [])
        # Rich escape_markup prepends \ before [, preventing markup interpretation
        assert r"\[world]" in result
        assert "hello" in result

    def test_refs_shown_as_filename_pills(self) -> None:
        ref = _FileRef("data.csv")
        result = _build_user_display("", [ref])
        assert "data.csv" in result

    def test_ref_and_text_combined(self) -> None:
        ref = _FileRef("foo.py")
        result = _build_user_display("check this", [ref])
        assert "foo.py" in result
        assert "check this" in result


class TestBuildAgentQuery:
    def test_no_refs_returns_clean_text(self) -> None:
        result = _build_agent_query("what is this?", [])
        assert result == "what is this?"

    def test_with_refs_adds_file_attachment_tags(self) -> None:
        ref = _FileRef("data.csv")
        result = _build_agent_query("analyse", [ref])
        assert "<file_attachment" in result
        assert "data.csv" in result
        assert "analyse" in result

    def test_no_lines_attr_in_tag(self) -> None:
        ref = _FileRef("foo.py")
        result = _build_agent_query("", [ref])
        assert "lines=" not in result

    def test_empty_text_only_attachment(self) -> None:
        ref = _FileRef("x.py")
        result = _build_agent_query("", [ref])
        assert result.startswith("<file_attachment")


class TestSplitExistingFileRefs:
    def test_existing_file_goes_to_existing(self, tmp_path: Path) -> None:
        f = tmp_path / "real.csv"
        f.write_text("a,b")
        ref = _FileRef(str(f))
        existing, missing = _split_existing_file_refs([ref])
        assert ref in existing
        assert missing == []

    def test_missing_file_goes_to_missing(self) -> None:
        ref = _FileRef("/no/such/file.csv")
        existing, missing = _split_existing_file_refs([ref])
        assert existing == []
        assert ref in missing

    def test_directory_treated_as_missing(self, tmp_path: Path) -> None:
        ref = _FileRef(str(tmp_path))
        existing, missing = _split_existing_file_refs([ref])
        assert existing == []
        assert ref in missing


# ---------------------------------------------------------------------------
# Fragment detection helpers
# ---------------------------------------------------------------------------


class TestFindSlashFragment:
    def test_slash_prefix_returned(self) -> None:
        assert _find_slash_fragment("/cle") == "/cle"

    def test_full_command_returned(self) -> None:
        assert _find_slash_fragment("/clear") == "/clear"

    def test_slash_with_space_returns_none(self) -> None:
        assert _find_slash_fragment("/clear foo") is None

    def test_no_slash_returns_none(self) -> None:
        assert _find_slash_fragment("hello") is None

    def test_empty_string_returns_none(self) -> None:
        assert _find_slash_fragment("") is None


class TestFindAtFragment:
    def test_at_prefix_returns_fragment_and_position(self) -> None:
        result = _find_at_fragment("@dat")
        assert result is not None
        fragment, pos = result
        assert fragment == "dat"
        assert pos == 0

    def test_at_in_middle_of_text(self) -> None:
        result = _find_at_fragment("look at @file")
        assert result is not None
        fragment, pos = result
        assert fragment == "file"
        assert pos == 8

    def test_colon_in_fragment_still_returns_result(self) -> None:
        # The :L1-L2 syntax is gone; colons are now treated as part of the
        # fragment (completion will simply find no matching files and hide).
        result = _find_at_fragment("@file.py:L1")
        assert result is not None
        fragment, pos = result
        assert fragment == "file.py:L1"
        assert pos == 0

    def test_no_at_returns_none(self) -> None:
        assert _find_at_fragment("no reference here") is None

    def test_fragment_stops_at_space(self) -> None:
        result = _find_at_fragment("@foo bar")
        assert result is not None
        fragment, _ = result
        assert fragment == "foo"


class TestDiscoverFiles:
    def test_finds_files_in_cwd(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.csv").write_text("")
        (tmp_path / "beta.csv").write_text("")
        with patch("opendatasci._tui.controller.Path.cwd", return_value=tmp_path):
            matches = _discover_files("")
        assert "alpha.csv" in matches
        assert "beta.csv" in matches

    def test_filters_by_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.csv").write_text("")
        (tmp_path / "beta.csv").write_text("")
        with patch("opendatasci._tui.controller.Path.cwd", return_value=tmp_path):
            matches = _discover_files("al")
        assert "alpha.csv" in matches
        assert "beta.csv" not in matches

    def test_hidden_files_excluded_by_default(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").write_text("")
        (tmp_path / "visible.txt").write_text("")
        with patch("opendatasci._tui.controller.Path.cwd", return_value=tmp_path):
            matches = _discover_files("")
        assert "visible.txt" in matches
        assert ".hidden" not in matches

    def test_hidden_files_included_when_prefix_is_dot(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("")
        with patch("opendatasci._tui.controller.Path.cwd", return_value=tmp_path):
            matches = _discover_files(".")
        assert ".env" in matches

    def test_directories_get_trailing_slash(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        with patch("opendatasci._tui.controller.Path.cwd", return_value=tmp_path):
            matches = _discover_files("")
        assert "subdir/" in matches

    def test_non_existent_search_dir_returns_empty(self) -> None:
        matches = _discover_files("nonexistent_dir/file")
        assert matches == []

    def test_capped_at_ten_results(self, tmp_path: Path) -> None:
        for i in range(15):
            (tmp_path / f"file{i}.csv").write_text("")
        with patch("opendatasci._tui.controller.Path.cwd", return_value=tmp_path):
            matches = _discover_files("")
        assert len(matches) == 10


# ---------------------------------------------------------------------------
# PasteAttachment — data model
# ---------------------------------------------------------------------------


class TestPasteAttachment:
    def test_line_count_single_line(self) -> None:
        a = PasteAttachment("hello")
        assert a._line_count == 1

    def test_line_count_multiline(self) -> None:
        a = PasteAttachment("line1\nline2\nline3")
        assert a._line_count == 3

    def test_display_label_singular(self) -> None:
        a = PasteAttachment("only one line")
        assert a.display_label == "Text: 1 line"

    def test_display_label_plural(self) -> None:
        a = PasteAttachment("a\nb")
        assert a.display_label == "Text: 2 lines"

    def test_pill_markup_contains_label(self) -> None:
        a = PasteAttachment("x\ny\nz")
        assert "Text: 3 lines" in a.pill_markup

    def test_xml_tag_wraps_content(self) -> None:
        a = PasteAttachment("def foo():\n    pass")
        assert a.xml_tag.startswith("<pasted_content>")
        assert "def foo():" in a.xml_tag
        assert a.xml_tag.endswith("</pasted_content>")

    def test_content_preserved_exactly(self) -> None:
        text = "a\nb\nc"
        a = PasteAttachment(text)
        assert a._content == text


# ---------------------------------------------------------------------------
# CLIController — paste attachment
# ---------------------------------------------------------------------------


class TestPasteAttachment_Controller:
    def test_on_paste_stores_attachment(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("line1\nline2")
        assert controller._paste_attachment is not None
        assert controller._paste_attachment._line_count == 2

    def test_on_paste_calls_show_attachment(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("a\nb\nc")
        mock_ui.show_attachment.assert_called_once_with("Text: 3 lines")

    def test_clear_paste_attachment_removes_it(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("a\nb")
        controller.clear_paste_attachment()
        assert controller._paste_attachment is None
        mock_ui.hide_attachment.assert_called_once()

    def test_clear_paste_attachment_noop_when_none(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.clear_paste_attachment()  # should not raise
        mock_ui.hide_attachment.assert_not_called()

    async def test_on_submit_includes_attachment_in_query(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("def foo():\n    pass")
        action, query = await controller.on_submit("explain this")
        assert action == "run"
        assert "<pasted_content>" in query
        assert "def foo():" in query

    async def test_on_submit_pill_shown_in_display(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("x\ny")
        await controller.on_submit("look")
        # The user bubble is the last add_message("user", ...) call.
        user_calls = [c for c in mock_ui.add_message.call_args_list if c[0][0] == "user"]
        assert user_calls, "No user message was added"
        display_arg = user_calls[-1][0][1]
        assert "Text: 2 lines" in display_arg

    async def test_on_submit_clears_attachment_afterward(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("a\nb")
        await controller.on_submit("query")
        assert controller._paste_attachment is None

    async def test_on_submit_attachment_only_no_text(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("code\nhere")
        action, query = await controller.on_submit("")
        assert action == "run"
        assert "<pasted_content>" in query

    async def test_on_submit_hides_attachment_bar(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_paste("a\nb")
        await controller.on_submit("q")
        mock_ui.hide_attachment.assert_called()


# ---------------------------------------------------------------------------
# CLIController — input change handling
# ---------------------------------------------------------------------------


class TestOnInputChanged:
    def test_completing_flag_skips_and_resets(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._completing = True
        result = controller.on_input_changed("anything")
        assert result is True
        assert controller._completing is False
        mock_ui.show_completion.assert_not_called()

    def test_slash_fragment_shows_completion(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.on_input_changed("/cl")
        mock_ui.show_completion.assert_called_once()
        assert controller._comp_mode == "slash"

    def test_exact_slash_command_hides_completion(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        # Simulate the popup being visible from a previous partial match.
        controller._comp_matches = ["/clear", "/compact"]
        controller.on_input_changed("/clear")
        mock_ui.hide_completion.assert_called()

    def test_at_fragment_with_matches_shows_completion(
        self, controller: CLIController, mock_ui: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "data.csv").write_text("")
        with patch("opendatasci._tui.completion._discover_files", return_value=["data.csv"]):
            controller.on_input_changed("@data")
        mock_ui.show_completion.assert_called_once()
        assert controller._comp_mode == "file"

    def test_at_fragment_no_matches_hides_completion(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        # Simulate the popup being visible from a previous @-scan that had results.
        controller._comp_matches = ["data.csv"]
        with patch("opendatasci._tui.completion._discover_files", return_value=[]):
            controller.on_input_changed("@nonexistent")
        mock_ui.hide_completion.assert_called()

    def test_plain_text_hides_completion(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        # Simulate the popup being visible before the user switches to plain text.
        controller._comp_matches = ["data.csv"]
        controller.on_input_changed("hello world")
        mock_ui.hide_completion.assert_called()

    def test_no_ui_call_when_completion_not_showing(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        # Popup was never shown; hide_completion on the UI adapter must NOT be
        # called so we avoid a gratuitous CompletionPopup.update("") re-render on
        # every regular keystroke.
        controller.on_input_changed("hello world")
        mock_ui.hide_completion.assert_not_called()


# ---------------------------------------------------------------------------
# CLIController — completion
# ---------------------------------------------------------------------------


class TestCompletion:
    def test_has_completion_matches_false_when_empty(self, controller: CLIController) -> None:
        assert controller.has_completion_matches is False

    def test_has_completion_matches_true_when_populated(self, controller: CLIController) -> None:
        controller._comp_matches = ["/clear", "/reset"]
        assert controller.has_completion_matches is True

    def test_cycle_completion_no_matches_returns_false(self, controller: CLIController) -> None:
        assert controller.cycle_completion("", direction=1) is False

    def test_cycle_completion_slash_mode_sets_input(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._comp_matches = ["/clear", "/compact"]
        controller._comp_displays = ["/clear  ...", "/compact  ..."]
        controller._comp_mode = "slash"
        controller.cycle_completion("", direction=1)
        mock_ui.set_input_value.assert_called_once_with("/clear", 6)
        assert controller._completing is True

    def test_cycle_completion_up_from_start_wraps_to_last(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._comp_matches = ["/clear", "/compact", "/help"]
        controller._comp_displays = controller._comp_matches
        controller._comp_mode = "slash"
        controller._comp_idx = -1
        controller.cycle_completion("", direction=-1)
        assert controller._comp_idx == 2  # wraps to last

    def test_cycle_completion_file_mode_updates_input(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._comp_matches = ["data.csv"]
        controller._comp_mode = "file"
        controller._comp_at_pos = 0  # @ at position 0
        controller.cycle_completion("@", direction=1)
        mock_ui.set_input_value.assert_called_once_with("@data.csv", 9)

    def test_hide_completion_clears_all_state(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._comp_matches = ["/clear"]
        controller._comp_displays = ["/clear  ..."]
        controller._comp_idx = 0
        controller._comp_at_pos = 2
        controller._comp_mode = "slash"
        controller.hide_completion()
        assert controller._comp_matches == []
        assert controller._comp_idx == -1
        assert controller._comp_at_pos == -1
        assert controller._comp_mode == "file"
        mock_ui.hide_completion.assert_called_once()

    def test_hide_completion_swallows_ui_error(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        mock_ui.hide_completion.side_effect = RuntimeError("boom")
        controller.hide_completion()  # should not raise


# ---------------------------------------------------------------------------
# CLIController — submit
# ---------------------------------------------------------------------------


class TestOnSubmit:
    async def test_empty_submit_returns_noop(self, controller: CLIController) -> None:
        action, payload = await controller.on_submit("")
        assert action == ""
        assert payload == ""

    async def test_slash_quit_returns_quit(self, controller: CLIController) -> None:
        action, _ = await controller.on_submit("/exit")
        assert action == "quit"

    async def test_slash_clear_returns_empty_action(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        action, _ = await loaded_controller.on_submit("/clear")
        assert action == ""
        mock_ui.clear_messages.assert_called()

    async def test_unknown_slash_command_shows_error(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await controller.on_submit("/nonexistent")
        mock_ui.add_message.assert_called()
        call_args = mock_ui.add_message.call_args
        assert "Unknown command" in call_args[0][1]

    async def test_agent_running_queues_message_instead_of_running(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._agent_running = True
        action, _ = await controller.on_submit("do something")
        assert action == ""
        mock_ui.set_input_placeholder.assert_not_called()

    async def test_agent_running_pins_message_in_ui(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._agent_running = True
        await controller.on_submit("do something")
        mock_ui.add_pending_message.assert_called_once_with("do something")
        assert len(controller._pending_queue) == 1

    async def test_normal_query_returns_run(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        action, query = await controller.on_submit("analyse the data")
        assert action == "run"
        assert query == "analyse the data"
        mock_ui.add_message.assert_called_with("user", "analyse the data")

    async def test_query_with_missing_ref_shows_warning(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        action, _ = await controller.on_submit("@/nonexistent/ghost.csv")
        mock_ui.add_message.assert_called()
        warning_calls = [
            c for c in mock_ui.add_message.call_args_list if "File not found" in str(c)
        ]
        assert warning_calls

    async def test_awaiting_choice_empty_is_noop(self, controller: CLIController) -> None:
        controller._awaiting_choice = True
        action, _ = await controller.on_submit("")
        assert action == ""

    async def test_awaiting_choice_quit_slash_exits(self, controller: CLIController) -> None:
        controller._awaiting_choice = True
        controller._pending_choices = ["yes", "no"]
        action, _ = await controller.on_submit("/exit")
        assert action == "quit"

    async def test_awaiting_choice_routes_answer_as_run(
        self, controller: CLIController, mock_service: MagicMock
    ) -> None:
        controller._service = mock_service
        controller._awaiting_choice = True
        controller._pending_choices = ["yes", "no"]
        controller._other_choice_label = None
        action, payload = await controller.on_submit("A")
        assert action == "run"
        assert payload == "yes"


# ---------------------------------------------------------------------------
# CLIController — actions
# ---------------------------------------------------------------------------


class TestReset:
    async def test_reset_no_service_shows_not_loaded(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await controller.reset()
        mock_ui.add_message.assert_called_with("agent", "Not loaded yet.")

    async def test_reset_with_service_resets(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        await loaded_controller.reset()
        mock_service.reset_session.assert_awaited_once()
        msg_calls = [c[0][1] for c in mock_ui.add_message.call_args_list]
        assert any("reset" in m.lower() for m in msg_calls)

    async def test_reset_failure_shows_error(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        mock_service.reset_session.side_effect = RuntimeError("disk full")
        await loaded_controller.reset()
        msg_calls = [c[0][1] for c in mock_ui.add_message.call_args_list]
        assert any("Reset failed" in m for m in msg_calls)


class TestClearConv:
    async def test_clear_conv_always_clears_messages(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await controller.clear_conv()
        mock_ui.clear_messages.assert_called_once()

    async def test_clear_conv_calls_service_clear_context(
        self, loaded_controller: CLIController, mock_service: MagicMock
    ) -> None:
        await loaded_controller.clear_conv()
        mock_service.clear_context.assert_awaited_once()

    async def test_clear_conv_swallows_service_error(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        mock_service.clear_context.side_effect = RuntimeError("oops")
        await loaded_controller.clear_conv()  # should not raise
        mock_ui.clear_messages.assert_called_once()


class TestCompact:
    async def test_compact_no_service_shows_not_loaded(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await controller.compact()
        mock_ui.add_message.assert_called_with("agent", "Not loaded yet.")

    async def test_compact_success_shows_confirmation_without_summary(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        mock_service.compact_chat_history = AsyncMock(return_value="key findings")
        await loaded_controller.compact()
        msg_calls = [c[0][1] for c in mock_ui.add_message.call_args_list]
        assert not any("key findings" in m for m in msg_calls)
        assert any("Compaction done" in m for m in msg_calls)

    async def test_compact_failure_shows_error(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        mock_service.compact_chat_history = AsyncMock(side_effect=RuntimeError("timeout"))
        status_handle = mock_ui.add_message.return_value
        await loaded_controller.compact()
        # The error is set via set_content on the existing status bubble, not add_message
        calls = [str(c) for c in status_handle.set_content.call_args_list]
        assert any("Compact failed" in c for c in calls)


class TestShowHelp:
    def test_show_help_includes_all_commands(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.show_help()
        content = mock_ui.add_message.call_args[0][1]
        for cmd in [
            "/help",
            "/clear",
            "/compact",
            "/ls-workspace",
            "/models",
            "/exit",
            "/reset",
            "/stop",
            "/themes",
        ]:
            assert cmd in content


class TestShowThemes:
    def test_show_themes_lists_all_palettes_and_marks_active(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        from opendatasci._tui import theme as _theme

        _theme.active_name = "dracula"
        try:
            controller.show_themes()
        finally:
            _theme.active_name = "default"
        content = mock_ui.add_message.call_args[0][1]
        for name in _theme.THEMES:
            assert name in content
        assert "*(active)*" in content
        # The active marker must sit on the dracula line specifically.
        dracula_line = next(line for line in content.splitlines() if "dracula" in line)
        assert "*(active)*" in dracula_line


class TestShowModels:
    def test_show_models_shows_model_info(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller.show_models()
        content = mock_ui.add_message.call_args[0][1]
        assert "Model" in content
        assert "Secondary Model" in content

    def test_show_models_uses_stored_cfg(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        cfg = MagicMock(spec=OpenDataSciConfig)
        cfg.provider = "anthropic"
        cfg.model = "claude-sonnet-4-6"
        cfg.secondary_provider = "anthropic"
        cfg.secondary_model = "claude-haiku-4-5"
        controller._cfg = cfg
        controller.show_models()
        content = mock_ui.add_message.call_args[0][1]
        assert "Claude" in content


class TestLsWorkspace:
    def test_ls_workspace_no_service(self, controller: CLIController, mock_ui: MagicMock) -> None:
        controller.ls_workspace()
        mock_ui.add_message.assert_called_with("agent", "_Not loaded yet._")

    def test_ls_workspace_shows_panel(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        mock_service.get_workspace_files.return_value = ["a.csv", "b.csv"]
        loaded_controller.ls_workspace()
        mock_ui.show_workspace_panel.assert_called_once_with(["a.csv", "b.csv"])

    def test_ls_workspace_error_shows_message(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        mock_service.get_workspace_files.side_effect = RuntimeError("permission denied")
        loaded_controller.ls_workspace()
        content = mock_ui.add_message.call_args[0][1]
        assert "permission denied" in content


# ---------------------------------------------------------------------------
# CLIController — choice handling
# ---------------------------------------------------------------------------


class TestChoiceHandling:
    def test_show_choice_prompt_enters_awaiting_state(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._show_choice_prompt("Pick one?", ["Alpha", "Beta"])
        assert controller._awaiting_choice is True
        assert controller._pending_choices == ["Alpha", "Beta"]
        mock_ui.add_input_class.assert_called_with("awaiting-choice")

    def test_handle_user_choice_by_letter_resolves_to_text(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._pending_choices = ["yes", "no"]
        controller._awaiting_choice = True
        controller._other_choice_label = None
        result = controller._handle_user_choice("A")
        assert result == "yes"

    def test_handle_user_choice_freeform_text_sent_as_is(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._pending_choices = ["yes", "no"]
        controller._awaiting_choice = True
        controller._other_choice_label = None
        result = controller._handle_user_choice("maybe later")
        assert result == "maybe later"

    def test_handle_user_choice_other_label_prompts_custom_input(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._pending_choices = ["yes", "no"]
        controller._awaiting_choice = True
        controller._other_choice_label = "C"
        controller._awaiting_custom_choice_input = False
        controller._handle_user_choice("C")
        assert controller._awaiting_custom_choice_input is True

    def test_cancel_choice_exits_mode_and_returns_cancel(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._awaiting_choice = True
        controller._pending_choices = ["yes"]
        result = controller.cancel_choice()
        assert controller._awaiting_choice is False
        assert result == "cancel"

    def test_cancel_choice_no_op_when_not_awaiting(
        self, controller: CLIController
    ) -> None:
        controller._awaiting_choice = False
        result = controller.cancel_choice()
        assert result is None


# ---------------------------------------------------------------------------
# CLIController — run_agent streaming
# ---------------------------------------------------------------------------


async def _aiter(*events: AgentStreamEvent):
    for e in events:
        yield e


class TestRunAgent:
    async def test_run_agent_no_service_shows_warning(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await controller.run_agent("test")
        call = mock_ui.add_message.call_args
        assert "Still loading" in call[0][1]

    async def test_run_agent_token_event_opens_bubble(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        event = TokenEvent(content="Hello")
        mock_service.astream.return_value = _aiter(event)
        await loaded_controller.run_agent("hi")
        mock_ui.add_message.assert_any_call("agent", "")

    async def test_run_agent_response_event_adds_final_bubble(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        event = ResponseEvent(content="Final answer")
        mock_service.astream.return_value = _aiter(event)
        await loaded_controller.run_agent("query")
        mock_ui.add_message.assert_any_call("agent", "Final answer")

    async def test_run_agent_error_event_appends_error(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        event = ErrorEvent(content="something went wrong")
        mock_service.astream.return_value = _aiter(event)
        handle = mock_ui.add_message.return_value
        await loaded_controller.run_agent("query")
        appended = "".join(str(c) for c in handle.append.call_args_list)
        assert "something went wrong" in appended

    async def test_run_agent_exception_sets_error_content(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        async def _raise():
            raise RuntimeError("boom")
            yield  # make it a generator

        mock_service.astream.return_value = _raise()
        handle = mock_ui.add_message.return_value
        await loaded_controller.run_agent("query")
        handle.set_content.assert_called()
        content = handle.set_content.call_args[0][0]
        assert "boom" in content

    async def test_run_agent_resets_agent_running_flag_on_finish(
        self, loaded_controller: CLIController, mock_service: MagicMock
    ) -> None:
        mock_service.astream.return_value = _aiter()
        await loaded_controller.run_agent("q")
        assert loaded_controller._agent_running is False

    async def test_run_agent_stops_turn_status_bar_on_finish(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        bar = mock_ui.add_turn_status_bar.return_value
        loaded_controller._active_turn_status = bar
        mock_service.astream.return_value = _aiter()
        await loaded_controller.run_agent("q")
        bar.stop.assert_called()

    async def test_run_agent_usage_event_updates_context(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        bar = mock_ui.add_turn_status_bar.return_value
        loaded_controller._active_turn_status = bar
        event = UsageEvent(input_tokens=1200, output_tokens=300, cache_read_tokens=600)
        mock_service.astream.return_value = _aiter(event)
        await loaded_controller.run_agent("q")
        bar.update_context.assert_called_once_with(1500, 600)

    async def test_run_agent_tool_call_creates_ephemeral(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        event = ToolCallEvent(tool="execute_python_code", tool_call_id="tc1", summary="sum")
        mock_service.astream.return_value = _aiter(event)
        await loaded_controller.run_agent("q")
        mock_ui.add_ephemeral_block.assert_called()

    async def test_run_agent_tool_result_marks_ephemeral_done(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        eph = mock_ui.add_ephemeral_block.return_value
        events = [
            ToolCallEvent(tool="execute_python_code", tool_call_id="tc1"),
            ToolResultEvent(tool_call_id="tc1"),
        ]
        mock_service.astream.return_value = _aiter(*events)
        await loaded_controller.run_agent("q")
        eph.set_done.assert_called()

    async def test_run_agent_reasoning_event_creates_thinking_bubble(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        event = ReasoningEvent(content="thinking…")
        mock_service.astream.return_value = _aiter(event)
        await loaded_controller.run_agent("q")
        mock_ui.add_thinking_block.assert_called()

    async def test_run_agent_adds_divider_at_end(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        mock_service.astream.return_value = _aiter()
        await loaded_controller.run_agent("q")
        mock_ui.add_divider.assert_called_once()

    async def test_run_agent_spawn_workers_block_marked_done_on_tool_result(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        """Worker block turns green (set_done) when spawn_workers tool_result arrives."""
        wb = mock_ui.add_worker_block.return_value
        events = [
            ToolCallEvent(tool="spawn_workers", tool_call_id="sw1", worker_summaries=["Task A", "Task B"]),
            WorkerDoneEvent(worker_idx=0, success=True),
            WorkerDoneEvent(worker_idx=1, success=True),
            ToolResultEvent(content="done", tool_call_id="sw1"),
        ]
        mock_service.astream.return_value = _aiter(*events)
        await loaded_controller.run_agent("q")
        wb.set_done.assert_called()

    async def test_run_agent_spawn_workers_worker_done_updates_block(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        """Each worker_done event calls mark_worker_done on the worker block."""
        wb = mock_ui.add_worker_block.return_value
        events = [
            ToolCallEvent(tool="spawn_workers", tool_call_id="sw1", worker_summaries=["Task A", "Task B"]),
            WorkerDoneEvent(worker_idx=0, success=True),
            WorkerDoneEvent(worker_idx=1, success=True),
            ToolResultEvent(content="done", tool_call_id="sw1"),
        ]
        mock_service.astream.return_value = _aiter(*events)
        await loaded_controller.run_agent("q")
        wb.mark_worker_done.assert_any_call(0)
        wb.mark_worker_done.assert_any_call(1)

    async def test_run_agent_spawn_workers_worker_done_not_lost_when_parallel_tool_result_fires_first(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        """Regression: when a parallel tool's tool_result fires before worker_done events,
        the worker block must still receive mark_worker_done for each completing worker.

        Without the fix, the tool_result handler reset _worker_block to None, causing
        subsequent worker_done events to be silently dropped — leaving the block blue.
        """
        wb = mock_ui.add_worker_block.return_value
        # Simulate: Tool A and spawn_workers called in parallel.
        # Tool A's tool_result arrives BEFORE the worker_done events (Tool A was faster).
        events = [
            ToolCallEvent(tool="execute_python_code", tool_call_id="tc_a"),
            ToolCallEvent(tool="spawn_workers", tool_call_id="sw1", worker_summaries=["Task A", "Task B"]),
            # Tool A finishes first — its tool_result arrives before worker_done
            ToolResultEvent(content="result", tool_call_id="tc_a"),
            # Workers complete after Tool A's result
            WorkerDoneEvent(worker_idx=0, success=True),
            WorkerDoneEvent(worker_idx=1, success=True),
            ToolResultEvent(content="done", tool_call_id="sw1"),
        ]
        mock_service.astream.return_value = _aiter(*events)
        await loaded_controller.run_agent("q")
        # Both workers must be individually marked done despite Tool A's result firing first
        wb.mark_worker_done.assert_any_call(0)
        wb.mark_worker_done.assert_any_call(1)
        # And the block itself must be set done
        wb.set_done.assert_called()


# ---------------------------------------------------------------------------
# CLIController — stop_agent
# ---------------------------------------------------------------------------


class TestStopAgent:
    async def test_stop_agent_not_running_shows_info_message(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._agent_running = False
        await controller.stop_agent()
        content = mock_ui.add_message.call_args[0][1]
        assert "No agent is currently running" in content

    async def test_stop_agent_not_running_does_not_call_ui_stop(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._agent_running = False
        await controller.stop_agent()
        mock_ui.stop_agent.assert_not_called()

    async def test_stop_agent_when_running_calls_ui_stop(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        loaded_controller._agent_running = True
        await loaded_controller.stop_agent()
        mock_ui.stop_agent.assert_called_once()

    async def test_stop_agent_when_running_rolls_back_turn(
        self, loaded_controller: CLIController, mock_service: MagicMock, mock_ui: MagicMock
    ) -> None:
        loaded_controller._agent_running = True
        await loaded_controller.stop_agent()
        mock_service.rewind_turn.assert_awaited_once()

    async def test_stop_agent_when_running_shows_stopped_message(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        loaded_controller._agent_running = True
        await loaded_controller.stop_agent()
        contents = [c[0][1] for c in mock_ui.add_message.call_args_list]
        assert any("stopped" in c.lower() for c in contents)

    async def test_stop_agent_no_service_still_cancels_ui(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._agent_running = True
        controller._service = None
        await controller.stop_agent()
        mock_ui.stop_agent.assert_called_once()

    async def test_slash_stop_dispatched_when_agent_running(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        loaded_controller._agent_running = True
        action, _ = await loaded_controller.on_submit("/stop")
        assert action == ""
        mock_ui.stop_agent.assert_called_once()

    async def test_slash_stop_dispatched_when_agent_idle(
        self, controller: CLIController, mock_ui: MagicMock
    ) -> None:
        controller._agent_running = False
        action, _ = await controller.on_submit("/stop")
        assert action == ""
        content = mock_ui.add_message.call_args[0][1]
        assert "No agent is currently running" in content


# ---------------------------------------------------------------------------
# CLIController — lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_close_no_service_is_noop(self, controller: CLIController) -> None:
        await controller.close()  # should not raise

    async def test_close_with_service_calls_close(
        self, loaded_controller: CLIController, mock_service: MagicMock
    ) -> None:
        await loaded_controller.close()
        mock_service.close.assert_called_once()

    def test_awaiting_choice_property(self, controller: CLIController) -> None:
        assert controller.awaiting_choice is False
        controller._awaiting_choice = True
        assert controller.awaiting_choice is True


# ---------------------------------------------------------------------------
# CLIController — session ID
# ---------------------------------------------------------------------------


class TestSessionId:
    def test_session_id_stored_on_controller(self) -> None:
        ui = MagicMock()
        ctrl = CLIController(
            ui=ui,
            workspace_path="/fake/data.csv",
            datasci_config=OpenDataSciConfig(provider="anthropic", model="claude-sonnet-4-6"),
            session_id="deadbeef",
        )
        assert ctrl._session_id == "deadbeef"

    async def test_boot_wires_agent_from_create_agent_into_service(self, mock_ui: MagicMock) -> None:
        # boot() now delegates agent construction (including the session context
        # store) to create_agent(), enters it as an async context manager, and
        # wraps the agent + its sandbox in the TUI service.
        ctrl = CLIController(
            ui=mock_ui,
            workspace_path="/fake/data.csv",
            datasci_config=OpenDataSciConfig(provider="anthropic", model="claude-sonnet-4-6"),
            session_id="cafebabe",
        )

        mock_workspace = MagicMock()
        mock_workspace.get_reference.return_value = "/tmp/fake_workspace"
        mock_sandbox = MagicMock()
        mock_sandbox.get_history = MagicMock(return_value=[])
        # The agent acts as its own async context manager.
        mock_agent = MagicMock()
        mock_agent.__aenter__ = AsyncMock(return_value=mock_agent)
        mock_agent.__aexit__ = AsyncMock(return_value=None)
        mock_agent._workspace = mock_workspace
        mock_agent._sandbox = mock_sandbox
        mock_create_agent = MagicMock(return_value=mock_agent)
        mock_service_instance = MagicMock()
        mock_service_cls = MagicMock(return_value=mock_service_instance)
        fake_info = MagicMock(
            is_directory=False, workspaces=[{"name": "data.csv"}], workspace_count=1
        )

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.is_dir", return_value=False),
            patch("opendatasci._tui.controller.create_agent", mock_create_agent),
            patch("opendatasci._tui.controller.OpenDataSciTuiService", mock_service_cls),
            patch("opendatasci._tui.session.CLISessionInfo.from_path", return_value=fake_info),
            patch("opendatasci.tools.mcp.load_mcp_servers", return_value=[]),
            patch("pathlib.Path.resolve", return_value=Path("/fake/data.csv")),
        ):
            await ctrl.boot()

        # create_agent was invoked for the workspace, and the resulting agent +
        # its sandbox were handed to the service.
        mock_create_agent.assert_called_once()
        _, svc_kwargs = mock_service_cls.call_args
        assert svc_kwargs["agent"] is mock_agent
        assert svc_kwargs["sandbox"] is mock_sandbox
        assert ctrl._service is mock_service_instance


# ---------------------------------------------------------------------------
# CLIController — boot failure messages
# ---------------------------------------------------------------------------

_BOOT_PATCHES = (
    "opendatasci.tools.mcp.load_mcp_servers",
    "pathlib.Path.resolve",
    "pathlib.Path.is_dir",
)


def _make_boot_ctrl(mock_ui: MagicMock, workspace_path: str = "/fake/data.csv") -> CLIController:
    return CLIController(
        ui=mock_ui,
        workspace_path=workspace_path,
        datasci_config=OpenDataSciConfig(provider="anthropic", model="claude-sonnet-4-6"),
        session_id="testsid0",
    )


class TestBootFailures:
    async def test_file_not_found_shows_tailored_message(self, mock_ui: MagicMock) -> None:
        ctrl = _make_boot_ctrl(mock_ui)
        with (
            patch("pathlib.Path.is_file", return_value=False),
            patch("pathlib.Path.is_dir", return_value=False),
            patch("opendatasci._tui.controller.create_agent", side_effect=FileNotFoundError()),
            patch("opendatasci._tui.controller.OpenDataSciTuiService"),
            patch("opendatasci._tui.session.CLISessionInfo.from_path"),
            patch("opendatasci.tools.mcp.load_mcp_servers", return_value=[]),
            patch("pathlib.Path.resolve", return_value=Path("/fake/data.csv")),
            patch("opendatasci._tui.controller.CLIController._did_you_mean", return_value=""),
        ):
            await ctrl.boot()

        content = mock_ui.add_message.return_value.set_content.call_args[0][0]
        assert "File not found" in content
        assert "/fake/data.csv" in content
        assert "Check the path" in content

    async def test_file_not_found_includes_did_you_mean_when_close_match_exists(
        self, mock_ui: MagicMock, tmp_path: Path
    ) -> None:
        existing = tmp_path / "data.csv"
        existing.touch()
        typo = str(tmp_path / "daata.csv")
        ctrl = _make_boot_ctrl(mock_ui, workspace_path=typo)

        with (
            patch("pathlib.Path.is_file", return_value=False),
            patch("pathlib.Path.is_dir", return_value=False),
            patch("opendatasci._tui.controller.create_agent", side_effect=FileNotFoundError()),
            patch("opendatasci._tui.controller.OpenDataSciTuiService"),
            patch("opendatasci._tui.session.CLISessionInfo.from_path"),
            patch("opendatasci.tools.mcp.load_mcp_servers", return_value=[]),
            patch("pathlib.Path.resolve", return_value=Path(typo)),
        ):
            await ctrl.boot()

        content = mock_ui.add_message.return_value.set_content.call_args[0][0]
        assert "Did you mean" in content
        assert "data.csv" in content

    async def test_file_not_found_no_hint_when_no_close_match(
        self, mock_ui: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "completely_different.csv").touch()
        typo = str(tmp_path / "xyz.csv")
        ctrl = _make_boot_ctrl(mock_ui, workspace_path=typo)

        with (
            patch("pathlib.Path.is_file", return_value=False),
            patch("pathlib.Path.is_dir", return_value=False),
            patch("opendatasci._tui.controller.create_agent", side_effect=FileNotFoundError()),
            patch("opendatasci._tui.controller.OpenDataSciTuiService"),
            patch("opendatasci._tui.session.CLISessionInfo.from_path"),
            patch("opendatasci.tools.mcp.load_mcp_servers", return_value=[]),
            patch("pathlib.Path.resolve", return_value=Path(typo)),
        ):
            await ctrl.boot()

        content = mock_ui.add_message.return_value.set_content.call_args[0][0]
        assert "Did you mean" not in content

    async def test_permission_error_shows_tailored_message(self, mock_ui: MagicMock) -> None:
        ctrl = _make_boot_ctrl(mock_ui)
        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.is_dir", return_value=False),
            patch("opendatasci._tui.controller.create_agent", side_effect=PermissionError()),
            patch("opendatasci._tui.controller.OpenDataSciTuiService"),
            patch("opendatasci._tui.session.CLISessionInfo.from_path"),
            patch("opendatasci.tools.mcp.load_mcp_servers", return_value=[]),
            patch("pathlib.Path.resolve", return_value=Path("/fake/data.csv")),
        ):
            await ctrl.boot()

        content = mock_ui.add_message.return_value.set_content.call_args[0][0]
        assert "Permission denied" in content
        assert "/fake/data.csv" in content

    async def test_llm_provider_error_shows_api_key_guidance(self, mock_ui: MagicMock) -> None:
        
        ctrl = _make_boot_ctrl(mock_ui)
        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.is_dir", return_value=False),
            patch(
                "opendatasci._tui.controller.create_agent",
                side_effect=ValueError("bad key"),
            ),
            patch("opendatasci._tui.controller.OpenDataSciTuiService"),
            patch("opendatasci._tui.session.CLISessionInfo.from_path"),
            patch("opendatasci.tools.mcp.load_mcp_servers", return_value=[]),
            patch("pathlib.Path.resolve", return_value=Path("/fake/data.csv")),
        ):
            await ctrl.boot()

        content = mock_ui.add_message.return_value.set_content.call_args[0][0]
        assert "Provider error" in content
        assert "bad key" in content

    async def test_generic_exception_falls_through_to_default_message(
        self, mock_ui: MagicMock
    ) -> None:
        ctrl = _make_boot_ctrl(mock_ui)
        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.is_dir", return_value=False),
            patch(
                "opendatasci._tui.controller.create_agent",
                side_effect=RuntimeError("something unexpected"),
            ),
            patch("opendatasci._tui.controller.OpenDataSciTuiService"),
            patch("opendatasci._tui.session.CLISessionInfo.from_path"),
            patch("opendatasci.tools.mcp.load_mcp_servers", return_value=[]),
            patch("pathlib.Path.resolve", return_value=Path("/fake/data.csv")),
        ):
            await ctrl.boot()

        content = mock_ui.add_message.return_value.set_content.call_args[0][0]
        assert "Failed to load" in content


# ---------------------------------------------------------------------------
# _did_you_mean — sibling lookup edge cases
# ---------------------------------------------------------------------------


class TestDidYouMean:
    def test_close_sibling_produces_hint(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").write_text("")
        hint = CLIController._did_you_mean(str(tmp_path / "dato.csv"))
        assert "Did you mean" in hint
        assert "data.csv" in hint

    def test_no_close_match_returns_empty_string(self, tmp_path: Path) -> None:
        (tmp_path / "z.csv").write_text("")
        assert CLIController._did_you_mean(str(tmp_path / "completely_different.txt")) == ""

    def test_oserror_on_iterdir_returns_empty_string(self) -> None:
        """If the parent directory can't be listed (e.g. permission denied) the
        hint must degrade gracefully to an empty string — not propagate the error."""
        with patch("pathlib.Path.iterdir", side_effect=OSError("denied")):
            assert CLIController._did_you_mean("/something/missing.csv") == ""


# ---------------------------------------------------------------------------
# _describe_data — data summary lines
# ---------------------------------------------------------------------------


class TestDescribeData:
    def test_directory_pluralises_count(self) -> None:
        info = MagicMock(is_directory=True, workspace_count=3)
        assert CLIController._describe_data(info) == "3 files"

    def test_directory_singular_count(self) -> None:
        info = MagicMock(is_directory=True, workspace_count=1)
        assert CLIController._describe_data(info) == "1 file"

    def test_single_file_returns_empty(self) -> None:
        info = MagicMock(is_directory=False)
        assert CLIController._describe_data(info) == ""


# ---------------------------------------------------------------------------
# run_agent — unexpected event-type warning path
# ---------------------------------------------------------------------------


class TestRunAgentSkipsNonStreamEvent:
    async def test_warns_and_skips_non_stream_event(
        self,
        loaded_controller: CLIController,
        mock_service: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If astream() yields something that isn't a AgentStreamEvent (e.g. a stray
        dict from a misbehaving service), the controller must log a warning and
        continue — never crash, never dispatch."""
        import logging

        async def mixed_stream():
            yield {"this": "should-be-skipped"}  # not a AgentStreamEvent
            yield ResponseEvent(content="final")

        mock_service.astream = MagicMock(return_value=mixed_stream())
        with caplog.at_level(logging.WARNING, logger="opendatasci._tui.controller"):
            await loaded_controller.run_agent("q")
        assert any("unexpected type" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Slash-command dispatch — branches not exercised in TestOnSubmit
# ---------------------------------------------------------------------------


class TestSlashDispatch:
    async def test_slash_reset_invokes_reset(
        self, loaded_controller: CLIController, mock_service: MagicMock
    ) -> None:
        await loaded_controller.on_submit("/reset")
        mock_service.reset_session.assert_awaited_once()

    async def test_slash_compact_invokes_compact(
        self, loaded_controller: CLIController, mock_service: MagicMock
    ) -> None:
        await loaded_controller.on_submit("/compact")
        mock_service.compact_chat_history.assert_awaited()

    async def test_slash_ls_workspace_invokes_workspace_listing(
        self, loaded_controller: CLIController, mock_service: MagicMock
    ) -> None:
        await loaded_controller.on_submit("/ls-workspace")
        mock_service.get_workspace_files.assert_called()

    async def test_slash_models_renders_model_info(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await loaded_controller.on_submit("/models")
        rendered = [c.args[1] for c in mock_ui.add_message.call_args_list]
        assert any(
            "claude-sonnet" in str(text).lower() or "model" in str(text).lower()
            for text in rendered
        )

    async def test_slash_help_renders_help(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await loaded_controller.on_submit("/help")
        rendered = " ".join(str(c.args[1]) for c in mock_ui.add_message.call_args_list)
        assert "/help" in rendered or "command" in rendered.lower()

    async def test_slash_themes_lists_themes(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        await loaded_controller.on_submit("/themes")
        rendered = " ".join(str(c.args[1]) for c in mock_ui.add_message.call_args_list)
        assert "default" in rendered and "accessible" in rendered and "dracula" in rendered

    async def test_slash_vars_shows_deprecation_message(
        self, loaded_controller: CLIController, mock_ui: MagicMock
    ) -> None:
        """``/vars`` was removed; the controller now responds with a deprecation hint."""
        await loaded_controller.on_submit("/vars")
        last_message = mock_ui.add_message.call_args.args[1]
        assert "/vars" in last_message
        assert "removed" in last_message


# ---------------------------------------------------------------------------
# compact — error branch must stop the timer
# ---------------------------------------------------------------------------


class TestCompactStopsTimer:
    async def test_compact_failure_stops_turn_status_bar(
        self, loaded_controller: CLIController, mock_ui: MagicMock, mock_service: MagicMock
    ) -> None:
        """When compact_chat_history raises, the inline turn-status timer must be
        stopped so the UI does not leak a ticking counter on a finished turn."""
        timer = mock_ui.add_turn_status_bar.return_value
        mock_service.compact_chat_history = AsyncMock(side_effect=RuntimeError("boom"))
        await loaded_controller.compact()
        timer.stop.assert_called()
