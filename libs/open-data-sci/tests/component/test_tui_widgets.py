"""Component tests: real Textual widgets mounted in a headless app.

The unit suite covers the widgets' pure logic without an app context; these
tests complement it by driving the *mounted* behaviour through Textual's test
pilot: composition, timers, key bindings, message posting, and the ChatPane
factory methods — i.e. the widget layer as the controller/app actually use it.

The harness app mirrors ``OpenDataSciApp``'s CSS setup: it loads the real
``styles.tcss`` and exposes the active theme palette as ``$ods-*`` variables,
so widget DEFAULT_CSS that references those variables resolves exactly as in
production.
"""

import asyncio
import time
from pathlib import Path

import pytest
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import Input, Static
from textual.widgets import Markdown as TUIMarkdown

import opendatasci._tui as _tui_pkg
from opendatasci._tui import theme as _theme
from opendatasci._tui.widgets import (
    AppHeader,
    AttachmentBar,
    ChatPane,
    CommandApprovalPrompt,
    CompletionPopup,
    MessageBubble,
    MessagesContainer,
    PendingMessageBubble,
    PendingMessagePanel,
    SmartInput,
    ThinkingBlock,
    ToolCallBlock,
    TurnStatusBar,
    WorkspacePanel,
    _scroll_is_at_bottom,
)


def _plain(widget: Static) -> str:
    """Return the plain text of a Static's current renderable."""
    renderable = widget.render()
    if hasattr(renderable, "plain"):
        return renderable.plain
    return Text.from_markup(str(renderable)).plain


def _markdown_plain(bubble: "MessageBubble") -> str:
    """Return the concatenated plain text actually rendered by an agent bubble.

    Black-box equivalent of inspecting ``bubble._content`` (the accumulator):
    this reads what Textual's ``Markdown`` widget has actually mounted and
    rendered as child blocks, independent of whatever internal buffering
    mechanism produced it.
    """
    assert isinstance(bubble._inner, TUIMarkdown)
    return "\n".join(_plain(child) for child in bubble._inner.children if isinstance(child, Static))


class _Harness(App[None]):
    """Minimal app hosting the production ChatPane + AppHeader."""

    CSS_PATH = str(Path(_tui_pkg.__file__).parent / "styles.tcss")

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[bool] = []
        self.pasted: list[str] = []
        self.tab_completes: list[int] = []

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(
            {f"ods-{key.replace('_', '-')}": value for key, value in _theme.active.items()}
        )
        return variables

    def compose(self) -> ComposeResult:
        yield AppHeader(version="9.9.9", workspace="/data")
        yield ChatPane()

    def on_command_approval_prompt_decision(self, message: CommandApprovalPrompt.Decision) -> None:
        self.decisions.append(message.approved)

    def on_smart_input_pasted(self, message: SmartInput.Pasted) -> None:
        self.pasted.append(message._text)

    def on_smart_input_tab_complete(self, message: SmartInput.TabComplete) -> None:
        self.tab_completes.append(message._direction)


@pytest.fixture
async def harness():
    app = _Harness()
    async with app.run_test(size=(100, 40)) as pilot:
        yield app, pilot, app.query_one(ChatPane)


# ---------------------------------------------------------------------------
# AppHeader
# ---------------------------------------------------------------------------


class TestAppHeader:
    async def test_renders_version_and_workspace(self, harness) -> None:
        app, pilot, _ = harness
        info = _plain(app.query_one("#header-info", Static))
        assert "v9.9.9" in info
        assert "/data" in info

    async def test_does_not_render_a_model_line(self, harness) -> None:
        # Model info now lives behind /models, not the always-on header.
        app, pilot, _ = harness
        info = _plain(app.query_one("#header-info", Static))
        assert "Model" not in info

    async def test_set_file_count_and_workspace_rerender(self, harness) -> None:
        app, pilot, _ = harness
        header = app.query_one(AppHeader)
        header.set_file_count("3 files")
        header.set_workspace("sales-analysis")
        info = _plain(app.query_one("#header-info", Static))
        assert "(3 files)" in info
        assert "sales-analysis" in info

    async def test_clearing_workspace_name_removes_it(self, harness) -> None:
        app, pilot, _ = harness
        header = app.query_one(AppHeader)
        header.set_workspace("temp-name")
        header.set_workspace(None)
        assert "temp-name" not in _plain(app.query_one("#header-info", Static))


# ---------------------------------------------------------------------------
# MessageBubble via ChatPane.add_message
# ---------------------------------------------------------------------------


class TestMessageBubble:
    async def test_user_message_renders_markup_as_text(self, harness) -> None:
        app, pilot, pane = harness
        bubble = pane.add_message("user", "[bold]hello[/bold] world")
        await pilot.pause()
        assert isinstance(bubble._inner, Static)
        assert _plain(bubble._inner) == "hello world"
        assert bubble.has_class("user")

    async def test_question_message_with_invalid_markup_falls_back_to_plain(
        self, harness
    ) -> None:
        app, pilot, pane = harness
        bubble = pane.add_message("question", "closing tag [/] without open")
        await pilot.pause()
        assert _plain(bubble._inner) == "closing tag [/] without open"

    async def test_agent_message_constructed_with_content_renders(self, harness) -> None:
        # Reproduces the original TUI bug: content set at construction time,
        # then finish() called synchronously, before compose()/on_mount() has
        # had a chance to run (mounting is deferred). The bubble must still
        # end up fully rendered once it does mount.
        app, pilot, pane = harness
        bubble = pane.add_message("agent", "# Title\n\nBody text")
        await bubble.finish()
        await pilot.pause()
        await pilot.pause(0.1)
        assert isinstance(bubble._inner, TUIMarkdown)
        assert len(list(bubble._inner.children)) >= 2  # heading + paragraph blocks
        rendered = _markdown_plain(bubble)
        assert "Title" in rendered
        assert "Body text" in rendered

    async def test_agent_streaming_appends_render_final_text(self, harness) -> None:
        app, pilot, pane = harness
        bubble = pane.add_message("agent")
        await pilot.pause()
        await bubble.append("Hello ")
        await bubble.append("world")
        await pilot.pause(0.1)
        await bubble.finish()
        await pilot.pause(0.1)
        assert bubble._content == "Hello world"
        assert _markdown_plain(bubble) == "Hello world"

    async def test_rapid_fire_appends_all_land_in_final_render(self, harness) -> None:
        # Streams many small chunks faster than any coalescing window, to
        # guard against dropped or reordered fragments under backpressure.
        app, pilot, pane = harness
        bubble = pane.add_message("agent")
        await pilot.pause()
        chunks = [f"tok{i} " for i in range(60)]
        await asyncio.gather(*(bubble.append(chunk) for chunk in chunks))
        await bubble.finish()
        await pilot.pause(0.2)
        expected = "".join(chunks)
        assert bubble._content == expected
        assert _markdown_plain(bubble) == expected.strip()

    async def test_set_content_replaces_streamed_content(self, harness) -> None:
        app, pilot, pane = harness
        bubble = pane.add_message("agent")
        await pilot.pause()
        await bubble.append("partial")
        await bubble.set_content("final answer")
        await pilot.pause(0.1)
        assert bubble._content == "final answer"
        rendered = _markdown_plain(bubble)
        assert rendered == "final answer"
        assert "partial" not in rendered

    async def test_set_content_before_mount_renders_correctly(self, harness) -> None:
        # Same finish-before-mount race as above, but via the error path:
        # add_message("agent", "") then set_content() immediately, both
        # before the widget has been mounted.
        app, pilot, pane = harness
        bubble = pane.add_message("agent", "")
        await bubble.set_content("boom")
        await pilot.pause()
        await pilot.pause(0.1)
        assert bubble._content == "boom"
        assert _markdown_plain(bubble) == "boom"

    async def test_set_content_then_finish_with_no_intervening_yield_does_not_raise(
        self, harness
    ) -> None:
        # This is the exact sequence presenter.py runs on every error turn
        # (handle_exception -> set_content, then cleanup -> finish, back to
        # back with no yield in between). set_content() closes the stream
        # without reopening it, so finish() has nothing to stop here — but
        # this is still worth pinning down as a regression test since it's
        # the call pattern that originally surfaced the CancelledError bug
        # described on test_bootstrap_immediately_finished_does_not_raise.
        app, pilot, pane = harness
        bubble = pane.add_message("agent", "")
        await pilot.pause()
        await bubble.set_content("❌ boom")
        await bubble.finish()  # must not raise CancelledError
        await pilot.pause(0.1)
        assert bubble._content == "❌ boom"
        assert _markdown_plain(bubble) == "❌ boom"

    async def test_bootstrap_immediately_finished_does_not_raise(self, harness) -> None:
        # Regression: on_mount() opens a MarkdownStream (_bootstrap_stream)
        # regardless of whether there's any content yet. If finish() is
        # awaited before that background task has ever run a single step,
        # Textual's own MarkdownStream.stop() (task.cancel(); await task)
        # re-raises CancelledError instead of the task's internal
        # try/except absorbing it, unless _open_stream_locked's sleep(0)
        # lets the task start first.
        app, pilot, pane = harness
        bubble = pane.add_message("agent", "")
        await pilot.pause()  # let on_mount / _bootstrap_stream run
        await bubble.finish()  # must not raise CancelledError, even with nothing appended
        await pilot.pause(0.1)
        assert bubble._content == ""


# ---------------------------------------------------------------------------
# TurnStatusBar via ChatPane.add_turn_status_bar
# ---------------------------------------------------------------------------


class TestTurnStatusBar:
    async def test_running_label_and_stop_label(self, harness) -> None:
        app, pilot, pane = harness
        bar = pane.add_turn_status_bar()
        await pilot.pause()
        assert "Working for 0s" in _plain(bar)
        bar.stop()
        assert _plain(bar).startswith("Worked for")
        assert bar._interval is not None and bar._stopped is True

    async def test_context_suffix_without_cache_info(self, harness) -> None:
        app, pilot, pane = harness
        bar = pane.add_turn_status_bar()
        await pilot.pause()
        bar.update_context(3250, None)
        assert "Context: 3.2k tokens" in _plain(bar)
        assert "cached" not in _plain(bar)

    async def test_context_suffix_with_cache_percentage(self, harness) -> None:
        app, pilot, pane = harness
        bar = pane.add_turn_status_bar()
        await pilot.pause()
        bar.update_context(10_000, 5_000)
        assert "Context: 10.0k tokens (50.0% cached)" in _plain(bar)

    async def test_cache_percentage_is_clamped_below_100(self, harness) -> None:
        app, pilot, pane = harness
        bar = pane.add_turn_status_bar()
        await pilot.pause()
        bar.update_context(1_000, 2_000)  # provider quirk: cached > context
        assert "(99.9% cached)" in _plain(bar)

    async def test_minutes_formatting(self, harness) -> None:
        app, pilot, pane = harness
        bar = pane.add_turn_status_bar()
        await pilot.pause()
        bar._start = time.monotonic() - 61
        bar._tick()
        assert "Working for 1min 01s" in _plain(bar)
        bar._start = time.monotonic() - 120
        bar._tick()
        assert "Working for 2min" in _plain(bar)

    async def test_new_bar_replaces_previous_one(self, harness) -> None:
        app, pilot, pane = harness
        pane.add_turn_status_bar()
        await pilot.pause()
        second = pane.add_turn_status_bar()
        await pilot.pause()
        bars = list(app.query(TurnStatusBar))
        assert bars == [second]


# ---------------------------------------------------------------------------
# ToolCallBlock via ChatPane.add_ephemeral_block / add_worker_block
# ---------------------------------------------------------------------------


class TestToolCallBlock:
    async def test_running_then_done_lifecycle(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_ephemeral_block("Loading the data", "load_data", "Loading sales.csv")
        await pilot.pause()
        text = _plain(block)
        assert "Loading the data" in text
        assert "Loading sales.csv" in text
        assert block.is_running() is True

        block.set_done()
        assert block.is_running() is False
        assert block._spin_timer is None
        assert "Loading sales.csv" in _plain(block)

    async def test_error_state_shows_cross(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_ephemeral_block("", "run_code", "Running code")
        await pilot.pause()
        block.set_error()
        assert "✗ Running code" in _plain(block)
        assert block.is_running() is False

    async def test_summary_falls_back_to_label(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_ephemeral_block("", "tool_label", "")
        await pilot.pause()
        assert "tool_label" in _plain(block)

    async def test_communication_only_block_becomes_plain_text_when_done(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_ephemeral_block("Thinking about the schema", "", "")
        await pilot.pause()
        running_text = _plain(block)
        assert "Thinking about the schema" in running_text
        block.set_done()
        assert _plain(block) == "Thinking about the schema"

    async def test_upgrade_replaces_pending_label(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_ephemeral_block("", "Preparing…", "")
        await pilot.pause()
        block.upgrade("run_python", "Running analysis")
        assert "Running analysis" in _plain(block)
        assert "Preparing…" not in _plain(block)

    async def test_set_communication_updates_narration(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_ephemeral_block("old narration", "tool", "summary")
        await pilot.pause()
        block.set_communication("new narration")
        assert "new narration" in _plain(block)
        assert "old narration" not in _plain(block)

    async def test_markup_in_tool_text_is_escaped_not_interpreted(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_ephemeral_block("", "tool", "df.loc[0] selected")
        await pilot.pause()
        assert "df.loc[0] selected" in _plain(block)


class TestWorkerBlock:
    async def test_worker_rows_and_parallelizing_header(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_worker_block("Fanning out", ["Clean data", "Fit model"])
        await pilot.pause()
        text = _plain(block)
        assert "Fanning out" in text
        assert "Parallelizing" in text
        assert "Worker 1: Clean data" in text
        assert "Worker 2: Fit model" in text

    async def test_activity_shown_while_running_and_cleared_when_done(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_worker_block("", ["Clean data", "Fit model"])
        await pilot.pause()
        block.update_worker_activity(0, "run_python")
        assert "Worker 1: run_python" in _plain(block)

        block.mark_worker_done(0)
        text = _plain(block)
        assert "Worker 1: Clean data" in text  # activity replaced by summary again
        assert block.is_running() is True  # worker 2 still running

    async def test_all_workers_terminal_stops_spinner_and_marks_done(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_worker_block("", ["a", "b"])
        await pilot.pause()
        block.mark_worker_done(0)
        block.mark_worker_error(1)
        assert block.is_running() is False
        assert block._spin_timer is None
        text = _plain(block)
        assert "✗ Worker 2: b" in text

    async def test_out_of_range_worker_indices_are_ignored(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_worker_block("", ["a"])
        await pilot.pause()
        block.mark_worker_done(5)
        block.mark_worker_error(-1)
        block.update_worker_activity(5, "x")
        assert block._worker_statuses == ["running"]

    async def test_force_done_promotes_running_rows(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_worker_block("", ["a", "b"])
        await pilot.pause()
        block.mark_worker_done(0)
        block.set_done()  # e.g. turn ended while worker 2 still running
        text = _plain(block)
        assert "Worker 1: a" in text
        assert "Worker 2: b" in text
        assert "✗" not in text  # promoted to done, not error


# ---------------------------------------------------------------------------
# CommandApprovalPrompt via ChatPane.show_approval_prompt
# ---------------------------------------------------------------------------


class TestCommandApprovalPrompt:
    async def test_enter_confirms_yes_by_default(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_approval_prompt("Install seaborn", "")
        await pilot.pause()
        assert app.focused is app.query_one(CommandApprovalPrompt)
        await pilot.press("enter")
        assert app.decisions == [True]

    async def test_arrow_down_then_enter_selects_no(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_approval_prompt("Delete rows", "Data loss possible")
        await pilot.pause()
        assert "Data loss possible" in _plain(app.query_one("#approval-prompt-content", Static))
        await pilot.press("down", "enter")
        assert app.decisions == [False]

    async def test_escape_declines(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_approval_prompt("Run rm -rf tmp", "")
        await pilot.pause()
        await pilot.press("escape")
        assert app.decisions == [False]

    async def test_prompt_freezes_after_decision(self, harness) -> None:
        app, pilot, pane = harness
        prompt = pane.show_approval_prompt("Do the thing", "")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("down", "enter", "escape")
        assert app.decisions == [True]  # no further decisions posted
        content = _plain(app.query_one("#approval-prompt-content", Static))
        assert "✓ Yes" in content
        assert "Enter confirm" not in content  # hint removed once resolved
        assert prompt._resolved is True

    async def test_heads_up_omitted_when_empty(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_approval_prompt("Benign action", "")
        await pilot.pause()
        lines = _plain(app.query_one("#approval-prompt-content", Static)).splitlines()
        # Compact layout: header+description line, then straight to Yes/No —
        # no heads-up line, and no blank-line padding anywhere.
        assert lines[0] == "Approval required — Benign action"
        assert lines[1].lstrip("▸ ").rstrip() == "Yes"
        assert "" not in lines

    async def test_content_layout_signposts_description_and_heads_up(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_approval_prompt("Deletes temporary files", "Files are gone for good")
        await pilot.pause()
        text = _plain(app.query_one("#approval-prompt-content", Static))
        lines = text.splitlines()
        assert lines[0] == "Approval required — Deletes temporary files"
        assert lines[1] == "Files are gone for good"


# ---------------------------------------------------------------------------
# ThinkingBlock
# ---------------------------------------------------------------------------


class TestThinkingBlock:
    async def test_shows_thinking_and_animates(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_thinking_block()
        await pilot.pause()
        assert "Thinking" in _plain(block)
        first = _plain(block)
        block._tick()
        second = _plain(block)
        # The spinner glyph cycles each tick, but the label text stays "Thinking".
        assert "Thinking" in second
        assert first != second

    async def test_finish_stops_animation_and_shows_summary(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_thinking_block()
        await pilot.pause()
        block.finish("Thought for 3s")
        assert block._spin_timer is None
        assert _plain(block) == "Thought for 3s"

    async def test_dismiss_removes_from_dom(self, harness) -> None:
        app, pilot, pane = harness
        block = pane.add_thinking_block()
        await pilot.pause()
        block.dismiss()
        await pilot.pause()
        assert not list(app.query(ThinkingBlock))


# ---------------------------------------------------------------------------
# SmartInput: paste, tab-complete, history
# ---------------------------------------------------------------------------


class TestSmartInput:
    async def test_multiline_paste_posts_pasted_message(self, harness) -> None:
        app, pilot, pane = harness
        inp = app.query_one("#user-input", SmartInput)
        inp.focus()
        inp._on_paste(events.Paste("line one\nline two"))
        await pilot.pause()
        assert app.pasted == ["line one\nline two"]
        assert inp.value == ""  # multi-line paste never lands in the input

    async def test_single_line_paste_inserts_into_input(self, harness) -> None:
        app, pilot, pane = harness
        inp = app.query_one("#user-input", SmartInput)
        inp.focus()
        await pilot.pause()
        inp._on_paste(events.Paste("just one line"))
        await pilot.pause()
        assert app.pasted == []
        assert inp.value == "just one line"

    async def test_tab_posts_tab_complete_instead_of_focus_change(self, harness) -> None:
        app, pilot, pane = harness
        app.query_one("#user-input", SmartInput).focus()
        await pilot.pause()
        await pilot.press("tab")
        assert app.tab_completes == [1]

    async def test_history_navigation_round_trip(self, harness) -> None:
        app, pilot, pane = harness
        inp = app.query_one("#user-input", SmartInput)
        inp.push_history("first")
        inp.push_history("second")
        inp.value = "draft"

        assert inp.navigate_history(-1) is True
        assert inp.value == "second"
        assert inp.navigate_history(-1) is True
        assert inp.value == "first"
        assert inp.navigate_history(-1) is False  # already at oldest
        assert inp.navigate_history(1) is True
        assert inp.value == "second"
        assert inp.navigate_history(1) is True
        assert inp.value == "draft"  # back to the saved draft

    async def test_consecutive_duplicate_history_entries_collapse(self, harness) -> None:
        app, pilot, pane = harness
        inp = app.query_one("#user-input", SmartInput)
        inp.push_history("same")
        inp.push_history("same")
        assert inp._input_history._history == ["same"]

    async def test_down_without_navigation_is_a_noop(self, harness) -> None:
        app, pilot, pane = harness
        inp = app.query_one("#user-input", SmartInput)
        inp.push_history("entry")
        inp.value = "typing"
        assert inp.navigate_history(1) is False
        assert inp.value == "typing"


# ---------------------------------------------------------------------------
# CompletionPopup / AttachmentBar
# ---------------------------------------------------------------------------


class TestCompletionPopup:
    async def test_show_matches_highlights_selection_and_escapes_markup(self, harness) -> None:
        app, pilot, pane = harness
        popup = app.query_one(CompletionPopup)
        popup.show_matches(["data[1].csv", "other.csv"], selected=0)
        assert popup.has_class("active")
        text = _plain(popup)
        assert "▸ data[1].csv" in text
        assert "other.csv" in text

    async def test_hide_clears_content_and_class(self, harness) -> None:
        app, pilot, pane = harness
        popup = app.query_one(CompletionPopup)
        popup.show_matches(["a.csv"], selected=0)
        popup.hide()
        assert not popup.has_class("active")
        assert _plain(popup) == ""


class TestAttachmentBar:
    async def test_show_pill_and_hide(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_attachment("Pasted text (4 lines)")
        bar = app.query_one(AttachmentBar)
        assert bar.has_class("active")
        text = _plain(bar)
        assert "Pasted text (4 lines)" in text
        assert "Esc to discard" in text

        pane.hide_attachment()
        assert not bar.has_class("active")
        assert _plain(bar) == ""


# ---------------------------------------------------------------------------
# WorkspacePanel via ChatPane.show_workspace_panel
# ---------------------------------------------------------------------------


class TestWorkspacePanel:
    async def test_show_files_focuses_panel_and_lists_files(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_workspace_panel(["a.csv", "b.csv", "c.csv"])
        await pilot.pause()
        panel = app.query_one(WorkspacePanel)
        assert panel.has_class("active")
        assert app.focused is panel
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ a.csv" in content
        assert "3 files" in content

    async def test_empty_file_list_shows_placeholder(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_workspace_panel([])
        await pilot.pause()
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "No files in active workspace." in content

    async def test_arrow_navigation_moves_selection(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_workspace_panel(["a.csv", "b.csv", "c.csv"])
        await pilot.pause()
        await pilot.press("down", "down")
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ c.csv" in content
        await pilot.press("up")
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ b.csv" in content

    async def test_scrolling_window_and_range_indicator(self, harness) -> None:
        app, pilot, pane = harness
        files = [f"file{i:02d}.csv" for i in range(30)]
        pane.show_workspace_panel(files)
        await pilot.pause()
        await pilot.press("end")
        panel = app.query_one(WorkspacePanel)
        assert panel._selected == 29
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ file29.csv" in content
        assert "19–30 of 30" in content

        await pilot.press("home")
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ file00.csv" in content
        assert "1–12 of 30" in content

    async def test_page_down_and_page_up_move_selection_and_window(self, harness) -> None:
        app, pilot, pane = harness
        files = [f"file{i:02d}.csv" for i in range(30)]
        pane.show_workspace_panel(files)
        await pilot.pause()
        panel = app.query_one(WorkspacePanel)

        await pilot.press("pagedown")
        assert panel._selected == 12
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ file12.csv" in content

        await pilot.press("pageup")
        assert panel._selected == 0
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ file00.csv" in content

    async def test_paging_within_a_single_page_rerenders_highlight(self, harness) -> None:
        # Fewer files than one page: PgDn/PgUp move the selection without
        # scrolling, and the highlighted row must still follow.
        app, pilot, pane = harness
        pane.show_workspace_panel(["a.csv", "b.csv", "c.csv"])
        await pilot.pause()
        panel = app.query_one(WorkspacePanel)

        await pilot.press("pagedown")
        assert panel._selected == 2
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ c.csv" in content

        await pilot.press("pageup")
        assert panel._selected == 0
        content = _plain(app.query_one("#workspace-panel-content", Static))
        assert "▸ a.csv" in content

    async def test_escape_closes_panel_and_refocuses_input(self, harness) -> None:
        app, pilot, pane = harness
        pane.show_workspace_panel(["a.csv"])
        await pilot.pause()
        await pilot.press("escape")
        panel = app.query_one(WorkspacePanel)
        assert not panel.has_class("active")
        assert panel._files == []
        assert app.focused is app.query_one("#user-input", Input)


# ---------------------------------------------------------------------------
# PendingMessagePanel / PendingMessageBubble
# ---------------------------------------------------------------------------


class TestPendingMessages:
    async def test_newest_pending_message_is_mounted_first(self, harness) -> None:
        app, pilot, pane = harness
        pane.add_pending_message("first queued")
        await pilot.pause()
        pane.add_pending_message("second queued")
        await pilot.pause()
        panel = app.query_one(PendingMessagePanel)
        bubbles = list(panel.query(PendingMessageBubble))
        assert _plain(bubbles[0]).endswith("second queued")
        assert _plain(bubbles[1]).endswith("first queued")

    async def test_pending_bubble_shows_queued_marker(self, harness) -> None:
        app, pilot, pane = harness
        bubble = pane.add_pending_message("run the model")
        await pilot.pause()
        assert "Queued" in _plain(bubble)


# ---------------------------------------------------------------------------
# MessagesContainer auto-scroll anchor + ChatPane housekeeping
# ---------------------------------------------------------------------------


class TestMessagesContainerAnchor:
    async def test_new_content_keeps_view_pinned_to_bottom(self, harness) -> None:
        app, pilot, pane = harness
        for i in range(40):
            await pane.add_message("user", f"message {i}").finish()
        await pilot.pause(0.3)  # let the anchor tick re-pin after layout
        container = app.query_one("#messages", MessagesContainer)
        assert container.max_scroll_y > 0
        assert _scroll_is_at_bottom(container)

    async def test_scrolling_up_releases_anchor(self, harness) -> None:
        app, pilot, pane = harness
        for i in range(40):
            await pane.add_message("user", f"message {i}").finish()
        await pilot.pause(0.3)
        container = app.query_one("#messages", MessagesContainer)
        container.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert container.scroll_offset.y == 0
        # New content must not drag the user back down once they've scrolled up.
        await pane.add_message("user", "late arrival").finish()
        await pilot.pause(0.3)
        assert container.scroll_offset.y == 0

    async def test_scrolling_back_to_bottom_rearms_anchor(self, harness) -> None:
        app, pilot, pane = harness
        for i in range(40):
            await pane.add_message("user", f"message {i}").finish()
        await pilot.pause(0.3)
        container = app.query_one("#messages", MessagesContainer)
        container.scroll_to(y=0, animate=False)
        await pilot.pause()
        container.scroll_end(animate=False)
        await pilot.pause()
        # New content must resume following the bottom once re-armed.
        await pane.add_message("user", "late arrival").finish()
        await pilot.pause(0.3)
        assert _scroll_is_at_bottom(container)


class TestChatPaneHousekeeping:
    async def test_add_divider_mounts_rule(self, harness) -> None:
        app, pilot, pane = harness
        pane.add_divider()
        await pilot.pause()
        assert len(list(app.query(".msg-divider"))) == 1

    async def test_clear_messages_empties_container(self, harness) -> None:
        app, pilot, pane = harness
        pane.add_message("user", "hello")
        pane.add_divider()
        pane.add_thinking_block()
        await pilot.pause()
        pane.clear_messages()
        await pilot.pause()
        container = app.query_one("#messages", MessagesContainer)
        assert len(list(container.children)) == 0
