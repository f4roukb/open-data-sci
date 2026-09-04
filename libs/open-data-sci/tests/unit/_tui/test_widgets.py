"""Unit tests for opendatasci._tui.chat.widgets — pure logic only (no Textual app context)."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from rich.text import Text
from textual import events as textual_events
from textual.widgets import Markdown as TUIMarkdown
from textual.widgets import Static

from opendatasci._tui.chat.models import SPINNER, SPINNER_INTERVAL
from opendatasci._tui.chat.widgets import (
    AppHeader,
    AttachmentBar,
    ChatPane,
    CommandApprovalPrompt,
    CommandHighlighter,
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
    _InputHistory,
    _scroll_is_at_bottom,
)

# ---------------------------------------------------------------------------
# Spinner constants
# ---------------------------------------------------------------------------


class TestSpinnerConstants:
    def test_spinner_has_multiple_frames(self) -> None:
        assert len(SPINNER) >= 4

    def test_spinner_interval_is_slower_than_old_80ms(self) -> None:
        assert SPINNER_INTERVAL > 0.08

    def test_spinner_interval_is_reasonable(self) -> None:
        assert 0.08 < SPINNER_INTERVAL <= 0.25

    def test_spinner_frames_are_unique(self) -> None:
        assert len(set(SPINNER)) == len(SPINNER)


# ---------------------------------------------------------------------------
# AppHeader — version string
# ---------------------------------------------------------------------------


def _make_header(version: str = "0.1.0") -> AppHeader:
    """Instantiate AppHeader bypassing Textual Widget.__init__."""
    header = AppHeader.__new__(AppHeader)
    header._version = version
    header._workspace = "/tmp/data"
    header._workspace_name = None
    header._file_count = ""
    header._model_info = ""
    header._background_tasks = ""
    return header


def _render_info_plain(header: AppHeader) -> str:
    """Call _render_info() and return the full plain text passed to Static.update()."""
    captured: list = []
    mock_static = MagicMock()
    mock_static.update.side_effect = captured.append
    header.query_one = MagicMock(return_value=mock_static)
    header._render_info()
    assert captured, "_render_info() never called Static.update()"
    return captured[-1].plain


class TestAppHeaderVersionString:
    def test_version_string_shown(self) -> None:
        text = _render_info_plain(_make_header(version="1.2.3"))
        assert "v1.2.3" in text

    def test_no_model_line_rendered_when_unset(self) -> None:
        text = _render_info_plain(_make_header())
        assert "Model" not in text

    def test_model_line_shown_once_set(self) -> None:
        header = _make_header()
        header._model_info = "Anthropic  claude-sonnet-5"
        text = _render_info_plain(header)
        assert "Model" in text
        assert "Anthropic  claude-sonnet-5" in text


# ---------------------------------------------------------------------------
# CommandHighlighter
# ---------------------------------------------------------------------------


class TestCommandHighlighter:
    def _highlight(self, text: str) -> Text:
        t = Text(text)
        CommandHighlighter().highlight(t)
        return t

    def test_known_command_has_bold_style(self) -> None:
        t = self._highlight("/clear")
        spans = [(s.start, s.end) for s in t._spans]
        assert (0, 6) in spans

    def test_valid_prefix_also_styled(self) -> None:
        t = self._highlight("/cl")
        spans = [(s.start, s.end) for s in t._spans]
        assert (0, 3) in spans

    def test_non_slash_text_no_spans(self) -> None:
        t = self._highlight("hello")
        assert t._spans == []

    def test_unrelated_slash_word_no_spans(self) -> None:
        t = self._highlight("/zzz_not_a_command")
        assert t._spans == []


# ---------------------------------------------------------------------------
# TurnStatusBar._fmt
# ---------------------------------------------------------------------------


class TestTurnStatusBarFmt:
    """Test time formatting without instantiating the full Textual widget."""

    @pytest.fixture
    def fmt(self) -> "function":  # type: ignore[type-arg]
        bar = TurnStatusBar.__new__(TurnStatusBar)
        return bar._fmt

    def test_sub_minute_shows_seconds(self, fmt) -> None:
        assert fmt(30) == "30s"

    def test_zero_seconds(self, fmt) -> None:
        assert fmt(0) == "0s"

    def test_exact_minute_no_seconds(self, fmt) -> None:
        assert fmt(60) == "1min"

    def test_over_minute_shows_minutes_and_seconds(self, fmt) -> None:
        assert fmt(90) == "1min 30s"

    def test_two_minutes(self, fmt) -> None:
        assert fmt(120) == "2min"


# ---------------------------------------------------------------------------
# TurnStatusBar._fmt_tokens
# ---------------------------------------------------------------------------


class TestTurnStatusBarFmtTokens:
    """Test token-count formatting without instantiating the full Textual widget."""

    def _fmt(self, n: int) -> str:
        return TurnStatusBar._fmt_tokens(n)

    def test_exact_thousands(self) -> None:
        assert self._fmt(3000) == "3.0k"

    def test_truncates_not_rounds(self) -> None:
        # 3250 / 1000 = 3.25 → should truncate to 3.2, not round to 3.3
        assert self._fmt(3250) == "3.2k"

    def test_truncates_fractional(self) -> None:
        assert self._fmt(3999) == "3.9k"

    def test_small_value(self) -> None:
        assert self._fmt(100) == "0.1k"

    def test_zero(self) -> None:
        assert self._fmt(0) == "0.0k"

    def test_large_value(self) -> None:
        assert self._fmt(12500) == "12.5k"


# ---------------------------------------------------------------------------
# TurnStatusBar._context_suffix
# ---------------------------------------------------------------------------


class TestTurnStatusBarContextSuffix:
    """Test context suffix formatting without a Textual event loop."""

    def _bar(
        self, context_tokens: int | None = None, cached_tokens: int | None = None
    ) -> TurnStatusBar:
        t = TurnStatusBar.__new__(TurnStatusBar)
        t._context_tokens = context_tokens
        t._cached_tokens = cached_tokens
        return t

    def test_no_context_returns_empty(self) -> None:
        assert self._bar()._context_suffix() == ""

    def test_context_with_no_cache_support(self) -> None:
        # cached_tokens=None means the API doesn't report cache metrics
        suffix = self._bar(6500, None)._context_suffix()
        assert suffix == " | Context: 6.5k tokens"

    def test_context_with_zero_cached(self) -> None:
        # cached_tokens=0 is valid: API supports it, nothing was cached
        suffix = self._bar(6500, 0)._context_suffix()
        assert suffix == " | Context: 6.5k tokens (0.0% cached)"

    def test_context_with_cache_percentage(self) -> None:
        # 2925 / 6500 = 45.0% exactly
        suffix = self._bar(6500, 2925)._context_suffix()
        assert suffix == " | Context: 6.5k tokens (45.0% cached)"

    def test_cache_percentage_rounds_up(self) -> None:
        # 2601 / 6500 = 40.015...% → ceil to one decimal → 40.1%
        suffix = self._bar(6500, 2601)._context_suffix()
        assert "(40.1% cached)" in suffix

    def test_full_cache_shows_capped_percent(self) -> None:
        # Cache-read is always a subset of input tokens, so a "full" cache hit
        # is capped just under 100% rather than displayed as a bare 100.0%.
        suffix = self._bar(5000, 5000)._context_suffix()
        assert "(99.9% cached)" in suffix


# ---------------------------------------------------------------------------
# TurnStatusBar.update_context
# ---------------------------------------------------------------------------


class TestTurnStatusBarUpdateContext:
    """Test update_context without a Textual event loop."""

    def _bar(self) -> TurnStatusBar:
        t = TurnStatusBar.__new__(TurnStatusBar)
        t._stopped = False
        t._mounted = True
        t._start = 0.0
        t._interval = None
        t._context_tokens = None
        t._cached_tokens = None
        return t

    def test_update_context_sets_values(self) -> None:
        t = self._bar()
        t.update = MagicMock()
        t.update_context(6500, 2925)
        assert t._context_tokens == 6500
        assert t._cached_tokens == 2925

    def test_update_context_triggers_redraw(self) -> None:
        t = self._bar()
        t.update = MagicMock()
        t.update_context(6500, 2925)
        t.update.assert_called_once()
        rendered = t.update.call_args[0][0]
        assert "Context:" in rendered
        assert "(45.0% cached)" in rendered

    def test_update_context_clamps_cached_over_context(self) -> None:
        # Reproduces the Anthropic streaming case where langchain_anthropic
        # sums usage across message_start/message_delta chunks, so cache_read
        # can come back larger than input+output tokens for the same call.
        t = self._bar()
        t.update = MagicMock()
        t.update_context(70, 9000)
        rendered = t.update.call_args[0][0]
        assert "(99.9% cached)" in rendered

    def test_update_context_none_cached_renders_without_parens(self) -> None:
        # None means the API doesn't provide cache metrics
        t = self._bar()
        t.update = MagicMock()
        t.update_context(4000, None)
        rendered = t.update.call_args[0][0]
        assert "Context: 4.0k" in rendered
        assert "cached" not in rendered

    def test_update_context_none_tokens_hides_context_size(self) -> None:
        t = self._bar()
        t.update = MagicMock()
        t.update_context(None, None)
        rendered = t.update.call_args[0][0]
        assert "Context size" not in rendered

    def test_update_context_noop_when_stopped(self) -> None:
        t = self._bar()
        t._stopped = True
        t.update = MagicMock()
        t.update_context(6500, None)
        t.update.assert_not_called()

    def test_update_context_noop_when_not_mounted(self) -> None:
        t = self._bar()
        t._mounted = False
        t.update = MagicMock()
        t.update_context(6500, None)
        t.update.assert_not_called()


# ---------------------------------------------------------------------------
# ToolCallBlock — state machine (no DOM)
# ---------------------------------------------------------------------------


def _make_block(
    communication: str = "doing something",
    label: str = "MyTool",
    summary: str = "ran code",
    task_summaries: list[str] | None = None,
) -> ToolCallBlock:
    """Instantiate ToolCallBlock bypassing Textual Widget.__init__."""
    block = ToolCallBlock.__new__(ToolCallBlock)
    block._communication = communication
    block._label = label
    block._summary = summary
    block._task_summaries = task_summaries or []
    block._task_statuses = ["running"] * len(block._task_summaries)
    block._task_activities = [""] * len(block._task_summaries)
    block._done = False
    block._error = False
    block._spin_idx = 0
    block._spin_timer = None
    return block


class TestToolCallBlockState:
    def test_is_running_true_initially(self) -> None:
        block = _make_block()
        assert block.is_running() is True

    def test_set_done_marks_not_running(self) -> None:
        block = _make_block()
        with patch.object(block, "_refresh"):
            block.set_done()
        assert block._done is True
        assert block.is_running() is False

    def test_upgrade_updates_label_and_summary(self) -> None:
        block = _make_block(label="old", summary="")
        with patch.object(block, "_refresh"):
            block.upgrade("NewLabel", "new summary")
        assert block._label == "NewLabel"
        assert block._summary == "new summary"

    def test_set_communication_updates_text(self) -> None:
        block = _make_block(communication="old comm")
        with patch.object(block, "_refresh"):
            block.set_communication("updated comm")
        assert block._communication == "updated comm"

    def test_set_communication_none_removes_it(self) -> None:
        block = _make_block(communication="old comm")
        with patch.object(block, "_refresh"):
            block.set_communication(None)
        assert block._communication is None

    def test_mark_task_done_sets_status(self) -> None:
        block = _make_block(task_summaries=["w1", "w2"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_task_done(0)
        assert block._task_statuses[0] == "done"
        assert block._task_statuses[1] == "running"

    def test_mark_task_error_sets_status(self) -> None:
        block = _make_block(task_summaries=["w1"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_task_error(0)
        assert block._task_statuses[0] == "error"

    def test_mark_task_done_all_terminal_stops_spinner(self) -> None:
        block = _make_block(task_summaries=["w1"])
        stop_called = []
        with (
            patch.object(block, "_refresh"),
            patch.object(block, "_stop_spinner", side_effect=lambda: stop_called.append(True)),
        ):
            block.mark_task_done(0)
        assert stop_called

    def test_mark_task_done_all_terminal_sets_done(self) -> None:
        block = _make_block(task_summaries=["w1", "w2"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_task_done(0)
        assert block._done is False  # only one of two workers done
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_task_done(1)
        assert block._done is True  # all workers done → block is definitively closed

    def test_mark_task_error_all_terminal_sets_done(self) -> None:
        block = _make_block(task_summaries=["w1"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_task_error(0)
        assert block._done is True

    def test_mark_task_done_partial_does_not_set_done(self) -> None:
        block = _make_block(task_summaries=["w1", "w2", "w3"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_task_done(0)
            block.mark_task_done(1)
        assert block._done is False  # third worker still running

    def test_update_task_activity_updates_running_worker(self) -> None:
        block = _make_block(task_summaries=["w1"])
        with patch.object(block, "_refresh"):
            block.update_task_activity(0, "running tool X")
        assert block._task_activities[0] == "running tool X"

    def test_update_task_activity_ignored_for_done_worker(self) -> None:
        block = _make_block(task_summaries=["w1"])
        block._task_statuses[0] = "done"
        with patch.object(block, "_refresh"):
            block.update_task_activity(0, "should be ignored")
        assert block._task_activities[0] == ""

    def test_update_task_activity_skips_refresh_when_unchanged(self) -> None:
        block = _make_block(task_summaries=["w1"])
        block._task_activities[0] = "tool X"
        with patch.object(block, "_refresh") as refresh:
            block.update_task_activity(0, "tool X")
        refresh.assert_not_called()

    def test_update_task_activity_calls_refresh_when_changed(self) -> None:
        block = _make_block(task_summaries=["w1"])
        with patch.object(block, "_refresh") as refresh:
            block.update_task_activity(0, "tool X")
        refresh.assert_called_once()

    def test_mark_task_done_clears_activity(self) -> None:
        block = _make_block(task_summaries=["w1"])
        block._task_activities[0] = "some activity"
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_task_done(0)
        assert block._task_activities[0] == ""


class TestToolCallBlockMarkup:
    """Test markup generation methods (pure string logic)."""

    def test_status_markup_running_contains_spinner(self) -> None:
        block = _make_block()
        markup = block._status_markup("some text")
        assert "some text" in markup
        assert "tool_running" in markup or "#" in markup

    def test_status_markup_done_no_checkmark(self) -> None:
        block = _make_block()
        block._done = True
        markup = block._status_markup("done text")
        assert "✓" not in markup
        assert "done text" in markup
        assert "[bold" in markup  # still styled

    def test_task_status_markup_error_contains_x(self) -> None:
        block = _make_block()
        markup = block._task_status_markup("worker", "error")
        assert "✗" in markup

    def test_task_status_markup_done_no_checkmark(self) -> None:
        block = _make_block()
        markup = block._task_status_markup("worker", "done")
        assert "✓" not in markup
        assert "worker" in markup
        assert "[bold" in markup  # still styled

    def test_task_status_markup_running_contains_spinner(self) -> None:
        block = _make_block()
        markup = block._task_status_markup("worker", "running")
        assert "worker" in markup


class TestToolCallBlockRefreshForcedTerminalState:
    """When the block is force-closed via set_done/set_error while individual
    worker rows are still 'running', _refresh must promote those rows to a
    state consistent with how the block itself closed: 'done' for set_done,
    'error' for set_error. Promoting an unfinished row to 'done' under
    set_error would hide a failure behind a green row."""

    def test_set_error_promotes_running_rows_to_error(self) -> None:
        block = _make_block(task_summaries=["w1", "w2"])
        block.update = MagicMock()  # type: ignore[assignment]
        block.set_error()
        rendered = str(block.update.call_args.args[0])
        # Both rows should reflect the error state — '✗' is the row-level error glyph.
        assert rendered.count("✗") >= 2

    def test_set_done_keeps_running_rows_as_done(self) -> None:
        block = _make_block(task_summaries=["w1"])
        block.update = MagicMock()  # type: ignore[assignment]
        block.set_done()
        rendered = str(block.update.call_args.args[0])
        # set_done should not flip running rows into the error glyph.
        assert "✗" not in rendered

    def test_set_error_keeps_already_done_rows_done(self) -> None:
        block = _make_block(task_summaries=["w1", "w2"])
        block._task_statuses[0] = "done"
        block.update = MagicMock()  # type: ignore[assignment]
        block.set_error()
        rendered = str(block.update.call_args.args[0])
        # Only the row that was still running at set_error time should become error;
        # the already-done row stays done — exactly one '✗' from the worker rows
        # plus one from the header.
        assert rendered.count("✗") == 2


# ---------------------------------------------------------------------------
# ChatPane.add_turn_status_bar — scroll behaviour
# ---------------------------------------------------------------------------


def _make_chat_pane() -> ChatPane:
    """Instantiate ChatPane bypassing Textual Widget.__init__."""
    return ChatPane.__new__(ChatPane)


class TestChatPaneAddTurnStatusBar:
    """add_turn_status_bar mounts the new bar into #status-bar, alongside TipsBar."""

    def _setup_pane(self, existing_bars: list | None = None) -> tuple[ChatPane, MagicMock]:
        pane = _make_chat_pane()
        status_bar = MagicMock()
        pane.query = MagicMock(return_value=existing_bars or [])
        pane.query_one = MagicMock(return_value=status_bar)
        return pane, status_bar

    def test_bar_is_mounted_into_status_bar(self) -> None:
        pane, status_bar = self._setup_pane()
        bar = MagicMock()

        with patch("opendatasci._tui.chat.widgets.TurnStatusBar", return_value=bar):
            result = pane.add_turn_status_bar()

        status_bar.mount.assert_called_once_with(bar)
        assert result is bar

    def test_existing_bars_are_removed_before_mounting(self) -> None:
        stale_a, stale_b = MagicMock(), MagicMock()
        pane, _ = self._setup_pane(existing_bars=[stale_a, stale_b])

        with patch("opendatasci._tui.chat.widgets.TurnStatusBar", return_value=MagicMock()):
            pane.add_turn_status_bar()

        stale_a.remove.assert_called_once()
        stale_b.remove.assert_called_once()

    def test_new_bar_is_mounted_into_status_bar(self) -> None:
        pane, status_bar = self._setup_pane()
        bar = MagicMock()

        with patch("opendatasci._tui.chat.widgets.TurnStatusBar", return_value=bar):
            pane.add_turn_status_bar()

        status_bar.mount.assert_called_once_with(bar)

    def test_returns_the_newly_created_bar(self) -> None:
        pane, _ = self._setup_pane()
        bar = MagicMock()

        with patch("opendatasci._tui.chat.widgets.TurnStatusBar", return_value=bar):
            result = pane.add_turn_status_bar()

        assert result is bar


class TestToolCallBlockRefreshSpacing:
    """_refresh must produce a blank line between the communication text and the
    tool status line so the visual gap matches the inter-block margin-bottom."""

    def _rendered_lines(self, block: ToolCallBlock) -> list[str]:
        captured: list[Text] = []
        with patch.object(block, "update", side_effect=captured.append):
            block._refresh()
        assert captured, "update() was never called"
        return captured[-1].plain.splitlines()

    def test_communication_separated_by_blank_line_from_status(self) -> None:
        block = _make_block(communication="Let me check this.", label="MyTool", summary="ran")
        lines = self._rendered_lines(block)
        # Expected layout: communication, blank line, status line
        assert lines[0] == "Let me check this."
        assert lines[1] == ""
        assert lines[2] != ""  # status line present

    def test_no_communication_no_blank_line_inserted(self) -> None:
        block = _make_block(communication="", label="MyTool", summary="ran")
        lines = self._rendered_lines(block)
        # No communication → just the status line, no leading blank
        assert lines[0] != ""
        assert len(lines) == 1

    def test_communication_shown_with_blank_separator_for_task_blocks(self) -> None:
        block = _make_block(task_summaries=["worker-1", "worker-2"])
        block._communication = "Running checks in parallel."
        lines = self._rendered_lines(block)
        # Worker path renders communication → blank separator → header → worker rows
        assert lines[0] == "Running checks in parallel."
        assert lines[1] == ""
        assert lines[2] != ""  # header line
        assert lines[3] != ""  # worker 1
        assert lines[4] != ""  # worker 2

    def test_no_communication_no_blank_line_for_task_blocks(self) -> None:
        block = _make_block(task_summaries=["worker-1", "worker-2"])
        block._communication = ""
        lines = self._rendered_lines(block)
        # No communication → header is first line, no leading blank
        assert lines[0] != ""
        assert "" not in lines

    def test_status_line_shows_summary_only_not_label(self) -> None:
        block = _make_block(communication="", label="MyTool", summary="ran")
        lines = self._rendered_lines(block)
        # Spinner is SPINNER[0] = "⣾"; status line must be exactly "⣾ ran".
        assert lines[0] == "⣾ ran"

    def test_status_line_with_communication_shows_summary_only(self) -> None:
        block = _make_block(communication="Let me check this.", label="MyTool", summary="ran")
        lines = self._rendered_lines(block)
        # lines[2] is the status line; it must be exactly "⣾ ran", not "⣾ MyTool — ran".
        assert lines[2] == "⣾ ran"

    def test_status_line_done_shows_summary_only(self) -> None:
        block = _make_block(communication="", label="MyTool", summary="ran")
        block._done = True
        lines = self._rendered_lines(block)
        # Done state shows just the summary in green — no tick prefix.
        assert lines[0] == "ran"

    def test_status_line_falls_back_to_label_when_no_summary(self) -> None:
        block = _make_block(communication="", label="…", summary="")
        lines = self._rendered_lines(block)
        # Placeholder phase (no summary yet) falls back to the label token.
        assert lines[0] == "⣾ …"


class TestToolCallBlockCommunicationOnly:
    """Communication-only mode (label == summary == ""): used for hidden
    (display_status=False) tools whose narration must stay visible while the tool
    identity does not.  The narration is the whole block — spinner-prefixed
    while running, plain text once finished, error glyph on failure."""

    def _rendered_lines(self, block: ToolCallBlock) -> list[str]:
        captured: list[Text] = []
        with patch.object(block, "update", side_effect=captured.append):
            block._refresh()
        assert captured, "update() was never called"
        return captured[-1].plain.splitlines()

    def test_running_shows_spinner_prefixed_narration_only(self) -> None:
        block = _make_block(communication="Checking dataset notes.", label="", summary="")
        lines = self._rendered_lines(block)
        # Single line: no separate status line, no blank separator.
        assert lines == ["⣾ Checking dataset notes."]

    def test_done_shows_plain_narration_without_status_styling(self) -> None:
        block = _make_block(communication="Checking dataset notes.", label="", summary="")
        block._done = True
        lines = self._rendered_lines(block)
        assert lines == ["Checking dataset notes."]

    def test_error_shows_x_glyph_on_narration(self) -> None:
        block = _make_block(communication="Checking dataset notes.", label="", summary="")
        block._done = True
        block._error = True
        lines = self._rendered_lines(block)
        assert lines == ["✗ Checking dataset notes."]

    def test_upgrade_to_real_label_restores_two_part_layout(self) -> None:
        block = _make_block(communication="Checking dataset notes.", label="", summary="")
        with patch.object(block, "_refresh"):
            block.upgrade("MyTool", "ran")
        lines = self._rendered_lines(block)
        assert lines == ["Checking dataset notes.", "", "⣾ ran"]


class TestToolCallBlockWorkerRowRendering:
    """Worker block renders a subtree:

        Parallelizing
          └─ ⣾ Worker 1 — …
          └─ ⣾ Worker 2 — …

    Each worker row is indented two spaces and prefixed with the L-shaped
    box-drawing character so the layout reads as a subtree under the header.
    The header itself has no trailing punctuation."""

    WORKER_UPDATE_BRANCH = "  └─ "

    def _rendered_lines(self, block: ToolCallBlock) -> list[str]:
        captured: list[Text] = []
        with patch.object(block, "update", side_effect=captured.append):
            block._refresh()
        assert captured, "update() was never called"
        return captured[-1].plain.splitlines()

    def test_header_has_no_trailing_colon(self) -> None:
        block = _make_block(communication="", task_summaries=["w1"])
        lines = self._rendered_lines(block)
        # While workers run, the spinner is prepended to the header by
        # _status_markup; expected layout is "{spin} Parallelizing".
        assert "Parallelizing" in lines[0]
        assert not lines[0].rstrip().endswith(":")

    def test_header_done_state_has_no_trailing_colon(self) -> None:
        block = _make_block(communication="", task_summaries=["w1"])
        block._task_statuses[0] = "done"
        lines = self._rendered_lines(block)
        # All workers terminal → header rendered in the done style (no spinner).
        assert lines[0] == "Parallelizing"

    def test_running_worker_row_has_indented_tree_prefix_before_spinner(self) -> None:
        block = _make_block(communication="", task_summaries=["w1"])
        lines = self._rendered_lines(block)
        # Spinner is SPINNER[0] = "⣾"; tree prefix sits BEFORE the spinner.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}⣾ Worker 1 — w1"

    def test_done_worker_row_preserves_tree_prefix(self) -> None:
        block = _make_block(communication="", task_summaries=["w1"])
        block._task_statuses[0] = "done"
        lines = self._rendered_lines(block)
        # No spinner in done state — prefix still leads, then the row text.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}Worker 1 — w1"

    def test_error_worker_row_preserves_tree_prefix_before_x_glyph(self) -> None:
        block = _make_block(communication="", task_summaries=["w1"])
        block._task_statuses[0] = "error"
        lines = self._rendered_lines(block)
        # Error glyph "✗" sits inside the status markup; prefix is still external.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}✗ Worker 1 — w1"

    def test_running_worker_row_with_activity_keeps_tree_prefix(self) -> None:
        block = _make_block(communication="", task_summaries=["w1"])
        block._task_activities[0] = "🐍 running pandas"
        lines = self._rendered_lines(block)
        # Activity replaces the subtask summary while running; prefix unchanged.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}⣾ Worker 1 — 🐍 running pandas"


def _make_bubble(role: str, content: str = "") -> MessageBubble:
    """Instantiate a MessageBubble bypassing Textual Widget.__init__."""
    bubble = MessageBubble.__new__(MessageBubble)
    bubble._role = role
    bubble._content = content
    bubble._inner = None
    bubble._stream = None
    bubble._written_len = 0
    bubble._write_lock = asyncio.Lock()
    bubble._ready = asyncio.Event()
    return bubble


# ---------------------------------------------------------------------------
# CommandApprovalPrompt — pure rendering logic (no DOM)
# ---------------------------------------------------------------------------


def _make_prompt(description: str = "Run script", heads_up: str = "") -> CommandApprovalPrompt:
    """Instantiate CommandApprovalPrompt bypassing Textual Widget.__init__."""
    prompt = CommandApprovalPrompt.__new__(CommandApprovalPrompt)
    prompt._description = description
    prompt._heads_up = heads_up
    prompt._selected = 0
    prompt._resolved = False
    return prompt


def _prompt_rendered_lines(prompt: CommandApprovalPrompt) -> list[str]:
    """Call _refresh_content() and return the plain lines passed to Static.update()."""
    captured: list[Text] = []
    mock_static = MagicMock()
    mock_static.update.side_effect = captured.append
    prompt.query_one = MagicMock(return_value=mock_static)
    prompt._refresh_content()
    assert captured, "_refresh_content() never called Static.update()"
    return captured[-1].plain.splitlines()


class TestCommandApprovalPromptRefreshContent:
    """Compact layout: header+description share a line, no blank-line padding.

    Unresolved:              Resolved:
        Approval required — <description>      Approval required — <description>
        <heads_up>                              <heads_up>
        ▸ Yes                                   ✓ Yes
          No
        ↑↓ select  Enter confirm  Esc decline
    """

    def test_header_and_description_share_first_line(self) -> None:
        lines = _prompt_rendered_lines(_make_prompt(description="Deletes temporary files"))
        assert lines[0] == "Approval required — Deletes temporary files"

    def test_no_blank_lines_in_unresolved_layout(self) -> None:
        lines = _prompt_rendered_lines(_make_prompt(heads_up="Files are gone for good"))
        assert "" not in lines

    def test_heads_up_shown_without_emoji_when_present(self) -> None:
        lines = _prompt_rendered_lines(_make_prompt(heads_up="Files are gone for good"))
        assert "Files are gone for good" in lines
        assert not any("⚠" in line for line in lines)

    def test_heads_up_omitted_when_empty(self) -> None:
        lines = _prompt_rendered_lines(_make_prompt(heads_up=""))
        assert not any("⚠" in line for line in lines)
        # Header line, then straight to Yes/No — no empty heads-up line either.
        assert lines[1].strip().lstrip("▸").strip() in ("Yes", "No")

    def test_selected_option_marked_with_arrow(self) -> None:
        prompt = _make_prompt()
        prompt._selected = 0
        lines = _prompt_rendered_lines(prompt)
        assert any(line.startswith("▸ Yes") for line in lines)

    def test_moving_selection_marks_no(self) -> None:
        prompt = _make_prompt()
        prompt._selected = 1
        lines = _prompt_rendered_lines(prompt)
        assert any(line.startswith("▸ No") for line in lines)

    def test_hint_line_present_while_unresolved(self) -> None:
        lines = _prompt_rendered_lines(_make_prompt())
        assert any("Enter confirm" in line for line in lines)

    def test_resolved_yes_shows_only_check_marked_choice(self) -> None:
        prompt = _make_prompt()
        prompt._selected = 0
        prompt._resolved = True
        lines = _prompt_rendered_lines(prompt)
        assert any(line.startswith("✓ Yes") for line in lines)
        assert not any("No" in line for line in lines)
        assert not any("Enter confirm" in line for line in lines)

    def test_resolved_layout_has_exactly_two_lines_without_heads_up(self) -> None:
        prompt = _make_prompt(description="Do the thing")
        prompt._selected = 0
        prompt._resolved = True
        lines = _prompt_rendered_lines(prompt)
        assert lines == ["Approval required — Do the thing", "✓ Yes"]

    def test_no_emoji_anywhere_in_output(self) -> None:
        lines = _prompt_rendered_lines(
            _make_prompt(description="Do the thing", heads_up="Heads up text")
        )
        text = "\n".join(lines)
        assert "🛡" not in text
        assert "⚠️" not in text


class TestMessageBubbleCompose:
    """compose() always produces an empty Markdown widget; content arrives via _flush_agent."""

    def _composed_inner(self, bubble: MessageBubble):
        widgets = list(bubble.compose())
        assert len(widgets) == 1, "compose() must yield exactly one inner widget"
        return widgets[0]

    def test_agent_role_yields_markdown_widget(self) -> None:
        bubble = _make_bubble("agent", "")
        inner = self._composed_inner(bubble)
        assert isinstance(inner, TUIMarkdown)
        assert bubble._inner is inner

    def test_agent_role_always_starts_empty(self) -> None:
        # compose() always seeds TUIMarkdown with "" so that _on_mount's
        # implicit update() is a no-op.  All rendering goes through _flush_agent
        # to avoid a race where two concurrent update() tasks both mount content
        # and produce duplicate text (the root cause of the /clear doubling bug).
        bubble = _make_bubble("agent", "Final answer with **markdown**")
        inner = self._composed_inner(bubble)
        assert isinstance(inner, TUIMarkdown)
        assert inner._markdown == ""

    def test_agent_role_starts_empty_regardless_of_pre_compose_appends(self) -> None:
        # Tokens may stream in before the bubble finishes mounting, but compose()
        # still produces an empty Markdown widget; the stream (created in
        # on_mount) is what flushes whatever has accumulated in _content.
        bubble = _make_bubble("agent", "")
        bubble._content = "tok1tok2"
        inner = self._composed_inner(bubble)
        assert isinstance(inner, TUIMarkdown)
        assert inner._markdown == ""

    def test_user_role_yields_static_widget(self) -> None:
        bubble = _make_bubble("user", "Hello")
        inner = self._composed_inner(bubble)
        assert isinstance(inner, Static)
        assert not isinstance(inner, TUIMarkdown)

    def test_question_role_yields_static_widget(self) -> None:
        bubble = _make_bubble("question", "Choose")
        inner = self._composed_inner(bubble)
        assert isinstance(inner, Static)


class TestMessageBubbleOnMount:
    """on_mount() must open the streaming gate for agent bubbles via a
    background worker (which creates the MarkdownStream); other roles open
    the gate immediately since they never touch a stream."""

    def test_agent_role_schedules_stream_bootstrap(self) -> None:
        bubble = _make_bubble("agent", "")
        with (
            patch.object(bubble, "run_worker") as run_worker,
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        run_worker.assert_called_once()
        run_worker.call_args[0][0].close()  # avoid an unawaited-coroutine ResourceWarning
        assert bubble._ready.is_set() is False  # gate stays closed until the worker runs

    def test_non_agent_role_opens_the_gate_immediately(self) -> None:
        bubble = _make_bubble("user", "hi")
        with patch.object(bubble, "_refresh_content"):
            bubble.on_mount()
        assert bubble._ready.is_set() is True


class TestMessageBubbleBootstrapStream:
    """_bootstrap_stream creates the stream, flushes pre-mount content, then
    opens the gate — the core regression coverage for the original
    "agent message never appears" bug: content set at construction (or via
    append()/finish() called before Textual has mounted the widget) must
    still reach the stream once it exists."""

    @pytest.mark.asyncio
    async def test_flushes_content_set_before_mount(self) -> None:
        bubble = _make_bubble("agent", "buffered before mount")
        bubble._inner = MagicMock(spec=TUIMarkdown)
        stream = MagicMock()
        stream.write = AsyncMock()
        with patch.object(TUIMarkdown, "get_stream", return_value=stream):
            await bubble._bootstrap_stream()
        stream.write.assert_awaited_once_with("buffered before mount")
        assert bubble._written_len == len("buffered before mount")
        assert bubble._ready.is_set() is True

    @pytest.mark.asyncio
    async def test_no_write_when_no_pre_mount_content(self) -> None:
        bubble = _make_bubble("agent", "")
        bubble._inner = MagicMock(spec=TUIMarkdown)
        stream = MagicMock()
        stream.write = AsyncMock()
        with patch.object(TUIMarkdown, "get_stream", return_value=stream):
            await bubble._bootstrap_stream()
        stream.write.assert_not_called()
        assert bubble._ready.is_set() is True

    @pytest.mark.asyncio
    async def test_opens_the_gate_even_if_get_stream_raises(self) -> None:
        # If the stream can't be created, callers must not deadlock forever
        # waiting on _ready — they'll just find _stream is None and skip.
        bubble = _make_bubble("agent", "x")
        bubble._inner = MagicMock(spec=TUIMarkdown)
        with patch.object(TUIMarkdown, "get_stream", side_effect=RuntimeError("boom")):
            await bubble._bootstrap_stream()  # must not raise
        assert bubble._ready.is_set() is True
        assert bubble._stream is None


class TestMessageBubbleWritePendingLocked:
    """The write cursor (_written_len) only ever sends the unwritten tail of
    _content — the guard that prevents double-writing content set both at
    construction and via a concurrent append()/finish()."""

    @pytest.mark.asyncio
    async def test_only_sends_unwritten_tail(self) -> None:
        bubble = _make_bubble("agent", "abcdef")
        bubble._written_len = 3
        stream = MagicMock()
        stream.write = AsyncMock()
        bubble._stream = stream
        await bubble._write_pending_locked()
        stream.write.assert_awaited_once_with("def")
        assert bubble._written_len == 6

    @pytest.mark.asyncio
    async def test_noop_when_fully_written(self) -> None:
        bubble = _make_bubble("agent", "abc")
        bubble._written_len = 3
        stream = MagicMock()
        stream.write = AsyncMock()
        bubble._stream = stream
        await bubble._write_pending_locked()
        stream.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_stream_is_none(self) -> None:
        bubble = _make_bubble("agent", "abc")
        bubble._stream = None
        await bubble._write_pending_locked()  # must not raise
        assert bubble._written_len == 0


class TestMessageBubbleAppend:
    @pytest.mark.asyncio
    async def test_agent_append_writes_the_new_chunk(self) -> None:
        bubble = _make_bubble("agent", "")
        stream = MagicMock()
        stream.write = AsyncMock()
        bubble._stream = stream
        bubble._ready.set()
        await bubble.append("hello")
        assert bubble._content == "hello"
        stream.write.assert_awaited_once_with("hello")
        assert bubble._written_len == len("hello")

    @pytest.mark.asyncio
    async def test_agent_append_blocks_until_ready(self) -> None:
        bubble = _make_bubble("agent", "")
        stream = MagicMock()
        stream.write = AsyncMock()
        bubble._stream = stream
        task = asyncio.ensure_future(bubble.append("late"))
        await asyncio.sleep(0)
        assert not task.done()  # still waiting on _ready
        bubble._ready.set()
        await task
        stream.write.assert_awaited_once_with("late")

    @pytest.mark.asyncio
    async def test_non_agent_append_calls_refresh_content_without_touching_ready(self) -> None:
        bubble = _make_bubble("user", "Hello")
        with patch.object(bubble, "_refresh_content") as refresh:
            await bubble.append(" world")
        refresh.assert_called_once()
        assert bubble._content == "Hello world"
        assert bubble._ready.is_set() is False  # non-agent roles never touch the gate

    @pytest.mark.asyncio
    async def test_agent_append_reopens_a_stream_closed_by_set_content(self) -> None:
        # set_content() closes the stream without reopening it (see
        # TestMessageBubbleSetContent); a later append() must lazily reopen
        # one rather than silently dropping the chunk.
        bubble = _make_bubble("agent", "")
        bubble._inner = MagicMock(spec=TUIMarkdown)
        bubble._stream = None
        bubble._ready.set()
        new_stream = MagicMock()
        new_stream.write = AsyncMock()
        with patch.object(TUIMarkdown, "get_stream", return_value=new_stream) as get_stream:
            await bubble.append("more")
        get_stream.assert_called_once_with(bubble._inner)
        new_stream.write.assert_awaited_once_with("more")
        assert bubble._stream is new_stream


class TestMessageBubbleSetContent:
    @pytest.mark.asyncio
    async def test_agent_set_content_stops_stream_and_replaces_without_reopening(self) -> None:
        # No fresh stream is opened here: every current caller follows
        # set_content() with finish() and nothing else, so eagerly reopening
        # would just create a stream that's cancelled unused (append()
        # reopens lazily via _ensure_stream_locked() if ever called after).
        bubble = _make_bubble("agent", "partial")
        bubble._written_len = 7
        old_stream = MagicMock()
        old_stream.stop = AsyncMock()
        bubble._stream = old_stream
        bubble._inner = MagicMock(spec=TUIMarkdown)
        bubble._inner.update = AsyncMock()
        bubble._ready.set()
        with patch.object(TUIMarkdown, "get_stream") as get_stream:
            await bubble.set_content("final")
        old_stream.stop.assert_awaited_once()
        bubble._inner.update.assert_awaited_once_with("final")
        assert bubble._content == "final"
        assert bubble._written_len == len("final")
        get_stream.assert_not_called()
        assert bubble._stream is None

    @pytest.mark.asyncio
    async def test_agent_set_content_noop_stop_when_no_stream_open(self) -> None:
        bubble = _make_bubble("agent", "")
        bubble._inner = MagicMock(spec=TUIMarkdown)
        bubble._inner.update = AsyncMock()
        bubble._ready.set()
        await bubble.set_content("final")  # must not raise despite _stream being None
        assert bubble._content == "final"

    @pytest.mark.asyncio
    async def test_non_agent_set_content_calls_refresh_content(self) -> None:
        bubble = _make_bubble("user", "old")
        with patch.object(bubble, "_refresh_content") as refresh:
            await bubble.set_content("new")
        assert bubble._content == "new"
        refresh.assert_called_once()


class TestMessageBubbleFinish:
    @pytest.mark.asyncio
    async def test_agent_finish_flushes_pending_then_stops_stream(self) -> None:
        bubble = _make_bubble("agent", "unflushed")
        stream = MagicMock()
        stream.write = AsyncMock()
        stream.stop = AsyncMock()
        bubble._stream = stream
        bubble._ready.set()
        await bubble.finish()
        stream.write.assert_awaited_once_with("unflushed")
        stream.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_agent_finish_calls_refresh_content(self) -> None:
        bubble = _make_bubble("user", "hi")
        with patch.object(bubble, "_refresh_content") as refresh:
            await bubble.finish()
        refresh.assert_called_once()


# ---------------------------------------------------------------------------
# ThinkingBlock
# ---------------------------------------------------------------------------


class TestThinkingBlock:
    def _make_block(self) -> ThinkingBlock:
        block = ThinkingBlock()
        block.update = MagicMock()  # type: ignore[assignment]
        return block

    def test_finish_stops_spin_timer(self) -> None:
        block = self._make_block()
        timer = MagicMock()
        block._spin_timer = timer
        block.finish("Thought for 3s")
        timer.stop.assert_called_once()
        assert block._spin_timer is None

    def test_finish_noop_when_timer_already_none(self) -> None:
        block = self._make_block()
        block._spin_timer = None
        block.finish("Thought for 0s")  # must not raise

    def test_finish_updates_display_with_summary(self) -> None:
        block = self._make_block()
        block._spin_timer = None
        block.finish("Thought for 5s")
        rendered: Text = block.update.call_args[0][0]
        assert "Thought for 5s" in rendered.plain

    def test_finish_uses_text_muted_color(self) -> None:
        from opendatasci._tui.style import theme as _theme

        block = self._make_block()
        block._spin_timer = None
        block.finish("Thought for 2s")
        rendered: Text = block.update.call_args[0][0]
        muted_color = _theme.active["text_muted"]
        assert any(muted_color in str(span.style) for span in rendered._spans)

    def test_finish_does_not_remove_widget(self) -> None:
        block = self._make_block()
        block.remove = MagicMock()  # type: ignore[assignment]
        block._spin_timer = None
        block.finish("Thought for 1s")
        block.remove.assert_not_called()

    def test_on_mount_uses_shared_spinner_interval(self) -> None:
        # Must use the same SPINNER_INTERVAL as ToolCallBlock, not a private
        # literal, so the two "in progress" indicators never drift apart again.
        block = self._make_block()
        block.set_interval = MagicMock(return_value=MagicMock())  # type: ignore[assignment]
        block.on_mount()
        block.set_interval.assert_called_once_with(SPINNER_INTERVAL, block._tick)

    def test_tick_cycles_through_shared_spinner_frames(self) -> None:
        block = self._make_block()
        assert block._spin_idx == 0
        block._tick()
        assert block._spin_idx == 1
        rendered: Text = block.update.call_args[0][0]
        assert rendered.plain.startswith(f"{SPINNER[1]} Thinking")

    def test_tick_wraps_around_after_last_frame(self) -> None:
        block = self._make_block()
        block._spin_idx = len(SPINNER) - 1
        block._tick()
        assert block._spin_idx == 0

    def test_display_has_no_emoji(self) -> None:
        block = self._make_block()
        block._update_display()
        rendered: Text = block.update.call_args[0][0]
        assert "💭" not in rendered.plain
        assert rendered.plain == f"{SPINNER[0]} Thinking"


# ---------------------------------------------------------------------------
# _InputHistory — navigation logic
# ---------------------------------------------------------------------------


class TestInputHistoryPush:
    def test_push_adds_entry(self) -> None:
        h = _InputHistory()
        h.push("hello")
        assert h._history == ["hello"]

    def test_push_empty_string_ignored(self) -> None:
        h = _InputHistory()
        h.push("")
        assert h._history == []

    def test_push_consecutive_duplicate_ignored(self) -> None:
        h = _InputHistory()
        h.push("hello")
        h.push("hello")
        assert h._history == ["hello"]

    def test_push_non_consecutive_duplicate_stored(self) -> None:
        h = _InputHistory()
        h.push("a")
        h.push("b")
        h.push("a")
        assert h._history == ["a", "b", "a"]

    def test_push_resets_navigation_state(self) -> None:
        h = _InputHistory()
        h.push("first")
        h._index = 0
        h._draft = "partial"
        h.push("second")
        assert h._index == -1
        assert h._draft == ""


class TestInputHistoryNavigate:
    def test_up_with_no_history_returns_none(self) -> None:
        h = _InputHistory()
        assert h.navigate(-1, "current") is None

    def test_down_with_no_history_returns_none(self) -> None:
        h = _InputHistory()
        assert h.navigate(1, "current") is None

    def test_down_when_not_navigating_returns_none(self) -> None:
        h = _InputHistory()
        h.push("a")
        assert h.navigate(1, "current") is None

    def test_first_up_returns_most_recent_entry(self) -> None:
        h = _InputHistory()
        h.push("first")
        h.push("second")
        assert h.navigate(-1, "") == "second"

    def test_first_up_saves_draft(self) -> None:
        h = _InputHistory()
        h.push("entry")
        h.navigate(-1, "my draft")
        assert h._draft == "my draft"

    def test_second_up_returns_older_entry(self) -> None:
        h = _InputHistory()
        h.push("first")
        h.push("second")
        h.navigate(-1, "")
        assert h.navigate(-1, "second") == "first"

    def test_up_at_oldest_returns_none(self) -> None:
        h = _InputHistory()
        h.push("only")
        h.navigate(-1, "")
        assert h.navigate(-1, "only") is None

    def test_down_after_up_returns_newer_entry(self) -> None:
        h = _InputHistory()
        h.push("a")
        h.push("b")
        h.push("c")
        h.navigate(-1, "")  # → c
        h.navigate(-1, "c")  # → b
        assert h.navigate(1, "b") == "c"

    def test_down_from_most_recent_returns_draft(self) -> None:
        h = _InputHistory()
        h.push("entry")
        h.navigate(-1, "my draft")  # → entry, draft saved
        result = h.navigate(1, "entry")  # → back to draft
        assert result == "my draft"

    def test_down_from_most_recent_resets_index(self) -> None:
        h = _InputHistory()
        h.push("entry")
        h.navigate(-1, "")
        h.navigate(1, "entry")
        assert h._index == -1

    def test_single_entry_up_then_down_restores_draft(self) -> None:
        h = _InputHistory()
        h.push("query")
        h.navigate(-1, "draft text")
        result = h.navigate(1, "query")
        assert result == "draft text"

    def test_multiple_entries_full_round_trip(self) -> None:
        h = _InputHistory()
        for entry in ["alpha", "beta", "gamma"]:
            h.push(entry)
        assert h.navigate(-1, "draft") == "gamma"
        assert h.navigate(-1, "gamma") == "beta"
        assert h.navigate(-1, "beta") == "alpha"
        assert h.navigate(-1, "alpha") is None  # already at oldest
        assert h.navigate(1, "alpha") == "beta"
        assert h.navigate(1, "beta") == "gamma"
        assert h.navigate(1, "gamma") == "draft"  # back to saved draft


# ---------------------------------------------------------------------------
# SmartInput — history navigation wrappers
# ---------------------------------------------------------------------------


class TestSmartInputHistory:
    """Tests for SmartInput.push_history / navigate_history — widget-level wrappers."""

    def _make_inp(self) -> SmartInput:
        inp = SmartInput.__new__(SmartInput)
        inp._input_history = _InputHistory()
        return inp

    def test_push_history_stores_entry_in_input_history(self) -> None:
        inp = self._make_inp()
        inp.push_history("my query")
        assert inp._input_history._history == ["my query"]

    def test_navigate_history_passes_direction_and_current_value_to_inner(self) -> None:
        inp = self._make_inp()
        mock_history = MagicMock(spec=_InputHistory)
        mock_history.navigate.return_value = None
        inp._input_history = mock_history
        # create=True is required because Input.value is a Textual reactive whose
        # __get__ raises AttributeError on class access, so patch.object must
        # create the shadow attribute on SmartInput rather than looking it up.
        with patch.object(
            type(inp), "value", new_callable=PropertyMock, return_value="draft", create=True
        ):
            inp.navigate_history(-1)
        mock_history.navigate.assert_called_once_with(-1, "draft")

    def test_navigate_history_returns_false_when_inner_returns_none(self) -> None:
        inp = self._make_inp()
        mock_history = MagicMock(spec=_InputHistory)
        mock_history.navigate.return_value = None
        inp._input_history = mock_history
        with patch.object(
            type(inp), "value", new_callable=PropertyMock, return_value="", create=True
        ):
            result = inp.navigate_history(-1)
        assert result is False

    def test_navigate_history_returns_true_and_updates_value(self) -> None:
        inp = self._make_inp()
        mock_history = MagicMock(spec=_InputHistory)
        mock_history.navigate.return_value = "previous query"
        inp._input_history = mock_history
        with (
            patch.object(type(inp), "value", new_callable=PropertyMock, create=True) as mock_value,
            patch.object(type(inp), "cursor_position", new_callable=PropertyMock, create=True),
        ):
            mock_value.return_value = ""
            result = inp.navigate_history(-1)
        assert result is True
        mock_value.assert_any_call("previous query")


# ---------------------------------------------------------------------------
# SmartInput — paste interception
# ---------------------------------------------------------------------------


def _make_smart_input() -> SmartInput:
    """Instantiate SmartInput bypassing Textual Widget.__init__."""
    return SmartInput.__new__(SmartInput)


class TestSmartInputPaste:
    def test_multiline_paste_posts_pasted_message(self) -> None:
        inp = _make_smart_input()
        posted: list = []
        inp.post_message = MagicMock(side_effect=posted.append)

        mock_event = MagicMock(spec=textual_events.Paste)
        mock_event.text = "line1\nline2\nline3"

        with patch.object(SmartInput.__bases__[0], "_on_paste") as super_paste:
            inp._on_paste(mock_event)

        super_paste.assert_not_called()
        assert len(posted) == 1
        assert isinstance(posted[0], SmartInput.Pasted)
        assert posted[0]._text == "line1\nline2\nline3"

    def test_single_line_paste_falls_through_to_super(self) -> None:
        inp = _make_smart_input()

        mock_event = MagicMock(spec=textual_events.Paste)
        mock_event.text = "single line paste"

        with patch.object(SmartInput.__bases__[0], "_on_paste") as super_paste:
            inp._on_paste(mock_event)

        super_paste.assert_called_once_with(mock_event)

    def test_empty_paste_falls_through_to_super(self) -> None:
        inp = _make_smart_input()

        mock_event = MagicMock(spec=textual_events.Paste)
        mock_event.text = ""

        with patch.object(SmartInput.__bases__[0], "_on_paste") as super_paste:
            inp._on_paste(mock_event)

        super_paste.assert_called_once_with(mock_event)

    def test_pasted_message_carries_full_text(self) -> None:
        inp = _make_smart_input()
        posted: list = []
        inp.post_message = MagicMock(side_effect=posted.append)

        code = "def foo():\n    return 42\n"
        mock_event = MagicMock(spec=textual_events.Paste)
        mock_event.text = code

        with patch.object(SmartInput.__bases__[0], "_on_paste"):
            inp._on_paste(mock_event)

        assert posted[0]._text == code


# ---------------------------------------------------------------------------
# AttachmentBar — pill display and hide
# ---------------------------------------------------------------------------


def _make_attachment_bar() -> AttachmentBar:
    """Instantiate AttachmentBar bypassing Textual Widget.__init__."""
    return AttachmentBar.__new__(AttachmentBar)


class TestAttachmentBar:
    def test_show_pill_updates_content(self) -> None:
        bar = _make_attachment_bar()
        captured: list = []
        bar.update = MagicMock(side_effect=captured.append)
        bar.add_class = MagicMock()

        bar.show_pill("Text: 5 lines")

        assert captured, "update() was never called"
        assert "Text: 5 lines" in captured[-1].plain

    def test_show_pill_adds_active_class(self) -> None:
        bar = _make_attachment_bar()
        bar.update = MagicMock()
        bar.add_class = MagicMock()

        bar.show_pill("Text: 3 lines")

        bar.add_class.assert_called_once_with("active")

    def test_hide_removes_active_class(self) -> None:
        bar = _make_attachment_bar()
        bar.remove_class = MagicMock()
        bar.update = MagicMock()

        bar.hide()

        bar.remove_class.assert_called_once_with("active")

    def test_hide_clears_content(self) -> None:
        bar = _make_attachment_bar()
        bar.remove_class = MagicMock()
        bar.update = MagicMock()

        bar.hide()

        bar.update.assert_called_once_with("")

    def test_show_pill_contains_discard_hint(self) -> None:
        bar = _make_attachment_bar()
        captured: list = []
        bar.update = MagicMock(side_effect=captured.append)
        bar.add_class = MagicMock()

        bar.show_pill("Text: 2 lines")

        assert "Esc" in captured[-1].plain


# ---------------------------------------------------------------------------
# CompletionPopup — show_matches and hide
# ---------------------------------------------------------------------------


def _make_completion_popup() -> CompletionPopup:
    return CompletionPopup.__new__(CompletionPopup)


class TestCompletionPopup:
    def test_show_matches_calls_update(self) -> None:
        popup = _make_completion_popup()
        popup.update = MagicMock()
        popup.add_class = MagicMock()
        popup.show_matches(["/clear", "/compact"], selected=0)
        popup.update.assert_called_once()

    def test_show_matches_includes_all_items_in_output(self) -> None:
        popup = _make_completion_popup()
        captured: list[Text] = []
        popup.update = MagicMock(side_effect=captured.append)
        popup.add_class = MagicMock()
        popup.show_matches(["/clear", "/compact"], selected=0)
        plain = captured[-1].plain
        assert "/clear" in plain
        assert "/compact" in plain

    def test_show_matches_selected_item_marked_with_arrow(self) -> None:
        popup = _make_completion_popup()
        captured: list[Text] = []
        popup.update = MagicMock(side_effect=captured.append)
        popup.add_class = MagicMock()
        popup.show_matches(["/clear", "/compact"], selected=0)
        assert "▸" in captured[-1].plain

    def test_show_matches_only_one_arrow_for_one_selection(self) -> None:
        popup = _make_completion_popup()
        captured: list[Text] = []
        popup.update = MagicMock(side_effect=captured.append)
        popup.add_class = MagicMock()
        popup.show_matches(["/clear", "/compact", "/exit"], selected=1)
        assert captured[-1].plain.count("▸") == 1

    def test_show_matches_second_item_selected_has_arrow_before_it(self) -> None:
        popup = _make_completion_popup()
        captured: list[Text] = []
        popup.update = MagicMock(side_effect=captured.append)
        popup.add_class = MagicMock()
        popup.show_matches(["/clear", "/compact"], selected=1)
        lines = captured[-1].plain.splitlines()
        # Second line is the selected item — it must contain ▸
        assert "▸" in lines[1]
        assert "▸" not in lines[0]

    def test_show_matches_adds_active_class(self) -> None:
        popup = _make_completion_popup()
        popup.update = MagicMock()
        popup.add_class = MagicMock()
        popup.show_matches(["/clear"], selected=0)
        popup.add_class.assert_called_once_with("active")

    def test_hide_removes_active_class(self) -> None:
        popup = _make_completion_popup()
        popup.remove_class = MagicMock()
        popup.update = MagicMock()
        popup.hide()
        popup.remove_class.assert_called_once_with("active")

    def test_hide_clears_content_to_empty_string(self) -> None:
        popup = _make_completion_popup()
        popup.remove_class = MagicMock()
        popup.update = MagicMock()
        popup.hide()
        popup.update.assert_called_once_with("")


# ---------------------------------------------------------------------------
# WorkspacePanel — navigation state machine (pure logic, no DOM)
# ---------------------------------------------------------------------------


def _make_panel(
    files: list[str],
    selected: int = 0,
    offset: int = 0,
) -> WorkspacePanel:
    panel = WorkspacePanel.__new__(WorkspacePanel)
    panel._files = list(files)
    panel._selected = selected
    panel._offset = offset
    return panel


class TestWorkspacePanelNavigation:
    """Verify that action_move_* methods update _selected and _offset correctly.

    All tests patch _update_content to keep tests pure — the DOM-level rendering
    path is tested separately or implicitly through the app integration tests.
    """

    def test_move_up_decrements_selected(self) -> None:
        panel = _make_panel(["a", "b", "c"], selected=2)
        with patch.object(panel, "_update_content"):
            panel.action_move_up()
        assert panel._selected == 1

    def test_move_up_noop_at_first_item(self) -> None:
        panel = _make_panel(["a", "b"], selected=0)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_up()
        mock_update.assert_not_called()
        assert panel._selected == 0

    def test_move_up_adjusts_offset_when_selection_scrolls_above_view(self) -> None:
        panel = _make_panel(list("abcde"), selected=3, offset=3)
        with patch.object(panel, "_update_content"):
            panel.action_move_up()
        assert panel._selected == 2
        assert panel._offset == 2

    def test_move_up_does_not_change_offset_when_selection_stays_visible(self) -> None:
        panel = _make_panel(list("abcde"), selected=2, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_up()
        assert panel._selected == 1
        assert panel._offset == 0

    def test_move_down_increments_selected(self) -> None:
        panel = _make_panel(["a", "b", "c"], selected=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_down()
        assert panel._selected == 1

    def test_move_down_noop_at_last_item(self) -> None:
        panel = _make_panel(["a", "b"], selected=1)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_down()
        mock_update.assert_not_called()
        assert panel._selected == 1

    def test_move_down_advances_offset_when_selection_leaves_page(self) -> None:
        # PAGE_SIZE = 12; with 14 files at selected=11, offset=0 → moving down
        # pushes selected to 12 which is >= 0 + 12, so offset becomes 1.
        files = [str(i) for i in range(14)]
        panel = _make_panel(files, selected=11, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_down()
        assert panel._selected == 12
        assert panel._offset == 1

    def test_move_down_does_not_change_offset_when_selection_stays_in_page(self) -> None:
        panel = _make_panel(list("abcde"), selected=0, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_down()
        assert panel._selected == 1
        assert panel._offset == 0

    def test_move_home_jumps_to_first_item(self) -> None:
        panel = _make_panel(["a", "b", "c"], selected=2, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_home()
        assert panel._selected == 0
        assert panel._offset == 0

    def test_move_home_resets_offset(self) -> None:
        files = [str(i) for i in range(20)]
        panel = _make_panel(files, selected=15, offset=4)
        with patch.object(panel, "_update_content"):
            panel.action_move_home()
        assert panel._selected == 0
        assert panel._offset == 0

    def test_move_home_noop_on_empty_list(self) -> None:
        panel = _make_panel([])
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_home()
        mock_update.assert_not_called()

    def test_move_end_jumps_to_last_item(self) -> None:
        panel = _make_panel(["a", "b", "c"], selected=0, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_end()
        assert panel._selected == 2

    def test_move_end_sets_offset_to_last_page(self) -> None:
        files = [str(i) for i in range(15)]
        panel = _make_panel(files, selected=0, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_end()
        assert panel._selected == 14
        # PAGE_SIZE=12; last page starts at max(0, 15-12)=3
        assert panel._offset == 3

    def test_move_end_noop_on_empty_list(self) -> None:
        panel = _make_panel([])
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_end()
        mock_update.assert_not_called()

    def test_move_page_up_jumps_by_page_size_and_adjusts_offset(self) -> None:
        # 20 files, selected=15, offset=4, PAGE_SIZE=12
        # → new_selected = max(0, 15-12) = 3; 3 < 4 → offset = 3
        files = [str(i) for i in range(20)]
        panel = _make_panel(files, selected=15, offset=4)
        with patch.object(panel, "_update_content"):
            panel.action_move_page_up()
        assert panel._selected == 3
        assert panel._offset == 3

    def test_move_page_up_noop_when_already_at_first(self) -> None:
        panel = _make_panel(["a", "b", "c"], selected=0)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_page_up()
        mock_update.assert_not_called()
        assert panel._selected == 0

    def test_move_page_up_noop_on_empty_list(self) -> None:
        panel = _make_panel([])
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_page_up()
        mock_update.assert_not_called()

    def test_move_page_up_clamps_to_first_item(self) -> None:
        files = [str(i) for i in range(5)]
        panel = _make_panel(files, selected=3, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_page_up()
        assert panel._selected == 0  # max(0, 3-12) = 0

    def test_move_page_down_jumps_by_page_size_and_adjusts_offset(self) -> None:
        # 20 files, selected=5, offset=0, PAGE_SIZE=12
        # → new_selected = min(19, 5+12) = 17; 17 >= 0+12 → offset = 17-12+1 = 6
        files = [str(i) for i in range(20)]
        panel = _make_panel(files, selected=5, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_page_down()
        assert panel._selected == 17
        assert panel._offset == 6

    def test_move_page_down_noop_when_already_at_last(self) -> None:
        panel = _make_panel(["a", "b", "c"], selected=2)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_page_down()
        mock_update.assert_not_called()
        assert panel._selected == 2

    def test_move_page_down_noop_on_empty_list(self) -> None:
        panel = _make_panel([])
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_page_down()
        mock_update.assert_not_called()

    def test_move_page_down_clamps_to_last_item(self) -> None:
        files = [str(i) for i in range(5)]
        panel = _make_panel(files, selected=2, offset=0)
        with patch.object(panel, "_update_content"):
            panel.action_move_page_down()
        assert panel._selected == 4  # min(4, 2+12) = 4

    def test_move_up_calls_update_content_on_actual_move(self) -> None:
        panel = _make_panel(["a", "b"], selected=1)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_up()
        mock_update.assert_called_once()

    def test_move_down_calls_update_content_on_actual_move(self) -> None:
        panel = _make_panel(["a", "b"], selected=0)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_down()
        mock_update.assert_called_once()

    def test_move_home_calls_update_content(self) -> None:
        panel = _make_panel(["a", "b"], selected=1)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_home()
        mock_update.assert_called_once()

    def test_move_end_calls_update_content(self) -> None:
        panel = _make_panel(["a", "b"], selected=0)
        with patch.object(panel, "_update_content") as mock_update:
            panel.action_move_end()
        mock_update.assert_called_once()


# ---------------------------------------------------------------------------
# PendingMessageBubble — on_mount markup
# ---------------------------------------------------------------------------


class TestPendingMessageBubbleMount:
    def test_on_mount_calls_update(self) -> None:
        bubble = PendingMessageBubble("my pending query")
        bubble.update = MagicMock()
        bubble.on_mount()
        bubble.update.assert_called_once()

    def test_on_mount_renders_queued_label(self) -> None:
        bubble = PendingMessageBubble("my pending query")
        bubble.update = MagicMock()
        bubble.on_mount()
        rendered: Text = bubble.update.call_args[0][0]
        assert "Queued" in rendered.plain

    def test_on_mount_includes_message_text(self) -> None:
        bubble = PendingMessageBubble("analyse this dataset")
        bubble.update = MagicMock()
        bubble.on_mount()
        rendered: Text = bubble.update.call_args[0][0]
        assert "analyse this dataset" in rendered.plain

    def test_on_mount_renders_no_emoji(self) -> None:
        bubble = PendingMessageBubble("query")
        bubble.update = MagicMock()
        bubble.on_mount()
        rendered: Text = bubble.update.call_args[0][0]
        assert "⏳" not in rendered.plain


# ---------------------------------------------------------------------------
# PendingMessagePanel — add_pending appends in queued order
# ---------------------------------------------------------------------------


class TestPendingMessagePanelAddPending:
    def test_add_pending_appends_bubble(self) -> None:
        panel = PendingMessagePanel.__new__(PendingMessagePanel)
        panel.mount = MagicMock()
        bubble_sentinel = MagicMock()

        with patch("opendatasci._tui.chat.widgets.PendingMessageBubble", return_value=bubble_sentinel):
            panel.add_pending("queued message")

        panel.mount.assert_called_once_with(bubble_sentinel)

    def test_add_pending_creates_bubble_with_correct_text(self) -> None:
        panel = PendingMessagePanel.__new__(PendingMessagePanel)
        panel.mount = MagicMock()

        with patch("opendatasci._tui.chat.widgets.PendingMessageBubble") as mock_cls:
            mock_cls.return_value = MagicMock()
            panel.add_pending("text to queue")

        mock_cls.assert_called_once_with("text to queue")

    def test_add_pending_returns_the_bubble(self) -> None:
        panel = PendingMessagePanel.__new__(PendingMessagePanel)
        panel.mount = MagicMock()
        sentinel = MagicMock()

        with patch("opendatasci._tui.chat.widgets.PendingMessageBubble", return_value=sentinel):
            result = panel.add_pending("msg")

        assert result is sentinel


# ---------------------------------------------------------------------------
# TurnStatusBar.stop — time label and interval teardown
# ---------------------------------------------------------------------------


class TestTurnStatusBarStop:
    def _bar(self) -> TurnStatusBar:
        t = TurnStatusBar.__new__(TurnStatusBar)
        t._stopped = False
        t._mounted = True
        t._start = time.monotonic()
        t._interval = MagicMock()
        t._context_tokens = None
        t._cached_tokens = None
        return t

    def test_stop_sets_stopped_flag(self) -> None:
        t = self._bar()
        t.update = MagicMock()
        t.stop()
        assert t._stopped is True

    def test_stop_calls_interval_stop(self) -> None:
        t = self._bar()
        t.update = MagicMock()
        t.stop()
        t._interval.stop.assert_called_once()

    def test_stop_updates_label_to_scienced_for(self) -> None:
        t = self._bar()
        t.update = MagicMock()
        t.stop()
        rendered: str = t.update.call_args[0][0]
        assert "Scienced for" in rendered

    def test_stop_noop_when_already_stopped(self) -> None:
        t = self._bar()
        t._stopped = True
        t.update = MagicMock()
        t.stop()
        t.update.assert_not_called()
        t._interval.stop.assert_not_called()

    def test_stop_noop_when_not_mounted(self) -> None:
        t = self._bar()
        t._mounted = False
        t.update = MagicMock()
        t.stop()
        t.update.assert_not_called()

    def test_stop_with_context_tokens_includes_context_in_label(self) -> None:
        t = self._bar()
        t._context_tokens = 5000
        t.update = MagicMock()
        t.stop()
        rendered: str = t.update.call_args[0][0]
        assert "Context:" in rendered

    def test_stop_interval_none_does_not_raise(self) -> None:
        t = self._bar()
        t._interval = None
        t.update = MagicMock()
        t.stop()  # must not raise


# ---------------------------------------------------------------------------
# TurnStatusBar.on_unmount — prevents double-stop
# ---------------------------------------------------------------------------


class TestTurnStatusBarOnUnmount:
    def test_on_unmount_stops_interval_when_mounted_and_not_yet_stopped(self) -> None:
        t = TurnStatusBar.__new__(TurnStatusBar)
        t._mounted = True
        t._stopped = False
        t._interval = MagicMock()
        t.on_unmount()
        assert t._stopped is True
        t._interval.stop.assert_called_once()

    def test_on_unmount_noop_when_already_stopped(self) -> None:
        t = TurnStatusBar.__new__(TurnStatusBar)
        t._mounted = True
        t._stopped = True
        t._interval = MagicMock()
        t.on_unmount()
        t._interval.stop.assert_not_called()

    def test_on_unmount_noop_when_never_mounted(self) -> None:
        t = TurnStatusBar.__new__(TurnStatusBar)
        t._mounted = False
        t._stopped = False
        t._interval = MagicMock()
        t.on_unmount()
        t._interval.stop.assert_not_called()

    def test_on_unmount_after_stop_is_idempotent(self) -> None:
        t = TurnStatusBar.__new__(TurnStatusBar)
        t._mounted = True
        t._stopped = False
        t._start = time.monotonic()
        t._interval = MagicMock()
        t._context_tokens = None
        t._cached_tokens = None
        t.update = MagicMock()
        t.stop()  # first stop
        t._interval.reset_mock()
        t.on_unmount()  # unmount must not double-stop
        t._interval.stop.assert_not_called()


# ---------------------------------------------------------------------------
# MessageBubble._refresh_content — non-agent roles
# ---------------------------------------------------------------------------


class TestMessageBubbleRefreshContentNonAgent:
    def _mounted_bubble(self, role: str, content: str) -> MessageBubble:
        bubble = _make_bubble(role, content)
        bubble._inner = MagicMock(spec=Static)
        return bubble

    def test_user_role_updates_inner_with_content(self) -> None:
        bubble = self._mounted_bubble("user", "Hello")
        bubble._refresh_content()
        bubble._inner.update.assert_called_once()
        rendered: Text = bubble._inner.update.call_args[0][0]
        assert "Hello" in rendered.plain

    def test_question_role_updates_inner_with_content(self) -> None:
        bubble = self._mounted_bubble("question", "What would you like?")
        bubble._refresh_content()
        bubble._inner.update.assert_called_once()
        rendered: Text = bubble._inner.update.call_args[0][0]
        assert "What would you like?" in rendered.plain

    def test_question_role_handles_rich_markup_error_gracefully(self) -> None:
        # Rich may reject invalid markup; the question role catches that and
        # falls back to plain Text(content).
        bubble = self._mounted_bubble("question", "[broken")
        bubble._refresh_content()  # must not raise
        bubble._inner.update.assert_called_once()

    def test_agent_role_does_nothing_synchronously(self) -> None:
        # Agent rendering is fully async via _flush_agent; _refresh_content
        # must be a no-op so it never races with the async update path.
        bubble = _make_bubble("agent", "response text")
        bubble._inner = MagicMock(spec=TUIMarkdown)
        bubble._refresh_content()
        bubble._inner.update.assert_not_called()

    def test_refresh_content_noop_when_inner_not_yet_set(self) -> None:
        bubble = _make_bubble("user", "Hello")
        bubble._inner = None
        bubble._refresh_content()  # must not raise


# ---------------------------------------------------------------------------
# Auto-scroll (2.1.1) — anchor detection and mount-time pinning
# ---------------------------------------------------------------------------


def _fake_scroll_container(offset_y: int, max_scroll_y: int) -> MagicMock:
    container = MagicMock()
    container.scroll_offset.y = offset_y
    container.max_scroll_y = max_scroll_y
    return container


class TestScrollIsAtBottom:
    def test_true_when_exactly_at_bottom(self) -> None:
        assert _scroll_is_at_bottom(_fake_scroll_container(10, 10)) is True

    def test_true_within_one_row_of_bottom(self) -> None:
        assert _scroll_is_at_bottom(_fake_scroll_container(9, 10)) is True

    def test_false_when_scrolled_up(self) -> None:
        assert _scroll_is_at_bottom(_fake_scroll_container(3, 10)) is False

    def test_true_when_content_fits_viewport(self) -> None:
        # Nothing to scroll yet — the anchor must engage from the start.
        assert _scroll_is_at_bottom(_fake_scroll_container(0, 0)) is True


class TestMessagesContainerAnchor:
    """Bottom-pinning is delegated to Textual's built-in scroll anchor."""

    def test_on_mount_arms_the_native_anchor(self) -> None:
        container = MessagesContainer.__new__(MessagesContainer)
        container.anchor = MagicMock()  # type: ignore[method-assign]
        container.on_mount()
        container.anchor.assert_called_once_with()


class TestChatPaneMountAutoScroll:
    """_mount_in_messages mounts new content in #messages."""

    def _pane(self) -> tuple[ChatPane, MagicMock]:
        pane = _make_chat_pane()
        container = MagicMock()
        pane.query_one = MagicMock(return_value=container)
        return pane, container

    def test_mounts_widget_in_messages_container(self) -> None:
        pane, container = self._pane()
        widget = MagicMock()
        pane._mount_in_messages(widget)
        container.mount.assert_called_once_with(widget)

    def test_add_message_uses_autoscroll_mount(self) -> None:
        pane = _make_chat_pane()
        pane._mount_in_messages = MagicMock()
        bubble = pane.add_message("user", "hi")
        pane._mount_in_messages.assert_called_once_with(bubble)

    def test_add_ephemeral_block_uses_autoscroll_mount(self) -> None:
        pane = _make_chat_pane()
        pane._mount_in_messages = MagicMock()
        block = pane.add_ephemeral_block("comm", "label", "summary")
        pane._mount_in_messages.assert_called_once_with(block)
