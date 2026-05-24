"""Unit tests for opendatasci._tui.widgets — pure logic only (no Textual app context)."""


from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from rich.text import Text
from textual import events as textual_events
from textual.widgets import Markdown as TUIMarkdown
from textual.widgets import Static

from opendatasci._tui.models import SPINNER, SPINNER_INTERVAL
from opendatasci._tui.widgets import (
    AppHeader,
    AttachmentBar,
    ChatPane,
    CommandHighlighter,
    MessageBubble,
    SmartInput,
    ToolCallBlock,
    TurnStatusBar,
    _InputHistory,
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
    header._provider = "anthropic"
    header._model = "claude-sonnet-4-6"
    header._workspace = "/tmp/data"
    header._workspace_name = None
    header._file_count = ""
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

    def test_full_cache_shows_100_percent(self) -> None:
        suffix = self._bar(5000, 5000)._context_suffix()
        assert "(100.0% cached)" in suffix


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
    worker_summaries: list[str] | None = None,
) -> ToolCallBlock:
    """Instantiate ToolCallBlock bypassing Textual Widget.__init__."""
    block = ToolCallBlock.__new__(ToolCallBlock)
    block._communication = communication
    block._label = label
    block._summary = summary
    block._worker_summaries = worker_summaries or []
    block._worker_statuses = ["running"] * len(block._worker_summaries)
    block._worker_activities = [""] * len(block._worker_summaries)
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

    def test_mark_worker_done_sets_status(self) -> None:
        block = _make_block(worker_summaries=["w1", "w2"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_worker_done(0)
        assert block._worker_statuses[0] == "done"
        assert block._worker_statuses[1] == "running"

    def test_mark_worker_error_sets_status(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_worker_error(0)
        assert block._worker_statuses[0] == "error"

    def test_mark_worker_done_all_terminal_stops_spinner(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        stop_called = []
        with (
            patch.object(block, "_refresh"),
            patch.object(block, "_stop_spinner", side_effect=lambda: stop_called.append(True)),
        ):
            block.mark_worker_done(0)
        assert stop_called

    def test_mark_worker_done_all_terminal_sets_done(self) -> None:
        block = _make_block(worker_summaries=["w1", "w2"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_worker_done(0)
        assert block._done is False  # only one of two workers done
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_worker_done(1)
        assert block._done is True  # all workers done → block is definitively closed

    def test_mark_worker_error_all_terminal_sets_done(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_worker_error(0)
        assert block._done is True

    def test_mark_worker_done_partial_does_not_set_done(self) -> None:
        block = _make_block(worker_summaries=["w1", "w2", "w3"])
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_worker_done(0)
            block.mark_worker_done(1)
        assert block._done is False  # third worker still running

    def test_update_worker_activity_updates_running_worker(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        with patch.object(block, "_refresh"):
            block.update_worker_activity(0, "running tool X")
        assert block._worker_activities[0] == "running tool X"

    def test_update_worker_activity_ignored_for_done_worker(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        block._worker_statuses[0] = "done"
        with patch.object(block, "_refresh"):
            block.update_worker_activity(0, "should be ignored")
        assert block._worker_activities[0] == ""

    def test_update_worker_activity_skips_refresh_when_unchanged(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        block._worker_activities[0] = "tool X"
        with patch.object(block, "_refresh") as refresh:
            block.update_worker_activity(0, "tool X")
        refresh.assert_not_called()

    def test_update_worker_activity_calls_refresh_when_changed(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        with patch.object(block, "_refresh") as refresh:
            block.update_worker_activity(0, "tool X")
        refresh.assert_called_once()

    def test_mark_worker_done_clears_activity(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        block._worker_activities[0] = "some activity"
        with patch.object(block, "_refresh"), patch.object(block, "_stop_spinner"):
            block.mark_worker_done(0)
        assert block._worker_activities[0] == ""


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

    def test_worker_status_markup_error_contains_x(self) -> None:
        block = _make_block()
        markup = block._worker_status_markup("worker", "error")
        assert "✗" in markup

    def test_worker_status_markup_done_no_checkmark(self) -> None:
        block = _make_block()
        markup = block._worker_status_markup("worker", "done")
        assert "✓" not in markup
        assert "worker" in markup
        assert "[bold" in markup  # still styled

    def test_worker_status_markup_running_contains_spinner(self) -> None:
        block = _make_block()
        markup = block._worker_status_markup("worker", "running")
        assert "worker" in markup


class TestToolCallBlockRefreshForcedTerminalState:
    """When the block is force-closed via set_done/set_error while individual
    worker rows are still 'running', _refresh must promote those rows to a
    state consistent with how the block itself closed: 'done' for set_done,
    'error' for set_error. Promoting an unfinished row to 'done' under
    set_error would hide a failure behind a green row."""

    def test_set_error_promotes_running_rows_to_error(self) -> None:
        block = _make_block(worker_summaries=["w1", "w2"])
        block.update = MagicMock()  # type: ignore[assignment]
        block.set_error()
        rendered = str(block.update.call_args.args[0])
        # Both rows should reflect the error state — '✗' is the row-level error glyph.
        assert rendered.count("✗") >= 2

    def test_set_done_keeps_running_rows_as_done(self) -> None:
        block = _make_block(worker_summaries=["w1"])
        block.update = MagicMock()  # type: ignore[assignment]
        block.set_done()
        rendered = str(block.update.call_args.args[0])
        # set_done should not flip running rows into the error glyph.
        assert "✗" not in rendered

    def test_set_error_keeps_already_done_rows_done(self) -> None:
        block = _make_block(worker_summaries=["w1", "w2"])
        block._worker_statuses[0] = "done"
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
    """add_turn_status_bar mounts the new bar on the pane after the input-bar."""

    def _setup_pane(self, existing_bars: list | None = None) -> tuple[ChatPane, MagicMock]:
        pane = _make_chat_pane()
        input_bar = MagicMock()
        pane.query = MagicMock(return_value=existing_bars or [])
        pane.query_one = MagicMock(return_value=input_bar)
        pane.mount = MagicMock()
        return pane, input_bar

    def test_bar_is_mounted_on_pane_after_input_bar(self) -> None:
        pane, input_bar = self._setup_pane()
        bar = MagicMock()

        with patch("opendatasci._tui.widgets.TurnStatusBar", return_value=bar):
            result = pane.add_turn_status_bar()

        pane.mount.assert_called_once_with(bar, after=input_bar)
        assert result is bar

    def test_existing_bars_are_removed_before_mounting(self) -> None:
        stale_a, stale_b = MagicMock(), MagicMock()
        pane, _ = self._setup_pane(existing_bars=[stale_a, stale_b])

        with patch("opendatasci._tui.widgets.TurnStatusBar", return_value=MagicMock()):
            pane.add_turn_status_bar()

        stale_a.remove.assert_called_once()
        stale_b.remove.assert_called_once()

    def test_new_bar_is_mounted_on_pane(self) -> None:
        pane, input_bar = self._setup_pane()
        bar = MagicMock()

        with patch("opendatasci._tui.widgets.TurnStatusBar", return_value=bar):
            pane.add_turn_status_bar()

        pane.mount.assert_called_once_with(bar, after=input_bar)

    def test_returns_the_newly_created_bar(self) -> None:
        pane, _ = self._setup_pane()
        bar = MagicMock()

        with patch("opendatasci._tui.widgets.TurnStatusBar", return_value=bar):
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

    def test_communication_shown_with_blank_separator_for_worker_blocks(self) -> None:
        block = _make_block(worker_summaries=["worker-1", "worker-2"])
        block._communication = "Running checks in parallel."
        lines = self._rendered_lines(block)
        # Worker path renders communication → blank separator → header → worker rows
        assert lines[0] == "Running checks in parallel."
        assert lines[1] == ""
        assert lines[2] != ""  # header line
        assert lines[3] != ""  # worker 1
        assert lines[4] != ""  # worker 2

    def test_no_communication_no_blank_line_for_worker_blocks(self) -> None:
        block = _make_block(worker_summaries=["worker-1", "worker-2"])
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


class TestToolCallBlockWorkerRowRendering:
    """Worker block renders a subtree:

        ⚡ Spawned parallel workers
          └─ ⣾ Worker 1: …
          └─ ⣾ Worker 2: …

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
        block = _make_block(communication="", worker_summaries=["w1"])
        lines = self._rendered_lines(block)
        # While workers run, the spinner is prepended to the header by
        # _status_markup; expected layout is "{spin} ⚡ Spawned parallel workers".
        assert "⚡ Spawned parallel workers" in lines[0]
        assert not lines[0].rstrip().endswith(":")

    def test_header_done_state_has_no_trailing_colon(self) -> None:
        block = _make_block(communication="", worker_summaries=["w1"])
        block._worker_statuses[0] = "done"
        lines = self._rendered_lines(block)
        # All workers terminal → header rendered in the done style (no spinner).
        assert lines[0] == "⚡ Spawned parallel workers"

    def test_running_worker_row_has_indented_tree_prefix_before_spinner(self) -> None:
        block = _make_block(communication="", worker_summaries=["w1"])
        lines = self._rendered_lines(block)
        # Spinner is SPINNER[0] = "⣾"; tree prefix sits BEFORE the spinner.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}⣾ Worker 1: w1"

    def test_done_worker_row_preserves_tree_prefix(self) -> None:
        block = _make_block(communication="", worker_summaries=["w1"])
        block._worker_statuses[0] = "done"
        lines = self._rendered_lines(block)
        # No spinner in done state — prefix still leads, then the row text.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}Worker 1: w1"

    def test_error_worker_row_preserves_tree_prefix_before_x_glyph(self) -> None:
        block = _make_block(communication="", worker_summaries=["w1"])
        block._worker_statuses[0] = "error"
        lines = self._rendered_lines(block)
        # Error glyph "✗" sits inside the status markup; prefix is still external.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}✗ Worker 1: w1"

    def test_running_worker_row_with_activity_keeps_tree_prefix(self) -> None:
        block = _make_block(communication="", worker_summaries=["w1"])
        block._worker_activities[0] = "🐍 running pandas"
        lines = self._rendered_lines(block)
        # Activity replaces the subtask summary while running; prefix unchanged.
        assert lines[1] == f"{self.WORKER_UPDATE_BRANCH}⣾ Worker 1: 🐍 running pandas"


def _make_bubble(role: str, content: str = "") -> MessageBubble:
    """Instantiate a MessageBubble bypassing Textual Widget.__init__."""
    bubble = MessageBubble.__new__(MessageBubble)
    bubble._role = role
    bubble._content = content
    bubble._spin_idx = 0
    bubble._spin_label = "Thinking"
    bubble._spin_timer = None
    bubble._inner = None
    bubble._summary_text = None
    bubble._flush_timer = None
    bubble._dirty = False
    bubble._flush_scheduled = False
    return bubble


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
        # still produces an empty Markdown widget; on_mount() schedules a flush
        # to render whatever has accumulated in _content.
        bubble = _make_bubble("agent", "")
        bubble._content = "tok1tok2"
        bubble._dirty = True
        inner = self._composed_inner(bubble)
        assert isinstance(inner, TUIMarkdown)
        assert inner._markdown == ""

    def test_user_role_yields_static_widget(self) -> None:
        bubble = _make_bubble("user", "Hello")
        inner = self._composed_inner(bubble)
        assert isinstance(inner, Static)
        assert not isinstance(inner, TUIMarkdown)

    def test_thinking_role_yields_static_widget(self) -> None:
        bubble = _make_bubble("thinking", "")
        inner = self._composed_inner(bubble)
        assert isinstance(inner, Static)

    def test_question_role_yields_static_widget(self) -> None:
        bubble = _make_bubble("question", "Choose")
        inner = self._composed_inner(bubble)
        assert isinstance(inner, Static)


class TestMessageBubbleOnMountSafetyNet:
    """on_mount() must catch up any flush that fired before mount completed.

    This is the core regression test for the "agent message never appears"
    bug.  When finish() / set_content() schedule a flush via
    call_after_refresh, that flush can fire before the bubble has finished
    mounting.  In that case _flush_agent early-returns leaving _dirty=True;
    on_mount must re-schedule a flush so the buffered content eventually
    reaches the Markdown widget.
    """

    def test_dirty_agent_bubble_schedules_flush_on_mount(self) -> None:
        bubble = _make_bubble("agent", "Hi")
        bubble._dirty = True
        bubble._flush_scheduled = False
        bubble._inner = MagicMock(spec=TUIMarkdown)
        with (
            patch.object(bubble, "_schedule_final_flush") as schedule,
            patch.object(bubble, "set_interval"),
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        schedule.assert_called_once()

    def test_clean_agent_bubble_does_not_schedule_flush_on_mount(self) -> None:
        bubble = _make_bubble("agent", "")
        bubble._dirty = False
        with (
            patch.object(bubble, "_schedule_final_flush") as schedule,
            patch.object(bubble, "set_interval"),
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        schedule.assert_not_called()

    def test_agent_bubble_with_initial_content_schedules_flush_on_mount(self) -> None:
        # Regression: add_message("agent", text).finish() sets content in the
        # constructor and calls finish() before mount completes.  on_mount() must
        # schedule a flush so the content reaches the Markdown widget even when
        # _dirty was False before on_mount ran (e.g. finish() already cleared it).
        bubble = _make_bubble("agent", "✓ Context cleared.")
        bubble._dirty = False  # simulate: finish() ran and _flush_agent already cleared dirty
        bubble._inner = MagicMock(spec=TUIMarkdown)
        with (
            patch.object(bubble, "_schedule_final_flush") as schedule,
            patch.object(bubble, "set_interval"),
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        schedule.assert_called_once()

    def test_dirty_user_bubble_does_not_schedule_flush_on_mount(self) -> None:
        # The flush mechanism is agent-only; other roles render synchronously
        # via _refresh_content.
        bubble = _make_bubble("user", "Hello")
        bubble._dirty = True
        with (
            patch.object(bubble, "_schedule_final_flush") as schedule,
            patch.object(bubble, "set_interval"),
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        schedule.assert_not_called()

    def test_dirty_thinking_bubble_does_not_schedule_flush_on_mount(self) -> None:
        bubble = _make_bubble("thinking", "")
        bubble._dirty = True
        with (
            patch.object(bubble, "_schedule_final_flush") as schedule,
            patch.object(bubble, "set_interval"),
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        schedule.assert_not_called()

    def test_thinking_bubble_starts_spinner_on_mount(self) -> None:
        bubble = _make_bubble("thinking", "")
        with (
            patch.object(bubble, "set_interval", return_value="timer-sentinel") as si,
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        si.assert_called_once()
        assert bubble._spin_timer == "timer-sentinel"

    def test_agent_bubble_does_not_start_spinner_on_mount(self) -> None:
        bubble = _make_bubble("agent", "")
        with (
            patch.object(bubble, "set_interval") as si,
            patch.object(bubble, "_refresh_content"),
            patch.object(bubble, "_schedule_final_flush"),
        ):
            bubble.on_mount()
        si.assert_not_called()
        assert bubble._spin_timer is None


class TestMessageBubbleScheduleFinalFlush:
    """_schedule_final_flush must be idempotent within one refresh cycle."""

    def test_first_call_posts_callback(self) -> None:
        bubble = _make_bubble("agent", "")
        bubble._flush_scheduled = False
        with patch.object(bubble, "call_after_refresh") as car:
            bubble._schedule_final_flush()
        car.assert_called_once_with(bubble._flush_agent)
        assert bubble._flush_scheduled is True

    def test_second_call_is_deduplicated(self) -> None:
        bubble = _make_bubble("agent", "")
        bubble._flush_scheduled = True  # already scheduled
        with patch.object(bubble, "call_after_refresh") as car:
            bubble._schedule_final_flush()
        car.assert_not_called()


class TestMessageBubbleAppend:
    """append() must mark the bubble dirty and start the rate-limit timer."""

    def test_agent_append_marks_dirty(self) -> None:
        bubble = _make_bubble("agent", "")
        with patch.object(bubble, "set_interval", return_value=MagicMock(name="timer")):
            bubble.append("hello")
        assert bubble._content == "hello"
        assert bubble._dirty is True

    def test_agent_append_starts_timer_only_once(self) -> None:
        bubble = _make_bubble("agent", "")
        timer_sentinel = MagicMock(name="flush_timer")
        with patch.object(bubble, "set_interval", return_value=timer_sentinel) as si:
            bubble.append("a")
            bubble.append("b")
        # The interval is created only on the first append; subsequent appends
        # reuse it.
        assert si.call_count == 1
        assert bubble._content == "ab"

    def test_agent_flush_timer_rate_is_throttled(self) -> None:
        bubble = _make_bubble("agent", "")
        with patch.object(bubble, "set_interval", return_value=MagicMock()) as si:
            bubble.append("hello")
        interval = si.call_args[0][0]
        assert (
            interval >= 0.2
        ), f"flush interval {interval}s is too fast — would cause excessive Markdown rebuilds"

    def test_non_agent_append_calls_refresh_content(self) -> None:
        bubble = _make_bubble("user", "Hello")
        with patch.object(bubble, "_refresh_content") as refresh:
            bubble.append(" world")
        refresh.assert_called_once()
        assert bubble._content == "Hello world"
        assert bubble._dirty is False  # not used for non-agent


class TestMessageBubbleSetContent:
    """set_content must replace the buffer and schedule a final flush for agents."""

    def test_agent_set_content_marks_dirty_and_schedules_flush(self) -> None:
        bubble = _make_bubble("agent", "")
        with (
            patch.object(bubble, "_stop_flush_timer") as stop,
            patch.object(bubble, "_schedule_final_flush") as schedule,
        ):
            bubble.set_content("replaced")
        assert bubble._content == "replaced"
        assert bubble._dirty is True
        stop.assert_called_once()
        schedule.assert_called_once()

    def test_non_agent_set_content_calls_refresh_content(self) -> None:
        bubble = _make_bubble("user", "old")
        with patch.object(bubble, "_refresh_content") as refresh:
            bubble.set_content("new")
        assert bubble._content == "new"
        refresh.assert_called_once()


class TestMessageBubbleFinish:
    """finish() must stop the streaming timer and request one final flush."""

    def test_agent_finish_marks_dirty_and_schedules_flush(self) -> None:
        bubble = _make_bubble("agent", "Hi")
        with (
            patch.object(bubble, "_stop_flush_timer") as stop,
            patch.object(bubble, "_schedule_final_flush") as schedule,
        ):
            bubble.finish()
        assert bubble._dirty is True
        stop.assert_called_once()
        schedule.assert_called_once()

    def test_thinking_finish_sets_done_summary_when_no_summary(self) -> None:
        bubble = _make_bubble("thinking", "")
        bubble._summary_text = None
        with patch.object(bubble, "_refresh_content"):
            bubble.finish()
        assert bubble._summary_text == "Done thinking"


class TestThinkingBubbleRefreshContent:
    """_refresh_content for the 'thinking' role must render streamed content
    (debug mode) instead of the spinner when _content is non-empty."""

    def _make_mounted_thinking(self, content: str = "") -> MessageBubble:
        bubble = _make_bubble("thinking", content)
        bubble._inner = MagicMock(spec=Static)
        return bubble

    def test_empty_content_shows_spinner(self) -> None:
        bubble = self._make_mounted_thinking("")
        bubble._refresh_content()
        rendered: Text = bubble._inner.update.call_args[0][0]
        assert "Thinking" in rendered.plain

    def test_non_empty_content_shows_content_not_spinner(self) -> None:
        bubble = self._make_mounted_thinking("step one → step two")
        bubble._refresh_content()
        rendered: Text = bubble._inner.update.call_args[0][0]
        assert "step one → step two" in rendered.plain
        assert "Thinking" not in rendered.plain

    def test_non_empty_content_stops_spin_timer(self) -> None:
        bubble = self._make_mounted_thinking("some reasoning")
        timer = MagicMock()
        bubble._spin_timer = timer
        bubble._refresh_content()
        timer.stop.assert_called_once()
        assert bubble._spin_timer is None

    def test_spin_timer_stop_is_idempotent_when_already_none(self) -> None:
        bubble = self._make_mounted_thinking("reasoning")
        bubble._spin_timer = None
        bubble._refresh_content()  # must not raise

    def test_summary_takes_precedence_over_content(self) -> None:
        bubble = self._make_mounted_thinking("reasoning text")
        bubble._summary_text = "Thought for 3s"
        bubble._refresh_content()
        rendered: Text = bubble._inner.update.call_args[0][0]
        assert "Thought for 3s" in rendered.plain
        assert "reasoning text" not in rendered.plain

    def test_content_rendered_in_muted_grey(self) -> None:
        from opendatasci._tui import theme as _theme

        bubble = self._make_mounted_thinking("ponder this")
        bubble._refresh_content()
        rendered: Text = bubble._inner.update.call_args[0][0]
        muted_color = _theme.active["text_muted"]
        assert any(muted_color in str(span.style) for span in rendered._spans)

    def test_content_prefixed_with_thoughts(self) -> None:
        bubble = self._make_mounted_thinking("step one")
        bubble._refresh_content()
        rendered: Text = bubble._inner.update.call_args[0][0]
        assert rendered.plain.startswith("Reasoning:")
        assert "step one" in rendered.plain


class TestMessageBubbleFlushAgent:
    """_flush_agent must NEVER drop content silently when the bubble isn't
    yet ready — it must leave _dirty=True so the on_mount safety net (or the
    streaming timer) can pick it back up."""

    @pytest.mark.asyncio
    async def test_flush_agent_no_op_when_not_dirty(self) -> None:
        bubble = _make_bubble("agent", "abc")
        bubble._dirty = False
        bubble._flush_scheduled = True
        bubble._inner = MagicMock(spec=TUIMarkdown)
        await bubble._flush_agent()
        assert bubble._flush_scheduled is False
        bubble._inner.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_agent_preserves_dirty_when_inner_is_none(self) -> None:
        # Regression: when _flush_agent runs before compose() has set _inner,
        # it must leave _dirty=True so on_mount re-schedules the flush.
        bubble = _make_bubble("agent", "buffered")
        bubble._dirty = True
        bubble._flush_scheduled = True
        bubble._inner = None
        with patch.object(MessageBubble, "is_mounted", property(lambda _: True)):
            await bubble._flush_agent()
        assert bubble._dirty is True
        # The scheduled-flag is reset so the on_mount safety net can schedule
        # a fresh call_after_refresh.
        assert bubble._flush_scheduled is False

    @pytest.mark.asyncio
    async def test_flush_agent_preserves_dirty_when_not_mounted(self) -> None:
        # Same regression but for the case where compose() has run (inner is
        # set) but the bubble's mount has not yet completed.
        bubble = _make_bubble("agent", "buffered")
        bubble._dirty = True
        bubble._flush_scheduled = True
        bubble._inner = MagicMock(spec=TUIMarkdown)
        with patch.object(MessageBubble, "is_mounted", property(lambda _: False)):
            await bubble._flush_agent()
        assert bubble._dirty is True
        assert bubble._flush_scheduled is False
        bubble._inner.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_agent_renders_content_when_ready(self) -> None:
        bubble = _make_bubble("agent", "rendered")
        bubble._dirty = True
        bubble._flush_scheduled = True
        inner = MagicMock(spec=TUIMarkdown)

        async def _async_update(_text):
            return None

        inner.update = MagicMock(side_effect=_async_update)
        bubble._inner = inner
        with patch.object(MessageBubble, "is_mounted", property(lambda _: True)):
            await bubble._flush_agent()
        inner.update.assert_called_once_with("rendered")
        assert bubble._dirty is False

    @pytest.mark.asyncio
    async def test_flush_agent_swallows_inner_update_exceptions(self) -> None:
        # The Markdown widget may raise from inside its async DOM operations.
        # _flush_agent must log and swallow rather than propagate, so the app
        # stays alive — but it must NOT clear _dirty (otherwise the stale
        # content is permanently lost).
        bubble = _make_bubble("agent", "boom")
        bubble._dirty = True
        bubble._flush_scheduled = True
        inner = MagicMock(spec=TUIMarkdown)

        async def _raise(_text):
            raise RuntimeError("markdown blew up")

        inner.update = MagicMock(side_effect=_raise)
        bubble._inner = inner
        with patch.object(MessageBubble, "is_mounted", property(lambda _: True)):
            await bubble._flush_agent()  # must not raise
        assert bubble._dirty is True


class TestMessageBubbleEndToEndFinishBeforeMount:
    """End-to-end coverage for the original TUI bug.

    Reproduces the exact sequence that produced empty agent bubbles:
      1. ui.add_message("agent", "")  → bubble created, mount scheduled.
      2. bubble.append("Hi")          → _dirty=True, streaming timer started.
      3. bubble.finish()              → timer stopped, single flush scheduled.
      4. The single scheduled flush fires BEFORE the bubble has finished
         mounting (early-return: _inner=None).
      5. compose() then runs, on_mount() then runs.
    The fix guarantees that step 5 schedules a fresh flush so the buffered
    content eventually reaches the Markdown widget.
    """

    @pytest.mark.asyncio
    async def test_finish_before_mount_recovers_via_on_mount(self) -> None:
        bubble = _make_bubble("agent", "")
        scheduled_callbacks: list = []
        timer_sentinel = MagicMock(name="flush_timer")

        with (
            patch.object(bubble, "set_interval", return_value=timer_sentinel),
            patch.object(bubble, "call_after_refresh", side_effect=scheduled_callbacks.append),
        ):
            # 2. tokens stream in
            bubble.append("Hi")
            assert bubble._dirty is True
            assert bubble._flush_timer is timer_sentinel

            # 3. finish() requests one final flush
            bubble.finish()
            assert bubble._dirty is True
            assert bubble._flush_scheduled is True
            assert scheduled_callbacks == [bubble._flush_agent]

            # 4. The scheduled flush fires WHILE the bubble is still mounting:
            #    inner has not been created yet and is_mounted is False.
            with patch.object(MessageBubble, "is_mounted", property(lambda _: False)):
                await bubble._flush_agent()
            # Content survived; only the scheduling flag was cleared.
            assert bubble._dirty is True
            assert bubble._flush_scheduled is False

            # 5. compose() then runs. It always creates an empty TUIMarkdown;
            # rendering is handled exclusively by _flush_agent to avoid the
            # race where two concurrent update() tasks produce duplicate text.
            inners = list(bubble.compose())
            assert len(inners) == 1
            assert isinstance(inners[0], TUIMarkdown)
            assert inners[0]._markdown == ""

            # 6. on_mount() finally fires (is_mounted has flipped to True).
            scheduled_callbacks.clear()
            with patch.object(bubble, "_refresh_content"):
                bubble.on_mount()
            # The safety net re-scheduled the flush.
            assert bubble._flush_scheduled is True
            assert scheduled_callbacks == [bubble._flush_agent]

            # 7. This time _flush_agent runs against a ready inner and renders.
            inner_mock = MagicMock(spec=TUIMarkdown)

            async def _async_update(_text):
                return None

            inner_mock.update = MagicMock(side_effect=_async_update)
            bubble._inner = inner_mock
            with patch.object(MessageBubble, "is_mounted", property(lambda _: True)):
                await bubble._flush_agent()
            inner_mock.update.assert_called_once_with("Hi")
            assert bubble._dirty is False

    @pytest.mark.asyncio
    async def test_construct_with_content_renders_via_flush_on_mount(self) -> None:
        # Covers the controller's `response`-only branch:
        #     ui.add_message("agent", "Final answer")
        # compose() always creates an empty TUIMarkdown; on_mount() detects
        # that _content is non-empty, marks the bubble dirty, and schedules a
        # flush so the content reaches the Markdown widget via _flush_agent.
        bubble = _make_bubble("agent", "Final answer")
        inners = list(bubble.compose())
        assert len(inners) == 1
        inner = inners[0]
        assert isinstance(inner, TUIMarkdown)
        assert inner._markdown == ""
        # on_mount must schedule a flush because _content is non-empty.
        with (
            patch.object(bubble, "_schedule_final_flush") as schedule,
            patch.object(bubble, "set_interval"),
            patch.object(bubble, "_refresh_content"),
        ):
            bubble.on_mount()
        schedule.assert_called_once()


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
