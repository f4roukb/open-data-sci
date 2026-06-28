"""Unit tests for opendatasci._tui.completion.CompletionState.

CompletionState is pure Python with no Textual dependency — it delegates all
UI updates through UIAdapter.  Tests supply a mock UIAdapter so the logic is
exercised without any widget tree.
"""

from unittest.mock import MagicMock, patch

import pytest

from opendatasci._tui.commands import SLASH_COMMANDS
from opendatasci._tui.completion import CompletionState


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    ui = MagicMock()
    ui.show_completion = MagicMock()
    ui.hide_completion = MagicMock()
    ui.set_input_value = MagicMock()
    return ui


def _state_with_slash_matches(value: str = "/c") -> tuple[CompletionState, MagicMock]:
    """Return a CompletionState that has slash matches for *value*."""
    state = CompletionState()
    ui = _make_ui()
    state.on_input_changed(value, ui)
    ui.reset_mock()
    return state, ui


# ---------------------------------------------------------------------------
# has_matches
# ---------------------------------------------------------------------------


class TestCompletionStateHasMatches:
    def test_false_on_fresh_state(self) -> None:
        assert CompletionState().has_matches is False

    def test_true_after_slash_match_found(self) -> None:
        state, _ = _state_with_slash_matches("/c")
        assert state.has_matches is True

    def test_false_after_hide(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.hide(ui)
        assert state.has_matches is False

    def test_false_when_no_commands_match(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        state.on_input_changed("/zzz_not_a_command", ui)
        assert state.has_matches is False


# ---------------------------------------------------------------------------
# on_input_changed — slash mode
# ---------------------------------------------------------------------------


class TestCompletionStateOnInputChangedSlash:
    def test_slash_prefix_with_matches_calls_show_completion(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        state.on_input_changed("/c", ui)
        ui.show_completion.assert_called_once()

    def test_slash_prefix_returns_false(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        result = state.on_input_changed("/c", ui)
        assert result is False

    def test_completing_flag_returns_true_and_clears_itself(self) -> None:
        # When _completing=True, on_input_changed returns True immediately without
        # triggering any popup update.  This prevents feedback loops when the tab-
        # completion handler itself changes the input value.
        state = CompletionState()
        state._completing = True
        ui = _make_ui()
        result = state.on_input_changed("/clear", ui)
        assert result is True
        assert state._completing is False
        ui.show_completion.assert_not_called()

    def test_completing_flag_suppresses_ui_updates(self) -> None:
        state = CompletionState()
        state._completing = True
        ui = _make_ui()
        state.on_input_changed("/clear", ui)
        ui.show_completion.assert_not_called()
        ui.hide_completion.assert_not_called()

    def test_exact_single_match_does_not_show_popup(self) -> None:
        # "/clear" is the only command starting with "/clear" — exact unique match.
        # The popup must not open; show_completion should not be called.
        state = CompletionState()
        ui = _make_ui()
        state.on_input_changed("/clear", ui)
        ui.show_completion.assert_not_called()
        assert state.has_matches is False

    def test_non_matching_slash_does_not_show_popup(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        state.on_input_changed("/zzz_not_a_command", ui)
        ui.show_completion.assert_not_called()
        assert state.has_matches is False

    def test_non_slash_text_hides_popup_and_clears_matches(self) -> None:
        # Populate matches first, then type plain text → popup must close.
        state, ui = _state_with_slash_matches("/c")
        state.on_input_changed("plain text", ui)
        # hide_completion IS called because there were prior matches to clear.
        ui.hide_completion.assert_called()
        assert state.has_matches is False

    def test_slash_prefix_matches_are_stored(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        state.on_input_changed("/c", ui)
        # /c matches /cancel-all-messages, /cancel-message, /clear, /compact
        assert len(state._matches) > 1
        assert all(cmd.startswith("/c") for cmd in state._matches)

    def test_slash_mode_is_set_when_matches_found(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        state.on_input_changed("/c", ui)
        assert state._mode == "slash"

    def test_show_completion_receives_display_strings_not_raw_commands(self) -> None:
        # Displays include the command description, not just the command name.
        state = CompletionState()
        ui = _make_ui()
        state.on_input_changed("/c", ui)
        # show_completion(displays, idx) — two positional args
        displays_arg, _idx_arg = ui.show_completion.call_args[0]
        # Displays must be longer strings than the raw command token alone
        assert any(len(d) > len(cmd) for d, cmd in zip(displays_arg, state._matches))

    def test_extra_commands_included_in_matching(self) -> None:
        state = CompletionState(extra_commands=["/custom-tool"])
        ui = _make_ui()
        state.on_input_changed("/cus", ui)
        assert "/custom-tool" in state._matches


# ---------------------------------------------------------------------------
# on_input_changed — @ (file) mode
# ---------------------------------------------------------------------------


class TestCompletionStateOnInputChangedAtMode:
    def test_at_fragment_with_matching_files_shows_popup(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        with patch(
            "opendatasci._tui.completion._discover_files", return_value=["data.csv"]
        ):
            state.on_input_changed("@data", ui)
        ui.show_completion.assert_called_once()

    def test_at_fragment_with_no_matches_does_not_show_popup(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        with patch("opendatasci._tui.completion._discover_files", return_value=[]):
            state.on_input_changed("@no_match_xyz", ui)
        ui.show_completion.assert_not_called()
        assert state.has_matches is False

    def test_at_mode_is_set_when_file_matches_found(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        with patch(
            "opendatasci._tui.completion._discover_files", return_value=["data.csv"]
        ):
            state.on_input_changed("@d", ui)
        assert state._mode == "file"

    def test_at_fragment_caches_discovery_results(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        with patch(
            "opendatasci._tui.completion._discover_files", return_value=["data.csv"]
        ) as mock_discover:
            state.on_input_changed("@data", ui)
            state.on_input_changed("@data", ui)  # same fragment → should use cache
        mock_discover.assert_called_once()  # only one filesystem scan

    def test_at_fragment_change_triggers_new_discovery(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        with patch(
            "opendatasci._tui.completion._discover_files", return_value=["data.csv"]
        ) as mock_discover:
            state.on_input_changed("@da", ui)
            state.on_input_changed("@dat", ui)  # fragment changed
        assert mock_discover.call_count == 2

    def test_at_position_stored_correctly(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        with patch(
            "opendatasci._tui.completion._discover_files", return_value=["data.csv"]
        ):
            state.on_input_changed("describe @d", ui)
        # "@" is at index 9 in "describe @d"
        assert state._at_pos == 9


# ---------------------------------------------------------------------------
# cycle — forward and backward navigation
# ---------------------------------------------------------------------------


class TestCompletionStateCycle:
    def test_no_matches_returns_false(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        result = state.cycle("/c", direction=1, ui=ui)
        assert result is False

    def test_slash_mode_cycle_forward_returns_true(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        result = state.cycle("/c", direction=1, ui=ui)
        assert result is True

    def test_slash_mode_cycle_forward_calls_set_input_value(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.cycle("/c", direction=1, ui=ui)
        ui.set_input_value.assert_called_once()

    def test_slash_mode_cycle_sets_completing_flag(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.cycle("/c", direction=1, ui=ui)
        assert state._completing is True

    def test_slash_mode_cycle_selected_value_is_a_command(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.cycle("/c", direction=1, ui=ui)
        selected_value = ui.set_input_value.call_args[0][0]
        assert selected_value.startswith("/")
        assert selected_value in SLASH_COMMANDS

    def test_slash_mode_cycle_forward_advances_index(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.cycle("/c", direction=1, ui=ui)
        assert state._idx == 0

    def test_slash_mode_cycle_forward_wraps_around(self) -> None:
        # Starting from idx=-1, after n+1 forward cycles we return to idx=0.
        # (First cycle lands at 0, then n more cycles complete a full lap.)
        state, ui = _state_with_slash_matches("/c")
        n = len(state._matches)
        for _ in range(n + 1):
            state.cycle("/c", direction=1, ui=ui)
        assert state._idx == 0

    def test_slash_mode_cycle_backward_from_start_goes_to_last(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.cycle("/c", direction=-1, ui=ui)
        assert state._idx == len(state._matches) - 1

    def test_slash_mode_show_completion_called_with_updated_index(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.cycle("/c", direction=1, ui=ui)
        ui.show_completion.assert_called_once()
        _, idx_arg = ui.show_completion.call_args[0]
        assert idx_arg == state._idx

    def test_file_mode_cycle_replaces_at_fragment_in_value(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        # Inject file mode state directly
        state._matches = ["data.csv", "report.csv"]
        state._displays = state._matches
        state._idx = -1
        state._at_pos = 0  # "@" is at position 0
        state._mode = "file"
        state.cycle("@d", direction=1, ui=ui)
        new_value = ui.set_input_value.call_args[0][0]
        # The @ is preserved and the fragment is replaced with the match
        assert "@data.csv" in new_value

    def test_file_mode_cycle_preserves_text_after_space(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        state._matches = ["data.csv"]
        state._displays = state._matches
        state._idx = -1
        state._at_pos = 0
        state._mode = "file"
        # Input: "@d rest" — rest comes after the @ reference
        state.cycle("@d rest", direction=1, ui=ui)
        new_value = ui.set_input_value.call_args[0][0]
        assert "rest" in new_value


# ---------------------------------------------------------------------------
# hide — popup teardown
# ---------------------------------------------------------------------------


class TestCompletionStateHide:
    def test_hide_calls_hide_completion_when_popup_was_showing(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.hide(ui)
        ui.hide_completion.assert_called_once()

    def test_hide_is_noop_for_ui_when_popup_was_not_showing(self) -> None:
        state = CompletionState()
        ui = _make_ui()
        state.hide(ui)
        ui.hide_completion.assert_not_called()

    def test_hide_clears_matches_list(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.hide(ui)
        assert state._matches == []

    def test_hide_clears_displays_list(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.hide(ui)
        assert state._displays == []

    def test_hide_resets_index_to_minus_one(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state._idx = 2
        state.hide(ui)
        assert state._idx == -1

    def test_hide_resets_mode_to_file(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.hide(ui)
        assert state._mode == "file"

    def test_hide_clears_at_cache(self) -> None:
        state = CompletionState()
        state._last_at_fragment = "data"
        state._cached_at_matches = ["data.csv"]
        ui = _make_ui()
        # Inject one match to ensure hide_completion is triggered
        state._matches = ["data.csv"]
        state.hide(ui)
        assert state._last_at_fragment is None
        assert state._cached_at_matches == []

    def test_hide_resets_has_matches_to_false(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        state.hide(ui)
        assert state.has_matches is False

    def test_hide_handles_ui_exception_without_propagating(self) -> None:
        state, ui = _state_with_slash_matches("/c")
        ui.hide_completion = MagicMock(side_effect=RuntimeError("widget gone"))
        state.hide(ui)  # must not raise — exception is caught and logged
