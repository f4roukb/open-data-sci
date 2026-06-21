"""Tab-completion state machine for the OpenDataSci TUI input bar.

Manages slash-command and @file-path completion independently of Textual
widgets, so it can be unit-tested without a running app.
"""

import logging

from .adapter import UIAdapter
from .commands import SLASH_COMMAND_DESCRIPTIONS, SLASH_COMMANDS
from .file_refs import _discover_files, _find_at_fragment, _find_slash_fragment

logger = logging.getLogger(__name__)


class CompletionState:
    """Encapsulates all mutable state for the input-bar completion popup.

    Owned by ``CLIController``; delegates UI updates through ``UIAdapter``
    so the logic remains fully testable without a Textual widget tree.
    """

    def __init__(self, extra_commands: list[str] | None = None) -> None:
        self._matches: list[str] = []
        self._displays: list[str] = []
        self._idx: int = -1
        self._at_pos: int = -1
        self._mode: str = "file"
        # Set to True by cycle() before it changes the input value, so that
        # the resulting on_input_changed callback knows to ignore the event.
        self._completing: bool = False
        # Cache the last @-fragment so _discover_files is not called on every
        # keystroke when the fragment hasn't changed.
        self._last_at_fragment: str | None = None
        self._cached_at_matches: list[str] = []
        self._all_commands: list[str] = SLASH_COMMANDS + list(extra_commands or [])

    @property
    def has_matches(self) -> bool:
        """True when the completion popup currently has items to navigate."""
        return bool(self._matches)

    def on_input_changed(self, value: str, ui: UIAdapter) -> bool:
        """Handle an input-text change.

        Returns ``True`` when the change was a programmatic tab-completion
        update (the caller should skip further processing).
        """
        if self._completing:
            self._completing = False
            return True

        slash_frag = _find_slash_fragment(value)
        if slash_frag is not None:
            matches = [cmd for cmd in self._all_commands if cmd.startswith(slash_frag)]
            if matches and not (len(matches) == 1 and slash_frag == matches[0]):
                self._matches = matches
                self._displays = [
                    f"{cmd}  {SLASH_COMMAND_DESCRIPTIONS.get(cmd, '')}" for cmd in matches
                ]
                self._idx = -1
                self._at_pos = -1
                self._mode = "slash"
                ui.show_completion(self._displays, self._idx)
            else:
                self.hide(ui)
            return False

        result = _find_at_fragment(value)
        if result is None:
            self.hide(ui)
            return False

        fragment, at_pos = result
        if fragment != self._last_at_fragment:
            self._cached_at_matches = _discover_files(fragment)
            self._last_at_fragment = fragment
        matches = self._cached_at_matches
        if not matches:
            self.hide(ui)
            return False

        self._matches = matches
        self._idx = -1
        self._at_pos = at_pos
        self._mode = "file"
        ui.show_completion(matches, self._idx)
        return False

    def cycle(self, current_value: str, direction: int, ui: UIAdapter) -> bool:
        """Cycle the completion selection by ``direction`` (+1 down, -1 up).

        Returns ``True`` if a completion was applied (caller should NOT call
        focus_next); ``False`` if there are no completions to cycle.
        """
        if not self._matches:
            return False

        if direction < 0 and self._idx == -1:
            self._idx = len(self._matches) - 1
        else:
            self._idx = (self._idx + direction) % len(self._matches)

        match = self._matches[self._idx]

        if self._mode == "slash":
            self._completing = True
            ui.set_input_value(match, len(match))
            ui.show_completion(self._displays, self._idx)
            return True

        after = current_value[self._at_pos + 1 :]
        space_pos = after.find(" ")
        rest = after[space_pos:] if space_pos != -1 else ""
        new_value = current_value[: self._at_pos + 1] + match + rest
        cursor = self._at_pos + 1 + len(match)
        self._completing = True
        ui.set_input_value(new_value, cursor)
        ui.show_completion(self._matches, self._idx)
        return True

    def hide(self, ui: UIAdapter) -> None:
        """Clear completion state and hide the popup (no-op if already hidden)."""
        was_showing = bool(self._matches)
        self._matches = []
        self._displays = []
        self._idx = -1
        self._at_pos = -1
        self._mode = "file"
        self._last_at_fragment = None
        self._cached_at_matches = []
        # Only update the UI when the popup was actually visible, to avoid
        # spurious re-renders on every normal keystroke.
        if not was_showing:
            return
        try:
            ui.hide_completion()
        except Exception:
            logger.exception("Failed to hide completion popup")
