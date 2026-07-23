"""Tab-completion state machine for the OpenDataSci TUI input bar.

Manages slash-command and @file-path completion independently of Textual
widgets, so it can be unit-tested without a running app.
"""

import logging
from typing import Callable

from opendatasci.configs import DEFAULT_MODEL, DEFAULT_SECONDARY_MODEL, OpenDataSciConfig
from opendatasci.models.providers import Provider

from . import theme as _theme
from .adapter import UIAdapter
from .commands import SLASH_COMMAND_DESCRIPTIONS, SLASH_COMMANDS
from .file_refs import _discover_files, _find_at_fragment, _find_slash_fragment

logger = logging.getLogger(__name__)

# Slash commands whose first argument can be tab-completed. Each maps to a
# candidate-generating branch in CompletionState._arg_candidates().
_ARG_COMPLETABLE_COMMANDS = frozenset(
    {"/theme", "/model", "/provider", "/secondary-model", "/secondary-provider"}
)


def _find_command_arg_fragment(text: str) -> tuple[str, str] | None:
    """Return (command, arg_fragment) while typing the first argument of a
    known slash command (e.g. "/theme dra" -> ("/theme", "dra")), or None.
    """
    if not text.startswith("/"):
        return None
    parts = text.split(" ")
    if len(parts) != 2:
        return None
    command, fragment = parts
    if command not in _ARG_COMPLETABLE_COMMANDS:
        return None
    return command, fragment


class CompletionState:
    """Encapsulates all mutable state for the input-bar completion popup.

    Owned by ``CLIController``; delegates UI updates through ``UIAdapter``
    so the logic remains fully testable without a Textual widget tree.
    """

    def __init__(
        self,
        extra_commands: list[str] | None = None,
        config_provider: Callable[[], OpenDataSciConfig | None] | None = None,
    ) -> None:
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
        # "<command> " prefix preserved by cycle() when completing an argument.
        self._arg_prefix: str = ""
        # Returns the config currently in effect (or None pre-boot), used to
        # scope /model and /secondary-model completions to the active provider.
        self._config_provider = config_provider

    @property
    def has_matches(self) -> bool:
        """True when the completion popup currently has items to navigate."""
        return bool(self._matches)

    @property
    def is_suppressing_input_change(self) -> bool:
        """True when the next ``on_input_changed`` call will be swallowed."""
        return self._completing

    def suppress_next_input_change(self) -> None:
        """Mark the next input-change as programmatic, so it gets ignored.

        Call this immediately before code sets the input value directly
        (tab-completion, history navigation), so the resulting change event
        isn't misread as the user typing a new completion trigger.
        """
        self._completing = True

    def cancel_input_change_suppression(self) -> None:
        """Undo ``suppress_next_input_change`` when the anticipated value update didn't happen."""
        self._completing = False

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

        arg_frag = _find_command_arg_fragment(value)
        if arg_frag is not None:
            command, fragment = arg_frag
            matches = [c for c in self._arg_candidates(command) if c.startswith(fragment)]
            if matches:
                self._matches = matches
                self._displays = matches
                self._idx = -1
                self._at_pos = -1
                self._mode = "arg"
                self._arg_prefix = f"{command} "
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

    def _arg_candidates(self, command: str) -> list[str]:
        """Candidate values for the first argument of *command*."""
        if command == "/theme":
            return list(_theme.THEMES.keys())
        if command in ("/provider", "/secondary-provider"):
            return [p.value for p in Provider]
        if command == "/model":
            return self._model_candidates(secondary=False)
        if command == "/secondary-model":
            return self._model_candidates(secondary=True)
        return []

    def _model_candidates(self, secondary: bool) -> list[str]:
        """Known model names for the currently active (primary/secondary) provider."""
        if self._config_provider is None:
            return []
        cfg = self._config_provider()
        if cfg is None:
            return []
        provider = cfg.secondary_provider if secondary else cfg.provider
        candidates = {DEFAULT_MODEL.get(provider), DEFAULT_SECONDARY_MODEL.get(provider)}
        return sorted(c for c in candidates if c)

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

        if self._mode == "arg":
            new_value = self._arg_prefix + match
            self._completing = True
            ui.set_input_value(new_value, len(new_value))
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
        self._arg_prefix = ""
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
