"""StartupWizardScreen — the mandatory, once-per-launch selection flow.

Walks a fixed sequence of ``ConfigLeaf`` steps (always theme first, then
whichever of provider/model/secondary_provider/secondary_model weren't
already resolved from ``--config``/env — see
``opendatasci._tui.config.onboarding.compute_missing_selection_fields``), one per
screen, using the same option-list/text-input rendering as ``ConfigScreen``.
No back-navigation and no Escape-to-cancel: this stands in for the CLI flags
that used to make these choices, so it must run to completion before boot.
"""

from typing import Callable

from rich.markup import escape
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from .. import theme as _theme
from ..theme import active as theme
from .config_tree import ConfigLeaf


class StartupWizardScreen(ModalScreen[None]):
    """One-shot linear picker for theme + whichever provider/model fields are unset."""

    DEFAULT_CSS = """
    StartupWizardScreen {
        align: center middle;
    }
    StartupWizardScreen > Vertical {
        width: 64;
        height: auto;
        border: round $ods-accent;
        padding: 1 2;
        background: $ods-surface;
    }
    StartupWizardScreen OptionList {
        height: auto;
        max-height: 14;
        margin-top: 1;
    }
    StartupWizardScreen Input {
        margin-top: 1;
    }
    """

    def __init__(
        self,
        steps: list[tuple[str, ConfigLeaf]],
        staged: dict[str, str],
        on_complete: Callable[[dict[str, str]], None],
    ) -> None:
        super().__init__()
        self._steps = steps
        self._staged = dict(staged)
        self._index = 0
        self._on_complete = on_complete

    @property
    def _current(self) -> tuple[str, ConfigLeaf]:
        return self._steps[self._index]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="wizard-title")
            yield Vertical(id="wizard-body")

    def on_mount(self) -> None:
        self._render_step()

    def _render_step(self) -> None:
        title, leaf = self._current
        text = (
            f"[bold {theme['text_primary']}]{escape(title)}[/bold {theme['text_primary']}]"
            f" [{theme['text_muted']}]({self._index + 1}/{len(self._steps)})[/{theme['text_muted']}]"
        )
        self.query_one("#wizard-title", Static).update(text)

        body = self.query_one("#wizard-body", Vertical)
        body.remove_children()
        choices = leaf.options(self._staged)
        if choices:
            current = self._staged.get(leaf.field)
            options = []
            for choice in choices:
                marker = "●" if choice.value == current else "○"
                label = f"{marker} {choice.label}"
                if choice.description:
                    label += f"  [dim]{choice.description}[/dim]"
                options.append(Option(label, id=choice.value))
            option_list = OptionList(*options)
            body.mount(option_list)
            option_list.focus()
        else:
            text_input = Input(
                value=self._staged.get(leaf.field, ""),
                placeholder=leaf.text_placeholder,
            )
            body.mount(text_input)
            text_input.focus()

    def _advance(self, field: str, value: str) -> None:
        _title, leaf = self._current
        self._staged[field] = value
        if leaf.linked_field and leaf.linked_default is not None:
            self._staged[leaf.linked_field] = leaf.linked_default(value)
        if field == "theme":
            if _theme.set_active(value):
                self.app.refresh_css()
        self._index += 1
        if self._index >= len(self._steps):
            # Dismiss (pop) this screen *before* the completion callback, which
            # may itself push a new screen (OnboardingScreen) — dismiss() pops
            # whatever is currently on top of the stack, not necessarily this
            # screen, so pushing first would pop the wrong one.
            self.dismiss()
            self._on_complete(self._staged)
        else:
            self._render_step()

    @on(OptionList.OptionSelected)
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        _title, leaf = self._current
        assert event.option.id is not None
        self._advance(leaf.field, event.option.id)

    @on(Input.Submitted)
    def _on_text_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        _title, leaf = self._current
        value = event.value.strip()
        if not value:
            return
        self._advance(leaf.field, value)
