"""Modal overlay collecting missing provider config, one field at a time."""

from typing import Callable

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from opendatasci._tui.config.global_config import save_global_config_value
from opendatasci._tui.config.onboarding import RequiredField
from opendatasci._tui.screens._wizard_layout import WIZARD_BOX_WIDTH
from opendatasci._tui.style.theme import active as theme


class OnboardingScreen(ModalScreen[None]):
    """Blocking overlay that prompts for each missing `RequiredField` in turn.

    Collected values are persisted immediately (one write per field) so that
    if the user quits partway through, whatever was already entered is not
    lost on the next launch.
    """

    DEFAULT_CSS = f"""
    OnboardingScreen {{
        align: center middle;
    }}
    OnboardingScreen > Vertical {{
        width: {WIZARD_BOX_WIDTH};
        height: auto;
        border: round $ods-accent;
        padding: 1 2;
        background: $ods-surface;
    }}
    OnboardingScreen #onboarding-help {{
        margin-top: 1;
    }}
    OnboardingScreen Input {{
        margin-top: 1;
    }}
    """

    def __init__(
        self,
        fields: list[RequiredField],
        on_complete: Callable[[dict[str, str]], None],
    ) -> None:
        super().__init__()
        self._fields = fields
        self._index = 0
        self._collected: dict[str, str] = {}
        self._on_complete = on_complete
        self._error: str = ""

    @property
    def _current(self) -> RequiredField:
        return self._fields[self._index]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="onboarding-title")
            yield Static(id="onboarding-help")
            yield Input(id="onboarding-input")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        rf = self._current
        title = Text.from_markup(
            f"[bold {theme['text_primary']}]Setup required[/bold {theme['text_primary']}]"
            f" [{theme['text_muted']}]({self._index + 1}/{len(self._fields)})[/{theme['text_muted']}]"
        )
        self.query_one("#onboarding-title", Static).update(title)

        help_lines = [f"[{theme['text_primary']}]{rf.label}[/{theme['text_primary']}]"]
        if rf.default:
            help_lines.append(
                f"[dim {theme['text_muted']}]Press Enter to accept the default: "
                f"{rf.default}[/dim {theme['text_muted']}]"
            )
        if self._error:
            help_lines.append(f"[{theme['error']}]{self._error}[/{theme['error']}]")
        self.query_one("#onboarding-help", Static).update(Text.from_markup("\n".join(help_lines)))

        inp = self.query_one("#onboarding-input", Input)
        inp.value = ""
        inp.password = rf.secret
        inp.placeholder = rf.default or ""
        inp.focus()

    @on(Input.Submitted, "#onboarding-input")
    def _submit(self, event: Input.Submitted) -> None:
        rf = self._current
        value = event.value.strip() or (rf.default or "")
        if not value:
            self._error = "This value is required."
            self._refresh()
            return
        self._error = ""
        self._collected[rf.field] = value
        save_global_config_value(rf.field, value)
        self._index += 1
        if self._index >= len(self._fields):
            self._on_complete(self._collected)
            self.dismiss()
        else:
            self._refresh()
