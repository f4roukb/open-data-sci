"""SystemDependenciesScreen -- offers to install missing sandbox OS packages.

Shown once, before the startup wizard, only when
``get_system_dependency_status()`` reports something missing on a supported
platform. Declining, failing, or having no known auto-install path all fall
through to the same manual-instructions view rather than blocking boot --
the sandbox is only needed once the agent actually executes code, so this
step must never be able to strand the user without a way to continue.
"""

import logging
import subprocess
from typing import Callable

from rich.markup import escape
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from opendatasci._tui.screens._wizard_layout import WIZARD_BOX_WIDTH
from opendatasci._tui.style.theme import active as theme
from opendatasci.sandbox.srt import (
    SystemDependencyStatus,
    build_auto_install_command,
    get_system_dependency_status,
)

logger = logging.getLogger(__name__)

_INSTALL_NOW = "install-now"
_SKIP = "skip"
_CONTINUE = "continue"


class SystemDependenciesScreen(ModalScreen[None]):
    """One-shot prompt to install missing sandbox dependencies (ripgrep, bubblewrap, socat)."""

    DEFAULT_CSS = f"""
    SystemDependenciesScreen {{
        align: center middle;
    }}
    SystemDependenciesScreen > Vertical {{
        width: {WIZARD_BOX_WIDTH};
        height: auto;
        border: round $ods-accent;
        background: $ods-surface;
    }}
    SystemDependenciesScreen #sysdeps-title {{
        background: $ods-surface-alt;
        border-bottom: solid $ods-separator;
        padding: 1 3;
    }}
    SystemDependenciesScreen #sysdeps-body {{
        padding: 1 3;
    }}
    SystemDependenciesScreen #sysdeps-body Static {{
        margin-bottom: 1;
    }}
    SystemDependenciesScreen OptionList {{
        height: auto;
        max-height: 6;
        margin-top: 1;
    }}
    """

    def __init__(
        self,
        status: SystemDependencyStatus,
        on_complete: Callable[[], None],
    ) -> None:
        super().__init__()
        self._status = status
        self._on_complete = on_complete
        self._install_argv = build_auto_install_command(status.platform)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="sysdeps-title")
            yield Vertical(id="sysdeps-body")

    def on_mount(self) -> None:
        self.query_one("#sysdeps-title", Static).update(
            f"[bold {theme['accent']}]System dependencies[/bold {theme['accent']}]"
        )
        self._render_confirm()

    def _render_confirm(self) -> None:
        body = self.query_one("#sysdeps-body", Vertical)
        body.remove_children()
        is_or_are = "isn't" if "," not in self._status.description else "aren't"
        lines = [
            f"[{theme['text_primary']}]OpenDataSci's sandboxed code execution needs "
            f"{escape(self._status.description)}, which {is_or_are} "
            f"installed.[/{theme['text_primary']}]",
        ]
        if self._install_argv is not None:
            command_str = " ".join(self._install_argv)
            lines.append(
                f"[{theme['text_muted']}]OpenDataSci can install {escape(command_str)} for you now"
                f"{' (you may be asked for your password)' if 'sudo' in self._install_argv else ''}."
                f"[/{theme['text_muted']}]"
            )
        body.mount(Static("\n".join(lines)))

        options = []
        if self._install_argv is not None:
            options.append(Option("Install now", id=_INSTALL_NOW))
        options.append(Option("Skip -- I'll install it myself", id=_SKIP))
        option_list = OptionList(*options)
        body.mount(option_list)
        option_list.focus()

    def _render_manual_instructions(self, *, install_failed: bool) -> None:
        body = self.query_one("#sysdeps-body", Vertical)
        body.remove_children()
        lines = []
        if install_failed:
            lines.append(
                f"[{theme['error']}]The automatic install didn't complete.[/{theme['error']}]"
            )
        lines.append(
            f"[{theme['text_primary']}]Run this in a terminal, then restart "
            f"OpenDataSci:[/{theme['text_primary']}]"
        )
        hint = self._status.manual_install_hint or "See your OS package manager's documentation."
        lines.append(f"[{theme['accent']}]{escape(hint)}[/{theme['accent']}]")
        body.mount(Static("\n\n".join(lines)))

        option_list = OptionList(Option("Continue", id=_CONTINUE))
        body.mount(option_list)
        option_list.focus()

    def _render_installing(self) -> None:
        body = self.query_one("#sysdeps-body", Vertical)
        body.remove_children()
        body.mount(
            Static(f"[{theme['text_muted']}]Installing in your terminal...[/{theme['text_muted']}]")
        )

    @on(OptionList.OptionSelected)
    def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        choice = event.option.id
        if choice == _INSTALL_NOW:
            self._render_installing()
            self._run_install()
        elif choice == _SKIP:
            self._render_manual_instructions(install_failed=False)
        elif choice == _CONTINUE:
            self.dismiss()
            self._on_complete()

    @work
    async def _run_install(self) -> None:
        assert self._install_argv is not None
        try:
            with self.app.suspend():
                subprocess.run(self._install_argv, check=True)
            succeeded = get_system_dependency_status().satisfied
        except Exception:
            logger.exception("Automatic system dependency install failed")
            succeeded = False

        if succeeded:
            self.dismiss()
            self._on_complete()
        else:
            self._render_manual_instructions(install_failed=True)
