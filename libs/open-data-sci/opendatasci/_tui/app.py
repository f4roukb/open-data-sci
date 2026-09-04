import argparse
import importlib.metadata
import logging
import os
import uuid
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import Footer, Input

from opendatasci._tui.adapter import SubmitAction
from opendatasci._tui.chat.widgets import (
    AppHeader,
    ChatPane,
    CommandApprovalPrompt,
    CompletionPopup,
    MessageBubble,
    PendingMessageBubble,
    SmartInput,
    ThinkingBlock,
    TipsBar,
    ToolCallBlock,
    TurnStatusBar,
)
from opendatasci._tui.config.config_tree import (
    ConfigLeaf,
    ConfigNode,
    build_model_leaf,
    build_provider_leaf,
    build_theme_leaf,
)
from opendatasci._tui.config.config_tree import (
    initial_values as build_initial_values,
)
from opendatasci._tui.config.global_config import load_global_config
from opendatasci._tui.config.onboarding import (
    compute_missing_fields,
    compute_missing_selection_fields,
)
from opendatasci._tui.controller import CLIController
from opendatasci._tui.screens.config_screen import ConfigScreen
from opendatasci._tui.screens.onboarding_screen import OnboardingScreen
from opendatasci._tui.screens.startup_wizard_screen import StartupWizardScreen
from opendatasci._tui.style import theme as _theme
from opendatasci.configs import DEFAULT_MODEL, OpenDataSciConfig
from opendatasci.tools.mcp import MCPServerSpec

logger = logging.getLogger(__name__)


def _print_providers() -> None:
    table = Table(title=None, show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Default model")
    for provider, model in DEFAULT_MODEL.items():
        table.add_row(provider, model)
    Console().print(table)


def _apply_global_config_fallback(kwargs: dict[str, object], global_cfg: dict[str, object]) -> None:
    """Fill *kwargs* in place from *global_cfg* for fields not set via CLI or env.

    A real environment variable for the field always wins over the persisted
    global config value.
    """
    for field_name, value in global_cfg.items():
        if field_name in kwargs:
            continue
        model_field = OpenDataSciConfig.model_fields.get(field_name)
        if model_field is None:
            continue
        if model_field.alias and os.environ.get(model_field.alias):
            continue
        kwargs[field_name] = value


def _get_version() -> str:
    try:
        return importlib.metadata.version("open-data-sci")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("open-data-sci package not found; falling back to hardcoded version '0.2.0'")
        return "0.2.0"


class OpenDataSciApp(App[None]):
    """OpenDataSci — full TUI for AI-powered data science."""

    CSS_PATH = "style/styles.tcss"

    BINDINGS = [
        Binding("ctrl+c", "request_quit", "Stop/Quit"),
        Binding("ctrl+r", "reset", "Reset"),
        Binding("ctrl+l", "clear_conv", "Clear", show=False),
        Binding("escape", "focus_input", "Focus", show=False),
    ]

    def __init__(
        self,
        workspace_path: str,
        session_id: str,
        datasci_config: OpenDataSciConfig,
        missing_selection: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._initial_datasci_config = datasci_config
        self._controller = CLIController(
            ui=self,  # type: ignore[arg-type]
            workspace_path=workspace_path,
            datasci_config=datasci_config,
            session_id=session_id,
        )
        self._missing_selection = missing_selection or []

    def get_css_variables(self) -> dict[str, str]:
        """Expose the active theme palette as $ods-* CSS variables.

        styles.tcss and widget DEFAULT_CSS reference these instead of color
        literals, so every registered theme restyles the whole app.
        """
        variables = super().get_css_variables()
        variables.update(
            {f"ods-{key.replace('_', '-')}": value for key, value in _theme.active.items()}
        )
        return variables

    def compose(self) -> ComposeResult:
        yield AppHeader(
            version=_get_version(),
            workspace=str(Path(self._controller._workspace_path).resolve()),
        )
        with Horizontal(id="main"):
            yield ChatPane()
        yield Footer()

    def on_mount(self) -> None:
        self._quit_requested = False
        self._quit_timer: Timer | None = None
        self.query_one("#user-input", Input).focus()
        steps = self._build_wizard_steps()
        values = build_initial_values(self._initial_datasci_config, _theme.active_name)
        self.push_screen(StartupWizardScreen(steps, values, self._on_wizard_complete))

    def _build_wizard_steps(self) -> list[tuple[str, ConfigLeaf]]:
        """Always-shown theme step, plus whichever provider/model fields weren't
        already resolved via --config/env (see ``compute_missing_selection_fields``)."""
        steps: list[tuple[str, ConfigLeaf]] = [("Theme", build_theme_leaf())]
        if "provider" in self._missing_selection:
            steps.append(("Provider", build_provider_leaf("provider", "model")))
        if "model" in self._missing_selection:
            steps.append(("Model", build_model_leaf("model", "provider")))
        if "secondary_provider" in self._missing_selection:
            steps.append(
                ("Secondary provider", build_provider_leaf("secondary_provider", "secondary_model"))
            )
        if "secondary_model" in self._missing_selection:
            steps.append(
                ("Secondary model", build_model_leaf("secondary_model", "secondary_provider"))
            )
        return steps

    def _on_wizard_complete(self, values: dict[str, str]) -> None:
        _theme.set_active(values["theme"])
        self.refresh_css()
        config_updates = {k: v for k, v in values.items() if k != "theme"}
        self._controller.apply_config_updates(config_updates)
        self.query_one("#user-input", Input).focus()

        base_config = self._controller.base_config
        global_cfg = load_global_config()
        missing_fields = compute_missing_fields(
            [base_config.provider, base_config.secondary_provider], {}, global_cfg
        )
        if missing_fields:
            self.push_screen(OnboardingScreen(missing_fields, self._on_onboarding_complete))
        else:
            self._boot()

    def _on_onboarding_complete(self, values: dict[str, str]) -> None:
        self._controller.apply_config_updates(values)
        self.query_one("#user-input", Input).focus()
        self._boot()

    async def on_unmount(self) -> None:
        await self._controller.close()

    # ── UIAdapter implementation ──────────────────────────────────────────────

    def add_message(self, role: str, content: str = "") -> MessageBubble:
        return self.query_one(ChatPane).add_message(role, content)

    def add_divider(self) -> None:
        self.query_one(ChatPane).add_divider()

    def add_turn_status_bar(self) -> TurnStatusBar:
        return self.query_one(ChatPane).add_turn_status_bar()

    def add_pending_message(self, text: str) -> PendingMessageBubble:
        return self.query_one(ChatPane).add_pending_message(text)

    def add_ephemeral_block(self, communication: str, label: str, summary: str) -> ToolCallBlock:
        return self.query_one(ChatPane).add_ephemeral_block(communication, label, summary)

    def add_task_block(self, communication: str, task_summaries: list[str]) -> ToolCallBlock:
        return self.query_one(ChatPane).add_task_block(communication, task_summaries)

    def add_thinking_block(self) -> ThinkingBlock:
        return self.query_one(ChatPane).add_thinking_block()

    def clear_messages(self) -> None:
        self.query_one(ChatPane).clear_messages()

    def set_workspace(self, name: str) -> None:
        self.query_one(AppHeader).set_workspace(name)

    def set_file_count(self, description: str) -> None:
        self.query_one(AppHeader).set_file_count(description)

    def set_background_tasks(self, description: str) -> None:
        self.query_one(AppHeader).set_background_tasks(description)

    def set_model_info(self, description: str) -> None:
        self.query_one(AppHeader).set_model_info(description)

    def show_workspace_panel(self, files: list[str]) -> None:
        self.query_one(ChatPane).show_workspace_panel(files)

    def show_approval_prompt(self, description: str, heads_up: str) -> None:
        self.query_one(ChatPane).show_approval_prompt(description, heads_up)

    def show_attachment(self, label: str) -> None:
        self.query_one(ChatPane).show_attachment(label)

    def hide_attachment(self) -> None:
        self.query_one(ChatPane).hide_attachment()

    def open_config_panel(
        self,
        root: ConfigNode,
        initial_values: dict[str, str],
        start_path: list[str],
        on_apply: Callable[[dict[str, str], list[MCPServerSpec] | None], Awaitable[str | None]],
        initial_mcp_servers: list[MCPServerSpec] | None = None,
    ) -> None:
        self.push_screen(
            ConfigScreen(root, initial_values, start_path, on_apply, initial_mcp_servers)
        )

    def refresh_theme(self) -> None:
        """Recompute $ods-* CSS variables from the (just-switched) active palette."""
        self.refresh_css()

    def refresh_tips(self) -> None:
        """Re-render the footer tips bar after tips.set_enabled() flips it."""
        self.query_one(TipsBar).apply_settings()

    def stop_agent(self) -> None:
        self.workers.cancel_group(self, "agent")

    def set_input_placeholder(self, text: str) -> None:
        self.query_one("#user-input", Input).placeholder = text

    def add_input_class(self, cls: str) -> None:
        self.query_one("#user-input", Input).add_class(cls)

    def remove_input_class(self, cls: str) -> None:
        self.query_one("#user-input", Input).remove_class(cls)

    def set_input_value(self, value: str, cursor: int | None = None) -> None:
        inp = self.query_one("#user-input", Input)
        inp.value = value
        if cursor is not None:
            inp.cursor_position = cursor

    def show_completion(self, matches: list[str], selected: int) -> None:
        self.query_one(CompletionPopup).show_matches(matches, selected)

    def hide_completion(self) -> None:
        self.query_one(CompletionPopup).hide()

    # ── Event handlers ────────────────────────────────────────────────────────

    @on(SmartInput.Pasted)
    def on_paste_attachment(self, event: SmartInput.Pasted) -> None:
        self._controller.on_paste(event._text)

    @on(Input.Changed, "#user-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self._controller.on_input_changed(event.value)

    @on(Input.Submitted, "#user-input")
    async def on_submit(self, event: Input.Submitted) -> None:
        if self._controller.accept_completion():
            # A completion popup was active: Enter confirms the selection
            # (already written into the input text) instead of sending it.
            return
        raw = event.value.strip()
        if raw:
            self.query_one("#user-input", SmartInput).push_history(raw)
        self.query_one("#user-input", Input).value = ""
        action, query = await self._controller.on_submit(raw)
        if action is SubmitAction.RUN:
            self._run_agent(query)
        elif action is SubmitAction.RESUME_INPUT:
            self._resume_with_input(query)
        elif action is SubmitAction.QUIT:
            self.exit()

    @on(CommandApprovalPrompt.Decision)
    async def on_approval_decision(self, event: CommandApprovalPrompt.Decision) -> None:
        self.query_one("#user-input", Input).focus()
        self._resume_with_approval(event.approved)

    @on(events.Key)
    def on_input_key(self, event: events.Key) -> None:
        if event.key not in {"up", "down"}:
            return
        inp = self.query_one("#user-input", Input)
        if self.focused is not inp:
            return
        direction = 1 if event.key == "down" else -1
        if self._controller.has_completion_matches:
            if self._controller.cycle_completion(inp.value, direction=direction):
                event.stop()
                event.prevent_default()
        else:
            self._controller.suppress_next_input_change()  # value update fires Input.Changed
            if self.query_one("#user-input", SmartInput).navigate_history(direction):
                event.stop()
                event.prevent_default()
            else:
                self._controller.cancel_input_change_suppression()

    # ── @work wrappers ────────────────────────────────────────────────────────

    @work
    async def _boot(self) -> None:
        await self._controller.boot()

    @work(exclusive=True, group="agent", exit_on_error=False)
    async def _run_agent(self, query: str) -> None:
        await self._controller.run_agent(query)

    @work(exclusive=True, group="agent", exit_on_error=False)
    async def _resume_with_input(self, answer: str) -> None:
        await self._controller.resume_with_input(answer)

    @work(exclusive=True, group="agent", exit_on_error=False)
    async def _resume_with_approval(self, approved: bool) -> None:
        await self._controller.resume_with_approval(approved)

    @work
    async def _compact(self) -> None:
        await self._controller.compact()

    # ── Action handlers ───────────────────────────────────────────────────────

    async def action_quit(self) -> None:
        self.exit()

    async def action_request_quit(self) -> None:
        if self._controller.agent_running:
            # While a turn is running, Ctrl+C means "stop the agent" (same as
            # /stop). Quitting remains a deliberate double-press while idle.
            await self._controller.stop_agent()
            return
        if self._quit_requested:
            self.exit()
            return
        self._quit_requested = True
        self.notify("Press Ctrl+C again to quit", severity="warning", timeout=3)
        if self._quit_timer is not None:
            self._quit_timer.stop()
        self._quit_timer = self.set_timer(3, self._reset_quit_request)

    def _reset_quit_request(self) -> None:
        self._quit_requested = False
        self._quit_timer = None

    async def action_reset(self) -> None:
        await self._controller.reset()

    async def action_clear_conv(self) -> None:
        await self._controller.clear_conv()

    def action_compact(self) -> None:
        self._compact()

    async def action_focus_input(self) -> None:
        had_completion = self._controller.has_completion_matches
        had_paste = self._controller.has_paste_attachment
        self._controller.hide_completion()
        self._controller.clear_paste_attachment()
        if self._controller.awaiting_choice:
            resume_input = await self._controller.cancel_choice()
            if resume_input is not None:
                self._resume_with_input(resume_input)
        elif not had_completion and not had_paste and self._controller.agent_running:
            # A bare Esc during a turn stops the agent, mirroring Ctrl+C.
            await self._controller.stop_agent()
        self.query_one("#user-input", Input).focus()

    @on(SmartInput.TabComplete)
    def on_smart_input_tab_complete(self, event: SmartInput.TabComplete) -> None:
        inp = self.query_one("#user-input", Input)
        if not self._controller.cycle_completion(inp.value, direction=event._direction):
            self.action_focus_next()


def _load_yaml_dict(path: str) -> dict[str, object]:
    """Raw contents of a --config YAML file, used only to detect which fields
    it sets explicitly (OpenDataSciConfig.from_yaml doesn't expose that)."""
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="OpenDataSci — AI-powered data analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  opendatasci data.xlsx
  opendatasci ./data_folder
  opendatasci data.csv --config path/to/datasci_config.yaml
        """,
    )
    parser.add_argument(
        "workspace_or_file",
        nargs="?",
        default=None,
        help="Data file or directory containing data files to work with",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help=(
            "Path to a YAML file containing OpenDataSciConfig fields. Provider/model "
            "fields it sets are used as-is; anything it doesn't set (including theme, "
            "which it never sets) is picked interactively on startup."
        ),
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List supported providers and their default models, then exit",
    )
    parser.add_argument("--version", action="version", version=f"OpenDataSci {_get_version()}")
    args = parser.parse_args()

    if args.list_providers:
        _print_providers()
        return

    workspace_or_file = args.workspace_or_file or str(Path.cwd())
    global_cfg = load_global_config()

    if args.config:
        datasci_config = OpenDataSciConfig.from_yaml(args.config)
        yaml_data = _load_yaml_dict(args.config)
    else:
        datasci_config = OpenDataSciConfig()
        yaml_data = {}

    overrides: dict[str, object] = {}
    _apply_global_config_fallback(overrides, global_cfg)
    if overrides:
        datasci_config = datasci_config.model_copy(update=overrides)

    missing_selection = compute_missing_selection_fields(yaml_data)
    session_id = uuid.uuid4().hex

    OpenDataSciApp(
        workspace_path=workspace_or_file,
        session_id=session_id,
        datasci_config=datasci_config,
        missing_selection=missing_selection,
    ).run()


if __name__ == "__main__":
    main()
