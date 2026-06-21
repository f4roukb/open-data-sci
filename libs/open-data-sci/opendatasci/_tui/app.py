import argparse
import importlib.metadata
import logging
import uuid
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.timer import Timer
from textual.widgets import Footer, Input

from opendatasci.configs import DEFAULT_MODEL, DEFAULT_SECONDARY_MODEL, OpenDataSciConfig
from opendatasci.models.providers import Provider

from . import theme as _theme
from .controller import CLIController, UIAdapter
from .widgets import (
    AppHeader,
    ChatPane,
    CompletionPopup,
    MessageBubble,
    SmartInput,
    ThinkingBlock,
    ToolCallBlock,
    TurnStatusBar,
)


def _print_providers() -> None:
    table = Table(title=None, show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Default model")
    for provider, model in DEFAULT_MODEL.items():
        table.add_row(provider, model)
    Console().print(table)


def _get_version() -> str:
    try:
        return importlib.metadata.version("open-data-sci")
    except importlib.metadata.PackageNotFoundError:
        logging.getLogger(__name__).warning(
            "open-data-sci package not found; falling back to hardcoded version '0.1.0'"
        )
        return "0.1.0"


class OpenDataSciApp(App[None]):
    """OpenDataSci — full TUI for AI-powered data science."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        Binding("ctrl+c", "request_quit", "Quit"),
        Binding("ctrl+d", "quit", "Quit", show=False),
        Binding("ctrl+r", "reset", "Reset"),
        Binding("ctrl+l", "clear_conv", "Clear", show=False),
        Binding("escape", "focus_input", "Focus", show=False),
    ]

    def __init__(
        self,
        workspace_path: str,
        session_id: str,
        datasci_config: OpenDataSciConfig,
        theme: str = "default",
    ) -> None:
        palette = _theme.THEMES.get(theme, _theme.DARK)
        _theme.active.update(palette)
        _theme.active_name = theme if theme in _theme.THEMES else "default"
        if theme == "accessible":
            self.CSS_PATH = str(Path(__file__).parent / "styles_visible.tcss")  # type: ignore[misc]
        super().__init__()
        self._controller = CLIController(
            ui=self,  # type: ignore[arg-type]
            workspace_path=workspace_path,
            datasci_config=datasci_config,
            session_id=session_id,
        )

    def compose(self) -> ComposeResult:
        yield AppHeader(
            version=_get_version(),
            provider=self._controller.provider,
            model=self._controller.model,
            workspace=str(Path(self._controller._workspace_path).resolve()),
        )
        with Horizontal(id="main"):
            yield ChatPane()
        yield Footer()

    def on_mount(self) -> None:
        self._quit_requested = False
        self._quit_timer: Timer | None = None
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

    def add_ephemeral_block(self, communication: str, label: str, summary: str) -> ToolCallBlock:
        return self.query_one(ChatPane).add_ephemeral_block(communication, label, summary)

    def add_worker_block(self, communication: str, worker_summaries: list[str]) -> ToolCallBlock:
        return self.query_one(ChatPane).add_worker_block(communication, worker_summaries)

    def add_thinking_block(self) -> ThinkingBlock:
        return self.query_one(ChatPane).add_thinking_block()

    def clear_messages(self) -> None:
        self.query_one(ChatPane).clear_messages()

    def set_workspace(self, name: str) -> None:
        self.query_one(AppHeader).set_workspace(name)

    def set_file_count(self, description: str) -> None:
        self.query_one(AppHeader).set_file_count(description)

    def show_workspace_panel(self, files: list[str]) -> None:
        self.query_one(ChatPane).show_workspace_panel(files)

    def show_attachment(self, label: str) -> None:
        self.query_one(ChatPane).show_attachment(label)

    def hide_attachment(self) -> None:
        self.query_one(ChatPane).hide_attachment()

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
        raw = event.value.strip()
        if raw:
            self.query_one("#user-input", SmartInput).push_history(raw)
        self.query_one("#user-input", Input).value = ""
        action, query = await self._controller.on_submit(raw)
        if action == "run":
            self._run_agent(query)
        elif action == "quit":
            self.exit()

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
            self._controller._completing = True  # suppress Input.Changed fired by value update
            if self.query_one("#user-input", SmartInput).navigate_history(direction):
                event.stop()
                event.prevent_default()
            else:
                self._controller._completing = False

    # ── @work wrappers ────────────────────────────────────────────────────────

    @work
    async def _boot(self) -> None:
        await self._controller.boot()

    @work(exclusive=True, group="agent", exit_on_error=False)
    async def _run_agent(self, query: str) -> None:
        await self._controller.run_agent(query)

    @work
    async def _compact(self) -> None:
        await self._controller.compact()

    # ── Action handlers ───────────────────────────────────────────────────────

    async def action_quit(self) -> None:
        self.exit()

    def action_request_quit(self) -> None:
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

    def action_focus_input(self) -> None:
        self._controller.hide_completion()
        self._controller.clear_paste_attachment()
        if self._controller.awaiting_choice:
            resume_input = self._controller.cancel_choice()
            if resume_input is not None:
                self._run_agent(resume_input)
        self.query_one("#user-input", Input).focus()

    @on(SmartInput.TabComplete)
    def on_smart_input_tab_complete(self, event: SmartInput.TabComplete) -> None:
        inp = self.query_one("#user-input", Input)
        if not self._controller.cycle_completion(inp.value, direction=event._direction):
            self.action_focus_next()


# Maps a provider name to the OpenDataSciConfig field that holds its API key.
# Providers that use cloud-native auth (bedrock, vertexai, ollama) have no key field.
_PROVIDER_KEY_FIELD: dict[Provider, str | None] = {
    Provider.ANTHROPIC: "anthropic_api_key",
    Provider.OPENAI: "openai_api_key",
    Provider.GEMINI: "google_api_key",
    Provider.AZURE: "azure_api_key",
    Provider.OPENAI_COMPATIBLE_SERVER: "openai_api_key",
    Provider.BEDROCK: None,
    Provider.VERTEXAI: None,
    Provider.OLLAMA: None,
}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="OpenDataSci — AI-powered data analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  opendatasci data.xlsx
  opendatasci data.csv --provider bedrock
  opendatasci ./data_folder --provider openai --model gpt-4o
  opendatasci data.csv --secondary-provider openai --secondary-model gpt-4o-mini
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
        "--provider",
        default=None,
        choices=list(Provider),
        help="LLM provider for the primary model (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=None,
        help="Primary model name (provider-specific)",
    )
    parser.add_argument(
        "--secondary-provider",
        dest="secondary_provider",
        default=None,
        choices=list(Provider),
        help="LLM provider for the secondary (auxiliary) model — may differ from --provider",
    )
    parser.add_argument(
        "--secondary-model",
        dest="secondary_model",
        default=None,
        help="Secondary model name (resolved against --secondary-provider or --provider)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key for the primary provider (or set via environment variable)",
    )
    parser.add_argument(
        "--theme",
        choices=list(_theme.THEMES.keys()),
        default="default",
        help=(
            "Colour palette. Choices: "
            + ", ".join(_theme.THEMES.keys())
            + ". Run `/themes` inside the TUI for descriptions."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help=(
            "Path to a YAML file containing OpenDataSciConfig fields. "
            "Explicit TUI flags take precedence over values in the file."
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

    if args.workspace_or_file is None:
        parser.error("the following arguments are required: path")

    # Build OpenDataSciConfig: YAML file provides the base; explicit TUI flags override.
    if args.config:
        datasci_config = OpenDataSciConfig.from_yaml(args.config)
        overrides: dict[str, object] = {}
        if args.provider is not None:
            overrides["provider"] = args.provider
        if args.model is not None:
            overrides["model"] = args.model
        if args.secondary_provider is not None:
            overrides["secondary_provider"] = args.secondary_provider
        if args.secondary_model is not None:
            overrides["secondary_model"] = args.secondary_model
        if args.api_key is not None:
            effective_provider = str(args.provider or datasci_config.provider)
            key_field = _PROVIDER_KEY_FIELD.get(Provider(effective_provider))
            if key_field:
                overrides[key_field] = args.api_key
            else:
                parser.error(
                    f"--api-key is not supported for provider '{effective_provider}' "
                    f"(uses cloud-native authentication)"
                )
        if overrides:
            datasci_config = datasci_config.model_copy(update=overrides)
    else:
        provider: Provider = args.provider or Provider.ANTHROPIC
        resolved_secondary_provider: Provider = args.secondary_provider or provider
        kwargs: dict[str, object] = {
            "provider": provider,
            "model": args.model or DEFAULT_MODEL[provider],
            "secondary_provider": resolved_secondary_provider,
            "secondary_model": args.secondary_model
            or DEFAULT_SECONDARY_MODEL[resolved_secondary_provider],
        }
        if args.api_key is not None:
            key_field = _PROVIDER_KEY_FIELD.get(provider)
            if key_field:
                kwargs[key_field] = args.api_key
            else:
                parser.error(
                    f"--api-key is not supported for provider '{provider}' "
                    f"(uses cloud-native authentication)"
                )
        datasci_config = OpenDataSciConfig(**kwargs)  # type: ignore[arg-type]

    session_id = uuid.uuid4().hex

    OpenDataSciApp(
        workspace_path=args.workspace_or_file,
        session_id=session_id,
        datasci_config=datasci_config,
        theme=args.theme,
    ).run()


# Register OpenDataSciApp as a virtual subclass of UIAdapter to avoid the metaclass
# conflict between Textual's _MessagePumpMeta and ABCMeta.
UIAdapter.register(OpenDataSciApp)


if __name__ == "__main__":
    main()
