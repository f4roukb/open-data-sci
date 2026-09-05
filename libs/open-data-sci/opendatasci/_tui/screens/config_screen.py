"""ConfigScreen — the navigable panel behind /config, /models and /providers.

A ``ModalScreen`` pushed on top of the running app, following the same
pattern as ``OnboardingScreen``: the chat screen underneath stays mounted and
keeps receiving controller-driven updates while this is up, so nothing needs
pausing/buffering for background messages to survive the panel being open.

Navigation: Up/Down move the highlight within a level (native to
``OptionList``). Enter, Right and mouse click all select the highlighted row
(Textual's own ``OptionList.action_select`` already posts ``OptionSelected``
for all three). Left moves up one level. Escape leaves the whole panel,
prompting Save/Discard first if anything was changed.
"""

from pathlib import Path
from typing import Awaitable, Callable

from rich.markup import escape
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from opendatasci._tui.config.config_tree import ConfigLeaf, ConfigNode, diff_values
from opendatasci._tui.style.theme import active as theme
from opendatasci.tools.mcp import (
    MCPServerSpec,
    MCPTransport,
    check_mcp_server,
    load_named_mcp_servers,
)

_SAVE = "Save changes"
_DISCARD = "Discard changes"

_MCP_TEXT_MODES = ("mcp_load_path", "mcp_manual_name", "mcp_manual_url", "mcp_manual_header")
# Every mode that belongs to an "add a server" sub-flow (load-from-file or
# manual) — Back/Escape from any of these cancels the whole sub-flow rather
# than stepping back one field at a time.
_MCP_WIZARD_MODES = (*_MCP_TEXT_MODES, "mcp_load_select", "mcp_manual_transport")
_MCP_TRANSPORT_LABELS = {
    MCPTransport.HTTP: "HTTP (Streamable, recommended)",
    MCPTransport.SSE: "SSE (Server-Sent Events, legacy)",
}


def _hint_chip(key: str, label: str) -> str:
    """A small "key + label" hint chip, styled like the footer's key hints."""
    chip_style = f"{theme['text_primary']} on {theme['separator']}"
    return f"[{chip_style}] {key} [/{chip_style}]  [{theme['text_muted']}]{label}[/{theme['text_muted']}]"


def _hint_bar(pairs: list[tuple[str, str]]) -> str:
    return "    ".join(_hint_chip(key, label) for key, label in pairs)


class _NavigateBack(Message):
    pass


class _RequestClose(Message):
    pass


class _ConfigOptionList(OptionList):
    BINDINGS = [
        Binding("left", "go_back", show=False),
        Binding("right", "select", show=False),
        Binding("escape", "request_close", show=False),
    ]

    def action_go_back(self) -> None:
        self.post_message(_NavigateBack())

    def action_request_close(self) -> None:
        self.post_message(_RequestClose())


class _ConfigTextInput(Input):
    BINDINGS = [
        Binding("left", "go_back", show=False),
        Binding("escape", "request_close", show=False),
    ]

    def action_go_back(self) -> None:
        self.post_message(_NavigateBack())

    def action_request_close(self) -> None:
        self.post_message(_RequestClose())


class _ToggleSelectAll(Message):
    pass


class _McpSelectOptionList(_ConfigOptionList):
    BINDINGS = [Binding("ctrl+a", "toggle_all", show=False)]

    def action_toggle_all(self) -> None:
        self.post_message(_ToggleSelectAll())


class ConfigScreen(ModalScreen[None]):
    """Tree-navigable settings panel: Display, Models, Providers."""

    DEFAULT_CSS = """
    ConfigScreen {
        align: center middle;
    }
    ConfigScreen > Vertical {
        width: 78;
        height: auto;
        max-height: 80%;
        border: round $ods-accent;
        background: $ods-surface;
    }
    ConfigScreen #config-breadcrumb {
        background: $ods-surface-alt;
        border-bottom: solid $ods-separator;
        padding: 1 3;
    }
    ConfigScreen #config-body {
        padding: 1 3;
    }
    ConfigScreen OptionList {
        height: auto;
        max-height: 16;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        root: ConfigNode,
        initial_values: dict[str, str],
        start_path: list[str],
        on_apply: Callable[[dict[str, str], list[MCPServerSpec] | None], Awaitable[str | None]],
        initial_mcp_servers: list[MCPServerSpec] | None = None,
    ) -> None:
        super().__init__()
        self._root = root
        self._initial_values = dict(initial_values)
        self._staged = dict(initial_values)
        self._on_apply = on_apply
        self._path: list[ConfigNode] = self._resolve_path(root, start_path)
        self._cursor_by_path: dict[tuple[str, ...], int] = {}
        self._mode: str = "browse"
        self._error: str = ""
        self._status: str = ""
        # MCP servers editor state — kept outside ``_staged`` since it's a
        # list of server specs, not a scalar field.
        self._initial_mcp_servers: list[MCPServerSpec] = list(initial_mcp_servers or [])
        self._mcp_servers: list[MCPServerSpec] = list(initial_mcp_servers or [])
        self._mcp_candidates: list[MCPServerSpec] = []
        self._mcp_selected: set[int] = set()
        self._mcp_select_cursor: int = 0
        self._mcp_checking: bool = False
        # Fields collected across the manual-add wizard (name -> url ->
        # transport -> headers) before the final connectivity check.
        self._mcp_pending_name: str | None = None
        self._mcp_pending_url: str | None = None
        self._mcp_pending_transport: MCPTransport = MCPTransport.HTTP
        self._mcp_pending_headers: dict[str, str] = {}

    @staticmethod
    def _resolve_path(root: ConfigNode, start_path: list[str]) -> list[ConfigNode]:
        path = [root]
        node = root
        for key in start_path:
            match = next((c for c in node.children if c.key == key), None)
            if match is None:
                break
            path.append(match)
            node = match
        return path

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="config-breadcrumb")
            yield Vertical(id="config-body")

    def on_mount(self) -> None:
        self._render_level()

    # ── Rendering ──────────────────────────────────────────────────────────

    def _path_key(self) -> tuple[str, ...]:
        return tuple(n.key for n in self._path)

    def _breadcrumb_text(self) -> None:
        crumbs = " › ".join(n.label for n in self._path)
        if self._mode == "confirm":
            hint = _hint_bar([("↑↓", "move"), ("Enter", "select"), ("Esc", "keep editing")])
        elif self._mode == "mcp_list":
            hint = _hint_bar(
                [("↑↓", "move"), ("Enter", "remove/choose"), ("←", "back"), ("Esc", "close")]
            )
        elif self._mode == "mcp_load_select":
            hint = _hint_bar(
                [
                    ("↑↓", "move"),
                    ("Enter", "toggle"),
                    ("Ctrl+A", "select/clear all"),
                    ("←", "cancel"),
                ]
            )
        elif self._mode == "mcp_manual_transport":
            hint = _hint_bar([("↑↓", "move"), ("Enter", "select"), ("←", "cancel")])
        elif self._mode in _MCP_TEXT_MODES:
            hint = _hint_bar([("Enter", "continue"), ("←/Esc", "cancel")])
        else:
            hint = _hint_bar(
                [("↑↓", "move"), ("Enter/→", "select"), ("←", "back"), ("Esc", "close")]
            )
        lines = [
            f"[bold {theme['accent']}]{escape(crumbs)}[/bold {theme['accent']}]",
            hint,
        ]
        if self._error:
            lines.append(f"[{theme['error']}]{escape(self._error)}[/{theme['error']}]")
        elif self._status:
            lines.append(f"[{theme['text_muted']}]{escape(self._status)}[/{theme['text_muted']}]")
        self.query_one("#config-breadcrumb", Static).update("\n".join(lines))

    def _clear_body(self) -> None:
        self.query_one("#config-body", Vertical).remove_children()

    def _render_level(self) -> None:
        self._error = ""
        self._mode = "browse"
        node = self._path[-1]
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)

        if node.children:
            options = [Option(self._child_label(child), id=child.key) for child in node.children]
            option_list = _ConfigOptionList(*options)
            body.mount(option_list)
            option_list.highlighted = self._cursor_by_path.get(self._path_key(), 0)
            option_list.focus()
            return

        assert node.leaf is not None
        leaf = node.leaf
        if leaf.kind == "mcp_servers":
            self._render_mcp_list()
            return

        choices = leaf.options(self._staged)
        if choices:
            options = []
            current = self._staged.get(leaf.field)
            for choice in choices:
                marker = "●" if choice.value == current else "○"
                options.append(Option(f"{marker} {choice.label}", id=choice.value))
            option_list = _ConfigOptionList(*options)
            body.mount(option_list)
            option_list.highlighted = self._cursor_by_path.get(self._path_key(), 0)
            option_list.focus()
        else:
            self._mode = "text"
            text_input = _ConfigTextInput(
                value=self._staged.get(leaf.field, ""),
                placeholder=leaf.text_placeholder,
            )
            body.mount(text_input)
            text_input.focus()

    def _child_label(self, child: ConfigNode) -> str:
        if child.leaf is not None and child.leaf.kind == "mcp_servers":
            count = len(self._mcp_servers)
            suffix = f"{count} server{'s' if count != 1 else ''}" if count else "none configured"
            return f"{child.label}  [dim]({suffix})[/dim]"
        if child.leaf is not None:
            value = self._staged.get(child.leaf.field, "")
            return f"{child.label}  [dim]({value})[/dim]" if value else child.label
        return child.label

    # ── MCP servers editor ──────────────────────────────────────────────────

    def _render_mcp_list(self) -> None:
        self._mode = "mcp_list"
        self._error = ""
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        options = [
            Option(
                f"✕ {escape(server.name)} — {escape(server.url)} [dim]({server.transport.value})[/dim]",
                id=f"remove:{idx}",
            )
            for idx, server in enumerate(self._mcp_servers)
        ]
        options.append(Option("+ Load from config file", id="mcp_load"))
        options.append(Option("+ Add manually", id="mcp_manual"))
        option_list = _ConfigOptionList(*options)
        body.mount(option_list)
        option_list.highlighted = self._cursor_by_path.get((*self._path_key(), "mcp_list"), 0)
        option_list.focus()

    def _render_mcp_load_path(self) -> None:
        self._mode = "mcp_load_path"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        text_input = _ConfigTextInput(placeholder="Path to mcp.json")
        body.mount(text_input)
        text_input.focus()

    def _render_mcp_manual_name(self) -> None:
        self._mode = "mcp_manual_name"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        text_input = _ConfigTextInput(placeholder="Server name")
        body.mount(text_input)
        text_input.focus()

    def _render_mcp_manual_url(self, value: str = "") -> None:
        self._mode = "mcp_manual_url"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        text_input = _ConfigTextInput(value=value, placeholder="Server URL")
        body.mount(text_input)
        text_input.focus()

    def _render_mcp_load_select(self) -> None:
        self._mode = "mcp_load_select"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        options = []
        for idx, server in enumerate(self._mcp_candidates):
            marker = "[x]" if idx in self._mcp_selected else "[ ]"
            options.append(
                Option(f"{marker} {escape(server.name)} — {escape(server.url)}", id=f"cand:{idx}")
            )
        options.append(
            Option(f"✔ Add selected ({len(self._mcp_selected)})", id="confirm_selection")
        )
        option_list = _McpSelectOptionList(*options)
        body.mount(option_list)
        option_list.highlighted = self._mcp_select_cursor
        option_list.focus()

    def _upsert_mcp_server(self, server: MCPServerSpec) -> None:
        for idx, existing in enumerate(self._mcp_servers):
            if existing.url == server.url:
                self._mcp_servers[idx] = server
                return
        self._mcp_servers.append(server)

    def _reset_mcp_transient_state(self) -> None:
        self._mcp_candidates = []
        self._mcp_selected = set()
        self._mcp_select_cursor = 0
        self._mcp_checking = False
        self._status = ""
        self._mcp_pending_name = None
        self._mcp_pending_url = None
        self._mcp_pending_transport = MCPTransport.HTTP
        self._mcp_pending_headers = {}

    def _handle_mcp_load_path_submit(self, value: str) -> None:
        if not value:
            self._render_mcp_list()
            return
        try:
            candidates = load_named_mcp_servers(Path(value))
        except Exception as exc:
            self._error = f"Couldn't read {value}: {exc}"
            self._render_mcp_load_path()
            return
        if not candidates:
            self._error = f"No MCP servers found in {value}"
            self._render_mcp_load_path()
            return
        self._error = ""
        self._mcp_candidates = candidates
        self._mcp_selected = set()
        self._mcp_select_cursor = 0
        self._render_mcp_load_select()

    def _handle_mcp_manual_name_submit(self, value: str) -> None:
        if not value:
            self._render_mcp_list()
            return
        self._mcp_pending_name = value
        self._render_mcp_manual_url()

    def _handle_mcp_manual_url_submit(self, value: str) -> None:
        if not value or not self._mcp_pending_name:
            self._reset_mcp_transient_state()
            self._render_mcp_list()
            return
        self._mcp_pending_url = value
        self._render_mcp_manual_transport()

    def _render_mcp_manual_transport(self) -> None:
        self._mode = "mcp_manual_transport"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        options = [
            Option(_MCP_TRANSPORT_LABELS[transport], id=transport.value)
            for transport in MCPTransport
        ]
        option_list = _ConfigOptionList(*options)
        body.mount(option_list)
        option_list.focus()

    def _render_mcp_manual_header(self) -> None:
        self._mode = "mcp_manual_header"
        if not self._mcp_checking:
            count = len(self._mcp_pending_headers)
            self._status = (
                f"{count} header{'s' if count != 1 else ''} added — "
                "Enter with nothing typed to finish"
                if count
                else "Optional — e.g. Authorization: Bearer <token>. Enter with nothing typed to skip"
            )
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        text_input = _ConfigTextInput(placeholder="Header (Name: Value)")
        body.mount(text_input)
        text_input.focus()

    async def _handle_mcp_manual_header_submit(self, value: str) -> None:
        if not value:
            await self._finalize_manual_mcp_add()
            return
        name, sep, header_value = value.partition(":")
        name, header_value = name.strip(), header_value.strip()
        if not sep or not name or not header_value:
            self._error = "Expected 'Header-Name: value'"
            self._render_mcp_manual_header()
            return
        self._error = ""
        self._mcp_pending_headers[name] = header_value
        self._render_mcp_manual_header()

    async def _finalize_manual_mcp_add(self) -> None:
        name = self._mcp_pending_name
        url = self._mcp_pending_url
        if not name or not url or self._mcp_checking:
            return
        server = MCPServerSpec(
            name=name,
            url=url,
            transport=self._mcp_pending_transport,
            headers=dict(self._mcp_pending_headers),
        )

        self._error = ""
        self._status = "Checking connection…"
        self._mcp_checking = True
        self._render_mcp_manual_header()
        try:
            await check_mcp_server(server)
        except Exception as exc:
            if self._mode != "mcp_manual_header" or not self._mcp_checking:
                return  # user navigated away while the check was in flight
            self._mcp_checking = False
            self._status = ""
            self._error = f"Couldn't connect to {url}: {exc}"
            self._render_mcp_manual_header()
            return

        if self._mode != "mcp_manual_header" or not self._mcp_checking:
            return  # user navigated away while the check was in flight
        self._upsert_mcp_server(server)
        self._reset_mcp_transient_state()
        self._render_mcp_list()

    def _render_confirm(self) -> None:
        self._mode = "confirm"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        options = [
            Option(
                Text.from_markup(f"[bold {theme['success']}]{_SAVE}[/bold {theme['success']}]"),
                id=_SAVE,
            ),
            Option(
                Text.from_markup(
                    f"[{theme['text_secondary']}]{_DISCARD}[/{theme['text_secondary']}]"
                ),
                id=_DISCARD,
            ),
        ]
        option_list = _ConfigOptionList(*options)
        body.mount(option_list)
        option_list.focus()

    # ── Selection handling ────────────────────────────────────────────────

    @on(OptionList.OptionSelected)
    async def _on_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = event.option.id
        assert option_id is not None

        if self._mode == "confirm":
            if option_id == _SAVE:
                diff = diff_values(self._initial_values, self._staged)
                mcp_changes = (
                    list(self._mcp_servers)
                    if self._mcp_servers != self._initial_mcp_servers
                    else None
                )
                error = await self._on_apply(diff, mcp_changes)
                if error is not None:
                    self._error = error
                    self._initial_values = dict(self._staged)
                    self._initial_mcp_servers = list(self._mcp_servers)
                    self._render_level()
                    return
            # Both Save (on success) and Discard close the whole panel.
            self.dismiss()
            return

        if self._mode == "mcp_list":
            self._cursor_by_path[(*self._path_key(), "mcp_list")] = (
                event.option_list.highlighted or 0
            )
            if option_id.startswith("remove:"):
                del self._mcp_servers[int(option_id.split(":", 1)[1])]
                self._render_mcp_list()
            elif option_id == "mcp_load":
                self._render_mcp_load_path()
            elif option_id == "mcp_manual":
                self._render_mcp_manual_name()
            return

        if self._mode == "mcp_load_select":
            self._mcp_select_cursor = event.option_list.highlighted or 0
            if option_id == "confirm_selection":
                for idx in sorted(self._mcp_selected):
                    self._upsert_mcp_server(self._mcp_candidates[idx])
                self._reset_mcp_transient_state()
                self._render_mcp_list()
                return
            idx = int(option_id.split(":", 1)[1])
            self._mcp_selected.symmetric_difference_update({idx})
            self._render_mcp_load_select()
            return

        if self._mode == "mcp_manual_transport":
            self._mcp_pending_transport = MCPTransport(option_id)
            self._render_mcp_manual_header()
            return

        node = self._path[-1]
        if node.children:
            self._cursor_by_path[self._path_key()] = event.option_list.highlighted or 0
            child = next(c for c in node.children if c.key == option_id)
            self._path.append(child)
            self._render_level()
            return

        assert node.leaf is not None
        self._stage_value(node.leaf, option_id)
        self._path.pop()
        self._render_level()

    @on(_ToggleSelectAll)
    def _on_toggle_select_all(self, event: _ToggleSelectAll) -> None:
        event.stop()
        if self._mode != "mcp_load_select":
            return
        if len(self._mcp_selected) == len(self._mcp_candidates):
            self._mcp_selected = set()
        else:
            self._mcp_selected = set(range(len(self._mcp_candidates)))
        self._render_mcp_load_select()

    @on(Input.Submitted)
    async def _on_text_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        value = event.value.strip()
        if self._mode == "mcp_load_path":
            self._handle_mcp_load_path_submit(value)
            return
        if self._mode == "mcp_manual_name":
            self._handle_mcp_manual_name_submit(value)
            return
        if self._mode == "mcp_manual_url":
            self._handle_mcp_manual_url_submit(value)
            return
        if self._mode == "mcp_manual_header":
            await self._handle_mcp_manual_header_submit(value)
            return

        node = self._path[-1]
        assert node.leaf is not None
        leaf = node.leaf
        if value or leaf.allow_empty:
            if value and leaf.validate is not None:
                error = leaf.validate(value)
                if error is not None:
                    self._error = error
                    self._render_leaf_text_retry(leaf, value)
                    return
            self._stage_value(leaf, value)
        self._path.pop()
        self._render_level()

    def _render_leaf_text_retry(self, leaf: ConfigLeaf, value: str) -> None:
        """Re-show a leaf's text input after a validation error.

        Unlike ``_render_level``, this keeps ``self._error`` (so the message
        set by the caller stays visible) and preserves the invalid draft the
        user just typed, so they can fix it in place instead of retyping.
        """
        self._mode = "text"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        text_input = _ConfigTextInput(value=value, placeholder=leaf.text_placeholder)
        body.mount(text_input)
        text_input.focus()

    def _stage_value(self, leaf: ConfigLeaf, value: str) -> None:
        self._staged[leaf.field] = value
        if leaf.linked_field and leaf.linked_default is not None:
            self._staged[leaf.linked_field] = leaf.linked_default(value)

    # ── Navigation ─────────────────────────────────────────────────────────

    @on(_NavigateBack)
    def _on_navigate_back(self, event: _NavigateBack) -> None:
        event.stop()
        if self._mode == "confirm":
            self._render_level()
            return
        if self._mode in _MCP_WIZARD_MODES:
            self._reset_mcp_transient_state()
            self._render_mcp_list()
            return
        if len(self._path) > 1:
            self._path.pop()
            self._render_level()

    @on(_RequestClose)
    def _on_request_close(self, event: _RequestClose) -> None:
        event.stop()
        if self._mode == "confirm":
            self.dismiss()
            return
        if self._mode in _MCP_WIZARD_MODES:
            self._reset_mcp_transient_state()
            self._render_mcp_list()
            return
        if diff_values(self._initial_values, self._staged) or (
            self._mcp_servers != self._initial_mcp_servers
        ):
            self._render_confirm()
        else:
            self.dismiss()
