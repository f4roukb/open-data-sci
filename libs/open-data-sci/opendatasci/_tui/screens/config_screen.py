"""ConfigScreen — the navigable panel behind /config, /models and /providers.

A ``ModalScreen`` pushed on top of the running app, following the same
pattern as ``OnboardingScreen``: the chat screen underneath stays mounted and
keeps receiving controller-driven updates while this is up, so nothing needs
pausing/buffering for background messages to survive the panel being open.

Navigation: Up/Down move the highlight within a level (native to
``OptionList``). Enter, Right and mouse click all select the highlighted row
(Textual's own ``OptionList.action_select`` already posts ``OptionSelected``
for all three). Left and Escape both move up one level; only at the root of
the tree does Escape instead leave the whole panel, prompting Save/Discard
first if anything was changed. A text field's staged value updates live as
the user types (no Enter needed) so leaving the field via Left/Escape still
picks up the latest typed value.
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

_MCP_TEXT_MODES = ("mcp_load_path",)
# Every mode that belongs to an "add a server" sub-flow (load-from-file or
# manual) — Back/Escape from any of these cancels the whole sub-flow rather
# than stepping back one field at a time.
_MCP_WIZARD_MODES = (*_MCP_TEXT_MODES, "mcp_load_select", "mcp_manual_form")
_MCP_TRANSPORT_LABELS = {
    MCPTransport.HTTP: "HTTP (Streamable, recommended)",
    MCPTransport.SSE: "SSE (Server-Sent Events, legacy)",
}
# Focus order for the "Add manually" form — Up/Down cycle through these.
_MCP_FORM_FIELD_IDS = (
    "mcp-form-name",
    "mcp-form-url",
    "mcp-form-transport",
    "mcp-form-headers",
    "mcp-form-action",
)


def _hint_chip(key: str, label: str) -> str:
    """A small "key + label" hint chip, styled like the footer's key hints."""
    chip_style = f"{theme['text_primary']} on {theme['separator']}"
    return f"[{chip_style}] {key} [/{chip_style}]  [{theme['text_muted']}]{label}[/{theme['text_muted']}]"


def _hint_bar(pairs: list[tuple[str, str]]) -> str:
    return "    ".join(_hint_chip(key, label) for key, label in pairs)


def _parse_headers_text(text: str) -> tuple[dict[str, str], str | None]:
    """Parse a comma-separated ``Name: Value, Name2: Value2`` string.

    Returns the parsed headers, or an error message (and an empty dict) if
    any comma-separated part isn't a well-formed ``Name: Value`` pair.
    """
    text = text.strip()
    if not text:
        return {}, None
    headers: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition(":")
        name, value = name.strip(), value.strip()
        if not sep or not name or not value:
            return {}, f"Expected 'Header-Name: value', got {part!r}"
        headers[name] = value
    return headers, None


class _NavigateBack(Message):
    pass


class _RequestClose(Message):
    pass


class _FormNavUp(Message):
    pass


class _FormNavDown(Message):
    pass


class _McpTransportToggled(Message):
    def __init__(self, transport: MCPTransport) -> None:
        self.transport = transport
        super().__init__()


class _McpFormSubmit(Message):
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
        Binding("up", "form_nav_up", show=False),
        Binding("down", "form_nav_down", show=False),
    ]

    def action_go_back(self) -> None:
        self.post_message(_NavigateBack())

    def action_request_close(self) -> None:
        self.post_message(_RequestClose())

    def action_form_nav_up(self) -> None:
        self.post_message(_FormNavUp())

    def action_form_nav_down(self) -> None:
        self.post_message(_FormNavDown())


class _ToggleSelectAll(Message):
    pass


class _McpSelectOptionList(_ConfigOptionList):
    BINDINGS = [Binding("ctrl+a", "toggle_all", show=False)]

    def action_toggle_all(self) -> None:
        self.post_message(_ToggleSelectAll())


class _MCPTransportField(Static, can_focus=True):
    """Focusable, non-Input row that cycles HTTP/SSE with Enter.

    Lives inside the "Add manually" form alongside plain ``Input`` fields —
    Up/Down/Left/Escape mirror ``_ConfigTextInput`` so the whole form
    navigates uniformly regardless of which row is focused.
    """

    DEFAULT_CSS = """
    _MCPTransportField {
        padding: 0 1;
    }
    _MCPTransportField:focus {
        background: $ods-surface-alt;
    }
    """
    BINDINGS = [
        Binding("enter", "toggle_transport", show=False),
        Binding("left", "go_back", show=False),
        Binding("escape", "request_close", show=False),
        Binding("up", "form_nav_up", show=False),
        Binding("down", "form_nav_down", show=False),
    ]

    def __init__(self, transport: MCPTransport, id: str | None = None) -> None:
        super().__init__(id=id)
        self.transport = transport
        self._refresh()

    def _refresh(self) -> None:
        self.update(f"‹ {_MCP_TRANSPORT_LABELS[self.transport]} ›")

    def action_toggle_transport(self) -> None:
        transports = list(MCPTransport)
        self.transport = transports[(transports.index(self.transport) + 1) % len(transports)]
        self._refresh()
        self.post_message(_McpTransportToggled(self.transport))

    def action_go_back(self) -> None:
        self.post_message(_NavigateBack())

    def action_request_close(self) -> None:
        self.post_message(_RequestClose())

    def action_form_nav_up(self) -> None:
        self.post_message(_FormNavUp())

    def action_form_nav_down(self) -> None:
        self.post_message(_FormNavDown())


class _MCPFormAction(Static, can_focus=True):
    """The "Add server" trigger row at the bottom of the manual-add form."""

    DEFAULT_CSS = """
    _MCPFormAction {
        padding: 0 1;
    }
    _MCPFormAction:focus {
        background: $ods-surface-alt;
    }
    """
    BINDINGS = [
        Binding("enter", "submit", show=False),
        Binding("left", "go_back", show=False),
        Binding("escape", "request_close", show=False),
        Binding("up", "form_nav_up", show=False),
    ]

    def action_submit(self) -> None:
        self.post_message(_McpFormSubmit())

    def action_go_back(self) -> None:
        self.post_message(_NavigateBack())

    def action_request_close(self) -> None:
        self.post_message(_RequestClose())

    def action_form_nav_up(self) -> None:
        self.post_message(_FormNavUp())


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
    ConfigScreen .mcp-form-label {
        color: $ods-text-muted;
        margin-top: 1;
    }
    ConfigScreen .mcp-form-label:first-child {
        margin-top: 0;
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
        # Fields staged live from the manual-add form (name, url, transport,
        # headers all shown at once) before the final connectivity check.
        self._mcp_pending_name: str | None = None
        self._mcp_pending_url: str | None = None
        self._mcp_pending_transport: MCPTransport = MCPTransport.HTTP
        self._mcp_pending_headers_text: str = ""

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
        elif self._mode == "mcp_manual_form":
            hint = _hint_bar(
                [("↑↓", "move field"), ("Enter", "next/toggle/add"), ("←/Esc", "cancel")]
            )
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
            options = [
                Option(self._section_header_label(child), id=child.key, disabled=True)
                if child.header
                else Option(self._child_label(child), id=child.key)
                for child in node.children
            ]
            option_list = _ConfigOptionList(*options)
            body.mount(option_list)
            default_highlight = next(
                (i for i, child in enumerate(node.children) if not child.header), 0
            )
            option_list.highlighted = self._cursor_by_path.get(self._path_key(), default_highlight)
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

    def _section_header_label(self, child: ConfigNode) -> str:
        return f"[bold {theme['text_muted']}]{escape(child.label)}[/bold {theme['text_muted']}]"

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

    def _render_mcp_manual_form(self) -> None:
        """The "Add manually" form — name, URL, transport and headers all on
        one screen at once, navigated with Up/Down between fields."""
        self._mode = "mcp_manual_form"
        self._breadcrumb_text()
        self._clear_body()
        body = self.query_one("#config-body", Vertical)
        body.mount(Static("Server name", classes="mcp-form-label"))
        body.mount(
            _ConfigTextInput(
                value=self._mcp_pending_name or "",
                placeholder="e.g. my-server",
                id="mcp-form-name",
            )
        )
        body.mount(Static("Server URL", classes="mcp-form-label"))
        body.mount(
            _ConfigTextInput(
                value=self._mcp_pending_url or "",
                placeholder="https://…",
                id="mcp-form-url",
            )
        )
        body.mount(Static("Transport", classes="mcp-form-label"))
        body.mount(_MCPTransportField(self._mcp_pending_transport, id="mcp-form-transport"))
        body.mount(Static("Headers (optional)", classes="mcp-form-label"))
        body.mount(
            _ConfigTextInput(
                value=self._mcp_pending_headers_text,
                placeholder="Name: Value, Name2: Value2",
                id="mcp-form-headers",
            )
        )
        action_label = "Checking connection…" if self._mcp_checking else "✔ Add server"
        body.mount(_MCPFormAction(action_label, id="mcp-form-action"))
        self.query_one("#mcp-form-name", _ConfigTextInput).focus()

    def _move_form_focus(self, delta: int) -> None:
        if self._mode != "mcp_manual_form":
            return
        focused = self.focused
        if focused is None or focused.id not in _MCP_FORM_FIELD_IDS:
            return
        new_idx = _MCP_FORM_FIELD_IDS.index(focused.id) + delta
        if 0 <= new_idx < len(_MCP_FORM_FIELD_IDS):
            self.query_one(f"#{_MCP_FORM_FIELD_IDS[new_idx]}").focus()

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
        self._mcp_pending_headers_text = ""

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

    @on(_McpFormSubmit)
    async def _on_mcp_form_submit(self, event: _McpFormSubmit) -> None:
        event.stop()
        if self._mode != "mcp_manual_form" or self._mcp_checking:
            return
        name = (self._mcp_pending_name or "").strip()
        url = (self._mcp_pending_url or "").strip()
        if not name or not url:
            self._error = "Server name and URL are required"
            self._render_mcp_manual_form()
            return
        headers, header_error = _parse_headers_text(self._mcp_pending_headers_text)
        if header_error:
            self._error = header_error
            self._render_mcp_manual_form()
            return

        server = MCPServerSpec(
            name=name, url=url, transport=self._mcp_pending_transport, headers=headers
        )
        self._error = ""
        self._status = "Checking connection…"
        self._mcp_checking = True
        self._render_mcp_manual_form()
        try:
            await check_mcp_server(server)
        except Exception as exc:
            if self._mode != "mcp_manual_form" or not self._mcp_checking:
                return  # user navigated away while the check was in flight
            self._mcp_checking = False
            self._status = ""
            self._error = f"Couldn't connect to {url}: {exc}"
            self._render_mcp_manual_form()
            return

        if self._mode != "mcp_manual_form" or not self._mcp_checking:
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
                self._render_mcp_manual_form()
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
        if self._mode == "mcp_manual_form":
            # Enter in a text field advances to the next field rather than
            # submitting the whole form — only the trailing action row does.
            self._move_form_focus(1)
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

    @on(Input.Changed)
    def _on_text_changed(self, event: Input.Changed) -> None:
        """Stage a text field's value as it's typed, so leaving the field via
        Left/Escape (rather than Enter) still keeps what was typed."""
        if self._mode == "text":
            node = self._path[-1]
            assert node.leaf is not None
            leaf = node.leaf
            value = event.value.strip()
            if value or leaf.allow_empty:
                if value and leaf.validate is not None and leaf.validate(value) is not None:
                    return  # invalid partial input — Enter will surface the error
                self._stage_value(leaf, value)
            return
        if self._mode != "mcp_manual_form":
            return
        if event.input.id == "mcp-form-name":
            self._mcp_pending_name = event.value
        elif event.input.id == "mcp-form-url":
            self._mcp_pending_url = event.value
        elif event.input.id == "mcp-form-headers":
            self._mcp_pending_headers_text = event.value

    # ── Navigation ─────────────────────────────────────────────────────────

    def _go_back_one_level(self) -> bool:
        """Pop one level of the config tree. Returns False at the root."""
        if len(self._path) > 1:
            self._path.pop()
            self._render_level()
            return True
        return False

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
        self._go_back_one_level()

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
        # Escape behaves like Left (step back one level) everywhere except
        # at the tree's root, where there's nothing left to step back to —
        # only there does it mean "leave the panel", prompting Save/Discard.
        if self._go_back_one_level():
            return
        if diff_values(self._initial_values, self._staged) or (
            self._mcp_servers != self._initial_mcp_servers
        ):
            self._render_confirm()
        else:
            self.dismiss()

    @on(_FormNavUp)
    def _on_form_nav_up(self, event: _FormNavUp) -> None:
        event.stop()
        self._move_form_focus(-1)

    @on(_FormNavDown)
    def _on_form_nav_down(self, event: _FormNavDown) -> None:
        event.stop()
        self._move_form_focus(1)

    @on(_McpTransportToggled)
    def _on_mcp_transport_toggled(self, event: _McpTransportToggled) -> None:
        event.stop()
        self._mcp_pending_transport = event.transport
