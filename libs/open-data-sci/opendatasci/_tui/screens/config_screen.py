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

from typing import Awaitable, Callable, Literal

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

_SAVE = "Save changes"
_DISCARD = "Discard changes"


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
        on_apply: Callable[[dict[str, str]], Awaitable[str | None]],
    ) -> None:
        super().__init__()
        self._root = root
        self._initial_values = dict(initial_values)
        self._staged = dict(initial_values)
        self._on_apply = on_apply
        self._path: list[ConfigNode] = self._resolve_path(root, start_path)
        self._cursor_by_path: dict[tuple[str, ...], int] = {}
        self._mode: Literal["browse", "text", "confirm"] = "browse"
        self._error: str = ""

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
        hint = (
            _hint_bar([("↑↓", "move"), ("Enter/→", "select"), ("←", "back"), ("Esc", "close")])
            if self._mode != "confirm"
            else _hint_bar([("↑↓", "move"), ("Enter", "select"), ("Esc", "keep editing")])
        )
        lines = [
            f"[bold {theme['accent']}]{escape(crumbs)}[/bold {theme['accent']}]",
            hint,
        ]
        if self._error:
            lines.append(f"[{theme['error']}]{escape(self._error)}[/{theme['error']}]")
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
        if child.leaf is not None:
            value = self._staged.get(child.leaf.field, "")
            return f"{child.label}  [dim]({value})[/dim]" if value else child.label
        return child.label

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
                Text.from_markup(f"[{theme['text_secondary']}]{_DISCARD}[/{theme['text_secondary']}]"),
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
                error = await self._on_apply(diff)
                if error is not None:
                    self._error = error
                    self._initial_values = dict(self._staged)
                    self._render_level()
                    return
            # Both Save (on success) and Discard close the whole panel.
            self.dismiss()
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

    @on(Input.Submitted)
    def _on_text_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        node = self._path[-1]
        assert node.leaf is not None
        value = event.value.strip()
        if value:
            self._stage_value(node.leaf, value)
        self._path.pop()
        self._render_level()

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
        if len(self._path) > 1:
            self._path.pop()
            self._render_level()

    @on(_RequestClose)
    def _on_request_close(self, event: _RequestClose) -> None:
        event.stop()
        if self._mode == "confirm":
            self.dismiss()
            return
        if diff_values(self._initial_values, self._staged):
            self._render_confirm()
        else:
            self.dismiss()
