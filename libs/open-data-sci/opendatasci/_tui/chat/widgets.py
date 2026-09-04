"""Textual widgets for OpenDataSci TUI v2."""

import asyncio
import bisect
import logging
import math
import time
from pathlib import Path

from rich.highlighter import Highlighter
from rich.markup import escape
from rich.rule import Rule
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Input, Static
from textual.widgets import Markdown as TUIMarkdown
from textual.widgets.markdown import MarkdownStream

try:
    from textual.widgets import Image as _TUIImage  # type: ignore[attr-defined]
except ImportError:
    _TUIImage = None

from opendatasci._tui.chat.commands import SLASH_COMMANDS
from opendatasci._tui.chat.models import SPINNER, SPINNER_INTERVAL
from opendatasci._tui.style.theme import active as theme

logger = logging.getLogger(__name__)


def _scroll_is_at_bottom(container: ScrollableContainer) -> bool:
    """True when *container* is scrolled to (within one row of) the bottom."""
    return container.scroll_offset.y >= container.max_scroll_y - 1


class CommandHighlighter(Highlighter):
    """Highlight a valid (or partial) slash command at the start of input."""

    _sorted_commands: list[str] = sorted(SLASH_COMMANDS)

    def highlight(self, text: Text) -> None:
        plain = text.plain
        if not plain.startswith("/"):
            return
        token = plain.split()[0] if plain.split() else plain
        is_valid = token in SLASH_COMMANDS
        if not is_valid:
            idx = bisect.bisect_left(self._sorted_commands, token)
            is_prefix = idx < len(self._sorted_commands) and self._sorted_commands[idx].startswith(
                token
            )
        else:
            is_prefix = False
        if is_valid or is_prefix:
            text.stylize(f"bold {theme['accent']}", 0, len(token))


class AppHeader(Widget):
    """Docked top bar: logo left, version/workspace info right."""

    DEFAULT_CSS = """
    #header-layout { layout: horizontal; height: 6; }
    """

    def __init__(
        self,
        version: str,
        workspace: str,
        workspace_name: str | None = None,
    ) -> None:
        super().__init__()
        self._version = version
        self._workspace = workspace
        self._workspace_name = workspace_name
        self._file_count: str = ""
        self._model_info: str = ""
        self._background_tasks: str = ""
        _logo_path = Path(__file__).parents[5] / "docs" / "logo.png"
        self._use_image = _TUIImage is not None and _logo_path.exists()
        self._logo_path = _logo_path

    def compose(self) -> ComposeResult:
        with Horizontal(id="header-layout"):
            if self._use_image:
                yield _TUIImage(self._logo_path, id="header-logo")
            else:
                yield Static(id="header-logo")
            yield Static(id="header-info")

    def on_mount(self) -> None:
        if not self._use_image:
            self._render_logo()
        self._render_info()

    def _render_logo(self) -> None:
        bold = f"bold {theme['logo']}"
        t = Text()
        t.append("OpenDataSci", style=bold)
        self.query_one("#header-logo", Static).update(t)

    def _render_info(self) -> None:
        lbl = theme["text_secondary"]
        t = Text()
        t.append("Version    ", style=lbl)
        version_str = f"v{self._version}"
        t.append(version_str, style=f"bold {theme['logo']}")
        t.append("\n")
        t.append("Workspace  ", style=lbl)
        t.append(self._workspace, style=theme["text_primary"])
        if self._file_count:
            t.append(f"  ({self._file_count})", style=theme["text_secondary"])
        if self._workspace_name:
            t.append("   Workspace  ", style=lbl)
            t.append(self._workspace_name, style=theme["accent"])
        if self._model_info:
            t.append("\n")
            t.append("Model      ", style=lbl)
            t.append(self._model_info, style=theme["accent"])
        if self._background_tasks:
            t.append("\n")
            t.append("Background ", style=lbl)
            t.append(self._background_tasks, style=theme["accent"])
        self.query_one("#header-info", Static).update(t)

    def set_workspace(self, name: str | None) -> None:
        self._workspace_name = name
        self._render_info()

    def set_file_count(self, description: str) -> None:
        self._file_count = description
        self._render_info()

    def set_model_info(self, description: str) -> None:
        self._model_info = description
        self._render_info()

    def set_background_tasks(self, description: str) -> None:
        """Update the "running background tasks" line, or clear it if *description* is empty."""
        self._background_tasks = description
        self._render_info()


class TurnStatusBar(Static):
    """Inline status bar appended at the end of the conversation during an agent turn."""

    DEFAULT_CSS = """
    TurnStatusBar {
        width: auto;
        height: auto;
        padding: 0 2;
        margin-bottom: 0;
        text-align: right;
    }
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        # Initialise instance variables so that stop() / on_unmount() are safe
        # to call even if on_mount() has not yet fired (e.g. when a stale
        # bar widget is removed by add_turn_status_bar's cleanup loop right
        # after mount() but before the event-loop dispatches on_mount).
        self._stopped: bool = False  # False = running once on_mount fires
        self._mounted: bool = False  # True only after on_mount has run
        self._start: float = 0.0
        self._interval: Timer | None = None
        self._context_tokens: int | None = None
        self._cached_tokens: int | None = None

    def on_mount(self) -> None:
        self._mounted = True
        self._start = time.monotonic()
        self._interval = self.set_interval(1, self._tick)
        self._stopped = False
        self._tick()

    def _fmt(self, s: int) -> str:
        if s < 60:
            return f"{s}s"
        mins, secs = divmod(s, 60)
        return f"{mins}min {secs:02d}s" if secs else f"{mins}min"

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        """Format token count as truncated-to-one-decimal thousands, e.g. 3250 → '3.2k'."""
        k = (n // 100) / 10
        return f"{k:.1f}k"

    def _context_suffix(self) -> str:
        if self._context_tokens is None:
            return ""
        size = self._fmt_tokens(self._context_tokens)
        if self._cached_tokens is None:
            return f" | Context: {size} tokens"
        # cached_tokens is the fraction of the total context (input + output)
        # that was served from cache. Clamped defensively: cache_read is a
        # subset of input_tokens by construction, so the ratio should never
        # exceed 1, but this guards against any upstream provider quirk that
        # breaks that invariant.
        pct = min(99.9, math.ceil(self._cached_tokens / max(self._context_tokens, 1) * 1000) / 10)
        return f" | Context: {size} tokens ({pct:.1f}% cached)"

    def _tick(self) -> None:
        s = int(time.monotonic() - self._start)
        label = f"Sciencing for {self._fmt(s)}{self._context_suffix()}"
        self.update(f"[{theme['text_muted']}]{label}[/{theme['text_muted']}]")

    def update_context(self, context_tokens: int | None, cached_tokens: int | None) -> None:
        """Update context size and cached token count and re-render the label."""
        self._context_tokens = context_tokens
        self._cached_tokens = cached_tokens
        if not self._stopped and self._mounted:
            self._tick()

    def stop(self) -> None:
        if self._stopped or not self._mounted:
            return
        self._stopped = True
        if self._interval is not None:
            self._interval.stop()
        s = int(time.monotonic() - self._start)
        label = f"Scienced for {self._fmt(s)}{self._context_suffix()}"
        self.update(f"[{theme['text_muted']}]{label}[/{theme['text_muted']}]")

    def on_unmount(self) -> None:
        # Guard against removal before stop() was explicitly called, and also
        # against removal before on_mount() ever fired (_mounted is False).
        if self._mounted and not self._stopped:
            self._stopped = True
            if self._interval is not None:
                self._interval.stop()


# Short, discoverable tips — cycled round-robin by TipsBar regardless of
# what the user is doing, so people find features they'd otherwise only see
# by reading /help.
_TIPS: tuple[str, ...] = (
    "Tip: type /help to see all commands",
    "Tip: type /config to change theme, model, or provider",
    "Tip: type /models to switch the active model",
    "Tip: type /providers to switch the active provider",
    "Tip: type @path/to/file to attach a file",
    "Tip: press Tab to autocomplete a command or file path",
    "Tip: press ↑ / ↓ to browse your input history",
    "Tip: press Esc to stop the agent mid-turn",
    "Tip: type /compact to summarise a long conversation",
    "Tip: type /clear to wipe the conversation and start fresh",
    "Tip: type /reset to reload your data from disk",
    "Tip: type /ls-workspace to list your workspace files",
    "Tip: type /cancel-message to drop the last queued message",
    "Tip: press Ctrl+R to reset the session",
    "Tip: press Ctrl+L to clear the conversation",
)

_TIP_INTERVAL_SECONDS = 7


class TipsBar(Static):
    """Rotating one-line tip docked at the footer's left — always cycling,
    independent of agent/turn state."""

    DEFAULT_CSS = """
    TipsBar {
        width: 1fr;
        height: auto;
        padding: 0 2;
        color: $ods-text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._index = 0

    def on_mount(self) -> None:
        self._render_tip()
        self.set_interval(_TIP_INTERVAL_SECONDS, self._tick)

    def _tick(self) -> None:
        self._index = (self._index + 1) % len(_TIPS)
        self._render_tip()

    def _render_tip(self) -> None:
        self.update(_TIPS[self._index])


class MessageBubble(Widget):
    """A single chat message — user, agent (streaming), or question.

    Agent-role bubbles stream their Markdown through Textual's own
    ``Markdown.get_stream()`` (``MarkdownStream``), which owns the
    coalescing/backpressure that used to be hand-rolled here with a flush
    timer and a dirty flag.

    Two invariants keep this race-free:

    - ``_ready`` (set once ``on_mount`` has created the stream) makes
      ``append``/``set_content``/``finish`` safe to call at any point,
      including synchronously right after construction, before Textual has
      even mounted the widget — the original source of the "empty agent
      bubble" bug this class used to work around with ``_dirty``.
    - ``_written_len`` is a cursor into ``_content``: only the *unwritten*
      tail is ever sent to the stream, and ``_write_lock`` serialises all
      writers (the mount-time bootstrap and any concurrent
      append/set_content/finish call) so content set at construction and a
      call arriving immediately after can never be double-written.
    """

    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self._role = role
        self._content = content
        self._inner: Static | TUIMarkdown | None = None
        self._stream: MarkdownStream | None = None
        self._written_len = 0  # how much of self._content has reached the stream
        self._write_lock = asyncio.Lock()
        self._ready = asyncio.Event()  # set once the stream exists (post on_mount)
        self.add_class(role)

    def compose(self) -> ComposeResult:
        inner: Static | TUIMarkdown
        if self._role == "agent":
            md = TUIMarkdown("")
            md.code_dark_theme = "github-dark"  # type: ignore[attr-defined]
            inner = md
        else:
            inner = Static("")
        self._inner = inner
        yield inner

    def on_mount(self) -> None:
        self._refresh_content()
        if self._role == "agent":
            self.run_worker(self._bootstrap_stream(), exit_on_error=False)
        else:
            self._ready.set()

    async def _bootstrap_stream(self) -> None:
        """Create the stream and flush any content set before mount, then open the gate."""
        try:
            async with self._write_lock:
                await self._open_stream_locked()
                await self._write_pending_locked()
        except Exception:
            logger.exception("MessageBubble stream bootstrap failed — agent content may be stale")
        finally:
            self._ready.set()

    async def _open_stream_locked(self) -> None:
        """Create a fresh MarkdownStream and let its background task take its first step.

        A task cancelled before it has ever run doesn't reach the
        try/except around ``MarkdownStream._run``'s loop, so
        ``MarkdownStream.stop()`` re-raises ``CancelledError`` instead of
        absorbing it. The ``sleep(0)`` yield lets the task start so a later
        stop() (e.g. from ``finish()`` right after ``_bootstrap_stream``, with
        nothing ever written) is clean. Caller holds ``_write_lock``.
        """
        assert isinstance(self._inner, TUIMarkdown)
        self._stream = TUIMarkdown.get_stream(self._inner)
        await asyncio.sleep(0)

    async def _ensure_stream_locked(self) -> None:
        """Open a stream if one isn't already open. Caller holds ``_write_lock``."""
        if self._stream is None:
            await self._open_stream_locked()

    async def _write_pending_locked(self) -> None:
        """Send whatever of self._content hasn't reached the stream yet. Caller holds the lock."""
        if self._stream is None:
            return
        pending = self._content[self._written_len :]
        if pending:
            self._written_len = len(self._content)
            await self._stream.write(pending)

    def _refresh_content(self) -> None:
        inner = self._inner
        if inner is None:
            return
        role = self._role
        content = self._content
        if role == "user":
            assert isinstance(inner, Static)
            inner.update(Text.from_markup(content))
        elif role == "question":
            assert isinstance(inner, Static)
            try:
                inner.update(Text.from_markup(content))
            except Exception:
                inner.update(Text(content))
        # "agent" rendering is handled entirely by the MarkdownStream.

    async def append(self, chunk: str) -> None:
        if self._role != "agent":
            self._content += chunk
            self._refresh_content()
            return
        await self._ready.wait()
        async with self._write_lock:
            await self._ensure_stream_locked()
            self._content += chunk
            await self._write_pending_locked()

    async def set_content(self, text: str) -> None:
        if self._role != "agent":
            self._content = text
            self._refresh_content()
            return
        await self._ready.wait()
        async with self._write_lock:
            assert isinstance(self._inner, TUIMarkdown)
            # MarkdownStream only appends; a wholesale replace stops the
            # current stream and rewrites the widget directly. No fresh
            # stream is opened here — every current caller follows
            # set_content() with finish() and nothing else, so eagerly
            # reopening one would just be cancelled unused. append()
            # reopens lazily via _ensure_stream_locked() if it's ever
            # called after a set_content().
            if self._stream is not None:
                await self._stream.stop()
                self._stream = None
            self._content = text
            await self._inner.update(text)
            self._written_len = len(text)

    async def finish(self) -> None:
        if self._role != "agent":
            self._refresh_content()
            return
        await self._ready.wait()
        async with self._write_lock:
            await self._write_pending_locked()
            if self._stream is not None:
                await self._stream.stop()


class CompletionPopup(Static):
    """File-path completion list shown above the input bar when typing @references."""

    def show_matches(self, matches: list[str], selected: int) -> None:
        lines = []
        for i, m in enumerate(matches):
            safe_match = escape(m)
            if i == selected:
                lines.append(f"[bold {theme['accent']}]▸ {safe_match}[/bold {theme['accent']}]")
            else:
                lines.append(f"  [{theme['text_muted']}]{safe_match}[/{theme['text_muted']}]")
        self.update(Text.from_markup("\n".join(lines)))
        self.add_class("active")

    def hide(self) -> None:
        self.remove_class("active")
        self.update("")


class _InputHistory:
    """Keyboard-navigable history of submitted inputs.

    Index convention: -1 = not navigating (showing live input or draft).
    0 = most-recent entry, 1 = second-most-recent, etc.
    """

    def __init__(self) -> None:
        self._history: list[str] = []
        self._index: int = -1
        self._draft: str = ""

    def push(self, text: str) -> None:
        """Append *text* to history, ignoring consecutive duplicates."""
        if text and (not self._history or self._history[-1] != text):
            self._history.append(text)
        self._index = -1
        self._draft = ""

    def navigate(self, direction: int, current_value: str) -> str | None:
        """Return the entry to display after a navigation key press.

        *direction* is -1 for UP (older) and +1 for DOWN (newer).
        Returns the text to show, or None when the key has no effect.
        """
        if not self._history:
            return None
        if self._index == -1:
            if direction == 1:
                return None  # DOWN with no active navigation — nothing to do
            self._draft = current_value
            self._index = 0
        elif direction == -1:
            if self._index >= len(self._history) - 1:
                return None  # Already at the oldest entry
            self._index += 1
        else:
            self._index -= 1
            if self._index < 0:
                self._index = -1
                return self._draft
        return self._history[-(self._index + 1)]


class SmartInput(Input):
    """Input widget that converts multi-line paste events into a typed Pasted message.

    Single-line pastes pass through to the default Input handler unchanged.
    Multi-line pastes (text containing a newline) are intercepted and posted
    as ``SmartInput.Pasted`` so the controller can store them as a
    ``PasteAttachment`` and display a compact pill in the UI.

    Tab is intercepted here (at the focused-widget level) so it fires before
    the Screen's default ``focus_next`` binding, enabling completion cycling.
    Up/Down navigate the submission history when no completion popup is active.
    """

    BINDINGS = [
        Binding("tab", "tab_complete_forward", show=False),
    ]

    class Pasted(Message):
        """Posted when the user pastes multi-line text into the input."""

        def __init__(self, text: str) -> None:
            self._text = text
            super().__init__()

    class TabComplete(Message):
        """Posted when Tab is pressed to trigger slash-command or @file completion."""

        def __init__(self, direction: int = 1) -> None:
            self._direction = direction
            super().__init__()

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._input_history = _InputHistory()

    def push_history(self, text: str) -> None:
        """Store a submitted text in history."""
        self._input_history.push(text)

    def navigate_history(self, direction: int) -> bool:
        """Navigate history (direction=-1=UP older, +1=DOWN newer).

        Returns True when navigation occurred and the input value was updated.
        """
        result = self._input_history.navigate(direction, self.value)
        if result is None:
            return False
        self.value = result
        self.cursor_position = len(result)
        return True

    def action_tab_complete_forward(self) -> None:
        self.post_message(self.TabComplete(direction=1))

    def _on_paste(self, event: events.Paste) -> None:
        if "\n" in event.text:
            self.post_message(self.Pasted(event.text))
        else:
            super()._on_paste(event)


class AttachmentBar(Static):
    """Shows a paste-attachment pill above the input bar.

    Becomes visible (via the ``active`` CSS class) when a paste attachment is
    pending; hidden again on submission or Esc.
    """

    def show_pill(self, label: str) -> None:
        safe = escape(label)
        markup = (
            f"[bold {theme['accent']}]{safe}[/bold {theme['accent']}]"
            f"  [dim {theme['text_muted']}](Esc to discard)[/dim {theme['text_muted']}]"
        )
        self.update(Text.from_markup(markup))
        self.add_class("active")

    def hide(self) -> None:
        self.remove_class("active")
        self.update("")


class WorkspacePanel(Widget):
    """Scrollable file listing panel shown below the input bar for /ls-workspace.

    Up/Down to navigate, Escape or Ctrl+C to close.
    """

    BINDINGS = [
        Binding("ctrl+c", "close_panel", show=False),
        Binding("escape", "close_panel", show=False),
        Binding("up", "move_up", show=False),
        Binding("down", "move_down", show=False),
        Binding("home", "move_home", show=False),
        Binding("end", "move_end", show=False),
        Binding("pageup", "move_page_up", show=False),
        Binding("pagedown", "move_page_down", show=False),
    ]

    can_focus = True
    PAGE_SIZE = 12

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._files: list[str] = []
        self._selected: int = 0
        self._offset: int = 0

    def compose(self) -> ComposeResult:
        yield Static(id="workspace-panel-content")

    def show_files(self, files: list[str]) -> None:
        self._files = files
        self._selected = 0
        self._offset = 0
        self.add_class("active")
        self.focus()
        self._update_content()

    def _update_content(self) -> None:
        files = self._files
        content_widget = self.query_one("#workspace-panel-content", Static)
        if not files:
            content_widget.update(
                Text.from_markup(
                    f"[dim {theme['text_secondary']}]No files in active workspace.[/dim {theme['text_secondary']}]"
                )
            )
            return

        count = len(files)
        hint = (
            f"[dim {theme['text_secondary']}]"
            f"↑↓ navigate  PgUp/PgDn page  Home/End jump  Esc close  "
            f"{count} file{'s' if count != 1 else ''}"
            f"[/dim {theme['text_secondary']}]"
        )
        lines = [hint]
        visible = files[self._offset : self._offset + self.PAGE_SIZE]
        for i, name in enumerate(visible):
            abs_idx = self._offset + i
            if abs_idx == self._selected:
                lines.append(f"[bold {theme['accent']}]▸ {name}[/bold {theme['accent']}]")
            else:
                lines.append(f"  [{theme['text_secondary']}]{name}[/{theme['text_secondary']}]")
        if count > self.PAGE_SIZE:
            lo = self._offset + 1
            hi = min(self._offset + self.PAGE_SIZE, count)
            lines.append(
                f"[dim {theme['text_secondary']}]  {lo}–{hi} of {count}[/dim {theme['text_secondary']}]"
            )
        content_widget.update(Text.from_markup("\n".join(lines)))

    def action_move_up(self) -> None:
        if self._selected > 0:
            self._selected -= 1
            if self._selected < self._offset:
                self._offset = self._selected
            self._update_content()

    def action_move_down(self) -> None:
        if self._selected < len(self._files) - 1:
            self._selected += 1
            if self._selected >= self._offset + self.PAGE_SIZE:
                self._offset = self._selected - self.PAGE_SIZE + 1
            self._update_content()

    def action_move_home(self) -> None:
        if not self._files:
            return
        self._selected = 0
        self._offset = 0
        self._update_content()

    def action_move_end(self) -> None:
        if not self._files:
            return
        self._selected = len(self._files) - 1
        self._offset = max(0, len(self._files) - self.PAGE_SIZE)
        self._update_content()

    def action_move_page_up(self) -> None:
        if not self._files or self._selected == 0:
            return
        self._selected = max(0, self._selected - self.PAGE_SIZE)
        if self._selected < self._offset:
            self._offset = self._selected
        self._update_content()

    def action_move_page_down(self) -> None:
        if not self._files or self._selected >= len(self._files) - 1:
            return
        self._selected = min(len(self._files) - 1, self._selected + self.PAGE_SIZE)
        if self._selected >= self._offset + self.PAGE_SIZE:
            self._offset = self._selected - self.PAGE_SIZE + 1
        self._update_content()

    def action_close_panel(self) -> None:
        self.remove_class("active")
        self._files = []
        try:
            self.app.query_one("#user-input", Input).focus()
        except NoMatches:
            logger.debug("action_close_panel: #user-input not found, skipping focus")


class CommandApprovalPrompt(Widget):
    """Yes/no prompt asking the user to approve a command the agent wants to run.

    Shows the LLM-generated description of the command, then a heads-up warning
    (only when a potential negative impact was identified), then Yes / No
    options. Up/Down moves the selection, Enter confirms, Esc declines.
    Posts a ``Decision`` message with the outcome and freezes afterwards.
    """

    DEFAULT_CSS = """
    CommandApprovalPrompt {
        height: auto;
        padding: 0 2;
        margin-bottom: 1;
        border-left: thick $ods-warning;
    }
    """

    BINDINGS = [
        Binding("up", "move_up", show=False),
        Binding("down", "move_down", show=False),
        Binding("enter", "confirm", show=False),
        Binding("escape", "decline", show=False),
    ]

    can_focus = True

    _OPTIONS = ("Yes", "No")

    class Decision(Message):
        """Posted when the user confirms a choice (``approved`` True for Yes)."""

        def __init__(self, approved: bool) -> None:
            self.approved = approved
            super().__init__()

    def __init__(self, description: str, heads_up: str = "") -> None:
        super().__init__()
        self._description = description
        self._heads_up = heads_up
        self._selected = 0
        self._resolved = False

    def compose(self) -> ComposeResult:
        yield Static(id="approval-prompt-content")

    def on_mount(self) -> None:
        self.focus()
        self._refresh_content()

    def _refresh_content(self) -> None:
        lines = [
            f"[bold {theme['warning']}]Approval required[/bold {theme['warning']}]"
            f" — [{theme['text_primary']}]{escape(self._description)}[/{theme['text_primary']}]"
        ]
        if self._heads_up:
            lines.append(f"[{theme['warning']}]{escape(self._heads_up)}[/{theme['warning']}]")
        for idx, option in enumerate(self._OPTIONS):
            if self._resolved:
                if idx == self._selected:
                    lines.append(
                        f"[bold {theme['tool_done']}]✓ {option}[/bold {theme['tool_done']}]"
                    )
                continue
            if idx == self._selected:
                lines.append(f"[bold {theme['accent']}]▸ {option}[/bold {theme['accent']}]")
            else:
                lines.append(f"  [{theme['text_secondary']}]{option}[/{theme['text_secondary']}]")
        if not self._resolved:
            lines.append(
                f"[dim {theme['text_secondary']}]↑↓ select  Enter confirm  "
                f"Esc decline[/dim {theme['text_secondary']}]"
            )
        self.query_one("#approval-prompt-content", Static).update(
            Text.from_markup("\n".join(lines))
        )

    def action_move_up(self) -> None:
        if not self._resolved and self._selected > 0:
            self._selected -= 1
            self._refresh_content()

    def action_move_down(self) -> None:
        if not self._resolved and self._selected < len(self._OPTIONS) - 1:
            self._selected += 1
            self._refresh_content()

    def action_confirm(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._refresh_content()
        self.post_message(self.Decision(approved=self._selected == 0))

    def action_decline(self) -> None:
        if self._resolved:
            return
        self._selected = self._OPTIONS.index("No")
        self.action_confirm()


class ThinkingBlock(Static):
    """Ephemeral 'Thinking...' indicator shown while the LLM is processing.

    Displays a cycling dots animation (Thinking → Thinking. → Thinking.. →
    Thinking...) in a muted grey so it doesn't dominate the screen.  Call
    ``dismiss()`` to remove the block from the DOM.
    """

    DEFAULT_CSS = """
    ThinkingBlock {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._spin_idx = 0
        self._spin_timer: Timer | None = None

    def on_mount(self) -> None:
        self._spin_timer = self.set_interval(SPINNER_INTERVAL, self._tick)
        self._update_display()

    def _tick(self) -> None:
        self._spin_idx = (self._spin_idx + 1) % len(SPINNER)
        self._update_display()

    def _update_display(self) -> None:
        spin = SPINNER[self._spin_idx]
        self.update(
            Text.from_markup(
                f"[dim {theme['text_muted']}]{spin} Thinking[/dim {theme['text_muted']}]"
            )
        )

    def dismiss(self) -> None:
        self.remove()

    def finish(self, summary: str) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        self.update(
            Text.from_markup(f"[dim {theme['text_muted']}]{summary}[/dim {theme['text_muted']}]")
        )

    def on_unmount(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None


class PendingMessageBubble(Static):
    """Pinned indicator for a user message queued while the agent is busy.

    Stays visible (and unprocessed-looking) until the agent picks it up or
    the user cancels it via /cancel-all-messages or /cancel-message.
    """

    DEFAULT_CSS = """
    PendingMessageBubble {
        height: auto;
        padding: 0 2;
        margin-bottom: 1;
        background: $ods-warning-bg;
        border-left: thick $ods-warning;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__("")
        self._text = text

    def on_mount(self) -> None:
        self.update(
            Text.from_markup(
                f"[bold {theme['warning']}]Queued[/bold {theme['warning']}]  {self._text}"
            )
        )


class PendingMessagePanel(Vertical):
    """Holds pinned PendingMessageBubble widgets between the chat and input bar.

    New bubbles are mounted at the top of the stack, closest to the live
    conversation, so the most recently queued message is the most visible.
    """

    DEFAULT_CSS = """
    PendingMessagePanel {
        height: auto;
        max-height: 8;
        overflow-y: auto;
    }
    """

    def add_pending(self, text: str) -> "PendingMessageBubble":
        bubble = PendingMessageBubble(text)
        self.mount(bubble, before=0)
        return bubble


class MessagesContainer(ScrollableContainer):
    """Message-history scroller with a releasable bottom anchor.

    Delegates entirely to Textual's built-in scroll anchoring
    (``Widget.anchor()``): while armed, the compositor re-pins the view to
    the bottom on every layout pass (which is what actually solves the
    "reflow after an async Markdown update" problem — no polling timer
    needed). Any user-initiated scroll releases the anchor automatically;
    scrolling back down to the bottom re-arms it. See
    ``textual.widget.Widget.anchor``/``release_anchor``/``is_anchored``.
    """

    def on_mount(self) -> None:
        self.anchor()


class ChatPane(Widget):
    """Left pane: scrollable message history + input bar."""

    def compose(self) -> ComposeResult:
        yield MessagesContainer(id="messages")
        yield PendingMessagePanel(id="pending-panel")
        with Vertical(id="input-bar"):
            yield CompletionPopup(id="completion-popup")
            yield AttachmentBar(id="attachment-bar")
            yield SmartInput(
                placeholder="Ask anything… (/ for commands, @ to attach files)",
                id="user-input",
                highlighter=CommandHighlighter(),
            )
        with Horizontal(id="status-bar"):
            yield TipsBar()
        yield WorkspacePanel(id="workspace-panel")

    def _mount_in_messages(self, widget: Widget) -> None:
        """Mount *widget* in #messages.

        The view follows new content while Textual's scroll anchor is armed
        (see ``MessagesContainer``); a user who scrolled up keeps their
        position.
        """
        container = self.query_one("#messages", MessagesContainer)
        container.mount(widget)

    def add_message(self, role: str, content: str = "") -> MessageBubble:
        bubble = MessageBubble(role, content)
        self._mount_in_messages(bubble)
        return bubble

    def add_divider(self) -> None:
        divider = Static(Rule(style=theme["separator"]), classes="msg-divider")
        self._mount_in_messages(divider)

    def add_turn_status_bar(self) -> "TurnStatusBar":
        for existing in self.query(TurnStatusBar):
            existing.remove()
        timer = TurnStatusBar()
        self.query_one("#status-bar", Horizontal).mount(timer)
        return timer

    def add_pending_message(self, text: str) -> "PendingMessageBubble":
        return self.query_one("#pending-panel", PendingMessagePanel).add_pending(text)

    def add_thinking_block(self) -> "ThinkingBlock":
        block = ThinkingBlock()
        self._mount_in_messages(block)
        return block

    def add_ephemeral_block(self, communication: str, label: str, summary: str) -> "ToolCallBlock":
        widget = ToolCallBlock(communication, label, summary)
        self._mount_in_messages(widget)
        return widget

    def add_task_block(self, communication: str, task_summaries: list[str]) -> "ToolCallBlock":
        widget = ToolCallBlock(communication, "", "", task_summaries=task_summaries)
        self._mount_in_messages(widget)
        return widget

    def show_workspace_panel(self, files: list[str]) -> None:
        self.query_one("#workspace-panel", WorkspacePanel).show_files(files)

    def show_approval_prompt(self, description: str, heads_up: str) -> "CommandApprovalPrompt":
        widget = CommandApprovalPrompt(description, heads_up)
        self._mount_in_messages(widget)
        return widget

    def show_attachment(self, label: str) -> None:
        self.query_one("#attachment-bar", AttachmentBar).show_pill(label)

    def hide_attachment(self) -> None:
        self.query_one("#attachment-bar", AttachmentBar).hide()

    def clear_messages(self) -> None:
        self.query_one("#messages", ScrollableContainer).remove_children()


class ToolCallBlock(Static):
    """Ephemeral status block: optional communication line + tool status line(s).

    Shows blue while the tool is running; call ``set_done()`` to turn green.
    Call ``dismiss()`` to remove from the DOM entirely.
    For ``task``, pass ``task_summaries`` to get one status line per worker.
    Worker rows can be individually marked done (green ✓) or error (red ✗).

    When both ``label`` and ``summary`` are empty the block is
    *communication-only* (used for hidden ``display_status=False`` tools): the
    narration is the whole block — spinner-prefixed while running, plain text
    once finished — and no tool-status line ever appears.
    """

    DEFAULT_CSS = """
    ToolCallBlock {
        height: auto;
        padding: 0;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        communication: str,
        label: str,
        summary: str,
        task_summaries: list[str] | None = None,
    ) -> None:
        super().__init__("")
        self._communication: str | None = communication
        self._label = label
        self._summary = summary
        self._task_summaries = task_summaries or []
        # Per-worker three-state status: "running" | "done" | "error"
        self._task_statuses: list[str] = ["running"] * len(self._task_summaries)
        # Current tool name / activity string shown inline for running workers.
        self._task_activities: list[str] = [""] * len(self._task_summaries)
        self._done = False
        self._error = False
        self._spin_idx = 0
        self._spin_timer: Timer | None = None

    def on_mount(self) -> None:
        if not self._done:
            self._spin_timer = self.set_interval(SPINNER_INTERVAL, self._tick)
        self._refresh()

    def _tick(self) -> None:
        self._spin_idx = (self._spin_idx + 1) % len(SPINNER)
        self._refresh()

    def _status_markup(self, text: str, done: bool | None = None) -> str:
        safe_text = escape(text)
        if self._error:
            return f"[bold {theme['error']}]✗ {safe_text}[/bold {theme['error']}]"
        is_done = self._done if done is None else done
        if is_done:
            return f"[bold {theme['tool_done']}]{safe_text}[/bold {theme['tool_done']}]"
        spin = SPINNER[self._spin_idx]
        return f"[bold {theme['tool_running']}]{spin} {safe_text}[/bold {theme['tool_running']}]"

    def _task_status_markup(self, text: str, status: str, prefix: str = "") -> str:
        """Return markup for a single worker row based on its status.

        ``prefix`` is placed inside the bold/colour span so it inherits the
        row's status colour (running/done/error)."""
        safe_text = escape(text)
        if status == "error":
            return f"[bold {theme['error']}]{prefix}✗ {safe_text}[/bold {theme['error']}]"
        if status == "done":
            return f"[bold {theme['tool_done']}]{prefix}{safe_text}[/bold {theme['tool_done']}]"
        spin = SPINNER[self._spin_idx]
        return f"[bold {theme['tool_running']}]{prefix}{spin} {safe_text}[/bold {theme['tool_running']}]"

    def _refresh(self) -> None:
        lines: list[str] = []
        if self._task_summaries:
            all_terminal = all(s != "running" for s in self._task_statuses)
            if self._communication:
                lines.append(escape(self._communication))
                lines.append("")
            lines.append(self._status_markup("Parallelizing", done=self._done or all_terminal))
            for i, s in enumerate(self._task_summaries):
                if self._done:
                    # Force-done: keep terminal rows as-is; promote any still-running row to
                    # "error" if the block itself errored, otherwise to "done".
                    cur = self._task_statuses[i]
                    if cur == "running":
                        st = "error" if self._error else "done"
                    else:
                        st = cur
                else:
                    st = self._task_statuses[i]
                activity = self._task_activities[i] if i < len(self._task_activities) else ""
                label = f"Worker {i + 1}: {activity if activity and st == 'running' else s}"
                display = self._task_status_markup(label, st, prefix="  └─ ")
                lines.append(display)
        else:
            display = self._summary if self._summary else self._label
            if self._communication and display:
                lines.append(escape(self._communication))
                lines.append("")  # blank line so the gap matches the inter-block margin
                lines.append(self._status_markup(display))
            elif self._communication:
                # Communication-only block (hidden tool): the narration is the
                # whole block — spinner while running, plain text once done.
                if self._done and not self._error:
                    lines.append(escape(self._communication))
                else:
                    lines.append(self._status_markup(self._communication))
            else:
                lines.append(self._status_markup(display))
        self.update(Text.from_markup("\n".join(lines)))

    def _stop_spinner(self) -> None:
        """Stop the spinner timer (called when all workers reach a terminal state)."""
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None

    def mark_task_done(self, idx: int) -> None:
        if 0 <= idx < len(self._task_statuses):
            self._task_statuses[idx] = "done"
            if idx < len(self._task_activities):
                self._task_activities[idx] = ""
        if all(s != "running" for s in self._task_statuses):
            self._stop_spinner()
            self._done = True
        self._refresh()

    def mark_task_error(self, idx: int) -> None:
        if 0 <= idx < len(self._task_statuses):
            self._task_statuses[idx] = "error"
            if idx < len(self._task_activities):
                self._task_activities[idx] = ""
        if all(s != "running" for s in self._task_statuses):
            self._stop_spinner()
            self._done = True
        self._refresh()

    def update_task_activity(self, idx: int, activity: str) -> None:
        """Update the inline activity label for a running worker row."""
        if 0 <= idx < len(self._task_activities) and self._task_statuses[idx] == "running":
            if self._task_activities[idx] == activity:
                return
            self._task_activities[idx] = activity
            self._refresh()

    def set_communication(self, text: str | None) -> None:
        """Update the communication line while the tool's args are still streaming."""
        self._communication = text
        self._refresh()

    def upgrade(self, label: str, summary: str) -> None:
        """Replace the generic pending label with the real tool label/summary once tool_call fires."""
        self._label = label
        self._summary = summary
        self._refresh()

    def set_done(self) -> None:
        self._done = True
        self._stop_spinner()
        self._refresh()

    def set_error(self) -> None:
        self._error = True
        self._done = True
        self._stop_spinner()
        self._refresh()

    def is_running(self) -> bool:  # type: ignore[override]
        return not self._done

    def on_unmount(self) -> None:
        self._stop_spinner()

    def dismiss(self) -> None:
        self.remove()
