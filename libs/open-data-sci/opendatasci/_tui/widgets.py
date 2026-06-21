"""Textual widgets for OpenDataSci TUI v2."""

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
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Input, Static
from textual.widgets import Markdown as TUIMarkdown

try:
    from textual.widgets import Image as _TUIImage  # type: ignore[attr-defined]
except ImportError:
    _TUIImage = None

from .adapter import EphemeralHandle, MessageHandle, ThinkingHandle, TurnStatusHandle
from .commands import SLASH_COMMANDS, _fmt_model
from .models import SPINNER, SPINNER_INTERVAL
from .theme import active as theme

logger = logging.getLogger(__name__)

_BUBBLE_FLUSH_INTERVAL = 0.25  # seconds — caps Markdown rebuilds at 4/sec during streaming


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
    AppHeader {
        dock: top;
        height: 5;
        background: #0d1117;
        border-bottom: solid #1a2030;
    }
    #header-layout { layout: horizontal; height: 5; }
    #header-logo {
        width: 21;
        height: 5;
        content-align: center middle;
    }
    """

    def __init__(
        self,
        version: str,
        provider: str,
        model: str,
        workspace: str,
        workspace_name: str | None = None,
    ) -> None:
        super().__init__()
        self._version = version
        self._provider = provider
        self._model = model
        self._workspace = workspace
        self._workspace_name = workspace_name
        self._file_count: str = ""
        _logo_path = Path(__file__).parents[4] / "docs" / "logo.png"
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
        t.append("\n")
        t.append("Model      ", style=lbl)
        t.append(_fmt_model(self._provider, self._model), style=theme["text_primary"])
        self.query_one("#header-info", Static).update(t)

    def set_workspace(self, name: str | None) -> None:
        self._workspace_name = name
        self._render_info()

    def set_file_count(self, description: str) -> None:
        self._file_count = description
        self._render_info()


class TurnStatusBar(Static):
    """Inline status bar appended at the end of the conversation during an agent turn."""

    DEFAULT_CSS = """
    TurnStatusBar {
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
        k = math.floor(n / 100) / 10
        return f"{k:.1f}k"

    def _context_suffix(self) -> str:
        if self._context_tokens is None:
            return ""
        size = self._fmt_tokens(self._context_tokens)
        if self._cached_tokens is None:
            return f" | Context: {size} tokens"
        pct = math.ceil(self._cached_tokens / max(self._context_tokens, 1) * 1000) / 10
        return f" | Context: {size} tokens ({pct:.1f}% cached)"

    def _tick(self) -> None:
        s = int(time.monotonic() - self._start)
        label = f"Working for {self._fmt(s)}{self._context_suffix()}"
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
        label = f"Worked for {self._fmt(s)}{self._context_suffix()}"
        self.update(f"[{theme['text_muted']}]{label}[/{theme['text_muted']}]")

    def on_unmount(self) -> None:
        # Guard against removal before stop() was explicitly called, and also
        # against removal before on_mount() ever fired (_mounted is False).
        if self._mounted and not self._stopped:
            self._stopped = True
            if self._interval is not None:
                self._interval.stop()


class MessageBubble(Widget):
    """A single chat message — user, agent (streaming), or thinking."""

    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self._role = role
        self._content = content
        self._spin_idx = 0
        self._spin_label: str = "Thinking"
        self._spin_timer: Timer | None = None
        self._inner: Static | TUIMarkdown | None = None
        self._summary_text: str | None = None
        self._flush_timer: Timer | None = None  # rate-limit Markdown rebuilds
        self._dirty: bool = False
        self._flush_scheduled: bool = False  # at most one call_after_refresh pending
        self.add_class(role)

    def compose(self) -> ComposeResult:
        inner: Static | TUIMarkdown
        if self._role == "agent":
            # Always start with an empty Markdown widget. All rendering goes
            # through _flush_agent so there is only ever one update() task in
            # flight, avoiding the race where TUIMarkdown._on_mount's own
            # update() task and _flush_agent's update() task both mount content
            # and produce duplicate text.
            md = TUIMarkdown("")
            md.code_dark_theme = "github-dark"  # type: ignore[attr-defined]
            inner = md
        else:
            inner = Static("")
        self._inner = inner
        yield inner

    def on_mount(self) -> None:
        self._spin_timer = (
            self.set_interval(SPINNER_INTERVAL, self._spin_tick)
            if self._role == "thinking"
            else None
        )
        self._refresh_content()
        # If the bubble already has content (set before mount completed) mark it
        # dirty so the flush below will render it.  This covers the common
        # pattern add_message("agent", text).finish() where finish() runs before
        # compose/on_mount, as well as the response-event path where the full
        # answer is passed directly to the constructor.
        if self._role == "agent":
            if self._content:
                self._dirty = True
            if self._dirty:
                self._flush_scheduled = False  # force a new call even if one was pending
                self._schedule_final_flush()

    def _spin_tick(self) -> None:
        self._spin_idx = (self._spin_idx + 1) % len(SPINNER)
        self._refresh_content()

    def _refresh_content(self) -> None:
        inner = self._inner
        if inner is None:
            return
        role = self._role
        content = self._content
        if role == "agent":
            # Agent rendering happens in two places:
            #   1. compose() seeds the Markdown widget with self._content so
            #      content present at construction (or accumulated before
            #      compose ran) is rendered automatically by Markdown's own
            #      mount hook.
            #   2. _flush_agent() applies subsequent updates (streaming tokens,
            #      set_content, finish) at most 10 times per second.
            # No work to do from _refresh_content itself.
            pass
        elif role == "user":
            assert isinstance(inner, Static)
            inner.update(Text.from_markup(content))
        elif role == "thinking":
            assert isinstance(inner, Static)
            if self._summary_text is not None:
                inner.update(
                    Text.from_markup(
                        f"[dim {theme['thinking']}]✓ {self._summary_text}[/dim {theme['thinking']}]"
                    )
                )
            elif content:
                # Debug mode: actual reasoning text streamed in — stop the
                # spinner and render the accumulated content in muted grey.
                if self._spin_timer is not None:
                    self._spin_timer.stop()
                    self._spin_timer = None
                inner.update(
                    Text.from_markup(
                        f"[bold {theme['text_muted']}]Reasoning:[/bold {theme['text_muted']}]"
                        f"[{theme['text_muted']}] {escape(content)}[/{theme['text_muted']}]"
                    )
                )
            else:
                spin = SPINNER[self._spin_idx]
                inner.update(
                    Text.from_markup(
                        f"[bold {theme['thinking']}]{spin} {self._spin_label}…[/bold {theme['thinking']}]"
                    )
                )
        elif role == "question":
            assert isinstance(inner, Static)
            try:
                inner.update(Text.from_markup(content))
            except Exception:
                inner.update(Text(content))

    def _schedule_final_flush(self) -> None:
        """Schedule one awaited Markdown rebuild via call_after_refresh (deduped)."""
        if not self._flush_scheduled:
            self._flush_scheduled = True
            self.call_after_refresh(self._flush_agent)

    def set_content(self, text: str) -> None:
        self._content = text
        if self._role == "agent":
            self._stop_flush_timer()
            self._dirty = True
            self._schedule_final_flush()
        else:
            self._refresh_content()

    def append(self, chunk: str) -> None:
        self._content += chunk
        if self._role == "agent":
            # Buffer tokens; the flush timer does a single awaited rebuild at ~10 Hz.
            self._dirty = True
            if self._flush_timer is None:
                self._flush_timer = self.set_interval(_BUBBLE_FLUSH_INTERVAL, self._flush_agent)
        else:
            self._refresh_content()

    def _stop_flush_timer(self) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None

    async def _flush_agent(self) -> None:
        """Flush buffered tokens to the Markdown widget (called at most 10×/sec).

        Awaiting inner.update() serialises rebuilds so they can never race each
        other.

        If the bubble is not yet fully mounted (``_inner`` is None or
        ``is_mounted`` is False), this returns early WITHOUT clearing
        ``_dirty``.  ``on_mount`` then schedules another flush once mount has
        completed, guaranteeing the buffered content eventually renders.

        Any exception is caught here rather than propagated: if Textual's
        Markdown widget raises (e.g. during the async DOM operations inside
        update()) the exception would otherwise travel through on_idle →
        _handle_exception and exit the entire app.  We log and swallow it so
        the app stays alive; the worst case is stale rendered content.
        """
        try:
            self._flush_scheduled = False
            if not self._dirty:
                return
            inner = self._inner
            if inner is None or not self.is_mounted:
                # Leave _dirty=True so on_mount or the streaming timer retries.
                return
            assert isinstance(inner, TUIMarkdown)
            # Snapshot content and clear _dirty BEFORE the await.  If any
            # append()/set_content()/finish() call arrives while
            # inner.update() is suspended, it will set _dirty=True again and
            # the next tick (or the final flush scheduled by finish()) will
            # pick up the newer content.  Clearing _dirty after the await
            # would overwrite that flag and silently drop those tokens.
            content = self._content
            self._dirty = False
            try:
                await inner.update(content)  # type: ignore[misc]
            except Exception:
                self._dirty = True  # render failed; retry on the next tick
                raise
        except Exception:
            logger.exception("_flush_agent failed — bubble content may be stale")

    def finish(self) -> None:
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        if self._role == "thinking" and self._summary_text is None:
            self._summary_text = "Done thinking"
        if self._role == "agent":
            self._dirty = True
            self._stop_flush_timer()
            # Bypass the dedup guard: finish() is the authoritative "done" signal
            # and must always schedule a render regardless of prior pending flushes.
            self._flush_scheduled = False
            self._schedule_final_flush()
        else:
            self._refresh_content()

    def finish_with_summary(self, text: str) -> None:
        """Stop the spinner and replace the bubble content with a static summary.

        Designed for thinking bubbles: transforms the animated "Thinking…" into
        a collapsed dim line like "✓ Thought for 12s".
        """
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        self._summary_text = text
        inner = self._inner
        if inner is None:
            return
        assert isinstance(inner, Static)
        inner.update(
            Text.from_markup(f"[dim {theme['thinking']}]✓ {text}[/dim {theme['thinking']}]")
        )


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
            f"[bold #58a6ff]📎 {safe}[/bold #58a6ff]"
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
        except Exception:
            pass


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

    _DOTS = ["", ".", "..", "..."]

    def __init__(self) -> None:
        super().__init__("")
        self._dot_idx = 0
        self._spin_timer: Timer | None = None

    def on_mount(self) -> None:
        self._spin_timer = self.set_interval(0.35, self._tick)
        self._update_display()

    def _tick(self) -> None:
        self._dot_idx = (self._dot_idx + 1) % len(self._DOTS)
        self._update_display()

    def _update_display(self) -> None:
        dots = self._DOTS[self._dot_idx]
        self.update(
            Text.from_markup(
                f"[dim {theme['text_muted']}]💭 Thinking{dots}[/dim {theme['text_muted']}]"
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


class ChatPane(Widget):
    """Left pane: scrollable message history + input bar."""

    def compose(self) -> ComposeResult:
        yield ScrollableContainer(id="messages")
        with Vertical(id="input-bar"):
            yield CompletionPopup(id="completion-popup")
            yield AttachmentBar(id="attachment-bar")
            yield SmartInput(
                placeholder="Ask anything… (/ for commands, @ to attach files)",
                id="user-input",
                highlighter=CommandHighlighter(),
            )
        yield WorkspacePanel(id="workspace-panel")

    def add_message(self, role: str, content: str = "") -> MessageBubble:
        bubble = MessageBubble(role, content)
        self.query_one("#messages", ScrollableContainer).mount(bubble)
        return bubble

    def add_divider(self) -> None:
        divider = Static(Rule(style=theme["separator"]), classes="msg-divider")
        self.query_one("#messages", ScrollableContainer).mount(divider)

    def add_turn_status_bar(self) -> "TurnStatusBar":
        for existing in self.query(TurnStatusBar):
            existing.remove()
        timer = TurnStatusBar()
        self.mount(timer, after=self.query_one("#input-bar"))
        return timer

    def add_thinking_block(self) -> "ThinkingBlock":
        block = ThinkingBlock()
        self.query_one("#messages", ScrollableContainer).mount(block)
        return block

    def add_ephemeral_block(self, communication: str, label: str, summary: str) -> "ToolCallBlock":
        widget = ToolCallBlock(communication, label, summary)
        self.query_one("#messages", ScrollableContainer).mount(widget)
        return widget

    def add_worker_block(self, communication: str, worker_summaries: list[str]) -> "ToolCallBlock":
        widget = ToolCallBlock(communication, "", "", worker_summaries=worker_summaries)
        self.query_one("#messages", ScrollableContainer).mount(widget)
        return widget

    def show_workspace_panel(self, files: list[str]) -> None:
        self.query_one("#workspace-panel", WorkspacePanel).show_files(files)

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
    For ``spawn_workers``, pass ``worker_summaries`` to get one status line per worker.
    Worker rows can be individually marked done (green ✓) or error (red ✗).
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
        worker_summaries: list[str] | None = None,
    ) -> None:
        super().__init__("")
        self._communication: str | None = communication
        self._label = label
        self._summary = summary
        self._worker_summaries = worker_summaries or []
        # Per-worker three-state status: "running" | "done" | "error"
        self._worker_statuses: list[str] = ["running"] * len(self._worker_summaries)
        # Current tool name / activity string shown inline for running workers.
        self._worker_activities: list[str] = [""] * len(self._worker_summaries)
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

    def _worker_status_markup(self, text: str, status: str, prefix: str = "") -> str:
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
        if self._worker_summaries:
            all_terminal = all(s != "running" for s in self._worker_statuses)
            if self._communication:
                lines.append(escape(self._communication))
                lines.append("")
            lines.append(self._status_markup("⚡ Parallelizing", done=self._done or all_terminal))
            for i, s in enumerate(self._worker_summaries):
                if self._done:
                    # Force-done: keep terminal rows as-is; promote any still-running row to
                    # "error" if the block itself errored, otherwise to "done".
                    cur = self._worker_statuses[i]
                    if cur == "running":
                        st = "error" if self._error else "done"
                    else:
                        st = cur
                else:
                    st = self._worker_statuses[i]
                activity = self._worker_activities[i] if i < len(self._worker_activities) else ""
                label = f"Worker {i + 1}: {activity if activity and st == 'running' else s}"
                display = self._worker_status_markup(label, st, prefix="  └─ ")
                lines.append(display)
        else:
            display = self._summary if self._summary else self._label
            if self._communication:
                lines.append(escape(self._communication))
                lines.append("")  # blank line so the gap matches the inter-block margin
            lines.append(self._status_markup(display))
        self.update(Text.from_markup("\n".join(lines)))

    def _stop_spinner(self) -> None:
        """Stop the spinner timer (called when all workers reach a terminal state)."""
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None

    def mark_worker_done(self, idx: int) -> None:
        if 0 <= idx < len(self._worker_statuses):
            self._worker_statuses[idx] = "done"
            if idx < len(self._worker_activities):
                self._worker_activities[idx] = ""
        if all(s != "running" for s in self._worker_statuses):
            self._stop_spinner()
            self._done = True
        self._refresh()

    def mark_worker_error(self, idx: int) -> None:
        if 0 <= idx < len(self._worker_statuses):
            self._worker_statuses[idx] = "error"
            if idx < len(self._worker_activities):
                self._worker_activities[idx] = ""
        if all(s != "running" for s in self._worker_statuses):
            self._stop_spinner()
            self._done = True
        self._refresh()

    def update_worker_activity(self, idx: int, activity: str) -> None:
        """Update the inline activity label for a running worker row."""
        if 0 <= idx < len(self._worker_activities) and self._worker_statuses[idx] == "running":
            if self._worker_activities[idx] == activity:
                return
            self._worker_activities[idx] = activity
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


# Register Textual widget implementations as virtual subclasses of their ABCs.
# Direct inheritance is not possible due to a metaclass conflict between
# Textual's _MessagePumpMeta and ABCMeta.
MessageHandle.register(MessageBubble)
EphemeralHandle.register(ToolCallBlock)
TurnStatusHandle.register(TurnStatusBar)
ThinkingHandle.register(ThinkingBlock)
