"""CLIController — application state and event routing for the OpenDataSci TUI.

Concerns deliberately kept here:
  - Lifecycle (boot, close)
  - Input routing (on_input_changed, on_submit)
  - Slash-command dispatch
  - Choice-prompt state machine
  - Action methods (reset, clear, compact, show_models, show_help, stop, ls_workspace)

Everything else has been extracted into focused sibling modules:
  - adapter.py   — UIAdapter + handle ABCs
  - commands.py  — SLASH_COMMANDS registry + display formatters
  - completion.py — CompletionState (tab-completion logic)
  - file_refs.py  — @file-ref parsing helpers
  - presenter.py  — _TurnPresenter (streaming event dispatch)
"""

import difflib
import logging
import string
from contextlib import AsyncExitStack
from pathlib import Path

from rich.markup import escape as escape_markup

from opendatasci._tui.service import OpenDataSciTuiService
from opendatasci._tui.session import CLISessionInfo
from opendatasci.agents.agents_factory import create_agent
from opendatasci.configs import OpenDataSciConfig
from opendatasci.streaming import BaseAgentStreamEvent
from opendatasci.streaming.events import (
    ErrorEvent,
    InputRequiredEvent,
    ReasoningEvent,
    ResponseEvent,
    SubagentEvent,
    TokenEvent,
    ToolCallEvent,
    ToolCommunicationEvent,
    ToolResultEvent,
    UsageEvent,
    WorkerDoneEvent,
)
from opendatasci.tools.mcp import load_mcp_servers

from . import theme as _theme
from .adapter import (
    PendingMessageHandle,
    TurnStatusHandle,
    UIAdapter,
)
from .commands import (
    format_help_message,
    format_models_message,
    format_themes_message,
)
from .completion import CompletionState
from .file_refs import (
    PasteAttachment,
    _build_agent_query,
    _build_user_display,
    _parse_file_refs,
    _split_existing_file_refs,
)
from .message_queue import PendingMessageQueue
from .presenter import _TurnPresenter
from .theme import active as theme

logger = logging.getLogger(__name__)


class CLIController:
    """Owns application state and all non-Textual business logic for the TUI."""

    def __init__(
        self,
        ui: UIAdapter,
        workspace_path: str,
        datasci_config: OpenDataSciConfig,
        session_id: str,
        completion: CompletionState | None = None,
    ) -> None:
        self._ui = ui
        self._workspace_path = workspace_path
        self._base_config = datasci_config
        self._session_id = session_id
        self._service: OpenDataSciTuiService | None = None
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._awaiting_choice: bool = False
        self._pending_choices: list[str] = []
        self._other_choice_label: str | None = None
        self._awaiting_custom_choice_input: bool = False
        self._active_turn_status: TurnStatusHandle | None = None
        self._agent_running: bool = False
        self._pending_queue = PendingMessageQueue()
        self._pending_handles: dict[int, PendingMessageHandle] = {}
        self._cfg: OpenDataSciConfig | None = None
        self._completion = (
            completion if completion is not None else CompletionState(extra_commands=[])
        )
        self._paste_attachment: PasteAttachment | None = None

    @property
    def provider(self) -> str:
        return self._base_config.provider

    @property
    def model(self) -> str:
        return self._base_config.model

    # ── Completion state delegation ───────────────────────────────────────────
    # These properties expose CompletionState internals under the names that
    # existed on CLIController before the extraction, so that existing tests
    # and any external callers that relied on the old attribute names keep
    # working without modification.

    @property
    def _completing(self) -> bool:
        return self._completion._completing

    @_completing.setter
    def _completing(self, value: bool) -> None:
        self._completion._completing = value

    @property
    def _comp_matches(self) -> list[str]:
        return self._completion._matches

    @_comp_matches.setter
    def _comp_matches(self, value: list[str]) -> None:
        self._completion._matches = value

    @property
    def _comp_displays(self) -> list[str]:
        return self._completion._displays

    @_comp_displays.setter
    def _comp_displays(self, value: list[str]) -> None:
        self._completion._displays = value

    @property
    def _comp_idx(self) -> int:
        return self._completion._idx

    @_comp_idx.setter
    def _comp_idx(self, value: int) -> None:
        self._completion._idx = value

    @property
    def _comp_at_pos(self) -> int:
        return self._completion._at_pos

    @_comp_at_pos.setter
    def _comp_at_pos(self, value: int) -> None:
        self._completion._at_pos = value

    @property
    def _comp_mode(self) -> str:
        return self._completion._mode

    @_comp_mode.setter
    def _comp_mode(self, value: str) -> None:
        self._completion._mode = value

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release the agent sandbox and any other resources held by the controller."""
        if self._service is not None:
            await self._service.close()
        await self._exit_stack.aclose()

    # ── Boot ──────────────────────────────────────────────────────────────────

    async def boot(self) -> None:
        ui = self._ui

        try:
            resolved_path = Path(self._workspace_path).resolve()
            config_search_path = resolved_path if resolved_path.is_dir() else resolved_path.parent
            mcp_servers = load_mcp_servers(config_search_path)

            cfg = self._base_config.model_copy(update={"mcp_servers": mcp_servers})
            self._cfg = cfg

            agent = await self._exit_stack.enter_async_context(
                create_agent(self._workspace_path, config=cfg)
            )
            workspace_path = Path(agent._workspace.get_reference())
            self._service = OpenDataSciTuiService(
                agent=agent,
                sandbox=agent._sandbox,
                workspace_path=workspace_path,
            )

            info = CLISessionInfo.from_path(self._workspace_path, workspace_path, cfg)
            ui.set_file_count(self._describe_data(info))
        except FileNotFoundError:
            hint = self._did_you_mean(self._workspace_path)
            msg_text = (
                f"❌ File not found: `{escape_markup(self._workspace_path)}`\n\n"
                f"Check the path and try again.{hint}"
            )
            msg = ui.add_message("agent", "")
            msg.set_content(msg_text)
            msg.finish()
        except PermissionError:
            msg = ui.add_message("agent", "")
            msg.set_content(f"❌ Permission denied: `{escape_markup(self._workspace_path)}`")
            msg.finish()
        except ValueError as exc:
            msg = ui.add_message("agent", "")
            msg.set_content(f"❌ Provider error: {exc}")
            msg.finish()
        except Exception as exc:
            msg = ui.add_message("agent", "")
            msg.set_content(f"❌ Failed to load: {exc}")
            msg.finish()

    @staticmethod
    def _did_you_mean(workspace_path: str) -> str:
        """Return a 'Did you mean …?' hint if a close filename exists in the same dir."""
        p = Path(workspace_path)
        try:
            siblings = [child.name for child in p.parent.iterdir()]
        except OSError:
            return ""
        close = difflib.get_close_matches(p.name, siblings, n=1, cutoff=0.6)
        if close:
            return f"\n\nDid you mean `{p.parent / close[0]}`?"
        return ""

    @staticmethod
    def _describe_data(info: object) -> str:
        """Derive a short human-readable description of the loaded data."""
        if getattr(info, "is_directory", False):
            count = getattr(info, "workspace_count", 0)
            return f"{count} file{'s' if count != 1 else ''}"
        return ""

    # ── Input change ──────────────────────────────────────────────────────────

    def on_input_changed(self, value: str) -> bool:
        """Handle input text change.

        Returns ``True`` when the change was a programmatic tab-completion
        update (caller should skip further processing).
        """
        return self._completion.on_input_changed(value, self._ui)

    # ── Tab completion ────────────────────────────────────────────────────────

    @property
    def has_completion_matches(self) -> bool:
        """True when the completion popup currently has items to navigate."""
        return self._completion.has_matches

    def cycle_completion(self, current_value: str, direction: int) -> bool:
        """Cycle completion selection up/down while the popup is visible."""
        return self._completion.cycle(current_value, direction=direction, ui=self._ui)

    def hide_completion(self) -> None:
        self._completion.hide(self._ui)

    # ── Paste attachment ──────────────────────────────────────────────────────

    def on_paste(self, text: str) -> None:
        """Store a multi-line paste as an attachment and show the pill in the UI."""
        self._paste_attachment = PasteAttachment(text)
        self._ui.show_attachment(self._paste_attachment.display_label)

    def clear_paste_attachment(self) -> None:
        """Discard the current paste attachment (Esc handler) and hide the bar."""
        if self._paste_attachment is not None:
            self._paste_attachment = None
            self._ui.hide_attachment()

    # ── Submit ────────────────────────────────────────────────────────────────

    async def on_submit(self, raw: str) -> tuple[str, str]:
        """Handle input submission.

        Returns ``(action, payload)`` where *action* is one of:
        - ``"run"``  — caller should run the agent with *payload* as the query
        - ``"quit"`` — caller should exit
        - ``""``     — action handled internally, nothing more to do
        """
        self.hide_completion()

        # Always capture and clear the paste attachment at the start of
        # submission so it is never accidentally carried into the next turn.
        attachment = self._paste_attachment
        self._paste_attachment = None
        self._ui.hide_attachment()

        if self._awaiting_choice:
            if not raw:
                return "", ""
            if raw in {"/exit", "/reset", "/clear"}:
                self._exit_choice_mode()
                should_quit = await self._handle_slash(raw)
                return ("quit" if should_quit else ""), ""
            answer = self._handle_user_choice(raw)
            if answer is not None:
                return "run", answer
            return "", ""

        if not raw and attachment is None:
            return "", ""

        if raw.startswith("/"):
            should_quit = await self._handle_slash(raw)
            return ("quit" if should_quit else ""), ""

        clean_text, refs = _parse_file_refs(raw)
        valid_refs, missing_refs = _split_existing_file_refs(refs)
        for ref in missing_refs:
            self._ui.add_message("agent", f"⚠️ File not found: {escape_markup(ref._path)}").finish()

        if refs and not clean_text and not valid_refs and attachment is None:
            return "", ""

        display = _build_user_display(clean_text, valid_refs) if refs else escape_markup(raw)
        agent_query = _build_agent_query(clean_text, valid_refs)

        if attachment is not None:
            display = attachment.pill_markup + ("\n" + display if display else "")
            agent_query = (agent_query + "\n\n" if agent_query else "") + attachment.xml_tag

        if self._agent_running:
            self._enqueue_pending(agent_query, display)
            return "", ""

        self._ui.add_message("user", display)
        self._active_turn_status = self._ui.add_turn_status_bar()
        return "run", agent_query

    def _enqueue_pending(self, agent_query: str, display: str) -> None:
        """Pin *display* in the UI and queue *agent_query* for when the agent is free."""
        message = self._pending_queue.enqueue(agent_query, display)
        self._pending_handles[message.id] = self._ui.add_pending_message(display)

    # ── Agent run ─────────────────────────────────────────────────────────────

    async def run_agent(self, query: str) -> None:
        """Run *query*, then keep draining the pending-message queue.

        Each queued message is run as its own turn, in submission order,
        as long as the previous turn didn't end on a choice prompt (which
        requires the user's input before anything else can proceed).
        """
        if self._service is None:
            self._ui.add_message(
                "agent", "⚠️ Still loading — please wait a moment and try again."
            ).finish()
            return

        while True:
            await self._run_turn(query)
            if self._awaiting_choice or self._pending_queue.is_empty():
                return
            query = self._dequeue_pending()

    def _dequeue_pending(self) -> str:
        """Pop the next queued message, surface it as a normal user turn, return its query."""
        message = self._pending_queue.pop_next()
        assert message is not None  # caller already checked the queue is non-empty
        handle = self._pending_handles.pop(message.id, None)
        if handle is not None:
            handle.remove()
        self._ui.add_message("user", message.display)
        self._active_turn_status = self._ui.add_turn_status_bar()
        return message.agent_query

    async def _run_turn(self, query: str) -> None:
        assert self._service is not None
        self._agent_running = True
        presenter = _TurnPresenter(self._ui)
        try:
            async for event in self._service.astream(query):
                if not isinstance(event, BaseAgentStreamEvent):
                    logger.warning("astream() yielded unexpected type %r; skipping", type(event))
                    continue
                self._dispatch_stream_event(event, presenter)
                if isinstance(event, (ResponseEvent, ErrorEvent)):
                    break
        except Exception as exc:
            presenter.handle_exception(exc)
        finally:
            self._agent_running = False
            presenter.cleanup()
            if self._active_turn_status is not None:
                self._active_turn_status.stop()
                self._active_turn_status = None
            if not self._awaiting_choice:
                self._ui.set_input_placeholder("Ask a question about your data…")
            self._ui.add_divider()

    def _dispatch_stream_event(
        self, event: BaseAgentStreamEvent, presenter: _TurnPresenter
    ) -> None:
        """Route a single stream event to the appropriate presenter handler."""
        if isinstance(event, ReasoningEvent):
            presenter.handle_reasoning(event)
        elif isinstance(event, TokenEvent):
            presenter.handle_token(event)
        elif isinstance(event, ToolCommunicationEvent):
            presenter.handle_tool_communication(event)
        elif isinstance(event, ToolCallEvent):
            presenter.handle_tool_call(event)
        elif isinstance(event, WorkerDoneEvent):
            presenter.handle_worker_done(event)
        elif isinstance(event, SubagentEvent):
            presenter.handle_subagent_event(event)
        elif isinstance(event, ToolResultEvent):
            presenter.handle_tool_result(event)
        elif isinstance(event, UsageEvent):
            presenter.handle_usage(event, self._active_turn_status)
        elif isinstance(event, InputRequiredEvent):
            self._show_choice_prompt(event.content, list(event.choices))
        elif isinstance(event, ResponseEvent):
            presenter.handle_response(event)
        elif isinstance(event, ErrorEvent):
            presenter.handle_error(event)

    # ── Choice handling ───────────────────────────────────────────────────────

    def _show_choice_prompt(self, question: str, choices: list[str]) -> None:
        labels = string.ascii_uppercase[: len(choices)]
        other_label = (
            string.ascii_uppercase[len(choices)]
            if len(choices) < len(string.ascii_uppercase)
            else None
        )
        lines = [
            f"[bold {theme['warning']}]❓[/bold {theme['warning']}]  "
            f"[bold {theme['text_primary']}]{question}[/bold {theme['text_primary']}]\n"
        ]
        for label, choice_text in zip(labels, choices):
            lines.append(
                f"  [bold {theme['warning']}]{label}[/bold {theme['warning']}]  {choice_text}"
            )
        if other_label is not None:
            lines.append(
                f"  [dim {theme['text_secondary']}]{other_label}"
                f"[/dim {theme['text_secondary']}]  "
                f"[dim {theme['text_secondary']}]Other (type your answer below)"
                f"[/dim {theme['text_secondary']}]"
            )
        lines.append(
            f"  [dim {theme['text_secondary']}]Press Esc to cancel[/dim {theme['text_secondary']}]"
        )
        self._ui.add_message("question", "\n".join(lines)).finish()
        self._pending_choices = list(choices)
        self._other_choice_label = other_label
        self._awaiting_custom_choice_input = False
        self._awaiting_choice = True
        prompt_labels = ", ".join(labels)
        if other_label is not None:
            self._ui.set_input_placeholder(
                f"Enter {prompt_labels}, {other_label}, type your answer, or press Esc to cancel…"
            )
        else:
            self._ui.set_input_placeholder(
                "Enter a choice, type your answer, or press Esc to cancel…"
            )
        self._ui.add_input_class("awaiting-choice")

    @property
    def awaiting_choice(self) -> bool:
        return self._awaiting_choice

    def _exit_choice_mode(self) -> None:
        self._awaiting_choice = False
        self._pending_choices = []
        self._other_choice_label = None
        self._awaiting_custom_choice_input = False
        self._ui.set_input_placeholder("Ask a question about your data…")
        self._ui.remove_input_class("awaiting-choice")

    def cancel_choice(self) -> str | None:
        """Exit choice mode and return the resume input to send to the agent.

        Returns ``"cancel"`` when a choice was active (caller must pass this
        to ``run_agent``), or ``None`` when there was nothing to cancel.
        """
        if not self._awaiting_choice:
            return None
        self._exit_choice_mode()
        self._ui.add_message("agent", "Choice cancelled.").finish()
        return "cancel"

    def _handle_user_choice(self, raw: str) -> str | None:
        raw_stripped = raw.strip()
        upper = raw_stripped.upper()
        if (
            self._other_choice_label is not None
            and not self._awaiting_custom_choice_input
            and upper == self._other_choice_label
        ):
            self._awaiting_custom_choice_input = True
            self._other_choice_label = None
            self._ui.add_message("agent", "Type your answer and press Enter.").finish()
            self._ui.set_input_placeholder("Type your answer and press Enter…")
            return None

        pending_choices = list(self._pending_choices)
        self._exit_choice_mode()

        choice_map = {
            label: idx for idx, label in enumerate(string.ascii_uppercase[: len(pending_choices)])
        }
        answer = (
            pending_choices[choice_map[upper]]
            if upper in choice_map and choice_map[upper] < len(pending_choices)
            else raw_stripped
        )

        self._ui.add_message("user", escape_markup(raw)).finish()
        return answer

    # ── Slash command dispatch ────────────────────────────────────────────────

    async def _handle_slash(self, cmd: str) -> bool:
        """Dispatch a slash command. Returns True if the app should quit."""
        if cmd == "/exit":
            return True
        elif cmd == "/clear":
            await self.clear_conv()
        elif cmd == "/reset":
            await self.reset()
        elif cmd == "/compact":
            await self.compact()
        elif cmd == "/ls-workspace":
            self.ls_workspace()
        elif cmd == "/models":
            self.show_models()
        elif cmd == "/stop":
            await self.stop_agent()
        elif cmd == "/cancel-all-messages":
            self.cancel_pending_messages()
        elif cmd == "/cancel-message":
            self.cancel_last_pending_message()
        elif cmd == "/help":
            self.show_help()
        elif cmd == "/themes":
            self.show_themes()
        elif cmd == "/vars":
            self._ui.add_message(
                "agent",
                "⚠️ `/vars` has been removed. Use `/help` to see available commands.",
            ).finish()
        else:
            self._ui.add_message(
                "agent",
                f"⚠️ Unknown command: `{cmd}`\n\nType `/help` to see all available commands.",
            ).finish()
        return False

    # ── Actions ───────────────────────────────────────────────────────────────

    async def reset(self) -> None:
        """Reset agent session and reload data from disk."""
        self._clear_pending_queue()
        self._ui.clear_messages()
        if self._service is not None:
            try:
                await self._service.reset_session()
                self._ui.add_message("agent", "✓ Session reset.").finish()
            except Exception as exc:
                self._ui.add_message("agent", f"❌ Reset failed: {exc}").finish()
        else:
            self._ui.add_message("agent", "Not loaded yet.").finish()

    async def clear_conv(self) -> None:
        """Clear conversation context (preserves session variables)."""
        self._clear_pending_queue()
        self._ui.clear_messages()
        if self._service is not None:
            try:
                await self._service.clear_context()
            except Exception:
                logger.exception("Failed to clear service context")
        self._ui.add_message("agent", "✓ Context cleared.").finish()

    async def compact(self) -> None:
        """Summarize the conversation and replace it with a compact context preamble."""
        if self._service is None:
            self._ui.add_message("agent", "Not loaded yet.").finish()
            return
        status = self._ui.add_message("agent", "Compacting conversation…")
        status.set_content("Compacting conversation…")
        compact_timer: TurnStatusHandle | None = self._ui.add_turn_status_bar()
        try:
            await self._service.compact_chat_history()
        except Exception as exc:
            status.set_content(f"❌ Compact failed: {exc}")
            if compact_timer is not None:
                compact_timer.stop()
            status.finish()
            return
        try:
            self._ui.clear_messages()
            compact_timer = None  # removed from DOM by clear_messages()
            self._ui.add_message(
                "agent",
                "**✓ Compaction done.** You may continue the conversation.",
            ).finish()
        finally:
            status.finish()
            if compact_timer is not None:
                compact_timer.stop()

    def show_models(self) -> None:
        """Display the primary and secondary model in use."""
        cfg = self._cfg or self._base_config
        self._ui.add_message(
            "agent",
            format_models_message(
                cfg.provider,
                cfg.model,
                cfg.secondary_provider,
                cfg.secondary_model,
            ),
        ).finish()

    def show_help(self) -> None:
        """Display all available slash commands with descriptions."""
        self._ui.add_message("agent", format_help_message()).finish()

    def show_themes(self) -> None:
        """Display the list of available colour themes and mark the active one."""
        self._ui.add_message(
            "agent",
            format_themes_message(_theme.active_name, _theme.THEME_DESCRIPTIONS),
        ).finish()

    async def stop_agent(self) -> None:
        """Stop the currently running agent turn."""
        if not self._agent_running:
            self._ui.add_message("agent", "No agent is currently running.").finish()
            return
        self._ui.stop_agent()
        if self._service is not None:
            await self._service.rewind_turn()
        self._ui.add_message("agent", "⏹ Agent stopped. You can continue from here.").finish()

    def cancel_pending_messages(self) -> None:
        """Discard every message currently queued behind a running agent turn."""
        removed = self._pending_queue.cancel_all()
        for message in removed:
            self._discard_pending_handle(message.id)
        if removed:
            count = len(removed)
            self._ui.add_message(
                "agent", f"✓ Cancelled {count} pending message{'s' if count != 1 else ''}."
            ).finish()
        else:
            self._ui.add_message("agent", "No pending messages to cancel.").finish()

    def cancel_last_pending_message(self) -> None:
        """Discard only the most recently queued message."""
        message = self._pending_queue.cancel_last()
        if message is None:
            self._ui.add_message("agent", "No pending messages to cancel.").finish()
            return
        self._discard_pending_handle(message.id)
        self._ui.add_message("agent", "✓ Cancelled last pending message.").finish()

    def _discard_pending_handle(self, message_id: int) -> None:
        handle = self._pending_handles.pop(message_id, None)
        if handle is not None:
            handle.remove()

    def _clear_pending_queue(self) -> None:
        """Silently drop all queued messages (used by /reset and /clear)."""
        for message in self._pending_queue.cancel_all():
            self._discard_pending_handle(message.id)

    def ls_workspace(self) -> None:
        if self._service is None:
            self._ui.add_message("agent", "_Not loaded yet._").finish()
            return
        try:
            files = self._service.get_workspace_files()
        except Exception as exc:
            self._ui.add_message("agent", f"❌ {exc}").finish()
            return
        self._ui.show_workspace_panel(files)
