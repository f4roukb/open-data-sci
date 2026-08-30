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

import asyncio
import difflib
import logging
import string
from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncIterator

from rich.markup import escape as escape_markup

from opendatasci._tui.service import OpenDataSciTuiService
from opendatasci._tui.session import CLISessionInfo
from opendatasci.agents.agents import Invocation
from opendatasci.agents.agents_factory import create_agent
from opendatasci.configs import OpenDataSciConfig
from opendatasci.memory.messages import MessageOrigin
from opendatasci.streaming import AgentStreamEvent, BaseAgentStreamEvent
from opendatasci.streaming.events import (
    ApprovalRequiredEvent,
    ErrorEvent,
    InputRequiredEvent,
    ReasoningEvent,
    ResponseEvent,
    SubagentEvent,
    TaskDoneEvent,
    TokenEvent,
    ToolCallEvent,
    ToolCommunicationEvent,
    ToolResultEvent,
    UsageEvent,
)
from opendatasci.tasks.base import AgentTaskStatus
from opendatasci.tools.mcp import load_mcp_servers

from . import theme as _theme
from .adapter import (
    PendingMessageHandle,
    SubmitAction,
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
from .message_queue import PendingMessageOrigin, PendingMessageQueue
from .presenter import _TurnPresenter, apply_usage_event
from .theme import active as theme

logger = logging.getLogger(__name__)

# The resume query sent to the agent when the user cancels a choice prompt —
# a real (if synthetic) turn of conversation, not a control-flow sentinel.
_CHOICE_CANCELLED_QUERY = "cancel"

# How often the header's "running background tasks" line refreshes.
_BACKGROUND_STATUS_POLL_SECONDS = 2


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
        self._boot_failed: bool = False
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._awaiting_choice: bool = False
        self._awaiting_approval: bool = False
        self._pending_choices: list[str] = []
        self._other_choice_label: str | None = None
        self._awaiting_custom_choice_input: bool = False
        self._active_turn_status: TurnStatusHandle | None = None
        self._agent_running: bool = False
        self._pending_queue = PendingMessageQueue()
        self._pending_handles: dict[int, PendingMessageHandle] = {}
        self._background_watcher_task: asyncio.Task[None] | None = None
        self._background_status_task: asyncio.Task[None] | None = None
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

    @property
    def agent_running(self) -> bool:
        """True while an agent turn is streaming."""
        return self._agent_running

    @property
    def has_paste_attachment(self) -> bool:
        """True when a multi-line paste is pending in the attachment bar."""
        return self._paste_attachment is not None

    # ── Completion state suppression ──────────────────────────────────────────
    # Used by app.py around programmatic input-value updates (e.g. history
    # navigation) that must not be misread as a new completion trigger.

    @property
    def is_suppressing_input_change(self) -> bool:
        return self._completion.is_suppressing_input_change

    def suppress_next_input_change(self) -> None:
        self._completion.suppress_next_input_change()

    def cancel_input_change_suppression(self) -> None:
        self._completion.cancel_input_change_suppression()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release the agent sandbox and any other resources held by the controller."""
        if self._background_watcher_task is not None:
            self._background_watcher_task.cancel()
            self._background_watcher_task = None
        if self._background_status_task is not None:
            self._background_status_task.cancel()
            self._background_status_task = None
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
            self._background_watcher_task = asyncio.create_task(self._watch_background_tasks())
            self._background_status_task = asyncio.create_task(self._poll_background_task_status())
        except FileNotFoundError:
            hint = self._did_you_mean(self._workspace_path)
            await self._fail_boot(
                ui,
                f"❌ File not found: `{escape_markup(self._workspace_path)}`\n\n"
                f"Check the path and try again.{hint}",
            )
        except PermissionError:
            await self._fail_boot(
                ui, f"❌ Permission denied: `{escape_markup(self._workspace_path)}`"
            )
        except ValueError as exc:
            await self._fail_boot(ui, f"❌ Provider error: {exc}")
        except Exception as exc:
            await self._fail_boot(ui, f"❌ Failed to load: {exc}")

    async def _fail_boot(self, ui: UIAdapter, msg_text: str) -> None:
        self._boot_failed = True
        msg = ui.add_message("agent", "")
        await msg.set_content(msg_text)
        await msg.finish()

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

    async def on_submit(self, raw: str) -> tuple[SubmitAction, str]:
        """Handle input submission.

        Returns ``(action, payload)`` where *action* is one of:
        - ``SubmitAction.RUN``  — caller should run the agent with *payload* as the query
        - ``SubmitAction.QUIT`` — caller should exit
        - ``SubmitAction.NONE`` — action handled internally, nothing more to do
        """
        self.hide_completion()

        # Always capture and clear the paste attachment at the start of
        # submission so it is never accidentally carried into the next turn.
        attachment = self._paste_attachment
        self._paste_attachment = None
        self._ui.hide_attachment()

        if self._awaiting_choice:
            if not raw:
                return SubmitAction.NONE, ""
            if raw.split()[0] in {"/exit", "/reset", "/clear"}:
                self._exit_choice_mode()
                should_quit = await self._handle_slash(raw)
                return (SubmitAction.QUIT if should_quit else SubmitAction.NONE), ""
            answer = await self._handle_user_choice(raw)
            if answer is not None:
                return SubmitAction.RESUME_INPUT, answer
            return SubmitAction.NONE, ""

        if self._awaiting_approval:
            # The decision is made in the approval prompt widget (↑/↓ + Enter);
            # typed input is ignored except for quitting the app.
            if raw.split() and raw.split()[0] == "/exit":
                return SubmitAction.QUIT, ""
            return SubmitAction.NONE, ""

        if not raw and attachment is None:
            return SubmitAction.NONE, ""

        if raw.startswith("/"):
            should_quit = await self._handle_slash(raw)
            return (SubmitAction.QUIT if should_quit else SubmitAction.NONE), ""

        clean_text, refs = _parse_file_refs(raw)
        valid_refs, missing_refs = _split_existing_file_refs(refs)
        for ref in missing_refs:
            await self._ui.add_message(
                "agent", f"⚠️ File not found: {escape_markup(ref._path)}"
            ).finish()

        if refs and not clean_text and not valid_refs and attachment is None:
            return SubmitAction.NONE, ""

        display = _build_user_display(clean_text, valid_refs) if refs else escape_markup(raw)
        agent_query = _build_agent_query(clean_text, valid_refs)

        if attachment is not None:
            display = attachment.pill_markup + ("\n" + display if display else "")
            agent_query = (agent_query + "\n\n" if agent_query else "") + attachment.xml_tag

        if self._agent_running:
            self._enqueue_pending(agent_query, display)
            return SubmitAction.NONE, ""

        self._ui.add_message("user", display)
        self._active_turn_status = self._ui.add_turn_status_bar()
        return SubmitAction.RUN, agent_query

    def _enqueue_pending(
        self,
        agent_query: str,
        display: str,
        *,
        origin: PendingMessageOrigin = PendingMessageOrigin.USER,
    ) -> None:
        """Queue *agent_query* for when the agent is free.

        Only user-typed text is pinned in the UI as a pending indicator —
        the user never typed a worker result, so it shouldn't appear to be
        waiting on them either.
        """
        message = self._pending_queue.enqueue(agent_query, display, origin=origin)
        if origin is PendingMessageOrigin.USER:
            self._pending_handles[message.id] = self._ui.add_pending_message(display)

    # ── Agent run ─────────────────────────────────────────────────────────────

    async def _ensure_service_ready(self) -> bool:
        """Show a status message and return False if there's no service to run against yet."""
        if self._service is not None:
            return True
        if self._boot_failed:
            await self._ui.add_message(
                "agent",
                "❌ Startup failed, so queries can't run in this session. "
                "Fix the problem shown above and restart the app "
                "(type `/exit` to quit).",
            ).finish()
        else:
            await self._ui.add_message(
                "agent", "⚠️ Still loading — please wait a moment and try again."
            ).finish()
        return False

    async def run_agent(self, query: str | list[Invocation]) -> None:
        """Run *query* as a new turn, then keep draining the pending-message queue.

        Once a turn finishes, every message queued in the meantime is sent
        together as a single new turn, as long as the turn didn't end on a
        choice prompt (which requires the user's input before anything else
        can proceed).
        """
        if not await self._ensure_service_ready():
            return
        assert self._service is not None
        invocation = query if isinstance(query, list) else Invocation.from_text(query)
        await self._run_turn(self._service.astream(invocation))
        await self._drain_loop()

    async def resume_with_input(self, answer: str) -> None:
        """Resume a pending question/choice prompt with *answer*, then drain the pending queue."""
        if not await self._ensure_service_ready():
            return
        assert self._service is not None
        await self._run_turn(self._service.resume_with_input(answer))
        await self._drain_loop()

    async def resume_with_approval(self, approved: bool) -> None:
        """Resolve a pending approval prompt with the user's Yes/No decision, then drain the pending queue."""
        self._awaiting_approval = False
        self._ui.set_input_placeholder("Ask a question about your data…")
        await self._ui.add_message("user", "Yes" if approved else "No").finish()
        if not await self._ensure_service_ready():
            return
        assert self._service is not None
        await self._run_turn(self._service.resume_with_approval(approved))
        await self._drain_loop()

    async def _drain_loop(self) -> None:
        """Keep starting new turns until nothing is left to send or a prompt opens.

        Drains both the user-message queue and any task-manager content the
        agent hasn't picked up yet, so neither has to wait for an unrelated
        future event to surface.
        """
        assert self._service is not None
        while not (
            self._awaiting_choice
            or self._awaiting_approval
            or self._service.is_user_input_required()
        ):
            if not self._pending_queue.is_empty():
                batch = self._drain_pending_batch()
                await self._run_turn(self._service.astream(batch))
                continue
            if self._service.task_manager.has_task_updates():
                self._active_turn_status = self._ui.add_turn_status_bar()
                await self._run_turn(self._service.astream([]))
                continue
            return

    async def _watch_background_tasks(self) -> None:
        """Proactively start a turn when a background task finishes while the agent is idle.

        Runs for the lifetime of the session (started in ``boot``, cancelled
        in ``close``). ``listen_task_updates`` blocks until the next terminal
        task, so this stays idle between completions rather than polling.

        The raw completion is never shown as a chat message, and this method
        never feeds it to the agent directly — the agent picks up finished
        tasks on its own. This only decides whether to kick off a turn now,
        so an idle agent doesn't wait on the next unrelated user message.
        """
        assert self._service is not None
        async for _record in self._service.task_manager.listen_task_updates():
            if self._agent_running or self._service.is_user_input_required():
                continue
            self._active_turn_status = self._ui.add_turn_status_bar()
            await self.run_agent([])

    async def _poll_background_task_status(self) -> None:
        """Refresh the header's "running background tasks" line every few seconds.

        Purely a status display of already-in-memory state — not the
        completion-delivery mechanism (``_watch_background_tasks`` is).
        """
        assert self._service is not None
        while True:
            await asyncio.sleep(_BACKGROUND_STATUS_POLL_SECONDS)
            records = await self._service.task_manager.list_tasks()
            running = [r for r in records if r.status == AgentTaskStatus.RUNNING]
            if not running:
                self._ui.set_background_tasks("")
                continue
            parts = []
            for record in running:
                latest = record.progress[-1].progress_update.ongoing if record.progress else ""
                parts.append(f"{record.summary} — {latest}" if latest else record.summary)
            self._ui.set_background_tasks("; ".join(parts))

    def _drain_pending_batch(self) -> list[Invocation]:
        """Drain every queued message, surface user-typed ones in the UI, and return the batch.

        Messages are shown in the order they arrived. Background-task
        completions are never shown as chat messages — they're queued only
        so the agent gets fed the result once it's free.
        """
        messages = self._pending_queue.drain_all()
        assert messages  # caller already checked the queue is non-empty
        batch: list[Invocation] = []
        for message in messages:
            handle = self._pending_handles.pop(message.id, None)
            if handle is not None:
                handle.remove()
            is_task = message.origin is PendingMessageOrigin.TASK
            if not is_task:
                self._ui.add_message("user", message.display)
            batch.append(
                Invocation.from_text(
                    message.content,
                    origin=MessageOrigin.TASK if is_task else MessageOrigin.USER,
                    created_at=message.created_at,
                )
            )
        self._active_turn_status = self._ui.add_turn_status_bar()
        return batch

    async def _run_turn(self, stream: AsyncIterator[AgentStreamEvent]) -> None:
        self._agent_running = True
        presenter = _TurnPresenter(self._ui)
        try:
            async for event in stream:
                if not isinstance(event, BaseAgentStreamEvent):
                    logger.warning("agent stream yielded unexpected type %r; skipping", type(event))
                    continue
                await self._dispatch_stream_event(event, presenter)
                if isinstance(event, (ResponseEvent, ErrorEvent)):
                    break
        except Exception as exc:
            await presenter.handle_exception(exc)
        finally:
            self._agent_running = False
            await presenter.cleanup()
            if self._active_turn_status is not None:
                self._active_turn_status.stop()
                self._active_turn_status = None
            if not self._awaiting_choice and not self._awaiting_approval:
                self._ui.set_input_placeholder("Ask a question about your data…")
            self._ui.add_divider()

    async def _dispatch_stream_event(
        self, event: BaseAgentStreamEvent, presenter: _TurnPresenter
    ) -> None:
        """Route a single stream event to the appropriate presenter handler."""
        if isinstance(event, ReasoningEvent):
            presenter.handle_reasoning(event)
        elif isinstance(event, TokenEvent):
            await presenter.handle_token(event)
        elif isinstance(event, ToolCommunicationEvent):
            presenter.handle_tool_communication(event)
        elif isinstance(event, ToolCallEvent):
            await presenter.handle_tool_call(event)
        elif isinstance(event, TaskDoneEvent):
            presenter.handle_task_done(event)
        elif isinstance(event, SubagentEvent):
            presenter.handle_subagent_event(event)
        elif isinstance(event, ToolResultEvent):
            presenter.handle_tool_result(event)
        elif isinstance(event, UsageEvent):
            apply_usage_event(event, self._active_turn_status)
        elif isinstance(event, InputRequiredEvent):
            await self._show_choice_prompt(event.content, list(event.choices))
        elif isinstance(event, ApprovalRequiredEvent):
            self._show_approval_prompt(event)
        elif isinstance(event, ResponseEvent):
            presenter.handle_response(event)
        elif isinstance(event, ErrorEvent):
            await presenter.handle_error(event)

    # ── Choice handling ───────────────────────────────────────────────────────

    async def _show_choice_prompt(self, question: str, choices: list[str]) -> None:
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
        await self._ui.add_message("question", "\n".join(lines)).finish()
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

    async def cancel_choice(self) -> str | None:
        """Exit choice mode and return the resume input to send to the agent.

        Returns ``_CHOICE_CANCELLED_QUERY`` when a choice was active (caller
        must pass this to ``run_agent``), or ``None`` when there was nothing
        to cancel.
        """
        if not self._awaiting_choice:
            return None
        self._exit_choice_mode()
        await self._ui.add_message("agent", "Choice cancelled.").finish()
        return _CHOICE_CANCELLED_QUERY

    async def _handle_user_choice(self, raw: str) -> str | None:
        raw_stripped = raw.strip()
        upper = raw_stripped.upper()
        if (
            self._other_choice_label is not None
            and not self._awaiting_custom_choice_input
            and upper == self._other_choice_label
        ):
            self._awaiting_custom_choice_input = True
            self._other_choice_label = None
            await self._ui.add_message("agent", "Type your answer and press Enter.").finish()
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

        await self._ui.add_message("user", escape_markup(raw)).finish()
        return answer

    # ── Approval handling ─────────────────────────────────────────────────────

    @property
    def awaiting_approval(self) -> bool:
        return self._awaiting_approval

    def _show_approval_prompt(self, event: ApprovalRequiredEvent) -> None:
        self._ui.show_approval_prompt(event.description, event.heads_up)
        self._awaiting_approval = True
        self._ui.set_input_placeholder("↑/↓ to select Yes or No, Enter to confirm, Esc to decline…")

    # ── Slash command dispatch ────────────────────────────────────────────────

    async def _handle_slash(self, raw: str) -> bool:
        """Dispatch a slash command. Returns True if the app should quit.

        Only the first whitespace-separated token is matched, so trailing
        text ("/help x") doesn't turn a valid command into an unknown one.
        """
        cmd = raw.split()[0] if raw.split() else raw
        if cmd == "/exit":
            return True
        elif cmd == "/clear":
            await self.clear_conv()
        elif cmd == "/reset":
            await self.reset()
        elif cmd == "/compact":
            await self.compact()
        elif cmd == "/ls-workspace":
            await self.ls_workspace()
        elif cmd == "/models":
            await self.show_models()
        elif cmd == "/stop":
            await self.stop_agent()
        elif cmd == "/cancel-all-messages":
            await self.cancel_pending_messages()
        elif cmd == "/cancel-message":
            await self.cancel_last_pending_message()
        elif cmd == "/help":
            await self.show_help()
        elif cmd == "/themes":
            await self.show_themes()
        elif cmd == "/vars":
            await self._ui.add_message(
                "agent",
                "⚠️ `/vars` has been removed. Use `/help` to see available commands.",
            ).finish()
        else:
            await self._ui.add_message(
                "agent",
                f"⚠️ Unknown command: `{cmd}`\n\nType `/help` to see all available commands.",
            ).finish()
        return False

    # ── Actions ───────────────────────────────────────────────────────────────

    async def reset(self) -> None:
        """Reset agent session and reload data from disk."""
        self._awaiting_approval = False  # the prompt widget is removed with the messages
        self._clear_pending_queue()
        self._ui.clear_messages()
        if self._service is not None:
            try:
                await self._service.reset_session()
                await self._ui.add_message("agent", "✓ Session reset.").finish()
            except Exception as exc:
                await self._ui.add_message("agent", f"❌ Reset failed: {exc}").finish()
        else:
            await self._ui.add_message("agent", "Not loaded yet.").finish()

    async def clear_conv(self) -> None:
        """Clear all conversation context (preserves session variables)."""
        if self._agent_running:
            # A still-running turn would write the cleared conversation back
            # into state (and schedule its summarization) when it finishes.
            self._ui.stop_agent()
        self._awaiting_approval = False  # the prompt widget is removed with the messages
        self._clear_pending_queue()
        self._ui.clear_messages()
        if self._service is not None:
            try:
                await self._service.clear_context()
            except Exception as exc:
                logger.exception("Failed to clear service context")
                await self._ui.add_message("agent", f"❌ Clear failed: {exc}").finish()
                return
        await self._ui.add_message("agent", "✓ Context cleared.").finish()

    async def compact(self) -> None:
        """Summarize the conversation and replace it with a compact context preamble."""
        if self._service is None:
            await self._ui.add_message("agent", "Not loaded yet.").finish()
            return
        status = self._ui.add_message("agent", "Compacting conversation…")
        await status.set_content("Compacting conversation…")
        compact_timer: TurnStatusHandle | None = self._ui.add_turn_status_bar()
        try:
            await self._service.compact_chat_history()
        except Exception as exc:
            await status.set_content(f"❌ Compact failed: {exc}")
            if compact_timer is not None:
                compact_timer.stop()
            await status.finish()
            return
        try:
            self._ui.clear_messages()
            compact_timer = None  # removed from DOM by clear_messages()
            await self._ui.add_message(
                "agent",
                "**✓ Compaction done.** You may continue the conversation.",
            ).finish()
        finally:
            await status.finish()
            if compact_timer is not None:
                compact_timer.stop()

    async def show_models(self) -> None:
        """Display the primary and secondary model in use."""
        cfg = self._cfg or self._base_config
        await self._ui.add_message(
            "agent",
            format_models_message(
                cfg.provider,
                cfg.model,
                cfg.secondary_provider,
                cfg.secondary_model,
            ),
        ).finish()

    async def show_help(self) -> None:
        """Display all available slash commands with descriptions."""
        await self._ui.add_message("agent", format_help_message()).finish()

    async def show_themes(self) -> None:
        """Display the list of available colour themes and mark the active one."""
        await self._ui.add_message(
            "agent",
            format_themes_message(_theme.active_name, _theme.THEME_DESCRIPTIONS),
        ).finish()

    async def stop_agent(self) -> None:
        """Stop the currently running agent turn."""
        if not self._agent_running:
            await self._ui.add_message("agent", "No agent is currently running.").finish()
            return
        self._ui.stop_agent()
        if self._service is not None:
            await self._service.rewind_turn()
        await self._ui.add_message("agent", "⏹ Agent stopped. You can continue from here.").finish()

    async def cancel_pending_messages(self) -> None:
        """Discard every user-typed message currently queued behind a running agent turn.

        Worker (background task) results are left queued — the user never
        typed them, so they aren't the user's to cancel.
        """
        removed = self._pending_queue.cancel_all_user_messages()
        for message in removed:
            self._discard_pending_handle(message.id)
        if removed:
            count = len(removed)
            await self._ui.add_message(
                "agent", f"✓ Cancelled {count} pending message{'s' if count != 1 else ''}."
            ).finish()
        else:
            await self._ui.add_message("agent", "No pending messages to cancel.").finish()

    async def cancel_last_pending_message(self) -> None:
        """Discard only the most recently queued user-typed message."""
        message = self._pending_queue.cancel_last_user_message()
        if message is None:
            await self._ui.add_message("agent", "No pending messages to cancel.").finish()
            return
        self._discard_pending_handle(message.id)
        await self._ui.add_message("agent", "✓ Cancelled last pending message.").finish()

    def _discard_pending_handle(self, message_id: int) -> None:
        handle = self._pending_handles.pop(message_id, None)
        if handle is not None:
            handle.remove()

    def _clear_pending_queue(self) -> None:
        """Silently drop all queued messages (used by /reset and /clear)."""
        for message in self._pending_queue.cancel_all():
            self._discard_pending_handle(message.id)

    async def ls_workspace(self) -> None:
        if self._service is None:
            await self._ui.add_message("agent", "_Not loaded yet._").finish()
            return
        try:
            files = self._service.get_workspace_files()
        except Exception as exc:
            await self._ui.add_message("agent", f"❌ {exc}").finish()
            return
        self._ui.show_workspace_panel(files)
