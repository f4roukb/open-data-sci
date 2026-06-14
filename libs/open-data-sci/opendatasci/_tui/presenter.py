"""_TurnPresenter — manages all UI state for a single agent-turn stream.

Extracted from ``CLIController.run_agent`` to reduce its complexity.
``CLIController`` creates one ``_TurnPresenter`` per turn, feeds it events,
and calls ``cleanup()`` in the ``finally`` block.
"""

import logging
import time

from opendatasci.streaming.events import (
    ErrorEvent,
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
from opendatasci.tools import ToolName

from .adapter import EphemeralHandle, MessageHandle, ThinkingHandle, TurnStatusHandle, UIAdapter
from .tools_display import REGISTRY, ToolDisplay

logger = logging.getLogger(__name__)


class _TurnPresenter:
    """Manages ephemeral UI state (bubbles, tool blocks, thinking) for one turn.

    All methods are synchronous; awaiting happens inside ``MessageBubble``
    and ``ToolCallBlock`` via Textual's own async machinery.
    """

    def __init__(self, ui: UIAdapter) -> None:
        self._ui = ui
        self._agent_msg: MessageHandle | None = None
        self._thinking_start: float = 0.0
        self._had_reasoning: bool = False
        self._ephemerals: list[EphemeralHandle] = []
        # tool_call_id → ephemeral (promoted once tool_call fires)
        self._ephemerals_by_id: dict[str, EphemeralHandle] = {}
        # Ephemerals created from leading tool_communication before tool_call fires
        self._pending_ephemerals: dict[str, EphemeralHandle] = {}
        self._worker_block: EphemeralHandle | None = None
        # tool_call_id → latest communication text (buffered until block is ready)
        self._comm_buffers: dict[str, str] = {}
        # tool_call_ids for tools with display=False — no UI created, result silently ignored
        self._hidden_tool_call_ids: set[str] = set()
        # Ephemeral "Thinking..." spinner shown while the LLM is processing
        self._thinking_block: ThinkingHandle | None = None
        self._show_thinking_block()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _show_thinking_block(self) -> None:
        if self._thinking_block is None:
            self._thinking_block = self._ui.add_thinking_block()
            self._had_reasoning = False

    def _dismiss_thinking_block(self) -> None:
        if self._thinking_block is not None:
            self._thinking_block.dismiss()
            self._thinking_block = None

    def _finish_thinking(self) -> None:
        if self._thinking_block is None:
            return
        if self._had_reasoning:
            elapsed = int(time.monotonic() - self._thinking_start)
            self._thinking_block.finish(f"Thought for {elapsed}s")
        else:
            self._thinking_block.dismiss()
        self._thinking_block = None

    @staticmethod
    def _make_label(tool_display: ToolDisplay | None, event: ToolCallEvent) -> str:
        icon = tool_display.icon if tool_display else ""
        label_text = (tool_display.label if tool_display else None) or event.tool
        return f"{icon} {label_text}".strip() if icon else label_text

    @staticmethod
    def _make_summary(tool_display: ToolDisplay | None, event: ToolCallEvent) -> str:
        icon = tool_display.icon if tool_display else ""
        return f"{icon} {event.summary}".strip() if (icon and event.summary) else event.summary

    # ── Event handlers ────────────────────────────────────────────────────────

    def handle_reasoning(self, event: ReasoningEvent) -> None:
        if not self._had_reasoning:
            self._thinking_start = time.monotonic()
            self._had_reasoning = True

    def handle_token(self, event: TokenEvent) -> None:
        self._finish_thinking()
        if self._agent_msg is None:
            self._agent_msg = self._ui.add_message("agent", "")
        self._agent_msg.append(event.content)

    def handle_tool_communication(self, event: ToolCommunicationEvent) -> None:
        self._dismiss_thinking_block()
        tc_id = event.tool_call_id
        comm = event.content
        self._comm_buffers[tc_id] = comm

        if tc_id and tc_id in self._ephemerals_by_id:
            target = self._ephemerals_by_id[tc_id]
            if target.is_running():
                target.set_communication(comm)
        elif tc_id and tc_id not in self._ephemerals_by_id:
            # First comm token — pre-mount a placeholder ephemeral.
            block = self._ui.add_ephemeral_block(comm, "…", "")
            self._pending_ephemerals[tc_id] = block
            self._ephemerals.append(block)
            self._ephemerals_by_id[tc_id] = block

    def handle_tool_call(self, event: ToolCallEvent) -> None:
        tool_call_id = event.tool_call_id or ""
        existing = self._pending_ephemerals.pop(tool_call_id, None) if tool_call_id else None
        tool_display = REGISTRY.get(str(event.tool))

        if tool_display is not None and not tool_display.display:
            # Tool is hidden — discard any pending comm block and never create a new one.
            if existing is not None:
                existing.dismiss()
                self._ephemerals = [e for e in self._ephemerals if e is not existing]
                self._ephemerals_by_id.pop(tool_call_id, None)
            self._comm_buffers.pop(tool_call_id, None)
            if tool_call_id:
                self._hidden_tool_call_ids.add(tool_call_id)
            return

        self._finish_thinking()
        has_narration = self._agent_msg is not None
        if self._agent_msg is not None:
            self._agent_msg.finish()
            self._agent_msg = None

        buffered_comm = self._comm_buffers.pop(tool_call_id, "")
        comm = "" if has_narration else buffered_comm

        if str(event.tool) == ToolName.SPAWN_WORKERS:
            if existing is not None:
                existing.dismiss()
                self._ephemerals = [e for e in self._ephemerals if e is not existing]
                self._ephemerals_by_id.pop(tool_call_id, None)
            block = self._ui.add_worker_block(comm, list(event.worker_summaries))
            self._worker_block = block
            self._ephemerals.append(block)
            if tool_call_id:
                self._ephemerals_by_id[tool_call_id] = block
        elif existing is not None:
            if has_narration:
                existing.set_communication(None)
            existing.upgrade(
                self._make_label(tool_display, event), self._make_summary(tool_display, event)
            )
        else:
            block = self._ui.add_ephemeral_block(
                comm, self._make_label(tool_display, event), self._make_summary(tool_display, event)
            )
            self._ephemerals.append(block)
            if tool_call_id:
                self._ephemerals_by_id[tool_call_id] = block

    def handle_worker_done(self, event: WorkerDoneEvent) -> None:
        if self._worker_block is not None and event.worker_idx is not None:
            if event.success:
                self._worker_block.mark_worker_done(event.worker_idx)
            else:
                self._worker_block.mark_worker_error(event.worker_idx)

    def handle_subagent_event(self, event: SubagentEvent) -> None:
        if self._worker_block is None or event.worker_idx is None:
            return
        if event.event_type == "worker_tool_call":
            tool_name = event.content
            tool_display = REGISTRY.get(tool_name)
            activity = event.summary
            icon = tool_display.icon if tool_display else ""
            if activity:
                activity = f"{icon} {activity}".strip() if icon else activity
            elif tool_display:
                activity = f"{icon} {tool_display.label}".strip() if icon else tool_display.label
            else:
                activity = tool_name
            self._worker_block.update_worker_activity(event.worker_idx, activity)
        elif event.event_type == "worker_tool_result":
            # Tool finished — drop the inline activity so the row reverts to the
            # worker's subtask summary while the LLM decides on the next step.
            self._worker_block.update_worker_activity(event.worker_idx, "")

    def handle_tool_result(self, event: ToolResultEvent) -> None:
        tool_call_id = event.tool_call_id
        if tool_call_id and tool_call_id in self._hidden_tool_call_ids:
            self._hidden_tool_call_ids.discard(tool_call_id)
            return
        if tool_call_id and tool_call_id in self._ephemerals_by_id:
            target = self._ephemerals_by_id.pop(tool_call_id)
            if event.is_error:
                target.set_error()
            else:
                target.set_done()
            self._ephemerals = [e for e in self._ephemerals if e is not target]
        else:
            logger.warning(
                "Received uncorrelated tool_result (tool_call_id=%r); "
                "leaving ephemerals running until cleanup",
                tool_call_id,
            )
        self._show_thinking_block()

    def handle_usage(self, event: UsageEvent, turn_status: TurnStatusHandle | None) -> None:
        input_tokens = event.input_tokens
        output_tokens = event.output_tokens
        cache_read_tokens = event.cache_read_tokens

        context_tokens: int | None = None
        if input_tokens is not None or output_tokens is not None:
            context_tokens = int(input_tokens or 0) + int(output_tokens or 0)

        cached_tokens: int | None = (
            int(cache_read_tokens) if cache_read_tokens is not None else None
        )

        if turn_status is not None:
            turn_status.update_context(context_tokens, cached_tokens)

    def handle_response(self, event: ResponseEvent) -> None:
        self._finish_thinking()
        if self._agent_msg is None and event.content:
            self._agent_msg = self._ui.add_message("agent", event.content)

    def handle_error(self, event: ErrorEvent) -> None:
        self._dismiss_thinking_block()
        if self._agent_msg is None:
            self._agent_msg = self._ui.add_message("agent", "")
        self._agent_msg.append(f"\n\n❌ {event.content}")

    def handle_exception(self, exc: Exception) -> None:
        self._dismiss_thinking_block()
        if self._agent_msg is None:
            self._agent_msg = self._ui.add_message("agent", "")
        self._agent_msg.set_content(f"❌ **Error:** {exc}")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Finalise all open UI elements (called from the run_agent finally block)."""
        self._finish_thinking()
        for e in self._ephemerals:
            e.set_done()
        if self._agent_msg is not None:
            self._agent_msg.finish()
