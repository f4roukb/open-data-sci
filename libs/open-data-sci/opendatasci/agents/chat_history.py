"""Agent-level chat memory: rolling turn summaries and per-call context assembly."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage

from opendatasci._utils.mixins import RenderableMessageMixin
from opendatasci.context.base import BaseContextStore
from opendatasci.context.plans import Plan
from opendatasci.memory.chat_memory import (
    ChatHistoryCompaction,
    ChatTurnContext,
    ChatTurnSummarizer,
    ChatTurnSummary,
)
from opendatasci.memory.messages import (
    HarnessMessage,
    PlanMessage,
    SummaryMessage,
    is_ongoing_turn,
)
from opendatasci.memory.turn_memory import AgentLoopCompactor

logger = logging.getLogger(__name__)

CHAT_MAX_TURN_SUMMARIES: int = 25
_CHAT_TURN_SUMMARY_WINDOW_SIZE: int = 10


class BaseChatHistoryBuilder(ABC):
    @abstractmethod
    def schedule_turn_summarization(self, turn_messages: list[BaseMessage]) -> None:
        """Schedule background summarization of a completed turn."""

    @abstractmethod
    def cancel_pending_tasks(self) -> None:
        """Discard any pending without recording it."""

    @abstractmethod
    async def flush_pending_tasks(self) -> ChatTurnSummary | None:
        """Await and clear the pending tasks, returning its result or ``None``."""

    @abstractmethod
    async def build(
        self,
        messages: list[BaseMessage],
        turn_summaries: list[ChatTurnSummary],
        chat_history_compaction: ChatHistoryCompaction | None,
    ) -> ChatTurnContext:
        """Build the :class:`ChatTurnContext` for the current LLM call."""


class ChatHistoryBuilder(BaseChatHistoryBuilder):
    """Builds the per-call :class:`ChatTurnContext` from agent state.

    Assembles the message list for each LLM call: rolls in pending turn summaries,
    optionally compacts an oversized ongoing turn, prepends summary and plan recall
    messages, then renders everything for the LLM.

    Pass a *loop_compactor_llm* and a *midturn_compaction_threshold* to enable
    mid-turn compaction; omit either to disable it. Pass *context_store* and
    *session_id* to include the session's current plan; omit either to skip it.
    """

    def __init__(
        self,
        summarizer_llm: Any,
        loop_compactor_llm: Any | None = None,
        midturn_compaction_threshold: int | None = None,
        context_store: BaseContextStore | None = None,
        session_id: str | None = None,
        window_size: int = _CHAT_TURN_SUMMARY_WINDOW_SIZE,
    ) -> None:
        if window_size > CHAT_MAX_TURN_SUMMARIES:
            raise ValueError(
                f"window_size ({window_size}) exceeds CHAT_MAX_TURN_SUMMARIES ({CHAT_MAX_TURN_SUMMARIES})"
            )

        self._summarizer = ChatTurnSummarizer(summarizer_llm=summarizer_llm)
        self._loop_compactor = (
            AgentLoopCompactor(llm=loop_compactor_llm) if loop_compactor_llm is not None else None
        )
        self._midturn_compaction_threshold = midturn_compaction_threshold
        self._context_store = context_store
        self._session_id = session_id
        self._window_size = window_size
        self._pending_task: asyncio.Task[ChatTurnSummary | None] | None = None

    def schedule_turn_summarization(self, turn_messages: list[BaseMessage]) -> None:
        """Schedule background summarization of *completed_turn_messages*.

        Raises:
            ValueError: if the turn is still in progress (incomplete).
            ValueError: if the turn is empty.
        """
        if not turn_messages:
            raise ValueError("Cannot summarize an empty turn")
        if is_ongoing_turn(turn_messages):
            raise ValueError("Cannot summarize an ongoing (incomplete) turn")
        self._pending_task = asyncio.create_task(self._summarizer.summarize_turn(turn_messages))

    def _build_plan_message(self, plan: Plan) -> PlanMessage:
        """Build a recall message for *plan*, preserving its original timestamp."""
        raw_ts = plan.metadata.get("created_at")
        created_at = datetime.fromisoformat(raw_ts) if raw_ts else datetime.now(timezone.utc)
        return PlanMessage(content=plan.to_content(), created_at=created_at)

    def _build_compaction_message(self, compaction: "ChatHistoryCompaction") -> HarnessMessage:
        """Convert *compaction* into a stamped recall message."""
        return HarnessMessage(content=compaction.to_content(), created_at=compaction.compacted_at)

    def _build_summary_messages(
        self, turn_summaries: list[ChatTurnSummary]
    ) -> list[SummaryMessage]:
        """Convert *turn_summaries* into stamped recall messages, oldest first.

        Returns an empty list when *turn_summaries* is empty.
        """
        return [
            SummaryMessage(
                content=summary.to_content(),
                created_at=summary.turn_end_timestamp,
                turn_start_timestamp=summary.turn_start_timestamp,
                turn_end_timestamp=summary.turn_end_timestamp,
            )
            for summary in turn_summaries
        ]

    def cancel_pending_tasks(self) -> None:
        """Discard any pending task without recording it."""
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None

    async def flush_pending_tasks(self) -> ChatTurnSummary | None:
        """Await and clear the pending tasks, returning its result or ``None``."""
        if self._pending_task is None:
            return None
        task, self._pending_task = self._pending_task, None
        try:
            return await task
        except asyncio.CancelledError:
            return None
        except Exception:
            logger.exception("Background summarization task failed")
            return None

    async def build(
        self,
        messages: list[BaseMessage],
        turn_summaries: list[ChatTurnSummary],
        chat_history_compaction: "ChatHistoryCompaction | None",
    ) -> ChatTurnContext:
        """Build the :class:`ChatTurnContext` for the current LLM call."""
        summaries = list(turn_summaries)

        record = await self.flush_pending_tasks()
        if record is not None:
            summaries.append(record)
        summaries = summaries[-self._window_size :]

        # Once the summary window is full the compaction is no longer needed —
        # recent summaries cover enough history on their own.
        if len(summaries) >= self._window_size:
            chat_history_compaction = None

        messages = list(messages)
        if (
            self._loop_compactor is not None
            and self._midturn_compaction_threshold is not None
            and is_ongoing_turn(messages)
            and self._loop_compactor.estimate_tokens(messages) > self._midturn_compaction_threshold
        ):
            messages = await self._loop_compactor.compact(messages)

        recap_messages: list[HarnessMessage | SummaryMessage] = []
        if chat_history_compaction is not None:
            recap_messages.append(self._build_compaction_message(chat_history_compaction))
        recap_messages.extend(self._build_summary_messages(summaries))

        plan_messages: list[PlanMessage] = []
        if self._context_store is not None and self._session_id is not None:
            plan = self._context_store.get_current_plan(self._session_id)
            if plan is not None:
                plan_messages = [self._build_plan_message(plan)]

        combined = recap_messages + plan_messages + messages
        return ChatTurnContext(
            messages=[m.render() if isinstance(m, RenderableMessageMixin) else m for m in combined],
            turn_summaries=summaries,
            chat_history_compaction=chat_history_compaction,
        )
