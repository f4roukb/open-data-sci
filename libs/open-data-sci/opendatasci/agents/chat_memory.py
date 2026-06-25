"""Agent-level chat memory: rolling turn summaries and per-call context assembly."""

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from opendatasci._utils.langchain_utils import (
    get_final_ai_message,
    get_message_text_content,
    is_ongoing_turn,
    render_turn,
)
from opendatasci._utils.mixins import LLMDigestibleMixin
from opendatasci.agents._chat_messages import ChatMessageOrigin, HumanMessageMetadata
from opendatasci.context.base import BaseContextStore
from opendatasci.context.plans import Plan

logger = logging.getLogger(__name__)

__all__ = [
    "ChatHistoryBuilder",
    "ChatHistoryCompactor",
    "ChatMessageOrigin",
    "ChatTurnContext",
    "ChatTurnSummary",
    "ChatTurnSummarizer",
    "build_chat_recap_messages",
    "build_plan_message",
    "stamp_chat_message_metadata",
    "render_messages_for_llm",
    "extract_thinking_and_text",
]


_CHAT_TURN_SUMMARY_WINDOW_SIZE: int = 3


# ---------------------------------------------------------------------------
# Message provenance tagging
# ---------------------------------------------------------------------------


def stamp_chat_message_metadata(
    message: HumanMessage,
    origin: ChatMessageOrigin,
    created_at: datetime | None = None,
) -> HumanMessage:
    """Attach *origin* and *created_at* to *message*, leaving already-set fields unchanged."""
    metadata = HumanMessageMetadata.from_message(message)
    if metadata.origin is not None and metadata.created_at is not None:
        return message
    updated = metadata.model_copy(
        update={
            "origin": metadata.origin if metadata.origin is not None else origin,
            "created_at": metadata.created_at
            if metadata.created_at is not None
            else (created_at or datetime.now(timezone.utc)),
        }
    )
    return updated.attach_to(message)


def render_human_message_for_llm(message: HumanMessage) -> HumanMessage:
    """Return a copy of *message* with its provenance tag prepended to the content."""
    tag = HumanMessageMetadata.from_message(message).to_content()
    return HumanMessage(content=f"{tag}\n{message.content}")


def render_messages_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Render every ``HumanMessage`` in *messages* for the LLM; pass other message types through."""
    return [render_human_message_for_llm(m) if isinstance(m, HumanMessage) else m for m in messages]


def build_plan_message(plan: Plan) -> HumanMessage:
    """Build a recall message for *plan*, preserving its original timestamp."""
    message = HumanMessage(content=plan.to_content())
    created_at = plan.metadata.get("created_at")
    return stamp_chat_message_metadata(
        message,
        ChatMessageOrigin.HARNESS,
        created_at=datetime.fromisoformat(created_at) if created_at else None,
    )


# ---------------------------------------------------------------------------
# Turn summaries
# ---------------------------------------------------------------------------


@dataclass
class ChatTurnSummary(LLMDigestibleMixin):
    """Summary of a single completed conversation turn.

    A ``turn`` value of ``None`` indicates a folded compaction summary (produced
    by :class:`ChatHistoryCompactor`) rather than a regular per-turn record.
    """

    turn: int | None
    user: str
    actions: str
    agent: str
    timestamp: str = ""

    def to_content(self) -> str:
        ts_line = f"- Timestamp: {self.timestamp}\n" if self.timestamp else ""
        if self.turn is None:
            return f"**Compacted Summary:**\n{ts_line}{self.agent}"
        if "\n" in self.actions:
            indented = self.actions.replace("\n", "\n  ")
            actions_line = f"- Outcomes:\n  {indented}"
        else:
            actions_line = f"- Outcomes: {self.actions}"
        return (
            f"**Turn {self.turn}:**\n"
            f"{ts_line}"
            f"- User request: {self.user}\n"
            f"{actions_line}\n"
            f"- Agent response: {self.agent}"
        )


def build_chat_recap_messages(turn_summaries: list[ChatTurnSummary]) -> list[HumanMessage]:
    """Convert *turn_summaries* into recall messages, oldest first.

    Returns an empty list when *turn_summaries* is empty.
    """
    messages: list[HumanMessage] = []
    for summary in turn_summaries:
        message = HumanMessage(content=summary.to_content())
        messages.append(
            stamp_chat_message_metadata(
                message,
                ChatMessageOrigin.HARNESS,
                created_at=datetime.fromisoformat(summary.timestamp) if summary.timestamp else None,
            )
        )
    return messages


class ChatTurnSummaryOutput(BaseModel):
    """Structured output the summarizer LLM produces for a single turn."""

    user_request: str = Field(
        description="One sentence: what did the user ask for? Include specific names, columns, files, or constraints."
    )
    outcomes: str = Field(
        description="Bullet points: what concretely resulted — numbers, metrics, errors, conclusions, anything produced. No filler."
    )
    agent_response: str = Field(
        description="One or two sentences: what answer or conclusion was given to the user? Be specific."
    )


class ChatTurnSummarizer:
    """Summarizes a single completed agent turn into a :class:`ChatTurnSummary`."""

    def __init__(self, summarizer_llm: Any) -> None:
        self._structured_llm: Any = None
        if summarizer_llm is not None:
            try:
                self._structured_llm = summarizer_llm.with_structured_output(ChatTurnSummaryOutput)
            except Exception:
                logger.warning(
                    "Could not bind structured output to summarizer LLM; summarization disabled",
                    exc_info=True,
                )

    async def summarize_turn(self, turn_messages: list[BaseMessage]) -> ChatTurnSummary | None:
        """Summarize *turn_messages* into a :class:`ChatTurnSummary`, or ``None`` for an empty turn."""
        if not turn_messages:
            return None

        opening = turn_messages[0]
        opening_created_at = HumanMessageMetadata.from_message(opening).created_at
        timestamp = opening_created_at.isoformat() if opening_created_at else ""

        if self._structured_llm is not None:
            try:
                from opendatasci.prompts.prompt_templates import (
                    TURN_SUMMARIZER_SYSTEM_PROMPT,  # noqa: PLC0415
                )

                output: ChatTurnSummaryOutput = await self._structured_llm.ainvoke(
                    [
                        SystemMessage(content=TURN_SUMMARIZER_SYSTEM_PROMPT),
                        HumanMessage(content=render_turn(turn_messages)),
                    ]
                )
                return ChatTurnSummary(
                    turn=0,
                    user=output.user_request,
                    actions=output.outcomes,
                    agent=output.agent_response,
                    timestamp=timestamp,
                )
            except Exception:
                logger.exception("Summarizer failed, using fallback")

        try:
            final_response = get_message_text_content(get_final_ai_message(turn_messages)).strip()
        except ValueError:
            final_response = ""
        return ChatTurnSummary(
            turn=0,
            user=get_message_text_content(opening),
            actions="(summary unavailable)",
            agent=final_response,
            timestamp=timestamp,
        )


# ---------------------------------------------------------------------------
# Per-call turn context
# ---------------------------------------------------------------------------


@dataclass
class ChatTurnContext:
    """The assembled messages for a single LLM call.

    Attributes:
        messages: Turn-summary recall messages, followed by the current plan (if any),
            followed by the ongoing turn's messages — all rendered for the LLM.
        turn_summaries: Updated rolling summary list to write back to agent state.
    """

    messages: list[BaseMessage]
    turn_summaries: list[ChatTurnSummary]


class ChatHistoryBuilder:
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
        from opendatasci.agents.turn_memory import AgentLoopCompactor  # noqa: PLC0415

        self._summarizer = ChatTurnSummarizer(summarizer_llm=summarizer_llm)
        self._loop_compactor = (
            AgentLoopCompactor(llm=loop_compactor_llm) if loop_compactor_llm is not None else None
        )
        self._midturn_compaction_threshold = midturn_compaction_threshold
        self._context_store = context_store
        self._session_id = session_id
        self._window_size = window_size
        self._pending_task: asyncio.Task[ChatTurnSummary | None] | None = None

    def schedule_turn_summarization(self, completed_turn_messages: list[BaseMessage]) -> None:
        """Schedule background summarization of *completed_turn_messages*.

        Raises:
            ValueError: if the turn is still in progress (incomplete).
        """
        if not completed_turn_messages:
            return
        if is_ongoing_turn(completed_turn_messages):
            raise ValueError("Cannot summarize an ongoing (incomplete) turn")
        self._pending_task = asyncio.create_task(
            self._summarizer.summarize_turn(completed_turn_messages)
        )

    def cancel_pending(self) -> None:
        """Discard any pending summarization without recording it."""
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None

    async def flush(self) -> ChatTurnSummary | None:
        """Await and clear the pending summarization task, returning its result or ``None``."""
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
        ongoing_turn_messages: list[BaseMessage],
        turn_summaries: list[ChatTurnSummary],
    ) -> ChatTurnContext:
        """Build the :class:`ChatTurnContext` for the current LLM call."""
        summaries = list(turn_summaries)

        record = await self.flush()
        if record is not None:
            next_turn = max((s.turn for s in summaries if s.turn is not None), default=0) + 1
            summaries.append(replace(record, turn=next_turn))
        summaries = summaries[-self._window_size :]

        turn_messages = list(ongoing_turn_messages)
        if (
            self._loop_compactor is not None
            and self._midturn_compaction_threshold is not None
            and is_ongoing_turn(turn_messages)
            and self._loop_compactor.estimate_tokens(turn_messages)
            > self._midturn_compaction_threshold
        ):
            turn_messages = await self._loop_compactor.compact(turn_messages)

        recap_messages = build_chat_recap_messages(summaries)
        plan_messages: list[HumanMessage] = []
        if self._context_store is not None and self._session_id is not None:
            plan = self._context_store.get_current_plan(self._session_id)
            if plan is not None:
                plan_messages = [build_plan_message(plan)]

        combined = recap_messages + plan_messages + turn_messages
        return ChatTurnContext(
            messages=render_messages_for_llm(combined),
            turn_summaries=summaries,
        )


# ---------------------------------------------------------------------------
# Explicit history compaction
# ---------------------------------------------------------------------------


class ChatHistoryCompactor:
    """Folds the rolling per-turn summaries into a single :class:`ChatTurnSummary`."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def compact(self, turn_summaries: list[ChatTurnSummary]) -> ChatTurnSummary:
        """Fold *turn_summaries* into one new compaction summary.

        Raises:
            ValueError: if *turn_summaries* is empty.
        """
        from opendatasci.prompts.prompt_templates import (
            CHAT_COMPACTOR_SYSTEM_PROMPT,  # noqa: PLC0415
        )

        if not turn_summaries:
            raise ValueError("Nothing to compact: no turn summaries")

        compaction_input = "\n\n".join(summary.to_content() for summary in turn_summaries)

        response = await self._llm.ainvoke(
            [
                SystemMessage(content=CHAT_COMPACTOR_SYSTEM_PROMPT),
                HumanMessage(content=compaction_input),
            ]
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        return ChatTurnSummary(
            turn=None,
            user="",
            actions="",
            agent=text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def extract_thinking_and_text(msg: AIMessage) -> tuple[str, str]:
    """Return ``(thinking, text)`` extracted from a model response message."""
    content = msg.content
    if isinstance(content, str):
        return "", content

    thinking_parts: list[str] = []
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif btype == "text":
                text_parts.append(block.get("text", ""))
        elif isinstance(block, str):
            text_parts.append(block)

    return "\n".join(thinking_parts), "\n".join(text_parts)
