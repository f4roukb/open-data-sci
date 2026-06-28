"""Agent-level chat memory: rolling turn summaries and per-call context assembly."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, Field

from opendatasci._utils.message_utils import (
    get_final_ai_message,
    get_message_text_content,
    render_turn,
)
from opendatasci._utils.mixins import LLMDigestibleMixin
from opendatasci.memory.messages import (
    HarnessMessage,
    get_turn_end_timestamp,
    get_turn_start_timestamp,
    is_user_message,
)
from opendatasci.prompts.prompt_templates import (
    CHAT_COMPACTOR_SYSTEM_PROMPT,
    TURN_SUMMARIZER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn summaries
# ---------------------------------------------------------------------------


@dataclass
class ChatTurnSummary(LLMDigestibleMixin):
    """Summary of a single completed conversation turn."""

    # Metadata
    turn_start_timestamp: datetime
    turn_end_timestamp: datetime
    # Content
    user_message_summary: str
    actions_summary: str
    agent_response_summary: str

    def to_content(self) -> str:
        return (
            f"<summary_content>\n"
            f"  <user_request>{self.user_message_summary}</user_request>\n"
            f"  <outcomes>{self.actions_summary}</outcomes>\n"
            f"  <agent_response>{self.agent_response_summary}</agent_response>\n"
            f"</summary_content>"
        )


@dataclass
class ChatHistoryCompaction(LLMDigestibleMixin):
    """A folded compaction of multiple :class:`ChatTurnSummary` records.

    Produced by :class:`ChatHistoryCompactor` when the user explicitly requests
    history compaction. Unlike a per-turn summary this record has no user
    message or structured fields — only a free-form LLM-generated narrative and
    the time range it covers.
    """

    compacted_at: datetime
    timespan: tuple[datetime, datetime] | None
    content: str

    def to_content(self) -> str:
        span_from = self.timespan[0] if self.timespan else None
        span_to = self.timespan[1] if self.timespan else None
        return (
            f"<compaction_metadata>\n"
            f"  <compacted_at>{self.compacted_at}</compacted_at>\n"
            f"  <covers_from>{span_from}</covers_from>\n"
            f"  <covers_to>{span_to}</covers_to>\n"
            f"</compaction_metadata>\n"
            f"<compaction_content>\n"
            f"{self.content}\n"
            f"</compaction_content>"
        )


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

    def _build_llm_context(self, turn_messages: list[BaseMessage]) -> list[BaseMessage]:
        return [
            SystemMessage(content=TURN_SUMMARIZER_SYSTEM_PROMPT),
            HarnessMessage(content=render_turn(turn_messages)),
        ]

    async def summarize_turn(self, turn_messages: list[BaseMessage]) -> ChatTurnSummary | None:
        """Summarize *turn_messages* into a :class:`ChatTurnSummary`, or ``None`` for an empty turn."""
        if not turn_messages:
            raise ValueError("Cannot summarize an empty turn")

        turn_start_timestamp = get_turn_start_timestamp(turn_messages)
        turn_end_timestamp = get_turn_end_timestamp(turn_messages) or turn_start_timestamp

        if self._structured_llm is not None:
            try:
                context = self._build_llm_context(turn_messages)
                output: ChatTurnSummaryOutput = await self._structured_llm.ainvoke(context)
                return ChatTurnSummary(
                    turn_start_timestamp=turn_start_timestamp,
                    turn_end_timestamp=turn_end_timestamp,
                    user_message_summary=output.user_request,
                    actions_summary=output.outcomes,
                    agent_response_summary=output.agent_response,
                )
            except Exception:
                logger.exception("Summarizer failed, using fallback")

        user_msg = turn_messages[0]
        if not is_user_message(user_msg):
            raise ValueError("First message in turn is not a user message")
        final_ai_msg = get_final_ai_message(turn_messages)

        user_msg_text_content = get_message_text_content(user_msg)
        final_ai_msg_text_content = get_message_text_content(final_ai_msg)

        return ChatTurnSummary(
            turn_start_timestamp=turn_start_timestamp,
            turn_end_timestamp=turn_end_timestamp,
            user_message_summary=user_msg_text_content,
            actions_summary="N/A",
            agent_response_summary=final_ai_msg_text_content,
        )


# ---------------------------------------------------------------------------
# Per-call turn context
# ---------------------------------------------------------------------------


@dataclass
class ChatTurnContext:
    """The assembled messages for a single LLM call.

    Attributes:
        messages: Compaction recall (if any), turn-summary recall messages, the current
            plan (if any), and the ongoing turn's messages — all rendered for the LLM.
        turn_summaries: Updated rolling summary list to write back to agent state.
        chat_history_compaction: Updated compaction to write back to agent state.
            ``None`` means either no compaction exists or it was cleared because the
            summary window became full and the compaction is no longer needed.
    """

    messages: list[BaseMessage]
    turn_summaries: list[ChatTurnSummary]
    chat_history_compaction: "ChatHistoryCompaction | None"


# ---------------------------------------------------------------------------
# Explicit history compaction
# ---------------------------------------------------------------------------


class ChatHistoryCompactor:
    """Folds turn summaries and completed turns into a single :class:`ChatHistoryCompaction`."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def _build_llm_context(
        self,
        existing_compaction: ChatHistoryCompaction | None,
        turn_summaries: list[ChatTurnSummary],
        completed_messages: list[BaseMessage],
    ) -> list[BaseMessage]:
        parts: list[str] = []
        if existing_compaction is not None:
            parts.append(existing_compaction.to_content())
        for summary in turn_summaries:
            parts.append(summary.to_content())
        if completed_messages:
            parts.append(render_turn(completed_messages))
        return [
            SystemMessage(content=CHAT_COMPACTOR_SYSTEM_PROMPT),
            HarnessMessage(content="\n\n".join(parts)),
        ]

    async def compact(
        self,
        existing_compaction: ChatHistoryCompaction | None,
        turn_summaries: list[ChatTurnSummary],
        completed_messages: list[BaseMessage],
    ) -> ChatHistoryCompaction:
        """Fold all inputs into one new :class:`ChatHistoryCompaction`.

        The inputs are fed to the LLM in order: the existing compaction (if any),
        then the turn summaries, then the completed turn messages.

        Raises:
            ValueError: if all inputs are empty (nothing to compact).
        """
        if not existing_compaction and not turn_summaries and not completed_messages:
            raise ValueError("Nothing to compact")

        timespan_starts: list[datetime] = []
        timespan_ends: list[datetime] = []

        if existing_compaction is not None and existing_compaction.timespan is not None:
            timespan_starts.append(existing_compaction.timespan[0])
            timespan_ends.append(existing_compaction.timespan[1])
        for summary in turn_summaries:
            timespan_starts.append(summary.turn_start_timestamp)
            timespan_ends.append(summary.turn_end_timestamp)
        if completed_messages:
            completed_start = get_turn_start_timestamp(completed_messages)
            timespan_starts.append(completed_start)
            timespan_ends.append(get_turn_end_timestamp(completed_messages) or completed_start)

        timespan = (
            (min(timespan_starts), max(timespan_ends))
            if timespan_starts and timespan_ends
            else None
        )

        response = await self._llm.ainvoke(
            self._build_llm_context(existing_compaction, turn_summaries, completed_messages)
        )
        text = response.content if isinstance(response.content, str) else str(response.content)
        return ChatHistoryCompaction(
            compacted_at=datetime.now(timezone.utc),
            timespan=timespan,
            content=text,
        )
