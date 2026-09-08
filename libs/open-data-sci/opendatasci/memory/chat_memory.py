"""Agent-level chat memory: rolling turn summaries and per-call context assembly."""

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import Field

from opendatasci._utils.message_utils import (
    get_final_ai_message,
    get_message_text_content,
    get_thoughts,
    render_turn,
)
from opendatasci._utils.mixins import LLMDigestibleMixin
from opendatasci._utils.pydantic_utils import (
    FrozenBaseModel,
    FrozenStrictBaseModel,
    MutableStrictBaseModel,
)
from opendatasci.memory.messages import (
    TaskMessage,
    UserMessage,
    get_turn_end_timestamp,
    get_turn_start_timestamp,
)
from opendatasci.models.factory import bind_structured_output
from opendatasci.prompts.prompt_templates import (
    CHAT_COMPACTOR_SYSTEM_PROMPT,
    TURN_SUMMARIZER_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn summaries
# ---------------------------------------------------------------------------


class TurnStepBatchSummary(FrozenStrictBaseModel):
    """Summary of one logical batch of consecutive steps within a turn.

    A batch is however many consecutive steps served one identifiable
    sub-goal (e.g. several exploratory reads, a failed attempt followed by a
    successful retry) — not a single raw tool call.
    """

    goal: str
    actions: str
    outcome: str
    key_observations: str
    artifacts: list[str]

    def render(self, index: int) -> str:
        artifacts = "; ".join(self.artifacts) if self.artifacts else "No artifacts"
        return (
            f'    <batch index="{index}">\n'
            f"      <goal>{self.goal}</goal>\n"
            f"      <actions>{self.actions}</actions>\n"
            f"      <outcome>{self.outcome}</outcome>\n"
            f"      <key_observations>{self.key_observations}</key_observations>\n"
            f"      <artifacts>{artifacts}</artifacts>\n"
            f"    </batch>"
        )


class ChatTurnSummary(MutableStrictBaseModel, LLMDigestibleMixin):
    """Summary of a single completed conversation turn."""

    # Metadata
    turn_start_timestamp: datetime
    turn_end_timestamp: datetime
    # Content
    user_message: str
    step_batches: list[TurnStepBatchSummary]
    agent_response: str

    def to_content(self) -> str:
        batches = (
            "\n".join(batch.render(i) for i, batch in enumerate(self.step_batches, start=1))
            if self.step_batches
            else "    (no steps)"
        )
        return (
            f"<summary_content>\n"
            f"  <user_message>{self.user_message}</user_message>\n"
            f"  <step_batches>\n"
            f"{batches}\n"
            f"  </step_batches>\n"
            f"  <agent_response>{self.agent_response}</agent_response>\n"
            f"</summary_content>"
        )


class ChatHistoryCompaction(MutableStrictBaseModel, LLMDigestibleMixin):
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


_MAX_STEP_BATCHES: int = 16


class _TurnStepBatchSummaryOutput(FrozenBaseModel):
    goal: str = Field(description="One sentence: what sub-goal was this batch of steps pursuing?")
    actions: str = Field(
        description="What actions the agent took to get there — approaches or tools used. May span several steps."
    )
    outcome: str = Field(
        description=(
            "The outcome of those actions: what worked, what didn't, and any results or "
            "errors produced. If an attempt failed and was retried differently, say so "
            "explicitly rather than only reporting the final state."
        )
    )
    key_observations: str = Field(
        description=(
            "Any notable facts, constraints, or discoveries surfaced in this batch that "
            "aren't already captured by goal/actions/outcome — e.g. a surprising finding, "
            "a caveat the user should know about."
        )
    )
    artifacts: list[str] = Field(
        description=(
            "Paths of files this batch created or modified, especially anything written "
            "under .opendatasci/artifacts/. Empty if nothing was created or modified."
        )
    )


class _ChatTurnSummaryOutput(FrozenBaseModel):
    step_batches: list[_TurnStepBatchSummaryOutput] = Field(
        max_length=_MAX_STEP_BATCHES,
        description=(
            "Logically segment the turn into consecutive batches of steps, one batch per "
            "identifiable sub-goal — not one entry per tool call. Draw a new batch boundary "
            f"only when the apparent sub-goal changes. Keep at most {_MAX_STEP_BATCHES} "
            "batches; if more occurred, keep the most consequential ones and fold the rest "
            "into the last batch's outcome."
        ),
    )


def _has_tool_calls(turn_messages: list[BaseMessage]) -> bool:
    """Return ``True`` if any message in *turn_messages* is an AIMessage with tool calls."""
    return any(isinstance(msg, AIMessage) and msg.tool_calls for msg in turn_messages)


def _build_fallback_step_batches(turn_messages: list[BaseMessage]) -> list[TurnStepBatchSummary]:
    """Build one degenerate (batch-size-1) :class:`TurnStepBatchSummary` per raw tool call.

    Used only when no summarizer LLM is available. Sub-goal segmentation is an LLM-only
    judgment call, so this can't reproduce true batching — it preserves the trace instead
    of discarding it.
    """
    tool_results: dict[str, str] = {}
    for msg in turn_messages:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tool_results[msg.tool_call_id] = content

    batches: list[TurnStepBatchSummary] = []
    for msg in turn_messages:
        if not (isinstance(msg, AIMessage) and msg.tool_calls):
            continue
        goal = get_thoughts(msg).strip()
        for tc in msg.tool_calls:
            outcome = tool_results.get(tc.get("id") or "")
            batches.append(
                TurnStepBatchSummary(
                    goal=goal,
                    actions=f"{tc['name']}({tc.get('args', {})})",
                    outcome=outcome.strip() if outcome else "(no result captured)",
                    key_observations="(none)",
                    artifacts=[],
                )
            )
    return batches


class ChatTurnSummarizer:
    """Summarizes a single completed agent turn into a :class:`ChatTurnSummary`."""

    def __init__(self, summarizer_llm: BaseChatModel | None) -> None:
        self._structured_llm: Any = None
        if summarizer_llm is not None:
            try:
                self._structured_llm = bind_structured_output(
                    summarizer_llm, _ChatTurnSummaryOutput
                )
            except Exception:
                logger.warning(
                    "Could not bind structured output to summarizer LLM; summarization disabled",
                    exc_info=True,
                )

    def _build_llm_context(self, turn_messages: list[BaseMessage]) -> list[BaseMessage]:
        return [
            SystemMessage(content=TURN_SUMMARIZER_SYSTEM_PROMPT),
            HumanMessage(content=render_turn(turn_messages)),
        ]

    async def summarize_turn(self, turn_messages: list[BaseMessage]) -> ChatTurnSummary | None:
        """Summarize *turn_messages* into a :class:`ChatTurnSummary`, or ``None`` for an empty turn.

        ``user_message`` and ``agent_response`` are always preserved verbatim — never
        paraphrased by the LLM. Only ``step_batches`` depends on the summarizer LLM;
        without one (or if it fails), a degenerate per-tool-call batching is used instead.

        A turn with no tool calls skips the summarizer LLM entirely: the user's message
        and the agent's response are the whole story already, so there's nothing a
        step-batch summary would add.
        """
        if not turn_messages:
            raise ValueError("Cannot summarize an empty turn")

        turn_start_timestamp = get_turn_start_timestamp(turn_messages)
        turn_end_timestamp = get_turn_end_timestamp(turn_messages) or turn_start_timestamp

        user_msg = turn_messages[0]
        if not isinstance(user_msg, (UserMessage, TaskMessage)):
            raise ValueError("First message in turn is not a UserMessage or TaskMessage")
        final_ai_msg = get_final_ai_message(turn_messages)

        user_message = get_message_text_content(user_msg)
        agent_response = get_message_text_content(final_ai_msg)

        if self._structured_llm is not None and _has_tool_calls(turn_messages):
            try:
                context = self._build_llm_context(turn_messages)
                output: _ChatTurnSummaryOutput = await self._structured_llm.ainvoke(context)
                return ChatTurnSummary(
                    turn_start_timestamp=turn_start_timestamp,
                    turn_end_timestamp=turn_end_timestamp,
                    user_message=user_message,
                    step_batches=[
                        TurnStepBatchSummary(
                            goal=batch.goal,
                            actions=batch.actions,
                            outcome=batch.outcome,
                            key_observations=batch.key_observations,
                            artifacts=batch.artifacts,
                        )
                        for batch in output.step_batches
                    ],
                    agent_response=agent_response,
                )
            except Exception:
                logger.exception("Summarizer failed, using fallback")

        return ChatTurnSummary(
            turn_start_timestamp=turn_start_timestamp,
            turn_end_timestamp=turn_end_timestamp,
            user_message=user_message,
            step_batches=_build_fallback_step_batches(turn_messages),
            agent_response=agent_response,
        )


# ---------------------------------------------------------------------------
# Per-call turn context
# ---------------------------------------------------------------------------


class ChatTurnContext(MutableStrictBaseModel):
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
            HumanMessage(content="\n\n".join(parts)),
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
