"""Agent-level memory: rolling conversation history and turn summarization."""

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from opendatasci._utils.langchain_utils import (
    get_final_ai_message,
    get_last_turn_messages,
    get_message_text_content,
    get_ongoing_turn_messages,
    is_ongoing_turn,
    render_turn,
    render_turns,
)

if TYPE_CHECKING:
    from opendatasci.agents.turn_memory import AgentLoopCompactor

logger = logging.getLogger(__name__)

__all__ = [
    "ChatHistoryBuilder",
    "ChatHistoryCompactor",
    "PreparedHistory",
    "TurnSummaryRecord",
    "TurnSummarizer",
    "render_memory",
    "extract_thinking_and_text",
]


_CHAT_MEMORY_WINDOW_SIZE: int = 3


class ChatHistoryCompactor:
    """Compacts older conversation turns into a summary, preserving recent turns verbatim."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    @staticmethod
    def _validate(chat_history: list[BaseMessage]) -> list[list[BaseMessage]]:
        """Parse and validate *chat_history*, returning a list of turns.

        Raises:
            ValueError: if the history contains an ongoing last turn, a message
                that cannot belong to any turn, or an incomplete final turn.
        """
        if is_ongoing_turn(chat_history):
            raise ValueError("Chat history has an ongoing (incomplete) last turn")

        turns: list[list[BaseMessage]] = []
        current_turn: list[BaseMessage] | None = None

        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                if current_turn is not None:
                    raise ValueError(
                        "Encountered a HumanMessage before the previous turn ended "
                        "(missing final AIMessage without tool calls)"
                    )
                current_turn = [msg]
            elif current_turn is None:
                raise ValueError(f"Unexpected {type(msg).__name__} before the first HumanMessage")
            else:
                current_turn.append(msg)
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    turns.append(current_turn)
                    current_turn = None

        if current_turn is not None:
            raise ValueError("Last turn is incomplete (no final AIMessage without tool calls)")

        return turns

    async def compact(
        self,
        chat_history: list[BaseMessage],
        cutoff: int = 1,
    ) -> list[BaseMessage]:
        """Compact all turns except the last *cutoff* turns.

        Turns before the cutoff are summarised by the LLM and replaced by a
        single SystemMessage.  The last *cutoff* turns are kept verbatim.

        Returns *chat_history* unchanged when there are not enough turns to
        compact (i.e. ``len(turns) <= cutoff``).

        Args:
            chat_history: The full conversation message list to compact.
            cutoff: Number of most-recent turns to keep uncompacted.  Defaults
                to ``1``, meaning only the very last turn is preserved verbatim.

        Returns:
            A new message list with the older turns replaced by a summary followed
            by the verbatim kept turns, or the original list if nothing needed
            compacting.
        """
        from opendatasci.prompts.prompt_templates import (
            CHAT_COMPACTOR_SYSTEM_PROMPT,  # noqa: PLC0415
        )

        turns = self._validate(chat_history)

        if len(turns) <= cutoff:
            return list(chat_history)

        turns_to_compact = turns[:-cutoff] if cutoff > 0 else turns
        kept_turns = turns[-cutoff:] if cutoff > 0 else []

        rendered = render_turns(turns_to_compact)

        response = await self._llm.ainvoke(
            [
                SystemMessage(content=CHAT_COMPACTOR_SYSTEM_PROMPT),
                HumanMessage(content=rendered),
            ]
        )
        summary = response.content if isinstance(response.content, str) else str(response.content)

        kept_messages: list[BaseMessage] = [msg for turn in kept_turns for msg in turn]
        compaction_message = SystemMessage(
            content=f"<compacted_history>\n{summary}\n</compacted_history>"
        )
        return [compaction_message, *kept_messages]


# ---------------------------------------------------------------------------
# Turn summaries
# ---------------------------------------------------------------------------


@dataclass
class TurnSummaryRecord:
    """Summary of a single completed conversation turn."""

    turn: int
    user: str
    actions: str
    agent: str
    timestamp: str = ""

    def format(self) -> str:
        ts_line = f"- Timestamp: {self.timestamp}\n" if self.timestamp else ""
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


def render_memory(
    preamble: str | None,
    turn_summaries: list[TurnSummaryRecord],
) -> str:
    """Render the session preamble and recent turn summaries as Markdown.

    Returns an empty string when there is nothing to render.
    """
    parts: list[str] = []
    if preamble:
        parts.append(f"## Previous Session Summary\n\n{preamble}")
    if turn_summaries:
        lines = ["## Recent Conversation History", ""]
        for summary in turn_summaries:
            lines.append(summary.format())
            lines.append("")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


@dataclass
class PreparedHistory:
    """The assembled conversation context for a single LLM call.

    Attributes:
        messages: Human/AI/Tool messages only — never contains SystemMessages.
            Mid-turn compaction is applied when the ongoing turn exceeds the budget.
        memory_text: Rendered recall context (preamble + turn summaries) as a plain
            string, to be passed to SystemContextBuilder. None when there is nothing
            to recall.
        turn_summaries: Updated rolling summary list to write back to agent state.
    """

    messages: list[BaseMessage]
    memory_text: str | None
    turn_summaries: list[TurnSummaryRecord]


class TurnSummary(BaseModel):
    user_request: str = Field(
        description="One sentence: what did the user ask for? Include specific names, columns, files, or constraints."
    )
    outcomes: str = Field(
        description="Bullet points: what concretely resulted — numbers, metrics, errors, conclusions, anything produced. No filler."
    )
    agent_response: str = Field(
        description="One or two sentences: what answer or conclusion was given to the user? Be specific."
    )


class TurnSummarizer:
    """Summarizes a single completed agent turn into a :class:`TurnSummaryRecord`."""

    def __init__(self, summarizer_llm: Any) -> None:
        self._structured_llm: Any = None
        if summarizer_llm is not None:
            try:
                self._structured_llm = summarizer_llm.with_structured_output(TurnSummary)
            except Exception:
                logger.warning(
                    "Could not bind structured output to summarizer LLM; summarization disabled",
                    exc_info=True,
                )

    async def summarize_turn(self, turn: list[BaseMessage]) -> TurnSummaryRecord | None:
        """Summarize *turn* and return a record, falling back to raw text on failure.

        The returned record's ``turn`` index is left as ``0``; the caller assigns
        the running turn number. Returns ``None`` for an empty turn.
        """
        if not turn:
            return None

        opening = turn[0]
        timestamp = opening.additional_kwargs.get("timestamp") or ""

        if self._structured_llm is not None:
            try:
                from opendatasci.prompts.prompt_templates import (
                    TURN_SUMMARIZER_SYSTEM_PROMPT,  # noqa: PLC0415
                )

                summary: TurnSummary = await self._structured_llm.ainvoke(
                    [
                        SystemMessage(content=TURN_SUMMARIZER_SYSTEM_PROMPT),
                        HumanMessage(content=render_turn(turn)),
                    ]
                )
                return TurnSummaryRecord(
                    turn=0,
                    user=summary.user_request,
                    actions=summary.outcomes,
                    agent=summary.agent_response,
                    timestamp=timestamp,
                )
            except Exception:
                logger.exception("Summarizer failed, using fallback")

        try:
            final_response = get_message_text_content(get_final_ai_message(turn)).strip()
        except ValueError:
            final_response = ""
        return TurnSummaryRecord(
            turn=0,
            user=get_message_text_content(opening),
            actions="(summary unavailable)",
            agent=final_response,
            timestamp=timestamp,
        )


class ChatHistoryBuilder:
    """Builds the per-call conversation history from agent state.

    Handles three concerns in sequence inside :meth:`build`:

    1. Flushing any pending background turn summary and updating the rolling window.
    2. Applying mid-turn compaction when the ongoing turn exceeds the token budget.
    3. Rendering the recalled context (preamble + summaries) as plain text for the
       system prompt — keeping it out of the message list entirely.

    Inject an :class:`~opendatasci.agents.turn_memory.AgentLoopCompactor` and a
    threshold to enable mid-turn compaction; omit both to disable it.
    """

    def __init__(
        self,
        summarizer: TurnSummarizer,
        loop_compactor: "AgentLoopCompactor | None" = None,
        midturn_compaction_threshold: int | None = None,
        window_size: int = _CHAT_MEMORY_WINDOW_SIZE,
    ) -> None:
        self._summarizer = summarizer
        self._loop_compactor = loop_compactor
        self._midturn_compaction_threshold = midturn_compaction_threshold
        self._window_size = window_size
        self._pending_task: asyncio.Task[TurnSummaryRecord | None] | None = None

    def schedule_turn_summarization(self, messages: list[BaseMessage]) -> None:
        """Schedule background summarization of the last completed turn in *messages*.

        A no-op when *messages* contains no turns. Raises when the last turn is
        still in progress.

        Raises:
            ValueError: if the last turn is still ongoing (incomplete).
        """
        turn = get_last_turn_messages(messages)
        if not turn:
            return
        if is_ongoing_turn(turn):
            raise ValueError("Cannot summarize an ongoing (incomplete) turn")
        self._pending_task = asyncio.create_task(self._summarizer.summarize_turn(turn))

    def cancel_pending(self) -> None:
        """Discard any pending summarization without recording it."""
        if self._pending_task is not None:
            self._pending_task.cancel()
            self._pending_task = None

    async def flush(self) -> TurnSummaryRecord | None:
        """Await and clear the pending summarization task.

        Returns ``None`` when there is no pending task or the task failed;
        exceptions are swallowed and logged.
        """
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
        turn_summaries: list[TurnSummaryRecord],
        preamble: str | None = None,
    ) -> PreparedHistory:
        """Build the :class:`PreparedHistory` for the current LLM call.

        Flushes any pending summary, trims the rolling window, applies mid-turn
        compaction if needed, and renders the recalled context as plain text.
        The returned :attr:`PreparedHistory.messages` never contains SystemMessages.
        """
        summaries = list(turn_summaries)

        record = await self.flush()
        if record is not None:
            next_turn = summaries[-1].turn + 1 if summaries else 1
            summaries.append(replace(record, turn=next_turn))
        summaries = summaries[-self._window_size :]

        history = list(messages)
        if self._loop_compactor is not None and self._midturn_compaction_threshold is not None:
            try:
                turn = get_ongoing_turn_messages(history)
            except ValueError:
                turn = []
            if (
                turn
                and self._loop_compactor.estimate_tokens(turn) > self._midturn_compaction_threshold
            ):
                compacted = await self._loop_compactor.compact(turn)
                n = len(turn)
                history = history[:-n] + compacted

        memory_text = render_memory(preamble, summaries) or None
        return PreparedHistory(messages=history, memory_text=memory_text, turn_summaries=summaries)


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
