"""AgentLoopCompactor: LLM-based in-turn context compaction for ReAct loops."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import BaseModel

from opendatasci._utils.message_utils import render_turn
from opendatasci.memory.messages import HarnessMessage, is_ongoing_turn
from opendatasci.prompts.prompt_templates import MIDTURN_COMPACTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class _CompactedAgentLoop(BaseModel):
    content: str

    def to_message(self) -> SystemMessage:
        """Return the compacted steps as a SystemMessage for injection into the turn."""
        return SystemMessage(
            content=f"<compacted_agent_loop>\n{self.content}\n</compacted_agent_loop>"
        )


class AgentLoopCompactor:
    """Reduces an in-progress agent turn to fit within the token budget.

    When a turn's intermediate steps exceed the configured threshold, the
    compactor summarises them before the next LLM call.  Only the prompt
    presented to the model is affected; the stored graph state is unchanged.
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._structured_llm = llm.with_structured_output(_CompactedAgentLoop)
        self._system_prompt = MIDTURN_COMPACTOR_SYSTEM_PROMPT

    def estimate_tokens(self, messages: list[BaseMessage]) -> int:
        """Return an approximate token count for *messages*."""
        return count_tokens_approximately(messages)

    async def compact(self, turn: list[BaseMessage]) -> list[BaseMessage]:
        """Compact *turn* by summarising intermediate steps and return the reduced message list.

        The original user message and the most recent agent step are always preserved;
        only the intermediate steps between them are replaced by a summary.

        Returns *turn* unchanged if it is not an ongoing turn, if there are no
        intermediate steps to compact, or if the LLM call fails.
        """
        if not is_ongoing_turn(turn):
            return turn

        # Find the last AIMessage in the turn.
        last_ai_idx = -1
        for i in range(len(turn) - 1, -1, -1):
            if isinstance(turn[i], AIMessage):
                last_ai_idx = i
                break

        if last_ai_idx == -1:
            return turn

        intermediate = turn[1:last_ai_idx]
        if not intermediate:
            return turn

        try:
            result: _CompactedAgentLoop = await self._structured_llm.ainvoke(
                [
                    SystemMessage(content=self._system_prompt),
                    HarnessMessage(content=render_turn(intermediate)),
                ]
            )

            user_message: BaseMessage = turn[0]
            compaction_message: BaseMessage = result.to_message()
            rest_of_turn: list[BaseMessage] = turn[last_ai_idx:]
            return [user_message, compaction_message, *rest_of_turn]
        except Exception:
            logger.exception("AgentLoopCompactor LLM call failed; return uncompacted turn")
            return turn


class TurnRewinder:
    """Removes turns from a conversation history."""

    def rewind_last_turn(
        self,
        chat_history: list[BaseMessage],
        keep_user_message: bool = False,
    ) -> list[BaseMessage]:
        """Remove the last turn from *chat_history*.

        The last turn is rewound regardless of whether it has completed — an
        in-progress turn (started but no final agent response yet) is treated
        the same as a completed one.

        By default the human message that opened the turn is also dropped.
        Pass ``keep_user_message=True`` to retain it and drop only the agent
        response and any intermediate tool messages.

        Args:
            chat_history: The full conversation message list.
            keep_user_message: When ``True``, the user message that opened
                the last turn is retained. Defaults to ``False``.

        Returns:
            A new list with the last turn removed, or a copy of
            *chat_history* unchanged if no turn start is found.
        """
        # Locate the HumanMessage that opened the last turn.
        start_idx = -1
        for i in range(len(chat_history) - 1, -1, -1):
            if isinstance(chat_history[i], HumanMessage):
                start_idx = i
                break

        if start_idx == -1:
            return list(chat_history)

        cut = start_idx + 1 if keep_user_message else start_idx
        return list(chat_history[:cut])
