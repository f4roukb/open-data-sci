from abc import ABC
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.types import Interrupt

from opendatasci._utils.langchain_utils import is_final_ai_message
from opendatasci.agents.chat_memory import ChatTurnSummary
from opendatasci.skills.base import Skill


def reduce_to_ongoing_turn(
    current_turn_messages: list[BaseMessage], incoming_messages: list[BaseMessage]
) -> list[BaseMessage]:
    """Keep ``AgentState.messages`` scoped to the current turn only.

    Appends *incoming_messages* (tool calls, tool results, the next agent step,
    or a maintenance ``RemoveMessage`` from ``rewind_turn``/``clear_chat_history``)
    while the turn is still in progress — same behaviour as LangGraph's
    ``add_messages``. Only when a new user message arrives *after* the current
    turn has completed (its last message is a final AIMessage with no pending
    tool calls — the same check ``graphs.py`` uses to route to ``end``) does the
    turn reset from scratch instead of accumulating onto the old one: completed
    turns live on only as :class:`ChatTurnSummary` entries, never as raw
    messages.
    """
    starts_new_turn = any(isinstance(m, HumanMessage) for m in incoming_messages)
    if current_turn_messages and starts_new_turn and is_final_ai_message(current_turn_messages[-1]):
        return cast(list[BaseMessage], add_messages([], incoming_messages))  # type: ignore[arg-type]
    return cast(list[BaseMessage], add_messages(current_turn_messages, incoming_messages))  # type: ignore[arg-type]


@dataclass
class BaseAgentState(ABC):
    """Base class for all agent states."""

    interrupts: list[Interrupt] = field(default_factory=list)


@dataclass
class AgentState(BaseAgentState):
    """Shared state passed between nodes in the main agent graph.

    ``messages`` holds only the *ongoing* conversation turn (see
    :func:`reduce_to_ongoing_turn`) — completed turns are folded into
    ``turn_summaries`` instead of accumulating here.
    """

    messages: Annotated[list[Any], reduce_to_ongoing_turn] = field(default_factory=list)
    active_skills: list[Skill] = field(default_factory=list)
    is_plan_mode: bool = False
    is_self_review_mode: bool = False
    turn_summaries: list[ChatTurnSummary] = field(default_factory=list)


@dataclass
class WorkerAgentState(BaseAgentState):
    """Shared state passed between nodes in a worker agent graph."""

    messages: list[Any] = field(default_factory=list)
    active_skills: list[Skill] = field(default_factory=list)
