from abc import ABC
from dataclasses import dataclass, field
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from langgraph.types import Interrupt

from opendatasci.agents.chat_memory import TurnSummaryRecord
from opendatasci.skills.base import Skill


@dataclass
class BaseAgentState(ABC):
    """Base class for all agent states."""

    interrupts: list[Interrupt] = field(default_factory=list)


@dataclass
class AgentState(BaseAgentState):
    """Shared state passed between nodes in the main agent graph."""

    messages: Annotated[list[Any], add_messages] = field(default_factory=list)
    active_skills: list[Skill] = field(default_factory=list)
    is_plan_mode: bool = False
    is_self_review_mode: bool = False
    turn_summaries: list[TurnSummaryRecord] = field(default_factory=list)
    session_preamble: str | None = None


@dataclass
class WorkerAgentState(BaseAgentState):
    """Shared state passed between nodes in a worker agent graph."""

    messages: list[Any] = field(default_factory=list)
    active_skills: list[Skill] = field(default_factory=list)
