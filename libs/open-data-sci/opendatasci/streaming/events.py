from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


@dataclass
class BaseAgentStreamEvent:
    """Base class for all streaming agent events."""

    type: ClassVar[str] = ""


@dataclass
class ReasoningEvent(BaseAgentStreamEvent):
    """Extended-thinking / reasoning token(s)."""

    type: ClassVar[str] = "reasoning"
    content: str = ""


@dataclass
class TokenEvent(BaseAgentStreamEvent):
    """Regular response text token."""

    type: ClassVar[str] = "token"
    content: str = ""


@dataclass
class ToolCallEvent(BaseAgentStreamEvent):
    """The agent is invoking a tool.

    ``worker_summaries`` is populated only for ``spawn_workers`` tool calls;
    ``summary`` carries the agent-provided summary argument for all other calls.
    """

    type: ClassVar[str] = "tool_call"
    content: str = ""
    tool: str = ""
    tool_call_id: str | None = None
    summary: str = ""
    worker_summaries: list[str] = field(default_factory=list)


@dataclass
class ToolCommunicationEvent(BaseAgentStreamEvent):
    """A progress message emitted by a tool before it returns."""

    type: ClassVar[str] = "tool_communication"
    content: str = ""
    tool_call_id: str = ""
    tool_name: str = ""


@dataclass
class ToolResultEvent(BaseAgentStreamEvent):
    """A tool returned a result."""

    type: ClassVar[str] = "tool_result"
    content: str = ""
    tool_call_id: str | None = None
    is_error: bool = False


@dataclass
class MessageEvent(BaseAgentStreamEvent):
    """A completed ``BaseMessage`` for callers that own conversation-history accumulation."""

    type: ClassVar[str] = "message"
    message: BaseMessage | None = None


@dataclass
class WorkerDoneEvent(BaseAgentStreamEvent):
    """A single parallel worker finished."""

    type: ClassVar[str] = "worker_done"
    worker_idx: int | None = None
    success: bool = True


@dataclass
class SubagentEvent(BaseAgentStreamEvent):
    """Lifecycle event from inside a running worker.

    ``event_type`` is one of ``"worker_tool_call"`` or ``"worker_tool_result"``.
    ``content`` carries the tool name for ``worker_tool_call`` events.
    """

    type: ClassVar[str] = "subagent_event"
    content: str = ""
    worker_idx: int | None = None
    event_type: str = ""
    success: bool = True
    summary: str = ""


@dataclass
class InputRequiredEvent(BaseAgentStreamEvent):
    """The agent is paused at an interrupt and needs input from the user.

    ``content`` is the question.  Call ``astream`` again with the user's
    answer to resume.
    """

    type: ClassVar[str] = "input_required"
    content: str = ""
    choices: list[str] = field(default_factory=list)


@dataclass
class UsageEvent(BaseAgentStreamEvent):
    """Per-call token usage.

    All fields are ``None`` when not reported by the underlying provider for
    this event (e.g. incremental estimates omit cache fields).
    """

    type: ClassVar[str] = "usage"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None


@dataclass
class ResponseEvent(BaseAgentStreamEvent):
    """Final assembled response for this turn (end-of-turn marker)."""

    type: ClassVar[str] = "response"
    content: str = ""


@dataclass
class ErrorEvent(BaseAgentStreamEvent):
    """An unrecoverable error occurred."""

    type: ClassVar[str] = "error"
    content: str = ""


AgentStreamEvent = (
    ReasoningEvent
    | TokenEvent
    | ToolCallEvent
    | ToolCommunicationEvent
    | ToolResultEvent
    | MessageEvent
    | WorkerDoneEvent
    | SubagentEvent
    | InputRequiredEvent
    | UsageEvent
    | ResponseEvent
    | ErrorEvent
)
