from typing import ClassVar

from langchain_core.messages import BaseMessage
from pydantic import Field

from opendatasci._utils.pydantic_utils import MutableStrictBaseModel


class BaseAgentStreamEvent(MutableStrictBaseModel):
    """Base class for all streaming agent events."""

    type: ClassVar[str] = ""


class ReasoningEvent(BaseAgentStreamEvent):
    """Extended-thinking / reasoning token(s)."""

    type: ClassVar[str] = "reasoning"
    content: str = ""


class TokenEvent(BaseAgentStreamEvent):
    """Regular response text token."""

    type: ClassVar[str] = "token"
    content: str = ""


class ToolCallEvent(BaseAgentStreamEvent):
    """The agent is invoking a tool.

    ``task_summaries`` is populated only for ``task`` tool calls;
    ``summary`` carries the agent-provided summary argument for all other calls.
    """

    type: ClassVar[str] = "tool_call"
    content: str = ""
    tool: str = ""
    tool_call_id: str | None = None
    summary: str = ""
    task_summaries: list[str] = Field(default_factory=list)


class ToolCommunicationEvent(BaseAgentStreamEvent):
    """A progress message emitted by a tool before it returns."""

    type: ClassVar[str] = "tool_communication"
    content: str = ""
    tool_call_id: str = ""
    tool_name: str = ""


class ToolResultEvent(BaseAgentStreamEvent):
    """A tool returned a result."""

    type: ClassVar[str] = "tool_result"
    content: str = ""
    tool_call_id: str | None = None
    is_error: bool = False


# Marker used in a ToolMessage.artifact dict (response_format="content_and_artifact")
# to say "this artifact is an image to display inline" — keeps the artifact
# payload self-describing without a new event/ToolMessage shape per artifact kind.
IMAGE_ARTIFACT_KIND = "image"


class ImageRenderEvent(BaseAgentStreamEvent):
    """A tool pointed at a static image to render inline in the conversation.

    Only ``path`` crosses this boundary — the TUI alone is responsible for
    resolving, decoding, and rendering the file; no image bytes are carried
    by this event or ever fed back into the LLM's context.
    """

    type: ClassVar[str] = "image_render"
    tool_call_id: str | None = None
    path: str = ""
    caption: str = ""


class MessageEvent(BaseAgentStreamEvent):
    """A completed ``BaseMessage`` for callers that own conversation-history accumulation."""

    type: ClassVar[str] = "message"
    message: BaseMessage | None = None


class TaskDoneEvent(BaseAgentStreamEvent):
    """A single concurrent worker finished."""

    type: ClassVar[str] = "task_done"
    task_idx: int | None = None
    success: bool = True


class SubagentEvent(BaseAgentStreamEvent):
    """Lifecycle event from inside a running worker.

    ``event_type`` is one of ``"task_tool_call"`` or ``"task_tool_result"``.
    ``content`` carries the tool name for ``task_tool_call`` events.
    """

    type: ClassVar[str] = "subagent_event"
    content: str = ""
    task_idx: int | None = None
    event_type: str = ""
    success: bool = True
    summary: str = ""


class InputRequiredEvent(BaseAgentStreamEvent):
    """The agent is paused at an interrupt and needs input from the user.

    ``content`` is the question.  Resume with ``resume_with_input(answer)``.
    """

    type: ClassVar[str] = "input_required"
    content: str = ""
    choices: list[str] = Field(default_factory=list)


class ApprovalRequiredEvent(BaseAgentStreamEvent):
    """The agent is paused waiting for the user to approve a command.

    ``description`` is an LLM-generated plain-language summary of what the
    command does; ``heads_up`` warns about potential negative impact and is
    empty when none was identified.  Resume with ``resume_with_approval(approved)``.
    """

    type: ClassVar[str] = "approval_required"
    command: str = ""
    description: str = ""
    heads_up: str = ""


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


class ResponseEvent(BaseAgentStreamEvent):
    """Final assembled response for this turn (end-of-turn marker)."""

    type: ClassVar[str] = "response"
    content: str = ""


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
    | ImageRenderEvent
    | MessageEvent
    | TaskDoneEvent
    | SubagentEvent
    | InputRequiredEvent
    | ApprovalRequiredEvent
    | UsageEvent
    | ResponseEvent
    | ErrorEvent
)
