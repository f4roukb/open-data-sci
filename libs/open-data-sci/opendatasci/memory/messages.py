"""Typed LangChain message subtypes — the only message classes used in this codebase."""

from datetime import datetime
from enum import StrEnum, auto
from typing import Any, final

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import Field, field_validator

from opendatasci._utils.datetime_utils import datetime_now, to_local_timezone
from opendatasci._utils.message_utils import get_final_ai_message
from opendatasci._utils.mixins import RenderableMessageMixin

# ---------------------------------------------------------------------------
# HumanMessage subtypes
# ---------------------------------------------------------------------------


class MessageOrigin(StrEnum):
    """The origin of a message"""

    USER = auto()
    HARNESS = auto()
    AGENT = auto()
    TASK = auto()


def _with_metadata_tag(content: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    """Prepend *tag* as a text block to *content*."""
    return [{"type": "text", "text": tag}, *content]


@final
class UserMessage(HumanMessage, RenderableMessageMixin["UserMessage"]):
    """A message that originated directly from the user."""

    content: list[dict[str, Any]]
    origin: MessageOrigin = MessageOrigin.USER
    created_at: datetime = Field(default_factory=datetime_now)
    is_input_on_interrupt: bool = False

    @field_validator("content")
    @classmethod
    def _validate_content_blocks(cls, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not all(isinstance(block, dict) and "type" in block for block in blocks):
            raise ValueError("content must be a list of content-block dicts, each with a 'type' key")
        return blocks

    def _get_content(self) -> list[dict[str, Any]]:
        tag = (
            f"<message_metadata>"
            f"<origin>{self.origin}</origin>"
            f"<timestamp>{to_local_timezone(self.created_at).isoformat()}</timestamp>"
            f"</message_metadata>"
        )
        return _with_metadata_tag(self.content, tag)

    def render(self) -> "UserMessage":
        return self.model_copy(update={"content": self._get_content()}, deep=True)


@final
class CompactionMessage(HumanMessage, RenderableMessageMixin["CompactionMessage"]):
    """A harness message carrying a compacted history recall block."""

    content: list[dict[str, Any]]
    origin: MessageOrigin = MessageOrigin.HARNESS
    created_at: datetime = Field(default_factory=datetime_now)

    @field_validator("content")
    @classmethod
    def _validate_content_blocks(cls, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not all(isinstance(block, dict) and "type" in block for block in blocks):
            raise ValueError("content must be a list of content-block dicts, each with a 'type' key")
        return blocks

    def _get_content(self) -> list[dict[str, Any]]:
        tag = (
            f"<message_metadata>"
            f"<origin>{self.origin}</origin>"
            f"<timestamp>{to_local_timezone(self.created_at).isoformat()}</timestamp>"
            f"</message_metadata>"
        )
        return _with_metadata_tag(self.content, tag)

    def render(self) -> "CompactionMessage":
        return self.model_copy(update={"content": self._get_content()}, deep=True)


@final
class AgentToAgentMessage(HumanMessage, RenderableMessageMixin["AgentToAgentMessage"]):
    """A message sent from one agent to another."""

    content: list[dict[str, Any]]
    origin: MessageOrigin
    created_at: datetime = Field(default_factory=datetime_now)

    @field_validator("content")
    @classmethod
    def _validate_content_blocks(cls, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not all(isinstance(block, dict) and "type" in block for block in blocks):
            raise ValueError("content must be a list of content-block dicts, each with a 'type' key")
        return blocks

    def _get_content(self) -> list[dict[str, Any]]:
        tag = (
            f"<message_metadata>"
            f"<origin>{self.origin}</origin>"
            f"<timestamp>{to_local_timezone(self.created_at).isoformat()}</timestamp>"
            f"</message_metadata>"
        )
        return _with_metadata_tag(self.content, tag)

    def render(self) -> "AgentToAgentMessage":
        return self.model_copy(update={"content": self._get_content()}, deep=True)


@final
class TaskMessage(HumanMessage, RenderableMessageMixin["TaskMessage"]):
    """A worker's output delivered back to the main agent (e.g. a finished background task)."""

    content: list[dict[str, Any]]
    origin: MessageOrigin = MessageOrigin.TASK
    created_at: datetime = Field(default_factory=datetime_now)

    @field_validator("content")
    @classmethod
    def _validate_content_blocks(cls, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not all(isinstance(block, dict) and "type" in block for block in blocks):
            raise ValueError("content must be a list of content-block dicts, each with a 'type' key")
        return blocks

    def _get_content(self) -> list[dict[str, Any]]:
        tag = (
            f"<message_metadata>"
            f"<origin>{self.origin}</origin>"
            f"<timestamp>{to_local_timezone(self.created_at).isoformat()}</timestamp>"
            f"</message_metadata>"
        )
        return _with_metadata_tag(self.content, tag)

    def render(self) -> "TaskMessage":
        return self.model_copy(update={"content": self._get_content()}, deep=True)


@final
class SummaryMessage(HumanMessage, RenderableMessageMixin["SummaryMessage"]):
    """Harness-constructed message carrying a turn-summary recall."""

    content: list[dict[str, Any]]
    origin: MessageOrigin = MessageOrigin.HARNESS
    created_at: datetime = Field(default_factory=datetime_now)
    turn_start_timestamp: datetime
    turn_end_timestamp: datetime

    @field_validator("content")
    @classmethod
    def _validate_content_blocks(cls, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not all(isinstance(block, dict) and "type" in block for block in blocks):
            raise ValueError("content must be a list of content-block dicts, each with a 'type' key")
        return blocks

    def _get_content(self) -> list[dict[str, Any]]:
        message_meta = (
            f"<message_metadata>"
            f"<origin>{self.origin}</origin>"
            f"<timestamp>{to_local_timezone(self.created_at).isoformat()}</timestamp>"
            f"</message_metadata>"
        )
        summary_meta = (
            f"<summary_metadata>\n"
            f"  <turn_start_timestamp>{to_local_timezone(self.turn_start_timestamp).isoformat()}</turn_start_timestamp>\n"
            f"  <turn_end_timestamp>{to_local_timezone(self.turn_end_timestamp).isoformat()}</turn_end_timestamp>\n"
            f"</summary_metadata>"
        )
        return [
            {"type": "text", "text": message_meta},
            {"type": "text", "text": summary_meta},
            *self.content,
        ]

    def render(self) -> "SummaryMessage":
        return self.model_copy(update={"content": self._get_content()}, deep=True)


@final
class PlanMessage(HumanMessage, RenderableMessageMixin["PlanMessage"]):
    """Harness-constructed message carrying the current session plan."""

    content: list[dict[str, Any]]
    origin: MessageOrigin = MessageOrigin.HARNESS
    created_at: datetime = Field(default_factory=datetime_now)

    @field_validator("content")
    @classmethod
    def _validate_content_blocks(cls, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not all(isinstance(block, dict) and "type" in block for block in blocks):
            raise ValueError("content must be a list of content-block dicts, each with a 'type' key")
        return blocks

    def _get_content(self) -> list[dict[str, Any]]:
        tag = (
            f"<message_metadata>"
            f"<origin>{self.origin}</origin>"
            f"<timestamp>{to_local_timezone(self.created_at).isoformat()}</timestamp>"
            f"</message_metadata>"
        )
        return _with_metadata_tag(self.content, tag)

    def render(self) -> "PlanMessage":
        return self.model_copy(update={"content": self._get_content()}, deep=True)


@final
class AgentMessage(AIMessage):
    """A message produced by the LLM agent."""

    origin: MessageOrigin = MessageOrigin.AGENT
    created_at: datetime = Field(default_factory=datetime_now)

    @classmethod
    def from_langchain(cls, msg: AIMessage) -> "AgentMessage":
        return cls.model_validate(msg.model_dump())


def is_user_message(msg: BaseMessage) -> bool:
    """Return ``True`` if *msg* is a message that originated from the user."""
    return isinstance(msg, UserMessage)


def is_ongoing_turn(turn: list[BaseMessage]) -> bool:
    """Return ``True`` if *turn* is an active, in-progress ReAct turn.

    A valid ongoing turn starts with any ``HumanMessage`` and ends with either
    an ``AIMessage`` carrying pending tool calls, a ``ToolMessage``, or a
    ``UserMessage`` flagged as an interrupt reply.
    """
    if not turn or not isinstance(turn[0], HumanMessage):
        return False
    last = turn[-1]
    if isinstance(last, ToolMessage):
        return True
    if isinstance(last, UserMessage):
        return last.is_input_on_interrupt
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def get_turn_start_timestamp(turn_messages: list[BaseMessage]) -> datetime:
    first = turn_messages[0]
    if not isinstance(first, (UserMessage, TaskMessage)):
        raise ValueError("First message in turn is not a UserMessage or TaskMessage")
    return first.created_at


def get_turn_end_timestamp(turn_messages: list[BaseMessage]) -> datetime | None:
    final_ai = get_final_ai_message(turn_messages)
    return final_ai.created_at if isinstance(final_ai, AgentMessage) else None
