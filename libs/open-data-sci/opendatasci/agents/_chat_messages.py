"""Internal representation of HumanMessage provenance metadata.

``additional_kwargs`` on a ``HumanMessage`` should only ever be read or
written through :class:`HumanMessageMetadata` — never accessed as a raw dict
directly — so its shape stays consistent across every producer/consumer.
"""

from datetime import datetime, timezone
from enum import StrEnum, auto
from typing import override

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from opendatasci._utils.mixins import LLMDigestibleMixin

__all__ = ["ChatMessageOrigin", "HumanMessageMetadata"]


class ChatMessageOrigin(StrEnum):
    """Who a ``HumanMessage`` actually originated from."""

    USER = auto()
    HARNESS = auto()
    UNSPECIFIED = auto()


class HumanMessageMetadata(BaseModel, LLMDigestibleMixin):
    """Structured view of a ``HumanMessage``'s ``additional_kwargs``.

    Fields default to "unset" (``None``/``False``) rather than a meaningful
    value, so callers can tell whether something was ever actually set —
    that's what lets producers fill in only the fields that are still missing
    instead of overwriting whatever's already there.
    """

    origin: ChatMessageOrigin | None = None
    created_at: datetime | None = None
    is_input_on_interrupt: bool = False

    @override
    def to_content(self) -> str:
        """Render this metadata as the ``<message_metadata>`` tag prefixed onto a message's content."""
        origin = self.origin or ChatMessageOrigin.UNSPECIFIED
        created_at = self.created_at or datetime.now(timezone.utc)
        return (
            f"<message_metadata><origin>{origin.value}</origin>"
            f"<timestamp>{created_at.isoformat()}</timestamp></message_metadata>"
        )

    @classmethod
    def from_message(cls, message: BaseMessage) -> "HumanMessageMetadata":
        """Parse *message*'s ``additional_kwargs`` into structured metadata."""
        return cls.model_validate(message.additional_kwargs)

    def attach_to(self, message: HumanMessage, in_place: bool = False) -> HumanMessage:
        """Return *message* with ``additional_kwargs`` replaced by this metadata.

        By default returns a new copy, leaving *message* untouched. Pass
        ``in_place=True`` to mutate *message* directly instead and return it.
        """
        kwargs = self.model_dump(mode="json", exclude_none=True)
        if in_place:
            message.additional_kwargs = kwargs
            return message
        return message.model_copy(update={"additional_kwargs": kwargs})
