"""FIFO queue for user-typed chat messages submitted while the agent is busy."""

import itertools
from datetime import datetime

from pydantic import Field

from opendatasci._utils.datetime_utils import datetime_now
from opendatasci._utils.pydantic_utils import FrozenStrictBaseModel


class PendingMessage(FrozenStrictBaseModel):
    """A user-typed message queued while the agent was processing a previous turn.

    ``created_at`` is stamped at enqueue time so it still reflects when the
    message actually arrived, even if it sits queued for a while before
    being processed.
    """

    id: int
    content: str
    display: str
    created_at: datetime = Field(default_factory=datetime_now)


class PendingMessageQueue:
    """Unbounded FIFO queue of user messages submitted while the agent is running."""

    def __init__(self) -> None:
        self._items: list[PendingMessage] = []
        self._next_id = itertools.count(1)

    def enqueue(self, content: str, display: str) -> PendingMessage:
        message = PendingMessage(
            id=next(self._next_id),
            content=content,
            display=display,
        )
        self._items.append(message)
        return message

    def pop_next(self) -> PendingMessage | None:
        """Remove and return the oldest queued message, or ``None`` if empty."""
        return self._items.pop(0) if self._items else None

    def drain_all(self) -> list[PendingMessage]:
        """Remove and return every queued message, in FIFO arrival order."""
        items, self._items = self._items, []
        return items

    def cancel_all(self) -> list[PendingMessage]:
        """Remove and return every queued message."""
        removed, self._items = self._items, []
        return removed

    def cancel_last(self) -> PendingMessage | None:
        """Remove and return the most recently queued message, or ``None`` if empty."""
        return self._items.pop() if self._items else None

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)
