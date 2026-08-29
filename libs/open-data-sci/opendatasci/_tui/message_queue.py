"""FIFO queue for chat messages submitted while the agent is busy."""

import itertools
from datetime import datetime
from enum import StrEnum, auto

from pydantic import Field

from opendatasci._utils.datetime_utils import datetime_now
from opendatasci._utils.pydantic_utils import FrozenStrictBaseModel


class PendingMessageOrigin(StrEnum):
    """Who originated a queued message.

    Kept separate from :class:`opendatasci.memory.messages.MessageOrigin`
    (the agent-layer enum) because this queue is TUI-local and currently
    only ever produces two kinds of message; more originators can be added
    here independently as new sources start feeding the queue.
    """

    USER = auto()
    TASK = auto()


class PendingMessage(FrozenStrictBaseModel):
    """A message queued while the agent was processing a previous turn.

    ``origin`` distinguishes a background-task completion from text the
    user typed. ``created_at`` is stamped at enqueue time so it still
    reflects when the message actually arrived, even if it sits queued for
    a while before being processed.
    """

    id: int
    agent_query: str
    display: str
    origin: PendingMessageOrigin = PendingMessageOrigin.USER
    created_at: datetime = Field(default_factory=datetime_now)


class PendingMessageQueue:
    """Unbounded FIFO queue of messages submitted while the agent is running.

    Messages are meant to be drained together as a single batch once the
    agent is free, rather than processed one at a time.
    """

    def __init__(self) -> None:
        self._items: list[PendingMessage] = []
        self._next_id = itertools.count(1)

    def enqueue(
        self,
        agent_query: str,
        display: str,
        *,
        origin: PendingMessageOrigin = PendingMessageOrigin.USER,
    ) -> PendingMessage:
        message = PendingMessage(
            id=next(self._next_id),
            agent_query=agent_query,
            display=display,
            origin=origin,
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

    def cancel_all_user_messages(self) -> list[PendingMessage]:
        """Remove and return every queued user-typed message, leaving worker items queued."""
        removed = [item for item in self._items if item.origin is PendingMessageOrigin.USER]
        self._items = [item for item in self._items if item.origin is not PendingMessageOrigin.USER]
        return removed

    def cancel_last_user_message(self) -> PendingMessage | None:
        """Remove and return the most recently queued user-typed message, or ``None``."""
        for i in range(len(self._items) - 1, -1, -1):
            if self._items[i].origin is PendingMessageOrigin.USER:
                return self._items.pop(i)
        return None

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)
