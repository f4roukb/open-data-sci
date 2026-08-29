"""FIFO queue for chat messages submitted while the agent is busy."""

import itertools
from dataclasses import dataclass, field
from datetime import datetime

from opendatasci._utils.datetime_utils import datetime_now


@dataclass(frozen=True)
class PendingMessage:
    """A message queued while the agent was processing a previous turn.

    ``from_worker`` marks a background-task completion rather than text the
    user typed. ``created_at`` is stamped at enqueue time so it still
    reflects when the message actually arrived, even if it sits queued for
    a while before being processed.
    """

    id: int
    agent_query: str
    display: str
    from_worker: bool = False
    created_at: datetime = field(default_factory=datetime_now)


class PendingMessageQueue:
    """Unbounded FIFO queue of messages submitted while the agent is running.

    Messages are meant to be drained together as a single batch once the
    agent is free, rather than processed one at a time.
    """

    def __init__(self) -> None:
        self._items: list[PendingMessage] = []
        self._next_id = itertools.count(1)

    def enqueue(
        self, agent_query: str, display: str, *, from_worker: bool = False
    ) -> PendingMessage:
        message = PendingMessage(
            id=next(self._next_id),
            agent_query=agent_query,
            display=display,
            from_worker=from_worker,
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
        removed = [item for item in self._items if not item.from_worker]
        self._items = [item for item in self._items if item.from_worker]
        return removed

    def cancel_last_user_message(self) -> PendingMessage | None:
        """Remove and return the most recently queued user-typed message, or ``None``."""
        for i in range(len(self._items) - 1, -1, -1):
            if not self._items[i].from_worker:
                return self._items.pop(i)
        return None

    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)
