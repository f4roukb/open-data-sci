"""FIFO queue for chat messages submitted while the agent is busy."""

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingMessage:
    """A user message queued while the agent was processing a previous turn."""

    id: int
    agent_query: str
    display: str


class PendingMessageQueue:
    """Unbounded FIFO queue of messages submitted while the agent is running.

    Messages are processed one at a time, in submission order, once the
    current agent turn finishes (see ``CLIController.run_agent``).
    """

    def __init__(self) -> None:
        self._items: list[PendingMessage] = []
        self._next_id = itertools.count(1)

    def enqueue(self, agent_query: str, display: str) -> PendingMessage:
        message = PendingMessage(id=next(self._next_id), agent_query=agent_query, display=display)
        self._items.append(message)
        return message

    def pop_next(self) -> PendingMessage | None:
        """Remove and return the oldest queued message, or ``None`` if empty."""
        return self._items.pop(0) if self._items else None

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
