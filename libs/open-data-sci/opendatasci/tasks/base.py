"""Abstract interface for managing background tasks.

Kept async-first throughout, even for the in-process implementation, so that a
future implementation backed by a remote/cloud job runner (where checking
status or cancelling requires network I/O) can satisfy the same contract.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class TaskStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskRecord:
    """Point-in-time snapshot of a background task's lifecycle."""

    task_id: str
    summary: str
    """Human-facing label for the task (e.g. the caller-supplied ``summary``)."""
    status: TaskStatus
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None


class BaseTaskManager(ABC):
    """Registers and tracks background tasks: submit, check on, list, cancel."""

    @abstractmethod
    async def submit(self, work: Callable[[], Awaitable[Any]], summary: str) -> str:
        """Schedule *work* to run in the background and return its task ID.

        *work* is a zero-arg callable returning an awaitable, rather than a live
        coroutine object, so that implementations aren't required to accept
        in-process Python coroutines directly.
        """
        ...

    @abstractmethod
    async def get_status(self, task_id: str) -> TaskRecord | None:
        """Return the current record for *task_id*, or ``None`` if unknown."""
        ...

    @abstractmethod
    async def list(self) -> list[TaskRecord]:
        """Return records for all tasks currently tracked."""
        ...

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """Request cancellation of *task_id*. Returns ``False`` if unknown.

        Cancellation may be best-effort depending on the implementation and how
        far the task has progressed (e.g. work already running on a dedicated
        OS thread cannot always be interrupted mid-flight).
        """
        ...
