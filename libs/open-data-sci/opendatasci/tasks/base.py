"""Abstract interface for managing background tasks."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Awaitable, Callable
from uuid import UUID


class TaskStatus(StrEnum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class TaskProgressUpdate:
    """A worker's self-reported snapshot of what it has done, is doing, and is blocked on."""

    done: str
    ongoing: str
    blockers: str


@dataclass
class TaskProgressReport:
    """One progress checkpoint recorded against a task, in call order."""

    progress_update: TaskProgressUpdate
    eta_seconds: float | None
    reported_at: float = field(default_factory=time.time)


@dataclass
class TaskRecord:
    """Point-in-time snapshot of a background task's lifecycle."""

    task_id: UUID
    summary: str
    """Human-facing label for the task (e.g. the caller-supplied ``summary``)."""
    status: TaskStatus
    result: Any = None
    error: str | None = None
    progress: list[TaskProgressReport] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None


class BaseTaskManager(ABC):
    """Registers and tracks background tasks: submit, check on, cancel."""

    @abstractmethod
    async def submit_task(self, work: Callable[[UUID], Awaitable[Any]], summary: str) -> UUID:
        """Create a :class:`TaskRecord`, schedule *work* to run against it, and return its ID.

        The record is created and stored before *work* starts; *work* only
        receives the ``task_id``, not the record itself. Recording progress
        against the record is deliberately not part of this interface —
        there is no method here for mutating a task's state from outside.
        That is a concern of whatever runs the work (see
        :meth:`~opendatasci.agents.agents.WorkerAgent.report_progress`), not
        of the task manager: a task manager exposes reading tasks, not
        writing to them.
        """
        ...

    @abstractmethod
    async def get_task(self, task_id: UUID) -> TaskRecord | None:
        """Return the current record for *task_id*, or ``None`` if unknown."""
        ...

    @abstractmethod
    async def list_tasks(self) -> list[TaskRecord]:
        """Return records for all tasks currently tracked."""
        ...

    @abstractmethod
    async def cancel_task(self, task_id: UUID) -> bool:
        """Request cancellation of *task_id*. Returns ``False`` if unknown.

        Cancellation may be best-effort depending on the implementation.
        """
        ...
