"""Abstract interface for managing background tasks."""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import StrEnum, auto
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import UUID

from pydantic import Field

from opendatasci._utils.message_utils import to_text_content_blocks
from opendatasci._utils.pydantic_utils import MutableStrictBaseModel
from opendatasci.memory.messages import TaskMessage


class AgentTaskStatus(StrEnum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class AgentTaskProgressUpdate(MutableStrictBaseModel):
    """A worker's self-reported snapshot of what it has done, is doing, and is blocked on."""

    done: str
    ongoing: str
    blockers: str


class AgentTaskProgressReport(MutableStrictBaseModel):
    """One progress checkpoint recorded against a task, in call order."""

    progress_update: AgentTaskProgressUpdate
    eta_seconds: float | None
    reported_at: float = Field(default_factory=time.time)


class WorkerTaskRecord(MutableStrictBaseModel):
    """Point-in-time snapshot of a background task's lifecycle."""

    task_id: UUID
    summary: str
    """Human-facing label for the task (e.g. the caller-supplied ``summary``)."""
    status: AgentTaskStatus
    result: Any = None
    error: str | None = None
    progress: list[AgentTaskProgressReport] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None

    def to_update_message(self) -> TaskMessage:
        """Render this finished background task as the content fed to the model."""
        if self.status == AgentTaskStatus.COMPLETED:
            text = f"Background task '{self.summary}' finished:\n\n{self.result}"
        elif self.status == AgentTaskStatus.FAILED:
            text = f"Background task '{self.summary}' failed: {self.error}"
        else:
            text = f"Background task '{self.summary}' was cancelled."
        return TaskMessage(content=to_text_content_blocks(text), created_at=datetime.now(timezone.utc))


class AgentTaskManagerBase(ABC):
    """Registers and tracks background tasks: submit, check on, cancel."""

    @abstractmethod
    async def submit_task(self, work: Callable[[UUID], Awaitable[Any]], summary: str) -> UUID:
        """Create a :class:`WorkerTaskRecord`, schedule *work* to run against it, and return its ID.

        The record is created and stored before *work* starts; *work* only
        receives the ``task_id``, not the record itself. There is no method
        here for mutating a task's state from outside: a task manager
        exposes reading tasks, not writing to them.
        """
        ...

    @abstractmethod
    async def get_task(self, task_id: UUID) -> WorkerTaskRecord | None:
        """Return the current record for *task_id*, or ``None`` if unknown."""
        ...

    @abstractmethod
    async def list_tasks(self) -> list[WorkerTaskRecord]:
        """Return records for all tasks currently tracked."""
        ...

    @abstractmethod
    async def cancel_task(self, task_id: UUID) -> bool:
        """Request cancellation of *task_id*. Returns ``False`` if unknown.

        Cancellation may be best-effort depending on the implementation.
        """
        ...

    @abstractmethod
    async def upsert_record(self, record: WorkerTaskRecord) -> None:
        """Insert or overwrite *record* wholesale, keyed by ``record.task_id``.

        Exposed publicly so a worker running independently of the manager
        (e.g. in a different process) can report its own state back by
        constructing a record and calling this directly.
        """
        ...

    @abstractmethod
    async def push_task_progress(
        self,
        task_id: UUID,
        update: AgentTaskProgressUpdate,
        eta_seconds: float | None = None,
    ) -> None:
        """Append one progress checkpoint to *task_id*'s record.

        No-op (aside from logging) if *task_id* is unknown.
        """
        ...

    @abstractmethod
    def listen_task_updates(self) -> AsyncIterator[WorkerTaskRecord]:
        """Yield each task's record exactly once, as soon as it reaches a terminal status.

        Blocks between completions — this is a push source, not a poll.
        Single-consumer by contract (one listener per manager instance/
        session): each terminal record is delivered exactly once, so two
        concurrent consumers would race for records rather than both seeing
        them.
        """
        ...

    @abstractmethod
    async def gather_task_updates(self) -> list[WorkerTaskRecord]:
        """Return and clear every completed record collected since the last call.

        Non-blocking: returns immediately, empty if nothing has completed.
        Independent of :meth:`listen_task_updates` — both observe the same
        underlying completions, but gathering one does not consume the other,
        so each can have its own consumer without racing.

        Assumes at most one caller gathers at a time; concurrent gatherers
        would race for the same records.
        """
        ...

    @abstractmethod
    def has_task_updates(self) -> bool:
        """Cheap, non-blocking peek: ``True`` iff :meth:`gather_task_updates` would return non-empty."""
        ...
