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


class BackgroundTaskStatus(StrEnum):
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class BackgroundTaskUpdateKind(StrEnum):
    PROGRESS = auto()
    COMPLETED = auto()


class BackgroundTaskProgressUpdate(MutableStrictBaseModel):
    """A worker's self-reported snapshot of what it has done, is doing, and is blocked on."""

    done: str
    ongoing: str
    blockers: str


class BackgroundTaskProgressReport(MutableStrictBaseModel):
    """One progress checkpoint recorded against a task, in call order."""

    progress_update: BackgroundTaskProgressUpdate
    eta_seconds: float | None
    reported_at: float = Field(default_factory=time.time)


class BackgroundTaskRecord(MutableStrictBaseModel):
    """Point-in-time snapshot of a background task's lifecycle."""

    task_id: UUID
    summary: str
    status: BackgroundTaskStatus
    result: Any = None
    error: str | None = None
    progress: list[BackgroundTaskProgressReport] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    finished_at: float | None = None

    def to_update_message(self) -> TaskMessage:
        """Render this finished background task as the content fed to the model."""
        if self.status == BackgroundTaskStatus.COMPLETED:
            text = f"Background task '{self.summary}' finished:\n\n{self.result}"
        elif self.status == BackgroundTaskStatus.FAILED:
            text = f"Background task '{self.summary}' failed: {self.error}"
        else:
            text = f"Background task '{self.summary}' was cancelled."
        return TaskMessage(
            content=to_text_content_blocks(text), created_at=datetime.now(timezone.utc)
        )


class BackgroundTaskUpdate(MutableStrictBaseModel):
    """One event pushed against a background task — a completion today, other kinds later.

    Fields below the ``kind`` line are kind-specific: only ``summary``,
    ``status``, ``result``, and ``error`` are populated for
    :attr:`BackgroundTaskUpdateKind.COMPLETED` today. A future
    :attr:`BackgroundTaskUpdateKind.PROGRESS` producer would populate its own
    fields the same way, rather than reintroducing an untyped payload.
    """

    update_id: UUID
    task_id: UUID
    kind: BackgroundTaskUpdateKind
    summary: str
    status: BackgroundTaskStatus | None = None
    result: Any = None
    error: str | None = None
    created_at: float = Field(default_factory=time.time)

    def to_message(self) -> TaskMessage:
        """Render this update as the content fed to the model, dispatching on kind."""
        if self.kind != BackgroundTaskUpdateKind.COMPLETED:
            raise ValueError(f"Unknown BackgroundTaskUpdateKind: {self.kind!r}")
        if self.status == BackgroundTaskStatus.COMPLETED:
            text = f"Background task '{self.summary}' finished:\n\n{self.result}"
        elif self.status == BackgroundTaskStatus.FAILED:
            text = f"Background task '{self.summary}' failed: {self.error}"
        else:
            text = f"Background task '{self.summary}' was cancelled."
        return TaskMessage(
            content=to_text_content_blocks(text),
            created_at=datetime.fromtimestamp(self.created_at, tz=timezone.utc),
        )


class BackgroundTaskUpdateEvent(MutableStrictBaseModel):
    """Doorbell notification: an update was recorded against a task."""

    task_id: UUID
    update_id: UUID


class BackgroundTaskManagerBase(ABC):
    """Registers and tracks background tasks: submit, check on, cancel."""

    @abstractmethod
    async def submit_task(self, work: Callable[[UUID], Awaitable[Any]], summary: str) -> UUID:
        """Create a :class:`BackgroundTaskRecord`, schedule *work* to run against it, and return its ID.

        The record is created and stored before *work* starts; *work* only
        receives the ``task_id``, not the record itself. There is no method
        here for mutating a task's state from outside: a task manager
        exposes reading tasks, not writing to them.
        """
        ...

    @abstractmethod
    async def get_task(self, task_id: UUID) -> BackgroundTaskRecord | None:
        """Return the current record for *task_id*, or ``None`` if unknown."""
        ...

    @abstractmethod
    async def list_tasks(self) -> list[BackgroundTaskRecord]:
        """Return records for all tasks currently tracked."""
        ...

    @abstractmethod
    async def cancel_task(self, task_id: UUID) -> bool:
        """Request cancellation of *task_id*. Returns ``False`` if unknown.

        Cancellation may be best-effort depending on the implementation.
        """
        ...

    @abstractmethod
    async def upsert_record(self, record: BackgroundTaskRecord) -> None:
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
        update: BackgroundTaskProgressUpdate,
        eta_seconds: float | None = None,
    ) -> None:
        """Append one progress checkpoint to *task_id*'s record.

        No-op (aside from logging) if *task_id* is unknown.
        """
        ...

    @abstractmethod
    async def record_task_update(
        self,
        task_id: UUID,
        kind: BackgroundTaskUpdateKind,
        summary: str,
        status: BackgroundTaskStatus | None = None,
        result: Any = None,
        error: str | None = None,
    ) -> UUID:
        """Store a :class:`BackgroundTaskUpdate` against *task_id* and notify both delivery paths.

        Returns the new update's ``update_id``. This is the single write
        path both :meth:`listen_task_updates` (the doorbell) and
        :meth:`pull_task_updates` (the content buffer) are fed from — a
        completion is one *kind* of update, not a separate mechanism.
        """
        ...

    @abstractmethod
    def listen_task_updates(self) -> AsyncIterator[BackgroundTaskUpdateEvent]:
        """Yield a :class:`BackgroundTaskUpdateEvent` exactly once per recorded update.

        Blocks between updates — this is a push source, not a poll.
        Single-consumer by contract (one listener per manager instance/
        session): each update is delivered exactly once, so two concurrent
        consumers would race for updates rather than both seeing them.
        """
        ...

    @abstractmethod
    async def pull_task_updates(self) -> list[BackgroundTaskUpdate]:
        """Return and clear every update collected since the last call.

        Non-blocking: returns immediately, empty if nothing is pending.
        Independent of :meth:`listen_task_updates` — both observe the same
        underlying updates, but pulling one does not consume the other, so
        each can have its own consumer without racing.

        Assumes at most one caller pulls at a time; concurrent pullers
        would race for the same updates.
        """
        ...

    @abstractmethod
    def has_task_updates(self) -> bool:
        """Cheap, non-blocking peek: ``True`` iff :meth:`pull_task_updates` would return non-empty."""
        ...
