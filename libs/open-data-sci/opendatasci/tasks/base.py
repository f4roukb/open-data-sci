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
    activity: list[str] = Field(default_factory=list)
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

    Fields below the ``kind`` line are kind-specific: ``status``, ``result``,
    and ``error`` are populated for :attr:`BackgroundTaskUpdateKind.COMPLETED`;
    ``monitor_id``, ``pattern``, and ``matched_texts`` are populated for
    :attr:`BackgroundTaskUpdateKind.PROGRESS` (a monitor's regex matching a
    running task's activity — the only non-terminal update kind today).
    """

    update_id: UUID
    task_id: UUID
    kind: BackgroundTaskUpdateKind
    summary: str
    status: BackgroundTaskStatus | None = None
    result: Any = None
    error: str | None = None
    monitor_id: UUID | None = None
    pattern: str | None = None
    matched_texts: list[str] | None = None
    created_at: float = Field(default_factory=time.time)

    def to_message(self) -> TaskMessage:
        """Render this update as the content fed to the model, dispatching on kind."""
        if self.kind == BackgroundTaskUpdateKind.PROGRESS:
            matches = "\n".join(
                f"Match {i}: {matched_text}"
                for i, matched_text in enumerate(self.matched_texts or [])
            )
            text = (
                f"monitor(id={self.monitor_id}) fired on the activity log of "
                f"task(id={self.task_id}). This monitor will not fire again, so create a new "
                f"one if you really need it.\n\n"
                f"This monitor caught the following from the task's activity log:\n\n"
                f"{matches}"
            )
        elif self.kind == BackgroundTaskUpdateKind.COMPLETED:
            if self.status == BackgroundTaskStatus.COMPLETED:
                text = f"Background task '{self.summary}' finished:\n\n{self.result}"
            elif self.status == BackgroundTaskStatus.FAILED:
                text = f"Background task '{self.summary}' failed: {self.error}"
            else:
                text = f"Background task '{self.summary}' was cancelled."
        else:
            raise ValueError(f"Unknown BackgroundTaskUpdateKind: {self.kind!r}")
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
    async def push_activity(self, task_id: UUID, entry: str) -> None:
        """Append one plain-text activity entry to *task_id*'s record.

        No-op (aside from logging) if *task_id* is unknown, same contract as
        :meth:`push_task_progress`.
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
        monitor_id: UUID | None = None,
        pattern: str | None = None,
        matched_texts: list[str] | None = None,
    ) -> UUID:
        """Store a :class:`BackgroundTaskUpdate` against *task_id* and notify both delivery paths.

        Returns the new update's ``update_id``. This is the single write
        path both :meth:`listen_task_updates` (the doorbell) and
        :meth:`pull_task_updates` (the content buffer) are fed from — a
        completion is one *kind* of update, not a separate mechanism.
        """
        ...

    @abstractmethod
    async def monitor_task(self, task_id: UUID, regex_patterns: list[str]) -> list[UUID]:
        """Register one fire-once monitor per pattern in *regex_patterns* against *task_id*.

        Each activity entry (see :meth:`push_activity`) is checked against
        every monitor currently registered for its task; the first entry that
        matches records one :attr:`BackgroundTaskUpdateKind.PROGRESS` update
        via :meth:`record_task_update`, carrying every match found in that
        entry (not just the first), identifying which monitor fired — and
        that monitor is then removed, so it will not fire again. Register a
        new monitor (call this again) to keep watching after a match.

        If a monitor for a given pattern is already registered on *task_id*,
        registering the same pattern again does nothing — the existing
        monitor's ID is reused rather than creating a duplicate.

        Returns one monitor ID per pattern, in the same order as
        *regex_patterns* (a deduplicated pattern's existing ID is returned in
        its place). Two distinct monitors never share an ID — not on the same
        task, and not across different tasks with the same pattern. No-op
        (aside from logging), returning ``[]``, if *task_id* is unknown.
        Raises ``re.error`` before registering anything if any pattern in
        *regex_patterns* fails to compile.
        """
        ...

    @abstractmethod
    async def list_task_monitors(self, task_id: UUID) -> dict[UUID, str]:
        """Return ``{monitor_id: pattern}`` for every monitor currently active on *task_id*.

        Empty if *task_id* is unknown or has no active monitors.
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
