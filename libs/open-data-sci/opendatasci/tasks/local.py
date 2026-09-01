"""In-process background task manager backed by asyncio tasks."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import UUID, uuid4

from opendatasci.tasks.base import (
    BackgroundTaskManagerBase,
    BackgroundTaskProgressReport,
    BackgroundTaskProgressUpdate,
    BackgroundTaskRecord,
    BackgroundTaskStatus,
    TaskUpdate,
    TaskUpdateKind,
)

logger = logging.getLogger(__name__)

_MAX_RECORDS = 128


class BackgroundTaskManager(BackgroundTaskManagerBase):
    """Runs submitted work as ``asyncio.tasks`` objects on the current event loop.

    Records are kept for the lifetime of this manager instance (i.e. for as long
    as the owning agent session is alive) so that ``get_task``/``list_tasks``
    remain answerable after a task finishes, up to a fixed number of most
    recent tasks (oldest evicted first).
    """

    def __init__(self, output_root: Path | None = None) -> None:
        self._records: dict[UUID, BackgroundTaskRecord] = {}
        self._tasks: dict[UUID, asyncio.Task[Any]] = {}
        self._output_root = output_root
        self._updates: dict[UUID, TaskUpdate] = {}
        self._pending_update_ids: list[UUID] = []
        self._doorbell: asyncio.Queue[tuple[UUID, UUID]] = asyncio.Queue()

    async def submit_task(self, work: Callable[[UUID], Awaitable[Any]], summary: str) -> UUID:
        task_id = uuid4()
        record = BackgroundTaskRecord(
            task_id=task_id, summary=summary, status=BackgroundTaskStatus.RUNNING
        )
        await self.upsert_record(record)

        async def _run() -> None:
            try:
                result = await work(task_id)
            except asyncio.CancelledError:
                record.status = BackgroundTaskStatus.CANCELLED
                record.finished_at = time.time()
                await self.upsert_record(record)
                await self.record_task_update(
                    task_id,
                    TaskUpdateKind.COMPLETION,
                    {"status": record.status, "summary": record.summary},
                )
                raise
            except Exception as exc:
                logger.exception("Background task %s (%s) failed", task_id, summary)
                record.status = BackgroundTaskStatus.FAILED
                record.error = str(exc)
                record.finished_at = time.time()
                await self.upsert_record(record)
                await self.record_task_update(
                    task_id,
                    TaskUpdateKind.COMPLETION,
                    {"status": record.status, "summary": record.summary, "error": record.error},
                )
            else:
                record.status = BackgroundTaskStatus.COMPLETED
                record.result = result
                record.finished_at = time.time()
                await self.upsert_record(record)
                await self._publish_task_result(task_id, record)
                await self.record_task_update(
                    task_id,
                    TaskUpdateKind.COMPLETION,
                    {"status": record.status, "summary": record.summary, "result": record.result},
                )
            finally:
                self._tasks.pop(task_id, None)

        self._tasks[task_id] = asyncio.create_task(_run())
        return task_id

    async def _publish_task_result(self, task_id: UUID, record: BackgroundTaskRecord) -> None:
        if self._output_root is None:
            return
        output_path = self._output_root / f"{task_id}.md"
        content = (
            f"# {record.summary}\n\ntask_id: {task_id}\nstatus: {record.status.value}\n\n"
            f"## Result\n\n{record.result}\n"
        )

        def _write() -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

        await asyncio.to_thread(_write)

    async def get_task(self, task_id: UUID) -> BackgroundTaskRecord | None:
        return self._records.get(task_id)

    async def list_tasks(self) -> list[BackgroundTaskRecord]:
        return list(self._records.values())

    async def cancel_task(self, task_id: UUID) -> bool:
        if task_id not in self._records:
            return False
        task = self._tasks.get(task_id)
        if task is not None:
            task.cancel()
        return True

    async def upsert_record(self, record: BackgroundTaskRecord) -> None:
        if record.task_id not in self._records and len(self._records) >= _MAX_RECORDS:
            oldest_task_id = next(iter(self._records))
            del self._records[oldest_task_id]
        self._records[record.task_id] = record

    async def push_task_progress(
        self,
        task_id: UUID,
        update: BackgroundTaskProgressUpdate,
        eta_seconds: float | None = None,
    ) -> None:
        record = self._records.get(task_id)
        if record is None:
            logger.warning("push_task_progress called with unknown task_id=%s", task_id)
            return
        record.progress.append(
            BackgroundTaskProgressReport(progress_update=update, eta_seconds=eta_seconds)
        )
        await self.upsert_record(record)

    async def record_task_update(
        self, task_id: UUID, kind: TaskUpdateKind, payload: dict[str, Any]
    ) -> UUID:
        update_id = uuid4()
        update = TaskUpdate(update_id=update_id, task_id=task_id, kind=kind, payload=payload)
        self._updates[update_id] = update
        self._pending_update_ids.append(update_id)
        self._doorbell.put_nowait((task_id, update_id))
        return update_id

    async def listen_task_updates(self) -> AsyncIterator[tuple[UUID, UUID]]:
        while True:
            yield await self._doorbell.get()

    async def pull_task_updates(self) -> list[TaskUpdate]:
        update_ids, self._pending_update_ids = self._pending_update_ids, []
        return [self._updates[update_id] for update_id in update_ids]

    def has_task_updates(self) -> bool:
        return bool(self._pending_update_ids)
