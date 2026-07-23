"""In-process background task manager backed by asyncio tasks."""

import asyncio
import logging
import time
import uuid
from typing import Any, Awaitable, Callable

from opendatasci.tasks.base import BaseTaskManager, TaskRecord, TaskStatus

logger = logging.getLogger(__name__)


class LocalTaskManager(BaseTaskManager):
    """Runs submitted work as ``asyncio.Task`` objects on the current event loop.

    Records are kept for the lifetime of this manager instance (i.e. for as long
    as the owning agent session is alive) so that ``get_status``/``list`` remain
    answerable after a task finishes.
    """

    def __init__(self) -> None:
        self._records: dict[str, TaskRecord] = {}
        self._asyncio_tasks: dict[str, asyncio.Task[Any]] = {}

    async def submit(self, work: Callable[[], Awaitable[Any]], summary: str) -> str:
        task_id = uuid.uuid4().hex
        record = TaskRecord(task_id=task_id, summary=summary, status=TaskStatus.RUNNING)
        self._records[task_id] = record

        async def _run() -> None:
            try:
                result = await work()
            except asyncio.CancelledError:
                record.status = TaskStatus.CANCELLED
                record.finished_at = time.monotonic()
                raise
            except Exception as exc:
                logger.exception("Background task %s (%s) failed", task_id, summary)
                record.status = TaskStatus.FAILED
                record.error = str(exc)
                record.finished_at = time.monotonic()
            else:
                record.status = TaskStatus.COMPLETED
                record.result = result
                record.finished_at = time.monotonic()

        self._asyncio_tasks[task_id] = asyncio.create_task(_run())
        return task_id

    async def get_status(self, task_id: str) -> TaskRecord | None:
        return self._records.get(task_id)

    async def list(self) -> list[TaskRecord]:
        return list(self._records.values())

    async def cancel(self, task_id: str) -> bool:
        task = self._asyncio_tasks.get(task_id)
        if task is None:
            return False
        task.cancel()
        return True
