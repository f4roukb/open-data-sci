"""In-process background task manager backed by asyncio tasks."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from opendatasci.tasks.base import AgentTaskManagerBase, TaskRecord, TaskStatus

logger = logging.getLogger(__name__)


class LocalAgentTaskManager(AgentTaskManagerBase):
    """Runs submitted work as ``asyncio.tasks`` objects on the current event loop.

    Records are kept for the lifetime of this manager instance (i.e. for as long
    as the owning agent session is alive) so that ``get_task``/``list_tasks``
    remain answerable after a task finishes.
    """

    def __init__(self, output_root: Path | None = None) -> None:
        self._records: dict[UUID, TaskRecord] = {}
        self._tasks: dict[UUID, asyncio.tasks[Any]] = {}
        self._output_root = output_root

    async def submit_task(self, work: Callable[[UUID], Awaitable[Any]], summary: str) -> UUID:
        task_id = uuid4()
        record = TaskRecord(task_id=task_id, summary=summary, status=TaskStatus.RUNNING)
        self._records[task_id] = record

        async def _run() -> None:
            try:
                result = await work(task_id)
            except asyncio.CancelledError:
                record.status = TaskStatus.CANCELLED
                record.finished_at = time.time()
                raise
            except Exception as exc:
                logger.exception("Background task %s (%s) failed", task_id, summary)
                record.status = TaskStatus.FAILED
                record.error = str(exc)
                record.finished_at = time.time()
            else:
                record.status = TaskStatus.COMPLETED
                record.result = result
                record.finished_at = time.time()
                await self._publish_task_result(task_id, record)

        self._tasks[task_id] = asyncio.create_task(_run())
        return task_id

    async def _publish_task_result(self, task_id: UUID, record: TaskRecord) -> None:
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

    async def get_task(self, task_id: UUID) -> TaskRecord | None:
        return self._records.get(task_id)

    async def list_tasks(self) -> list[TaskRecord]:
        return list(self._records.values())

    async def cancel_task(self, task_id: UUID) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.cancel()
        return True
