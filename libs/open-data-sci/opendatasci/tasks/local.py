"""In-process background task manager backed by asyncio tasks."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable
from uuid import UUID, uuid4

from opendatasci.tasks.base import (
    AgentTaskManagerBase,
    AgentTaskProgressReport,
    AgentTaskProgressUpdate,
    AgentTaskRecord,
    AgentTaskStatus,
)

logger = logging.getLogger(__name__)

_MAX_RECORDS = 128


class LocalAgentTaskManager(AgentTaskManagerBase):
    """Runs submitted work as ``asyncio.tasks`` objects on the current event loop.

    Records are kept for the lifetime of this manager instance (i.e. for as long
    as the owning agent session is alive) so that ``get_task``/``list_tasks``
    remain answerable after a task finishes, up to a fixed number of most
    recent tasks (oldest evicted first).
    """

    def __init__(self, output_root: Path | None = None) -> None:
        self._records: dict[UUID, AgentTaskRecord] = {}
        self._tasks: dict[UUID, asyncio.Task[Any]] = {}
        self._output_root = output_root
        self._completions: asyncio.Queue[AgentTaskRecord] = asyncio.Queue()
        self._context_updates: list[AgentTaskRecord] = []

    async def submit_task(self, work: Callable[[UUID], Awaitable[Any]], summary: str) -> UUID:
        task_id = uuid4()
        record = AgentTaskRecord(task_id=task_id, summary=summary, status=AgentTaskStatus.RUNNING)
        await self.upsert_record(record)

        async def _run() -> None:
            try:
                result = await work(task_id)
            except asyncio.CancelledError:
                record.status = AgentTaskStatus.CANCELLED
                record.finished_at = time.time()
                await self.upsert_record(record)
                self._completions.put_nowait(record)
                self._context_updates.append(record)
                raise
            except Exception as exc:
                logger.exception("Background task %s (%s) failed", task_id, summary)
                record.status = AgentTaskStatus.FAILED
                record.error = str(exc)
                record.finished_at = time.time()
                await self.upsert_record(record)
                self._completions.put_nowait(record)
                self._context_updates.append(record)
            else:
                record.status = AgentTaskStatus.COMPLETED
                record.result = result
                record.finished_at = time.time()
                await self.upsert_record(record)
                await self._publish_task_result(task_id, record)
                self._completions.put_nowait(record)
                self._context_updates.append(record)
            finally:
                self._tasks.pop(task_id, None)

        self._tasks[task_id] = asyncio.create_task(_run())
        return task_id

    async def _publish_task_result(self, task_id: UUID, record: AgentTaskRecord) -> None:
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

    async def get_task(self, task_id: UUID) -> AgentTaskRecord | None:
        return self._records.get(task_id)

    async def list_tasks(self) -> list[AgentTaskRecord]:
        return list(self._records.values())

    async def cancel_task(self, task_id: UUID) -> bool:
        if task_id not in self._records:
            return False
        task = self._tasks.get(task_id)
        if task is not None:
            task.cancel()
        return True

    async def upsert_record(self, record: AgentTaskRecord) -> None:
        if record.task_id not in self._records and len(self._records) >= _MAX_RECORDS:
            oldest_task_id = next(iter(self._records))
            del self._records[oldest_task_id]
        self._records[record.task_id] = record

    async def push_task_progress(
        self,
        task_id: UUID,
        update: AgentTaskProgressUpdate,
        eta_seconds: float | None = None,
    ) -> None:
        record = self._records.get(task_id)
        if record is None:
            logger.warning("push_task_progress called with unknown task_id=%s", task_id)
            return
        record.progress.append(
            AgentTaskProgressReport(progress_update=update, eta_seconds=eta_seconds)
        )
        await self.upsert_record(record)

    async def listen_task_updates(self) -> AsyncIterator[AgentTaskRecord]:
        while True:
            yield await self._completions.get()

    async def gather_task_updates(self) -> list[AgentTaskRecord]:
        records, self._context_updates = self._context_updates, []
        return records

    def has_task_updates(self) -> bool:
        return bool(self._context_updates)
