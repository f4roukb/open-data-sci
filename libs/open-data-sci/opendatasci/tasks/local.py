"""In-process background task manager backed by asyncio tasks."""

import asyncio
import logging
import re
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
    BackgroundTaskUpdate,
    BackgroundTaskUpdateEvent,
    BackgroundTaskUpdateKind,
)

logger = logging.getLogger(__name__)

_MAX_RECORDS = 128
_MAX_ACTIVITY_ENTRIES = 200
_MAX_ACTIVITY_ENTRY_LEN = 32768


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
        self._updates_by_id: dict[UUID, BackgroundTaskUpdate] = {}
        self._unpulled_update_ids: list[UUID] = []
        self._update_event_queue: asyncio.Queue[BackgroundTaskUpdateEvent] = asyncio.Queue()
        self._monitors: dict[UUID, dict[UUID, re.Pattern[str]]] = {}
        self._monitor_task_ids: dict[UUID, UUID] = {}

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
                    BackgroundTaskUpdateKind.COMPLETED,
                    summary=record.summary,
                    status=record.status,
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
                    BackgroundTaskUpdateKind.COMPLETED,
                    summary=record.summary,
                    status=record.status,
                    error=record.error,
                )
            else:
                record.status = BackgroundTaskStatus.COMPLETED
                record.result = result
                record.finished_at = time.time()
                await self.upsert_record(record)
                await self._publish_task_result(task_id, record)
                await self.record_task_update(
                    task_id,
                    BackgroundTaskUpdateKind.COMPLETED,
                    summary=record.summary,
                    status=record.status,
                    result=record.result,
                )
            finally:
                self._tasks.pop(task_id, None)
                self._remove_monitors_for_task(task_id)

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

    async def push_activity(self, task_id: UUID, entry: str) -> None:
        record = self._records.get(task_id)
        if record is None:
            logger.warning("push_activity called with unknown task_id=%s", task_id)
            return

        # Monitors scan the full, untruncated entry — truncation below is a
        # storage bound on the persisted activity log, not a matching window,
        # so a match past the truncation cutoff must never be missed.
        for monitor_id, regex in list(self._monitors.get(task_id, {}).items()):
            for match in regex.finditer(entry):
                await self.record_task_update(
                    task_id,
                    BackgroundTaskUpdateKind.PROGRESS,
                    summary=record.summary,
                    monitor_id=monitor_id,
                    pattern=regex.pattern,
                    matched_text=match.group(0),
                )

        if len(entry) > _MAX_ACTIVITY_ENTRY_LEN:
            entry = entry[:_MAX_ACTIVITY_ENTRY_LEN] + "... (truncated)"
        record.activity.append(entry)
        if len(record.activity) > _MAX_ACTIVITY_ENTRIES:
            del record.activity[: len(record.activity) - _MAX_ACTIVITY_ENTRIES]
        await self.upsert_record(record)

    async def monitor_task(self, task_id: UUID, regex_patterns: list[str]) -> list[UUID]:
        if task_id not in self._records:
            logger.warning("monitor_task called with unknown task_id=%s", task_id)
            return []
        compiled = [re.compile(regex_pattern) for regex_pattern in regex_patterns]
        task_monitors = self._monitors.setdefault(task_id, {})
        monitor_ids: list[UUID] = []
        for regex in compiled:
            monitor_id = uuid4()
            task_monitors[monitor_id] = regex
            self._monitor_task_ids[monitor_id] = task_id
            monitor_ids.append(monitor_id)
        return monitor_ids

    async def stop_monitoring_task(
        self,
        task_id: UUID | None = None,
        monitor_ids: list[UUID] | None = None,
    ) -> None:
        if task_id is not None:
            self._remove_monitors_for_task(task_id)
        if monitor_ids is not None:
            for monitor_id in monitor_ids:
                owning_task_id = self._monitor_task_ids.pop(monitor_id, None)
                if owning_task_id is None:
                    continue
                task_monitors = self._monitors.get(owning_task_id)
                if task_monitors is None:
                    continue
                task_monitors.pop(monitor_id, None)
                if not task_monitors:
                    del self._monitors[owning_task_id]

    async def list_task_monitors(self, task_id: UUID) -> dict[UUID, str]:
        return {
            monitor_id: regex.pattern
            for monitor_id, regex in self._monitors.get(task_id, {}).items()
        }

    def _remove_monitors_for_task(self, task_id: UUID) -> None:
        for monitor_id in self._monitors.pop(task_id, {}):
            self._monitor_task_ids.pop(monitor_id, None)

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
        matched_text: str | None = None,
    ) -> UUID:
        update_id = uuid4()
        new_update = BackgroundTaskUpdate(
            update_id=update_id,
            task_id=task_id,
            kind=kind,
            summary=summary,
            status=status,
            result=result,
            error=error,
            monitor_id=monitor_id,
            pattern=pattern,
            matched_text=matched_text,
        )
        self._updates_by_id[update_id] = new_update
        self._unpulled_update_ids.append(update_id)
        self._update_event_queue.put_nowait(
            BackgroundTaskUpdateEvent(task_id=task_id, update_id=update_id)
        )
        return update_id

    async def listen_task_updates(self) -> AsyncIterator[BackgroundTaskUpdateEvent]:
        while True:
            yield await self._update_event_queue.get()

    async def pull_task_updates(self) -> list[BackgroundTaskUpdate]:
        update_ids, self._unpulled_update_ids = self._unpulled_update_ids, []
        return [self._updates_by_id[update_id] for update_id in update_ids]

    def has_task_updates(self) -> bool:
        return bool(self._unpulled_update_ids)
