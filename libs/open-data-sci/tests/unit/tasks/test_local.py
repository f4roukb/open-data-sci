"""Unit tests for opendatasci.tasks.local.LocalTaskManager."""

import asyncio

import pytest

from opendatasci.tasks.base import TaskStatus
from opendatasci.tasks.local import LocalTaskManager


class TestSubmitAndStatus:
    @pytest.mark.asyncio
    async def test_completed_task_reports_result(self) -> None:
        manager = LocalTaskManager()

        async def _work() -> str:
            return "done"

        task_id = await manager.submit(_work, summary="s")
        await asyncio.sleep(0)

        record = await manager.get_status(task_id)
        assert record is not None
        assert record.status == TaskStatus.COMPLETED
        assert record.result == "done"
        assert record.error is None
        assert record.finished_at is not None

    @pytest.mark.asyncio
    async def test_failed_task_reports_error_not_raised(self) -> None:
        manager = LocalTaskManager()

        async def _work() -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit(_work, summary="s")
        await asyncio.sleep(0)

        record = await manager.get_status(task_id)
        assert record is not None
        assert record.status == TaskStatus.FAILED
        assert record.error == "boom"
        assert record.result is None

    @pytest.mark.asyncio
    async def test_running_task_reports_running_status(self) -> None:
        manager = LocalTaskManager()
        started = asyncio.Event()

        async def _work() -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        record = await manager.get_status(task_id)
        assert record is not None
        assert record.status == TaskStatus.RUNNING

        await manager.cancel(task_id)

    @pytest.mark.asyncio
    async def test_unknown_task_id_returns_none(self) -> None:
        manager = LocalTaskManager()
        assert await manager.get_status("no-such-id") is None


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty_by_default(self) -> None:
        manager = LocalTaskManager()
        assert await manager.list() == []

    @pytest.mark.asyncio
    async def test_list_includes_all_submitted_tasks(self) -> None:
        manager = LocalTaskManager()

        async def _work() -> str:
            return "done"

        id1 = await manager.submit(_work, summary="one")
        id2 = await manager.submit(_work, summary="two")
        await asyncio.sleep(0)

        records = await manager.list()
        assert {r.task_id for r in records} == {id1, id2}


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_running_task_marks_cancelled(self) -> None:
        manager = LocalTaskManager()
        started = asyncio.Event()

        async def _work() -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        cancelled = await manager.cancel(task_id)
        assert cancelled is True
        await asyncio.sleep(0)

        record = await manager.get_status(task_id)
        assert record is not None
        assert record.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_unknown_task_id_returns_false(self) -> None:
        manager = LocalTaskManager()
        assert await manager.cancel("no-such-id") is False

    @pytest.mark.asyncio
    async def test_cancel_already_completed_task_still_returns_true(self) -> None:
        # asyncio.Task.cancel() on an already-finished task is a no-op that
        # returns False from asyncio's perspective, but the manager only cares
        # whether the task_id was known, so it should still report True.
        manager = LocalTaskManager()

        async def _work() -> str:
            return "done"

        task_id = await manager.submit(_work, summary="s")
        await asyncio.sleep(0)

        assert await manager.cancel(task_id) is True
        record = await manager.get_status(task_id)
        assert record is not None
        assert record.status == TaskStatus.COMPLETED
