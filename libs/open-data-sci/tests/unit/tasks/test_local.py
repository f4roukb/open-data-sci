"""Unit tests for opendatasci.tasks.local.LocalAgentTaskManager."""

import asyncio
from pathlib import Path

import pytest

from opendatasci.tasks.base import TaskStatus
from opendatasci.tasks.local import LocalAgentTaskManager


class TestSubmitAndStatus:
    @pytest.mark.asyncio
    async def test_completed_task_reports_result(self) -> None:
        manager = LocalAgentTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == TaskStatus.COMPLETED
        assert record.result == "done"
        assert record.error is None
        assert record.finished_at is not None

    @pytest.mark.asyncio
    async def test_failed_task_reports_error_not_raised(self) -> None:
        manager = LocalAgentTaskManager()

        async def _work(task_id: object) -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == TaskStatus.FAILED
        assert record.error == "boom"
        assert record.result is None

    @pytest.mark.asyncio
    async def test_running_task_reports_running_status(self) -> None:
        manager = LocalAgentTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == TaskStatus.RUNNING

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_unknown_task_id_returns_none(self) -> None:
        manager = LocalAgentTaskManager()
        assert await manager.get_task("no-such-id") is None

    @pytest.mark.asyncio
    async def test_work_receives_only_the_task_id(self) -> None:
        # submit_task hands `work` just the task_id, not the record — there is
        # no AgentTaskManagerBase method for mutating a record from outside.
        manager = LocalAgentTaskManager()
        received: list[object] = []

        async def _work(task_id: object) -> str:
            received.append(task_id)
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert received == [task_id]


class TestList:
    @pytest.mark.asyncio
    async def test_list_empty_by_default(self) -> None:
        manager = LocalAgentTaskManager()
        assert await manager.list_tasks() == []

    @pytest.mark.asyncio
    async def test_list_includes_all_submitted_tasks(self) -> None:
        manager = LocalAgentTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        id1 = await manager.submit_task(_work, summary="one")
        id2 = await manager.submit_task(_work, summary="two")
        await asyncio.sleep(0)

        records = await manager.list_tasks()
        assert {r.task_id for r in records} == {id1, id2}


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_running_task_marks_cancelled(self) -> None:
        manager = LocalAgentTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        cancelled = await manager.cancel_task(task_id)
        assert cancelled is True
        await asyncio.sleep(0)

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_unknown_task_id_returns_false(self) -> None:
        manager = LocalAgentTaskManager()
        assert await manager.cancel_task("no-such-id") is False

    @pytest.mark.asyncio
    async def test_cancel_already_completed_task_still_returns_true(self) -> None:
        # asyncio.tasks.cancel() on an already-finished task is a no-op that
        # returns False from asyncio's perspective, but the manager only cares
        # whether the task_id was known, so it should still report True.
        manager = LocalAgentTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert await manager.cancel_task(task_id) is True
        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == TaskStatus.COMPLETED


class TestPublishResult:
    @pytest.mark.asyncio
    async def test_completed_task_written_to_output_root(self, tmp_path: Path) -> None:
        manager = LocalAgentTaskManager(output_root=tmp_path)

        async def _work(task_id: object) -> str:
            return "the answer"

        task_id = await manager.submit_task(_work, summary="s")
        await manager._tasks[task_id]

        output_file = tmp_path / f"{task_id}.md"
        assert output_file.exists()
        content = output_file.read_text(encoding="utf-8")
        assert "the answer" in content
        assert str(task_id) in content

    @pytest.mark.asyncio
    async def test_no_output_root_skips_publish(self, tmp_path: Path) -> None:
        manager = LocalAgentTaskManager()

        async def _work(task_id: object) -> str:
            return "the answer"

        await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_failed_task_not_published(self, tmp_path: Path) -> None:
        manager = LocalAgentTaskManager(output_root=tmp_path)

        async def _work(task_id: object) -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert not (tmp_path / f"{task_id}.md").exists()
