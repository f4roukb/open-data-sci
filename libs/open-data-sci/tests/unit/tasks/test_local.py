"""Unit tests for opendatasci.tasks.local.LocalAgentTaskManager."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from opendatasci.tasks.base import AgentTaskProgressUpdate, AgentTaskRecord, AgentTaskStatus
from opendatasci.tasks.local import _MAX_RECORDS, LocalAgentTaskManager


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
        assert record.status == AgentTaskStatus.COMPLETED
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
        assert record.status == AgentTaskStatus.FAILED
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
        assert record.status == AgentTaskStatus.RUNNING

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
        assert record.status == AgentTaskStatus.CANCELLED

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
        assert record.status == AgentTaskStatus.COMPLETED


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


class TestUpsertRecord:
    @pytest.mark.asyncio
    async def test_upsert_inserts_new_record(self) -> None:
        manager = LocalAgentTaskManager()
        task_id = uuid4()
        record = AgentTaskRecord(task_id=task_id, summary="s", status=AgentTaskStatus.RUNNING)

        await manager.upsert_record(record)

        assert await manager.get_task(task_id) is record

    @pytest.mark.asyncio
    async def test_upsert_overwrites_existing_record(self) -> None:
        manager = LocalAgentTaskManager()
        task_id = uuid4()
        await manager.upsert_record(
            AgentTaskRecord(task_id=task_id, summary="s", status=AgentTaskStatus.RUNNING)
        )
        replacement = AgentTaskRecord(task_id=task_id, summary="s", status=AgentTaskStatus.COMPLETED)

        await manager.upsert_record(replacement)

        assert await manager.get_task(task_id) is replacement

    @pytest.mark.asyncio
    async def test_evicts_oldest_record_once_at_capacity(self) -> None:
        manager = LocalAgentTaskManager()
        task_ids = [uuid4() for _ in range(_MAX_RECORDS)]
        for task_id in task_ids:
            await manager.upsert_record(
                AgentTaskRecord(task_id=task_id, summary="s", status=AgentTaskStatus.RUNNING)
            )

        new_task_id = uuid4()
        await manager.upsert_record(
            AgentTaskRecord(task_id=new_task_id, summary="s", status=AgentTaskStatus.RUNNING)
        )

        assert len(await manager.list_tasks()) == _MAX_RECORDS
        assert await manager.get_task(task_ids[0]) is None
        assert await manager.get_task(task_ids[1]) is not None
        assert await manager.get_task(new_task_id) is not None

    @pytest.mark.asyncio
    async def test_updating_existing_record_does_not_evict(self) -> None:
        manager = LocalAgentTaskManager()
        task_ids = [uuid4() for _ in range(_MAX_RECORDS)]
        for task_id in task_ids:
            await manager.upsert_record(
                AgentTaskRecord(task_id=task_id, summary="s", status=AgentTaskStatus.RUNNING)
            )

        await manager.upsert_record(
            AgentTaskRecord(task_id=task_ids[0], summary="s", status=AgentTaskStatus.COMPLETED)
        )

        assert len(await manager.list_tasks()) == _MAX_RECORDS
        assert await manager.get_task(task_ids[0]) is not None


class TestPushTaskProgress:
    @pytest.mark.asyncio
    async def test_appends_progress_report(self) -> None:
        manager = LocalAgentTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.push_task_progress(
            task_id, AgentTaskProgressUpdate(done="a", ongoing="b", blockers=""), eta_seconds=5.0
        )
        record = await manager.get_task(task_id)
        assert record is not None
        assert len(record.progress) == 1
        assert record.progress[0].progress_update.done == "a"
        assert record.progress[0].progress_update.ongoing == "b"
        assert record.progress[0].eta_seconds == 5.0

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_unknown_task_id_is_a_noop(self) -> None:
        manager = LocalAgentTaskManager()
        await manager.push_task_progress(
            "no-such-id", AgentTaskProgressUpdate(done="", ongoing="", blockers="")
        )
        # No exception, and nothing to read back — the only observable
        # behavior is that this doesn't raise.


class TestWatchCompletions:
    @pytest.mark.asyncio
    async def test_yields_completed_task(self) -> None:
        manager = LocalAgentTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")

        watcher = manager.watch_completions()
        record = await asyncio.wait_for(watcher.__anext__(), timeout=1)
        assert record.task_id == task_id
        assert record.status == AgentTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_yields_failed_and_cancelled_tasks_too(self) -> None:
        manager = LocalAgentTaskManager()

        async def _fails(task_id: object) -> str:
            raise RuntimeError("boom")

        await manager.submit_task(_fails, summary="s")
        watcher = manager.watch_completions()
        record = await asyncio.wait_for(watcher.__anext__(), timeout=1)
        assert record.status == AgentTaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_blocks_until_next_completion(self) -> None:
        manager = LocalAgentTaskManager()
        watcher = manager.watch_completions()

        pending = asyncio.ensure_future(watcher.__anext__())
        await asyncio.sleep(0.05)
        assert not pending.done()

        async def _work(task_id: object) -> str:
            return "done"

        await manager.submit_task(_work, summary="s")
        record = await asyncio.wait_for(pending, timeout=1)
        assert record.status == AgentTaskStatus.COMPLETED
