"""Unit tests for opendatasci.tasks.local.BackgroundTaskManager."""

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from opendatasci.tasks.base import BackgroundTaskProgressUpdate, BackgroundTaskRecord, BackgroundTaskStatus
from opendatasci.tasks.local import _MAX_RECORDS, BackgroundTaskManager


class TestSubmitAndStatus:
    @pytest.mark.asyncio
    async def test_completed_task_reports_result(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == BackgroundTaskStatus.COMPLETED
        assert record.result == "done"
        assert record.error is None
        assert record.finished_at is not None

    @pytest.mark.asyncio
    async def test_failed_task_reports_error_not_raised(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == BackgroundTaskStatus.FAILED
        assert record.error == "boom"
        assert record.result is None

    @pytest.mark.asyncio
    async def test_running_task_reports_running_status(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == BackgroundTaskStatus.RUNNING

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_unknown_task_id_returns_none(self) -> None:
        manager = BackgroundTaskManager()
        assert await manager.get_task("no-such-id") is None

    @pytest.mark.asyncio
    async def test_work_receives_only_the_task_id(self) -> None:
        # submit_task hands `work` just the task_id, not the record — there is
        # no BackgroundTaskManagerBase method for mutating a record from outside.
        manager = BackgroundTaskManager()
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
        manager = BackgroundTaskManager()
        assert await manager.list_tasks() == []

    @pytest.mark.asyncio
    async def test_list_includes_all_submitted_tasks(self) -> None:
        manager = BackgroundTaskManager()

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
        manager = BackgroundTaskManager()
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
        assert record.status == BackgroundTaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_unknown_task_id_returns_false(self) -> None:
        manager = BackgroundTaskManager()
        assert await manager.cancel_task("no-such-id") is False

    @pytest.mark.asyncio
    async def test_cancel_already_completed_task_still_returns_true(self) -> None:
        # asyncio.tasks.cancel() on an already-finished task is a no-op that
        # returns False from asyncio's perspective, but the manager only cares
        # whether the task_id was known, so it should still report True.
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert await manager.cancel_task(task_id) is True
        record = await manager.get_task(task_id)
        assert record is not None
        assert record.status == BackgroundTaskStatus.COMPLETED


class TestPublishResult:
    @pytest.mark.asyncio
    async def test_completed_task_written_to_output_root(self, tmp_path: Path) -> None:
        manager = BackgroundTaskManager(output_root=tmp_path)

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
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "the answer"

        await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_failed_task_not_published(self, tmp_path: Path) -> None:
        manager = BackgroundTaskManager(output_root=tmp_path)

        async def _work(task_id: object) -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert not (tmp_path / f"{task_id}.md").exists()


class TestUpsertRecord:
    @pytest.mark.asyncio
    async def test_upsert_inserts_new_record(self) -> None:
        manager = BackgroundTaskManager()
        task_id = uuid4()
        record = BackgroundTaskRecord(task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING)

        await manager.upsert_record(record)

        assert await manager.get_task(task_id) is record

    @pytest.mark.asyncio
    async def test_upsert_overwrites_existing_record(self) -> None:
        manager = BackgroundTaskManager()
        task_id = uuid4()
        await manager.upsert_record(
            BackgroundTaskRecord(task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING)
        )
        replacement = BackgroundTaskRecord(task_id=task_id, summary="s", status=BackgroundTaskStatus.COMPLETED)

        await manager.upsert_record(replacement)

        assert await manager.get_task(task_id) is replacement

    @pytest.mark.asyncio
    async def test_evicts_oldest_record_once_at_capacity(self) -> None:
        manager = BackgroundTaskManager()
        task_ids = [uuid4() for _ in range(_MAX_RECORDS)]
        for task_id in task_ids:
            await manager.upsert_record(
                BackgroundTaskRecord(task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING)
            )

        new_task_id = uuid4()
        await manager.upsert_record(
            BackgroundTaskRecord(task_id=new_task_id, summary="s", status=BackgroundTaskStatus.RUNNING)
        )

        assert len(await manager.list_tasks()) == _MAX_RECORDS
        assert await manager.get_task(task_ids[0]) is None
        assert await manager.get_task(task_ids[1]) is not None
        assert await manager.get_task(new_task_id) is not None

    @pytest.mark.asyncio
    async def test_updating_existing_record_does_not_evict(self) -> None:
        manager = BackgroundTaskManager()
        task_ids = [uuid4() for _ in range(_MAX_RECORDS)]
        for task_id in task_ids:
            await manager.upsert_record(
                BackgroundTaskRecord(task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING)
            )

        await manager.upsert_record(
            BackgroundTaskRecord(task_id=task_ids[0], summary="s", status=BackgroundTaskStatus.COMPLETED)
        )

        assert len(await manager.list_tasks()) == _MAX_RECORDS
        assert await manager.get_task(task_ids[0]) is not None


class TestPushTaskProgress:
    @pytest.mark.asyncio
    async def test_appends_progress_report(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.push_task_progress(
            task_id, BackgroundTaskProgressUpdate(done="a", ongoing="b", blockers=""), eta_seconds=5.0
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
        manager = BackgroundTaskManager()
        await manager.push_task_progress(
            "no-such-id", BackgroundTaskProgressUpdate(done="", ongoing="", blockers="")
        )
        # No exception, and nothing to read back — the only observable
        # behavior is that this doesn't raise.


class TestListenTaskUpdates:
    @pytest.mark.asyncio
    async def test_yields_completed_task(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")

        watcher = manager.listen_task_updates()
        record = await asyncio.wait_for(watcher.__anext__(), timeout=1)
        assert record.task_id == task_id
        assert record.status == BackgroundTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_yields_failed_and_cancelled_tasks_too(self) -> None:
        manager = BackgroundTaskManager()

        async def _fails(task_id: object) -> str:
            raise RuntimeError("boom")

        await manager.submit_task(_fails, summary="s")
        watcher = manager.listen_task_updates()
        record = await asyncio.wait_for(watcher.__anext__(), timeout=1)
        assert record.status == BackgroundTaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_blocks_until_next_completion(self) -> None:
        manager = BackgroundTaskManager()
        watcher = manager.listen_task_updates()

        pending = asyncio.ensure_future(watcher.__anext__())
        await asyncio.sleep(0.05)
        assert not pending.done()

        async def _work(task_id: object) -> str:
            return "done"

        await manager.submit_task(_work, summary="s")
        record = await asyncio.wait_for(pending, timeout=1)
        assert record.status == BackgroundTaskStatus.COMPLETED


class TestTaskUpdates:
    @pytest.mark.asyncio
    async def test_has_task_updates_false_by_default(self) -> None:
        manager = BackgroundTaskManager()
        assert manager.has_task_updates() is False

    @pytest.mark.asyncio
    async def test_completed_task_becomes_a_task_update(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        assert manager.has_task_updates() is True
        records = await manager.gather_task_updates()
        assert [r.task_id for r in records] == [task_id]

    @pytest.mark.asyncio
    async def test_multiple_completions_accumulate_and_gather_together(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        id1 = await manager.submit_task(_work, summary="one")
        id2 = await manager.submit_task(_work, summary="two")
        await asyncio.sleep(0)

        records = await manager.gather_task_updates()
        assert {r.task_id for r in records} == {id1, id2}

    @pytest.mark.asyncio
    async def test_gather_clears_the_buffer(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        await manager.gather_task_updates()

        assert manager.has_task_updates() is False
        assert await manager.gather_task_updates() == []

    @pytest.mark.asyncio
    async def test_failed_and_cancelled_tasks_are_also_task_updates(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _fails(task_id: object) -> str:
            raise RuntimeError("boom")

        async def _hangs(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        await manager.submit_task(_fails, summary="s")
        await asyncio.sleep(0)

        hang_id = await manager.submit_task(_hangs, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)
        await manager.cancel_task(hang_id)
        await asyncio.sleep(0)

        records = await manager.gather_task_updates()
        assert {r.status for r in records} == {BackgroundTaskStatus.FAILED, BackgroundTaskStatus.CANCELLED}

    @pytest.mark.asyncio
    async def test_independent_of_listen_task_updates(self) -> None:
        # Gathering must not consume the listener's stream: both observe the
        # same completions, but each has its own consumer.
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")

        listener = manager.listen_task_updates()
        listened_record = await asyncio.wait_for(listener.__anext__(), timeout=1)
        assert listened_record.task_id == task_id

        assert manager.has_task_updates() is True
        gathered = await manager.gather_task_updates()
        assert [r.task_id for r in gathered] == [task_id]
