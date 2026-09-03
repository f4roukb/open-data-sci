"""Unit tests for opendatasci.tasks.local.BackgroundTaskManager."""

import asyncio
import re
from pathlib import Path
from uuid import uuid4

import pytest

from opendatasci.tasks.base import (
    BackgroundTaskRecord,
    BackgroundTaskStatus,
    BackgroundTaskUpdateKind,
)
from opendatasci.tasks.local import (
    _MAX_ACTIVITY_ENTRIES,
    _MAX_ACTIVITY_ENTRY_LEN,
    _MAX_RECORDS,
    BackgroundTaskManager,
)


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
        record = BackgroundTaskRecord(
            task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING
        )

        await manager.upsert_record(record)

        assert await manager.get_task(task_id) is record

    @pytest.mark.asyncio
    async def test_upsert_overwrites_existing_record(self) -> None:
        manager = BackgroundTaskManager()
        task_id = uuid4()
        await manager.upsert_record(
            BackgroundTaskRecord(task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING)
        )
        replacement = BackgroundTaskRecord(
            task_id=task_id, summary="s", status=BackgroundTaskStatus.COMPLETED
        )

        await manager.upsert_record(replacement)

        assert await manager.get_task(task_id) is replacement

    @pytest.mark.asyncio
    async def test_evicts_oldest_record_once_at_capacity(self) -> None:
        manager = BackgroundTaskManager()
        task_ids = [uuid4() for _ in range(_MAX_RECORDS)]
        for task_id in task_ids:
            await manager.upsert_record(
                BackgroundTaskRecord(
                    task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING
                )
            )

        new_task_id = uuid4()
        await manager.upsert_record(
            BackgroundTaskRecord(
                task_id=new_task_id, summary="s", status=BackgroundTaskStatus.RUNNING
            )
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
                BackgroundTaskRecord(
                    task_id=task_id, summary="s", status=BackgroundTaskStatus.RUNNING
                )
            )

        await manager.upsert_record(
            BackgroundTaskRecord(
                task_id=task_ids[0], summary="s", status=BackgroundTaskStatus.COMPLETED
            )
        )

        assert len(await manager.list_tasks()) == _MAX_RECORDS
        assert await manager.get_task(task_ids[0]) is not None


class TestPushActivity:
    @pytest.mark.asyncio
    async def test_appends_activity_entry(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.push_activity(task_id, "tool: execute\nresult: ok")
        record = await manager.get_task(task_id)
        assert record is not None
        assert record.activity == ["tool: execute\nresult: ok"]

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_unknown_task_id_is_a_noop(self) -> None:
        manager = BackgroundTaskManager()
        await manager.push_activity("no-such-id", "entry")
        # No exception, and nothing to read back — the only observable
        # behavior is that this doesn't raise.

    @pytest.mark.asyncio
    async def test_caps_at_max_activity_entries(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        for i in range(_MAX_ACTIVITY_ENTRIES + 5):
            await manager.push_activity(task_id, f"entry {i}")

        record = await manager.get_task(task_id)
        assert record is not None
        assert len(record.activity) == _MAX_ACTIVITY_ENTRIES
        assert record.activity[0] == "entry 5"
        assert record.activity[-1] == f"entry {_MAX_ACTIVITY_ENTRIES + 4}"

        await manager.cancel_task(task_id)


class TestMonitorTask:
    @pytest.mark.asyncio
    async def test_matching_activity_produces_a_progress_update(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        (monitor_id,) = await manager.monitor_task(task_id, [r"error: \d+"])
        await manager.push_activity(task_id, "tool: execute\nresult: error: 42")

        updates = await manager.pull_task_updates()
        assert len(updates) == 1
        assert updates[0].task_id == task_id
        assert updates[0].kind == BackgroundTaskUpdateKind.PROGRESS
        assert updates[0].monitor_id == monitor_id
        assert updates[0].pattern == r"error: \d+"
        assert updates[0].matched_texts == ["error: 42"]

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_fires_once_then_is_removed(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.monitor_task(task_id, ["boom"])
        await manager.push_activity(task_id, "boom")
        await manager.push_activity(task_id, "boom again")

        updates = await manager.pull_task_updates()
        assert len(updates) == 1
        assert await manager.list_task_monitors(task_id) == {}

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_multiple_matches_in_one_entry_are_numbered_in_one_update(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.monitor_task(task_id, [r"error: \d+"])
        await manager.push_activity(task_id, "error: 1 then error: 2 then error: 3")

        updates = await manager.pull_task_updates()
        assert len(updates) == 1
        assert updates[0].matched_texts == ["error: 1", "error: 2", "error: 3"]

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_does_not_rematch_earlier_activity_entries(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.push_activity(task_id, "boom")
        await manager.monitor_task(task_id, ["boom"])
        await manager.push_activity(task_id, "quiet")

        assert manager.has_task_updates() is False

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_multiple_patterns_registered_in_one_call_get_distinct_monitor_ids(
        self,
    ) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        monitor_ids = await manager.monitor_task(task_id, ["a", "b"])
        assert len(monitor_ids) == 2
        assert len(set(monitor_ids)) == 2

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_same_pattern_on_different_tasks_gets_different_monitor_ids(self) -> None:
        manager = BackgroundTaskManager()
        started1 = asyncio.Event()

        async def _hangs1(task_id: object) -> str:
            started1.set()
            await asyncio.sleep(10)
            return "never"

        task_id1 = await manager.submit_task(_hangs1, summary="one")
        await asyncio.wait_for(started1.wait(), timeout=1)

        started2 = asyncio.Event()

        async def _hangs2(task_id: object) -> str:
            started2.set()
            await asyncio.sleep(10)
            return "never"

        task_id2 = await manager.submit_task(_hangs2, summary="two")
        await asyncio.wait_for(started2.wait(), timeout=1)

        (monitor_id1,) = await manager.monitor_task(task_id1, ["same"])
        (monitor_id2,) = await manager.monitor_task(task_id2, ["same"])
        assert monitor_id1 != monitor_id2

        await manager.cancel_task(task_id1)
        await manager.cancel_task(task_id2)

    @pytest.mark.asyncio
    async def test_does_not_match_past_the_truncation_cutoff(self) -> None:
        # Monitors scan the truncated entry, not the raw one, so regex time is
        # bounded by _MAX_ACTIVITY_ENTRY_LEN regardless of how much a single
        # tool call actually printed — a match placed past the cutoff is
        # therefore not found.
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.monitor_task(task_id, ["needle"])
        entry = ("x" * (_MAX_ACTIVITY_ENTRY_LEN + 10)) + "needle"
        await manager.push_activity(task_id, entry)

        assert manager.has_task_updates() is False

        record = await manager.get_task(task_id)
        assert record is not None
        assert record.activity[0].endswith("... (truncated)")

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_matches_within_the_truncation_cutoff(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        await manager.monitor_task(task_id, ["needle"])
        entry = "needle" + ("x" * (_MAX_ACTIVITY_ENTRY_LEN + 10))
        await manager.push_activity(task_id, entry)

        updates = await manager.pull_task_updates()
        assert updates[0].matched_texts == ["needle"]

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_unknown_task_id_is_a_noop(self) -> None:
        manager = BackgroundTaskManager()
        assert await manager.monitor_task("no-such-id", ["pattern"]) == []

    @pytest.mark.asyncio
    async def test_invalid_pattern_raises_and_registers_nothing(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        with pytest.raises(re.error):
            await manager.monitor_task(task_id, ["valid", "(unclosed"])

        assert await manager.list_task_monitors(task_id) == {}

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_monitors_dropped_on_terminal_status_without_a_match(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")
        await manager.monitor_task(task_id, ["never matches"])
        await asyncio.sleep(0)

        assert await manager.list_task_monitors(task_id) == {}

    @pytest.mark.asyncio
    async def test_monitors_dropped_when_task_fails(self) -> None:
        manager = BackgroundTaskManager()

        async def _fails(task_id: object) -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit_task(_fails, summary="s")
        await manager.monitor_task(task_id, ["never matches"])
        await asyncio.sleep(0)

        assert await manager.list_task_monitors(task_id) == {}

    @pytest.mark.asyncio
    async def test_monitors_dropped_when_task_cancelled(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _hangs(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_hangs, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)
        await manager.monitor_task(task_id, ["never matches"])

        await manager.cancel_task(task_id)
        await asyncio.sleep(0)

        assert await manager.list_task_monitors(task_id) == {}

    @pytest.mark.asyncio
    async def test_registering_the_same_pattern_twice_reuses_the_monitor_id(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        (monitor_id1,) = await manager.monitor_task(task_id, ["a"])
        (monitor_id2,) = await manager.monitor_task(task_id, ["a"])

        assert monitor_id1 == monitor_id2
        assert await manager.list_task_monitors(task_id) == {monitor_id1: "a"}

        await manager.cancel_task(task_id)

    @pytest.mark.asyncio
    async def test_dedup_is_scoped_to_the_task_not_global(self) -> None:
        manager = BackgroundTaskManager()
        started1 = asyncio.Event()

        async def _hangs1(task_id: object) -> str:
            started1.set()
            await asyncio.sleep(10)
            return "never"

        task_id1 = await manager.submit_task(_hangs1, summary="one")
        await asyncio.wait_for(started1.wait(), timeout=1)

        started2 = asyncio.Event()

        async def _hangs2(task_id: object) -> str:
            started2.set()
            await asyncio.sleep(10)
            return "never"

        task_id2 = await manager.submit_task(_hangs2, summary="two")
        await asyncio.wait_for(started2.wait(), timeout=1)

        (monitor_id1,) = await manager.monitor_task(task_id1, ["same"])
        (monitor_id2,) = await manager.monitor_task(task_id2, ["same"])

        assert monitor_id1 != monitor_id2

        await manager.cancel_task(task_id1)
        await manager.cancel_task(task_id2)

    @pytest.mark.asyncio
    async def test_dedup_alongside_a_new_pattern_in_the_same_call(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _work(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        (monitor_id_a,) = await manager.monitor_task(task_id, ["a"])
        monitor_id_a_again, monitor_id_b = await manager.monitor_task(task_id, ["a", "b"])

        assert monitor_id_a_again == monitor_id_a
        assert monitor_id_b != monitor_id_a
        assert set((await manager.list_task_monitors(task_id)).values()) == {"a", "b"}

        await manager.cancel_task(task_id)


class TestListTaskMonitors:
    @pytest.mark.asyncio
    async def test_empty_for_unknown_task_id(self) -> None:
        manager = BackgroundTaskManager()
        assert await manager.list_task_monitors("no-such-id") == {}

    @pytest.mark.asyncio
    async def test_returns_pattern_per_monitor_id(self) -> None:
        manager = BackgroundTaskManager()
        started = asyncio.Event()

        async def _hangs(task_id: object) -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit_task(_hangs, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        monitor_id1, monitor_id2 = await manager.monitor_task(task_id, ["a", "b"])

        assert await manager.list_task_monitors(task_id) == {monitor_id1: "a", monitor_id2: "b"}

        await manager.cancel_task(task_id)


class TestListenTaskUpdates:
    @pytest.mark.asyncio
    async def test_yields_completed_task(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")

        watcher = manager.listen_task_updates()
        event = await asyncio.wait_for(watcher.__anext__(), timeout=1)
        assert event.task_id == task_id
        assert manager._updates_by_id[event.update_id].status == BackgroundTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_yields_failed_and_cancelled_tasks_too(self) -> None:
        manager = BackgroundTaskManager()

        async def _fails(task_id: object) -> str:
            raise RuntimeError("boom")

        await manager.submit_task(_fails, summary="s")
        watcher = manager.listen_task_updates()
        event = await asyncio.wait_for(watcher.__anext__(), timeout=1)
        assert manager._updates_by_id[event.update_id].status == BackgroundTaskStatus.FAILED

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
        event = await asyncio.wait_for(pending, timeout=1)
        assert manager._updates_by_id[event.update_id].status == BackgroundTaskStatus.COMPLETED


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
        updates = await manager.pull_task_updates()
        assert [u.task_id for u in updates] == [task_id]
        assert updates[0].kind == BackgroundTaskUpdateKind.COMPLETED
        assert updates[0].status == BackgroundTaskStatus.COMPLETED
        assert updates[0].result == "done"

    @pytest.mark.asyncio
    async def test_multiple_completions_accumulate_and_pull_together(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        id1 = await manager.submit_task(_work, summary="one")
        id2 = await manager.submit_task(_work, summary="two")
        await asyncio.sleep(0)

        updates = await manager.pull_task_updates()
        assert {u.task_id for u in updates} == {id1, id2}

    @pytest.mark.asyncio
    async def test_pull_clears_the_buffer(self) -> None:
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        await manager.submit_task(_work, summary="s")
        await asyncio.sleep(0)

        await manager.pull_task_updates()

        assert manager.has_task_updates() is False
        assert await manager.pull_task_updates() == []

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

        updates = await manager.pull_task_updates()
        assert {u.status for u in updates} == {
            BackgroundTaskStatus.FAILED,
            BackgroundTaskStatus.CANCELLED,
        }

    @pytest.mark.asyncio
    async def test_independent_of_listen_task_updates(self) -> None:
        # Pulling must not consume the listener's stream: both observe the
        # same updates, but each has its own consumer.
        manager = BackgroundTaskManager()

        async def _work(task_id: object) -> str:
            return "done"

        task_id = await manager.submit_task(_work, summary="s")

        listener = manager.listen_task_updates()
        listened_event = await asyncio.wait_for(listener.__anext__(), timeout=1)
        assert listened_event.task_id == task_id

        assert manager.has_task_updates() is True
        pulled = await manager.pull_task_updates()
        assert [u.task_id for u in pulled] == [task_id]
