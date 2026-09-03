"""Unit tests for opendatasci.tasks.base.BackgroundTaskUpdate.to_message."""

from uuid import uuid4

from opendatasci.tasks.base import (
    BackgroundTaskStatus,
    BackgroundTaskUpdate,
    BackgroundTaskUpdateKind,
)


class TestToMessage:
    def test_progress_kind_states_monitor_id_and_task_id(self) -> None:
        task_id = uuid4()
        monitor_id = uuid4()
        update = BackgroundTaskUpdate(
            update_id=uuid4(),
            task_id=task_id,
            kind=BackgroundTaskUpdateKind.PROGRESS,
            summary="s",
            monitor_id=monitor_id,
            pattern=r"error: \d+",
            matched_texts=["error: 42"],
        )

        text = update.to_message().content[0]["text"]

        assert f"monitor(id={monitor_id})" in text
        assert f"task(id={task_id})" in text
        assert "will not fire again" in text
        assert "Match 0: error: 42" in text

    def test_progress_kind_numbers_multiple_matches(self) -> None:
        update = BackgroundTaskUpdate(
            update_id=uuid4(),
            task_id=uuid4(),
            kind=BackgroundTaskUpdateKind.PROGRESS,
            summary="s",
            monitor_id=uuid4(),
            pattern=r"error: \d+",
            matched_texts=["error: 1", "error: 2", "error: 3"],
        )

        text = update.to_message().content[0]["text"]

        assert "Match 0: error: 1" in text
        assert "Match 1: error: 2" in text
        assert "Match 2: error: 3" in text

    def test_completed_kind_states_result(self) -> None:
        update = BackgroundTaskUpdate(
            update_id=uuid4(),
            task_id=uuid4(),
            kind=BackgroundTaskUpdateKind.COMPLETED,
            summary="s",
            status=BackgroundTaskStatus.COMPLETED,
            result="the answer",
        )

        text = update.to_message().content[0]["text"]

        assert "the answer" in text

    def test_failed_kind_states_error(self) -> None:
        update = BackgroundTaskUpdate(
            update_id=uuid4(),
            task_id=uuid4(),
            kind=BackgroundTaskUpdateKind.COMPLETED,
            summary="s",
            status=BackgroundTaskStatus.FAILED,
            error="boom",
        )

        text = update.to_message().content[0]["text"]

        assert "boom" in text

    def test_cancelled_kind_states_cancelled(self) -> None:
        update = BackgroundTaskUpdate(
            update_id=uuid4(),
            task_id=uuid4(),
            kind=BackgroundTaskUpdateKind.COMPLETED,
            summary="s",
            status=BackgroundTaskStatus.CANCELLED,
        )

        text = update.to_message().content[0]["text"]

        assert "cancelled" in text
