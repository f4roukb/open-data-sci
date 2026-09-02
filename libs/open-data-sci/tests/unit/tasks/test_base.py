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
            matched_text="error: 42",
        )

        text = update.to_message().content[0]["text"]

        assert text.startswith(f"Update from monitor_id={monitor_id} on task_id={task_id} ")
        assert "error: 42" in text
        assert repr(r"error: \d+") in text

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
