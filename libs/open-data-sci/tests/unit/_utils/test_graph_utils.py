"""Unit tests for opendatasci._utils.graph_utils."""

from langgraph.types import Interrupt, PregelTask, StateSnapshot

from opendatasci._utils.graph_utils import is_interrupt_state_snapshot


def _snapshot(*task_interrupts: tuple[Interrupt, ...]) -> StateSnapshot:
    tasks = tuple(
        PregelTask(id=str(i), name=f"task_{i}", path=(), interrupts=interrupts)
        for i, interrupts in enumerate(task_interrupts)
    )
    return StateSnapshot(
        values={},
        next=(),
        config={"configurable": {}},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=tasks,
    )


def _interrupt(value: str = "question") -> Interrupt:
    return Interrupt(value=value)


class TestIsInterruptStateSnapshot:
    def test_no_tasks_returns_false(self) -> None:
        assert is_interrupt_state_snapshot(_snapshot()) is False

    def test_task_with_no_interrupts_returns_false(self) -> None:
        assert is_interrupt_state_snapshot(_snapshot(())) is False

    def test_multiple_tasks_all_without_interrupts_returns_false(self) -> None:
        assert is_interrupt_state_snapshot(_snapshot((), ())) is False

    def test_single_task_with_interrupt_returns_true(self) -> None:
        assert is_interrupt_state_snapshot(_snapshot((_interrupt(),))) is True

    def test_multiple_interrupts_on_one_task_returns_true(self) -> None:
        assert is_interrupt_state_snapshot(_snapshot((_interrupt("q1"), _interrupt("q2")))) is True

    def test_only_second_task_has_interrupt_returns_true(self) -> None:
        assert is_interrupt_state_snapshot(_snapshot((), (_interrupt(),))) is True

    def test_interrupt_value_does_not_affect_result(self) -> None:
        assert is_interrupt_state_snapshot(_snapshot((_interrupt(""),))) is True
