"""LangGraph graph and state utilities."""

from langgraph.types import StateSnapshot


def is_interrupt_state_snapshot(state: StateSnapshot) -> bool:
    """Return True if *state* contains at least one pending LangGraph interrupt."""
    return any(intr for task in state.tasks for intr in task.interrupts)
