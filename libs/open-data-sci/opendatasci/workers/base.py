"""Strategy interface for executing a batch of spawned worker subtasks."""

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Coroutine

from langchain_core.runnables import RunnableConfig

# (idx, subtask, outer_config, schedule) -> worker result text.
# ``schedule`` is None for cooperative execution (use the running loop directly)
# or a callable that hands an event-dispatch coroutine back to the owning loop.
RunOne = Callable[
    [int, Any, RunnableConfig, "Callable[[Coroutine[Any, Any, None]], Any] | None"],
    Awaitable[str],
]


class BaseWorker(ABC):
    """Strategy for executing a batch of spawned worker subtasks and collecting results."""

    @abstractmethod
    async def run(
        self,
        subtasks: list[Any],
        outer_config: RunnableConfig,
        run_one: RunOne,
    ) -> list[Any]:
        """Execute *run_one* for each subtask, returning results/exceptions in order."""
