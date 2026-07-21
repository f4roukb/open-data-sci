"""True-parallel (dedicated thread + event loop per worker) execution strategy."""

import asyncio
from typing import Any, Coroutine

from langchain_core.runnables import RunnableConfig

from opendatasci.workers.base import BaseWorker, RunOne


class ParallelWorker(BaseWorker):
    """Runs each subtask on its own OS thread with a dedicated event loop.

    Genuinely parallel rather than cooperatively concurrent: each worker gets
    its own thread and event loop, so a worker blocked on CPU-bound or
    blocking I/O work cannot stall the others. Worker events are dispatched
    back onto the caller's loop via ``run_coroutine_threadsafe`` since the
    callback machinery bound to *outer_config* lives there.
    """

    async def run(
        self,
        subtasks: list[Any],
        outer_config: RunnableConfig,
        run_one: RunOne,
    ) -> list[Any]:
        main_loop = asyncio.get_running_loop()

        def schedule(coro: Coroutine[Any, Any, None]) -> None:
            asyncio.run_coroutine_threadsafe(coro, main_loop)

        def run_in_thread(idx: int, subtask: Any) -> str:
            thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(thread_loop)
            try:
                return thread_loop.run_until_complete(
                    run_one(idx, subtask, outer_config, schedule)
                )
            finally:
                asyncio.set_event_loop(None)
                thread_loop.close()

        return await asyncio.gather(
            *[
                main_loop.run_in_executor(None, run_in_thread, i, t)
                for i, t in enumerate(subtasks)
            ],
            return_exceptions=True,
        )
