"""Cooperative (single event loop) worker execution strategy."""

import asyncio
from typing import Any

from langchain_core.runnables import RunnableConfig

from opendatasci.workers.base import BaseWorker, RunOne


class ConcurrentWorker(BaseWorker):
    """Runs all subtasks cooperatively, interleaved on the current event loop.

    All workers share a single thread: genuine I/O overlap (e.g. concurrent
    LLM calls), but a blocking segment in one worker stalls the others.
    """

    async def run(
        self,
        subtasks: list[Any],
        outer_config: RunnableConfig,
        run_one: RunOne,
    ) -> list[Any]:
        return await asyncio.gather(
            *[run_one(i, t, outer_config, None) for i, t in enumerate(subtasks)],
            return_exceptions=True,
        )
