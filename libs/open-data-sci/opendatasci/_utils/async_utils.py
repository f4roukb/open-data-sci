import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, TypeVar

_T = TypeVar("_T")

_DEFAULT_MAX_WORKERS: int = 32


@lru_cache(1)
def get_default_threadpool_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=_DEFAULT_MAX_WORKERS)


async def run_in_executor(
    func: Callable[..., _T], *args: Any, executor: ThreadPoolExecutor | None = None
) -> _T:
    """Run a blocking callable in a thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor or get_default_threadpool_executor(), func, *args)
