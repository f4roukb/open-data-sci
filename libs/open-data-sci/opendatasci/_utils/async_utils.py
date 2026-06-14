import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_executor = ThreadPoolExecutor(max_workers=16)


async def run_in_executor(func: Any, *args: Any, executor: ThreadPoolExecutor | None = None) -> Any:
    """Run a blocking callable in a thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor or _executor, func, *args)
