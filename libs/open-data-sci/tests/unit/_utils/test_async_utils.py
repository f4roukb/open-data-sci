"""Unit tests for opendatasci._utils.async_utils."""


import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from opendatasci._utils.async_utils import run_in_executor


class TestRunInExecutor:
    @pytest.mark.asyncio
    async def test_returns_function_result(self) -> None:
        result = await run_in_executor(lambda: 42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_passes_positional_args(self) -> None:
        result = await run_in_executor(lambda a, b: a + b, 3, 4)
        assert result == 7

    @pytest.mark.asyncio
    async def test_runs_in_separate_thread(self) -> None:
        import threading

        caller_thread = threading.current_thread()
        executor_thread: list[threading.Thread] = []

        def capture_thread() -> None:
            executor_thread.append(threading.current_thread())

        await run_in_executor(capture_thread)
        assert executor_thread[0] is not caller_thread

    @pytest.mark.asyncio
    async def test_custom_executor_is_used(self) -> None:
        custom = ThreadPoolExecutor(max_workers=1)
        try:
            result = await run_in_executor(lambda: "ok", executor=custom)
            assert result == "ok"
        finally:
            custom.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_propagates_exception_from_callable(self) -> None:
        def boom() -> None:
            raise ValueError("from thread")

        with pytest.raises(ValueError, match="from thread"):
            await run_in_executor(boom)

    @pytest.mark.asyncio
    async def test_concurrent_calls_all_complete(self) -> None:
        results = await asyncio.gather(
            run_in_executor(lambda: 1),
            run_in_executor(lambda: 2),
            run_in_executor(lambda: 3),
        )
        assert sorted(results) == [1, 2, 3]
