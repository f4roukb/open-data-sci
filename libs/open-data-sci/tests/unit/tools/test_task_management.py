"""Unit tests for opendatasci.tools.task_management."""

import asyncio

import pytest

from opendatasci.tasks.local import LocalTaskManager
from opendatasci.tools.task_management import (
    CancelTaskTool,
    GetTaskStatusTool,
    create_task_management_tools,
)


class TestCreateTaskManagementTools:
    def test_returns_status_and_cancel_tools(self) -> None:
        tools = create_task_management_tools(LocalTaskManager())
        names = {t.name for t in tools}
        assert names == {"get_task_status", "cancel_task"}


class TestGetTaskStatusTool:
    @pytest.mark.asyncio
    async def test_unknown_task_id(self) -> None:
        tool = GetTaskStatusTool(task_manager=LocalTaskManager())
        result = await tool.ainvoke({"task_id": "no-such-id"})
        assert "no-such-id" in result
        assert "no background task found" in result.lower()

    @pytest.mark.asyncio
    async def test_completed_task_reports_result(self) -> None:
        manager = LocalTaskManager()

        async def _work() -> str:
            return "the answer"

        task_id = await manager.submit(_work, summary="s")
        await asyncio.sleep(0)

        tool = GetTaskStatusTool(task_manager=manager)
        result = await tool.ainvoke({"task_id": task_id})
        assert "completed" in result.lower()
        assert "the answer" in result

    @pytest.mark.asyncio
    async def test_failed_task_reports_error(self) -> None:
        manager = LocalTaskManager()

        async def _work() -> str:
            raise RuntimeError("boom")

        task_id = await manager.submit(_work, summary="s")
        await asyncio.sleep(0)

        tool = GetTaskStatusTool(task_manager=manager)
        result = await tool.ainvoke({"task_id": task_id})
        assert "failed" in result.lower()
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_no_task_id_lists_all_tasks(self) -> None:
        manager = LocalTaskManager()

        async def _work() -> str:
            return "done"

        id1 = await manager.submit(_work, summary="first")
        id2 = await manager.submit(_work, summary="second")
        await asyncio.sleep(0)

        tool = GetTaskStatusTool(task_manager=manager)
        result = await tool.ainvoke({})
        assert id1 in result
        assert id2 in result

    @pytest.mark.asyncio
    async def test_no_task_id_and_no_tasks_reports_none_scheduled(self) -> None:
        tool = GetTaskStatusTool(task_manager=LocalTaskManager())
        result = await tool.ainvoke({})
        assert "no background tasks" in result.lower()


class TestCancelTaskTool:
    @pytest.mark.asyncio
    async def test_unknown_task_id(self) -> None:
        tool = CancelTaskTool(task_manager=LocalTaskManager())
        result = await tool.ainvoke({"task_id": "no-such-id"})
        assert "no background task found" in result.lower()

    @pytest.mark.asyncio
    async def test_cancels_running_task(self) -> None:
        manager = LocalTaskManager()
        started = asyncio.Event()

        async def _work() -> str:
            started.set()
            await asyncio.sleep(10)
            return "never"

        task_id = await manager.submit(_work, summary="s")
        await asyncio.wait_for(started.wait(), timeout=1)

        tool = CancelTaskTool(task_manager=manager)
        result = await tool.ainvoke({"task_id": task_id})
        assert "cancellation requested" in result.lower()
        assert task_id in result
