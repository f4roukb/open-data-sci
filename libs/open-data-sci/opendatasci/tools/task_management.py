"""Main-agent tools for checking on and cancelling background tasks.

These are main-agent-only: background (``synch_mode="async"``) work is
scheduled by the ``task`` tool, but workers themselves cannot spawn further
background work, so they have no need to check on or cancel it.
"""

from typing import Any, override

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from opendatasci.tasks.base import BaseTaskManager, TaskRecord
from opendatasci.tools.base import OpenDataSciBaseTool


def _format_record(record: TaskRecord) -> str:
    lines = [f"task_id: {record.task_id}", f"summary: {record.summary}", f"status: {record.status.value}"]
    if record.status.value == "completed":
        lines.append(f"result:\n{record.result}")
    elif record.status.value == "failed":
        lines.append(f"error: {record.error}")
    return "\n".join(lines)


class GetTaskStatusTool(OpenDataSciBaseTool):
    """Check on a previously scheduled background task."""

    class CallArgs(BaseModel):
        task_id: str | None = None

    name: str = "get_task_status"
    description: str = """
Check the status of a background task previously scheduled via the `task` tool with
`synch_mode="async"`.

Args:
    task_id: The task ID returned when the background task was scheduled. Omit it to
             list every background task tracked in this session (id, summary, status).
""".strip()

    args_schema: type[BaseModel] = CallArgs

    task_manager: BaseTaskManager

    @override
    async def _arun(self, task_id: str | None = None, **kwargs: Any) -> str:
        if task_id is None:
            records = await self.task_manager.list()
            if not records:
                return "No background tasks have been scheduled in this session."
            return "\n\n---\n\n".join(_format_record(r) for r in records)

        record = await self.task_manager.get_status(task_id)
        if record is None:
            return f"No background task found with task_id={task_id}."
        return _format_record(record)


class CancelTaskTool(OpenDataSciBaseTool):
    """Cancel a previously scheduled background task."""

    class CallArgs(BaseModel):
        task_id: str

    name: str = "cancel_task"
    description: str = """
Cancel a background task previously scheduled via the `task` tool with `synch_mode="async"`.

Cancellation is best-effort: work already running on a dedicated OS thread (`run_mode="parallel"`
in the `task` tool) may keep running to completion in the background even after cancellation is
requested; its result is simply discarded.

Args:
    task_id: The task ID returned when the background task was scheduled.
""".strip()

    args_schema: type[BaseModel] = CallArgs

    task_manager: BaseTaskManager

    @override
    async def _arun(self, task_id: str, **kwargs: Any) -> str:
        cancelled = await self.task_manager.cancel(task_id)
        if not cancelled:
            return f"No background task found with task_id={task_id}."
        return f"Cancellation requested for task_id={task_id}."


def create_task_management_tools(task_manager: BaseTaskManager) -> list[BaseTool]:
    """Return the ``get_task_status`` and ``cancel_task`` tools.

    *task_manager* must be the same instance passed to
    :func:`opendatasci.tools.workers.create_worker_tools` so these tools can see
    the background tasks scheduled by the ``task`` tool.
    """
    return [
        GetTaskStatusTool(task_manager=task_manager),
        CancelTaskTool(task_manager=task_manager),
    ]
