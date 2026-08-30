"""Task tools: spawn worker subtasks (``task``) and manage background tasks
(``check_task``, ``list_tasks``, ``stop_task``).
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, override
from uuid import UUID

from annotated_types import MaxLen, MinLen
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from opendatasci.agents.workers import WorkerAgent
from opendatasci.configs import OpenDataSciConfig
from opendatasci.prompts.prompt_templates import WORKER_SYSTEM_PROMPT
from opendatasci.sandbox.base import BaseSandboxFactory
from opendatasci.skills import BaseSkillStore
from opendatasci.tasks.base import (
    AgentTaskManagerBase,
    AgentTaskProgressUpdate,
    WorkerTaskRecord,
    AgentTaskStatus,
)
from opendatasci.tools.base import OpenDataSciBaseTool
from opendatasci.tools.coding import create_cli_tools, create_coding_tools
from opendatasci.tools.skills import create_skill_tools
from opendatasci.workspace.base import BaseWorkspace

logger = logging.getLogger(__name__)


class RunMode(StrEnum):
    """Whether the ``task`` tool waits for results (foreground) or schedules them in the background."""

    FOREGROUND = auto()
    BACKGROUND = auto()


class ReportProgressTool(OpenDataSciBaseTool):
    """Report progress on the background task this worker is running under.

    Bound with a fixed ``task_id`` at construction time — only handed to a
    worker running in the background (``run_mode="background"``), since a
    foreground worker has no task record to report against.
    """

    class CallArgs(BaseModel):
        done: str
        """What has been completed so far."""
        ongoing: str
        """What is currently being worked on."""
        blockers: str
        """Anything blocking progress, or an empty string if none."""
        eta_seconds: float | None = None
        """Estimated seconds remaining, if it can be reasonably guessed."""

    name: str = "report_progress"
    description: str = """
Report progress on the background task you are currently running as. Call this
periodically on long-running work so whoever scheduled you can see what's done,
what's ongoing, and what's blocking further progress without waiting for you to finish.

Args:
    done:        What has been completed so far.
    ongoing:     What you are currently working on.
    blockers:    Anything blocking progress, or an empty string if none.
    eta_seconds: Estimated seconds remaining, if it can be reasonably guessed.
""".strip()

    args_schema: type[BaseModel] = CallArgs

    task_id: UUID
    background_task_manager: AgentTaskManagerBase

    @override
    async def _arun(
        self,
        done: str,
        ongoing: str,
        blockers: str,
        eta_seconds: float | None = None,
        **kwargs: Any,
    ) -> str:
        await self.background_task_manager.push_task_progress(
            self.task_id,
            AgentTaskProgressUpdate(done=done, ongoing=ongoing, blockers=blockers),
            eta_seconds=eta_seconds,
        )
        return "Progress recorded."


class TaskTool(OpenDataSciBaseTool):
    """Spawn parallel worker agents to execute independent subtasks."""

    class TaskDetails(BaseModel):
        """Everything a worker needs to run a single subtask to completion."""

        subtask: str
        """Specific, self-contained subtask with all context the worker needs."""
        summary: str
        """3-4 word status label (e.g. ``'Shapiro-Wilk on age'``)."""
        skill: str | None = None
        """Optional skill profile to preload before the subtask runs
        (e.g. ``'data_science'``, ``'ml_engineering'``). ``None`` = no skill."""

    class CallArgs(BaseModel):
        subtasks: Annotated[list["TaskTool.TaskDetails"], MinLen(1), MaxLen(3)]
        summary: str
        communication: str
        run_mode: RunMode = RunMode.FOREGROUND

    name: str = "task"
    description: str = """
Spawn 1-3 independent workers to execute narrow, concrete subtasks.

Workers are fully isolated: no shared state, no conversation history, no context from
other subtasks. Each subtask runs to completion independently before results are collected.

# When to use this tool
- For specific, orthogonal actions with a clearly defined outcome that can run concurrently:
  e.g. "Run Shapiro-Wilk on `age`", "Investigate the distribution of `revenue`".
- When the task has already been planned and workers execute individual, independent steps.

# When NOT to use this tool
- When one subtask's result informs another — workers cannot pass data to each other.
- For broad exploration or re-planning — workers execute, they don't strategise.
- For a single task: just execute directly; one worker adds latency with no benefit.

# How to use this tool
- Write every subtask description as fully self-contained: include dataset names,
  variable names, target columns, and any context the worker needs from the conversation.
- Assign a ``skill`` when the subtask benefits from domain-specific guidance.
- Assign a ``run_mode``:

  # When to use "background"
  - You don't need the result right away and can keep helping the user with
    something else in the meantime.
  - The subtask is expected to take a while (heavy training runs, large-scale
    data processing, anything that would otherwise stall the conversation).

  # When to use "foreground" (default)
  - You need the result before you can proceed with anything else.
  - The subtask is quick — scheduling overhead outweighs the benefit.

  ``"background"`` schedules each subtask in the background and returns immediately
  with one task ID per subtask instead of blocking on completion. Check on scheduled
  work with `check_task`/`list_tasks`, stop it with `stop_task`.

Args:
    subtasks:      1-3 subtask descriptors (see TaskDetails fields).
    communication: Brief message to the user about what you're doing
                   (e.g. "Running three checks in parallel.").
    run_mode:      "foreground" to wait for and return the result; "background" to schedule
                   each subtask in the background and return its task ID immediately. Prefer
                   "background" for long-running work.
""".strip()

    args_schema: type[BaseModel] = CallArgs

    workspace: BaseWorkspace
    datasci_config: OpenDataSciConfig
    sandbox_factory: BaseSandboxFactory
    skill_store: BaseSkillStore
    background_task_manager: AgentTaskManagerBase

    async def _arun_one(
        self,
        idx: int,
        subtask: TaskDetails,
        task_id: UUID | None,
        outer_config: RunnableConfig,
    ) -> str:
        """Run a single worker subtask inside its own sandbox.

        *task_id* is the background task ID this worker is running under, or
        ``None`` when running in the foreground.
        """
        initial_skill = None
        if subtask.skill is not None:
            initial_skill = self.skill_store.load(subtask.skill)
            if initial_skill is None:
                logger.warning("Subtask %d: skill %r not found", idx, subtask.skill)

        def emit(event_type: str, content: str, metadata: dict[str, Any] | None = None) -> None:
            coro = adispatch_custom_event(
                "task_event",
                {
                    "task_idx": idx,
                    "event_type": event_type,
                    "content": content,
                    **(metadata or {}),
                },
                config=outer_config,
            )
            asyncio.get_running_loop().create_task(coro)

        cancelled = False
        exc_info: BaseException | None = None

        async with self.sandbox_factory.create(
            workspace_path=Path(self.workspace.get_reference())
        ) as worker_sandbox:
            tools: list[BaseTool] = [
                *create_coding_tools(worker_sandbox),
                *create_cli_tools(worker_sandbox),
                *create_skill_tools(self.skill_store),
            ]
            if task_id is not None:
                tools.append(
                    ReportProgressTool(task_id=task_id, background_task_manager=self.background_task_manager)
                )

            agent = WorkerAgent(tools=tools, config=self.datasci_config)
            emit("task_started", subtask.summary)

            try:
                return await agent.ainvoke(
                    subtask.subtask,
                    WORKER_SYSTEM_PROMPT,
                    on_event=emit,
                    initial_active_skills=[initial_skill] if initial_skill is not None else [],
                )
            except asyncio.CancelledError:
                cancelled = True
                raise
            except RuntimeError as exc:
                exc_info = exc
                return str(exc)
            except Exception as exc:
                exc_info = exc
                raise
            finally:
                if not cancelled:
                    success = exc_info is None
                    emit("task_finished", subtask.summary, {"success": success})
                    emit("task_done", subtask.summary, {"success": success})

    async def _run_to_completion(
        self,
        subtasks: list[TaskDetails],
        outer_config: RunnableConfig,
    ) -> str:
        timeout = self.datasci_config.worker_timeout_seconds
        results = await asyncio.wait_for(
            asyncio.gather(
                *[self._arun_one(i, t, None, outer_config) for i, t in enumerate(subtasks)],
                return_exceptions=True,
            ),
            timeout=timeout,
        )

        sections: list[str] = []
        for i, (subtask, result) in enumerate(zip(subtasks, results), 1):
            if isinstance(result, BaseException):
                logger.error(
                    "Worker %d (%s) failed: %s: %s",
                    i,
                    subtask.summary,
                    type(result).__name__,
                    result,
                )
                output = f"Error: worker failed — {type(result).__name__}: {result}"
            else:
                output = result
            sections.append(f"### WorkerAgent {i}: {subtask.subtask}\n\n{output}")
        return "\n\n---\n\n".join(sections)

    @override
    async def _arun(
        self,
        subtasks: Annotated[list[TaskDetails], MinLen(1), MaxLen(3)],
        summary: str,
        communication: str,
        run_mode: RunMode = RunMode.FOREGROUND,
        **kwargs: Any,
    ) -> str:
        outer_config = ensure_config()

        if run_mode == RunMode.BACKGROUND:
            scheduled: list[tuple[UUID, str]] = []
            for i, subtask in enumerate(subtasks):

                def _make_work(
                    i: int = i, subtask: "TaskTool.TaskDetails" = subtask
                ) -> Callable[[UUID], Awaitable[str]]:
                    async def _work(tid: UUID) -> str:
                        return await self._arun_one(i, subtask, tid, outer_config)

                    return _work

                task_id = await self.background_task_manager.submit_task(
                    _make_work(),
                    summary=subtask.summary,
                )
                scheduled.append((task_id, subtask.summary))
            lines = [f"Scheduled {len(scheduled)} background task(s):"]
            lines.extend(f"- task_id={tid} — {summary}" for tid, summary in scheduled)
            lines.append(
                "Use `check_task`/`list_tasks` to monitor, `stop_task` to stop any of them."
            )
            return "\n".join(lines)

        return await self._run_to_completion(subtasks, outer_config)


def create_task_tools(
    workspace: BaseWorkspace,
    datasci_config: OpenDataSciConfig,
    sandbox_factory: BaseSandboxFactory,
    background_task_manager: AgentTaskManagerBase,
    skill_store: BaseSkillStore,
) -> list[BaseTool]:
    """Return the ``task`` tool — task creation only.

    Each spawned worker receives its own isolated sandbox created through
    *sandbox_factory* so that teardown is guaranteed on completion or error.
    Worker lifecycle events are dispatched directly into the calling graph's
    event stream via :func:`langchain_core.callbacks.manager.adispatch_custom_event`
    under the name ``"task_event"``, eliminating the need for side-channel queues.

    Args:
        workspace:       Workspace the workers operate on.
        datasci_config:  LLM configuration forwarded to each worker.
        sandbox_factory: Factory used to create an isolated sandbox for each worker.
        background_task_manager:    Shared task manager used to submit and track background
                         (``run_mode="background"``) task runs. Callers should share
                         the same instance with :func:`create_task_management_tools`
                         so ``check_task``/``list_tasks``/``stop_task`` can see these tasks.
        skill_store:     Skill store shared across all spawned workers.
    """
    return [
        TaskTool(
            workspace=workspace,
            datasci_config=datasci_config,
            sandbox_factory=sandbox_factory,
            skill_store=skill_store,
            background_task_manager=background_task_manager,
        )
    ]


def _isoformat(timestamp: float | None) -> str | None:
    return datetime.fromtimestamp(timestamp).isoformat() if timestamp is not None else None


def _record_to_dict(record: WorkerTaskRecord) -> dict[str, Any]:
    data: dict[str, Any] = {
        "task_id": str(record.task_id),
        "summary": record.summary,
        "status": record.status.value,
        "created_at": _isoformat(record.created_at),
        "finished_at": _isoformat(record.finished_at),
        "progress": [
            {
                "done": report.progress_update.done,
                "ongoing": report.progress_update.ongoing,
                "blockers": report.progress_update.blockers,
                "eta_seconds": report.eta_seconds,
                "reported_at": _isoformat(report.reported_at),
            }
            for report in record.progress
        ],
    }
    if record.status == AgentTaskStatus.COMPLETED:
        data["result"] = record.result
    elif record.status == AgentTaskStatus.FAILED:
        data["error"] = record.error
    return data


class CheckTaskTool(OpenDataSciBaseTool):
    """Check on a single previously scheduled background task."""

    class CallArgs(BaseModel):
        task_id: UUID

    name: str = "check_task"
    description: str = """
Check the status of a background task previously scheduled via the `task` tool with
`run_mode="background"`. Returns the task's summary, status, timestamps, and any
progress reported by the worker, plus its result or error once it reaches a terminal state.

Args:
    task_id: The task ID returned when the background task was scheduled.
""".strip()

    args_schema: type[BaseModel] = CallArgs

    background_task_manager: AgentTaskManagerBase

    @override
    async def _arun(self, task_id: UUID, **kwargs: Any) -> str:
        record = await self.background_task_manager.get_task(task_id)
        if record is None:
            return f"No background task found with task_id={task_id}."
        return json.dumps(_record_to_dict(record), indent=2, default=str)


class ListTasksTool(OpenDataSciBaseTool):
    """List previously scheduled background tasks, filtered by status."""

    class CallArgs(BaseModel):
        status_in: set[AgentTaskStatus] = Field(default_factory=lambda: {AgentTaskStatus.RUNNING})

    name: str = "list_tasks"
    description: str = """
List background tasks previously scheduled via the `task` tool with `run_mode="background"`,
filtered by status. Returns a table of task_id, summary, status, and start time.

Args:
    status_in: Set of statuses to include (e.g. {"running"}, {"completed", "failed"}).
               Defaults to {"running"} — currently active background tasks.
""".strip()

    args_schema: type[BaseModel] = CallArgs

    background_task_manager: AgentTaskManagerBase

    @override
    async def _arun(self, status_in: set[AgentTaskStatus] | None = None, **kwargs: Any) -> str:
        status_in = status_in or {AgentTaskStatus.RUNNING}
        records = [r for r in await self.background_task_manager.list_tasks() if r.status in status_in]
        if not records:
            return "No background tasks match the given status filter."

        header = "| task_id | summary | status | started_at |"
        separator = "|---|---|---|---|"
        rows = [
            f"| {r.task_id} | {r.summary} | {r.status.value} | {_isoformat(r.created_at)} |"
            for r in records
        ]
        return "\n".join([header, separator, *rows])


class StopTaskTool(OpenDataSciBaseTool):
    """Stop a previously scheduled background task."""

    class CallArgs(BaseModel):
        task_id: UUID

    name: str = "stop_task"
    description: str = """
Stop a background task previously scheduled via the `task` tool with `run_mode="background"`.

Stopping is best-effort: a worker deep inside a tool call may take a moment to unwind, and its
result is discarded once stopped.

Args:
    task_id: The task ID returned when the background task was scheduled.
""".strip()

    args_schema: type[BaseModel] = CallArgs

    background_task_manager: AgentTaskManagerBase

    @override
    async def _arun(self, task_id: UUID, **kwargs: Any) -> str:
        cancelled = await self.background_task_manager.cancel_task(task_id)
        if not cancelled:
            return f"No background task found with task_id={task_id}."
        return f"Stop requested for task_id={task_id}."


def create_task_management_tools(background_task_manager: AgentTaskManagerBase) -> list[BaseTool]:
    """Return the ``check_task``, ``list_tasks``, and ``stop_task`` tools.

    *background_task_manager* must be the same instance passed to :func:`create_task_tools`
    so these tools can see the background tasks scheduled by the ``task`` tool.
    """
    return [
        CheckTaskTool(background_task_manager=background_task_manager),
        ListTasksTool(background_task_manager=background_task_manager),
        StopTaskTool(background_task_manager=background_task_manager),
    ]
