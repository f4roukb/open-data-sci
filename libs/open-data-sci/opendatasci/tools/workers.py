"""ConcurrentWorkerAgent spawning tool: task."""

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Any, Callable, Coroutine, Literal, override

from annotated_types import MaxLen, MinLen
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.base import BaseContextStore
from opendatasci.prompts.prompt_templates import WORKER_SYSTEM_PROMPT
from opendatasci.sandbox.base import BaseSandboxFactory
from opendatasci.skills import BaseSkillStore
from opendatasci.skills.local import LocalSkillStore
from opendatasci.tasks.base import BaseTaskManager
from opendatasci.tools.base import OpenDataSciBaseTool
from opendatasci.tools.coding import create_cli_tools, create_coding_tools
from opendatasci.tools.skills import create_skill_tools
from opendatasci.tools.web import create_web_tools
from opendatasci.workers import BaseWorker, ConcurrentWorker, ParallelWorker
from opendatasci.workspace.base import BaseWorkspace

logger = logging.getLogger(__name__)


class SpawnWorkersTool(OpenDataSciBaseTool):
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
        allow_web_tools: bool = False
        """When ``True``, the worker can use ``web_search`` and ``fetch_url``
        to look up documentation, papers, or API references."""

    class CallArgs(BaseModel):
        subtasks: Annotated[list["SpawnWorkersTool.TaskDetails"], MinLen(1), MaxLen(3)]
        summary: str
        communication: str
        synch_mode: Literal["sync", "async"] = "sync"

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
- Set ``synch_mode``: use ``"sync"`` (default) to wait for the result and get it back
  immediately. Prefer ``"async"`` for long-running subtasks (e.g. heavy training runs,
  large-scale data processing, anything that would otherwise stall the conversation) —
  the tool schedules the work in the background and returns immediately with a task ID,
  so you can keep helping the user instead of blocking on completion.

Args:
    subtasks:      1-3 subtask descriptors (see TaskDetails fields).
    communication: Brief message to the user about what you're doing
                   (e.g. "Running three checks in parallel.").
    synch_mode:    "sync" to wait for and return the result; "async" to schedule the
                   subtasks in the background and return a task ID immediately. Prefer
                   "async" for long-running work.
""".strip()

    args_schema: type[BaseModel] = CallArgs

    workspace: BaseWorkspace
    datasci_config: OpenDataSciConfig | None
    sandbox_factory: BaseSandboxFactory
    store: BaseSkillStore
    task_manager: BaseTaskManager
    run_mode: Literal["parallel", "concurrent"] = "concurrent"

    async def _arun_one(
        self,
        idx: int,
        subtask: TaskDetails,
        outer_config: RunnableConfig,
        schedule: "Callable[[Coroutine[Any, Any, None]], Any] | None" = None,
    ) -> str:
        """Run a single worker subtask inside its own sandbox.

        Args:
            idx:          Zero-based worker index used to tag emitted events.
            subtask:      Subtask descriptor including instructions and options.
            outer_config: LangChain config from the calling graph, captured before
                          any inner graph run can overwrite the context var — ensures
                          ``adispatch_custom_event`` always targets the right callback
                          chain regardless of which async context is active at fire time.
            schedule:     How to schedule the ``adispatch_custom_event`` coroutine.
                          ``None`` schedules it on the currently running loop (the
                          worker's own loop, whether that's the shared concurrent
                          loop or this worker's dedicated parallel-mode loop). When
                          running in parallel mode, this is instead
                          ``run_coroutine_threadsafe`` targeting the caller's loop,
                          since the callback machinery in *outer_config* lives there.
        """
        initial_skill = None
        if subtask.skill is not None:
            initial_skill = self.store.load(subtask.skill)
            if initial_skill is None:
                logger.warning(
                    idx,
                    subtask.skill,
                )

        def emit(event_type: str, content: str, metadata: dict[str, Any] | None = None) -> None:
            coro = adispatch_custom_event(
                "worker_event",
                {
                    "worker_idx": idx,
                    "event_type": event_type,
                    "content": content,
                    **(metadata or {}),
                },
                config=outer_config,
            )
            if schedule is not None:
                schedule(coro)
            else:
                asyncio.get_running_loop().create_task(coro)

        cancelled = False
        exc_info: BaseException | None = None
        datasci_config = self.datasci_config or OpenDataSciConfig()

        async with self.sandbox_factory.create(
            workspace_path=Path(self.workspace.get_reference())
        ) as worker_sandbox:
            tools: list[BaseTool] = [
                *create_coding_tools(worker_sandbox),
                *create_cli_tools(worker_sandbox),
                *create_skill_tools(self.store),
            ]
            if subtask.allow_web_tools:
                tools.extend(create_web_tools())
            from opendatasci.agents.agents import (
                ConcurrentWorkerAgent,
            )  # local import breaks circular dependency

            agent = ConcurrentWorkerAgent(tools=tools, config=datasci_config)
            emit("worker_started", subtask.summary)

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
                    emit("worker_finished", subtask.summary, {"success": success})
                    emit("worker_done", subtask.summary, {"success": success})

    def _make_worker(self) -> BaseWorker:
        return ParallelWorker() if self.run_mode == "parallel" else ConcurrentWorker()

    async def _run_to_completion(
        self,
        subtasks: list[TaskDetails],
        outer_config: RunnableConfig,
    ) -> str:
        timeout = (self.datasci_config or OpenDataSciConfig()).worker_timeout_seconds
        results = await asyncio.wait_for(
            self._make_worker().run(subtasks, outer_config, self._arun_one),
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
            sections.append(f"### ConcurrentWorkerAgent {i}: {subtask.subtask}\n\n{output}")
        return "\n\n---\n\n".join(sections)

    @override
    async def _arun(
        self,
        subtasks: Annotated[list[TaskDetails], MinLen(1), MaxLen(3)],
        summary: str,
        communication: str,
        synch_mode: Literal["sync", "async"] = "sync",
        **kwargs: Any,
    ) -> str:
        outer_config = ensure_config()

        if synch_mode == "async":
            task_id = await self.task_manager.submit(
                lambda: self._run_to_completion(subtasks, outer_config),
                summary=summary,
            )
            return (
                f"Task scheduled successfully in the background (task_id={task_id}). "
                "It is running asynchronously; no result is returned here. Use the "
                "get_task_status tool with this task_id to check on it, or cancel_task "
                "to stop it."
            )

        return await self._run_to_completion(subtasks, outer_config)


def create_worker_tools(
    workspace: BaseWorkspace,
    context: BaseContextStore | None,
    datasci_config: OpenDataSciConfig | None,
    sandbox_factory: BaseSandboxFactory,
    task_manager: BaseTaskManager,
    store: BaseSkillStore | None = None,
    run_mode: Literal["parallel", "concurrent"] = "concurrent",
) -> list[BaseTool]:
    """Return the task tool.

    Each spawned worker receives its own isolated sandbox created through
    *sandbox_factory* so that teardown is guaranteed on completion or error.
    Worker lifecycle events are dispatched directly into the calling graph's
    event stream via :func:`langchain_core.callbacks.manager.adispatch_custom_event`
    under the name ``"worker_event"``, eliminating the need for side-channel queues.

    Args:
        workspace:       Workspace the workers operate on.
        context:         Work context from the main agent; used to resolve the
                         skills directory.
        datasci_config:  LLM configuration forwarded to each worker.
        sandbox_factory: Factory used to create an isolated sandbox for each worker.
        task_manager:    Shared task manager used to submit and track background
                         (``synch_mode="async"``) task runs. Callers should share
                         the same instance with :func:`create_task_management_tools`
                         so ``get_task_status``/``cancel_task`` can see these tasks.
        store:           Skill store shared across all spawned workers.  Defaults
                         to a :class:`~opendatasci.skills.local.LocalSkillStore`
                         rooted at ``<context.root>/skills``.
        run_mode:        ``"concurrent"`` (default) runs subtasks cooperatively on a
                         single event loop, as before. ``"parallel"`` runs each
                         subtask on its own OS thread with a dedicated event loop
                         for true parallel execution.
    """
    if store is None:
        user_skills_dir = Path(context.root) / "skills" if context is not None else None
        user_domains_dir = Path(context.root) / "skill_domains" if context is not None else None
        store = LocalSkillStore(
            [user_skills_dir] if user_skills_dir is not None else None,
            [user_domains_dir] if user_domains_dir is not None else None,
        )

    return [
        SpawnWorkersTool(
            workspace=workspace,
            datasci_config=datasci_config,
            sandbox_factory=sandbox_factory,
            store=store,
            task_manager=task_manager,
            run_mode=run_mode,
        )
    ]
