"""ParallelWorkerAgent spawning tool: spawn_workers."""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from annotated_types import MaxLen, MinLen
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables.config import RunnableConfig, ensure_config
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from opendatasci.context.base import BaseContextStore
from opendatasci.prompts.prompt_templates import WORKER_SYSTEM_PROMPT
from opendatasci.sandbox.base import BaseSandboxFactory
from opendatasci.skills import BaseSkillStore
from opendatasci.skills.local import LocalSkillStore
from opendatasci.tools.coding import create_cli_tools, create_coding_tools
from opendatasci.tools.skills import create_skill_tools
from opendatasci.tools.web import create_web_tools
from opendatasci.workspace.base import BaseWorkspace

if TYPE_CHECKING:
    from opendatasci.configs import OpenDataSciConfig

logger = logging.getLogger(__name__)


async def _run_one(
    idx: int,
    subtask: "WorkerTask",
    outer_config: RunnableConfig,
    *,
    sandbox_factory: BaseSandboxFactory,
    workspace: BaseWorkspace,
    store: BaseSkillStore,
    datasci_config: "OpenDataSciConfig | None",
) -> str:
    """Run a single worker subtask inside its own sandbox.

    Args:
        idx:             Zero-based worker index used to tag emitted events.
        subtask:         Subtask descriptor including instructions and options.
        outer_config:    LangChain config from the calling graph, captured before
                         any inner graph run can overwrite the context var — ensures
                         ``adispatch_custom_event`` always targets the right callback
                         chain regardless of which async context is active at fire time.
        sandbox_factory: Factory used to create the worker's isolated sandbox.
        workspace:       Workspace the worker operates on.
        store:           Skill store to resolve ``subtask.skill``.
        datasci_config:  LLM configuration forwarded to the worker agent.
    """
    initial_skill = None
    if subtask.skill is not None:
        initial_skill = store.load(subtask.skill)
        if initial_skill is None:
            logger.warning(
                "Worker %d: requested skill %r is unknown; starting without a preloaded skill.",
                idx,
                subtask.skill,
            )

    def emit(event_type: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        asyncio.get_running_loop().create_task(
            adispatch_custom_event(
                "worker_event",
                {
                    "worker_idx": idx,
                    "event_type": event_type,
                    "content": content,
                    **(metadata or {}),
                },
                config=outer_config,
            )
        )

    cancelled = False
    exc_info: BaseException | None = None

    async with sandbox_factory.create(
        workspace_path=Path(workspace.get_reference())
    ) as worker_sandbox:
        tools: list[BaseTool] = [
            *create_coding_tools(worker_sandbox),
            *create_cli_tools(worker_sandbox),
            *create_skill_tools(store),
        ]
        if subtask.allow_web_tools:
            tools.extend(
                create_web_tools(
                    datasci_config.extra_web_domains if datasci_config is not None else (),
                    datasci_config.override_web_domains if datasci_config is not None else None,
                )
            )
        from opendatasci.agents.agents import (
            ParallelWorkerAgent,
        )  # local import breaks circular dependency

        agent = ParallelWorkerAgent(tools=tools, config=datasci_config)
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


class WorkerTask(BaseModel):
    """A subtask descriptor for a worker."""

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


def create_worker_tools(
    workspace: BaseWorkspace,
    context: "BaseContextStore | None",
    datasci_config: "OpenDataSciConfig | None",
    sandbox_factory: BaseSandboxFactory,
    store: BaseSkillStore | None = None,
) -> list[BaseTool]:
    """Return the spawn_workers tool.

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
        store:           Skill store shared across all spawned workers.  Defaults
                         to a :class:`~opendatasci.skills.local.LocalSkillStore`
                         rooted at ``<context.root>/skills``.
    """
    if store is None:
        user_skills_dir = Path(context.root) / "skills" if context is not None else None
        store = LocalSkillStore([user_skills_dir] if user_skills_dir is not None else None)

    @tool
    async def spawn_workers(
        subtasks: Annotated[list[WorkerTask], MinLen(1), MaxLen(3)],
        communication: str,
    ) -> str:
        """Spawn 1–3 independent workers to execute narrow, concrete subtasks in parallel.

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

        Args:
            subtasks:      1–3 subtask descriptors (see WorkerTask fields).
            communication: Brief message to the user about what you're doing
                           (e.g. "Running three checks in parallel.").
        """
        outer_config = ensure_config()
        timeout = datasci_config.worker_timeout_seconds if datasci_config is not None else 300.0
        results = await asyncio.wait_for(
            asyncio.gather(
                *[
                    _run_one(
                        i,
                        t,
                        outer_config,
                        sandbox_factory=sandbox_factory,
                        workspace=workspace,
                        store=store,
                        datasci_config=datasci_config,
                    )
                    for i, t in enumerate(subtasks)
                ],
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
            sections.append(f"### ParallelWorkerAgent {i}: {subtask.subtask}\n\n{output}")
        return "\n\n---\n\n".join(sections)

    return [spawn_workers]
