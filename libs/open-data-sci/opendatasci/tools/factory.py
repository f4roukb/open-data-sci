"""Tool factories: assemble the right tool sets for main and worker agents."""

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool

from opendatasci.context.base import BaseContextStore
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory
from opendatasci.skills import BaseSkillStore
from opendatasci.skills.local import LocalSkillStore
from opendatasci.tools.coding import (
    create_cli_tools,
    create_code_verification_tools,
    create_coding_tools,
)
from opendatasci.tools.critic import create_critic_tools
from opendatasci.tools.dataset_info import create_data_context_tools
from opendatasci.tools.mcp import create_mcp_tools
from opendatasci.tools.planning import create_planning_tools
from opendatasci.tools.skills import create_skill_tools
from opendatasci.tools.user_interaction import create_user_interaction_tools
from opendatasci.tools.web import create_web_tools
from opendatasci.tools.workers import create_worker_tools
from opendatasci.tools.workspace import create_workspace_tools
from opendatasci.workspace.base import BaseWorkspace
from opendatasci.workspace.local import LocalWorkspace

if TYPE_CHECKING:
    from opendatasci.configs import OpenDataSciConfig


class ToolName(str, Enum):
    """Canonical names for all agent tools."""

    EXECUTE_PYTHON_CODE = "execute_python_code"
    EXECUTE_CLI = "execute_cli_command"
    LIST_PYTHON_LIBS = "list_python_libs"
    LOAD_SKILL = "load_skill"
    ENTER_PLAN_MODE = "enter_plan_mode"
    EXIT_PLAN_MODE = "exit_plan_mode"
    ENTER_SELF_REVIEW_MODE = "enter_self_review_mode"
    EXIT_SELF_REVIEW_MODE = "exit_self_review_mode"
    SPAWN_WORKERS = "spawn_workers"
    READ_DATASET_INFO = "read_dataset_info"
    UPDATE_DATASET_INFO = "update_dataset_info"
    PROFILE_DATASET = "profile_dataset"
    LIST_WORKSPACE_FILES = "list_workspace_files"
    WEB_SEARCH = "web_search"
    FETCH_URL = "fetch_url"
    ASK_USER_MCQ = "ask_user_mcq"
    VERIFY_PYTHON_CODE = "verify_python_code"


def _base_tools(
    workspace: BaseWorkspace,
    sandbox: BaseSandbox,
    context: "BaseContextStore | None",
    store: BaseSkillStore,
    persist: bool = True,
) -> list[BaseTool]:
    """Return the tools shared by both main and worker agents.

    Args:
        workspace:    Workspace container.
        sandbox:      Code execution sandbox.
        context: I/O boundary for dataset notes and profiles.
        store:        Skill store used by the ``load_skill`` tool.
        persist:      When ``False``, write-side tools (``update_dataset_info``)
                      are excluded and ``profile_dataset`` will not write profiles
                      to disk.
    """
    tools: list[BaseTool] = [
        *create_coding_tools(sandbox),
        *create_cli_tools(sandbox),
        *create_data_context_tools(context, sandbox, persist=persist),
        *create_skill_tools(store),
    ]
    if isinstance(workspace, LocalWorkspace):
        tools.extend(create_workspace_tools(Path(workspace.get_reference())))
    return tools


def create_worker_agent_tools(
    workspace: BaseWorkspace,
    context: "BaseContextStore | None",
    sandbox: BaseSandbox | None = None,
    store: BaseSkillStore | None = None,
) -> list[BaseTool]:
    """Return the tool list for a worker agent.

    Workers share the same core tools as the main agent but cannot spawn
    further workers, plan, or access the web.

    Args:
        workspace:    Workspace container.
        context: I/O boundary for dataset notes and profiles.
        sandbox:      Code execution sandbox.  A new :class:`~opendatasci.sandbox.srt.SRTSandbox`
                      is created when ``None``.
        store:        Skill store injected from the caller.  Defaults to a
                      :class:`~opendatasci.skills.local.LocalSkillStore` rooted
                      at ``<context.root>/skills``.
    """
    if sandbox is None:
        from opendatasci.sandbox.srt import SRTSandbox

        sandbox = SRTSandbox(workspace_path=Path(workspace.get_reference()))
    if store is None:
        user_skills_dir = Path(context.root) / "skills" if context is not None else None
        store = LocalSkillStore([user_skills_dir] if user_skills_dir is not None else None)
    return _base_tools(workspace, sandbox, context, store, persist=False)


def create_agent_tools(
    workspace: BaseWorkspace,
    sandbox: BaseSandbox,
    context: "BaseContextStore | None",
    sandbox_factory: BaseSandboxFactory,
    store: BaseSkillStore | None = None,
    datasci_config: "OpenDataSciConfig | None" = None,
    save_plan: "Callable[[str], None] | None" = None,
) -> list[BaseTool]:
    """Return the tool list for the main agent.

    Extends the worker tool set with planning, worker spawning, web access,
    and user interaction.

    Args:
        workspace:       Workspace container.
        sandbox:         Code execution sandbox.
        context:         I/O boundary for dataset notes and profiles.
        sandbox_factory: Factory used by spawned workers to create their own
                         isolated sandboxes.
        store:           Skill store injected from the caller.  Defaults to a
                         :class:`~opendatasci.skills.local.LocalSkillStore` rooted
                         at ``<context.root>/skills``.
        datasci_config:  LLM configuration forwarded to spawned workers.
        save_plan:       Callback that persists the final plan via
                         ``BaseContextStore``.  When provided,
                         ``enter_plan_mode`` and ``exit_plan_mode`` are added
                         to the tool list.
    """
    if store is None:
        user_skills_dir = Path(context.root) / "skills" if context is not None else None
        store = LocalSkillStore([user_skills_dir] if user_skills_dir is not None else None)
    tools = _base_tools(workspace, sandbox, context, store)
    if datasci_config is not None:
        tools.extend(create_code_verification_tools(datasci_config))
    if save_plan is not None:
        tools.extend(create_planning_tools(save_plan))
    tools.extend(create_critic_tools(store))
    tools.extend(
        create_worker_tools(
            workspace,
            context,
            datasci_config,
            store=store,
            sandbox_factory=sandbox_factory,
        )
    )
    tools.extend(
        create_web_tools(
            datasci_config.extra_web_domains if datasci_config else (),
            datasci_config.override_web_domains if datasci_config else None,
        )
    )
    tools.extend(create_user_interaction_tools())
    if datasci_config is not None and datasci_config.mcp_servers:
        tools.extend(create_mcp_tools(datasci_config.mcp_servers))
    return tools
