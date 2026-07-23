"""Tool factories: assemble the right tool sets for main and worker agents."""

from enum import Enum
from pathlib import Path

from langchain_core.tools import BaseTool

from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.base import BaseContextStore
from opendatasci.human_inputs.human_approval import (
    HumanApprovalBaseManager,
    HumanApprovalManager,
)
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory
from opendatasci.skills import BaseSkillStore
from opendatasci.skills.local import LocalSkillStore
from opendatasci.tools.coding import (
    create_cli_tools,
    create_code_verification_tools,
    create_coding_tools,
)
from opendatasci.tools.dataset_info import create_data_context_tools
from opendatasci.tools.mcp import create_mcp_tools
from opendatasci.tools.modes import create_mode_tools
from opendatasci.tools.skills import create_skill_tools
from opendatasci.tools.user_interaction import create_user_interaction_tools
from opendatasci.tools.web import create_web_tools
from opendatasci.tools.workers import create_worker_tools
from opendatasci.tools.workspace import create_workspace_tools
from opendatasci.workspace.base import BaseWorkspace
from opendatasci.workspace.local import LocalWorkspace


class ToolName(str, Enum):
    """Canonical names for all agent tools."""

    EXECUTE_PYTHON_CODE = "execute_python_code"
    EXECUTE_CLI = "execute_cli_command"
    LIST_PYTHON_LIBS = "list_python_libs"
    LOAD_SKILL = "load_skill"
    LIST_SKILLS = "list_skills"
    SWITCH_AGENTIC_MODE = "switch_agentic_mode"
    EXIT_PLAN_MODE = "exit_plan_mode"
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
    context: BaseContextStore | None,
    store: BaseSkillStore,
    persist: bool = True,
    approval_manager: HumanApprovalBaseManager | None = None,
) -> list[BaseTool]:
    tools: list[BaseTool] = [
        *create_coding_tools(sandbox),
        *create_cli_tools(sandbox, approval_manager=approval_manager),
        *create_data_context_tools(context, sandbox, persist=persist),
        *create_skill_tools(store),
    ]
    if isinstance(workspace, LocalWorkspace):
        tools.extend(create_workspace_tools(Path(workspace.get_reference())))
    return tools


def create_worker_agent_tools(
    workspace: BaseWorkspace,
    context: BaseContextStore | None,
    sandbox: BaseSandbox | None = None,
    store: BaseSkillStore | None = None,
) -> list[BaseTool]:
    """Return the tool list for a worker agent.

    Workers share the same core tools as the main agent but cannot spawn
    further sub-workers, plan, or access the web.
    """
    if sandbox is None:
        from opendatasci.sandbox.srt import SRTSandbox

        sandbox = SRTSandbox(workspace_path=Path(workspace.get_reference()))
    if store is None:
        user_skills_dir = Path(context.root) / "skills" if context is not None else None
        user_domains_dir = Path(context.root) / "skill_domains" if context is not None else None
        store = LocalSkillStore(
            [user_skills_dir] if user_skills_dir is not None else None,
            [user_domains_dir] if user_domains_dir is not None else None,
        )
    return _base_tools(workspace, sandbox, context, store, persist=False)


def create_execution_mode_tools(
    workspace: BaseWorkspace,
    sandbox: BaseSandbox,
    context: BaseContextStore | None,
    sandbox_factory: BaseSandboxFactory,
    session_id: str | None = None,
    store: BaseSkillStore | None = None,
    datasci_config: OpenDataSciConfig | None = None,
) -> list[BaseTool]:
    """Return the main agent's full tool set — the default, execution-mode list.

    Extends the worker tool set with mode-switching, worker spawning, web
    access, and user interaction. This is also the superset every other
    main-agent tool list is derived from: pass the result to
    ``create_plan_mode_tools`` / ``create_self_review_mode_tools`` to get the
    subset the LLM should see once it has switched into that mode, and keep
    this full list bound to the graph's tool-executing node so it can still
    run whichever tool the model actually called (e.g. ``exit_plan_mode``,
    which never appears in the execution-mode list itself).
    """
    datasci_config = datasci_config or OpenDataSciConfig()
    if store is None:
        user_skills_dir = Path(context.root) / "skills" if context is not None else None
        user_domains_dir = Path(context.root) / "skill_domains" if context is not None else None
        store = LocalSkillStore(
            [user_skills_dir] if user_skills_dir is not None else None,
            [user_domains_dir] if user_domains_dir is not None else None,
        )
    # A single manager instance is shared by every tool that supports human approval.
    approval_manager: HumanApprovalBaseManager = HumanApprovalManager(datasci_config)
    tools = _base_tools(workspace, sandbox, context, store, approval_manager=approval_manager)
    tools.extend(create_code_verification_tools(datasci_config))
    tools.extend(create_mode_tools(store, context, session_id))
    tools.extend(
        create_worker_tools(
            workspace,
            context,
            datasci_config,
            store=store,
            sandbox_factory=sandbox_factory,
        )
    )
    tools.extend(create_web_tools())
    tools.extend(create_user_interaction_tools())
    if datasci_config.mcp_servers:
        tools.extend(create_mcp_tools(datasci_config.mcp_servers))
    return tools


def create_plan_mode_tools(execution_tools: list[BaseTool]) -> list[BaseTool]:
    """Return the tool subset the LLM should see while in Plan Mode.

    Derived from *execution_tools* (see ``create_execution_mode_tools``):
    drops ``spawn_workers`` (no delegating out of plan mode) and
    ``switch_agentic_mode``/``exit_self_review_mode`` (only ``exit_plan_mode``
    is a legal way out of this mode).
    """
    excluded = {ToolName.SPAWN_WORKERS, ToolName.SWITCH_AGENTIC_MODE, ToolName.EXIT_SELF_REVIEW_MODE}
    return [tool for tool in execution_tools if tool.name not in excluded]


def create_self_review_mode_tools(execution_tools: list[BaseTool]) -> list[BaseTool]:
    """Return the tool subset the LLM should see while in Self-Review Mode.

    Derived from *execution_tools* (see ``create_execution_mode_tools``):
    drops ``spawn_workers`` and ``switch_agentic_mode``/``exit_plan_mode``
    (only ``exit_self_review_mode`` is a legal way out of this mode).
    """
    excluded = {ToolName.SPAWN_WORKERS, ToolName.SWITCH_AGENTIC_MODE, ToolName.EXIT_PLAN_MODE}
    return [tool for tool in execution_tools if tool.name not in excluded]
