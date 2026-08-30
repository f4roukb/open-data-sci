"""Tool factories: assemble the right tool sets for main and worker agents."""

from enum import StrEnum, auto
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
from opendatasci.tasks.base import AgentTaskManagerBase
from opendatasci.tasks.local import LocalAgentTaskManager
from opendatasci.tools.coding import (
    create_cli_tools,
    create_code_verification_tools,
    create_coding_tools,
)
from opendatasci.tools.dataset_info import create_data_context_tools
from opendatasci.tools.mcp import create_mcp_tools
from opendatasci.tools.modes import create_mode_tools
from opendatasci.tools.skills import create_skill_tools
from opendatasci.tools.tasks import create_task_management_tools, create_task_tools
from opendatasci.tools.user_interaction import create_user_interaction_tools
from opendatasci.tools.web import create_web_tools
from opendatasci.tools.workspace import create_workspace_tools
from opendatasci.workspace.base import BaseWorkspace
from opendatasci.workspace.local import LocalWorkspace


class ToolName(StrEnum):
    """Canonical names for all agent tools."""

    EXECUTE_PYTHON_CODE = auto()
    EXECUTE_CLI_COMMAND = auto()
    LIST_PYTHON_LIBS = auto()
    LOAD_SKILL = auto()
    LIST_SKILLS = auto()
    SWITCH_AGENTIC_MODE = auto()
    EXIT_PLAN_MODE = auto()
    EXIT_SELF_REVIEW_MODE = auto()
    TASK = auto()
    CHECK_TASK = auto()
    LIST_TASKS = auto()
    STOP_TASK = auto()
    REPORT_PROGRESS = auto()
    READ_DATASET_INFO = auto()
    UPDATE_DATASET_INFO = auto()
    PROFILE_DATASET = auto()
    LIST_WORKSPACE_FILES = auto()
    WEB_SEARCH = auto()
    FETCH_URL = auto()
    ASK_USER_MCQ = auto()
    VERIFY_PYTHON_CODE = auto()


def _base_tools(
    workspace: BaseWorkspace,
    sandbox: BaseSandbox,
    context_store: BaseContextStore | None,
    skill_store: BaseSkillStore,
    persist: bool = True,
    approval_manager: HumanApprovalBaseManager | None = None,
) -> list[BaseTool]:
    tools: list[BaseTool] = [
        *create_coding_tools(sandbox),
        *create_cli_tools(sandbox, approval_manager=approval_manager),
        *create_data_context_tools(context_store, sandbox, persist=persist),
        *create_skill_tools(skill_store),
    ]
    if isinstance(workspace, LocalWorkspace):
        tools.extend(create_workspace_tools(Path(workspace.get_reference())))
    return tools


def create_worker_agent_tools(
    workspace: BaseWorkspace,
    context_store: BaseContextStore | None,
    sandbox: BaseSandbox | None = None,
    skill_store: BaseSkillStore | None = None,
) -> list[BaseTool]:
    """Return the tool list for a worker agent.

    Workers share the same core tools as the main agent but cannot spawn
    further sub-workers, plan, or access the web.
    """
    if sandbox is None:
        from opendatasci.sandbox.srt import SRTSandbox

        sandbox = SRTSandbox(workspace_path=Path(workspace.get_reference()))
    if skill_store is None:
        user_skills_dir = Path(context_store.root) / "skills" if context_store is not None else None
        user_domains_dir = (
            Path(context_store.root) / "skill_domains" if context_store is not None else None
        )
        skill_store = LocalSkillStore(
            [user_skills_dir] if user_skills_dir is not None else None,
            [user_domains_dir] if user_domains_dir is not None else None,
        )
    return _base_tools(workspace, sandbox, context_store, skill_store, persist=False)


def create_execution_mode_tools(
    workspace: BaseWorkspace,
    sandbox: BaseSandbox,
    context_store: BaseContextStore | None,
    sandbox_factory: BaseSandboxFactory,
    session_id: str | None = None,
    skill_store: BaseSkillStore | None = None,
    datasci_config: OpenDataSciConfig | None = None,
    background_task_manager: AgentTaskManagerBase | None = None,
) -> list[BaseTool]:
    """Return the main agent's full tool set — the default, execution-mode list.

    Extends the worker tool set with mode-switching, worker spawning, background
    task management, web access, and user interaction. This is also the superset
    every other main-agent tool list is derived from: pass the result to
    ``create_plan_mode_tools`` / ``create_self_review_mode_tools`` to get the
    subset the LLM should see once it has switched into that mode, and keep
    this full list bound to the graph's tool-executing node so it can still
    run whichever tool the model actually called (e.g. ``exit_plan_mode``,
    which never appears in the execution-mode list itself).

    *skill_store*, *datasci_config*, and *background_task_manager* are provided by
    the caller when it already has an instance to share (e.g. across agent
    turns); otherwise a default is created here. Every tool downstream of this
    factory receives its instance as a required argument — this is the only
    layer that tolerates ``None``.
    """
    datasci_config = datasci_config or OpenDataSciConfig()
    if skill_store is None:
        user_skills_dir = Path(context_store.root) / "skills" if context_store is not None else None
        user_domains_dir = (
            Path(context_store.root) / "skill_domains" if context_store is not None else None
        )
        skill_store = LocalSkillStore(
            [user_skills_dir] if user_skills_dir is not None else None,
            [user_domains_dir] if user_domains_dir is not None else None,
        )
    if background_task_manager is None:
        output_root = (
            Path(context_store.root) / "workers" / "outputs" if context_store is not None else None
        )
        background_task_manager = LocalAgentTaskManager(output_root=output_root)
    # A single manager instance is shared by every tool that supports human approval.
    approval_manager: HumanApprovalBaseManager = HumanApprovalManager(datasci_config)
    tools = _base_tools(
        workspace, sandbox, context_store, skill_store, approval_manager=approval_manager
    )
    tools.extend(create_code_verification_tools(datasci_config))
    tools.extend(create_mode_tools(skill_store, context_store, session_id))
    tools.extend(
        create_task_tools(
            workspace,
            datasci_config,
            skill_store=skill_store,
            sandbox_factory=sandbox_factory,
            background_task_manager=background_task_manager,
        )
    )
    tools.extend(create_task_management_tools(background_task_manager))
    tools.extend(create_web_tools())
    tools.extend(create_user_interaction_tools())
    if datasci_config.mcp_servers:
        tools.extend(create_mcp_tools(datasci_config.mcp_servers))
    return tools


def create_plan_mode_tools(execution_tools: list[BaseTool]) -> list[BaseTool]:
    """Return the tool subset the LLM should see while in Plan Mode.

    Derived from *execution_tools* (see ``create_execution_mode_tools``):
    drops ``task`` (no delegating out of plan mode) and
    ``switch_agentic_mode``/``exit_self_review_mode`` (only ``exit_plan_mode``
    is a legal way out of this mode).
    """
    excluded = {
        ToolName.TASK,
        ToolName.SWITCH_AGENTIC_MODE,
        ToolName.EXIT_SELF_REVIEW_MODE,
    }
    return [tool for tool in execution_tools if tool.name not in excluded]


def create_self_review_mode_tools(execution_tools: list[BaseTool]) -> list[BaseTool]:
    """Return the tool subset the LLM should see while in Self-Review Mode.

    Derived from *execution_tools* (see ``create_execution_mode_tools``):
    drops ``task`` and ``switch_agentic_mode``/``exit_plan_mode``
    (only ``exit_self_review_mode`` is a legal way out of this mode).
    """
    excluded = {ToolName.TASK, ToolName.SWITCH_AGENTIC_MODE, ToolName.EXIT_PLAN_MODE}
    return [tool for tool in execution_tools if tool.name not in excluded]
