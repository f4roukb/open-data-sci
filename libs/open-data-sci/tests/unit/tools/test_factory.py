"""Unit tests for opendatasci.tools.factory."""


from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.context.base import BaseContextStore
from opendatasci.human_inputs.human_approval import HumanApprovalBaseManager
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory
from opendatasci.skills.base import BaseSkillStore
from opendatasci.tools.factory import (
    ToolName,
    create_execution_mode_tools,
    create_plan_mode_tools,
    create_self_review_mode_tools,
    create_worker_agent_tools,
)
from opendatasci.workspace.base import BaseWorkspace
from opendatasci.workspace.local import LocalWorkspace

# ---------------------------------------------------------------------------
# ToolName enum
# ---------------------------------------------------------------------------


class TestToolName:
    def test_execute_python_value(self) -> None:
        assert ToolName.EXECUTE_PYTHON_CODE == "execute_python_code"

    def test_execute_cli_value(self) -> None:
        assert ToolName.EXECUTE_CLI_COMMAND == "execute_cli_command"

    def test_is_string_subclass(self) -> None:
        assert isinstance(ToolName.EXECUTE_PYTHON_CODE, str)

    def test_all_expected_names_present(self) -> None:
        expected = {
            "execute_python_code",
            "execute_cli_command",
            "load_skill",
            "list_skills",
            "switch_agentic_mode",
            "exit_plan_mode",
            "exit_self_review_mode",
            "task",
            "check_task",
            "list_tasks",
            "cancel_task",
            "read_dataset_info",
            "update_dataset_info",
            "profile_dataset",
            "list_workspace_files",
            "list_python_libs",
            "web_search",
            "fetch_url",
            "ask_user_mcq",
            "verify_python_code",
        }
        actual = {member.value for member in ToolName}
        assert expected == actual

    def test_equality_with_plain_string(self) -> None:
        assert ToolName.TASK == "task"
        assert "task" == ToolName.TASK


# ---------------------------------------------------------------------------
# create_worker_agent_tools
# ---------------------------------------------------------------------------


def _make_workspace(has_workspace: bool = False) -> MagicMock:
    # Workspace tools are gated on isinstance(workspace, LocalWorkspace), so a
    # LocalWorkspace-spec'd mock opts in and a plain mock opts out.
    wb = MagicMock(spec=LocalWorkspace) if has_workspace else MagicMock(spec=BaseWorkspace)
    # _base_tools does Path(workspace.get_reference()), so a path-like is needed.
    wb.get_reference.return_value = "/tmp/workspace" if has_workspace else None
    return wb


def _make_sandbox() -> MagicMock:
    sb = MagicMock(spec=BaseSandbox)
    sb.execute = AsyncMock()
    sb.execute_cli = AsyncMock()
    return sb


def _make_sandbox_factory() -> MagicMock:
    return MagicMock(spec=BaseSandboxFactory)


class TestCreateWorkerAgentTools:
    def test_cli_tool_has_no_request_approval_arg(self) -> None:
        # Worker graphs have no checkpointer, so approval interrupts are
        # impossible there; workers must get the plain CLI tool.
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        cli_tool = next(t for t in tools if t.name == ToolName.EXECUTE_CLI_COMMAND)
        assert "request_approval" not in cli_tool.args

    def test_returns_list_of_tools(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_includes_execute_python(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "execute_python_code" in names

    def test_includes_cli_tool(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "execute_cli_command" in names

    def test_includes_read_dataset_info(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "read_dataset_info" in names

    def test_excludes_update_dataset_info(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "update_dataset_info" not in names

    def test_excludes_task(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "task" not in names

    def test_excludes_web_tools(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "web_search" not in names
        assert "fetch_url" not in names

    def test_includes_workspace_tools_when_path_set(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(has_workspace=True), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "list_workspace_files" in names

    def test_excludes_workspace_tools_when_no_path(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(has_workspace=False), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "list_workspace_files" not in names


# ---------------------------------------------------------------------------
# Worker tool set — exhaustive
# ---------------------------------------------------------------------------


class TestWorkerToolSetExact:
    """Verify the exact set of tools available to workers — no more, no less."""

    _BASE: frozenset[str] = frozenset(
        {
            "execute_python_code",
            "list_python_libs",
            "execute_cli_command",
            "read_dataset_info",
            "profile_dataset",
            "load_skill",
            "list_skills",
        }
    )

    def test_exact_set_without_workspace(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(has_workspace=False), None, sandbox=_make_sandbox())}
        assert names == self._BASE

    def test_exact_set_with_workspace(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(has_workspace=True), None, sandbox=_make_sandbox())}
        assert names == self._BASE | {"list_workspace_files"}

    def test_excludes_update_dataset_info(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "update_dataset_info" not in names

    def test_excludes_task(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "task" not in names

    def test_excludes_web_tools(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "web_search" not in names
        assert "fetch_url" not in names

    def test_excludes_mode_switching_tool(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "switch_agentic_mode" not in names

    def test_excludes_plan_and_self_review_exit_tools(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "exit_plan_mode" not in names
        assert "exit_self_review_mode" not in names

    def test_excludes_ask_user_mcq(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "ask_user_mcq" not in names

    def test_excludes_verify_python_code(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "verify_python_code" not in names


# ---------------------------------------------------------------------------
# create_execution_mode_tools
# ---------------------------------------------------------------------------


class TestCreateMainAgentTools:
    @pytest.fixture(autouse=True)
    def _patch_llm_deps(self):
        with (
            patch(
                "opendatasci.tools.factory.HumanApprovalManager",
                return_value=MagicMock(spec=HumanApprovalBaseManager),
            ),
            patch("opendatasci.tools.factory.create_code_verification_tools", return_value=[]),
        ):
            yield

    def test_includes_cli_tool_via_base(self) -> None:
        tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "execute_cli_command" in names

    def test_includes_task(self) -> None:
        tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "task" in names

    def test_includes_task_tools(self) -> None:
        tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "check_task" in names
        assert "list_tasks" in names
        assert "cancel_task" in names

    def test_includes_web_tools(self) -> None:
        tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "fetch_url" in names

    def test_includes_exit_plan_mode_when_context_and_session_id_provided(self) -> None:
        tools = create_execution_mode_tools(
            _make_workspace(),
            _make_sandbox(),
            MagicMock(spec=BaseContextStore),
            sandbox_factory=_make_sandbox_factory(),
            session_id="sess1",
            skill_store=MagicMock(spec=BaseSkillStore),
        )
        names = {t.name for t in tools}
        assert "switch_agentic_mode" in names
        assert "exit_plan_mode" in names

    def test_excludes_exit_plan_mode_when_no_context(self) -> None:
        tools = create_execution_mode_tools(
            _make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory(), session_id="sess1"
        )
        names = {t.name for t in tools}
        assert "switch_agentic_mode" in names
        assert "exit_plan_mode" not in names

    def test_excludes_exit_plan_mode_when_no_session_id(self) -> None:
        tools = create_execution_mode_tools(
            _make_workspace(),
            _make_sandbox(),
            MagicMock(spec=BaseContextStore),
            sandbox_factory=_make_sandbox_factory(),
            skill_store=MagicMock(spec=BaseSkillStore),
        )
        names = {t.name for t in tools}
        assert "switch_agentic_mode" in names
        assert "exit_plan_mode" not in names

    def test_includes_ask_user_mcq(self) -> None:
        tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "ask_user_mcq" in names

    def test_excludes_mcp_tools_when_config_has_no_urls(self) -> None:
        config = OpenDataSciConfig()
        mock_tool = MagicMock()
        mock_tool.name = "verify_python_code"
        with (
            patch(
                "opendatasci.tools.factory.create_code_verification_tools", return_value=mock_tool
            ),
            patch(
                "opendatasci.tools.factory.HumanApprovalManager",
                return_value=MagicMock(spec=HumanApprovalBaseManager),
            ),
        ):
            tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, datasci_config=config, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "mcp" not in " ".join(names).lower()

    def test_cli_tool_has_request_approval_arg(self) -> None:
        config = OpenDataSciConfig()
        with patch(
            "opendatasci.tools.factory.HumanApprovalManager",
            return_value=MagicMock(spec=HumanApprovalBaseManager),
        ) as mock_manager_cls:
            tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, datasci_config=config, sandbox_factory=_make_sandbox_factory())
        mock_manager_cls.assert_called_once_with(config)
        cli_tool = next(t for t in tools if t.name == ToolName.EXECUTE_CLI_COMMAND)
        assert "request_approval" in cli_tool.args

    def test_includes_mode_tools_unconditionally(self) -> None:
        tools = create_execution_mode_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "switch_agentic_mode" in names
        assert "exit_self_review_mode" in names


# ---------------------------------------------------------------------------
# create_plan_mode_tools / create_self_review_mode_tools
# ---------------------------------------------------------------------------


def _fake_tools(*names: str) -> list[MagicMock]:
    tools = []
    for name in names:
        tool = MagicMock()
        tool.name = name
        tools.append(tool)
    return tools


_FULL_TOOL_SET = _fake_tools(
    "execute_python_code",
    "task",
    "switch_agentic_mode",
    "exit_plan_mode",
    "exit_self_review_mode",
)


class TestCreatePlanModeTools:
    def test_keeps_execution_and_exit_plan_mode_tools(self) -> None:
        names = {t.name for t in create_plan_mode_tools(_FULL_TOOL_SET)}
        assert names == {"execute_python_code", "exit_plan_mode"}

    def test_excludes_task(self) -> None:
        names = {t.name for t in create_plan_mode_tools(_FULL_TOOL_SET)}
        assert "task" not in names

    def test_excludes_switch_agentic_mode(self) -> None:
        names = {t.name for t in create_plan_mode_tools(_FULL_TOOL_SET)}
        assert "switch_agentic_mode" not in names

    def test_excludes_exit_self_review_mode(self) -> None:
        names = {t.name for t in create_plan_mode_tools(_FULL_TOOL_SET)}
        assert "exit_self_review_mode" not in names

    def test_absent_exit_plan_mode_stays_absent(self) -> None:
        """If plan mode was never wired up (no context store), there is nothing to add back in."""
        tools = _fake_tools("execute_python_code", "task", "switch_agentic_mode")
        names = {t.name for t in create_plan_mode_tools(tools)}
        assert "exit_plan_mode" not in names


class TestCreateSelfReviewModeTools:
    def test_keeps_execution_and_exit_self_review_mode_tools(self) -> None:
        names = {t.name for t in create_self_review_mode_tools(_FULL_TOOL_SET)}
        assert names == {"execute_python_code", "exit_self_review_mode"}

    def test_excludes_task(self) -> None:
        names = {t.name for t in create_self_review_mode_tools(_FULL_TOOL_SET)}
        assert "task" not in names

    def test_excludes_switch_agentic_mode(self) -> None:
        names = {t.name for t in create_self_review_mode_tools(_FULL_TOOL_SET)}
        assert "switch_agentic_mode" not in names

    def test_excludes_exit_plan_mode(self) -> None:
        names = {t.name for t in create_self_review_mode_tools(_FULL_TOOL_SET)}
        assert "exit_plan_mode" not in names

