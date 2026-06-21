"""Unit tests for opendatasci.tools.factory."""


from unittest.mock import AsyncMock, MagicMock, patch

from opendatasci.sandbox.base import BaseSandboxFactory
from opendatasci.tools.factory import ToolName, create_agent_tools, create_worker_agent_tools
from opendatasci.workspace.local import LocalWorkspace

# ---------------------------------------------------------------------------
# ToolName enum
# ---------------------------------------------------------------------------


class TestToolName:
    def test_execute_python_value(self) -> None:
        assert ToolName.EXECUTE_PYTHON_CODE == "execute_python_code"

    def test_execute_cli_value(self) -> None:
        assert ToolName.EXECUTE_CLI == "execute_cli_command"

    def test_is_string_subclass(self) -> None:
        assert isinstance(ToolName.EXECUTE_PYTHON_CODE, str)

    def test_all_expected_names_present(self) -> None:
        expected = {
            "execute_python_code",
            "execute_cli_command",
            "load_skill",
            "enter_plan_mode",
            "exit_plan_mode",
            "enter_self_review_mode",
            "exit_self_review_mode",
            "spawn_workers",
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
        assert ToolName.SPAWN_WORKERS == "spawn_workers"
        assert "spawn_workers" == ToolName.SPAWN_WORKERS


# ---------------------------------------------------------------------------
# create_worker_agent_tools
# ---------------------------------------------------------------------------


def _make_workspace(has_workspace: bool = False) -> MagicMock:
    # Workspace tools are gated on isinstance(workspace, LocalWorkspace), so a
    # LocalWorkspace-spec'd mock opts in and a plain mock opts out.
    wb = MagicMock(spec=LocalWorkspace) if has_workspace else MagicMock()
    # _base_tools does Path(workspace.get_reference()), so a path-like is needed.
    wb.get_reference.return_value = "/tmp/workspace" if has_workspace else None
    return wb


def _make_sandbox() -> MagicMock:
    sb = MagicMock()
    sb.execute = AsyncMock()
    sb.execute_cli = AsyncMock()
    return sb


def _make_sandbox_factory() -> MagicMock:
    return MagicMock(spec=BaseSandboxFactory)


class TestCreateWorkerAgentTools:
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

    def test_excludes_spawn_workers(self) -> None:
        tools = create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())
        names = {t.name for t in tools}
        assert "spawn_workers" not in names

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

    def test_excludes_spawn_workers(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "spawn_workers" not in names

    def test_excludes_web_tools(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "web_search" not in names
        assert "fetch_url" not in names

    def test_excludes_planning_tools(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "enter_plan_mode" not in names
        assert "exit_plan_mode" not in names

    def test_excludes_self_review_tools(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "enter_self_review_mode" not in names
        assert "exit_self_review_mode" not in names

    def test_excludes_ask_user_mcq(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "ask_user_mcq" not in names

    def test_excludes_verify_python_code(self) -> None:
        names = {t.name for t in create_worker_agent_tools(_make_workspace(), None, sandbox=_make_sandbox())}
        assert "verify_python_code" not in names


# ---------------------------------------------------------------------------
# create_agent_tools
# ---------------------------------------------------------------------------


class TestCreateMainAgentTools:
    def test_includes_cli_tool_via_base(self) -> None:
        tools = create_agent_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "execute_cli_command" in names

    def test_includes_spawn_workers(self) -> None:
        tools = create_agent_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "spawn_workers" in names

    def test_includes_web_tools(self) -> None:
        tools = create_agent_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "web_search" in names
        assert "fetch_url" in names

    def test_includes_planning_tools_when_save_plan_provided(self) -> None:
        tools = create_agent_tools(
            _make_workspace(),
            _make_sandbox(),
            None,
            sandbox_factory=_make_sandbox_factory(),
            save_plan=lambda p: None,
        )
        names = {t.name for t in tools}
        assert "enter_plan_mode" in names
        assert "exit_plan_mode" in names

    def test_excludes_planning_tools_when_no_save_plan(self) -> None:
        tools = create_agent_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "enter_plan_mode" not in names
        assert "exit_plan_mode" not in names

    def test_includes_ask_user_mcq(self) -> None:
        tools = create_agent_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "ask_user_mcq" in names

    def test_excludes_mcp_tools_when_config_has_no_urls(self) -> None:
        config = MagicMock()
        config.mcp_servers = []
        config.extra_web_domains = []
        config.override_web_domains = None
        mock_tool = MagicMock()
        mock_tool.name = "verify_python_code"
        with patch(
            "opendatasci.tools.factory.create_code_verification_tools", return_value=mock_tool
        ):
            tools = create_agent_tools(_make_workspace(), _make_sandbox(), None, datasci_config=config, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "mcp" not in " ".join(names).lower()

    def test_includes_critic_tools_unconditionally(self) -> None:
        tools = create_agent_tools(_make_workspace(), _make_sandbox(), None, sandbox_factory=_make_sandbox_factory())
        names = {t.name for t in tools}
        assert "enter_self_review_mode" in names
        assert "exit_self_review_mode" in names

