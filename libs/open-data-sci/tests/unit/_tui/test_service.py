"""Unit tests for opendatasci._tui.service.OpenDataSci."""


from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


from opendatasci.configs import OpenDataSciConfig
from opendatasci._tui.service import OpenDataSciTuiService
from opendatasci._tui.session import CLISessionInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.clear_chat_history = AsyncMock()
    agent.compact_chat_history = AsyncMock(return_value="(no conversation to compact)")
    agent.rewind_turn = AsyncMock()

    return agent


def _make_mock_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.close = AsyncMock()
    sandbox.reset = MagicMock()
    return sandbox


def _make_service(
    workspace_path: Path | None = None,
) -> tuple[OpenDataSciTuiService, MagicMock, MagicMock]:
    agent = _make_mock_agent()
    sandbox = _make_mock_sandbox()
    svc = OpenDataSciTuiService(agent=agent, sandbox=sandbox, workspace_path=workspace_path)
    return svc, agent, sandbox


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestOpenDataSciServiceInit:
    def test_agent_is_stored(self) -> None:
        svc, agent, _ = _make_service()
        assert svc._agent is agent

    def test_sandbox_is_stored(self) -> None:
        svc, _, sandbox = _make_service()
        assert svc._sandbox is sandbox

# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------


class TestOpenDataSciServiceContextManager:
    async def test_aenter_returns_self(self) -> None:
        svc, _, _ = _make_service()
        result = await svc.__aenter__()
        assert result is svc

    async def test_aexit_calls_close(self) -> None:
        svc, _, _ = _make_service()
        svc.close = AsyncMock()
        await svc.__aexit__(None, None, None)
        svc.close.assert_awaited_once()

    async def test_close_delegates_to_sandbox(self) -> None:
        svc, _, sandbox = _make_service()
        await svc.close()
        sandbox.close.assert_awaited_once()

    async def test_context_manager_closes_on_exit(self) -> None:
        svc, _, sandbox = _make_service()
        async with svc:
            pass
        sandbox.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# astream
# ---------------------------------------------------------------------------


class TestOpenDataSciServiceAstream:
    async def test_astream_delegates_to_agent(self) -> None:
        svc, agent, _ = _make_service()
        fake_event = MagicMock()

        async def _fake_astream(query: str):
            yield fake_event

        agent.astream = _fake_astream

        events = [e async for e in svc.astream("test query")]
        assert fake_event in events

    async def test_astream_passes_query_to_agent(self) -> None:
        svc, agent, _ = _make_service()
        received_queries: list[str] = []

        async def _fake_astream(query: str):
            received_queries.append(query)
            return
            yield  # make it a generator

        agent.astream = _fake_astream

        async for _ in svc.astream("my special query"):
            pass

        assert received_queries == ["my special query"]


# ---------------------------------------------------------------------------
# get_workspace_files
# ---------------------------------------------------------------------------


class TestOpenDataSciCLIServiceGetWorkspaceFiles:
    def test_returns_empty_list_when_no_workspace_path(self) -> None:
        svc, _, _ = _make_service(workspace_path=None)
        assert svc.get_workspace_files() == []

    def test_returns_file_names(self, tmp_path: Path) -> None:
        (tmp_path / "alpha.csv").touch()
        (tmp_path / "beta.xlsx").touch()
        svc, _, _ = _make_service(workspace_path=tmp_path)

        files = svc.get_workspace_files()

        assert "alpha.csv" in files
        assert "beta.xlsx" in files

    def test_directories_suffixed_with_slash(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        svc, _, _ = _make_service(workspace_path=tmp_path)

        files = svc.get_workspace_files()

        assert "subdir/" in files

    def test_files_sorted_dirs_last(self, tmp_path: Path) -> None:
        (tmp_path / "aaa").mkdir()
        (tmp_path / "file.csv").touch()
        svc, _, _ = _make_service(workspace_path=tmp_path)

        files = svc.get_workspace_files()

        assert files.index("file.csv") < files.index("aaa/")

    def test_returns_empty_list_on_iterdir_error(self, tmp_path: Path) -> None:
        mock_path = MagicMock(spec=Path)
        mock_path.iterdir = MagicMock(side_effect=PermissionError("denied"))
        svc, _, _ = _make_service(workspace_path=mock_path)

        assert svc.get_workspace_files() == []


# ---------------------------------------------------------------------------
# reset_session
# ---------------------------------------------------------------------------


class TestOpenDataSciServiceResetSession:
    async def test_reset_session_calls_sandbox_reset(self) -> None:
        svc, _, sandbox = _make_service()
        await svc.reset_session()
        sandbox.reset.assert_called_once()

    async def test_reset_session_calls_agent_clear_chat_history(self) -> None:
        svc, agent, _ = _make_service()
        await svc.reset_session()
        agent.clear_chat_history.assert_awaited_once()


# ---------------------------------------------------------------------------
# clear_context
# ---------------------------------------------------------------------------


class TestOpenDataSciServiceClearContext:
    async def test_clear_context_calls_clear_chat_history(self) -> None:
        svc, agent, _ = _make_service()
        await svc.clear_context()
        agent.clear_chat_history.assert_awaited_once()


# ---------------------------------------------------------------------------
# rewind_turn
# ---------------------------------------------------------------------------


class TestOpenDataSciServiceRewindTurn:
    async def test_rewind_turn_delegates_to_agent(self) -> None:
        svc, agent, _ = _make_service()
        await svc.rewind_turn()
        agent.rewind_turn.assert_awaited_once()


# ---------------------------------------------------------------------------
# compact_chat_history
# ---------------------------------------------------------------------------


class TestOpenDataSciServiceCompactConversation:
    async def test_compact_chat_history_delegates_to_agent(self) -> None:
        svc, agent, _ = _make_service()
        agent.compact_chat_history = AsyncMock(return_value="agent summary")
        result = await svc.compact_chat_history()
        agent.compact_chat_history.assert_awaited_once()
        assert result == "agent summary"


# ---------------------------------------------------------------------------
# CLISessionInfo.from_path
# ---------------------------------------------------------------------------


class TestBuildCLISessionInfo:
    def test_file_path_produces_single_workspace(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.touch()

        cfg = OpenDataSciConfig()
        info = CLISessionInfo.from_path(str(data_file), None, cfg)

        assert info.workspace_count == 1
        assert info.workspaces[0]["name"] == "data.csv"

    def test_file_path_is_not_directory(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.touch()

        info = CLISessionInfo.from_path(str(data_file), None, OpenDataSciConfig())

        assert info.is_directory is False

    def test_directory_path_is_directory(self, tmp_path: Path) -> None:
        info = CLISessionInfo.from_path(str(tmp_path), tmp_path, OpenDataSciConfig())

        assert info.is_directory is True

    def test_directory_counts_supported_data_files(self, tmp_path: Path) -> None:
        (tmp_path / "data1.csv").touch()
        (tmp_path / "data2.xlsx").touch()
        (tmp_path / "image.png").touch()  # not a supported data extension

        info = CLISessionInfo.from_path(str(tmp_path), tmp_path, OpenDataSciConfig())

        assert info.workspace_count == 2

    def test_directory_excludes_hidden_files(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.csv").touch()
        (tmp_path / "visible.csv").touch()

        info = CLISessionInfo.from_path(str(tmp_path), tmp_path, OpenDataSciConfig())

        assert info.workspace_count == 1

    def test_session_info_carries_provider(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.touch()

        info = CLISessionInfo.from_path(str(data_file), None, OpenDataSciConfig(provider="openai"))

        assert info.provider == "openai"

    def test_session_info_carries_model(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.touch()

        info = CLISessionInfo.from_path(
            str(data_file),
            None,
            OpenDataSciConfig(provider="openai", model="gpt-4o"),
        )

        assert info.model == "gpt-4o"

    def test_session_info_path_matches_input(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.touch()

        info = CLISessionInfo.from_path(str(data_file), None, OpenDataSciConfig())

        assert info.path == str(data_file)

    def test_returns_session_info_instance(self, tmp_path: Path) -> None:
        data_file = tmp_path / "data.csv"
        data_file.touch()

        info = CLISessionInfo.from_path(str(data_file), None, OpenDataSciConfig())

        assert isinstance(info, CLISessionInfo)
