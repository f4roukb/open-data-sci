"""Component tests for OpenDataSci service paths not covered elsewhere.

Each test mocks the LLM and sandbox at the lowest level and exercises a single
service method end-to-end through a real ``Agent``:

* ``compact_chat_history`` (LLM-driven history compaction)
* ``rewind_turn`` (last-turn removal from conversation history)
* ``get_workspace_files`` edge cases (no session, missing path)
* ``load_file`` error-emission path
* ``RuntimeError`` guard on compact
"""


from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from opendatasci._tui.service import OpenDataSciTuiService


def _seed_messages(agent, messages):
    """Add messages directly to the agent's graph state for testing."""
    agent.graph.update_state(agent._graph_config, {"messages": messages})


def _get_messages(agent):
    """Read current messages from the agent's graph state."""
    return agent.graph.get_state(agent._graph_config).values.get("messages", [])


class TestCompactConversation:
    """Compact summarises older turns via ChatHistoryCompactor."""

    async def test_compact_with_empty_history_returns_placeholder(self, loaded_opendatasci_service):
        result = await loaded_opendatasci_service.compact_chat_history()
        assert "no conversation" in result.lower()

    async def test_compact_summarizes_via_llm(
        self, loaded_opendatasci_service, mock_llm
    ):
        agent = loaded_opendatasci_service._agent
        # Two complete turns are needed; cutoff=1 keeps the last turn verbatim.
        _seed_messages(agent, [
            HumanMessage(content="What are the trends?"),
            AIMessage(content="Upward."),
            HumanMessage(content="Confirm?"),
            AIMessage(content="Confirmed."),
        ])

        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="They chatted briefly."))

        summary = await loaded_opendatasci_service.compact_chat_history()

        assert summary == "They chatted briefly."
        # Only the kept turn (last verbatim) remains; no SystemMessage in graph state.
        remaining = _get_messages(agent)
        assert len(remaining) == 2
        assert isinstance(remaining[0], HumanMessage)
        # Rolling summaries are reset and the compacted summary becomes the preamble.
        state = agent.graph.get_state(agent._graph_config).values
        assert state.get("turn_summaries", []) == []
        assert state.get("session_preamble") == "They chatted briefly."


class TestRewindTurn:
    """rewind_turn delegates to the agent's async rewind_turn method."""

    async def test_rewind_invokes_agent_method(self, loaded_opendatasci_service):
        from unittest.mock import AsyncMock, patch

        with patch.object(
            loaded_opendatasci_service._agent, "rewind_turn", new_callable=AsyncMock
        ) as mock_rewind:
            await loaded_opendatasci_service.rewind_turn()
        mock_rewind.assert_awaited_once()


class TestGetWorkspaceFiles:
    """get_workspace_files handles missing-path and directory cases.

    These depend only on the service's workspace_path, so the service is built
    directly with stub agent/sandbox rather than through the heavy agent fixture.
    """

    def _service(self, workspace_path) -> OpenDataSciTuiService:
        return OpenDataSciTuiService(
            agent=MagicMock(),
            sandbox=MagicMock(),
            workspace_path=workspace_path,
        )

    def test_returns_empty_when_no_workspace_path(self):
        assert self._service(None).get_workspace_files() == []

    def test_returns_directory_listing(self, data_dir):
        files = self._service(data_dir).get_workspace_files()
        # data_dir fixture has two CSVs.
        assert "sales.csv" in files
        assert "costs.csv" in files

    async def test_handles_iterdir_exception_gracefully(
        self, loaded_opendatasci_service, monkeypatch
    ):
        # Force iterdir to blow up and assert we return [] rather than propagating.
        def _boom(self):
            raise OSError("denied")

        monkeypatch.setattr(Path, "iterdir", _boom)
        assert loaded_opendatasci_service.get_workspace_files() == []
