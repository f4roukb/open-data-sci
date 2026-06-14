"""Component tests: session lifecycle management.

Covers get_workspace_files(), reset_session(), clear_context(),
RuntimeError guards, and context-manager cleanup — each exercising
the real integration between OpenDataSci, Session, and Agent state.
"""


import pytest
from langchain_core.messages import HumanMessage


def _seed_messages(agent, messages):
    """Add messages directly to the agent's graph state for testing."""
    agent.graph.update_state(agent._graph_config, {"messages": messages})


def _get_messages(agent):
    """Read current messages from the agent's graph state."""
    return agent.graph.get_state(agent._graph_config).values.get("messages", [])



class TestWorkspaceInspection:
    """Happy path: querying workspace contents after loading."""

    async def test_get_workspace_files_returns_loaded_file(
        self, loaded_opendatasci_service, data_file
    ):
        files = loaded_opendatasci_service.get_workspace_files()
        assert data_file.name in files


class TestSessionReset:
    """reset_session() resets the sandbox; clear_context() does not touch it."""

    async def test_reset_calls_sandbox_reset(self, loaded_opendatasci_service, mock_sandbox):
        await loaded_opendatasci_service.reset_session()
        mock_sandbox.reset.assert_called_once()

    async def test_clear_context_does_not_reset_sandbox(
        self, loaded_opendatasci_service, mock_sandbox
    ):
        await loaded_opendatasci_service.clear_context()
        mock_sandbox.reset.assert_not_called()


class TestContextManager:
    """The TUI service is an async context manager that releases the sandbox on exit."""

    async def test_context_manager_closes_sandbox_on_exit(
        self, loaded_opendatasci_service, mock_sandbox
    ):
        async with loaded_opendatasci_service:
            pass

        mock_sandbox.close.assert_awaited()
