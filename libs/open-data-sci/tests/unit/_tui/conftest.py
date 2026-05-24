"""Shared fixtures for TUI unit tests."""


from unittest.mock import AsyncMock, MagicMock

import pytest

from opendatasci._tui.controller import CLIController, UIAdapter
from opendatasci.configs import OpenDataSciConfig


def _make_message_handle() -> MagicMock:
    handle = MagicMock()
    handle.append = MagicMock()
    handle.set_content = MagicMock()
    handle.finish = MagicMock()
    handle.finish_with_summary = MagicMock()
    return handle


def _make_ephemeral_handle() -> MagicMock:
    handle = MagicMock()
    handle.dismiss = MagicMock()
    handle.set_done = MagicMock()
    handle.is_running = MagicMock(return_value=True)
    handle.mark_worker_done = MagicMock()
    handle.mark_worker_error = MagicMock()
    handle.update_worker_activity = MagicMock()
    handle.set_communication = MagicMock()
    handle.upgrade = MagicMock()
    return handle


def _make_timer_handle() -> MagicMock:
    handle = MagicMock()
    handle.stop = MagicMock()
    handle.update_tokens = MagicMock()
    return handle


@pytest.fixture
def mock_ui() -> MagicMock:
    """A MagicMock UIAdapter with properly configured return values."""
    ui = MagicMock(spec=UIAdapter)
    ui.add_message.return_value = _make_message_handle()
    ui.add_turn_status_bar.return_value = _make_timer_handle()
    ui.add_ephemeral_block.return_value = _make_ephemeral_handle()
    ui.add_worker_block.return_value = _make_ephemeral_handle()
    ui.stop_agent = MagicMock()
    return ui


@pytest.fixture
def controller(mock_ui: MagicMock) -> CLIController:
    """An unloaded CLIController backed by a mock UI."""
    return CLIController(
        ui=mock_ui,
        workspace_path="/fake/data.csv",
        datasci_config=OpenDataSciConfig(provider="anthropic", model="claude-sonnet-4-6"),
        session_id="testsid0",
    )


@pytest.fixture
def mock_service() -> MagicMock:
    """A minimal mock for OpenDataSci."""
    svc = MagicMock()
    svc.close = AsyncMock()
    svc.reset_session = AsyncMock()
    svc.clear_context = AsyncMock()
    svc.compact_chat_history = AsyncMock(return_value="compact summary")
    svc.get_workspace_files = MagicMock(return_value=["data.csv", "output.csv"])
    svc.rewind_turn = AsyncMock()
    svc.astream = MagicMock(return_value=_empty_aiter())
    return svc


@pytest.fixture
def loaded_controller(controller: CLIController, mock_service: MagicMock) -> CLIController:
    """A CLIController with a mock service already attached."""
    controller._service = mock_service
    return controller


async def _empty_aiter():
    """An empty async iterator."""
    return
    yield  # make it a generator
