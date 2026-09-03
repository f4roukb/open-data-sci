"""Shared fixtures for TUI unit tests."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from opendatasci._tui import theme as _theme
from opendatasci._tui.controller import CLIController, UIAdapter
from opendatasci.configs import OpenDataSciConfig


@pytest.fixture(autouse=True)
def _restore_active_theme():
    """theme.set_active() mutates module-level globals; keep tests isolated."""
    saved_active = dict(_theme.active)
    saved_name = _theme.active_name
    yield
    _theme.active.clear()
    _theme.active.update(saved_active)
    _theme.active_name = saved_name


def _make_message_handle() -> MagicMock:
    handle = MagicMock()
    handle.append = AsyncMock()
    handle.set_content = AsyncMock()
    handle.finish = AsyncMock()
    return handle


def _make_ephemeral_handle() -> MagicMock:
    handle = MagicMock()
    handle.dismiss = MagicMock()
    handle.set_done = MagicMock()
    handle.is_running = MagicMock(return_value=True)
    handle.mark_task_done = MagicMock()
    handle.mark_task_error = MagicMock()
    handle.update_task_activity = MagicMock()
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
    ui.add_task_block.return_value = _make_ephemeral_handle()
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
    svc.resume_with_input = MagicMock(return_value=_empty_aiter())
    svc.resume_with_approval = MagicMock(return_value=_empty_aiter())
    svc.is_user_input_required = MagicMock(return_value=False)
    svc.task_manager = MagicMock()
    svc.task_manager.has_task_updates = MagicMock(return_value=False)
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
