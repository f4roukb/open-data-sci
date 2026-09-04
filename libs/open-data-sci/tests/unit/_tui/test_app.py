"""Unit tests for opendatasci._tui.app."""

import importlib.metadata
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from opendatasci._tui.adapter import SubmitAction
from opendatasci._tui.app import OpenDataSciApp, _get_version, main
from opendatasci.configs import OpenDataSciConfig

# ---------------------------------------------------------------------------
# _get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_returns_installed_version(self) -> None:
        with patch.object(importlib.metadata, "version", return_value="1.2.3"):
            assert _get_version() == "1.2.3"

    def test_falls_back_to_hardcoded_when_package_missing(self) -> None:
        with patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            assert _get_version() == "0.2.0"


# ---------------------------------------------------------------------------
# main() — argument parsing
# ---------------------------------------------------------------------------


class TestMainArgparse:
    """Verify that main() correctly parses TUI arguments and passes them to OpenDataSciApp."""

    def _run_main(self, argv: list[str]) -> MagicMock:
        """Run main() with the given argv and return the OpenDataSciApp class mock."""
        app_instance = MagicMock()
        app_instance.run = MagicMock()
        app_cls = MagicMock(return_value=app_instance)

        with (
            patch("sys.argv", ["opendatasci"] + argv),
            patch("opendatasci._tui.app.OpenDataSciApp", app_cls),
            patch("dotenv.load_dotenv"),
        ):
            main()

        return app_cls

    def _agent_config(self, argv: list[str]) -> OpenDataSciConfig:
        """Return the OpenDataSciConfig passed to OpenDataSciApp for the given argv."""
        return self._run_main(argv).call_args[1]["datasci_config"]

    def test_positional_path_passed_to_app(self) -> None:
        app_cls = self._run_main(["data.csv"])
        assert app_cls.call_args[1]["workspace_path"] == "data.csv"

    def test_default_provider_is_anthropic(self) -> None:
        assert self._agent_config(["data.csv"]).provider == "anthropic"

    def test_default_model_anthropic_is_claude_sonnet(self) -> None:
        assert self._agent_config(["data.csv"]).model == "claude-sonnet-5"

    def test_region_flag_is_removed(self) -> None:
        """--region is no longer a valid flag; aws_region defaults via OpenDataSciConfig."""
        with pytest.raises(SystemExit):
            self._run_main(["data.csv", "--region", "us-east-2"])

    def test_secondary_provider_defaults_to_anthropic(self) -> None:
        assert self._agent_config(["data.csv"]).secondary_provider == "anthropic"

    def test_missing_selection_includes_all_four_fields_by_default(self) -> None:
        app_cls = self._run_main(["data.csv"])
        missing = app_cls.call_args[1]["missing_selection"]
        assert set(missing) == {"provider", "model", "secondary_provider", "secondary_model"}

    def test_missing_selection_excludes_fields_set_by_config_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("provider: openai\nmodel: gpt-4o\n")
        app_cls = self._run_main(["data.csv", "--config", str(yaml_file)])
        missing = app_cls.call_args[1]["missing_selection"]
        assert "provider" not in missing
        assert "model" not in missing
        assert "secondary_provider" in missing
        assert "secondary_model" in missing

    def test_session_id_is_passed_to_app(self) -> None:
        app_cls = self._run_main(["data.csv"])
        assert "session_id" in app_cls.call_args[1]

    def test_session_id_is_full_hex(self) -> None:
        session_id = self._run_main(["data.csv"]).call_args[1]["session_id"]
        # session_id is a full, untruncated uuid4().hex (32 hex chars).
        assert len(session_id) == 32
        assert all(c in "0123456789abcdef" for c in session_id)

    def test_session_id_is_unique_per_run(self) -> None:
        ids = {self._run_main(["data.csv"]).call_args[1]["session_id"] for _ in range(5)}
        assert len(ids) == 5

    def test_config_flag_loads_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text(
            "provider: openai\nmodel: gpt-4o\nsecondary_provider: anthropic\nsecondary_model: claude-haiku-4-5\n"
        )
        cfg = self._agent_config(["data.csv", "--config", str(yaml_file)])
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o"
        assert cfg.secondary_provider == "anthropic"
        assert cfg.secondary_model == "claude-haiku-4-5"



# ---------------------------------------------------------------------------
# OpenDataSciApp.on_submit — history wiring
# ---------------------------------------------------------------------------


def _make_app() -> tuple[OpenDataSciApp, MagicMock]:
    """Bare OpenDataSciApp with mocked query_one and controller."""
    app = OpenDataSciApp.__new__(OpenDataSciApp)
    mock_input = MagicMock()
    app.query_one = MagicMock(return_value=mock_input)
    app._controller = MagicMock()
    return app, mock_input


# ---------------------------------------------------------------------------
# OpenDataSciApp.refresh_theme — live /theme <name> switching
# ---------------------------------------------------------------------------


class TestRefreshTheme:
    def test_refresh_theme_calls_refresh_css(self) -> None:
        app, _ = _make_app()
        app.refresh_css = MagicMock()

        app.refresh_theme()

        app.refresh_css.assert_called_once()


class TestOnSubmitHistory:
    async def test_push_history_called_for_non_empty_submission(self) -> None:
        app, mock_input = _make_app()
        app._controller.on_submit = AsyncMock(return_value=(SubmitAction.NONE, ""))
        event = MagicMock()
        event.value = "  analyse the data  "

        await app.on_submit(event)

        mock_input.push_history.assert_called_once_with("analyse the data")

    async def test_push_history_not_called_for_whitespace_only(self) -> None:
        app, mock_input = _make_app()
        app._controller.on_submit = AsyncMock(return_value=(SubmitAction.NONE, ""))
        event = MagicMock()
        event.value = "   "

        await app.on_submit(event)

        mock_input.push_history.assert_not_called()


# ---------------------------------------------------------------------------
# OpenDataSciApp.on_input_key — history wiring
# ---------------------------------------------------------------------------


class TestOnInputKeyHistory:
    def _app(self) -> tuple[OpenDataSciApp, MagicMock]:
        return _make_app()

    def test_up_navigates_history_when_no_completions(self) -> None:
        app, mock_input = self._app()
        app._controller.has_completion_matches = False
        mock_input.navigate_history.return_value = True
        event = MagicMock()
        event.key = "up"

        with patch.object(type(app), "focused", new_callable=PropertyMock, return_value=mock_input):
            app.on_input_key(event)

        mock_input.navigate_history.assert_called_once_with(-1)
        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        app._controller.suppress_next_input_change.assert_called_once()
        app._controller.cancel_input_change_suppression.assert_not_called()

    def test_down_navigates_history_when_no_completions(self) -> None:
        app, mock_input = self._app()
        app._controller.has_completion_matches = False
        mock_input.navigate_history.return_value = True
        event = MagicMock()
        event.key = "down"

        with patch.object(type(app), "focused", new_callable=PropertyMock, return_value=mock_input):
            app.on_input_key(event)

        mock_input.navigate_history.assert_called_once_with(1)
        event.stop.assert_called_once()

    def test_completion_takes_precedence_over_history(self) -> None:
        app, mock_input = self._app()
        app._controller.has_completion_matches = True
        app._controller.cycle_completion.return_value = True
        event = MagicMock()
        event.key = "up"

        with patch.object(type(app), "focused", new_callable=PropertyMock, return_value=mock_input):
            app.on_input_key(event)

        mock_input.navigate_history.assert_not_called()
        app._controller.cycle_completion.assert_called_once()

    def test_event_not_consumed_and_completing_reset_when_navigation_fails(self) -> None:
        app, mock_input = self._app()
        app._controller.has_completion_matches = False
        mock_input.navigate_history.return_value = False
        event = MagicMock()
        event.key = "up"

        with patch.object(type(app), "focused", new_callable=PropertyMock, return_value=mock_input):
            app.on_input_key(event)

        event.stop.assert_not_called()
        app._controller.suppress_next_input_change.assert_called_once()
        app._controller.cancel_input_change_suppression.assert_called_once()


# ---------------------------------------------------------------------------
# Ctrl+C / Esc during a running turn (2.1.4)
# ---------------------------------------------------------------------------


class TestCtrlCDuringTurn:
    """Ctrl+C stops the running turn first; quitting stays a double-press while idle."""

    def _quit_ready_app(self, agent_running: bool) -> OpenDataSciApp:
        app, _ = _make_app()
        app._controller.agent_running = agent_running
        app._controller.stop_agent = AsyncMock()
        app._quit_requested = False
        app._quit_timer = None
        app.exit = MagicMock()
        app.notify = MagicMock()
        app.set_timer = MagicMock()
        return app

    async def test_ctrl_c_stops_running_turn_instead_of_arming_quit(self) -> None:
        app = self._quit_ready_app(agent_running=True)

        await app.action_request_quit()

        app._controller.stop_agent.assert_awaited_once()
        assert app._quit_requested is False
        app.exit.assert_not_called()

    async def test_ctrl_c_while_idle_arms_quit(self) -> None:
        app = self._quit_ready_app(agent_running=False)

        await app.action_request_quit()

        app._controller.stop_agent.assert_not_awaited()
        assert app._quit_requested is True
        app.exit.assert_not_called()
        app.notify.assert_called_once()

    async def test_second_ctrl_c_while_idle_quits(self) -> None:
        app = self._quit_ready_app(agent_running=False)
        app._quit_requested = True

        await app.action_request_quit()

        app.exit.assert_called_once()


class TestEscDuringTurn:
    """A bare Esc during a turn stops the agent; UI dismissals take precedence."""

    def _esc_app(
        self,
        *,
        agent_running: bool = True,
        has_completion: bool = False,
        has_paste: bool = False,
        awaiting_choice: bool = False,
    ) -> OpenDataSciApp:
        app, _ = _make_app()
        controller = app._controller
        controller.agent_running = agent_running
        controller.has_completion_matches = has_completion
        controller.has_paste_attachment = has_paste
        controller.awaiting_choice = awaiting_choice
        controller.stop_agent = AsyncMock()
        app._run_agent = MagicMock()
        app._resume_with_input = MagicMock()
        return app

    async def test_bare_esc_stops_running_turn(self) -> None:
        app = self._esc_app()

        await app.action_focus_input()

        app._controller.stop_agent.assert_awaited_once()

    async def test_esc_does_not_stop_turn_when_idle(self) -> None:
        app = self._esc_app(agent_running=False)

        await app.action_focus_input()

        app._controller.stop_agent.assert_not_awaited()

    async def test_esc_dismisses_completion_without_stopping_turn(self) -> None:
        app = self._esc_app(has_completion=True)

        await app.action_focus_input()

        app._controller.hide_completion.assert_called_once()
        app._controller.stop_agent.assert_not_awaited()

    async def test_esc_discards_paste_without_stopping_turn(self) -> None:
        app = self._esc_app(has_paste=True)

        await app.action_focus_input()

        app._controller.clear_paste_attachment.assert_called_once()
        app._controller.stop_agent.assert_not_awaited()

    async def test_esc_cancels_choice_without_stopping_turn(self) -> None:
        app = self._esc_app(agent_running=False, awaiting_choice=True)
        app._controller.cancel_choice = AsyncMock(return_value="cancel")

        await app.action_focus_input()

        app._resume_with_input.assert_called_once_with("cancel")
        app._controller.stop_agent.assert_not_awaited()
