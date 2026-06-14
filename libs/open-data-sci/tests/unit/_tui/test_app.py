"""Unit tests for opendatasci._tui.app."""


import importlib.metadata
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

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
            assert _get_version() == "0.1.0"


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
        assert self._agent_config(["data.csv"]).model == "claude-sonnet-4-6"

    def test_default_model_openai_resolves_from_provider_default(self) -> None:
        cfg = self._agent_config(["data.csv", "--provider", "openai"])
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5.5"

    def test_explicit_model_overrides_default(self) -> None:
        assert (
            self._agent_config(["data.csv", "--model", "claude-opus-4-6"]).model
            == "claude-opus-4-6"
        )

    @pytest.mark.parametrize(
        "theme_name", ["default", "accessible", "light", "solarized", "dracula"]
    )
    def test_theme_flag_accepts_all_registered_palettes(self, theme_name: str) -> None:
        app_cls = self._run_main(["data.csv", "--theme", theme_name])
        assert app_cls.call_args[1]["theme"] == theme_name

    def test_api_key_stored_in_agent_config(self) -> None:
        assert (
            self._agent_config(["data.csv", "--api-key", "sk-test"]).anthropic_api_key == "sk-test"
        )

    def test_region_flag_is_removed(self) -> None:
        """--region is no longer a valid flag."""
        with pytest.raises(SystemExit):
            self._run_main(["data.csv", "--region", "us-east-2"])

    def test_secondary_model_flag(self) -> None:
        cfg = self._agent_config(["data.csv", "--secondary-model", "gpt-4o-mini"])
        assert cfg.secondary_model == "gpt-4o-mini"

    def test_secondary_provider_flag(self) -> None:
        cfg = self._agent_config(
            ["data.csv", "--secondary-provider", "openai", "--secondary-model", "gpt-4o-mini"]
        )
        assert cfg.secondary_provider == "openai"
        assert cfg.secondary_model == "gpt-4o-mini"

    def test_secondary_provider_defaults_to_primary_provider(self) -> None:
        assert self._agent_config(["data.csv"]).secondary_provider == "anthropic"

    def test_secondary_provider_resolves_to_main_when_not_set(self) -> None:
        cfg = self._agent_config(["data.csv", "--provider", "openai"])
        assert cfg.secondary_provider == "openai"

    def test_cross_provider_resolved_secondary_model(self) -> None:
        cfg = self._agent_config(
            [
                "data.csv",
                "--provider",
                "anthropic",
                "--secondary-provider",
                "openai",
                "--secondary-model",
                "gpt-4o-mini",
            ]
        )
        assert cfg.provider == "anthropic"
        assert cfg.secondary_provider == "openai"
        assert cfg.secondary_model == "gpt-4o-mini"

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

    def test_config_flag_cli_overrides_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cfg.yaml"
        yaml_file.write_text("provider: openai\nmodel: gpt-4o\n")
        cfg = self._agent_config(
            ["data.csv", "--config", str(yaml_file), "--model", "gpt-5.5"]
        )
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-5.5"


# ---------------------------------------------------------------------------
# OpenDataSciApp.on_submit — history wiring
# ---------------------------------------------------------------------------


def _make_app() -> tuple[OpenDataSciApp, MagicMock]:
    """Bare OpenDataSciApp with mocked query_one and controller."""
    app = OpenDataSciApp.__new__(OpenDataSciApp)
    mock_input = MagicMock()
    app.query_one = MagicMock(return_value=mock_input)
    app._controller = MagicMock()
    app._controller._completing = False
    return app, mock_input


class TestOnSubmitHistory:
    async def test_push_history_called_for_non_empty_submission(self) -> None:
        app, mock_input = _make_app()
        app._controller.on_submit = AsyncMock(return_value=("", ""))
        event = MagicMock()
        event.value = "  analyse the data  "

        await app.on_submit(event)

        mock_input.push_history.assert_called_once_with("analyse the data")

    async def test_push_history_not_called_for_whitespace_only(self) -> None:
        app, mock_input = _make_app()
        app._controller.on_submit = AsyncMock(return_value=("", ""))
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
        assert app._controller._completing is False
