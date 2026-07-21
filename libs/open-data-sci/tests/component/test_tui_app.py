"""Component tests: ``opendatasci._tui.app`` — the Textual app shell and CLI entry.

Two seams are exercised:

* ``OpenDataSciApp`` run headless with a stubbed ``CLIController``: composition,
  the UIAdapter methods the controller calls back into, input submission,
  approval decisions, quit/stop semantics, and history/completion key routing.
* ``main()``: argument parsing and ``OpenDataSciConfig`` assembly for the
  default, ``--config`` and ``--api-key`` paths, with the app itself replaced
  by a recording stub.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Input

import opendatasci._tui.app as app_module
from opendatasci._tui.adapter import SubmitAction
from opendatasci.configs import DEFAULT_MODEL, OpenDataSciConfig
from opendatasci.models.providers import Provider
from opendatasci._tui.app import OpenDataSciApp, _get_version, main
from opendatasci._tui.widgets import (
    AppHeader,
    ChatPane,
    CommandApprovalPrompt,
    CompletionPopup,
    MessageBubble,
    PendingMessageBubble,
    SmartInput,
    ThinkingBlock,
    ToolCallBlock,
    TurnStatusBar,
    WorkspacePanel,
)

# ---------------------------------------------------------------------------
# OpenDataSciApp with a stubbed controller
# ---------------------------------------------------------------------------


def _make_controller_stub(workspace_path: str) -> MagicMock:
    stub = MagicMock()
    stub.provider = "anthropic"
    stub.model = "claude-sonnet-4-6"
    stub._workspace_path = workspace_path
    stub.agent_running = False
    stub.awaiting_choice = False
    stub.has_completion_matches = False
    stub.has_paste_attachment = False
    stub.boot = AsyncMock()
    stub.close = AsyncMock()
    stub.stop_agent = AsyncMock()
    stub.reset = AsyncMock()
    stub.clear_conv = AsyncMock()
    stub.compact = AsyncMock()
    stub.run_agent = AsyncMock()
    stub.on_submit = AsyncMock(return_value=(SubmitAction.NONE, ""))
    stub.cycle_completion = MagicMock(return_value=False)
    stub.cancel_choice = AsyncMock(return_value=None)
    return stub


@pytest.fixture
async def running_app(tmp_path, datasci_config):
    """Yield ``(app, pilot, controller_stub)`` for a headless OpenDataSciApp."""
    stub = _make_controller_stub(str(tmp_path))
    with patch.object(app_module, "CLIController", return_value=stub):
        app = OpenDataSciApp(
            workspace_path=str(tmp_path),
            session_id="sess",
            datasci_config=datasci_config,
        )
        async with app.run_test(size=(100, 40)) as pilot:
            yield app, pilot, stub


class TestAppShell:
    async def test_composes_header_chat_pane_and_boots_controller(self, running_app) -> None:
        app, pilot, stub = running_app
        assert app.query_one(AppHeader) is not None
        assert app.query_one(ChatPane) is not None
        assert app.focused is app.query_one("#user-input", Input)
        stub.boot.assert_awaited_once()

    async def test_css_variables_expose_theme_palette(self, running_app) -> None:
        app, _, _ = running_app
        variables = app.get_css_variables()
        assert "ods-accent" in variables
        assert "ods-text-primary" in variables
        assert "ods-warning-bg" in variables

    async def test_adapter_methods_drive_chat_pane(self, running_app) -> None:
        app, pilot, _ = running_app
        bubble = app.add_message("user", "hi")
        assert isinstance(bubble, MessageBubble)
        app.add_divider()
        assert isinstance(app.add_turn_status_bar(), TurnStatusBar)
        assert isinstance(app.add_pending_message("queued"), PendingMessageBubble)
        assert isinstance(app.add_ephemeral_block("", "tool", "summary"), ToolCallBlock)
        assert isinstance(app.add_worker_block("", ["w1"]), ToolCallBlock)
        assert isinstance(app.add_thinking_block(), ThinkingBlock)
        await pilot.pause()

        app.clear_messages()
        await pilot.pause()
        assert not list(app.query(MessageBubble))

    async def test_adapter_input_and_header_helpers(self, running_app) -> None:
        app, pilot, _ = running_app
        inp = app.query_one("#user-input", Input)

        app.set_input_placeholder("waiting…")
        assert inp.placeholder == "waiting…"

        app.set_input_value("hello", cursor=2)
        assert inp.value == "hello"
        assert inp.cursor_position == 2

        app.add_input_class("busy")
        assert inp.has_class("busy")
        app.remove_input_class("busy")
        assert not inp.has_class("busy")

        app.set_workspace("ws-name")
        app.set_file_count("2 files")
        app.show_completion(["a.csv"], 0)
        assert app.query_one(CompletionPopup).has_class("active")
        app.hide_completion()
        assert not app.query_one(CompletionPopup).has_class("active")

        app.show_workspace_panel(["a.csv"])
        await pilot.pause()
        assert app.query_one(WorkspacePanel).has_class("active")

    async def test_submit_runs_agent_and_records_history(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.on_submit.return_value = (SubmitAction.RUN, "the query")
        inp = app.query_one("#user-input", SmartInput)
        inp.value = "analyse this"
        await pilot.press("enter")
        await pilot.pause()

        stub.on_submit.assert_awaited_once_with("analyse this")
        stub.run_agent.assert_awaited_once_with("the query")
        assert inp.value == ""
        assert inp._input_history._history == ["analyse this"]

    async def test_submit_quit_action_exits_app(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.on_submit.return_value = (SubmitAction.QUIT, "")
        app.exit = MagicMock()
        inp = app.query_one("#user-input", SmartInput)
        inp.value = "/exit"
        await pilot.press("enter")
        await pilot.pause()
        app.exit.assert_called_once()

    async def test_blank_submission_is_not_pushed_to_history(self, running_app) -> None:
        app, pilot, stub = running_app
        inp = app.query_one("#user-input", SmartInput)
        inp.value = "   "
        await pilot.press("enter")
        await pilot.pause()
        stub.on_submit.assert_awaited_once_with("")
        assert inp._input_history._history == []

    async def test_paste_message_is_forwarded_to_controller(self, running_app) -> None:
        app, pilot, stub = running_app
        inp = app.query_one("#user-input", SmartInput)
        inp.post_message(SmartInput.Pasted("a\nb"))
        await pilot.pause()
        stub.on_paste.assert_called_once_with("a\nb")

    async def test_input_changes_are_forwarded_to_controller(self, running_app) -> None:
        app, pilot, stub = running_app
        await pilot.press("@", "s", "a", "l")
        await pilot.pause()
        stub.on_input_changed.assert_called_with("@sal")

    async def test_approval_decision_resumes_agent(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.resolve_approval = AsyncMock(return_value="resume-query")
        app.query_one(ChatPane).show_approval_prompt("Do it?", "")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        stub.resolve_approval.assert_called_once_with(True)
        stub.run_agent.assert_awaited_once_with("resume-query")
        assert app.focused is app.query_one("#user-input", Input)


class TestQuitSemantics:
    async def test_ctrl_c_while_idle_requires_double_press(self, running_app) -> None:
        app, pilot, stub = running_app
        app.exit = MagicMock()
        await pilot.press("ctrl+c")
        assert app._quit_requested is True
        app.exit.assert_not_called()
        await pilot.press("ctrl+c")
        app.exit.assert_called_once()

    async def test_ctrl_c_while_agent_running_stops_agent_instead(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.agent_running = True
        app.exit = MagicMock()
        await pilot.press("ctrl+c")
        stub.stop_agent.assert_awaited_once()
        app.exit.assert_not_called()
        assert app._quit_requested is False

    async def test_quit_request_resets_after_timeout(self, running_app) -> None:
        app, pilot, stub = running_app
        await pilot.press("ctrl+c")
        assert app._quit_requested is True
        app._reset_quit_request()
        assert app._quit_requested is False
        app.exit = MagicMock()
        await pilot.press("ctrl+c")
        app.exit.assert_not_called()  # back to first-press semantics

    async def test_quit_action_exits_immediately(self, running_app) -> None:
        # There is no direct quit key (Ctrl+C double-press and /exit are the
        # user-facing paths); the action itself must still exit unconditionally.
        app, pilot, stub = running_app
        app.exit = MagicMock()
        await app.run_action("quit")
        app.exit.assert_called_once()


class TestKeyRouting:
    async def test_ctrl_r_resets_and_ctrl_l_clears(self, running_app) -> None:
        app, pilot, stub = running_app
        await pilot.press("ctrl+r")
        stub.reset.assert_awaited_once()
        await pilot.press("ctrl+l")
        stub.clear_conv.assert_awaited_once()

    async def test_up_down_cycle_completion_when_matches_active(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.has_completion_matches = True
        stub.cycle_completion = MagicMock(return_value=True)
        app.query_one("#user-input", Input).value = "@sal"
        await pilot.pause()
        await pilot.press("down")
        stub.cycle_completion.assert_called_with("@sal", direction=1)
        await pilot.press("up")
        stub.cycle_completion.assert_called_with("@sal", direction=-1)

    async def test_up_navigates_history_when_no_completion(self, running_app) -> None:
        app, pilot, stub = running_app
        inp = app.query_one("#user-input", SmartInput)
        inp.push_history("previous command")
        await pilot.press("up")
        assert inp.value == "previous command"

    async def test_up_with_no_history_resets_completing_flag(self, running_app) -> None:
        app, pilot, stub = running_app
        await pilot.press("up")
        stub.suppress_next_input_change.assert_called_once()
        stub.cancel_input_change_suppression.assert_called_once()

    async def test_escape_cancels_choice_and_resumes(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.awaiting_choice = True
        stub.cancel_choice = AsyncMock(return_value="resume-input")
        await pilot.press("escape")
        await pilot.pause()
        stub.cancel_choice.assert_called_once()
        stub.run_agent.assert_awaited_once_with("resume-input")

    async def test_escape_stops_running_agent(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.agent_running = True
        await pilot.press("escape")
        stub.stop_agent.assert_awaited_once()

    async def test_escape_only_dismisses_completion_when_one_is_open(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.agent_running = True
        stub.has_completion_matches = True
        await pilot.press("escape")
        stub.hide_completion.assert_called()
        stub.stop_agent.assert_not_awaited()

    async def test_tab_falls_back_to_focus_next_without_completion(self, running_app) -> None:
        app, pilot, stub = running_app
        stub.cycle_completion = MagicMock(return_value=False)
        await pilot.press("tab")
        await pilot.pause()
        stub.cycle_completion.assert_called_once()


async def test_controller_closed_on_unmount(tmp_path, datasci_config) -> None:
    stub = _make_controller_stub(str(tmp_path))
    with patch.object(app_module, "CLIController", return_value=stub):
        app = OpenDataSciApp(
            workspace_path=str(tmp_path), session_id="sess", datasci_config=datasci_config
        )
        async with app.run_test(size=(100, 40)):
            pass
    # Textual may unmount the app more than once during shutdown; the contract
    # is simply that the controller gets closed.
    stub.close.assert_awaited()


async def test_unknown_theme_falls_back_to_default(tmp_path, datasci_config) -> None:
    from opendatasci._tui import theme as _theme

    stub = _make_controller_stub(str(tmp_path))
    with patch.object(app_module, "CLIController", return_value=stub):
        OpenDataSciApp(
            workspace_path=str(tmp_path),
            session_id="sess",
            datasci_config=datasci_config,
            theme="does-not-exist",
        )
    assert _theme.active_name == "default"
    assert _theme.active == _theme.THEMES["default"]


# ---------------------------------------------------------------------------
# _get_version
# ---------------------------------------------------------------------------


class TestGetVersion:
    def test_returns_installed_version(self) -> None:
        assert _get_version() != ""

    def test_falls_back_when_package_not_installed(self) -> None:
        import importlib.metadata as md

        with patch.object(app_module.importlib.metadata, "version") as version_mock:
            version_mock.side_effect = md.PackageNotFoundError
            assert _get_version() == "0.2.0"


# ---------------------------------------------------------------------------
# main() — CLI argument parsing and config assembly
# ---------------------------------------------------------------------------


@pytest.fixture
def app_cls_stub():
    """Patch OpenDataSciApp inside main() and capture its constructor kwargs."""
    with patch.object(app_module, "OpenDataSciApp") as cls:
        cls.return_value.run = MagicMock()
        yield cls


def _run_main(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["opendatasci", *argv])
    main()


class TestMainArgParsing:
    def test_defaults_to_anthropic_provider_and_model(
        self, monkeypatch, app_cls_stub, tmp_path
    ) -> None:
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n")
        _run_main(monkeypatch, str(data))
        kwargs = app_cls_stub.call_args.kwargs
        config: OpenDataSciConfig = kwargs["datasci_config"]
        assert kwargs["workspace_path"] == str(data)
        assert config.provider == Provider.ANTHROPIC
        assert config.model == DEFAULT_MODEL[Provider.ANTHROPIC]
        assert config.secondary_provider == Provider.ANTHROPIC
        app_cls_stub.return_value.run.assert_called_once()

    def test_explicit_provider_model_and_api_key(
        self, monkeypatch, app_cls_stub, tmp_path
    ) -> None:
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n")
        _run_main(
            monkeypatch,
            str(data),
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "--secondary-provider",
            "openai",
            "--secondary-model",
            "gpt-4o-mini",
            "--api-key",
            "sk-cli",
        )
        config = app_cls_stub.call_args.kwargs["datasci_config"]
        assert config.provider == Provider.OPENAI
        assert config.model == "gpt-4o"
        assert config.secondary_model == "gpt-4o-mini"
        assert config.openai_api_key == "sk-cli"

    def test_secondary_defaults_follow_primary_provider(
        self, monkeypatch, app_cls_stub, tmp_path
    ) -> None:
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n")
        _run_main(monkeypatch, str(data), "--provider", "openai")
        config = app_cls_stub.call_args.kwargs["datasci_config"]
        assert config.secondary_provider == Provider.OPENAI

    def test_api_key_rejected_for_cloud_native_provider(
        self, monkeypatch, app_cls_stub, tmp_path
    ) -> None:
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, str(data), "--provider", "bedrock", "--api-key", "k")
        app_cls_stub.assert_not_called()

    def test_missing_path_errors(self, monkeypatch, app_cls_stub) -> None:
        with pytest.raises(SystemExit):
            _run_main(monkeypatch)
        app_cls_stub.assert_not_called()

    def test_list_providers_prints_table_and_exits(
        self, monkeypatch, app_cls_stub, capsys
    ) -> None:
        _run_main(monkeypatch, "--list-providers")
        out = capsys.readouterr().out
        assert "anthropic" in out
        assert "openai" in out
        app_cls_stub.assert_not_called()

    def test_config_file_provides_base_and_flags_override(
        self, monkeypatch, app_cls_stub, tmp_path
    ) -> None:
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "provider: openai\n"
            "model: gpt-4o\n"
            "secondary_provider: openai\n"
            "secondary_model: gpt-4o-mini\n"
        )
        _run_main(monkeypatch, str(data), "--config", str(cfg), "--model", "gpt-4.1")
        config = app_cls_stub.call_args.kwargs["datasci_config"]
        assert config.provider == Provider.OPENAI
        assert config.model == "gpt-4.1"  # flag wins over the YAML value
        assert config.secondary_model == "gpt-4o-mini"

    def test_config_file_with_api_key_targets_effective_provider(
        self, monkeypatch, app_cls_stub, tmp_path
    ) -> None:
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text("provider: gemini\nmodel: gemini-2.5-pro\n")
        _run_main(monkeypatch, str(data), "--config", str(cfg), "--api-key", "g-key")
        config = app_cls_stub.call_args.kwargs["datasci_config"]
        assert config.google_api_key == "g-key"

    def test_config_file_api_key_rejected_for_cloud_native_provider(
        self, monkeypatch, app_cls_stub, tmp_path
    ) -> None:
        data = tmp_path / "d.csv"
        data.write_text("a\n1\n")
        cfg = tmp_path / "config.yaml"
        cfg.write_text("provider: bedrock\nmodel: some-model\n")
        with pytest.raises(SystemExit):
            _run_main(monkeypatch, str(data), "--config", str(cfg), "--api-key", "k")
        app_cls_stub.assert_not_called()

    def test_version_flag_exits_cleanly(self, monkeypatch, app_cls_stub, capsys) -> None:
        with pytest.raises(SystemExit) as excinfo:
            _run_main(monkeypatch, "--version")
        assert excinfo.value.code == 0
        assert "OpenDataSci" in capsys.readouterr().out
        app_cls_stub.assert_not_called()
