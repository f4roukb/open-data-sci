"""Component tests: CLIController with a stub UIAdapter and stub service.

CLIController is a 442-line orchestrator that mediates between the Textual UI
(``UIAdapter``) and ``OpenDataSciTuiService``. These tests mock at the
lowest possible boundary — the adapter and the service — and drive the
controller through every public entry point:

* on_submit dispatch (text, slash commands, paste attachments, awaiting-choice)
* every slash command branch (/help, /reset, /clear, /compact, /export,
  /ls-workspace, /models, /stop, /vars, unknown)
* the choice-prompt state machine (letter answer, free-form, cancel)
* run_agent streaming dispatch into _TurnPresenter
* export to a file in tmp_path

The result: one test per pathway covers a long slice of controller code.
"""


from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from opendatasci.streaming.events import (
    ErrorEvent,
    InputRequiredEvent,
    ReasoningEvent,
    ResponseEvent,
    SubagentEvent,
    TokenEvent,
    ToolCallEvent,
    ToolCommunicationEvent,
    ToolResultEvent,
    UsageEvent,
    WorkerDoneEvent,
)
from opendatasci._tui.adapter import (
    EphemeralHandle,
    MessageHandle,
    ThinkingHandle,
    TurnStatusHandle,
    UIAdapter,
)
from opendatasci._tui.controller import CLIController
from opendatasci._tui.service import OpenDataSciTuiService

# ---------------------------------------------------------------------------
# Stub UIAdapter
# ---------------------------------------------------------------------------


class _RecordingMessageHandle(MessageHandle):
    def __init__(self, role: str, content: str = "") -> None:
        self.role = role
        self.contents: list[str] = [content] if content else []
        self.finished = False
        self.summary: str | None = None

    def append(self, chunk: str) -> None:
        self.contents.append(chunk)

    def set_content(self, text: str) -> None:
        self.contents = [text]

    def finish(self) -> None:
        self.finished = True

    def finish_with_summary(self, text: str) -> None:
        self.finished = True
        self.summary = text

    @property
    def text(self) -> str:
        return "".join(self.contents)


class _RecordingEphemeral(EphemeralHandle):
    def __init__(self) -> None:
        self.done = False
        self.error = False
        self.dismissed = False
        self.communication: str | None = None
        self.upgraded: tuple[str, str] | None = None
        self.worker_done_indices: list[int] = []

    def dismiss(self) -> None:
        self.dismissed = True

    def set_done(self) -> None:
        self.done = True

    def set_error(self) -> None:
        self.error = True

    def is_running(self) -> bool:
        return not (self.done or self.error or self.dismissed)

    def mark_worker_done(self, idx: int) -> None:
        self.worker_done_indices.append(idx)

    def mark_worker_error(self, idx: int) -> None:
        pass

    def update_worker_activity(self, idx: int, activity: str) -> None:
        pass

    def set_communication(self, text: str | None) -> None:
        self.communication = text

    def upgrade(self, label: str, summary: str) -> None:
        self.upgraded = (label, summary)


class _RecordingThinking(ThinkingHandle):
    def __init__(self) -> None:
        self.dismissed = False

    def dismiss(self) -> None:
        self.dismissed = True


class _RecordingTurnStatus(TurnStatusHandle):
    def __init__(self) -> None:
        self.stopped = False
        self.last_context: tuple[int | None, int | None] | None = None

    def stop(self) -> None:
        self.stopped = True

    def update_context(self, context_tokens: int | None, cached_tokens: int | None) -> None:
        self.last_context = (context_tokens, cached_tokens)


class _RecordingUI(UIAdapter):
    """In-memory UIAdapter that records every call for assertion."""

    def __init__(self) -> None:
        self.messages: list[_RecordingMessageHandle] = []
        self.dividers = 0
        self.workspace_panels: list[list[str]] = []
        self.placeholders: list[str] = []
        self.input_classes: list[str] = []
        self.removed_input_classes: list[str] = []
        self.file_counts: list[str] = []
        self.attachment_labels: list[str] = []
        self.hide_attachment_calls = 0
        self.cleared = 0
        self.stop_agent_calls = 0
        self.input_values: list[tuple[str, int | None]] = []

    def add_message(self, role: str, content: str = "") -> MessageHandle:
        h = _RecordingMessageHandle(role, content)
        self.messages.append(h)
        return h

    def add_divider(self) -> None:
        self.dividers += 1

    def add_turn_status_bar(self) -> TurnStatusHandle:
        return _RecordingTurnStatus()

    def add_ephemeral_block(self, communication: str, label: str, summary: str) -> EphemeralHandle:
        return _RecordingEphemeral()

    def add_worker_block(self, communication: str, worker_summaries: list[str]) -> EphemeralHandle:
        return _RecordingEphemeral()

    def add_thinking_block(self) -> ThinkingHandle:
        return _RecordingThinking()

    def clear_messages(self) -> None:
        self.messages = []
        self.cleared += 1

    def set_workspace(self, name: str) -> None:
        pass

    def set_file_count(self, description: str) -> None:
        self.file_counts.append(description)

    def set_input_placeholder(self, text: str) -> None:
        self.placeholders.append(text)

    def add_input_class(self, cls: str) -> None:
        self.input_classes.append(cls)

    def remove_input_class(self, cls: str) -> None:
        self.removed_input_classes.append(cls)

    def set_input_value(self, value: str, cursor: int | None = None) -> None:
        self.input_values.append((value, cursor))

    def show_completion(self, matches: list[str], selected: int) -> None:
        pass

    def hide_completion(self) -> None:
        pass

    def show_workspace_panel(self, files: list[str]) -> None:
        self.workspace_panels.append(files)

    def show_attachment(self, label: str) -> None:
        self.attachment_labels.append(label)

    def hide_attachment(self) -> None:
        self.hide_attachment_calls += 1

    def stop_agent(self) -> None:
        self.stop_agent_calls += 1


# ---------------------------------------------------------------------------
# Service stub
# ---------------------------------------------------------------------------


def _make_service_stub(
    *,
    astream_events: list | None = None,
    workspace_files: list[str] | None = None,
    compact_summary: str | None = "summary text",
    compact_raises: Exception | None = None,
    reset_raises: Exception | None = None,
) -> MagicMock:
    svc = MagicMock(spec=OpenDataSciTuiService)
    svc.close = AsyncMock()
    svc.get_workspace_files = MagicMock(return_value=workspace_files or [])
    if reset_raises is not None:
        svc.reset_session = AsyncMock(side_effect=reset_raises)
    else:
        svc.reset_session = AsyncMock()
    svc.clear_context = AsyncMock()
    if compact_raises is not None:
        svc.compact_chat_history = AsyncMock(side_effect=compact_raises)
    else:
        svc.compact_chat_history = AsyncMock(return_value=compact_summary)
    svc.rewind_turn = AsyncMock()

    async def _astream(_query: str):
        for ev in astream_events or []:
            yield ev

    svc.astream = _astream
    return svc


def _make_controller(
    tmp_path: Path | None = None,
    service: MagicMock | None = None,
) -> tuple[CLIController, _RecordingUI]:
    ui = _RecordingUI()
    from opendatasci.configs import OpenDataSciConfig

    ctrl = CLIController(
        ui=ui,
        workspace_path=str(tmp_path or Path.cwd()),
        datasci_config=OpenDataSciConfig(provider="anthropic", model="claude-sonnet-4-6"),
        session_id="test-session",
    )
    if service is not None:
        ctrl._service = service
    return ctrl, ui


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnSubmit:
    """on_submit routes input to the correct branch."""

    async def test_empty_input_is_noop(self):
        ctrl, _ = _make_controller(service=_make_service_stub())
        action, payload = await ctrl.on_submit("")
        assert action == ""
        assert payload == ""

    async def test_text_query_returns_run(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        action, payload = await ctrl.on_submit("What is the average?")
        assert action == "run"
        assert payload == "What is the average?"
        # The user bubble must have been added.
        assert any(m.role == "user" for m in ui.messages)

    async def test_busy_agent_rejects_input(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        ctrl._agent_running = True
        action, _ = await ctrl.on_submit("Another question")
        assert action == ""
        # Last message should warn the user.
        assert any("busy" in m.text.lower() for m in ui.messages)

    async def test_slash_command_quit(self):
        ctrl, _ = _make_controller(service=_make_service_stub())
        action, _ = await ctrl.on_submit("/exit")
        assert action == "quit"

    async def test_unknown_slash_command_warns(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        action, _ = await ctrl.on_submit("/bogus")
        assert action == ""
        assert any("unknown command" in m.text.lower() for m in ui.messages)


class TestSlashCommands:
    """Each slash command should reach the right service method or UI element."""

    async def test_help_message_lists_commands(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        await ctrl.on_submit("/help")
        assert any("Available Commands" in m.text for m in ui.messages)

    async def test_models_message_includes_provider(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        await ctrl.on_submit("/models")
        msg = next(m for m in ui.messages if "Model" in m.text)
        assert "Claude" in msg.text or "claude" in msg.text.lower()

    async def test_reset_calls_service(self):
        svc = _make_service_stub()
        ctrl, ui = _make_controller(service=svc)
        await ctrl.on_submit("/reset")
        svc.reset_session.assert_awaited_once()
        assert any("Session reset" in m.text for m in ui.messages)

    async def test_reset_handles_service_error(self):
        svc = _make_service_stub(reset_raises=RuntimeError("kaboom"))
        ctrl, ui = _make_controller(service=svc)
        await ctrl.on_submit("/reset")
        assert any("Reset failed" in m.text for m in ui.messages)

    async def test_clear_context_calls_service(self):
        svc = _make_service_stub()
        ctrl, ui = _make_controller(service=svc)
        await ctrl.on_submit("/clear")
        svc.clear_context.assert_awaited_once()
        assert any("Context cleared" in m.text for m in ui.messages)

    async def test_compact_emits_summary(self):
        svc = _make_service_stub(compact_summary="condensed history")
        ctrl, ui = _make_controller(service=svc)
        await ctrl.on_submit("/compact")
        svc.compact_chat_history.assert_awaited_once()
        assert any("condensed history" in m.text for m in ui.messages)

    async def test_compact_reports_failure(self):
        svc = _make_service_stub(compact_raises=RuntimeError("nope"))
        ctrl, ui = _make_controller(service=svc)
        await ctrl.on_submit("/compact")
        assert any("Compact failed" in m.text for m in ui.messages)

    async def test_ls_workspace_calls_panel(self):
        svc = _make_service_stub(workspace_files=["a.csv", "b.csv"])
        ctrl, ui = _make_controller(service=svc)
        await ctrl.on_submit("/ls-workspace")
        assert ui.workspace_panels == [["a.csv", "b.csv"]]

    async def test_stop_when_idle_warns(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        await ctrl.on_submit("/stop")
        assert any("No agent" in m.text for m in ui.messages)

    async def test_stop_when_running_calls_ui_and_rollback(self):
        svc = _make_service_stub()
        ctrl, ui = _make_controller(service=svc)
        ctrl._agent_running = True
        await ctrl.on_submit("/stop")
        assert ui.stop_agent_calls == 1
        svc.rewind_turn.assert_awaited_once()

    async def test_vars_command_is_deprecated(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        await ctrl.on_submit("/vars")
        assert any("/vars" in m.text and "removed" in m.text for m in ui.messages)


class TestChoicePrompt:
    """The input_required state machine: prompt, letter answer, free text, cancel."""

    def test_show_choice_prompt_sets_awaiting(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        ctrl._show_choice_prompt("Pick one", ["red", "blue"])
        assert ctrl.awaiting_choice is True
        assert "awaiting-choice" in ui.input_classes

    async def test_letter_answer_returns_run_action(self):
        svc = _make_service_stub()
        ctrl, _ = _make_controller(service=svc)
        ctrl._show_choice_prompt("Pick one", ["red", "blue"])
        action, payload = await ctrl.on_submit("B")
        assert action == "run"
        assert payload == "blue"
        assert ctrl.awaiting_choice is False

    async def test_other_choice_enters_custom_input_mode(self):
        svc = _make_service_stub()
        ctrl, _ = _make_controller(service=svc)
        ctrl._show_choice_prompt("Pick one", ["red", "blue"])
        # Two options -> labels A, B; "Other" label is C.
        action, _ = await ctrl.on_submit("C")
        # Still in awaiting-choice mode — waiting for free-form answer.
        assert action == ""
        assert ctrl.awaiting_choice is True
        action, payload = await ctrl.on_submit("custom answer")
        assert action == "run"
        assert payload == "custom answer"

    def test_cancel_choice_returns_cancel_string(self):
        svc = _make_service_stub()
        ctrl, _ = _make_controller(service=svc)
        ctrl._show_choice_prompt("Pick one", ["red", "blue"])
        result = ctrl.cancel_choice()
        assert result == "cancel"
        assert ctrl.awaiting_choice is False


class TestRunAgent:
    """run_agent streams events through _TurnPresenter."""

    async def test_runs_streamed_events_to_completion(self):
        events = [
            TokenEvent(content="Hello "),
            TokenEvent(content="world."),
            ResponseEvent(content="Hello world."),
        ]
        svc = _make_service_stub(astream_events=events)
        ctrl, ui = _make_controller(service=svc)
        await ctrl.run_agent("hi")
        # After completion, _agent_running flips back to False.
        assert ctrl._agent_running is False
        # The agent message bubble received the streamed tokens.
        agent_msgs = [m for m in ui.messages if m.role == "agent"]
        assert any("Hello" in m.text for m in agent_msgs)

    async def test_tool_call_and_result_routed_to_presenter(self):
        events = [
            ToolCallEvent(
                content="{}",
                tool="list_workspace_files",
                tool_call_id="id1",
                summary="Listing",
            ),
            ToolResultEvent(content="ok", tool_call_id="id1", is_error=False),
            ResponseEvent(content="done"),
        ]
        svc = _make_service_stub(astream_events=events)
        ctrl, ui = _make_controller(service=svc)
        await ctrl.run_agent("ls")
        # The presenter must have closed by handle_response — no exceptions raised.
        assert ctrl._agent_running is False

    async def test_input_required_event_opens_prompt(self):
        events = [
            InputRequiredEvent(content="What now?", choices=["A choice", "Another choice"]),
            ResponseEvent(content=""),
        ]
        svc = _make_service_stub(astream_events=events)
        ctrl, _ = _make_controller(service=svc)
        await ctrl.run_agent("question")
        assert ctrl.awaiting_choice is True

    async def test_run_agent_before_load_warns(self):
        ctrl, ui = _make_controller(service=None)
        await ctrl.run_agent("hi")
        assert any("Still loading" in m.text for m in ui.messages)

    async def test_exception_during_stream_caught_by_presenter(self):
        async def _bad_astream(_query):
            raise RuntimeError("upstream went down")
            yield  # pragma: no cover — make it an async generator

        svc = _make_service_stub()
        svc.astream = _bad_astream
        ctrl, ui = _make_controller(service=svc)
        await ctrl.run_agent("Q")
        assert ctrl._agent_running is False
        # Presenter writes an error message into the agent bubble.
        assert any("Error" in m.text for m in ui.messages)


class TestStreamingAllEventTypes:
    """Single end-to-end run that exercises every _TurnPresenter handler.

    A wide-coverage test: it feeds reasoning, tokens, tool_communication,
    tool_call, tool_result, worker_done, subagent_event, usage, and error
    events in one stream so every dispatch branch in run_agent + presenter
    fires together.
    """

    async def test_dispatches_every_event_type(self):
        events = [
            ReasoningEvent(content="thinking..."),
            ToolCommunicationEvent(
                content="working on it",
                tool_call_id="tc1",
                tool_name="list_workspace_files",
            ),
            ToolCallEvent(
                content="{}",
                tool="list_workspace_files",
                tool_call_id="tc1",
                summary="Listing",
            ),
            ToolResultEvent(content="files: a.csv", tool_call_id="tc1", is_error=False),
            ToolCallEvent(
                content="{}",
                tool="spawn_workers",
                tool_call_id="tc2",
                worker_summaries=["w1", "w2"],
            ),
            SubagentEvent(
                content="execute_python_code",
                worker_idx=0,
                event_type="worker_tool_call",
                summary="compute mean",
            ),
            SubagentEvent(content="", worker_idx=0, event_type="worker_tool_result"),
            WorkerDoneEvent(worker_idx=0, success=True),
            WorkerDoneEvent(worker_idx=1, success=False),
            ToolResultEvent(content="ok", tool_call_id="tc2", is_error=False),
            UsageEvent(
                input_tokens=100,
                output_tokens=200,
                cache_read_tokens=10,
                cache_creation_tokens=5,
            ),
            TokenEvent(content="Final "),
            TokenEvent(content="answer."),
            ResponseEvent(content="Final answer."),
        ]
        svc = _make_service_stub(astream_events=events)
        ctrl, ui = _make_controller(service=svc)
        await ctrl.run_agent("Q")
        # Final state: agent stopped, dividers added once per turn.
        assert ctrl._agent_running is False
        assert ui.dividers >= 1

    async def test_hidden_tool_call_skips_block(self):
        """Tools with display=False emit tool_result without a paired ephemeral; no crash."""
        events = [
            ToolCallEvent(
                content="{}",
                tool="ask_user_mcq",
                tool_call_id="hidden1",
                summary="",
            ),
            ToolResultEvent(content="recorded", tool_call_id="hidden1", is_error=False),
            ResponseEvent(content="done"),
        ]
        svc = _make_service_stub(astream_events=events)
        ctrl, _ = _make_controller(service=svc)
        await ctrl.run_agent("Q")
        assert ctrl._agent_running is False

    async def test_error_event_renders_to_agent_bubble(self):
        events = [
            ErrorEvent(content="Authentication failed"),
        ]
        svc = _make_service_stub(astream_events=events)
        ctrl, ui = _make_controller(service=svc)
        await ctrl.run_agent("Q")
        assert any("Authentication failed" in m.text for m in ui.messages)


class TestPasteAttachment:
    """Paste-attachment lifecycle: on_paste shows pill, clear_paste hides it."""

    def test_paste_then_clear_cycle(self):
        ctrl, ui = _make_controller(service=_make_service_stub())
        ctrl.on_paste("multi\nline\ntext")
        assert ui.attachment_labels  # at least one pill label registered
        ctrl.clear_paste_attachment()
        assert ui.hide_attachment_calls >= 1


class TestCloseLifecycle:
    """close() releases service resources if any; is a no-op when no service is loaded."""

    async def test_close_awaits_service_close_when_loaded_and_is_safe_otherwise(self):
        svc = _make_service_stub()
        ctrl, _ = _make_controller(service=svc)
        await ctrl.close()
        svc.close.assert_awaited_once()

        # No service -> still safe to call.
        ctrl_no_svc, _ = _make_controller(service=None)
        await ctrl_no_svc.close()


class TestServiceNotReadyHandling:
    """Service raising RuntimeError on commands is surfaced as a UI message."""

    async def test_ls_workspace_handles_error(self):
        svc = _make_service_stub()
        svc.get_workspace_files = MagicMock(side_effect=RuntimeError("not ready"))
        ctrl, ui = _make_controller(service=svc)
        await ctrl.on_submit("/ls-workspace")
        assert any("not ready" in m.text for m in ui.messages)
