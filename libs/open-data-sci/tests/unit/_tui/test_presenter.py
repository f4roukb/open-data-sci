"""Unit tests for _TurnPresenter (opendatasci._tui.presenter)."""


from unittest.mock import MagicMock

import pytest

from opendatasci.streaming.events import (
    ReasoningEvent,
    SubagentEvent,
    TokenEvent,
    ToolCallEvent,
    ToolCommunicationEvent,
    ToolResultEvent,
    UsageEvent,
    WorkerDoneEvent,
)
from opendatasci._tui.presenter import _TurnPresenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ui() -> MagicMock:
    ui = MagicMock()
    msg = MagicMock()
    msg.append = MagicMock()
    msg.finish = MagicMock()
    msg.finish_with_summary = MagicMock()
    ui.add_message.return_value = msg
    ui.add_thinking_block.return_value = MagicMock()
    return ui


def _reasoning(content: str = "let me think") -> ReasoningEvent:
    return ReasoningEvent(content=content)


# ---------------------------------------------------------------------------
# Thinking-block spinner
# ---------------------------------------------------------------------------


class TestThinkingBlockSpinner:
    def test_spinner_shown_on_init(self) -> None:
        ui = _make_ui()
        _TurnPresenter(ui)
        ui.add_thinking_block.assert_called_once()

    def test_spinner_reshown_after_tool_result(self) -> None:
        # The spinner is dismissed by tool_call and must reappear after tool_result.
        ui = _make_ui()
        eph = MagicMock()
        ui.add_ephemeral_block.return_value = eph
        p = _TurnPresenter(ui)
        p.handle_tool_call(
            ToolCallEvent(
                tool="execute_python_code",
                tool_call_id="tc1",
                summary="",
            )
        )
        ui.add_thinking_block.reset_mock()
        p.handle_tool_result(ToolResultEvent(tool_call_id="tc1"))
        ui.add_thinking_block.assert_called_once()


# ---------------------------------------------------------------------------
# handle_reasoning
# ---------------------------------------------------------------------------


class TestHandleReasoning:
    def test_does_not_create_thinking_message_bubble(self) -> None:
        ui = _make_ui()
        p = _TurnPresenter(ui)
        p.handle_reasoning(_reasoning("I am thinking hard"))
        thinking_calls = [c for c in ui.add_message.call_args_list if c.args[0] == "thinking"]
        assert len(thinking_calls) == 0

    def test_thinking_block_not_dismissed_on_reasoning(self) -> None:
        ui = _make_ui()
        p = _TurnPresenter(ui)
        block = ui.add_thinking_block.return_value
        p.handle_reasoning(_reasoning("x"))
        block.dismiss.assert_not_called()

    def test_thinking_start_recorded_only_on_first_chunk(self) -> None:
        ui = _make_ui()
        p = _TurnPresenter(ui)
        p.handle_reasoning(_reasoning("a"))
        first_start = p._thinking_start
        p.handle_reasoning(_reasoning("b"))
        assert p._thinking_start == first_start


# ---------------------------------------------------------------------------
# cleanup — thinking message finalised correctly
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_finalises_thinking_block_with_summary(self) -> None:
        ui = _make_ui()
        p = _TurnPresenter(ui)
        block = ui.add_thinking_block.return_value
        p.handle_reasoning(_reasoning("pondering"))
        p.cleanup()
        block.finish.assert_called_once()
        summary_arg = block.finish.call_args.args[0]
        assert "Thought for" in summary_arg

    def test_cleanup_dismisses_thinking_block_when_no_reasoning(self) -> None:
        ui = _make_ui()
        p = _TurnPresenter(ui)
        block = ui.add_thinking_block.return_value
        p.cleanup()
        block.dismiss.assert_called_once()
        block.finish.assert_not_called()


# ---------------------------------------------------------------------------
# handle_usage
# ---------------------------------------------------------------------------


class TestHandleUsage:
    def _presenter(self) -> _TurnPresenter:
        return _TurnPresenter(_make_ui())

    def _usage(self, **kwargs: object) -> UsageEvent:
        return UsageEvent(**kwargs)  # type: ignore[arg-type]

    def test_context_tokens_is_input_plus_output(self) -> None:
        p = self._presenter()
        bar = MagicMock()
        p.handle_usage(self._usage(input_tokens=1200, output_tokens=300), bar)
        bar.update_context.assert_called_once_with(1500, None)

    def test_cache_read_tokens_forwarded(self) -> None:
        p = self._presenter()
        bar = MagicMock()
        p.handle_usage(
            self._usage(input_tokens=1000, output_tokens=200, cache_read_tokens=600), bar
        )
        bar.update_context.assert_called_once_with(1200, 600)

    def test_missing_token_keys_yields_none_context(self) -> None:
        p = self._presenter()
        bar = MagicMock()
        p.handle_usage(self._usage(), bar)
        bar.update_context.assert_called_once_with(None, None)

    def test_missing_cache_key_yields_none_cached(self) -> None:
        p = self._presenter()
        bar = MagicMock()
        p.handle_usage(self._usage(input_tokens=500, output_tokens=100), bar)
        _, cached = bar.update_context.call_args.args
        assert cached is None

    def test_none_turn_status_does_not_raise(self) -> None:
        p = self._presenter()
        p.handle_usage(self._usage(input_tokens=100, output_tokens=50), None)


# ---------------------------------------------------------------------------
# display=False — tool calls must be invisible to the user
# ---------------------------------------------------------------------------


def _hidden_tool_call(tool_call_id: str = "h1") -> ToolCallEvent:
    # ask_user_mcq is registered with display=False in _tui/tools_display.py
    return ToolCallEvent(
        tool="ask_user_mcq",
        tool_call_id=tool_call_id,
        summary="",
    )


def _visible_tool_call(tool_call_id: str = "v1") -> ToolCallEvent:
    return ToolCallEvent(
        tool="execute_python_code",
        tool_call_id=tool_call_id,
        summary="Training model",
    )


class TestDisplayFalse:
    def _setup(self) -> tuple[_TurnPresenter, MagicMock]:
        ui = _make_ui()
        eph = MagicMock()
        eph.is_running.return_value = True
        ui.add_ephemeral_block.return_value = eph
        return _TurnPresenter(ui), ui

    def test_no_ephemeral_block_created_for_hidden_tool(self) -> None:
        p, ui = self._setup()
        p.handle_tool_call(_hidden_tool_call())
        ui.add_ephemeral_block.assert_not_called()

    def test_thinking_block_not_dismissed_for_hidden_tool(self) -> None:
        p, ui = self._setup()
        thinking = ui.add_thinking_block.return_value
        p.handle_tool_call(_hidden_tool_call())
        thinking.dismiss.assert_not_called()

    def test_tool_result_for_hidden_tool_does_not_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        p, ui = self._setup()
        p.handle_tool_call(_hidden_tool_call("h1"))
        with caplog.at_level(logging.WARNING):
            p.handle_tool_result(ToolResultEvent(tool_call_id="h1"))
        assert not caplog.records

    def test_tool_result_for_hidden_tool_does_not_reshow_thinking_block(self) -> None:
        p, ui = self._setup()
        p.handle_tool_call(_hidden_tool_call("h1"))
        ui.add_thinking_block.reset_mock()
        p.handle_tool_result(ToolResultEvent(tool_call_id="h1"))
        ui.add_thinking_block.assert_not_called()

    def test_pending_comm_ephemeral_dismissed_when_tool_is_hidden(self) -> None:
        p, ui = self._setup()
        comm_block = MagicMock()
        comm_block.is_running.return_value = True
        ui.add_ephemeral_block.return_value = comm_block
        # Communication arrives before tool_call (creates a pending ephemeral)
        p.handle_tool_communication(
            ToolCommunicationEvent(
                content="Saving notes…",
                tool_call_id="h1",
                tool_name="internal_action",
            )
        )
        assert ui.add_ephemeral_block.call_count == 1
        # tool_call fires with display=False — pending block must be dismissed
        p.handle_tool_call(_hidden_tool_call("h1"))
        comm_block.dismiss.assert_called_once()

    def test_visible_tool_after_hidden_tool_still_creates_ephemeral(self) -> None:
        p, ui = self._setup()
        p.handle_tool_call(_hidden_tool_call("h1"))
        p.handle_tool_result(ToolResultEvent(tool_call_id="h1"))
        ui.add_ephemeral_block.reset_mock()
        p.handle_tool_call(_visible_tool_call("v1"))
        ui.add_ephemeral_block.assert_called_once()

    def test_hidden_tool_call_id_cleared_after_result(self) -> None:
        p, _ = self._setup()
        p.handle_tool_call(_hidden_tool_call("h1"))
        assert "h1" in p._hidden_tool_call_ids
        p.handle_tool_result(ToolResultEvent(tool_call_id="h1"))
        assert "h1" not in p._hidden_tool_call_ids


# ---------------------------------------------------------------------------
# communication suppressed when narration precedes the tool call
# ---------------------------------------------------------------------------


def _tool_comm(tool_call_id: str = "tc1", content: str = "Fetching results…") -> ToolCommunicationEvent:
    return ToolCommunicationEvent(
        content=content,
        tool_call_id=tool_call_id,
        tool_name="execute_python_code",
    )


def _token(content: str = "Let me run that.") -> TokenEvent:
    return TokenEvent(content=content)


class TestCommunicationSuppressedByNarration:
    def _setup(self) -> tuple[_TurnPresenter, MagicMock]:
        ui = _make_ui()
        eph = MagicMock()
        eph.is_running.return_value = True
        ui.add_ephemeral_block.return_value = eph
        return _TurnPresenter(ui), ui

    def test_communication_shown_when_no_narration(self) -> None:
        p, ui = self._setup()
        p.handle_tool_communication(_tool_comm("tc1", "Fetching results…"))
        p.handle_tool_call(_visible_tool_call("tc1"))
        _, call_kwargs = (
            ui.add_ephemeral_block.call_args_list[-1].args,
            ui.add_ephemeral_block.call_args_list[-1],
        )
        # The pre-mounted ephemeral is upgraded (not a new add_ephemeral_block call), so
        # check that set_communication was NOT called (comm not cleared).
        eph = ui.add_ephemeral_block.return_value
        eph.set_communication.assert_not_called()

    def test_communication_suppressed_for_new_block_when_narration_present(self) -> None:
        p, ui = self._setup()
        # Token arrives before tool_call → narration present
        p.handle_token(_token("Let me run that."))
        p.handle_tool_communication(_tool_comm("tc1", "Fetching results…"))
        ui.add_ephemeral_block.reset_mock()
        p.handle_tool_call(_visible_tool_call("tc1"))
        # The pre-mounted block is upgraded; check set_communication(None) was called on it
        eph = ui.add_ephemeral_block.return_value
        eph.set_communication.assert_called_once_with(None)

    def test_communication_suppressed_for_direct_new_block_when_narration_present(self) -> None:
        """No pre-mounted ephemeral but comm buffered: block gets empty comm when narration present."""
        p, ui = self._setup()
        p.handle_token(_token("Here we go."))
        # Directly buffer comm without pre-mounting (edge case: comm arrived after narration)
        p._comm_buffers["tc1"] = "Running analysis…"
        p.handle_tool_call(_visible_tool_call("tc1"))
        comm_arg = ui.add_ephemeral_block.call_args.args[0]
        assert comm_arg == ""

    def test_communication_passed_through_when_no_narration(self) -> None:
        """No narration → comm forwarded verbatim to the new ephemeral block."""
        p, ui = self._setup()
        # Pre-buffer communication without going through a pre-mounted block
        p._comm_buffers["tc1"] = "Running analysis…"
        p.handle_tool_call(_visible_tool_call("tc1"))
        comm_arg = ui.add_ephemeral_block.call_args.args[0]
        assert comm_arg == "Running analysis…"


# ---------------------------------------------------------------------------
# handle_tool_result — success vs error dispatch
# ---------------------------------------------------------------------------


def _tool_result(tool_call_id: str, *, is_error: bool = False) -> ToolResultEvent:
    return ToolResultEvent(content="output", tool_call_id=tool_call_id, is_error=is_error)


class TestToolResultErrorHandling:
    def _setup(self) -> tuple[_TurnPresenter, MagicMock]:
        ui = _make_ui()
        eph = MagicMock()
        eph.is_running.return_value = True
        ui.add_ephemeral_block.return_value = eph
        return _TurnPresenter(ui), ui

    def test_successful_result_calls_set_done(self) -> None:
        p, ui = self._setup()
        eph = ui.add_ephemeral_block.return_value
        p.handle_tool_call(_visible_tool_call("tc1"))
        p.handle_tool_result(_tool_result("tc1", is_error=False))
        eph.set_done.assert_called_once()
        eph.set_error.assert_not_called()

    def test_error_result_calls_set_error(self) -> None:
        p, ui = self._setup()
        eph = ui.add_ephemeral_block.return_value
        p.handle_tool_call(_visible_tool_call("tc1"))
        p.handle_tool_result(_tool_result("tc1", is_error=True))
        eph.set_error.assert_called_once()
        eph.set_done.assert_not_called()

    def test_missing_is_error_key_defaults_to_set_done(self) -> None:
        p, ui = self._setup()
        eph = ui.add_ephemeral_block.return_value
        p.handle_tool_call(_visible_tool_call("tc1"))
        p.handle_tool_result(ToolResultEvent(content="output", tool_call_id="tc1"))
        eph.set_done.assert_called_once()
        eph.set_error.assert_not_called()

    def test_spinner_reshown_after_error_result(self) -> None:
        p, ui = self._setup()
        p.handle_tool_call(_visible_tool_call("tc1"))
        ui.add_thinking_block.reset_mock()
        p.handle_tool_result(_tool_result("tc1", is_error=True))
        ui.add_thinking_block.assert_called_once()


# ---------------------------------------------------------------------------
# spawn_workers — worker block lifecycle
# ---------------------------------------------------------------------------


def _spawn_workers_call(
    tool_call_id: str = "sw1",
    worker_summaries: list[str] | None = None,
) -> ToolCallEvent:
    return ToolCallEvent(
        tool="spawn_workers",
        tool_call_id=tool_call_id,
        worker_summaries=worker_summaries if worker_summaries is not None else ["Task A", "Task B"],
    )


def _worker_done(worker_idx: int, success: bool = True) -> WorkerDoneEvent:
    return WorkerDoneEvent(worker_idx=worker_idx, success=success)


def _subagent_event(
    worker_idx: int,
    tool_name: str = "execute_python_code",
    summary: str = "Running",
) -> SubagentEvent:
    return SubagentEvent(
        content=tool_name,
        worker_idx=worker_idx,
        event_type="worker_tool_call",
        summary=summary,
    )


class TestSpawnWorkers:
    def _setup(self) -> tuple[_TurnPresenter, MagicMock, MagicMock]:
        ui = _make_ui()
        worker_block = MagicMock()
        ui.add_worker_block.return_value = worker_block
        eph = MagicMock()
        eph.is_running.return_value = True
        ui.add_ephemeral_block.return_value = eph
        return _TurnPresenter(ui), ui, worker_block

    def test_spawn_workers_creates_worker_block_with_summaries(self) -> None:
        p, ui, _ = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1", ["Task A", "Task B"]))
        ui.add_worker_block.assert_called_once_with("", ["Task A", "Task B"])

    def test_worker_done_success_marks_worker_done(self) -> None:
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        p.handle_worker_done(_worker_done(0, success=True))
        wb.mark_worker_done.assert_called_once_with(0)

    def test_worker_done_failure_marks_worker_error(self) -> None:
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        p.handle_worker_done(_worker_done(1, success=False))
        wb.mark_worker_error.assert_called_once_with(1)

    def test_subagent_event_updates_worker_activity(self) -> None:
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        p.handle_subagent_event(_subagent_event(0, "execute_python_code", "Training model"))
        wb.update_worker_activity.assert_called_once()
        idx, activity = wb.update_worker_activity.call_args.args
        assert idx == 0
        assert "Training model" in activity

    def test_worker_block_survives_subsequent_tool_call(self) -> None:
        """Regression: a non-spawn_workers tool call after spawn_workers must not clobber _worker_block."""
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        # A second tool call fires while workers are still running
        p.handle_tool_call(_visible_tool_call("v1"))
        p.handle_worker_done(_worker_done(0, success=True))
        wb.mark_worker_done.assert_called_once_with(0)

    def test_subagent_event_reaches_block_after_subsequent_tool_call(self) -> None:
        """Regression: activity updates must not be dropped after a concurrent tool call."""
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        p.handle_tool_call(_visible_tool_call("v1"))
        p.handle_subagent_event(_subagent_event(1, "execute_python_code", "Fitting"))
        wb.update_worker_activity.assert_called_once()

    def test_tool_result_for_spawn_workers_marks_block_done(self) -> None:
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        p.handle_tool_result(ToolResultEvent(tool_call_id="sw1"))
        wb.set_done.assert_called_once()

    def test_tool_result_error_for_spawn_workers_marks_block_error(self) -> None:
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        p.handle_tool_result(ToolResultEvent(tool_call_id="sw1", is_error=True))
        wb.set_error.assert_called_once()
        wb.set_done.assert_not_called()

    def test_worker_done_is_no_op_before_spawn_workers(self) -> None:
        p, _, wb = self._setup()
        p.handle_worker_done(_worker_done(0))
        wb.mark_worker_done.assert_not_called()

    def test_subagent_event_is_no_op_before_spawn_workers(self) -> None:
        p, _, wb = self._setup()
        p.handle_subagent_event(_subagent_event(0))
        wb.update_worker_activity.assert_not_called()

    def test_worker_tool_result_clears_activity(self) -> None:
        """Regression: a finished tool call inside a worker must clear the inline
        activity so the row reverts to the subtask summary while the LLM is
        deciding the next step — otherwise the row would keep displaying the
        previous tool's name long after it finished."""
        p, _, wb = self._setup()
        p.handle_tool_call(_spawn_workers_call("sw1"))
        result_event = SubagentEvent(
            content="execute_python_code",
            worker_idx=0,
            event_type="worker_tool_result",
            success=True,
        )
        p.handle_subagent_event(result_event)
        wb.update_worker_activity.assert_called_once_with(0, "")

    def test_spawn_workers_replaces_pre_mounted_ephemeral(self) -> None:
        """When tool_communication arrives before the tool_call, a placeholder
        ephemeral is pre-mounted.  For spawn_workers the placeholder must be
        dismissed and replaced with a proper worker block."""
        ui = _make_ui()
        pre_mount = MagicMock()
        pre_mount.is_running.return_value = True
        worker_block = MagicMock()
        ui.add_ephemeral_block.return_value = pre_mount
        ui.add_worker_block.return_value = worker_block
        p = _TurnPresenter(ui)

        p.handle_tool_communication(_tool_comm("sw1", "Preparing…"))
        assert pre_mount in p._ephemerals

        p.handle_tool_call(_spawn_workers_call("sw1", ["Task A"]))

        pre_mount.dismiss.assert_called_once()
        assert pre_mount not in p._ephemerals
        assert p._ephemerals_by_id["sw1"] is worker_block


# ---------------------------------------------------------------------------
# tool_communication forwarded to already-running block
# ---------------------------------------------------------------------------


class TestToolCommunicationToRunningBlock:
    def test_set_communication_forwarded_when_block_is_running(self) -> None:
        """Once a tool block is mounted and running, subsequent comm chunks must
        be forwarded to it via set_communication so the user sees them stream in."""
        ui = _make_ui()
        running_block = MagicMock()
        running_block.is_running.return_value = True
        ui.add_ephemeral_block.return_value = running_block
        p = _TurnPresenter(ui)

        # Pre-mount the ephemeral by first sending a comm so the block is registered.
        p.handle_tool_communication(_tool_comm("tc1", "first"))
        # A second comm chunk for the same tool_call_id must route to set_communication.
        p.handle_tool_communication(_tool_comm("tc1", "second"))

        running_block.set_communication.assert_called_once_with("second")

    def test_set_communication_not_forwarded_when_block_finished(self) -> None:
        """If the block has finished running, no further communication should be pushed."""
        ui = _make_ui()
        finished_block = MagicMock()
        finished_block.is_running.return_value = False
        ui.add_ephemeral_block.return_value = finished_block
        p = _TurnPresenter(ui)

        p.handle_tool_communication(_tool_comm("tc1", "first"))
        p.handle_tool_communication(_tool_comm("tc1", "second"))

        finished_block.set_communication.assert_not_called()


# ---------------------------------------------------------------------------
# handle_subagent_event — activity fallbacks
# ---------------------------------------------------------------------------


class TestSubagentEventActivityFallbacks:
    def _setup(self) -> tuple[_TurnPresenter, MagicMock]:
        ui = _make_ui()
        worker_block = MagicMock()
        ui.add_worker_block.return_value = worker_block
        ui.add_ephemeral_block.return_value = MagicMock(is_running=MagicMock(return_value=True))
        p = _TurnPresenter(ui)
        p.handle_tool_call(_spawn_workers_call("sw1"))
        return p, worker_block

    def test_activity_falls_back_to_registry_label_when_no_summary(self) -> None:
        """When the subagent event carries no summary but the tool has a
        ToolDisplay in REGISTRY, the row activity must surface the registry
        label (with icon) instead of the raw tool name."""
        from opendatasci._tui.tools_display import ToolDisplay, _registry, register

        original = _registry.get("fake_demo_tool")
        register("fake_demo_tool", ToolDisplay(label="Demo Tool", icon="🎯"))
        try:
            p, wb = self._setup()
            p.handle_subagent_event(
                SubagentEvent(
                    content="fake_demo_tool",
                    worker_idx=0,
                    event_type="worker_tool_call",
                    summary="",
                )
            )
            _, activity = wb.update_worker_activity.call_args.args
            assert "🎯" in activity
            assert "Demo Tool" in activity
        finally:
            if original is not None:
                _registry["fake_demo_tool"] = original
            else:
                _registry.pop("fake_demo_tool", None)

    def test_activity_falls_back_to_raw_tool_name_when_no_display_and_no_summary(self) -> None:
        """Last-ditch fallback: unknown tool + no summary → render the bare tool name."""
        # "unregistered_xyz" is never registered; no setup needed.
        p, wb = self._setup()
        p.handle_subagent_event(
            SubagentEvent(
                content="unregistered_xyz",
                worker_idx=0,
                event_type="worker_tool_call",
                summary="",
            )
        )
        _, activity = wb.update_worker_activity.call_args.args
        assert activity == "unregistered_xyz"


# ---------------------------------------------------------------------------
# uncorrelated tool_result
# ---------------------------------------------------------------------------


class TestUncorrelatedToolResult:
    def test_unknown_tool_call_id_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A tool_result event with a tool_call_id that was never registered must
        be logged at WARNING level so operators can spot a stream-routing bug,
        but it must NOT raise — ephemerals are left running until cleanup()."""
        import logging

        ui = _make_ui()
        p = _TurnPresenter(ui)
        with caplog.at_level(logging.WARNING, logger="opendatasci._tui.presenter"):
            p.handle_tool_result(ToolResultEvent(tool_call_id="ghost-id"))
        assert any("uncorrelated" in r.message for r in caplog.records)
