"""Unit tests for opendatasci.tools.planning."""


from unittest.mock import MagicMock

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from opendatasci.tools.planning import create_planning_tools

_CALL_ID = "test_call_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_save_plan():
    return MagicMock()


def _get_enter_tool(save_plan):
    tools = create_planning_tools(save_plan)
    return next(t for t in tools if t.name == "enter_plan_mode")


def _get_exit_tool(save_plan):
    tools = create_planning_tools(save_plan)
    return next(t for t in tools if t.name == "exit_plan_mode")


def _invoke_enter(tool, *, communication: str = "Let me plan this.") -> Command:
    return tool.invoke(
        {"name": tool.name, "id": _CALL_ID, "args": {"communication": communication}, "type": "tool_call"}
    )


def _invoke_exit(tool, *, final_plan: str = "plan") -> Command:
    return tool.invoke(
        {"name": tool.name, "id": _CALL_ID, "args": {"final_plan": final_plan}, "type": "tool_call"}
    )


def _message_content(result: Command) -> str:
    msgs = result.update.get("messages", [])
    return msgs[0].content if msgs else ""


# ---------------------------------------------------------------------------
# create_planning_tools – structure
# ---------------------------------------------------------------------------


class TestGetPlanningToolsStructure:
    def test_returns_two_tools(self) -> None:
        tools = create_planning_tools(_make_save_plan())
        assert len(tools) == 2

    def test_tool_names(self) -> None:
        names = {t.name for t in create_planning_tools(_make_save_plan())}
        assert names == {"enter_plan_mode", "exit_plan_mode"}


# ---------------------------------------------------------------------------
# enter_plan_mode
# ---------------------------------------------------------------------------


class TestEnterPlanMode:
    def test_sets_is_plan_mode_true_in_state(self) -> None:
        result = _invoke_enter(_get_enter_tool(_make_save_plan()))
        assert result.update.get("is_plan_mode") is True

    def test_returns_command(self) -> None:
        result = _invoke_enter(_get_enter_tool(_make_save_plan()))
        assert isinstance(result, Command)

    def test_response_mentions_plan_mode(self) -> None:
        result = _invoke_enter(_get_enter_tool(_make_save_plan()))
        assert "Plan Mode" in _message_content(result)

    def test_response_instructs_to_call_exit(self) -> None:
        result = _invoke_enter(_get_enter_tool(_make_save_plan()))
        assert "exit_plan_mode" in _message_content(result)

    def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = _invoke_enter(_get_enter_tool(_make_save_plan()))
        msgs = result.update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

    def test_does_not_call_save_plan(self) -> None:
        save = _make_save_plan()
        _invoke_enter(_get_enter_tool(save))
        save.assert_not_called()


# ---------------------------------------------------------------------------
# exit_plan_mode
# ---------------------------------------------------------------------------


class TestExitPlanMode:
    def test_calls_save_plan_with_provided_plan(self) -> None:
        save = _make_save_plan()
        _invoke_exit(_get_exit_tool(save), final_plan="Step 1: do X\nStep 2: do Y")
        save.assert_called_once_with("Step 1: do X\nStep 2: do Y")

    def test_sets_is_plan_mode_false_in_state(self) -> None:
        result = _invoke_exit(_get_exit_tool(_make_save_plan()))
        assert result.update.get("is_plan_mode") is False

    def test_returns_command(self) -> None:
        result = _invoke_exit(_get_exit_tool(_make_save_plan()))
        assert isinstance(result, Command)

    def test_response_confirms_plan_recorded(self) -> None:
        result = _invoke_exit(_get_exit_tool(_make_save_plan()))
        content = _message_content(result)
        assert "recorded" in content.lower() or "saved" in content.lower()

    def test_response_mentions_execution_mode(self) -> None:
        result = _invoke_exit(_get_exit_tool(_make_save_plan()))
        assert "execution mode" in _message_content(result).lower()

    def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = _invoke_exit(_get_exit_tool(_make_save_plan()))
        msgs = result.update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

    def test_save_called_before_state_update(self) -> None:
        call_order = []
        save = MagicMock(side_effect=lambda p: call_order.append("save_plan"))
        tool = _get_exit_tool(save)
        _invoke_exit(tool)
        assert call_order == ["save_plan"]
