"""Unit tests for opendatasci.tools.modes."""


from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from opendatasci.agents.states import AgentState
from opendatasci.context.base import BaseContextStore
from opendatasci.skills import LocalSkillStore
from opendatasci.tools.modes import (
    AgentMode,
    ExitPlanModeTool,
    ExitSelfReviewModeTool,
    SwitchAgentModeTool,
    create_mode_tools,
)

_CALL_ID = "test_call_id"
_SESSION_ID = "test_session"
_STORE = LocalSkillStore()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context_store() -> MagicMock:
    return MagicMock(spec=BaseContextStore)


def _switch_tool(
    *, context_store: MagicMock | None = None, session_id: str | None = _SESSION_ID
) -> SwitchAgentModeTool:
    return SwitchAgentModeTool(skill_store=_STORE, context_store=context_store, session_id=session_id)


async def _invoke_switch(
    tool: SwitchAgentModeTool,
    *,
    mode: AgentMode,
    skill: str | None = None,
    is_plan_mode: bool = False,
    is_self_review_mode: bool = False,
) -> Command:
    state = AgentState(is_plan_mode=is_plan_mode, is_self_review_mode=is_self_review_mode)
    args: dict = {"mode": mode.value, "summary": "s", "communication": "c", "state": state}
    if skill is not None:
        args["skill"] = skill
    return await tool.ainvoke({"name": tool.name, "id": _CALL_ID, "args": args, "type": "tool_call"})


async def _invoke_exit_plan(tool: ExitPlanModeTool, *, final_plan: str = "plan") -> Command:
    return await tool.ainvoke(
        {
            "name": tool.name,
            "id": _CALL_ID,
            "args": {"final_plan": final_plan, "summary": "s", "communication": "c"},
            "type": "tool_call",
        }
    )


async def _invoke_exit_self_review(
    tool: ExitSelfReviewModeTool, *, review: str = "Looks good."
) -> Command:
    return await tool.ainvoke(
        {
            "name": tool.name,
            "id": _CALL_ID,
            "args": {"review": review, "summary": "s", "communication": "c"},
            "type": "tool_call",
        }
    )


def _message_content(result: Command) -> str:
    msgs = result.update.get("messages", [])
    return msgs[0].content if msgs else ""


# ---------------------------------------------------------------------------
# create_mode_tools – structure
# ---------------------------------------------------------------------------


class TestCreateModeToolsStructure:
    def test_returns_three_tools_with_context_and_session(self) -> None:
        tools = create_mode_tools(_STORE, _make_context_store(), _SESSION_ID)
        names = {t.name for t in tools}
        assert names == {"switch_agentic_mode", "exit_self_review_mode", "exit_plan_mode"}

    def test_omits_exit_plan_mode_without_context_store(self) -> None:
        tools = create_mode_tools(_STORE, None, _SESSION_ID)
        names = {t.name for t in tools}
        assert names == {"switch_agentic_mode", "exit_self_review_mode"}

    def test_omits_exit_plan_mode_without_session_id(self) -> None:
        tools = create_mode_tools(_STORE, _make_context_store(), None)
        names = {t.name for t in tools}
        assert names == {"switch_agentic_mode", "exit_self_review_mode"}


# ---------------------------------------------------------------------------
# switch_agentic_mode – plan
# ---------------------------------------------------------------------------


class TestSwitchToPlanMode:
    @pytest.mark.asyncio
    async def test_sets_is_plan_mode_true_in_state(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store()), mode=AgentMode.PLAN
        )
        assert result.update.get("is_plan_mode") is True

    @pytest.mark.asyncio
    async def test_returns_command(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store()), mode=AgentMode.PLAN
        )
        assert isinstance(result, Command)

    @pytest.mark.asyncio
    async def test_response_mentions_plan_mode(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store()), mode=AgentMode.PLAN
        )
        assert "Plan Mode" in _message_content(result)

    @pytest.mark.asyncio
    async def test_response_instructs_to_call_exit(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store()), mode=AgentMode.PLAN
        )
        assert "exit_plan_mode" in _message_content(result)

    @pytest.mark.asyncio
    async def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store()), mode=AgentMode.PLAN
        )
        msgs = result.update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

    @pytest.mark.asyncio
    async def test_unavailable_without_context_store(self) -> None:
        result = await _invoke_switch(_switch_tool(context_store=None), mode=AgentMode.PLAN)
        assert result.update.get("is_plan_mode") is None
        assert "unavailable" in _message_content(result).lower()

    @pytest.mark.asyncio
    async def test_unavailable_without_session_id(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store(), session_id=None), mode=AgentMode.PLAN
        )
        assert result.update.get("is_plan_mode") is None
        assert "unavailable" in _message_content(result).lower()

    @pytest.mark.asyncio
    async def test_blocked_when_self_review_mode_active(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store()),
            mode=AgentMode.PLAN,
            is_self_review_mode=True,
        )
        assert result.update.get("is_plan_mode") is None
        assert "already in self-review mode" in _message_content(result).lower()

    @pytest.mark.asyncio
    async def test_blocked_when_already_in_plan_mode(self) -> None:
        result = await _invoke_switch(
            _switch_tool(context_store=_make_context_store()),
            mode=AgentMode.PLAN,
            is_plan_mode=True,
        )
        assert "already in plan mode" in _message_content(result).lower()


# ---------------------------------------------------------------------------
# switch_agentic_mode – self_review
# ---------------------------------------------------------------------------


class TestSwitchToSelfReviewMode:
    @pytest.mark.asyncio
    async def test_blocked_when_plan_mode_active(self) -> None:
        result = await _invoke_switch(_switch_tool(), mode=AgentMode.SELF_REVIEW, is_plan_mode=True)
        assert "already in plan mode" in _message_content(result).lower()

    @pytest.mark.asyncio
    async def test_does_not_set_self_review_when_plan_mode_active(self) -> None:
        result = await _invoke_switch(_switch_tool(), mode=AgentMode.SELF_REVIEW, is_plan_mode=True)
        assert "is_self_review_mode" not in result.update

    @pytest.mark.asyncio
    async def test_sets_is_self_review_mode_true_in_state(self) -> None:
        result = await _invoke_switch(_switch_tool(), mode=AgentMode.SELF_REVIEW)
        assert result.update.get("is_self_review_mode") is True

    @pytest.mark.asyncio
    async def test_response_mentions_self_review_mode(self) -> None:
        result = await _invoke_switch(_switch_tool(), mode=AgentMode.SELF_REVIEW)
        assert "self-review" in _message_content(result).lower()

    @pytest.mark.asyncio
    async def test_response_instructs_to_call_exit(self) -> None:
        result = await _invoke_switch(_switch_tool(), mode=AgentMode.SELF_REVIEW)
        assert "exit_self_review_mode" in _message_content(result)

    @pytest.mark.asyncio
    async def test_no_skill_does_not_update_active_skills(self) -> None:
        result = await _invoke_switch(_switch_tool(), mode=AgentMode.SELF_REVIEW)
        assert "active_skills" not in result.update

    @pytest.mark.asyncio
    async def test_valid_skill_sets_active_skills(self) -> None:
        result = await _invoke_switch(
            _switch_tool(), mode=AgentMode.SELF_REVIEW, skill="data_science::exploratory_analysis"
        )
        skills = result.update.get("active_skills", [])
        assert len(skills) == 1
        assert skills[0].name == "data_science::exploratory_analysis"

    @pytest.mark.asyncio
    async def test_valid_skill_still_enables_self_review_mode(self) -> None:
        result = await _invoke_switch(
            _switch_tool(), mode=AgentMode.SELF_REVIEW, skill="data_science::exploratory_analysis"
        )
        assert result.update.get("is_self_review_mode") is True

    @pytest.mark.asyncio
    async def test_unknown_skill_returns_error(self) -> None:
        result = await _invoke_switch(
            _switch_tool(), mode=AgentMode.SELF_REVIEW, skill="nonexistent_skill"
        )
        assert "unknown skill" in _message_content(result).lower()

    @pytest.mark.asyncio
    async def test_unknown_skill_does_not_enable_self_review_mode(self) -> None:
        result = await _invoke_switch(
            _switch_tool(), mode=AgentMode.SELF_REVIEW, skill="nonexistent_skill"
        )
        assert "is_self_review_mode" not in result.update

    @pytest.mark.asyncio
    async def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = await _invoke_switch(_switch_tool(), mode=AgentMode.SELF_REVIEW)
        msgs = result.update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

    @pytest.mark.asyncio
    async def test_blocked_when_already_in_self_review_mode(self) -> None:
        result = await _invoke_switch(
            _switch_tool(), mode=AgentMode.SELF_REVIEW, is_self_review_mode=True
        )
        assert "already in self-review mode" in _message_content(result).lower()


# ---------------------------------------------------------------------------
# exit_plan_mode
# ---------------------------------------------------------------------------


class TestExitPlanMode:
    @pytest.mark.asyncio
    async def test_calls_save_plan_with_session_and_plan(self) -> None:
        context_store = _make_context_store()
        tool = ExitPlanModeTool(context_store=context_store, session_id=_SESSION_ID)
        await _invoke_exit_plan(tool, final_plan="Step 1: do X\nStep 2: do Y")
        context_store.save_plan.assert_called_once_with(_SESSION_ID, "Step 1: do X\nStep 2: do Y")

    @pytest.mark.asyncio
    async def test_sets_is_plan_mode_false_in_state(self) -> None:
        tool = ExitPlanModeTool(context_store=_make_context_store(), session_id=_SESSION_ID)
        result = await _invoke_exit_plan(tool)
        assert result.update.get("is_plan_mode") is False

    @pytest.mark.asyncio
    async def test_returns_command(self) -> None:
        tool = ExitPlanModeTool(context_store=_make_context_store(), session_id=_SESSION_ID)
        result = await _invoke_exit_plan(tool)
        assert isinstance(result, Command)

    @pytest.mark.asyncio
    async def test_response_confirms_plan_recorded(self) -> None:
        tool = ExitPlanModeTool(context_store=_make_context_store(), session_id=_SESSION_ID)
        content = _message_content(await _invoke_exit_plan(tool))
        assert "recorded" in content.lower() or "saved" in content.lower()

    @pytest.mark.asyncio
    async def test_response_mentions_execution_mode(self) -> None:
        tool = ExitPlanModeTool(context_store=_make_context_store(), session_id=_SESSION_ID)
        assert "execution mode" in _message_content(await _invoke_exit_plan(tool)).lower()

    @pytest.mark.asyncio
    async def test_command_includes_tool_message_with_correct_id(self) -> None:
        tool = ExitPlanModeTool(context_store=_make_context_store(), session_id=_SESSION_ID)
        msgs = (await _invoke_exit_plan(tool)).update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

    @pytest.mark.asyncio
    async def test_save_called_before_state_update(self) -> None:
        call_order = []
        context_store = MagicMock(spec=BaseContextStore)
        context_store.save_plan.side_effect = lambda sid, p: call_order.append("save_plan")
        tool = ExitPlanModeTool(context_store=context_store, session_id=_SESSION_ID)
        await _invoke_exit_plan(tool)
        assert call_order == ["save_plan"]


# ---------------------------------------------------------------------------
# exit_self_review_mode
# ---------------------------------------------------------------------------


class TestExitSelfReviewMode:
    @pytest.mark.asyncio
    async def test_sets_is_self_review_mode_false_in_state(self) -> None:
        result = await _invoke_exit_self_review(ExitSelfReviewModeTool())
        assert result.update.get("is_self_review_mode") is False

    @pytest.mark.asyncio
    async def test_response_contains_review_text(self) -> None:
        result = await _invoke_exit_self_review(ExitSelfReviewModeTool(), review="Analysis is on track.")
        assert "Analysis is on track." in _message_content(result)

    @pytest.mark.asyncio
    async def test_response_mentions_execution_mode(self) -> None:
        result = await _invoke_exit_self_review(ExitSelfReviewModeTool(), review="Done.")
        assert "execution mode" in _message_content(result).lower()

    @pytest.mark.asyncio
    async def test_command_includes_tool_message_with_correct_id(self) -> None:
        msgs = (await _invoke_exit_self_review(ExitSelfReviewModeTool())).update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID
