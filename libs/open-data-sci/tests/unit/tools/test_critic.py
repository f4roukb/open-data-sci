"""Unit tests for opendatasci.tools.critic."""


from langchain_core.messages import ToolMessage
from langgraph.types import Command

from opendatasci.agents.states import AgentState
from opendatasci.skills import LocalSkillStore
from opendatasci.tools.critic import create_critic_tools

_STORE = LocalSkillStore()
_CALL_ID = "test_call_id"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_enter_tool():
    tools = create_critic_tools(_STORE)
    return next(t for t in tools if t.name == "enter_self_review_mode")


def _get_exit_tool():
    tools = create_critic_tools(_STORE)
    return next(t for t in tools if t.name == "exit_self_review_mode")


def _invoke_enter(tool, *, skill: str | None = None, is_plan_mode: bool = False) -> Command:
    state = AgentState(is_plan_mode=is_plan_mode)
    args: dict = {"state": state}
    if skill is not None:
        args["skill"] = skill
    return tool.invoke({"name": tool.name, "id": _CALL_ID, "args": args, "type": "tool_call"})


def _invoke_exit(tool, *, review: str = "Looks good.") -> Command:
    return tool.invoke(
        {"name": tool.name, "id": _CALL_ID, "args": {"review": review}, "type": "tool_call"}
    )


def _message_content(result: Command) -> str:
    msgs = result.update.get("messages", [])
    return msgs[0].content if msgs else ""


# ---------------------------------------------------------------------------
# create_critic_tools – structure
# ---------------------------------------------------------------------------


class TestGetCriticToolsStructure:
    def test_returns_two_tools(self) -> None:
        assert len(create_critic_tools(_STORE)) == 2

    def test_tool_names(self) -> None:
        names = {t.name for t in create_critic_tools(_STORE)}
        assert names == {"enter_self_review_mode", "exit_self_review_mode"}


# ---------------------------------------------------------------------------
# enter_self_review_mode
# ---------------------------------------------------------------------------


class TestEnterSelfReviewMode:
    def test_blocked_when_plan_mode_active(self) -> None:
        result = _invoke_enter(_get_enter_tool(), is_plan_mode=True)
        assert "plan mode" in _message_content(result).lower()

    def test_does_not_set_self_review_when_plan_mode_active(self) -> None:
        result = _invoke_enter(_get_enter_tool(), is_plan_mode=True)
        assert "is_self_review_mode" not in result.update

    def test_sets_is_self_review_mode_true_in_state(self) -> None:
        result = _invoke_enter(_get_enter_tool())
        assert result.update.get("is_self_review_mode") is True

    def test_response_mentions_self_review_mode(self) -> None:
        result = _invoke_enter(_get_enter_tool())
        assert "self-review" in _message_content(result).lower()

    def test_response_instructs_to_call_exit(self) -> None:
        result = _invoke_enter(_get_enter_tool())
        assert "exit_self_review_mode" in _message_content(result)

    def test_no_skill_does_not_update_active_skills(self) -> None:
        result = _invoke_enter(_get_enter_tool())
        assert "active_skills" not in result.update

    def test_valid_skill_sets_active_skills(self) -> None:
        result = _invoke_enter(_get_enter_tool(), skill="data_science")
        skills = result.update.get("active_skills", [])
        assert len(skills) == 1
        assert skills[0].name == "data_science"

    def test_valid_skill_still_enables_self_review_mode(self) -> None:
        result = _invoke_enter(_get_enter_tool(), skill="data_science")
        assert result.update.get("is_self_review_mode") is True

    def test_unknown_skill_returns_error(self) -> None:
        result = _invoke_enter(_get_enter_tool(), skill="nonexistent_skill")
        assert "unknown skill" in _message_content(result).lower()

    def test_unknown_skill_does_not_enable_self_review_mode(self) -> None:
        result = _invoke_enter(_get_enter_tool(), skill="nonexistent_skill")
        assert "is_self_review_mode" not in result.update

    def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = _invoke_enter(_get_enter_tool())
        msgs = result.update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID


# ---------------------------------------------------------------------------
# exit_self_review_mode
# ---------------------------------------------------------------------------


class TestExitSelfReviewMode:
    def test_sets_is_self_review_mode_false_in_state(self) -> None:
        result = _invoke_exit(_get_exit_tool())
        assert result.update.get("is_self_review_mode") is False

    def test_response_contains_review_text(self) -> None:
        result = _invoke_exit(_get_exit_tool(), review="Analysis is on track.")
        assert "Analysis is on track." in _message_content(result)

    def test_response_mentions_execution_mode(self) -> None:
        result = _invoke_exit(_get_exit_tool(), review="Done.")
        assert "execution mode" in _message_content(result).lower()

    def test_command_includes_tool_message_with_correct_id(self) -> None:
        result = _invoke_exit(_get_exit_tool())
        msgs = result.update.get("messages", [])
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].tool_call_id == _CALL_ID

