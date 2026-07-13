"""Unit tests for opendatasci.agents.states."""


from langchain_core.messages import ToolMessage

from opendatasci.agents.states import _reduce_to_ongoing_turn
from opendatasci.memory.messages import AgentMessage, UserMessage


class TestReduceToOngoingTurn:
    def test_first_message_starts_the_turn(self) -> None:
        human = UserMessage(content="hi")
        assert _reduce_to_ongoing_turn([], [human]) == [human]

    def test_appends_while_turn_is_ongoing(self) -> None:
        human = UserMessage(content="q")
        ai_with_tool_call = AgentMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]
        )
        result = _reduce_to_ongoing_turn([human], [ai_with_tool_call])
        assert result == [human, ai_with_tool_call]

    def test_tool_result_appended_mid_turn(self) -> None:
        human = UserMessage(content="q")
        ai_with_tool_call = AgentMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]
        )
        tool_result = ToolMessage(content="result", tool_call_id="1")
        result = _reduce_to_ongoing_turn([human, ai_with_tool_call], [tool_result])
        assert result == [human, ai_with_tool_call, tool_result]

    def test_new_user_message_resets_after_completed_turn(self) -> None:
        completed_turn = [UserMessage(content="q1"), AgentMessage(content="a1")]
        next_message = UserMessage(content="q2")
        result = _reduce_to_ongoing_turn(completed_turn, [next_message])
        assert result == [next_message]

    def test_does_not_reset_while_turn_is_incomplete(self) -> None:
        human = UserMessage(content="q")
        ai_with_tool_call = AgentMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]
        )
        ongoing = [human, ai_with_tool_call]
        final_ai = AgentMessage(content="done")
        result = _reduce_to_ongoing_turn(ongoing, [final_ai])
        assert result == [human, ai_with_tool_call, final_ai]
