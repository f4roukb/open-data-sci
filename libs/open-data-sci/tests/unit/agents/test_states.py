"""Unit tests for opendatasci.agents.states."""


from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from opendatasci.agents.states import reduce_to_ongoing_turn


class TestReduceToOngoingTurn:
    def test_first_message_starts_the_turn(self) -> None:
        human = HumanMessage(content="hi")
        assert reduce_to_ongoing_turn([], [human]) == [human]

    def test_appends_while_turn_is_ongoing(self) -> None:
        human = HumanMessage(content="q")
        ai_with_tool_call = AIMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]
        )
        result = reduce_to_ongoing_turn([human], [ai_with_tool_call])
        assert result == [human, ai_with_tool_call]

    def test_tool_result_appended_mid_turn(self) -> None:
        human = HumanMessage(content="q")
        ai_with_tool_call = AIMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]
        )
        tool_result = ToolMessage(content="result", tool_call_id="1")
        result = reduce_to_ongoing_turn([human, ai_with_tool_call], [tool_result])
        assert result == [human, ai_with_tool_call, tool_result]

    def test_new_user_message_resets_after_completed_turn(self) -> None:
        completed_turn = [HumanMessage(content="q1"), AIMessage(content="a1")]
        next_message = HumanMessage(content="q2")
        result = reduce_to_ongoing_turn(completed_turn, [next_message])
        assert result == [next_message]

    def test_does_not_reset_while_turn_is_incomplete(self) -> None:
        human = HumanMessage(content="q")
        ai_with_tool_call = AIMessage(
            content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]
        )
        ongoing = [human, ai_with_tool_call]
        final_ai = AIMessage(content="done")
        result = reduce_to_ongoing_turn(ongoing, [final_ai])
        # The turn is still being built up, not reset.
        assert result == [human, ai_with_tool_call, final_ai]
