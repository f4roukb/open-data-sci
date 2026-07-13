"""Unit tests for TurnRewinder in opendatasci.agents.turn_memory."""


from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from opendatasci.memory.messages import AgentMessage, UserMessage
from opendatasci.memory.turn_memory import TurnRewinder


class TestTurnRewinder:
    def setup_method(self) -> None:
        self.rewinder = TurnRewinder()

    # ------------------------------------------------------------------
    # Empty / no-turn-start cases
    # ------------------------------------------------------------------

    def test_empty_history_returns_empty(self) -> None:
        assert self.rewinder.rewind_last_turn([]) == []

    def test_no_human_message_returns_copy(self) -> None:
        messages = [AgentMessage(content="hello")]
        result = self.rewinder.rewind_last_turn(messages)
        assert result == messages
        assert result is not messages  # must be a copy

    # ------------------------------------------------------------------
    # Single completed turn
    # ------------------------------------------------------------------

    def test_single_turn_drops_all(self) -> None:
        messages = [
            UserMessage(content="hi"),
            AgentMessage(content="there"),
        ]
        assert self.rewinder.rewind_last_turn(messages) == []

    def test_single_turn_keep_user_message(self) -> None:
        messages = [
            UserMessage(content="hi"),
            AgentMessage(content="there"),
        ]
        result = self.rewinder.rewind_last_turn(messages, keep_user_message=True)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    # ------------------------------------------------------------------
    # Multi-turn: only last turn is removed
    # ------------------------------------------------------------------

    def test_two_turns_drops_second_only(self) -> None:
        messages = [
            UserMessage(content="turn1"),
            AgentMessage(content="resp1"),
            UserMessage(content="turn2"),
            AgentMessage(content="resp2"),
        ]
        result = self.rewinder.rewind_last_turn(messages)
        assert len(result) == 2
        assert result[0].content == "turn1"
        assert result[1].content == "resp1"

    def test_two_turns_keep_user_message(self) -> None:
        messages = [
            UserMessage(content="turn1"),
            AgentMessage(content="resp1"),
            UserMessage(content="turn2"),
            AgentMessage(content="resp2"),
        ]
        result = self.rewinder.rewind_last_turn(messages, keep_user_message=True)
        assert len(result) == 3
        assert result[-1].content == "turn2"

    # ------------------------------------------------------------------
    # In-progress turn (no final AI response yet)
    # ------------------------------------------------------------------

    def test_in_progress_turn_drops_partial_turn(self) -> None:
        messages = [
            UserMessage(content="prev"),
            AgentMessage(content="prev_resp"),
            UserMessage(content="ongoing"),
            AgentMessage(content="", tool_calls=[{"id": "1", "name": "tool", "args": {}}]),
            ToolMessage(content="tool result", tool_call_id="1"),
        ]
        result = self.rewinder.rewind_last_turn(messages)
        assert len(result) == 2
        assert result[0].content == "prev"
        assert result[1].content == "prev_resp"

    def test_in_progress_turn_keep_user_message(self) -> None:
        messages = [
            UserMessage(content="prev"),
            AgentMessage(content="prev_resp"),
            UserMessage(content="ongoing"),
            ToolMessage(content="tool result", tool_call_id="1"),
        ]
        result = self.rewinder.rewind_last_turn(messages, keep_user_message=True)
        assert len(result) == 3
        assert result[-1].content == "ongoing"

    # ------------------------------------------------------------------
    # Turn with intermediate tool messages
    # ------------------------------------------------------------------

    def test_turn_with_tool_messages_removed(self) -> None:
        messages = [
            UserMessage(content="q"),
            AgentMessage(content="", tool_calls=[{"id": "t1", "name": "search", "args": {}}]),
            ToolMessage(content="result", tool_call_id="t1"),
            AgentMessage(content="final answer"),
        ]
        result = self.rewinder.rewind_last_turn(messages)
        assert result == []

    # ------------------------------------------------------------------
    # System messages before the turn are preserved
    # ------------------------------------------------------------------

    def test_system_messages_before_turn_are_preserved(self) -> None:
        messages = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AgentMessage(content="there"),
        ]
        result = self.rewinder.rewind_last_turn(messages)
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    # ------------------------------------------------------------------
    # Return value is always a new list (not a view)
    # ------------------------------------------------------------------

    def test_returns_new_list(self) -> None:
        messages = [
            UserMessage(content="a"),
            AgentMessage(content="b"),
            UserMessage(content="c"),
        ]
        result = self.rewinder.rewind_last_turn(messages)
        assert result is not messages

    def test_no_human_message_returns_new_list(self) -> None:
        messages = [AgentMessage(content="x")]
        result = self.rewinder.rewind_last_turn(messages)
        assert result is not messages
