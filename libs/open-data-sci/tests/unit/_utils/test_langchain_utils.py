"""Unit tests for opendatasci._utils.langchain_utils."""


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from opendatasci._utils.langchain_utils import is_ongoing_turn, prepend_messages, render_turn


# ---------------------------------------------------------------------------
# prepend_messages
# ---------------------------------------------------------------------------


class TestPrependMessages:
    def test_prepends_to_empty_history(self) -> None:
        msg = SystemMessage(content="sys")
        result = prepend_messages([], [msg])
        assert result == [msg]

    def test_prepends_before_non_system_messages(self) -> None:
        human = HumanMessage(content="q")
        sys = SystemMessage(content="sys")
        result = prepend_messages([human], [sys])
        assert result == [sys, human]

    def test_drops_existing_system_messages_from_history(self) -> None:
        old_sys = SystemMessage(content="old")
        human = HumanMessage(content="q")
        new_sys = SystemMessage(content="new")
        result = prepend_messages([old_sys, human], [new_sys])
        assert result == [new_sys, human]

    def test_prepends_multiple_messages(self) -> None:
        s1 = SystemMessage(content="first")
        s2 = SystemMessage(content="second")
        human = HumanMessage(content="q")
        result = prepend_messages([human], [s1, s2])
        assert result == [s1, s2, human]

    def test_empty_prepend_strips_system_messages(self) -> None:
        old_sys = SystemMessage(content="old")
        human = HumanMessage(content="q")
        result = prepend_messages([old_sys, human], [])
        assert result == [human]

    def test_preserves_order_of_non_system_messages(self) -> None:
        h1 = HumanMessage(content="first")
        h2 = HumanMessage(content="second")
        sys = SystemMessage(content="sys")
        result = prepend_messages([h1, h2], [sys])
        assert result == [sys, h1, h2]


# ---------------------------------------------------------------------------
# is_ongoing_turn
# ---------------------------------------------------------------------------


class TestIsOngoingTurn:
    def test_empty_list_returns_false(self) -> None:
        assert is_ongoing_turn([]) is False

    def test_does_not_start_with_human_returns_false(self) -> None:
        assert is_ongoing_turn([AIMessage(content="hi")]) is False

    def test_ends_with_tool_message_returns_true(self) -> None:
        turn = [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
            ToolMessage(content="result", tool_call_id="1"),
        ]
        assert is_ongoing_turn(turn) is True

    def test_ends_with_ai_with_tool_calls_returns_true(self) -> None:
        turn = [
            HumanMessage(content="q"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ]
        assert is_ongoing_turn(turn) is True

    def test_ends_with_ai_without_tool_calls_returns_false(self) -> None:
        turn = [HumanMessage(content="q"), AIMessage(content="done")]
        assert is_ongoing_turn(turn) is False

    def test_ends_with_human_message_returns_false(self) -> None:
        assert is_ongoing_turn([HumanMessage(content="q")]) is False

    def test_ends_with_interrupt_reply_returns_true(self) -> None:
        turn = [
            HumanMessage(content="start"),
            AIMessage(content="", tool_calls=[{"name": "ask", "args": {}, "id": "1"}]),
            HumanMessage(content="answer", additional_kwargs={"is_input_on_interrupt": True}),
        ]
        assert is_ongoing_turn(turn) is True


# ---------------------------------------------------------------------------
# render_turn — individual message types
# ---------------------------------------------------------------------------


class TestRenderTurnHumanMessage:
    def test_plain_string_content(self) -> None:
        result = render_turn([HumanMessage(content="hello")])
        assert result == "User: hello"

    def test_list_content_cast_to_string(self) -> None:
        result = render_turn([HumanMessage(content=["part1", "part2"])])
        assert "User:" in result

    def test_whitespace_only_content_omitted(self) -> None:
        result = render_turn([HumanMessage(content="   ")])
        assert result == "(no messages)"


class TestRenderTurnAIMessageNoToolCalls:
    def test_plain_string_content(self) -> None:
        result = render_turn([AIMessage(content="answer")])
        assert result == "Agent: answer"

    def test_thinking_block_skipped(self) -> None:
        msg = AIMessage(content=[
            {"type": "thinking", "thinking": "internal monologue"},
            {"type": "text", "text": "final answer"},
        ])
        result = render_turn([msg])
        assert "internal monologue" not in result
        assert result == "Agent: final answer"

    def test_multiple_text_blocks_joined(self) -> None:
        msg = AIMessage(content=[
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ])
        result = render_turn([msg])
        assert "part one" in result
        assert "part two" in result

    def test_bare_string_in_content_list(self) -> None:
        msg = AIMessage(content=["plain string block"])
        result = render_turn([msg])
        assert "Agent: plain string block" == result

    def test_empty_text_content_omitted(self) -> None:
        result = render_turn([AIMessage(content="")])
        assert result == "(no messages)"


class TestRenderTurnAIMessageWithToolCalls:
    def test_single_tool_call_formatted(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "my_tool", "args": {"x": 1}, "id": "abc"}],
        )
        result = render_turn([msg])
        assert "[TOOL CALL: my_tool]" in result
        assert "{'x': 1}" in result

    def test_multiple_tool_calls_each_rendered(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "args": {}, "id": "1"},
                {"name": "tool_b", "args": {"k": "v"}, "id": "2"},
            ],
        )
        result = render_turn([msg])
        assert "[TOOL CALL: tool_a]" in result
        assert "[TOOL CALL: tool_b]" in result

    def test_tool_call_without_args_key_defaults_to_empty_dict(self) -> None:
        # langchain_core now requires 'args' at construction time; model_construct
        # bypasses validation to exercise render_turn's .get("args", {}) fallback.
        msg = AIMessage.model_construct(content="", tool_calls=[{"name": "t", "id": "1"}])
        assert render_turn([msg]) == "[TOOL CALL: t]\n{}"


class TestRenderTurnToolMessage:
    def test_plain_string_content(self) -> None:
        result = render_turn([ToolMessage(content="42 rows", tool_call_id="1")])
        assert result == "[TOOL OUTPUT]\n42 rows"

    def test_non_string_content_cast(self) -> None:
        result = render_turn([ToolMessage(content=["a", "b"], tool_call_id="1")])
        assert "[TOOL OUTPUT]" in result


class TestRenderTurnUnknownMessageType:
    def test_system_message_silently_skipped(self) -> None:
        result = render_turn([SystemMessage(content="sys prompt")])
        assert result == "(no messages)"


# ---------------------------------------------------------------------------
# render_turn — composite / ordering
# ---------------------------------------------------------------------------


class TestRenderTurnComposite:
    def test_empty_list_returns_sentinel(self) -> None:
        assert render_turn([]) == "(no messages)"

    def test_full_turn_ordering(self) -> None:
        turn = [
            HumanMessage(content="query"),
            AIMessage(
                content="",
                tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "1"}],
            ),
            ToolMessage(content="result", tool_call_id="1"),
            AIMessage(content="answer"),
        ]
        result = render_turn(turn)
        parts = result.split("\n\n")
        assert parts[0] == "User: query"
        assert "[TOOL CALL: search]" in parts[1]
        assert parts[2] == "[TOOL OUTPUT]\nresult"
        assert parts[3] == "Agent: answer"

    def test_intermediate_slice_no_human_message(self) -> None:
        # AgentLoopCompactor passes turn[1:last_ai_idx] — no HumanMessage
        intermediate = [
            AIMessage(
                content="",
                tool_calls=[{"name": "calc", "args": {"n": 2}, "id": "2"}],
            ),
            ToolMessage(content="4", tool_call_id="2"),
        ]
        result = render_turn(intermediate)
        assert "[TOOL CALL: calc]" in result
        assert "[TOOL OUTPUT]\n4" in result
        assert "User:" not in result

    def test_parts_separated_by_double_newline(self) -> None:
        turn = [HumanMessage(content="q"), AIMessage(content="a")]
        assert render_turn(turn) == "User: q\n\nAgent: a"
