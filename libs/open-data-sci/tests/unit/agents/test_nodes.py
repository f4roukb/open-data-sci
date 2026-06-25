"""Unit tests for opendatasci.agents.nodes."""


from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from opendatasci.agents.nodes import AgentNode
from opendatasci.agents.states import AgentState


def _make_state(messages: list | None = None) -> AgentState:
    return AgentState(messages=messages or [HumanMessage(content="question")])


def _no_system(state):
    return []


def _one_system(state):
    return [SystemMessage(content="system")]


class TestAgentNode:
    def _make_node(
        self, response: AIMessage | None = None
    ) -> tuple[AgentNode, AsyncMock]:
        if response is None:
            response = AIMessage(content="hello")
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=response)
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_no_system,
        )
        return node, llm

    async def test_ainvoke_returns_messages_key(self) -> None:
        response = AIMessage(content="answer")
        node, _ = self._make_node(response=response)
        result = await node.ainvoke(_make_state())
        assert "messages" in result
        assert result["messages"][0] is response

    async def test_ainvoke_calls_llm_with_state_messages(self) -> None:
        node, llm = self._make_node()
        state = _make_state([HumanMessage(content="hi")])
        await node.ainvoke(state)
        llm.ainvoke.assert_called_once()

    async def test_system_context_prepended_before_history(self) -> None:
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_one_system,
        )
        await node.ainvoke(_make_state())
        call_args = llm.ainvoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "system"

    async def test_history_follows_system_messages(self) -> None:
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        human = HumanMessage(content="hello")
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_one_system,
        )
        await node.ainvoke(_make_state([human]))
        call_args = llm.ainvoke.call_args[0][0]
        # The no-builder branch renders HumanMessages too — same message, new object.
        assert "hello" in call_args[-1].content

    async def test_recap_messages_precede_ongoing_turn_messages(self) -> None:
        from opendatasci.agents.chat_memory import ChatHistoryBuilder, ChatTurnContext

        recap_message = HumanMessage(content="[Earlier session summary]\nold stuff")
        mock_builder = MagicMock(spec=ChatHistoryBuilder)
        mock_builder.build = AsyncMock(return_value=ChatTurnContext(
            messages=[recap_message, HumanMessage(content="q")],
            turn_summaries=[],
        ))

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_one_system,
            chat_history_builder=mock_builder,
        )

        await node.ainvoke(_make_state())
        sent = llm.ainvoke.call_args[0][0]
        assert sent == [
            SystemMessage(content="system"),
            recap_message,
            HumanMessage(content="q"),
        ]

    async def test_build_system_context_called_without_recap_param(self) -> None:
        from opendatasci.agents.chat_memory import ChatHistoryBuilder, ChatTurnContext

        received_states: list[AgentState] = []

        def capture_system(state):
            received_states.append(state)
            return []

        mock_builder = MagicMock(spec=ChatHistoryBuilder)
        mock_builder.build = AsyncMock(return_value=ChatTurnContext(
            messages=[HumanMessage(content="q")],
            turn_summaries=[],
        ))

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=capture_system,
            chat_history_builder=mock_builder,
        )

        await node.ainvoke(_make_state())
        assert len(received_states) == 1

    async def test_turn_summaries_written_back_when_builder_present(self) -> None:
        from opendatasci.agents.chat_memory import ChatHistoryBuilder, ChatTurnContext, ChatTurnSummary

        record = ChatTurnSummary(turn=1, user="q", actions="", agent="a", timestamp="")
        mock_builder = MagicMock(spec=ChatHistoryBuilder)
        mock_builder.build = AsyncMock(return_value=ChatTurnContext(
            messages=[HumanMessage(content="q")],
            turn_summaries=[record],
        ))

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_no_system,
            chat_history_builder=mock_builder,
        )

        result = await node.ainvoke(_make_state())
        assert result["turn_summaries"] == [record]

    async def test_no_turn_summaries_key_without_builder(self) -> None:
        node, _ = self._make_node()
        result = await node.ainvoke(_make_state())
        assert "turn_summaries" not in result

    def test_to_async_callable_returns_callable(self) -> None:
        node, _ = self._make_node()
        fn = node.to_async_callable()
        assert callable(fn)

    async def test_to_async_callable_delegates_to_ainvoke(self) -> None:
        response = AIMessage(content="from callable")
        node, _ = self._make_node(response=response)
        fn = node.to_async_callable()
        result = await fn(_make_state())
        assert result["messages"][0] is response

    async def test_ainvoke_forwards_config_to_llm(self) -> None:
        node, llm = self._make_node()
        config = {"callbacks": [], "tags": ["test"]}
        await node.ainvoke(_make_state(), config=config)
        positional_args = llm.ainvoke.call_args[0]
        assert len(positional_args) == 2
        assert positional_args[1] is config

    async def test_ainvoke_forwards_none_config_when_not_provided(self) -> None:
        node, llm = self._make_node()
        await node.ainvoke(_make_state())
        positional_args = llm.ainvoke.call_args[0]
        assert len(positional_args) == 2
        assert positional_args[1] is None

    async def test_to_async_callable_forwards_config(self) -> None:
        node, llm = self._make_node()
        config = {"callbacks": [], "tags": ["streaming"]}
        fn = node.to_async_callable()
        await fn(_make_state(), config)
        positional_args = llm.ainvoke.call_args[0]
        assert len(positional_args) == 2
        assert positional_args[1] is config

    async def test_to_async_callable_forwards_none_when_config_omitted(self) -> None:
        node, llm = self._make_node()
        fn = node.to_async_callable()
        await fn(_make_state())
        positional_args = llm.ainvoke.call_args[0]
        assert positional_args[1] is None

    async def test_get_llm_with_tools_receives_state(self) -> None:
        received_states: list[AgentState] = []
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        def get_llm(state: AgentState):
            received_states.append(state)
            return llm

        node = AgentNode(
            get_llm_with_tools=get_llm,
            build_system_context=_no_system,
        )
        state = AgentState(messages=[HumanMessage(content="hi")], is_plan_mode=True)
        await node.ainvoke(state)

        assert len(received_states) == 1
        assert received_states[0].is_plan_mode is True


# ---------------------------------------------------------------------------
# Compaction — now lives in ChatHistoryBuilder; AgentNode tests verify the
# node honours whatever ChatTurnContext.messages the builder returns.
# ---------------------------------------------------------------------------


class TestAgentNodeWithCompaction:
    """AgentNode passes builder output straight to the LLM — no compaction logic of its own."""

    async def test_node_uses_compacted_messages_from_builder(self) -> None:
        from opendatasci.agents.chat_memory import ChatHistoryBuilder, ChatTurnContext

        compacted = [HumanMessage(content="compacted summary")]
        mock_builder = MagicMock(spec=ChatHistoryBuilder)
        mock_builder.build = AsyncMock(return_value=ChatTurnContext(
            messages=compacted,
            turn_summaries=[],
        ))

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_no_system,
            chat_history_builder=mock_builder,
        )

        await node.ainvoke(_make_state([HumanMessage(content="original")]))
        called_messages = llm.ainvoke.call_args[0][0]
        assert called_messages == compacted
