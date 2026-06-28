"""Unit tests for opendatasci.agents.nodes."""


from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, SystemMessage

from opendatasci.agents.nodes import AgentNode
from opendatasci.agents.states import AgentState
from opendatasci.memory.messages import AgentMessage, UserMessage


def _make_state(messages: list | None = None) -> AgentState:
    return AgentState(messages=messages or [UserMessage(content="question")])


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
        node, _ = self._make_node(response=AIMessage(content="answer"))
        result = await node.ainvoke(_make_state())
        assert "messages" in result
        assert isinstance(result["messages"][0], AgentMessage)
        assert result["messages"][0].content == "answer"

    async def test_ainvoke_calls_llm_with_state_messages(self) -> None:
        node, llm = self._make_node()
        state = _make_state([UserMessage(content="hi")])
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
        human = UserMessage(content="hello")
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_one_system,
        )
        await node.ainvoke(_make_state([human]))
        call_args = llm.ainvoke.call_args[0][0]
        # The no-builder branch renders UserMessages too — same message, new object.
        assert "hello" in call_args[-1].content

    async def test_recap_messages_precede_ongoing_turn_messages(self) -> None:
        from opendatasci.agents.chat_history import ChatHistoryBuilder
        from opendatasci.memory.chat_memory import ChatTurnContext

        recap_message = UserMessage(content="[Earlier session summary]\nold stuff")
        inline_turn_message = UserMessage(content="q")
        mock_builder = MagicMock(spec=ChatHistoryBuilder)
        mock_builder.build = AsyncMock(return_value=ChatTurnContext(
            messages=[recap_message, inline_turn_message],
            turn_summaries=[],
            chat_history_compaction=None,
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
        assert len(sent) == 3
        assert isinstance(sent[0], SystemMessage) and sent[0].content == "system"
        assert sent[1] is recap_message
        assert sent[2].content == "q"

    async def test_build_system_context_called_without_recap_param(self) -> None:
        from opendatasci.agents.chat_history import ChatHistoryBuilder
        from opendatasci.memory.chat_memory import ChatTurnContext

        received_states: list[AgentState] = []

        def capture_system(state):
            received_states.append(state)
            return []

        mock_builder = MagicMock(spec=ChatHistoryBuilder)
        mock_builder.build = AsyncMock(return_value=ChatTurnContext(
            messages=[UserMessage(content="q")],
            turn_summaries=[],
            chat_history_compaction=None,
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

    async def test_no_turn_summaries_key_without_builder(self) -> None:
        node, _ = self._make_node()
        result = await node.ainvoke(_make_state())
        assert "turn_summaries" not in result

    def test_to_async_callable_returns_callable(self) -> None:
        node, _ = self._make_node()
        fn = node.to_async_callable()
        assert callable(fn)

    async def test_to_async_callable_delegates_to_ainvoke(self) -> None:
        node, _ = self._make_node(response=AIMessage(content="from callable"))
        fn = node.to_async_callable()
        result = await fn(_make_state())
        assert isinstance(result["messages"][0], AgentMessage)
        assert result["messages"][0].content == "from callable"

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
        state = AgentState(messages=[UserMessage(content="hi")], is_plan_mode=True)
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
        from opendatasci.agents.chat_history import ChatHistoryBuilder
        from opendatasci.memory.chat_memory import ChatTurnContext

        compacted = [UserMessage(content="compacted summary")]
        mock_builder = MagicMock(spec=ChatHistoryBuilder)
        mock_builder.build = AsyncMock(return_value=ChatTurnContext(
            messages=compacted,
            turn_summaries=[],
            chat_history_compaction=None,
        ))

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        node = AgentNode(
            get_llm_with_tools=lambda state: llm,
            build_system_context=_no_system,
            chat_history_builder=mock_builder,
        )

        await node.ainvoke(_make_state([UserMessage(content="original")]))
        called_messages = llm.ainvoke.call_args[0][0]
        assert called_messages == compacted
