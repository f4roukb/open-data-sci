"""Unit tests for opendatasci.agents.agents."""


import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict

from opendatasci.agents.agents import Agent, ConcurrentWorkerAgent, SUBAGENT_TAG
from opendatasci.agents.states import AgentState
from opendatasci.agents.chat_memory import ChatHistoryBuilder, ChatTurnSummary
from opendatasci.configs import OpenDataSciConfig
from pathlib import Path

from opendatasci.context.local import LocalContextStore
from opendatasci.sandbox.base import BaseSandboxFactory
from opendatasci.skills import BaseSkillStore
from opendatasci.skills.base import Skill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_sandbox() -> MagicMock:
    sb = MagicMock()
    sb.get_history = MagicMock(return_value=[])
    sb.execute = AsyncMock()
    sb.execute_cli = AsyncMock()
    return sb


def _make_mock_factory(sandbox: MagicMock | None = None) -> tuple[MagicMock, MagicMock]:
    """Return ``(factory, sandbox)`` where factory.create() yields the sandbox."""
    if sandbox is None:
        sandbox = _make_mock_sandbox()

    factory = MagicMock(spec=BaseSandboxFactory)

    @asynccontextmanager
    async def _create(*_args: object, **_kwargs: object) -> AsyncIterator[MagicMock]:
        yield sandbox

    factory.create = _create
    return factory, sandbox


def _make_mock_llm() -> MagicMock:
    """Return a MagicMock LLM with distinct bound instances per bind_tools call."""
    llm = MagicMock()
    llm.bind_tools.side_effect = lambda _tools: MagicMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=""))
    return llm


def _seed_messages(agent: Agent, messages: list) -> None:
    """Seed messages into the agent's graph state via the real checkpointer."""
    agent.graph.update_state(agent._graph_config, {"messages": messages})


def _get_messages(agent: Agent) -> list:
    """Read the current messages from the agent's graph state."""
    return agent.graph.get_state(agent._graph_config).values.get("messages", [])


def _get_state_value(agent: Agent, key: str, default: object = None) -> object:
    """Read an arbitrary field from the agent's graph state."""
    return agent.graph.get_state(agent._graph_config).values.get(key, default)


@asynccontextmanager
async def _make_agent_ctx(
    context_store: LocalContextStore | None = None,
    skill_store: BaseSkillStore | None = None,
    sandbox: MagicMock | None = None,
) -> AsyncIterator[Agent]:
    """Async context manager that yields a fully set-up ``Agent`` with mocked deps."""
    workspace = MagicMock()
    workspace.get_reference.return_value = "/tmp/fake_workspace"

    factory, mock_sandbox = _make_mock_factory(sandbox)
    if sandbox is None:
        sandbox = mock_sandbox

    mock_llm = _make_mock_llm()

    with (
        patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x),
        patch("opendatasci.agents.agents.create_agent_tools", return_value=[]),
        patch("opendatasci.agents.agents.create_model", return_value=mock_llm),
        patch("opendatasci.agents.agents.create_secondary_model", return_value=mock_llm),
    ):
        agent = Agent(
            workspace=workspace,
            sandbox_factory=factory,
            context_store=context_store or LocalContextStore(Path("/tmp/fake_workspace")),
            skill_store=skill_store or MagicMock(spec=BaseSkillStore),
            config=OpenDataSciConfig(),
            checkpointer=MemorySaver(),
        )
        async with agent:
            yield agent


@asynccontextmanager
async def _agent_with_overrides_ctx(**kwargs: object) -> AsyncIterator[Agent]:
    """Build an agent passing arbitrary keyword overrides to the constructor."""
    workspace = MagicMock()
    workspace.get_reference.return_value = "/tmp/fake_workspace"

    factory, _sandbox = _make_mock_factory()
    mock_llm = _make_mock_llm()

    with (
        patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x),
        patch("opendatasci.agents.agents.create_agent_tools", return_value=[]),
        patch("opendatasci.agents.agents.create_model", return_value=mock_llm),
        patch("opendatasci.agents.agents.create_secondary_model", return_value=mock_llm),
    ):
        agent = Agent(
            workspace=workspace,
            sandbox_factory=kwargs.pop("sandbox_factory", factory),
            context_store=kwargs.pop("context_store", LocalContextStore(Path("/tmp/fake_workspace"))),
            skill_store=kwargs.pop("skill_store", MagicMock(spec=BaseSkillStore)),
            config=OpenDataSciConfig(),
            checkpointer=MemorySaver(),
            **kwargs,
        )
        async with agent:
            yield agent


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestAgentInit:
    async def test_creates_chat_history_builder(self) -> None:
        async with _make_agent_ctx() as agent:
            assert isinstance(agent._chat_history_builder, ChatHistoryBuilder)

    async def test_messages_empty_on_fresh_session(self) -> None:
        async with _make_agent_ctx() as agent:
            assert _get_messages(agent) == []


# ---------------------------------------------------------------------------
# Conversation management
# ---------------------------------------------------------------------------


class TestAgentConversation:
    async def test_clear_chat_history_removes_all_messages(self) -> None:
        async with _make_agent_ctx() as agent:
            _seed_messages(agent, [HumanMessage(content="hello"), AIMessage(content="hi")])
            await agent.clear_chat_history()
            assert _get_messages(agent) == []

    async def test_clear_chat_history_no_op_when_empty(self) -> None:
        async with _make_agent_ctx() as agent:
            await agent.clear_chat_history()
            assert _get_messages(agent) == []

    async def test_clear_chat_history_empties_memory(self) -> None:
        async with _make_agent_ctx() as agent:
            agent.graph.update_state(
                agent._graph_config,
                {"turn_summaries": [ChatTurnSummary(turn=1, user="q", actions="", agent="a")]},
            )
            await agent.clear_chat_history()
            assert _get_state_value(agent, "turn_summaries", []) == []

    async def test_clear_chat_history_drops_unfinished_pending_summary(self) -> None:
        """A background summarization still running when /clear fires must be discarded."""
        async with _make_agent_ctx() as agent:
            never_completes: asyncio.Future = asyncio.get_event_loop().create_future()

            async def _hang_forever(_messages: list) -> ChatTurnSummary:
                await never_completes  # never resolved — simulates an in-flight summary
                raise AssertionError("should have been cancelled before completing")

            builder = agent._chat_history_builder
            builder._summarizer.summarize_turn = _hang_forever  # type: ignore[method-assign]
            builder.schedule_turn_summarization(
                [HumanMessage(content="q"), AIMessage(content="a")]
            )
            pending_task = builder._pending_task
            assert pending_task is not None and not pending_task.done()

            await agent.clear_chat_history()
            await asyncio.sleep(0)  # let the cancellation actually propagate into the task

            assert pending_task.cancelled()
            assert builder._pending_task is None
            assert _get_state_value(agent, "turn_summaries", []) == []

    async def test_compact_chat_history_returns_placeholder_for_empty_history(self) -> None:
        async with _make_agent_ctx() as agent:
            result = await agent.compact_chat_history()
            assert "no conversation" in result.lower()

    async def test_compact_chat_history_returns_placeholder_when_only_ongoing_turn_present(self) -> None:
        async with _make_agent_ctx() as agent:
            _seed_messages(agent, [HumanMessage(content="q"), AIMessage(content="a")])
            result = await agent.compact_chat_history()
            assert "no conversation" in result.lower()

    async def test_compact_chat_history_folds_turn_summaries_into_one_record(self) -> None:
        async with _make_agent_ctx() as agent:
            agent.graph.update_state(
                agent._graph_config,
                {
                    "turn_summaries": [
                        ChatTurnSummary(turn=1, user="question one", actions="", agent="answer one"),
                        ChatTurnSummary(turn=2, user="question two", actions="", agent="answer two"),
                    ]
                },
            )
            agent._llm.ainvoke = AsyncMock(return_value=AIMessage(content="compact summary"))
            result = await agent.compact_chat_history()
            assert result == "compact summary"
            summaries = _get_state_value(agent, "turn_summaries", [])
            assert len(summaries) == 1
            assert summaries[0].turn is None
            assert summaries[0].agent == "compact summary"

    async def test_compact_chat_history_wipes_ongoing_turn_messages(self) -> None:
        async with _make_agent_ctx() as agent:
            _seed_messages(agent, [HumanMessage(content="q"), AIMessage(content="a")])
            agent.graph.update_state(
                agent._graph_config,
                {"turn_summaries": [ChatTurnSummary(turn=1, user="old", actions="", agent="ans")]},
            )
            agent._llm.ainvoke = AsyncMock(return_value=AIMessage(content="summary"))
            await agent.compact_chat_history()
            # Compaction has the same effect on messages as clear_chat_history.
            assert _get_messages(agent) == []

    async def test_compact_chat_history_folds_existing_compaction_summary_too(self) -> None:
        async with _make_agent_ctx() as agent:
            agent.graph.update_state(
                agent._graph_config,
                {
                    "turn_summaries": [
                        ChatTurnSummary(turn=None, user="", actions="", agent="earlier summary"),
                        ChatTurnSummary(turn=2, user="q2", actions="", agent="a2"),
                    ],
                },
            )
            agent._llm.ainvoke = AsyncMock(return_value=AIMessage(content="compact summary"))
            result = await agent.compact_chat_history()
            assert result == "compact summary"
            summaries = _get_state_value(agent, "turn_summaries", [])
            assert len(summaries) == 1
            assert summaries[0].turn is None
            assert summaries[0].agent == "compact summary"

    async def test_rewind_turn_removes_incomplete_turn(self) -> None:
        async with _make_agent_ctx() as agent:
            _seed_messages(agent, [AIMessage(content="prev"), HumanMessage(content="interrupted")])
            await agent.rewind_turn()
            remaining = _get_messages(agent)
            assert len(remaining) == 1
            assert isinstance(remaining[0], AIMessage)

    async def test_rewind_turn_empty_messages_is_noop(self) -> None:
        async with _make_agent_ctx() as agent:
            await agent.rewind_turn()
            assert _get_messages(agent) == []

    async def test_rewind_turn_removes_completed_turn(self) -> None:
        async with _make_agent_ctx() as agent:
            _seed_messages(agent, [HumanMessage(content="q"), AIMessage(content="a")])
            await agent.rewind_turn()
            assert _get_messages(agent) == []

    async def test_rewind_turn_single_human_message_empties_history(self) -> None:
        async with _make_agent_ctx() as agent:
            _seed_messages(agent, [HumanMessage(content="only message")])
            await agent.rewind_turn()
            assert _get_messages(agent) == []

    async def test_rewind_turn_preserves_earlier_turns(self) -> None:
        async with _make_agent_ctx() as agent:
            _seed_messages(agent, [
                HumanMessage(content="q1"),
                AIMessage(content="a1"),
                HumanMessage(content="interrupted"),
            ])
            await agent.rewind_turn()
            remaining = _get_messages(agent)
            assert len(remaining) == 2
            assert remaining[0].content == "q1"
            assert remaining[1].content == "a1"


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class TestAgentInterruptState:
    async def test_prepare_user_message_records_query_and_timestamp_in_metadata(self) -> None:
        async with _make_agent_ctx() as agent:
            msg = agent._prepare_user_message("hello")
            assert msg.content == "hello"
            assert msg.additional_kwargs["created_at"]
            assert msg.additional_kwargs["origin"] == "user"
            assert msg.additional_kwargs["is_input_on_interrupt"] is False


# ---------------------------------------------------------------------------
# Constructor — explicit-dependency branches
# ---------------------------------------------------------------------------


class TestAgentExplicitDependencies:
    async def test_explicit_context_store_used_as_is(self) -> None:
        explicit_store = LocalContextStore(Path("/tmp/fake_workspace"))
        async with _agent_with_overrides_ctx(context_store=explicit_store) as agent:
            assert agent._context_store is explicit_store

    async def test_explicit_tools_used_as_is(self) -> None:
        from langchain_core.tools import StructuredTool

        sentinel_tool = StructuredTool.from_function(
            func=lambda: "x",
            name="custom_tool",
            description="a custom tool",
        )
        async with _agent_with_overrides_ctx(tools=[sentinel_tool]) as agent:
            assert agent._tools == [sentinel_tool]


# ---------------------------------------------------------------------------
# _get_active_llm_with_tools — mode dispatch
# ---------------------------------------------------------------------------


class TestActiveLlmWithToolsDispatch:
    async def test_normal_mode_returns_full_binding(self) -> None:
        async with _make_agent_ctx() as agent:
            assert agent._get_active_llm_with_tools(AgentState()) is agent._llm_with_tools

    async def test_plan_mode_returns_plan_binding(self) -> None:
        async with _make_agent_ctx() as agent:
            assert (
                agent._get_active_llm_with_tools(AgentState(is_plan_mode=True))
                is agent._llm_with_tools_plan
            )

    async def test_self_review_mode_takes_priority_over_plan_mode(self) -> None:
        async with _make_agent_ctx() as agent:
            assert agent._get_active_llm_with_tools(
                AgentState(is_plan_mode=True, is_self_review_mode=True)
            ) is agent._llm_with_tools_self_review


# ---------------------------------------------------------------------------
# _prepare_user_message
# ---------------------------------------------------------------------------


class TestPrepareUserMessage:
    async def test_prepare_user_message_records_timestamp_in_metadata(self) -> None:
        async with _make_agent_ctx() as agent:
            msg = agent._prepare_user_message("a query")
            assert msg.additional_kwargs["created_at"] is not None

    async def test_prepare_user_message_returns_plain_human_message_marked_not_interrupt(self) -> None:
        async with _make_agent_ctx() as agent:
            msg = agent._prepare_user_message("hello")
            assert isinstance(msg, HumanMessage)
            assert msg.content == "hello"
            assert msg.additional_kwargs["is_input_on_interrupt"] is False

    async def test_prepare_user_message_uses_session_id_not_per_turn_uuid(self) -> None:
        async with _make_agent_ctx() as agent:
            session_id = agent._session_id
            agent._prepare_user_message("first")
            agent._prepare_user_message("second")
            assert agent._session_id == session_id


# ---------------------------------------------------------------------------
# astream() — end-to-end through the orchestrator
# ---------------------------------------------------------------------------


class TestAgentAstream:
    def _wire_astream_mocks(
        self,
        agent: Agent,
        stream_events: list,
    ):
        """Replace graph.astream_events and AgentTurnStreamProcessor so astream
        yields a controlled sequence of AgentStreamEvents.

        Returns a context manager that activates the processor patch; tests must
        use it with ``with self._wire_astream_mocks(...):``."""
        from opendatasci.streaming.events import MessageEvent

        final_ai = AIMessage(content="final-explanation")
        # The first event is always the final AIMessage (captured as "message"),
        # followed by the caller-supplied stream events (forwarded to the caller).
        events_to_produce = [
            MessageEvent(message=final_ai),
            *stream_events,
        ]
        event_queue = list(events_to_produce)

        # Capture kwargs so the session_id test can inspect them
        captured: list[dict] = []

        async def fake_astream_events(state, **kwargs):
            captured.append(kwargs)
            # Persist the turn's messages so the final AIMessage (read from the
            # checkpointed state at finalization) can be recovered.
            agent.graph.update_state(
                agent._graph_config,
                {"messages": [HumanMessage(content="q"), final_ai]},
            )
            for _ in events_to_produce:
                yield {}

        agent.graph.astream_events = fake_astream_events  # type: ignore[attr-defined]
        agent._astream_captured = captured  # expose for tests

        mock_processor = MagicMock()
        mock_processor.process_event.side_effect = (
            lambda _: [event_queue.pop(0)] if event_queue else []
        )

        agent._context_store.prune = MagicMock()

        return patch("opendatasci.agents.agents.AgentTurnStreamProcessor", return_value=mock_processor)

    async def test_astream_emits_response_event_at_end(self) -> None:
        from opendatasci.streaming.events import TokenEvent

        upstream = [TokenEvent(content="hi")]
        async with _make_agent_ctx() as agent:
            with self._wire_astream_mocks(agent, upstream):
                events = [ev async for ev in agent.astream("hello")]
        assert events[-1].type == "response"
        assert events[-1].content == "final-explanation"

    async def test_astream_yields_orchestrator_events_in_order(self) -> None:
        from opendatasci.streaming.events import TokenEvent

        upstream = [
            TokenEvent(content="a"),
            TokenEvent(content="b"),
        ]
        async with _make_agent_ctx() as agent:
            with self._wire_astream_mocks(agent, upstream):
                events = [ev async for ev in agent.astream("q")]
        prefix = [(e.type, e.content) for e in events[:-1]]
        assert prefix == [("token", "a"), ("token", "b")]

    async def test_astream_uses_session_id_as_thread_id(self) -> None:
        from opendatasci.streaming.events import TokenEvent

        upstream = [TokenEvent(content="x")]
        async with _make_agent_ctx() as agent:
            with self._wire_astream_mocks(agent, upstream):
                session_id = agent._session_id
                [ev async for ev in agent.astream("q")]

        config = agent._astream_captured[0]["config"]
        assert config["configurable"]["thread_id"] == session_id

    async def test_astream_schedules_summarization_at_end(self) -> None:
        from opendatasci.streaming.events import TokenEvent

        scheduled: list[list] = []

        upstream = [TokenEvent(content="x")]
        async with _make_agent_ctx() as agent:
            agent._chat_history_builder.schedule_turn_summarization = (  # type: ignore[method-assign]
                lambda messages: scheduled.append(messages)
            )
            with self._wire_astream_mocks(agent, upstream):
                async for _ in agent.astream("q"):
                    pass

        assert len(scheduled) == 1
        assert any(isinstance(m, HumanMessage) for m in scheduled[0])


# ===========================================================================
# ConcurrentWorkerAgent (opendatasci.agents.agents.ConcurrentWorkerAgent)
# ===========================================================================


class _EmptyToolSchema(BaseModel):
    """A real (field-less) pydantic schema so ToolNode's schema introspection
    (``_get_state_args``/``_get_store_arg``) terminates instead of recursing on a
    bare Mock."""


def _make_tool(name: str, return_value: object = "ok result") -> MagicMock:
    """Build a BaseTool-shaped mock accepted by langgraph's ``ToolNode``.

    ``ToolNode`` only treats objects as tools when ``isinstance(t, BaseTool)``,
    invokes them as ``tool.ainvoke({**call, "type": "tool_call"}, config)``, and
    requires a ``ToolMessage`` (or ``Command``) back.  ``spec=BaseTool`` makes the
    ``isinstance`` check pass while keeping ``.ainvoke`` a settable/observable
    mock; ``get_input_schema`` returns a real empty model so ToolNode's
    construction-time introspection does not recurse.
    """
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.get_input_schema = MagicMock(return_value=_EmptyToolSchema)

    async def _ainvoke(call: dict, config: object = None, **_: object) -> ToolMessage:
        return ToolMessage(content=str(return_value), tool_call_id=call.get("id", "tc1"))

    tool.ainvoke = AsyncMock(side_effect=_ainvoke)
    return tool


class _AnyArgs(BaseModel):
    """Permissive schema so a tool accepts whatever args a tool-call carries."""

    model_config = ConfigDict(extra="allow")


def _make_real_tool(name: str, return_value: str = "ok result") -> StructuredTool:
    """A genuine ``StructuredTool`` so it participates in ``astream_events`` and
    emits ``on_tool_start``/``on_tool_end`` (needed to exercise the worker's
    ``on_event`` callback path)."""

    async def _arun(**_: object) -> str:
        return return_value

    return StructuredTool(
        name=name, description="test tool", args_schema=_AnyArgs, coroutine=_arun
    )


def _make_agent(
    tools: list | None = None,
    llm_responses: list | None = None,
) -> tuple[ConcurrentWorkerAgent, AsyncMock]:
    """Create a ConcurrentWorkerAgent with a mocked LLM."""
    if tools is None:
        tools = []
    mock_llm = MagicMock()
    mock_bound = AsyncMock()
    if llm_responses:
        mock_bound.ainvoke = AsyncMock(side_effect=llm_responses)
    else:
        mock_bound.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
    mock_llm.bind_tools.return_value = mock_bound

    with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
        agent = ConcurrentWorkerAgent(tools=tools, llm=mock_llm)

    return agent, mock_bound


class TestWorkerAgentRun:
    async def test_returns_string_response(self) -> None:
        agent, _ = _make_agent(llm_responses=[AIMessage(content="final answer")])
        result = await agent.ainvoke("task", "system prompt")
        assert result == "final answer"

    async def test_handles_list_text_content(self) -> None:
        response = AIMessage(
            content=[
                {"type": "text", "text": "part1"},
                {"type": "text", "text": "part2"},
            ]
        )
        agent, _ = _make_agent(llm_responses=[response])
        result = await agent.ainvoke("task", "system")
        assert "part1" in result
        assert "part2" in result

    async def test_handles_list_string_parts(self) -> None:
        response = AIMessage(content=["string block"])
        agent, _ = _make_agent(llm_responses=[response])
        result = await agent.ainvoke("task", "system")
        assert "string block" in result

    async def test_handles_non_string_content(self) -> None:
        # model_construct bypasses pydantic validation so we can carry a
        # non-string, non-list content (42) through the real graph; run() must
        # stringify it.
        response = AIMessage.model_construct(content=42, tool_calls=[], id="nonstr-1")
        agent, _ = _make_agent(llm_responses=[response])
        result = await agent.ainvoke("task", "system")
        assert result == "42"

    async def test_calls_tool_and_continues(self) -> None:
        tool = _make_tool("my_tool", "tool output")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {"x": 1}, "id": "tc1"}],
            ),
            AIMessage(content="final after tool"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        result = await agent.ainvoke("task", "system")
        assert result == "final after tool"
        tool.ainvoke.assert_called_once()
        call_dict = tool.ainvoke.call_args.args[0]
        assert call_dict["args"]["x"] == 1

    async def test_unknown_tool_returns_error_message_to_llm(self) -> None:
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "tc1"}],
            ),
            AIMessage(content="handled unknown tool"),
        ]
        agent, mock_bound = _make_agent(llm_responses=responses)
        result = await agent.ainvoke("task", "system")
        assert result == "handled unknown tool"
        # Verify the LLM was called twice (tool call + final response)
        assert mock_bound.ainvoke.call_count == 2

    async def test_tool_failure_is_caught_and_reported(self) -> None:
        tool = _make_tool("bad_tool")
        tool.ainvoke = AsyncMock(side_effect=RuntimeError("tool exploded"))
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "bad_tool", "args": {}, "id": "tc1"}],
            ),
            AIMessage(content="recovered from failure"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        result = await agent.ainvoke("task", "system")
        assert result == "recovered from failure"

    async def test_tool_result_passed_back_to_llm(self) -> None:
        tool = _make_tool("read_tool", "file contents here")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "read_tool", "args": {}, "id": "tc1"}],
            ),
            AIMessage(content="done"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        await agent.ainvoke("task", "system")
        # Second call to ainvoke should receive the ToolMessage
        second_call_messages = mock_bound.ainvoke.call_args_list[1][0][0]
        assert any(isinstance(m, ToolMessage) for m in second_call_messages)

    async def test_budget_exceeded_raises(self) -> None:
        tool = _make_tool("my_tool", "result")

        # A fresh AIMessage per call (unique auto id) so add_messages appends
        # rather than dedup-replaces, letting the loop hit the recursion limit.
        def _always_tool_call(*_a: object, **_k: object) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {}, "id": "tc1"}],
            )

        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=_always_tool_call)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        # GraphRecursionError subclasses RecursionError -> RuntimeError.
        with pytest.raises(RuntimeError):
            await agent.ainvoke("task", "system")


class TestWorkerAgentCallbacks:
    async def test_on_event_fired_for_tool_call(self) -> None:
        tool = _make_real_tool("my_tool", "result")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {"a": "b"}, "id": "tc1"}],
            ),
            AIMessage(content="done"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        events: list[tuple] = []

        def on_event(event_type: str, content: str, metadata: dict | None) -> None:
            events.append((event_type, content, metadata))

        await agent.ainvoke("task", "system", on_event=on_event)

        event_types = [e[0] for e in events]
        assert "worker_tool_call" in event_types
        assert "worker_tool_result" in event_types

    async def test_on_event_tool_call_includes_args_preview(self) -> None:
        tool = _make_real_tool("my_tool")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {"key": "value"}, "id": "tc1"}],
            ),
            AIMessage(content="done"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        events: list[tuple] = []
        await agent.ainvoke("task", "system", on_event=lambda t, c, m: events.append((t, c, m)))

        tool_call_event = next(e for e in events if e[0] == "worker_tool_call")
        assert "args_preview" in tool_call_event[2]

    async def test_on_event_tool_result_success_flag(self) -> None:
        tool = _make_real_tool("my_tool", "success output")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {}, "id": "tc1"}],
            ),
            AIMessage(content="done"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        events: list[tuple] = []
        await agent.ainvoke("task", "system", on_event=lambda t, c, m: events.append((t, c, m)))

        result_event = next(e for e in events if e[0] == "worker_tool_result")
        assert result_event[2]["success"] is True

    async def test_on_event_not_called_without_tool_calls(self) -> None:
        agent, _ = _make_agent(llm_responses=[AIMessage(content="direct answer")])
        events: list = []
        await agent.ainvoke("task", "system", on_event=lambda t, c, m: events.append(t))
        assert events == []

    async def test_no_on_event_runs_without_error(self) -> None:
        tool = _make_tool("my_tool")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {}, "id": "tc1"}],
            ),
            AIMessage(content="done"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        result = await agent.ainvoke("task", "system")
        assert result == "done"


class TestWorkerAgentSubagentTagging:
    """Worker LLM and tool invocations must carry the SUBAGENT_TAG so the main
    agent's astream_events stream can filter their nested events out.
    """

    @staticmethod
    def _config_tags(call_args) -> list[str]:
        config = call_args.kwargs.get("config") or (
            call_args.args[1] if len(call_args.args) > 1 else None
        )
        if not config:
            return []
        return list(config.get("tags") or [])

    async def test_llm_ainvoke_tagged_with_subagent_tag(self) -> None:
        agent, mock_bound = _make_agent(llm_responses=[AIMessage(content="answer")])
        await agent.ainvoke("task", "system")
        assert mock_bound.ainvoke.call_count == 1
        tags = self._config_tags(mock_bound.ainvoke.call_args)
        assert SUBAGENT_TAG in tags

    async def test_llm_ainvoke_tagged_across_multiple_steps(self) -> None:
        tool = _make_tool("my_tool", "tool output")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {"x": 1}, "id": "tc1"}],
            ),
            AIMessage(content="final"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        await agent.ainvoke("task", "system")
        assert mock_bound.ainvoke.call_count == 2
        for call in mock_bound.ainvoke.call_args_list:
            assert SUBAGENT_TAG in self._config_tags(call)

    async def test_tool_ainvoke_tagged_with_subagent_tag(self) -> None:
        tool = _make_tool("my_tool", "tool output")
        responses = [
            AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {"x": 1}, "id": "tc1"}],
            ),
            AIMessage(content="done"),
        ]
        mock_llm = MagicMock()
        mock_bound = AsyncMock()
        mock_bound.ainvoke = AsyncMock(side_effect=responses)
        mock_llm.bind_tools.return_value = mock_bound

        with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
            agent = ConcurrentWorkerAgent(tools=[tool], llm=mock_llm)

        await agent.ainvoke("task", "system")
        tool.ainvoke.assert_called_once()
        tags = self._config_tags(tool.ainvoke.call_args)
        assert SUBAGENT_TAG in tags


class TestWorkerAgentMessagesOut:
    async def test_messages_out_populated_after_run(self) -> None:
        agent, _ = _make_agent(llm_responses=[AIMessage(content="answer")])
        messages_out: list = []
        await agent.ainvoke("task", "system", messages_out=messages_out)
        # At minimum: SystemMessage, HumanMessage, AIMessage
        assert len(messages_out) >= 3

    async def test_messages_out_none_does_not_raise(self) -> None:
        agent, _ = _make_agent(llm_responses=[AIMessage(content="answer")])
        result = await agent.ainvoke("task", "system", messages_out=None)
        assert result == "answer"


def _has_cache_marker(content: object) -> bool:
    """Return True if a SystemMessage carries an Anthropic or Bedrock cache marker."""
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and ("cache_control" in b or "cachePoint" in b) for b in content)


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _make_agent_with_provider(
    provider: str,
    llm_responses: list | None = None,
) -> tuple[ConcurrentWorkerAgent, AsyncMock]:
    """Build a ConcurrentWorkerAgent wired to a specific provider config, with a mocked LLM."""
    if llm_responses is None:
        llm_responses = [AIMessage(content="done")]
    mock_llm = MagicMock()
    mock_bound = AsyncMock()
    mock_bound.ainvoke = AsyncMock(side_effect=llm_responses)
    mock_llm.bind_tools.return_value = mock_bound
    config = OpenDataSciConfig(provider=provider)  # type: ignore[arg-type]
    with patch("opendatasci.agents.agents.with_retry", side_effect=lambda x: x):
        agent = ConcurrentWorkerAgent(tools=[], config=config, llm=mock_llm)
    return agent, mock_bound


class TestWorkerAgentSystemPromptCaching:
    """The worker mirrors SystemPromptBuilder's caching contract.

    Same invariants: base prompt always cached, skill (when the session has
    one) sits immediately after the base prompt and is also cached. Total
    cache markers per request: 1 without a skill, 2 with a skill.
    """

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    async def test_base_prompt_carries_cache_marker(self, provider: str) -> None:
        agent, _ = _make_agent_with_provider(provider)
        messages_out: list = []
        await agent.ainvoke("task", "worker base prompt", messages_out=messages_out)
        # The base SystemMessage is first.
        assert _has_cache_marker(messages_out[0].content)
        assert "worker base prompt" in _extract_text(messages_out[0].content)

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    async def test_skill_message_immediately_after_base_and_cached(self, provider: str) -> None:
        agent, _ = _make_agent_with_provider(provider)
        messages_out: list = []
        await agent.ainvoke(
            "task",
            "worker base prompt",
            messages_out=messages_out,
            initial_active_skills=[Skill(name="skill", content="worker skill body")],
        )

        # messages_out = [base, skill, HumanMessage(task), AIMessage(done)]
        assert "worker base prompt" in _extract_text(messages_out[0].content)
        assert "worker skill body" in _extract_text(messages_out[1].content)
        assert _has_cache_marker(messages_out[1].content)

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    async def test_exactly_one_cache_marker_without_skill(self, provider: str) -> None:
        agent, _ = _make_agent_with_provider(provider)
        messages_out: list = []
        await agent.ainvoke("task", "worker base", messages_out=messages_out)
        # Only SystemMessages can carry cache markers — count them.
        marked = [m for m in messages_out if hasattr(m, "content") and _has_cache_marker(m.content)]
        assert len(marked) == 1

    @pytest.mark.parametrize("provider", ["anthropic", "bedrock"])
    async def test_exactly_two_cache_markers_with_skill(self, provider: str) -> None:
        agent, _ = _make_agent_with_provider(provider)
        messages_out: list = []
        await agent.ainvoke(
            "task",
            "worker base",
            messages_out=messages_out,
            initial_active_skills=[Skill(name="skill", content="skill body")],
        )
        marked = [m for m in messages_out if hasattr(m, "content") and _has_cache_marker(m.content)]
        assert len(marked) == 2

    async def test_session_without_skill_keeps_single_cache_marker(self) -> None:
        agent, _ = _make_agent_with_provider("anthropic")
        messages_out: list = []
        await agent.ainvoke(
            "task",
            "worker base",
            messages_out=messages_out,
            initial_active_skills=[],
        )
        marked = [m for m in messages_out if hasattr(m, "content") and _has_cache_marker(m.content)]
        assert len(marked) == 1

    @pytest.mark.parametrize("provider", ["openai", "gemini", "ollama", "openai_compatible_server"])
    async def test_non_breakpoint_providers_use_plain_strings(self, provider: str) -> None:
        # Providers with server-side automatic caching get plain-string content
        # for both base and skill so the prefix stays byte-identical.
        agent, _ = _make_agent_with_provider(provider)
        messages_out: list = []
        await agent.ainvoke(
            "task",
            "worker base",
            messages_out=messages_out,
            initial_active_skills=[Skill(name="skill", content="skill body")],
        )
        assert isinstance(messages_out[0].content, str)
        assert isinstance(messages_out[1].content, str)
        assert messages_out[1].content == "skill body"
