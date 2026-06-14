"""Component tests: full agent loop with the real AgentStreamingOrchestrator.

These tests replace the deterministic ``_FixedOrchestrator`` fixture with the
real :class:`AgentStreamingOrchestrator` and a scripted ``GenericFakeChatModel``.
Each test exercises a long code path:

* ``OpenDataSci.astream`` → ``Agent.astream`` → ``AgentStreamingOrchestrator.run``
* The compiled LangGraph (agent → tools → agent) and ``AgentTurnStreamProcessor``
* Real tool execution through ``ToolNode``
* ``TurnSummarizer`` post-turn bookkeeping (background summarizer task)

The LLM is the lowest-level boundary mock; everything above is real.
"""


import asyncio

from langchain_core.messages import AIMessage


def _ai_with_tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """Build an AIMessage that requests a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


class TestSingleTurnTextOnly:
    """Plain text response: no tool calls.

    Exercises graph → agent node → chain_end fallback, response metadata,
    token counters, and the service-level query_started/query_completed events
    in one pass.
    """

    async def test_single_turn_emits_full_event_set(self, loaded_scripted_service):
        svc = await loaded_scripted_service([AIMessage(content="Hello, world.")])

        events = [e async for e in svc.astream("hi")]
        await asyncio.sleep(0)  # drain background summarizer task

        types = [e.type for e in events]
        # _handle_chain_end emits a fallback token event when _current_text is empty
        # (the scripted chat model doesn't stream individual chunks).
        assert "token" in types
        assert "response" in types

        response = next(e for e in events if e.type == "response")
        assert response.content == "Hello, world."


class TestToolCallCycle:
    """Two-step LLM loop: agent calls a tool, gets a result, then produces final text.

    Exercises ToolNode invocation, processor's _handle_tool_call/_handle_tool_end,
    the orchestrator graph-event branch, the display-metadata registry, and the
    message redactor on the second turn — all in one scripted scenario.
    """

    async def test_tool_call_cycle_end_to_end(self, loaded_scripted_service):
        svc = await loaded_scripted_service(
            [
                _ai_with_tool_call(
                    "list_workspace_files",
                    {"summary": "Listing files", "communication": "Looking."},
                    call_id="call_ls",
                ),
                AIMessage(content="There is one CSV file."),
            ]
        )
        events = [e async for e in svc.astream("what files exist?")]
        await asyncio.sleep(0)

        types = [e.type for e in events]
        # Full event set for a tool-using turn.
        assert "tool_communication" in types  # emitted from finalized args
        assert "tool_call" in types
        assert "tool_result" in types
        assert "token" in types  # final text from the second LLM turn
        assert "response" in types

        # tool_call carries the tool name directly.
        tool_call = next(e for e in events if e.type == "tool_call")
        assert tool_call.tool == "list_workspace_files"

        # tool_result content reflects the real tool's output (against the fixture CSV).
        tool_result = next(e for e in events if e.type == "tool_result")
        assert "sales.csv" in tool_result.content
        assert tool_result.is_error is False


class TestStreamingErrorPath:
    """Graph-level exception should yield an error AgentStreamEvent and emit an error event."""

    async def test_graph_failure_yields_error_event(self, loaded_scripted_service):
        # Empty scripted-message iterator → StopIteration → wrapped to RuntimeError
        # by the framework when invoked.
        svc = await loaded_scripted_service([])

        events = [e async for e in svc.astream("trigger failure")]
        await asyncio.sleep(0)

        types = [e.type for e in events]
        assert "error" in types
