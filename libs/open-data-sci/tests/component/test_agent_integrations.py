"""Component tests for the most fragile multi-component integrations.

Each test mocks at the LLM/sandbox boundary and exercises an end-to-end flow
that involves several real subsystems wired together:

* ``task`` → ``WorkerAgent`` lifecycle (queues, sub-agent events, redactor)
* Plan-mode round-trip (``switch_agentic_mode`` → ``exit_plan_mode``, ``LocalContextStore``)
* Self-review-mode round-trip (``switch_agentic_mode`` → ``exit_self_review_mode``)
* Turn summarization (``TurnSummarizer`` writing to ``ChatMemory``)
"""

import asyncio
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, ToolMessage

from opendatasci.agents.agents import Invocation
from opendatasci.agents.states import AgentState
from opendatasci.tools.factory import ToolName


def _ai_with_tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ---------------------------------------------------------------------------
# Worker spawning
# ---------------------------------------------------------------------------


class TestSpawnWorkersFlow:
    """End-to-end task exercising tools/workers.py + agent/worker.py.

    Patches ``opendatasci.agents.workers.create_model`` so the spawned
    :class:`WorkerAgent` uses a scripted LLM rather than calling a real provider.
    The WorkerAgent has no tools required for the test — it just returns a
    final text on its first LLM call.
    """

    async def test_task_runs_real_worker_agent(self, loaded_scripted_service):
        from tests.component.conftest import _ScriptedChatModel  # local import to avoid cycle

        # The worker LLM yields a single message: a final text response with no tool_calls,
        # so WorkerAgent's loop exits after one iteration.
        worker_msgs = [
            AIMessage(content="Worker 1: found 2 columns."),
            AIMessage(content="Worker 2: rev=300."),
        ]
        # `iter(worker_msgs)` is shared, but each WorkerAgent constructs its own
        # scripted model via create_model — we need a fresh scripted model per worker.
        # Use a factory side_effect.
        worker_iter = iter(worker_msgs)

        def _worker_llm(_config):
            return _ScriptedChatModel(messages=iter([next(worker_iter)]))

        with (
            patch("opendatasci.agents.workers.create_model", side_effect=_worker_llm),
            patch("opendatasci.agents.workers.with_retry", side_effect=lambda x: x),
        ):
            svc = await loaded_scripted_service(
                [
                    _ai_with_tool_call(
                        "task",
                        {
                            "subtasks": [
                                {
                                    "subtask": "Count columns",
                                    "summary": "Count columns",
                                    "skill": None,
                                    "allow_web_tools": False,
                                },
                                {
                                    "subtask": "Sum revenue",
                                    "summary": "Sum revenue",
                                    "skill": None,
                                    "allow_web_tools": False,
                                },
                            ],
                            "summary": "Running workers",
                            "communication": "Running two workers in parallel.",
                        },
                        call_id="sw_1",
                    ),
                    AIMessage(content="Workers complete. The data has 2 columns and revenue=300."),
                ]
            )

            events = [e async for e in svc.astream(Invocation.from_text("analyse in parallel"))]
            await asyncio.sleep(0)

        types = [e.type for e in events]
        # task produced a tool_call AgentStreamEvent.
        assert "tool_call" in types
        # Both workers emitted at least one task_done signal (consumed by orchestrator).
        task_done = [e for e in events if e.type == "task_done"]
        assert len(task_done) >= 2
        # Sub-agent events (task_started, task_finished) propagate as subagent_event.
        subagent_events = [e for e in events if e.type == "subagent_event"]
        assert subagent_events  # at least one (task_started or task_finished)
        # Final tool_result for task contains both workers' outputs.
        tool_result = next(e for e in events if e.type == "tool_result")
        assert "found 2 columns" in tool_result.content
        assert "rev=300" in tool_result.content


# ---------------------------------------------------------------------------
# Plan mode
# ---------------------------------------------------------------------------


class TestPlanModeFlow:
    """switch_agentic_mode(mode="plan") → exit_plan_mode round-trip.

    Exercises tools/modes.py, context/local.py (LocalContextStore), and the
    plan-mode branch of SystemPromptBuilder via mode flag mutation.
    """

    async def test_plan_mode_round_trip_saves_plan_and_clears_flag(self, loaded_scripted_service):
        svc = await loaded_scripted_service(
            [
                _ai_with_tool_call(
                    "switch_agentic_mode",
                    {
                        "mode": "plan",
                        "summary": "Planning task",
                        "communication": "Need to plan first.",
                    },
                    call_id="ep1",
                ),
                _ai_with_tool_call(
                    "exit_plan_mode",
                    {
                        "final_plan": "1. Profile the data\n2. Compute revenue",
                        "summary": "Plan ready",
                        "communication": "Plan complete.",
                    },
                    call_id="ep2",
                ),
                AIMessage(content="Plan executed."),
            ]
        )

        agent = svc._agent

        events = [e async for e in svc.astream(Invocation.from_text("complex task"))]
        await asyncio.sleep(0)

        # The plan was persisted into the LocalContextStore.
        plan = agent._context_store.get_current_plan(agent._session_id)
        assert plan is not None
        assert plan.content == "1. Profile the data\n2. Compute revenue"

        # Two tool_call events fired (switch + exit).
        tool_calls = [e for e in events if e.type == "tool_call"]
        assert {e.tool for e in tool_calls} == {
            "switch_agentic_mode",
            "exit_plan_mode",
        }

    async def test_plan_mode_active_switches_llm_binding(self, loaded_scripted_service):
        """_get_active_llm_with_tools returns the right binding for each mode state."""
        svc = await loaded_scripted_service([AIMessage(content="ok")])
        agent = svc._agent

        assert agent._get_active_llm_with_tools(AgentState()) is agent._llm_with_tools
        assert (
            agent._get_active_llm_with_tools(AgentState(is_plan_mode=True))
            is agent._llm_with_tools_plan
        )
        assert (
            agent._get_active_llm_with_tools(AgentState(is_self_review_mode=True))
            is agent._llm_with_tools_self_review
        )

    def test_context_store_plan_persists_to_disk(self, tmp_path):
        from opendatasci.context.local import LocalContextStore

        store = LocalContextStore(tmp_path)
        assert store.get_current_plan("abc12345") is None

        store.save_plan("abc12345", "Step 1\nStep 2")
        # Plan is read back fresh from disk — no in-memory caching.
        plan = store.get_current_plan("abc12345")
        assert plan is not None
        assert plan.content == "Step 1\nStep 2"

        # Save a newer plan — older one is pruned.
        store.save_plan("abc12345", "Step 1\nStep 2\nStep 3")
        plan_files = list(store._plans_root.glob("abc12345_*.json"))
        assert len(plan_files) == 1
        plan = store.get_current_plan("abc12345")
        assert plan is not None
        assert plan.content == "Step 1\nStep 2\nStep 3"


# ---------------------------------------------------------------------------
# Self-review mode
# ---------------------------------------------------------------------------


class TestSelfReviewModeFlow:
    """switch_agentic_mode(mode="self_review") → exit_self_review_mode round-trip."""

    async def test_self_review_round_trip_fires_tool_events(self, loaded_scripted_service):
        svc = await loaded_scripted_service(
            [
                _ai_with_tool_call(
                    "switch_agentic_mode",
                    {
                        "mode": "self_review",
                        "summary": "Reviewing work",
                        "communication": "Reviewing.",
                        "skill": None,
                    },
                    call_id="sr1",
                ),
                _ai_with_tool_call(
                    "exit_self_review_mode",
                    {
                        "review": "All looks good.",
                        "summary": "Review complete",
                        "communication": "Review done.",
                    },
                    call_id="sr2",
                ),
                AIMessage(content="Continuing."),
            ]
        )

        events = [e async for e in svc.astream(Invocation.from_text("review yourself"))]
        await asyncio.sleep(0)

        tool_calls = [e for e in events if e.type == "tool_call"]
        assert {e.tool for e in tool_calls} == {
            "switch_agentic_mode",
            "exit_self_review_mode",
        }

    async def test_self_review_blocked_while_in_plan_mode(self, loaded_scripted_service):
        """switch_agentic_mode refuses to enter self-review while plan mode is active.

        Plan mode is established by scripting a first switch_agentic_mode call
        so the is_plan_mode flag is set in LangGraph state before the second
        switch_agentic_mode(mode="self_review") call runs.
        """
        svc = await loaded_scripted_service(
            [
                _ai_with_tool_call(
                    "switch_agentic_mode",
                    {"mode": "plan", "summary": "Planning task", "communication": "planning first"},
                    call_id="ep1",
                ),
                _ai_with_tool_call(
                    "switch_agentic_mode",
                    {
                        "mode": "self_review",
                        "summary": "Reviewing work",
                        "communication": "Reviewing.",
                        "skill": None,
                    },
                    call_id="sr_blocked",
                ),
                AIMessage(content="OK."),
            ]
        )

        events = [e async for e in svc.astream(Invocation.from_text("try to review"))]
        await asyncio.sleep(0)

        tool_results = [e for e in events if e.type == "tool_result"]
        # The second switch_agentic_mode call is the one that gets blocked.
        blocked = tool_results[1]
        assert "already in plan mode" in blocked.content.lower()


# ---------------------------------------------------------------------------
# Turn summarizer
# ---------------------------------------------------------------------------


class TestTurnSummarizerIntegration:
    """A real summarizer LLM is invoked after a turn and writes into ChatMemory."""

    async def test_summarizer_adds_turn_to_memory(self, loaded_scripted_service):
        svc = await loaded_scripted_service([AIMessage(content="Hello, world.")])
        agent = svc._agent

        # Inject a summarizer LLM whose with_structured_output returns an
        # AsyncMock that yields a real ChatTurnSummaryOutput on ainvoke. We patch
        # the private _structured_llm attribute on the existing
        # ChatTurnSummarizer to avoid rewiring the whole turn finalizer.
        from opendatasci.memory.chat_memory import ChatTurnSummaryOutput

        structured_summary = ChatTurnSummaryOutput(
            user_request="Asked to greet.",
            outcomes="No tools used.",
            agent_response="The agent said hello.",
        )
        structured_llm = AsyncMock()
        structured_llm.ainvoke = AsyncMock(return_value=structured_summary)
        agent._chat_history_builder._summarizer._structured_llm = structured_llm

        async for _ in svc.astream(Invocation.from_text("Greet me")):
            pass
        await asyncio.sleep(0)
        record = await agent._chat_history_builder.flush_pending_tasks()

        # The flushed summary carries the turn details.
        assert record is not None
        recap = agent._chat_history_builder._build_summary_messages([record])
        formatted = recap[0].content[0]["text"]
        assert "Asked to greet" in formatted
        assert "The agent said hello" in formatted

    async def test_summarizer_falls_back_when_llm_disabled(self, loaded_scripted_service):
        """No summarizer LLM (the default in the test fixture) — fallback to raw query."""
        svc = await loaded_scripted_service([AIMessage(content="Hi.")])
        agent = svc._agent
        # Conftest patches create_secondary_model to None, so _summarizer_llm is None.
        assert agent._summarizer_llm is None

        async for _ in svc.astream(Invocation.from_text("Greet me!")):
            pass
        await asyncio.sleep(0)
        record = await agent._chat_history_builder.flush_pending_tasks()

        # Even without a structured LLM, the fallback path still records the turn
        # using the raw query and explanation text.
        assert record is not None
        recap = agent._chat_history_builder._build_summary_messages([record])
        assert "Greet me!" in recap[0].content[0]["text"]
