"""Unit tests for opendatasci.memory.chat_memory and opendatasci.agents.chat_history.

Covers ChatTurnSummary rendering, ChatTurnSummarizer fallback path,
ChatHistoryBuilder context assembly, and ChatHistoryCompactor.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from opendatasci._utils.mixins import LLMDigestibleMixin
from opendatasci.memory.messages import (
    AgentMessage,
    HarnessMessage,
    PlanMessage,
    SummaryMessage,
    UserMessage,
)
from opendatasci.memory.chat_memory import (
    ChatHistoryCompaction,
    ChatHistoryCompactor,
    ChatTurnContext,
    ChatTurnSummary,
    ChatTurnSummarizer,
)
from opendatasci.agents.chat_history import ChatHistoryBuilder
from opendatasci.context.plans import Plan

_TS1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
_TS2 = datetime(2024, 6, 2, tzinfo=timezone.utc)


def _completed_turn(query: str = "my query", answer: str = "my answer") -> list[BaseMessage]:
    return [
        UserMessage(content=query, created_at=_TS1),
        AgentMessage(content=answer),
    ]


def _summary(
    user: str = "q",
    agent: str = "a",
    actions: str = "did something",
    ts_start: datetime = _TS1,
    ts_end: datetime = _TS2,
) -> ChatTurnSummary:
    return ChatTurnSummary(
        turn_start_timestamp=ts_start,
        turn_end_timestamp=ts_end,
        user_message_summary=user,
        actions_summary=actions,
        agent_response_summary=agent,
    )


# ---------------------------------------------------------------------------
# ChatTurnSummary
# ---------------------------------------------------------------------------


class TestChatTurnSummary:
    def test_is_llm_digestible(self) -> None:
        assert isinstance(_summary(), LLMDigestibleMixin)

    def test_to_content_includes_all_fields(self) -> None:
        s = _summary(user="What is X?", agent="X is Y.", actions="called tool A")
        content = s.to_content()
        assert "What is X?" in content
        assert "X is Y." in content
        assert "called tool A" in content

    def test_to_content_has_expected_structure(self) -> None:
        content = _summary().to_content()
        assert "<summary_content>" in content
        assert "<user_request>" in content
        assert "<outcomes>" in content
        assert "<agent_response>" in content

    def test_to_content_does_not_include_timestamps(self) -> None:
        content = _summary().to_content()
        assert "<summary_metadata>" not in content


# ---------------------------------------------------------------------------
# ChatHistoryCompaction
# ---------------------------------------------------------------------------


class TestChatHistoryCompaction:
    def test_is_llm_digestible(self) -> None:
        c = ChatHistoryCompaction(compacted_at=_TS1, timespan=(_TS1, _TS2), content="folded")
        assert isinstance(c, LLMDigestibleMixin)

    def test_to_content_includes_content(self) -> None:
        c = ChatHistoryCompaction(compacted_at=_TS1, timespan=(_TS1, _TS2), content="folded history")
        assert "folded history" in c.to_content()

    def test_to_content_handles_none_timespan(self) -> None:
        c = ChatHistoryCompaction(compacted_at=_TS1, timespan=None, content="x")
        assert "None" in c.to_content()


# ---------------------------------------------------------------------------
# ChatTurnSummarizer (fallback path: no LLM)
# ---------------------------------------------------------------------------


class TestChatTurnSummarizer:
    async def test_fallback_returns_summary_with_text_content(self) -> None:
        s = ChatTurnSummarizer(summarizer_llm=None)
        record = await s.summarize_turn(_completed_turn("the query", "the answer"))
        assert record is not None
        assert record.user_message_summary == "the query"
        assert record.agent_response_summary == "the answer"

    async def test_fallback_sets_actions_to_na(self) -> None:
        s = ChatTurnSummarizer(summarizer_llm=None)
        record = await s.summarize_turn(_completed_turn())
        assert record.actions_summary == "N/A"

    async def test_raises_for_empty_turn(self) -> None:
        s = ChatTurnSummarizer(summarizer_llm=None)
        with pytest.raises(ValueError, match="empty"):
            await s.summarize_turn([])

    async def test_reads_timestamps_from_turn(self) -> None:
        s = ChatTurnSummarizer(summarizer_llm=None)
        turn = [
            UserMessage(content="q", created_at=_TS1),
            AgentMessage(content="a"),
        ]
        record = await s.summarize_turn(turn)
        assert record is not None
        assert record.turn_start_timestamp == _TS1


# ---------------------------------------------------------------------------
# ChatHistoryBuilder
# ---------------------------------------------------------------------------


class TestChatHistoryBuilder:
    def _builder(self, window_size: int = 3, **kwargs) -> ChatHistoryBuilder:
        return ChatHistoryBuilder(summarizer_llm=None, window_size=window_size, **kwargs)

    async def test_build_no_summaries_renders_ongoing_turn_only(self) -> None:
        builder = self._builder()
        ctx = await builder.build([UserMessage(content="hi")], [], None)
        assert isinstance(ctx, ChatTurnContext)
        assert ctx.turn_summaries == []
        assert len(ctx.messages) == 1
        assert "hi" in ctx.messages[0].content

    async def test_build_flushes_and_prepends_recap_messages(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn("first query", "first answer"))
        ctx = await builder.build([UserMessage(content="second")], [], None)
        assert len(ctx.turn_summaries) == 1
        assert len(ctx.messages) == 2
        assert "first query" in ctx.messages[0].content
        assert "second" in ctx.messages[1].content

    async def test_messages_are_rendered_with_metadata_tag(self) -> None:
        builder = self._builder()
        ctx = await builder.build([UserMessage(content="hi")], [], None)
        assert "<message_metadata>" in ctx.messages[0].content

    async def test_all_messages_are_human_messages(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn("q", "a"))
        ctx = await builder.build([UserMessage(content="next")], [], None)
        assert not any(isinstance(m, SystemMessage) for m in ctx.messages)
        assert all(isinstance(m, HumanMessage) for m in ctx.messages)

    async def test_flush_pending_tasks_no_task_is_none(self) -> None:
        builder = self._builder()
        assert await builder.flush_pending_tasks() is None

    async def test_cancel_pending_tasks_discards_summary(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn())
        builder.cancel_pending_tasks()
        ctx = await builder.build([UserMessage(content="x")], [], None)
        assert ctx.turn_summaries == []
        assert len(ctx.messages) == 1

    async def test_flush_pending_tasks_swallows_failure(self) -> None:
        builder = self._builder()

        async def boom() -> ChatTurnSummary:
            raise RuntimeError("boom")

        builder._pending_task = asyncio.create_task(boom())
        assert await builder.flush_pending_tasks() is None

    async def test_schedule_ongoing_turn_raises(self) -> None:
        builder = self._builder()
        ongoing = [
            UserMessage(content="q"),
            AgentMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ]
        with pytest.raises(ValueError, match="ongoing"):
            builder.schedule_turn_summarization(ongoing)
        assert builder._pending_task is None

    async def test_schedule_empty_history_raises(self) -> None:
        builder = self._builder()
        with pytest.raises(ValueError, match="empty"):
            builder.schedule_turn_summarization([])

    async def test_plan_message_appended_after_recap_before_ongoing_turn(self) -> None:
        plan = Plan(content="do the thing", metadata={})
        context_store = MagicMock()
        context_store.get_current_plan.return_value = plan
        builder = self._builder(context_store=context_store, session_id="s1")
        builder.schedule_turn_summarization(_completed_turn())
        ctx = await builder.build([UserMessage(content="x")], [], None)
        assert len(ctx.messages) == 3
        assert "current_plan" in ctx.messages[1].content
        assert "x" in ctx.messages[2].content

    async def test_no_plan_message_when_context_store_returns_none(self) -> None:
        context_store = MagicMock()
        context_store.get_current_plan.return_value = None
        builder = self._builder(context_store=context_store, session_id="s1")
        ctx = await builder.build([UserMessage(content="x")], [], None)
        assert len(ctx.messages) == 1

    async def test_no_plan_message_without_context_store(self) -> None:
        builder = self._builder()
        ctx = await builder.build([UserMessage(content="x")], [], None)
        assert len(ctx.messages) == 1

    async def test_compaction_message_included_when_below_window(self) -> None:
        builder = self._builder(window_size=5)
        compaction = ChatHistoryCompaction(compacted_at=_TS1, timespan=(_TS1, _TS2), content="folded")
        ctx = await builder.build([UserMessage(content="x")], [], compaction)
        assert len(ctx.messages) == 2
        assert "folded" in ctx.messages[0].content

    async def test_compaction_cleared_when_window_full(self) -> None:
        builder = self._builder(window_size=2)
        summaries = [_summary(), _summary()]
        compaction = ChatHistoryCompaction(compacted_at=_TS1, timespan=None, content="folded")
        ctx = await builder.build([UserMessage(content="x")], summaries, compaction)
        assert ctx.chat_history_compaction is None

    async def test_summary_window_trims_oldest(self) -> None:
        builder = self._builder(window_size=2)
        summaries = [_summary(user=f"q{i}") for i in range(5)]
        ctx = await builder.build([UserMessage(content="x")], summaries, None)
        assert len(ctx.turn_summaries) == 2
        rendered = " ".join(m.content for m in ctx.messages)
        assert "q3" in rendered or "q4" in rendered
        assert "q0" not in rendered


# ---------------------------------------------------------------------------
# ChatHistoryBuilder._build_plan_message
# ---------------------------------------------------------------------------


class TestBuildPlanMessage:
    _builder = ChatHistoryBuilder(summarizer_llm=None)

    def test_wraps_content_in_current_plan_tags(self) -> None:
        plan = Plan(content="Step 1: do X", metadata={})
        message = self._builder._build_plan_message(plan)
        assert isinstance(message, PlanMessage)
        assert "<current_plan>" in message.content
        assert "Step 1: do X" in message.content

    def test_returns_plan_message_type(self) -> None:
        message = self._builder._build_plan_message(Plan(content="x", metadata={}))
        assert isinstance(message, PlanMessage)

    def test_reuses_created_at_metadata_as_timestamp(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        message = self._builder._build_plan_message(
            Plan(content="x", metadata={"created_at": "2024-06-01T00:00:00+00:00"})
        )
        assert message.created_at == ts


# ---------------------------------------------------------------------------
# ChatHistoryBuilder._build_summary_messages
# ---------------------------------------------------------------------------


class TestBuildSummaryMessages:
    _builder = ChatHistoryBuilder(summarizer_llm=None)

    def test_empty_renders_nothing(self) -> None:
        assert self._builder._build_summary_messages([]) == []

    def test_one_message_per_summary(self) -> None:
        result = self._builder._build_summary_messages([_summary()])
        assert len(result) == 1
        assert isinstance(result[0], SummaryMessage)

    def test_content_contains_summary_text(self) -> None:
        result = self._builder._build_summary_messages([_summary(user="What is X?", agent="X is Y.")])
        assert "What is X?" in result[0].content
        assert "X is Y." in result[0].content

    def test_returns_summary_message_type(self) -> None:
        result = self._builder._build_summary_messages([_summary()])
        assert isinstance(result[0], SummaryMessage)

    def test_summaries_kept_in_order(self) -> None:
        result = self._builder._build_summary_messages([_summary(user="first"), _summary(user="second")])
        assert "first" in result[0].content
        assert "second" in result[1].content


# ---------------------------------------------------------------------------
# ChatHistoryCompactor
# ---------------------------------------------------------------------------


class TestChatHistoryCompactor:
    async def test_raises_when_nothing_to_compact(self) -> None:
        compactor = ChatHistoryCompactor(llm=None)
        with pytest.raises(ValueError, match="Nothing to compact"):
            await compactor.compact(None, [], [])

    async def test_returns_compaction_with_content(self) -> None:
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="folded summary"))
        compactor = ChatHistoryCompactor(llm=llm)

        result = await compactor.compact(None, [_summary()], [])

        assert isinstance(result, ChatHistoryCompaction)
        assert result.content == "folded summary"
        assert result.timespan is not None

    async def test_compaction_timespan_spans_all_summaries(self) -> None:
        ts_early = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts_late = datetime(2024, 12, 31, tzinfo=timezone.utc)
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="x"))
        compactor = ChatHistoryCompactor(llm=llm)
        summaries = [
            _summary(ts_start=ts_early, ts_end=_TS1),
            _summary(ts_start=_TS1, ts_end=ts_late),
        ]
        result = await compactor.compact(None, summaries, [])
        assert result.timespan is not None
        assert result.timespan[0] == ts_early
        assert result.timespan[1] == ts_late
