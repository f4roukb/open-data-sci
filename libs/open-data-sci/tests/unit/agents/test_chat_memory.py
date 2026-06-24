"""Unit tests for opendatasci.agents.chat_memory.

Covers message provenance tagging/rendering, ChatHistoryBuilder, ChatTurnContext,
ChatTurnSummary/build_chat_recap_messages rendering, and the ChatTurnSummarizer
fallback path.
"""


import asyncio
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from opendatasci._utils.mixins import LLMDigestibleMixin
from opendatasci.agents.chat_memory import (
    ChatHistoryBuilder,
    ChatHistoryCompactor,
    ChatMessageOrigin,
    ChatTurnContext,
    ChatTurnSummarizer,
    ChatTurnSummary,
    build_chat_recap_messages,
    build_plan_message,
    render_messages_for_llm,
    stamp_chat_message_metadata,
)
from opendatasci.context.plans import Plan


def _completed_turn(query: str = "my query", answer: str = "my answer") -> list[BaseMessage]:
    return [
        HumanMessage(content=query, additional_kwargs={"is_input_on_interrupt": False}),
        AIMessage(content=answer),
    ]


def _record(turn: int | None, user: str = "q", agent: str = "a", **kw: str) -> ChatTurnSummary:
    return ChatTurnSummary(turn=turn, user=user, actions=kw.get("actions", ""), agent=agent,
                            timestamp=kw.get("timestamp", ""))


# ---------------------------------------------------------------------------
# Message provenance: stamping (additional_kwargs) and rendering (content)
# ---------------------------------------------------------------------------


class TestStampChatMessageMetadata:
    def test_stamps_origin_and_timestamp(self) -> None:
        message = stamp_chat_message_metadata(HumanMessage(content="hi"), ChatMessageOrigin.USER)
        assert message.additional_kwargs["origin"] == "user"
        assert message.additional_kwargs["created_at"]

    def test_does_not_mutate_original_in_place(self) -> None:
        original = HumanMessage(content="hi")
        stamped = stamp_chat_message_metadata(original, ChatMessageOrigin.USER)
        assert original.additional_kwargs == {}
        assert stamped is not original

    def test_idempotent_keeps_existing_values(self) -> None:
        once = stamp_chat_message_metadata(HumanMessage(content="hi"), ChatMessageOrigin.USER)
        twice = stamp_chat_message_metadata(once, ChatMessageOrigin.HARNESS)
        # Re-stamping with a different default origin/timestamp is a no-op.
        assert twice.additional_kwargs == once.additional_kwargs
        assert twice is once

    def test_explicit_timestamp_is_used_when_provided(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        message = stamp_chat_message_metadata(
            HumanMessage(content="hi"), ChatMessageOrigin.HARNESS, created_at=ts
        )
        assert message.additional_kwargs["created_at"] == "2024-06-01T00:00:00Z"

    def test_content_is_untouched(self) -> None:
        message = stamp_chat_message_metadata(HumanMessage(content="hi"), ChatMessageOrigin.USER)
        assert message.content == "hi"


class TestRenderMessagesForLlm:
    def test_renders_tag_into_content(self) -> None:
        ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
        message = stamp_chat_message_metadata(
            HumanMessage(content="hi"), ChatMessageOrigin.USER, created_at=ts
        )
        rendered = render_messages_for_llm([message])
        assert rendered[0].content == (
            "<message_metadata><origin>user</origin>"
            "<timestamp>2024-06-01T00:00:00+00:00</timestamp></message_metadata>\nhi"
        )

    def test_unstamped_message_falls_back_to_unspecified(self) -> None:
        rendered = render_messages_for_llm([HumanMessage(content="hi")])
        assert "<origin>unspecified</origin>" in rendered[0].content

    def test_does_not_mutate_original(self) -> None:
        original = HumanMessage(content="hi")
        render_messages_for_llm([original])
        assert original.content == "hi"

    def test_non_human_messages_pass_through_unchanged(self) -> None:
        ai = AIMessage(content="answer")
        sys = SystemMessage(content="sys")
        rendered = render_messages_for_llm([ai, sys])
        assert rendered == [ai, sys]

    def test_mixed_list_only_renders_human_messages(self) -> None:
        human = stamp_chat_message_metadata(HumanMessage(content="hi"), ChatMessageOrigin.USER)
        ai = AIMessage(content="answer")
        rendered = render_messages_for_llm([human, ai])
        assert rendered[1] is ai
        assert "<message_metadata>" in rendered[0].content


class TestBuildPlanMessage:
    def test_wraps_content_in_current_plan_tags(self) -> None:
        plan = Plan(content="Step 1: do X", metadata={})
        message = build_plan_message(plan)
        assert isinstance(message, HumanMessage)
        assert message.content == "<current_plan>\nStep 1: do X\n</current_plan>"

    def test_stamped_as_harness(self) -> None:
        message = build_plan_message(Plan(content="x", metadata={}))
        assert message.additional_kwargs["origin"] == "harness"

    def test_reuses_created_at_metadata_as_timestamp(self) -> None:
        message = build_plan_message(
            Plan(content="x", metadata={"created_at": "2024-06-01T00:00:00+00:00"})
        )
        assert message.additional_kwargs["created_at"] == "2024-06-01T00:00:00Z"


# ---------------------------------------------------------------------------
# build_chat_recap_messages / ChatTurnSummary
# ---------------------------------------------------------------------------


class TestChatTurnSummaryIsLLMDigestible:
    def test_is_instance_of_mixin(self) -> None:
        assert isinstance(_record(1), LLMDigestibleMixin)

    def test_to_content_used_by_recap(self) -> None:
        record = _record(1, user="q", agent="a")
        assert record.to_content() in build_chat_recap_messages([record])[0].content

    def test_compaction_summary_uses_distinct_header(self) -> None:
        record = ChatTurnSummary(turn=None, user="", actions="", agent="folded text", timestamp="")
        assert record.to_content().startswith("**Compacted Summary:**")
        assert "folded text" in record.to_content()


class TestBuildChatRecapMessages:
    def test_empty_renders_nothing(self) -> None:
        assert build_chat_recap_messages([]) == []

    def test_one_message_per_summary(self) -> None:
        result = build_chat_recap_messages(
            [_record(1, user="What is X?", agent="X is Y.", actions="called tool A")]
        )
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert "Turn 1" in result[0].content
        assert "What is X?" in result[0].content and "X is Y." in result[0].content
        assert "called tool A" in result[0].content

    def test_summaries_kept_in_list_order(self) -> None:
        result = build_chat_recap_messages([_record(1), _record(2)])
        assert len(result) == 2
        assert "Turn 1" in result[0].content
        assert "Turn 2" in result[1].content

    def test_compaction_summary_is_just_another_entry(self) -> None:
        compaction = ChatTurnSummary(turn=None, user="", actions="", agent="folded", timestamp="")
        result = build_chat_recap_messages([compaction, _record(5)])
        assert len(result) == 2
        assert "Compacted Summary" in result[0].content
        assert "Turn 5" in result[1].content

    def test_timestamp_rendered(self) -> None:
        result = build_chat_recap_messages([_record(1, timestamp="2024-06-01T12:00:00+00:00")])
        assert "2024-06-01" in result[0].content

    def test_multiline_actions_indented(self) -> None:
        result = build_chat_recap_messages([_record(1, actions="line1\nline2")])
        assert "- Outcomes:\n  line1\n  line2" in result[0].content

    def test_stamped_with_harness_origin_and_summary_timestamp(self) -> None:
        result = build_chat_recap_messages([_record(1, timestamp="2024-06-01T00:00:00+00:00")])
        assert result[0].additional_kwargs["origin"] == "harness"
        assert result[0].additional_kwargs["created_at"] == "2024-06-01T00:00:00Z"


# ---------------------------------------------------------------------------
# ChatTurnSummarizer (LLM-less fallback)
# ---------------------------------------------------------------------------


class TestChatTurnSummarizer:
    async def test_summarize_completed_turn_fallback(self) -> None:
        s = ChatTurnSummarizer(summarizer_llm=None)
        record = await s.summarize_turn(_completed_turn("the query", "the answer"))
        assert record is not None
        assert record.turn == 0  # caller assigns the running index
        assert record.user == "the query"
        assert record.agent == "the answer"
        assert record.actions == "(summary unavailable)"

    async def test_summarize_empty_turn_is_none(self) -> None:
        s = ChatTurnSummarizer(summarizer_llm=None)
        assert await s.summarize_turn([]) is None

    async def test_summarize_reads_timestamp(self) -> None:
        s = ChatTurnSummarizer(summarizer_llm=None)
        turn = [
            HumanMessage(content="q", additional_kwargs={"created_at": "2024-06-01T00:00:00+00:00"}),
            AIMessage(content="a"),
        ]
        record = await s.summarize_turn(turn)
        assert record is not None and record.timestamp == "2024-06-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# ChatHistoryBuilder
# ---------------------------------------------------------------------------


class TestChatHistoryBuilder:
    def _builder(self, window_size: int = 3, **kwargs) -> ChatHistoryBuilder:
        return ChatHistoryBuilder(summarizer_llm=None, window_size=window_size, **kwargs)

    async def test_build_no_summaries_renders_ongoing_turn_only(self) -> None:
        builder = self._builder()
        ctx = await builder.build([HumanMessage(content="hi")], [])
        assert isinstance(ctx, ChatTurnContext)
        assert ctx.turn_summaries == []
        assert len(ctx.messages) == 1
        assert "hi" in ctx.messages[0].content

    async def test_build_flushes_and_prepends_recap_messages(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn("first query", "first answer"))
        ctx = await builder.build([HumanMessage(content="second")], [])
        assert len(ctx.turn_summaries) == 1
        assert ctx.turn_summaries[0].turn == 1
        # One recap HumanMessage prepended, then the ongoing turn's own message.
        assert len(ctx.messages) == 2
        assert "first query" in ctx.messages[0].content
        assert "second" in ctx.messages[1].content

    async def test_messages_never_contain_system_messages(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn("q", "a"))
        ctx = await builder.build([HumanMessage(content="next")], [])
        assert not any(isinstance(m, SystemMessage) for m in ctx.messages)
        assert all(isinstance(m, HumanMessage) for m in ctx.messages)

    async def test_messages_are_rendered_with_metadata_tag(self) -> None:
        builder = self._builder()
        ctx = await builder.build([HumanMessage(content="hi")], [])
        assert "<message_metadata>" in ctx.messages[0].content

    async def test_turn_numbers_are_monotonic_with_eviction(self) -> None:
        builder = self._builder(window_size=2)
        summaries: list[ChatTurnSummary] = []
        for i in range(5):
            builder.schedule_turn_summarization(_completed_turn(f"q{i}", f"a{i}"))
            ctx = await builder.build([HumanMessage(content=f"q{i}")], summaries)
            summaries = ctx.turn_summaries

        assert [r.turn for r in summaries] == [4, 5]  # absolute numbering preserved past eviction
        recap = build_chat_recap_messages(summaries)
        rendered = "\n\n".join(m.content for m in recap)
        assert "Turn 4" in rendered and "Turn 5" in rendered
        assert "q0" not in rendered and "Turn 3" not in rendered

    async def test_turn_numbering_skips_compaction_summary(self) -> None:
        builder = self._builder()
        compaction = ChatTurnSummary(turn=None, user="", actions="", agent="folded", timestamp="")
        builder.schedule_turn_summarization(_completed_turn("q", "a"))
        ctx = await builder.build([HumanMessage(content="q")], [compaction])
        assert ctx.turn_summaries[0].turn is None
        assert ctx.turn_summaries[1].turn == 1

    async def test_flush_no_task_is_none(self) -> None:
        builder = self._builder()
        assert await builder.flush() is None

    async def test_cancel_pending_discards_summary(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn())
        builder.cancel_pending()
        ctx = await builder.build([HumanMessage(content="x")], [])
        assert ctx.turn_summaries == []
        assert len(ctx.messages) == 1

    async def test_flush_swallows_failure(self) -> None:
        builder = self._builder()

        async def boom() -> ChatTurnSummary:
            raise RuntimeError("boom")

        builder._pending_task = asyncio.create_task(boom())
        assert await builder.flush() is None  # must not raise

    async def test_schedule_ongoing_turn_raises(self) -> None:
        builder = self._builder()
        ongoing = [
            HumanMessage(content="q", additional_kwargs={"is_input_on_interrupt": False}),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ]
        with pytest.raises(ValueError, match="ongoing"):
            builder.schedule_turn_summarization(ongoing)
        assert builder._pending_task is None

    async def test_schedule_empty_history_is_noop(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization([])
        assert builder._pending_task is None

    async def test_plan_message_appended_after_recap_before_ongoing_turn(self) -> None:
        plan = Plan(content="do the thing", metadata={})
        builder = self._builder(get_current_plan=lambda: plan)
        ctx = await builder.build([HumanMessage(content="x")], [_record(1)])
        assert len(ctx.messages) == 3
        assert "Turn 1" in ctx.messages[0].content
        assert "current_plan" in ctx.messages[1].content
        assert "x" in ctx.messages[2].content

    async def test_no_plan_message_when_get_current_plan_returns_none(self) -> None:
        builder = self._builder(get_current_plan=lambda: None)
        ctx = await builder.build([HumanMessage(content="x")], [])
        assert len(ctx.messages) == 1

    async def test_no_plan_message_without_get_current_plan(self) -> None:
        builder = self._builder()
        ctx = await builder.build([HumanMessage(content="x")], [])
        assert len(ctx.messages) == 1


# ---------------------------------------------------------------------------
# ChatHistoryCompactor
# ---------------------------------------------------------------------------


class TestChatHistoryCompactor:
    async def test_raises_when_nothing_to_compact(self) -> None:
        compactor = ChatHistoryCompactor(llm=None)
        with pytest.raises(ValueError, match="Nothing to compact"):
            await compactor.compact([])

    async def test_returns_compaction_summary_with_turn_none(self) -> None:
        from unittest.mock import AsyncMock

        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="folded summary"))
        compactor = ChatHistoryCompactor(llm=llm)

        result = await compactor.compact([_record(1, user="q1", agent="a1")])

        assert result.turn is None
        assert result.agent == "folded summary"
        assert result.user == ""
        assert result.actions == ""
        assert result.timestamp
