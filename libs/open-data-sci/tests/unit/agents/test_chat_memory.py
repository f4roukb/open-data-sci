"""Unit tests for opendatasci.agents.chat_memory.

Covers ChatHistoryBuilder, PreparedHistory, TurnSummaryRecord/render_memory
rendering, and the TurnSummarizer fallback path.
"""


import asyncio

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from opendatasci.agents.chat_memory import (
    ChatHistoryBuilder,
    PreparedHistory,
    TurnSummarizer,
    TurnSummaryRecord,
    render_memory,
)


def _completed_turn(query: str = "my query", answer: str = "my answer") -> list[BaseMessage]:
    return [
        HumanMessage(content=query, additional_kwargs={"is_input_on_interrupt": False}),
        AIMessage(content=answer),
    ]


def _record(turn: int, user: str = "q", agent: str = "a", **kw: str) -> TurnSummaryRecord:
    return TurnSummaryRecord(turn=turn, user=user, actions=kw.get("actions", ""), agent=agent,
                             timestamp=kw.get("timestamp", ""))


# ---------------------------------------------------------------------------
# render_memory / TurnSummaryRecord
# ---------------------------------------------------------------------------


class TestRenderMemory:
    def test_empty_renders_nothing(self) -> None:
        assert render_memory(None, []) == ""

    def test_preamble_only(self) -> None:
        result = render_memory("Prior session covered topic X.", [])
        assert "Previous Session Summary" in result
        assert "Prior session covered topic X." in result
        assert "Recent Conversation History" not in result

    def test_turns_only(self) -> None:
        result = render_memory(None, [_record(1, user="What is X?", agent="X is Y.", actions="called tool A")])
        assert "## Recent Conversation History" in result
        assert "Turn 1" in result
        assert "What is X?" in result and "X is Y." in result and "called tool A" in result

    def test_preamble_and_turns(self) -> None:
        result = render_memory("Summary.", [_record(1)])
        assert "Previous Session Summary" in result
        assert "Recent Conversation History" in result

    def test_timestamp_rendered(self) -> None:
        result = render_memory(None, [_record(1, timestamp="2024-06-01T12:00:00+00:00")])
        assert "2024-06-01" in result

    def test_multiline_actions_indented(self) -> None:
        result = render_memory(None, [_record(1, actions="line1\nline2")])
        assert "- Outcomes:\n  line1\n  line2" in result


# ---------------------------------------------------------------------------
# TurnSummarizer (LLM-less fallback)
# ---------------------------------------------------------------------------


class TestTurnSummarizer:
    async def test_summarize_completed_turn_fallback(self) -> None:
        s = TurnSummarizer(summarizer_llm=None)
        record = await s.summarize_turn(_completed_turn("the query", "the answer"))
        assert record is not None
        assert record.turn == 0  # caller assigns the running index
        assert record.user == "the query"
        assert record.agent == "the answer"
        assert record.actions == "(summary unavailable)"

    async def test_summarize_empty_turn_is_none(self) -> None:
        s = TurnSummarizer(summarizer_llm=None)
        assert await s.summarize_turn([]) is None

    async def test_summarize_reads_timestamp(self) -> None:
        s = TurnSummarizer(summarizer_llm=None)
        turn = [
            HumanMessage(content="q", additional_kwargs={"timestamp": "2024-06-01T00:00:00+00:00"}),
            AIMessage(content="a"),
        ]
        record = await s.summarize_turn(turn)
        assert record is not None and record.timestamp == "2024-06-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# ChatHistoryBuilder
# ---------------------------------------------------------------------------


class TestChatHistoryBuilder:
    def _builder(self, window_size: int = 3) -> ChatHistoryBuilder:
        return ChatHistoryBuilder(summarizer=TurnSummarizer(summarizer_llm=None), window_size=window_size)

    async def test_build_no_summaries_returns_history_unchanged(self) -> None:
        builder = self._builder()
        history = [HumanMessage(content="hi")]
        ph = await builder.build(history, [])
        assert isinstance(ph, PreparedHistory)
        assert ph.turn_summaries == []
        assert ph.messages == history
        assert ph.memory_text is None

    async def test_build_flushes_and_produces_memory_text(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn("first query", "first answer"))
        ph = await builder.build([HumanMessage(content="second")], [])
        assert len(ph.turn_summaries) == 1
        assert ph.turn_summaries[0].turn == 1
        assert ph.memory_text is not None
        assert "first query" in ph.memory_text
        assert ph.messages == [HumanMessage(content="second")]

    async def test_messages_never_contain_system_messages(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn("q", "a"))
        ph = await builder.build([HumanMessage(content="next")], [])
        assert not any(isinstance(m, SystemMessage) for m in ph.messages)

    async def test_turn_numbers_are_monotonic_with_eviction(self) -> None:
        builder = self._builder(window_size=2)
        summaries: list[TurnSummaryRecord] = []
        for i in range(5):
            builder.schedule_turn_summarization(_completed_turn(f"q{i}", f"a{i}"))
            ph = await builder.build([HumanMessage(content=f"q{i}")], summaries)
            summaries = ph.turn_summaries

        assert [r.turn for r in summaries] == [4, 5]  # absolute numbering preserved past eviction
        rendered = render_memory(None, summaries)
        assert "Turn 4" in rendered and "Turn 5" in rendered
        assert "q0" not in rendered and "Turn 3" not in rendered

    async def test_flush_no_task_is_none(self) -> None:
        builder = self._builder()
        assert await builder.flush() is None

    async def test_cancel_pending_discards_summary(self) -> None:
        builder = self._builder()
        builder.schedule_turn_summarization(_completed_turn())
        builder.cancel_pending()
        ph = await builder.build([HumanMessage(content="x")], [])
        assert ph.turn_summaries == []
        assert ph.memory_text is None

    async def test_flush_swallows_failure(self) -> None:
        builder = self._builder()

        async def boom() -> TurnSummaryRecord:
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

    async def test_preamble_flows_into_memory_text(self) -> None:
        builder = self._builder()
        ph = await builder.build([HumanMessage(content="x")], [_record(1)], preamble="carried over")
        assert ph.memory_text is not None
        assert "carried over" in ph.memory_text
        assert "Previous Session Summary" in ph.memory_text
        assert not any(isinstance(m, SystemMessage) for m in ph.messages)
