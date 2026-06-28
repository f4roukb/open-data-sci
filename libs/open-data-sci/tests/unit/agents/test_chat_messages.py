"""Unit tests for opendatasci.memory.messages typed subtypes."""

from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from opendatasci._utils.mixins import RenderableMessageMixin
from opendatasci.memory.messages import (
    AgentMessage,
    HarnessMessage,
    PlanMessage,
    SummaryMessage,
    UserMessage,
    is_user_message,
)

_TS = datetime(2024, 6, 1, tzinfo=timezone.utc)


class TestUserMessage:
    def test_is_subtype_of_human_message(self) -> None:
        assert isinstance(UserMessage(content="hi"), HumanMessage)

    def test_is_renderable(self) -> None:
        assert isinstance(UserMessage(content="hi"), RenderableMessageMixin)

    def test_is_user_message(self) -> None:
        assert is_user_message(UserMessage(content="hi"))

    def test_render_returns_user_message(self) -> None:
        msg = UserMessage(content="hi", created_at=_TS)
        assert isinstance(msg.render(), UserMessage)

    def test_render_includes_user_origin_tag(self) -> None:
        msg = UserMessage(content="hi", created_at=_TS)
        assert "<origin>user</origin>" in msg.render().content

    def test_render_includes_timestamp(self) -> None:
        msg = UserMessage(content="hi", created_at=_TS)
        assert "2024-06-01" in msg.render().content

    def test_render_preserves_original_text_in_content(self) -> None:
        msg = UserMessage(content="hello world", created_at=_TS)
        assert "hello world" in msg.render().content

    def test_render_does_not_mutate_original(self) -> None:
        msg = UserMessage(content="hi")
        msg.render()
        assert msg.content == "hi"


class TestHarnessMessage:
    def test_is_subtype_of_human_message(self) -> None:
        assert isinstance(HarnessMessage(content="ctx"), HumanMessage)

    def test_is_not_user_message(self) -> None:
        assert not is_user_message(HarnessMessage(content="ctx"))

    def test_render_returns_harness_message(self) -> None:
        msg = HarnessMessage(content="ctx", created_at=_TS)
        assert isinstance(msg.render(), HarnessMessage)

    def test_render_includes_harness_origin_tag(self) -> None:
        msg = HarnessMessage(content="ctx", created_at=_TS)
        assert "<origin>harness</origin>" in msg.render().content

    def test_render_does_not_mutate_original(self) -> None:
        msg = HarnessMessage(content="ctx")
        msg.render()
        assert msg.content == "ctx"


_TS_START = datetime(2024, 5, 31, tzinfo=timezone.utc)
_TS_END = datetime(2024, 6, 1, tzinfo=timezone.utc)


def _summary_msg(content: str = "summary") -> SummaryMessage:
    return SummaryMessage(
        content=content,
        created_at=_TS_END,
        turn_start_timestamp=_TS_START,
        turn_end_timestamp=_TS_END,
    )


class TestSummaryMessage:
    def test_is_subtype_of_human_message(self) -> None:
        assert isinstance(_summary_msg(), HumanMessage)

    def test_render_returns_summary_message(self) -> None:
        assert isinstance(_summary_msg().render(), SummaryMessage)

    def test_render_includes_harness_origin_tag(self) -> None:
        assert "<origin>harness</origin>" in _summary_msg().render().content

    def test_render_includes_summary_metadata_block(self) -> None:
        assert "<summary_metadata>" in _summary_msg().render().content

    def test_render_includes_turn_start_timestamp(self) -> None:
        assert "2024-05-31" in _summary_msg().render().content

    def test_render_includes_turn_end_timestamp(self) -> None:
        assert "2024-06-01" in _summary_msg().render().content

    def test_render_preserves_content(self) -> None:
        assert "my summary text" in _summary_msg("my summary text").render().content

    def test_render_does_not_mutate_original(self) -> None:
        msg = _summary_msg("original")
        msg.render()
        assert msg.content == "original"


class TestPlanMessage:
    def test_is_subtype_of_human_message(self) -> None:
        assert isinstance(PlanMessage(content="plan"), HumanMessage)

    def test_render_returns_plan_message(self) -> None:
        msg = PlanMessage(content="plan", created_at=_TS)
        assert isinstance(msg.render(), PlanMessage)

    def test_render_includes_harness_origin_tag(self) -> None:
        msg = PlanMessage(content="p", created_at=_TS)
        assert "<origin>harness</origin>" in msg.render().content


class TestAgentMessage:
    def test_is_subtype_of_ai_message(self) -> None:
        assert isinstance(AgentMessage(content="response"), AIMessage)

    def test_from_langchain_preserves_content(self) -> None:
        raw = AIMessage(content="hello")
        msg = AgentMessage.from_langchain(raw)
        assert isinstance(msg, AgentMessage)
        assert msg.content == "hello"

    def test_from_langchain_preserves_tool_calls(self) -> None:
        raw = AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {"x": 1}, "id": "abc"}],
        )
        msg = AgentMessage.from_langchain(raw)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "t"
        assert msg.tool_calls[0]["args"] == {"x": 1}

    def test_created_at_defaults_to_now(self) -> None:
        msg = AgentMessage(content="resp")
        assert msg.created_at is not None


class TestIsUserMessage:
    def test_user_message_returns_true(self) -> None:
        assert is_user_message(UserMessage(content="hi"))

    def test_harness_message_returns_false(self) -> None:
        assert not is_user_message(HarnessMessage(content="ctx"))

    def test_ai_message_returns_false(self) -> None:
        assert not is_user_message(AIMessage(content="resp"))

    def test_plain_human_message_returns_false(self) -> None:
        assert not is_user_message(HumanMessage(content="hi"))
