"""Unit tests for opendatasci.agents._chat_messages."""


from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from opendatasci._utils.mixins import LLMDigestibleMixin
from opendatasci.agents._chat_messages import ChatMessageOrigin, HumanMessageMetadata


class TestHumanMessageMetadataToContent:
    def test_is_llm_digestible(self) -> None:
        assert isinstance(HumanMessageMetadata(), LLMDigestibleMixin)

    def test_renders_origin_and_timestamp(self) -> None:
        metadata = HumanMessageMetadata(
            origin=ChatMessageOrigin.USER,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        assert metadata.to_content() == (
            "<message_metadata><origin>user</origin>"
            "<timestamp>2024-06-01T00:00:00+00:00</timestamp></message_metadata>"
        )

    def test_falls_back_to_unspecified_and_now_when_unset(self) -> None:
        content = HumanMessageMetadata().to_content()
        assert "<origin>unspecified</origin>" in content


class TestHumanMessageMetadataFromMessage:
    def test_unset_message_parses_to_all_defaults(self) -> None:
        metadata = HumanMessageMetadata.from_message(HumanMessage(content="hi"))
        assert metadata.origin is None
        assert metadata.created_at is None
        assert metadata.is_input_on_interrupt is False

    def test_parses_existing_additional_kwargs(self) -> None:
        message = HumanMessage(
            content="hi",
            additional_kwargs={
                "origin": "user",
                "created_at": "2024-06-01T00:00:00+00:00",
                "is_input_on_interrupt": True,
            },
        )
        metadata = HumanMessageMetadata.from_message(message)
        assert metadata.origin is ChatMessageOrigin.USER
        assert metadata.created_at == datetime(2024, 6, 1, tzinfo=timezone.utc)
        assert metadata.is_input_on_interrupt is True


class TestHumanMessageMetadataAttachTo:
    def test_returns_new_message_by_default(self) -> None:
        original = HumanMessage(content="hi")
        metadata = HumanMessageMetadata(origin=ChatMessageOrigin.USER)
        result = metadata.attach_to(original)
        assert result is not original
        assert original.additional_kwargs == {}

    def test_applied_kwargs_round_trip(self) -> None:
        metadata = HumanMessageMetadata(
            origin=ChatMessageOrigin.HARNESS,
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            is_input_on_interrupt=True,
        )
        result = metadata.attach_to(HumanMessage(content="hi"))
        assert HumanMessageMetadata.from_message(result) == metadata

    def test_excludes_unset_fields(self) -> None:
        metadata = HumanMessageMetadata(origin=ChatMessageOrigin.USER)
        result = metadata.attach_to(HumanMessage(content="hi"))
        assert "created_at" not in result.additional_kwargs

    def test_in_place_mutates_and_returns_same_message(self) -> None:
        original = HumanMessage(content="hi")
        metadata = HumanMessageMetadata(origin=ChatMessageOrigin.USER)
        result = metadata.attach_to(original, in_place=True)
        assert result is original
        assert original.additional_kwargs["origin"] == "user"
