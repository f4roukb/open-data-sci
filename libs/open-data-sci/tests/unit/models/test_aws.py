"""Unit tests for opendatasci.models.aws factory functions.

Covers create_bedrock_model / create_bedrock_secondary_model, the error paths that
fire when langchain-aws is unavailable, and the streaming-fix regressions
(disable_streaming override and _strip_list_usage_fields for Bedrock's
list-valued cacheDetails usage field).
"""


from unittest.mock import patch

import pytest

pytest.importorskip("langchain_aws")

import opendatasci.models.aws as aws_module
from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.aws import _strip_list_usage_fields, create_bedrock_model
from opendatasci.prompts.caching import cached_system_prompt


def _create_model(model_id: str):
    config = OpenDataSciConfig(provider="bedrock", model=model_id)  # type: ignore[arg-type]
    return create_bedrock_model(config)


class TestCachedSystemPrompt:
    def test_appends_default_cache_point(self) -> None:
        content = cached_system_prompt("system prompt body")
        assert content == [
            {"type": "text", "text": "system prompt body"},
            {"cachePoint": {"type": "default"}},
        ]


class TestCreateBedrockSecondaryModel:
    def test_returns_bedrock_base_instance(self) -> None:
        # Secondary model uses the plain _BedrockBase (no streaming override needed).
        model = aws_module.create_bedrock_secondary_model(
            OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
        )
        assert isinstance(model, aws_module._BedrockBase)

    def test_uses_resolved_secondary_model(self) -> None:
        config = OpenDataSciConfig(provider="bedrock", secondary_model="us.anthropic.fake-haiku")  # type: ignore[arg-type]
        model = aws_module.create_bedrock_secondary_model(config)
        assert model.model_id == "us.anthropic.fake-haiku"

    def test_region_from_config_preferred_over_env(self, monkeypatch) -> None:
        monkeypatch.setenv("REGION", "eu-west-1")
        config = OpenDataSciConfig(provider="bedrock", aws_region="us-east-2")  # type: ignore[arg-type]
        model = aws_module.create_bedrock_secondary_model(config)
        assert model.region_name == "us-east-2"

    def test_region_falls_back_to_env(self, monkeypatch) -> None:
        monkeypatch.setenv("REGION", "ap-south-1")
        config = OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
        model = aws_module.create_bedrock_secondary_model(config)
        assert model.region_name == "ap-south-1"

    def test_region_defaults_to_us_east_1_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("REGION", raising=False)
        config = OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
        model = aws_module.create_bedrock_secondary_model(config)
        assert model.region_name == "us-east-1"

    def test_temperature_is_zero(self) -> None:
        model = aws_module.create_bedrock_secondary_model(
            OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
        )
        assert model.temperature == 0

    def test_max_tokens_is_capped_low(self) -> None:
        model = aws_module.create_bedrock_secondary_model(
            OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
        )
        assert model.max_tokens == 1000


class TestMissingLangchainAws:
    """When langchain-aws is not installed, both factories must raise ValueError."""

    def test_create_bedrock_model_raises_when_custom_base_is_none(self) -> None:
        with patch.object(aws_module, "_CustomBedrockConverse", None):
            with pytest.raises(ValueError, match="langchain-aws"):
                aws_module.create_bedrock_model(
                    OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
                )

    def test_create_bedrock_secondary_model_raises_when_base_is_none(self) -> None:
        with patch.object(aws_module, "_BedrockBase", None):
            with pytest.raises(ValueError, match="langchain-aws"):
                aws_module.create_bedrock_secondary_model(
                    OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
                )


class TestCustomBedrockConverseStream:
    """_CustomBedrockConverse._stream must wrap the parent's _stream output and pass
    each yielded chunk through _strip_list_usage_fields before yielding it on."""

    def test_stream_passes_chunks_through_strip_helper(self) -> None:
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        # cacheDetails is a list — must be stripped on its way out of _stream.
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content="hi",
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                    "cache_details": [{"ttl": 60}],
                },
            )
        )

        with patch.object(
            aws_module._BedrockBase, "_stream", return_value=iter([chunk]), create=False
        ):
            model = aws_module._CustomBedrockConverse.__new__(aws_module._CustomBedrockConverse)
            results = list(model._stream(messages=[]))

        assert len(results) == 1
        assert results[0].message.usage_metadata == {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }


# ===========================================================================
# Streaming-fix regressions (create_bedrock_model disable_streaming override
# and _strip_list_usage_fields for Bedrock cacheDetails)
# ===========================================================================


class TestBedrockStreamingEnabled:
    """disable_streaming=False must be passed explicitly to ChatBedrockConverse
    so that Claude 4 (and any future) model IDs get streaming, regardless of
    what the library's auto-detection decides."""

    def test_disable_streaming_is_false_for_default_model(self) -> None:
        config = OpenDataSciConfig(provider="bedrock")  # type: ignore[arg-type]
        model = create_bedrock_model(config)
        assert model.disable_streaming is False, (
            "disable_streaming must be explicitly False; without it the library "
            "validator defaults to True for Claude 4 model IDs, silently "
            "disabling streaming and making every response arrive as one chunk."
        )

    @pytest.mark.parametrize(
        "model_id",
        [
            # Cross-region inference prefixed IDs (us., eu., ap.)
            "us.anthropic.claude-sonnet-4-6",
            "eu.anthropic.claude-sonnet-4-6",
            # Direct Bedrock model IDs for Claude 4
            "anthropic.claude-sonnet-4-6-20250514-v1:0",
            "us.anthropic.claude-haiku-4-5",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-opus-4-7",
        ],
    )
    def test_disable_streaming_false_for_claude4_ids(self, model_id: str) -> None:
        """None of these Claude 4 model IDs contain 'claude-3', so without the
        explicit override they would all get disable_streaming=True."""
        model = _create_model(model_id)
        assert model.disable_streaming is False

    def test_model_id_is_forwarded_to_constructor(self) -> None:
        model_id = "us.anthropic.claude-sonnet-4-6"
        model = _create_model(model_id)
        assert model.model_id == model_id

    def test_thinking_fields_are_present(self) -> None:
        """Adaptive thinking configuration must still be forwarded."""
        model = _create_model("us.anthropic.claude-sonnet-4-6")
        amrf = model.additional_model_request_fields or {}
        assert "thinking" in amrf


class TestStripListUsageFields:
    """_strip_list_usage_fields must remove list-valued keys from usage_metadata
    so that langchain_core's _dict_int_op can combine streaming chunks without
    raising ValueError when Bedrock includes cacheDetails (a list) in the
    token-usage response."""

    def _make_chunk(self, usage: dict | None):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        msg = AIMessageChunk(content="hi", usage_metadata=usage)
        return ChatGenerationChunk(message=msg)

    def test_passthrough_when_no_usage(self) -> None:
        chunk = self._make_chunk(None)
        result = _strip_list_usage_fields(chunk)
        assert result is chunk

    def test_passthrough_when_no_list_values(self) -> None:
        chunk = self._make_chunk({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        result = _strip_list_usage_fields(chunk)
        assert result is chunk

    def test_strips_cache_details_list(self) -> None:
        """cacheDetails is the Bedrock field that triggers the bug."""
        chunk = self._make_chunk(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cache_details": [{"ttl": 3600, "tokens": 30}],
            }
        )
        result = _strip_list_usage_fields(chunk)
        assert result is not chunk
        assert result.message.usage_metadata == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_strips_multiple_list_fields(self) -> None:
        chunk = self._make_chunk(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "list_a": [1, 2],
                "list_b": ["x"],
            }
        )
        result = _strip_list_usage_fields(chunk)
        assert result.message.usage_metadata == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    def test_preserves_dict_values(self) -> None:
        """Dict values (e.g. input_token_details) must not be stripped."""
        chunk = self._make_chunk(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": {"cache_read": 3},
            }
        )
        result = _strip_list_usage_fields(chunk)
        assert result is chunk  # no list fields → same object returned
        assert result.message.usage_metadata["input_token_details"] == {"cache_read": 3}
