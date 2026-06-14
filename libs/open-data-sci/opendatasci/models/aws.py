from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from opendatasci.configs import OpenDataSciConfig

try:
    from langchain_aws import ChatBedrockConverse as _BedrockBase
except ImportError:
    _BedrockBase = None


def _strip_list_usage_fields(chunk: ChatGenerationChunk) -> ChatGenerationChunk:
    """Remove list-valued fields from a chunk's usage_metadata.

    Bedrock's TokenUsage includes a ``cacheDetails`` field (type: list) which
    langchain_core's ``_dict_int_op`` cannot combine — it only accepts int/dict
    values.  Stripping the field here prevents the ValueError that would
    otherwise surface as a spurious error event at the end of each streamed
    response when prompt caching is active.
    """
    msg = chunk.message
    if not (isinstance(msg, AIMessageChunk) and msg.usage_metadata):
        return chunk
    if not any(isinstance(v, list) for v in msg.usage_metadata.values()):
        return chunk
    cleaned = {k: v for k, v in msg.usage_metadata.items() if not isinstance(v, list)}
    return ChatGenerationChunk(
        message=msg.model_copy(update={"usage_metadata": cleaned}),
        generation_info=chunk.generation_info,
    )


if _BedrockBase is not None:

    class _CustomBedrockConverse(_BedrockBase):  # type: ignore[misc]
        def _stream(
            self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
        ) -> Any:
            for chunk in super()._stream(messages, stop, run_manager, **kwargs):
                yield _strip_list_usage_fields(chunk)
else:
    _CustomBedrockConverse = None  # type: ignore[assignment,misc]


def create_bedrock_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a Bedrock Converse model with adaptive thinking and streaming enabled."""
    if _CustomBedrockConverse is None:
        raise ValueError("langchain-aws is not installed.")
    return _CustomBedrockConverse(
        model=config.model,
        region_name=config.aws_region,
        # Temperature must be 1 when extended thinking is enabled.
        temperature=1,
        max_tokens=16000,
        additional_model_request_fields={
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        },
        # langchain-aws's set_disable_streaming validator only recognises
        # "claude-3" model IDs as streaming-capable.  Claude 4 models
        # (claude-sonnet-4-6, etc.) match none of the patterns and default to
        # disable_streaming=True, causing every response to arrive as a single
        # non-streamed chunk.  Claude 4 supports Bedrock ConverseStream with
        # tools, so we override the auto-detection explicitly.
        disable_streaming=False,
    )


def create_bedrock_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap Bedrock model for auxiliary tasks (thinking disabled)."""
    if _BedrockBase is None:
        raise ValueError("langchain-aws is not installed.")
    return _BedrockBase(  # type: ignore[no-any-return]
        model=config.secondary_model,
        region_name=config.aws_region,
        temperature=0,
        max_tokens=1000,
    )


def cached_system_prompt(prompt: str) -> list[dict[str, Any]]:
    """Wrap *prompt* with a Bedrock Converse cache point breakpoint."""
    return [{"type": "text", "text": prompt}, {"cachePoint": {"type": "default"}}]
