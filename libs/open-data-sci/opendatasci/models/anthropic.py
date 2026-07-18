import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from opendatasci.configs import OpenDataSciConfig

# Matches modern Claude IDs, including provider-prefixed Bedrock IDs
# (us.anthropic.claude-sonnet-5) and IDs without a minor version
# (claude-sonnet-5).
_CLAUDE_ID = re.compile(r"claude-(opus|sonnet|haiku)-(\d+)(?:-(\d+))?")


def supports_adaptive_thinking(model_id: str) -> bool:
    """Return True for Claude models that use adaptive thinking.

    Claude 4.6+ (Opus 4.6/4.7/4.8, Sonnet 4.6, Sonnet 5) uses
    ``thinking: {"type": "adaptive"}``.  On Opus 4.7+ and Sonnet 5 the legacy
    ``budget_tokens`` config and explicit sampling parameters
    (``temperature``/``top_p``/``top_k``) are rejected with a 400.
    """
    m = _CLAUDE_ID.search(model_id)
    if m is None:
        # Legacy Claude 3.x IDs place the version before the variant
        # (claude-3-5-haiku-...) and never support adaptive thinking.
        return False
    variant, major, minor = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    if variant == "haiku":
        return False  # haiku-4-5 still uses budget_tokens-style thinking
    return (major, minor) >= (4, 6)


def create_anthropic_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a ``ChatAnthropic`` model with adaptive thinking where supported."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ValueError("langchain-anthropic is not installed.") from exc
    if supports_adaptive_thinking(config.model):
        # Claude 4.6+ / Sonnet 5: adaptive thinking; explicit sampling
        # parameters are rejected.
        return ChatAnthropic(
            model=config.model,
            api_key=config.anthropic_api_key,
            max_tokens=16000,
            thinking={"type": "adaptive"},
        )
    # Models without adaptive thinking (e.g. haiku-4-5) run without thinking.
    return ChatAnthropic(
        model=config.model,
        api_key=config.anthropic_api_key,
        temperature=config.temperature,
        max_tokens=16000,
    )


def create_anthropic_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap ``ChatAnthropic`` model for auxiliary tasks (thinking disabled)."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ValueError("langchain-anthropic is not installed.") from exc
    if supports_adaptive_thinking(config.secondary_model):
        # Explicit temperature is rejected on Opus 4.7+ and Sonnet 5.
        return ChatAnthropic(
            model=config.secondary_model,
            api_key=config.anthropic_api_key,
            max_tokens=1000,
        )
    return ChatAnthropic(
        model=config.secondary_model,
        api_key=config.anthropic_api_key,
        temperature=0,
        max_tokens=1000,
    )


def cached_system_prompt(prompt: str) -> list[dict[str, Any]]:
    """Wrap *prompt* with Anthropic's ephemeral cache breakpoint."""
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
