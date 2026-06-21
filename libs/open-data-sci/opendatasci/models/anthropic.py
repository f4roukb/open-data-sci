from typing import Any

from langchain_core.language_models import BaseChatModel

from opendatasci.configs import OpenDataSciConfig


def create_anthropic_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a ``ChatAnthropic`` model with extended thinking enabled."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ValueError("langchain-anthropic is not installed.") from exc
    return ChatAnthropic(
        model=config.model,
        api_key=config.anthropic_api_key,
        # Temperature must be 1 when extended thinking is enabled.
        temperature=1,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": config.thinking_budget},
    )


def create_anthropic_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap ``ChatAnthropic`` model for auxiliary tasks (thinking disabled)."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ValueError("langchain-anthropic is not installed.") from exc
    return ChatAnthropic(
        model=config.secondary_model,
        api_key=config.anthropic_api_key,
        temperature=0,
        max_tokens=1000,
    )


def cached_system_prompt(prompt: str) -> list[dict[str, Any]]:
    """Wrap *prompt* with Anthropic's ephemeral cache breakpoint."""
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
