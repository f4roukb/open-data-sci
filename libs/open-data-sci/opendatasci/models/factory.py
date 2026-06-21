"""
LLM provider factory for the Agent.
"""

import asyncio
import logging
import random
from typing import Any

from langchain_core.language_models import BaseChatModel

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.providers import Provider

_LOG = logging.getLogger(__name__)

_TRANSIENT_KEYWORDS = frozenset(
    [
        "rate limit",
        "ratelimit",
        "rate_limit",
        "429",
        "too many requests",
        "overloaded",
        "overload",
        "503",
        "service unavailable",
        "throttl",
        "connection error",
        "connection refused",
        "connecterror",
        "apiconnectionerror",
    ]
)

_MAX_RETRY_WAIT = 60.0
_DEFAULT_MAX_ATTEMPTS = 5


def _is_transient(exc: Exception) -> bool:
    text = (type(exc).__name__ + " " + str(exc)).lower()
    return any(kw in text for kw in _TRANSIENT_KEYWORDS)


class _RetryRunnable:
    """Wraps a LangChain Runnable, retrying transient errors with exponential backoff."""

    def __init__(self, runnable: Any, max_attempts: int = _DEFAULT_MAX_ATTEMPTS) -> None:
        self._runnable = runnable
        self._max_attempts = max_attempts

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(self._max_attempts):
            try:
                return await self._runnable.ainvoke(*args, **kwargs)
            except Exception as exc:
                if _is_transient(exc) and attempt < self._max_attempts - 1:
                    wait = min(_MAX_RETRY_WAIT, (2**attempt) + random.random())
                    _LOG.warning(
                        "LLM transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_attempts,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise


def with_retry(runnable: Any, max_attempts: int = _DEFAULT_MAX_ATTEMPTS) -> _RetryRunnable:
    """Wrap *runnable* so that transient LLM errors are retried with exponential backoff."""
    return _RetryRunnable(runnable, max_attempts=max_attempts)


def _supported_providers() -> str:
    return ", ".join(f"'{p}'" for p in Provider)


def create_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate the primary LLM for the agent and workers.

    Thinking is enabled by default for providers that support it (Anthropic, Bedrock).
    Falls back to the provider's default primary model when none is specified.
    """
    match config.provider:
        case Provider.ANTHROPIC:
            from opendatasci.models.anthropic import create_anthropic_model as _create

            return _create(config)
        case Provider.BEDROCK:
            from opendatasci.models.aws import create_bedrock_model as _create

            return _create(config)
        case Provider.OPENAI:
            from opendatasci.models.openai import create_openai_model as _create

            return _create(config)
        case Provider.GEMINI:
            from opendatasci.models.google import create_gemini_model as _create

            return _create(config)
        case Provider.VERTEXAI:
            from opendatasci.models.google import create_vertexai_model as _create

            return _create(config)
        case Provider.AZURE:
            from opendatasci.models.microsoft import create_azure_model as _create

            return _create(config)
        case Provider.OLLAMA:
            from opendatasci.models.local import create_ollama_model as _create

            return _create(config)
        case Provider.OPENAI_COMPATIBLE_SERVER:
            from opendatasci.models.local import create_openai_compatible_model as _create

            return _create(config)

    raise ValueError(
        f"Unknown provider '{config.provider}'. Supported providers: {_supported_providers()}."
    )


def create_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate the secondary (auxiliary) LLM for lightweight tasks such as summarization.

    Thinking is disabled. Falls back to the provider's default secondary model when none is
    specified.

    When ``config.secondary_provider`` differs from ``config.provider``, a
    lightweight config copy is used so the secondary model's factory resolves its
    API key from its own environment variable rather than the primary provider's.
    """
    secondary_provider = config.secondary_provider

    # When the secondary model uses a different provider, switch the provider
    # field so the factory dispatches correctly.  Each provider reads its own
    # dedicated API-key field, so no credential clearing is needed.
    if secondary_provider != config.provider:
        config = config.model_copy(update={"provider": secondary_provider})

    match secondary_provider:
        case Provider.ANTHROPIC:
            from opendatasci.models.anthropic import create_anthropic_secondary_model as _create

            return _create(config)
        case Provider.BEDROCK:
            from opendatasci.models.aws import create_bedrock_secondary_model as _create

            return _create(config)
        case Provider.OPENAI:
            from opendatasci.models.openai import create_openai_secondary_model as _create

            return _create(config)
        case Provider.GEMINI:
            from opendatasci.models.google import create_gemini_secondary_model as _create

            return _create(config)
        case Provider.VERTEXAI:
            from opendatasci.models.google import create_vertexai_secondary_model as _create

            return _create(config)
        case Provider.AZURE:
            from opendatasci.models.microsoft import create_azure_secondary_model as _create

            return _create(config)
        case Provider.OLLAMA:
            from opendatasci.models.local import create_ollama_secondary_model as _create

            return _create(config)
        case Provider.OPENAI_COMPATIBLE_SERVER:
            from opendatasci.models.local import create_openai_compatible_secondary_model as _create

            return _create(config)

    raise ValueError(
        f"Unknown provider '{secondary_provider}'. Supported providers: {_supported_providers()}."
    )
