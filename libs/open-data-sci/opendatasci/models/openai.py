from langchain_core.language_models import BaseChatModel

from opendatasci.configs import OpenDataSciConfig

# Stable routing hint passed via OpenAI's `prompt_cache_key`. Caching itself is
# automatic for prompts >= 1024 tokens; this key just ensures repeated requests
# from the same OpenDataSci session land on the same backend, maximising cache hits.
_PROMPT_CACHE_KEY = "open-data-sci-system-v1"


def create_openai_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a ``ChatOpenAI`` model with a stable prompt cache key."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ValueError("langchain-openai is not installed.") from exc
    return ChatOpenAI(
        model=config.model,
        api_key=config.openai_api_key,
        temperature=config.temperature,
        reasoning_effort="medium",
        model_kwargs={"prompt_cache_key": _PROMPT_CACHE_KEY},
    )


def create_openai_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap ``ChatOpenAI`` model for auxiliary tasks."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ValueError("langchain-openai is not installed.") from exc
    return ChatOpenAI(
        model=config.secondary_model,
        api_key=config.openai_api_key,
        temperature=0,
        max_tokens=1000,
        model_kwargs={"prompt_cache_key": _PROMPT_CACHE_KEY},
    )


def cached_system_prompt(prompt: str) -> str:
    """Return the system prompt unchanged.

    OpenAI performs automatic prompt caching for any prompt prefix >= 1024
    tokens on `gpt-4o` and newer models, so no client-side cache markers are
    needed. The model is constructed with a stable `prompt_cache_key` to keep
    routing consistent and maximise cache hit rates.
    """
    return prompt
