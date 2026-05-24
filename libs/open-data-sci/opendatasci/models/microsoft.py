from langchain_core.language_models import BaseChatModel

from opendatasci.configs import OpenDataSciConfig

# Stable routing hint for Azure OpenAI's prompt cache.  Automatic prefix
# caching is available on gpt-4o and newer deployments; this key keeps
# repeated requests from the same session routed consistently.
_PROMPT_CACHE_KEY = "open-data-sci-system-v1"


def _resolve_azure_endpoint(config: OpenDataSciConfig) -> str:
    if not config.azure_endpoint:
        raise ValueError(
            "Azure OpenAI endpoint is not configured. "
            "Set the AZURE_OPENAI_ENDPOINT environment variable or pass "
            "azure_endpoint in OpenDataSciConfig."
        )
    return config.azure_endpoint


def create_azure_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate an ``AzureChatOpenAI`` model with a stable prompt cache key."""
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as exc:
        raise ValueError("langchain-openai is not installed.") from exc
    return AzureChatOpenAI(
        azure_deployment=config.model,
        azure_endpoint=_resolve_azure_endpoint(config),
        api_key=config.azure_api_key,
        api_version=config.azure_api_version,
        temperature=config.temperature,
        model_kwargs={"prompt_cache_key": _PROMPT_CACHE_KEY},
    )


def create_azure_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap ``AzureChatOpenAI`` model for auxiliary tasks."""
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as exc:
        raise ValueError("langchain-openai is not installed.") from exc
    return AzureChatOpenAI(
        azure_deployment=config.secondary_model,
        azure_endpoint=_resolve_azure_endpoint(config),
        api_key=config.azure_api_key,
        api_version=config.azure_api_version,
        temperature=0,
        max_tokens=1000,
        model_kwargs={"prompt_cache_key": _PROMPT_CACHE_KEY},
    )


def cached_system_prompt(prompt: str) -> str:
    """Return the system prompt unchanged.

    Azure OpenAI performs automatic prompt caching for prefixes >= 1024
    tokens on supported deployments. No client-side cache markers are
    required; a stable ``prompt_cache_key`` is passed at model construction
    to maximise routing consistency and cache hit rates.
    """
    return prompt
