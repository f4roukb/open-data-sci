from langchain_core.language_models import BaseChatModel

from opendatasci.configs import OpenDataSciConfig


def create_ollama_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a ``ChatOllama`` model against a local Ollama server."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ValueError(
            "langchain-ollama is not installed. Run: pip install 'open-data-sci[ollama]'"
        ) from exc
    model: BaseChatModel = ChatOllama(
        model=config.model,
        llm_server_base_url=config.llm_server_base_url or "http://localhost:11434",
        temperature=config.temperature,
    )
    return model


def create_ollama_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap ``ChatOllama`` model for auxiliary tasks."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ValueError(
            "langchain-ollama is not installed. Run: pip install 'open-data-sci[ollama]'"
        ) from exc
    model: BaseChatModel = ChatOllama(
        model=config.secondary_model,
        llm_server_base_url=config.llm_server_base_url or "http://localhost:11434",
        temperature=0,
        num_predict=1000,
    )
    return model


def create_vllm_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate an OpenAI-compatible ``ChatOpenAI`` model against a local vLLM server."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ValueError("langchain-openai is not installed.") from exc
    return ChatOpenAI(
        model=config.model,
        base_url=config.llm_server_base_url or "http://localhost:8000/v1",
        api_key=config.openai_api_key or "EMPTY",
        temperature=config.temperature,
    )


def create_vllm_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap vLLM-backed ``ChatOpenAI`` model for auxiliary tasks."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ValueError("langchain-openai is not installed.") from exc
    return ChatOpenAI(
        model=config.secondary_model,
        base_url=config.llm_server_base_url or "http://localhost:8000/v1",
        api_key=config.openai_api_key or "EMPTY",
        temperature=0,
        max_tokens=1000,
    )


def cached_system_prompt(prompt: str) -> str:
    """Return the system prompt unchanged.

    Both Ollama and vLLM perform automatic prefix caching server-side: Ollama
    enables it by default for recent versions, and vLLM enables it when
    started with `--enable-prefix-caching`. Caching is keyed on the leading
    prompt prefix, so no client-side cache markers are required.
    """
    return prompt
