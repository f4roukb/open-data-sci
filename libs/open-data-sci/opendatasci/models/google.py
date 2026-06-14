from langchain_core.language_models import BaseChatModel

from opendatasci.configs import OpenDataSciConfig


def create_gemini_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a ``ChatGoogleGenerativeAI`` model via the Gemini API."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ValueError(
            "langchain-google-genai is not installed. Run: pip install 'open-data-sci[gemini]'"
        ) from exc
    return ChatGoogleGenerativeAI(  # type: ignore[no-any-return]
        model=config.model,
        google_api_key=config.google_api_key,
        temperature=config.temperature,
    )


def create_gemini_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap Gemini model for auxiliary tasks."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ValueError(
            "langchain-google-genai is not installed. Run: pip install 'open-data-sci[gemini]'"
        ) from exc
    return ChatGoogleGenerativeAI(  # type: ignore[no-any-return]
        model=config.secondary_model,
        google_api_key=config.google_api_key,
        temperature=0,
        max_output_tokens=1000,
    )


def create_vertexai_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a ``ChatVertexAI`` model via Google Cloud Vertex AI."""
    try:
        from langchain_google_vertexai import ChatVertexAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ValueError(
            "langchain-google-vertexai is not installed. Run: pip install 'open-data-sci[gcp]'"
        ) from exc
    return ChatVertexAI(  # type: ignore[no-any-return]
        model=config.model,
        project=config.google_cloud_project,
        location=config.google_cloud_location,
        temperature=config.temperature,
    )


def create_vertexai_secondary_model(config: OpenDataSciConfig) -> BaseChatModel:
    """Instantiate a cheap Vertex AI model for auxiliary tasks."""
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError as exc:
        raise ValueError(
            "langchain-google-vertexai is not installed. Run: pip install 'open-data-sci[gcp]'"
        ) from exc
    return ChatVertexAI(  # type: ignore[no-any-return]
        model=config.secondary_model,
        project=config.google_cloud_project,
        location=config.google_cloud_location,
        temperature=0,
        max_output_tokens=1000,
    )


def cached_system_prompt(prompt: str) -> str:
    """Return the system prompt unchanged.

    Gemini 2.5+ models perform implicit context caching automatically for
    prompts above the per-model minimum (>= 1024 tokens for Flash, >= 4096 for
    Pro), keying off the request's leading prefix. No client-side cache
    markers are required, and explicit `cached_content` setup is intentionally
    not used here to keep the model factory side-effect free.
    """
    return prompt
