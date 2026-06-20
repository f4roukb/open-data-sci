from typing import Any

from opendatasci.models.providers import Provider


def cached_system_prompt(prompt: str, provider: Provider) -> str | list[dict[str, Any]]:
    """Format *prompt* as ``SystemMessage`` content with provider-specific caching.

    Each backend opts into prompt caching in the way its API expects:
    Anthropic and Bedrock embed an explicit cache breakpoint in the message;
    OpenAI, Gemini, Ollama and OpenAI-compatible servers (e.g. vLLM) rely on
    automatic server-side caching of the prompt prefix and return the prompt
    as a plain string.
    """
    match provider:
        case Provider.ANTHROPIC:
            from opendatasci.models.anthropic import cached_system_prompt as _impl

            return _impl(prompt)
        case Provider.BEDROCK:
            from opendatasci.models.aws import cached_system_prompt as _impl

            return _impl(prompt)
        case Provider.OPENAI:
            from opendatasci.models.openai import cached_system_prompt as _impl  # type: ignore[assignment]  # noqa: I001

            return _impl(prompt)
        case Provider.GEMINI | Provider.VERTEXAI:
            from opendatasci.models.google import cached_system_prompt as _impl  # type: ignore[assignment]  # noqa: I001

            return _impl(prompt)
        case Provider.AZURE:
            from opendatasci.models.microsoft import cached_system_prompt as _impl  # type: ignore[assignment]  # noqa: I001

            return _impl(prompt)
        case Provider.OLLAMA | Provider.OPENAI_COMPATIBLE_SERVER:
            from opendatasci.models.local import cached_system_prompt as _local_impl

            return _local_impl(prompt)

    supported = ", ".join(f"'{p}'" for p in Provider)
    raise ValueError(f"Unknown provider '{provider}'. Supported providers: {supported}.")
