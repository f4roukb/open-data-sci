"""Unit tests for the cached_system_prompt dispatcher in opendatasci.prompts.caching.

Per-provider cached_system_prompt helpers are tested in their own model factory test
files (test_anthropic, test_aws, test_openai, etc.).
"""


import pytest

from opendatasci.models.providers import Provider
from opendatasci.prompts.caching import cached_system_prompt

_PROMPT = "system prompt body"


class TestDispatcher:
    @pytest.mark.parametrize("provider", [Provider.ANTHROPIC, Provider.BEDROCK])
    def test_breakpoint_providers_return_structured_content(self, provider: Provider) -> None:
        content = cached_system_prompt(_PROMPT, provider)
        assert isinstance(content, list)
        assert any(isinstance(b, dict) and b.get("text") == _PROMPT for b in content)

    @pytest.mark.parametrize(
        "provider",
        [Provider.OPENAI, Provider.GEMINI, Provider.VERTEXAI, Provider.AZURE, Provider.OLLAMA, Provider.VLLM],
    )
    def test_automatic_providers_return_plain_string(self, provider: Provider) -> None:
        assert cached_system_prompt(_PROMPT, provider) == _PROMPT

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError):
            cached_system_prompt(_PROMPT, "mystery")  # type: ignore[arg-type]
