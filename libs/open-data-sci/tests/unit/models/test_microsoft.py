"""Unit tests for opendatasci.models.microsoft factory functions."""


import sys
from unittest.mock import MagicMock

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.microsoft import (
    _PROMPT_CACHE_KEY,
    cached_system_prompt,
    create_azure_model,
    create_azure_secondary_model,
)

_ENDPOINT = "https://my-resource.openai.azure.com"


@pytest.fixture
def fake_azure_chat_openai(monkeypatch):
    """Insert a fake ``langchain_openai`` module with a recording ``AzureChatOpenAI``."""
    captured: dict = {}

    def _ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="AzureChatOpenAI", **kwargs)

    fake_module = MagicMock(name="langchain_openai_stub")
    fake_module.AzureChatOpenAI = _ctor
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    return captured


class TestCreateAzurePrimaryModel:
    def test_uses_resolved_model(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(provider="azure", model="gpt-4o-test", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["azure_deployment"] == "gpt-4o-test"

    def test_falls_back_to_provider_default(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["azure_deployment"] == config.model

    def test_endpoint_from_config_field(self, fake_azure_chat_openai, monkeypatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["azure_endpoint"] == _ENDPOINT

    def test_endpoint_from_env_var(self, fake_azure_chat_openai, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", _ENDPOINT)
        config = OpenDataSciConfig(provider="azure")  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["azure_endpoint"] == _ENDPOINT

    def test_missing_endpoint_raises_llm_provider_error(
        self, fake_azure_chat_openai, monkeypatch
    ) -> None:
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        config = OpenDataSciConfig(provider="azure")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
            create_azure_model(config)

    def test_api_key_from_config_preferred_over_env(
        self, fake_azure_chat_openai, monkeypatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
        config = OpenDataSciConfig(
            provider="azure", azure_endpoint=_ENDPOINT, azure_api_key="config-key"
        )  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["api_key"] == "config-key"

    def test_api_key_falls_back_to_env_var(self, fake_azure_chat_openai, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-fallback")
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["api_key"] == "env-fallback"

    def test_temperature_propagated(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT, temperature=0.5)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["temperature"] == 0.5

    def test_prompt_cache_key_forwarded(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["model_kwargs"] == {"prompt_cache_key": _PROMPT_CACHE_KEY}

    def test_api_version_from_config_field(self, fake_azure_chat_openai, monkeypatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        config = OpenDataSciConfig(
            provider="azure", azure_endpoint=_ENDPOINT, azure_api_version="2024-06-01"
        )  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["api_version"] == "2024-06-01"

    def test_api_version_from_env_var(self, fake_azure_chat_openai, monkeypatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["api_version"] == "2024-06-01"

    def test_api_version_has_default(self, fake_azure_chat_openai, monkeypatch) -> None:
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_model(config)
        assert fake_azure_chat_openai["api_version"] == "2025-01-01-preview"


class TestCreateAzureSecondaryModel:
    def test_uses_resolved_secondary_model(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(
            provider="azure", secondary_model="gpt-4o-mini-test", azure_endpoint=_ENDPOINT
        )  # type: ignore[arg-type]
        create_azure_secondary_model(config)
        assert fake_azure_chat_openai["azure_deployment"] == "gpt-4o-mini-test"

    def test_missing_endpoint_raises_llm_provider_error(
        self, fake_azure_chat_openai, monkeypatch
    ) -> None:
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        config = OpenDataSciConfig(provider="azure")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
            create_azure_secondary_model(config)

    def test_temperature_is_zero(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_secondary_model(config)
        assert fake_azure_chat_openai["temperature"] == 0

    def test_max_tokens_capped(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_secondary_model(config)
        assert fake_azure_chat_openai["max_tokens"] == 1000

    def test_prompt_cache_key_also_set(self, fake_azure_chat_openai, monkeypatch) -> None:
        config = OpenDataSciConfig(provider="azure", azure_endpoint=_ENDPOINT)  # type: ignore[arg-type]
        create_azure_secondary_model(config)
        assert fake_azure_chat_openai["model_kwargs"] == {"prompt_cache_key": _PROMPT_CACHE_KEY}


class TestCachedSystemPrompt:
    def test_returns_prompt_unchanged(self) -> None:
        assert cached_system_prompt("hello world") == "hello world"

    def test_empty_string_returned_as_is(self) -> None:
        assert cached_system_prompt("") == ""
