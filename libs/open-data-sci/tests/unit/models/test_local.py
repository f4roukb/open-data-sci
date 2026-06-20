"""Unit tests for opendatasci.models.local factory functions (Ollama and OpenAI-compatible servers)."""


import sys
from unittest.mock import MagicMock, patch

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.local import (
    cached_system_prompt,
    create_ollama_model,
    create_ollama_secondary_model,
    create_openai_compatible_model,
    create_openai_compatible_secondary_model,
)


@pytest.fixture
def fake_chat_ollama(monkeypatch):
    """Insert a fake ``langchain_ollama`` module with a recording ``ChatOllama``."""
    captured: dict = {}

    def _ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ChatOllama", **kwargs)

    fake_module = MagicMock(name="langchain_ollama_stub")
    fake_module.ChatOllama = _ctor
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_module)
    return captured


@pytest.fixture
def fake_chat_openai_compatible(monkeypatch):
    """Insert a fake ``langchain_openai`` module with a recording ``ChatOpenAI``."""
    captured: dict = {}

    def _ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ChatOpenAI", **kwargs)

    fake_module = MagicMock(name="langchain_openai_stub")
    fake_module.ChatOpenAI = _ctor
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    return captured


class TestCreateOllamaPrimaryModel:
    def test_uses_resolved_model(self, fake_chat_ollama) -> None:
        config = OpenDataSciConfig(provider="ollama", model="llama3-test")  # type: ignore[arg-type]
        create_ollama_model(config)
        assert fake_chat_ollama["model"] == "llama3-test"

    def test_falls_back_to_provider_default(self, fake_chat_ollama) -> None:
        config = OpenDataSciConfig(provider="ollama")  # type: ignore[arg-type]
        create_ollama_model(config)
        assert fake_chat_ollama["model"] == config.model

    def test_temperature_propagated(self, fake_chat_ollama) -> None:
        config = OpenDataSciConfig(provider="ollama", temperature=0.6)  # type: ignore[arg-type]
        create_ollama_model(config)
        assert fake_chat_ollama["temperature"] == 0.6

    def test_base_url_from_config(self, fake_chat_ollama, monkeypatch) -> None:
        monkeypatch.delenv("LLM_SERVER_BASE_URL", raising=False)
        config = OpenDataSciConfig(provider="ollama", llm_server_base_url="http://custom:11434")  # type: ignore[arg-type]
        create_ollama_model(config)
        assert fake_chat_ollama["llm_server_base_url"] == "http://custom:11434"

    def test_base_url_from_env_var(self, fake_chat_ollama, monkeypatch) -> None:
        monkeypatch.setenv("LLM_SERVER_BASE_URL", "http://env-host:11434")
        config = OpenDataSciConfig(provider="ollama")  # type: ignore[arg-type]
        create_ollama_model(config)
        assert fake_chat_ollama["llm_server_base_url"] == "http://env-host:11434"

    def test_base_url_defaults_to_localhost(self, fake_chat_ollama, monkeypatch) -> None:
        monkeypatch.delenv("LLM_SERVER_BASE_URL", raising=False)
        config = OpenDataSciConfig(provider="ollama")  # type: ignore[arg-type]
        create_ollama_model(config)
        assert fake_chat_ollama["llm_server_base_url"] == "http://localhost:11434"

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_ollama", raising=False)
        with patch.dict(sys.modules, {"langchain_ollama": None}):
            with pytest.raises(ValueError, match="langchain-ollama"):
                create_ollama_model(OpenDataSciConfig(provider="ollama"))  # type: ignore[arg-type]


class TestCreateOllamaSecondaryModel:
    def test_uses_resolved_secondary_model(self, fake_chat_ollama) -> None:
        config = OpenDataSciConfig(provider="ollama", secondary_model="phi3-test")  # type: ignore[arg-type]
        create_ollama_secondary_model(config)
        assert fake_chat_ollama["model"] == "phi3-test"

    def test_temperature_is_zero(self, fake_chat_ollama) -> None:
        create_ollama_secondary_model(OpenDataSciConfig(provider="ollama"))  # type: ignore[arg-type]
        assert fake_chat_ollama["temperature"] == 0

    def test_num_predict_capped(self, fake_chat_ollama) -> None:
        create_ollama_secondary_model(OpenDataSciConfig(provider="ollama"))  # type: ignore[arg-type]
        assert fake_chat_ollama["num_predict"] == 1000

    def test_base_url_defaults_to_localhost(self, fake_chat_ollama, monkeypatch) -> None:
        monkeypatch.delenv("LLM_SERVER_BASE_URL", raising=False)
        create_ollama_secondary_model(OpenDataSciConfig(provider="ollama"))  # type: ignore[arg-type]
        assert fake_chat_ollama["llm_server_base_url"] == "http://localhost:11434"

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_ollama", raising=False)
        with patch.dict(sys.modules, {"langchain_ollama": None}):
            with pytest.raises(ValueError, match="langchain-ollama"):
                create_ollama_secondary_model(OpenDataSciConfig(provider="ollama"))  # type: ignore[arg-type]


class TestCreateOpenAICompatiblePrimaryModel:
    def test_uses_resolved_model(self, fake_chat_openai_compatible) -> None:
        config = OpenDataSciConfig(provider="openai_compatible_server", model="llama-test")  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["model"] == "llama-test"

    def test_falls_back_to_provider_default(self, fake_chat_openai_compatible) -> None:
        config = OpenDataSciConfig(provider="openai_compatible_server")  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["model"] == config.model

    def test_temperature_propagated(self, fake_chat_openai_compatible) -> None:
        config = OpenDataSciConfig(provider="openai_compatible_server", temperature=0.4)  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["temperature"] == 0.4

    def test_base_url_from_config(self, fake_chat_openai_compatible, monkeypatch) -> None:
        monkeypatch.delenv("LLM_SERVER_BASE_URL", raising=False)
        config = OpenDataSciConfig(
            provider="openai_compatible_server", llm_server_base_url="http://custom:8000/v1"
        )  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["base_url"] == "http://custom:8000/v1"

    def test_base_url_from_env_var(self, fake_chat_openai_compatible, monkeypatch) -> None:
        monkeypatch.setenv("LLM_SERVER_BASE_URL", "http://gpu-server:8000/v1")
        config = OpenDataSciConfig(provider="openai_compatible_server")  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["base_url"] == "http://gpu-server:8000/v1"

    def test_base_url_defaults_to_localhost(self, fake_chat_openai_compatible, monkeypatch) -> None:
        monkeypatch.delenv("LLM_SERVER_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = OpenDataSciConfig(provider="openai_compatible_server")  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["base_url"] == "http://localhost:8000/v1"

    def test_api_key_defaults_to_empty(self, fake_chat_openai_compatible, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = OpenDataSciConfig(provider="openai_compatible_server")  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["api_key"] == "EMPTY"

    def test_api_key_from_config_preferred(self, fake_chat_openai_compatible, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = OpenDataSciConfig(provider="openai_compatible_server", openai_api_key="my-server-token")  # type: ignore[arg-type]
        create_openai_compatible_model(config)
        assert fake_chat_openai_compatible["api_key"] == "my-server-token"

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_openai", raising=False)
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ValueError, match="langchain-openai"):
                create_openai_compatible_model(OpenDataSciConfig(provider="openai_compatible_server"))  # type: ignore[arg-type]


class TestCreateOpenAICompatibleSecondaryModel:
    def test_uses_resolved_secondary_model(self, fake_chat_openai_compatible) -> None:
        config = OpenDataSciConfig(provider="openai_compatible_server", secondary_model="llama-mini-test")  # type: ignore[arg-type]
        create_openai_compatible_secondary_model(config)
        assert fake_chat_openai_compatible["model"] == "llama-mini-test"

    def test_temperature_is_zero(self, fake_chat_openai_compatible) -> None:
        create_openai_compatible_secondary_model(OpenDataSciConfig(provider="openai_compatible_server"))  # type: ignore[arg-type]
        assert fake_chat_openai_compatible["temperature"] == 0

    def test_max_tokens_capped(self, fake_chat_openai_compatible) -> None:
        create_openai_compatible_secondary_model(OpenDataSciConfig(provider="openai_compatible_server"))  # type: ignore[arg-type]
        assert fake_chat_openai_compatible["max_tokens"] == 1000

    def test_base_url_defaults_to_localhost(self, fake_chat_openai_compatible, monkeypatch) -> None:
        monkeypatch.delenv("LLM_SERVER_BASE_URL", raising=False)
        create_openai_compatible_secondary_model(OpenDataSciConfig(provider="openai_compatible_server"))  # type: ignore[arg-type]
        assert fake_chat_openai_compatible["base_url"] == "http://localhost:8000/v1"

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_openai", raising=False)
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ValueError, match="langchain-openai"):
                create_openai_compatible_secondary_model(OpenDataSciConfig(provider="openai_compatible_server"))  # type: ignore[arg-type]


class TestCachedSystemPrompt:
    def test_returns_prompt_unchanged(self) -> None:
        assert cached_system_prompt("hello world") == "hello world"

    def test_empty_string_returned_as_is(self) -> None:
        assert cached_system_prompt("") == ""
