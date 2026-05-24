"""Unit tests for opendatasci.models.openai factory functions."""


import sys
from unittest.mock import MagicMock, patch

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.openai import (
    _PROMPT_CACHE_KEY,
    cached_system_prompt,
    create_openai_model,
    create_openai_secondary_model,
)


@pytest.fixture
def fake_chat_openai(monkeypatch):
    """Insert a fake ``langchain_openai`` module with a recording ``ChatOpenAI``."""
    captured: dict = {}

    def _ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ChatOpenAI", **kwargs)

    fake_module = MagicMock(name="langchain_openai_stub")
    fake_module.ChatOpenAI = _ctor
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    return captured


class TestCreateOpenAIPrimaryModel:
    def test_uses_resolved_model(self, fake_chat_openai) -> None:
        config = OpenDataSciConfig(provider="openai", model="gpt-test")  # type: ignore[arg-type]
        create_openai_model(config)
        assert fake_chat_openai["model"] == "gpt-test"

    def test_falls_back_to_provider_default_when_model_unset(self, fake_chat_openai) -> None:
        config = OpenDataSciConfig(provider="openai")  # type: ignore[arg-type]
        create_openai_model(config)
        assert fake_chat_openai["model"] == config.model

    def test_temperature_propagated_from_config(self, fake_chat_openai) -> None:
        # OpenAI does not have extended thinking, so the user-set temperature is honoured.
        config = OpenDataSciConfig(provider="openai", temperature=0.42)  # type: ignore[arg-type]
        create_openai_model(config)
        assert fake_chat_openai["temperature"] == 0.42

    def test_reasoning_effort_set_to_medium(self, fake_chat_openai) -> None:
        create_openai_model(OpenDataSciConfig(provider="openai"))  # type: ignore[arg-type]
        assert fake_chat_openai["reasoning_effort"] == "medium"

    def test_prompt_cache_key_forwarded_via_model_kwargs(self, fake_chat_openai) -> None:
        # A stable prompt_cache_key is needed so repeated calls route to the same backend.
        create_openai_model(OpenDataSciConfig(provider="openai"))  # type: ignore[arg-type]
        assert fake_chat_openai["model_kwargs"] == {"prompt_cache_key": _PROMPT_CACHE_KEY}

    def test_api_key_from_config_preferred_over_env(self, fake_chat_openai, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = OpenDataSciConfig(provider="openai", openai_api_key="config-key")  # type: ignore[arg-type]
        create_openai_model(config)
        assert fake_chat_openai["api_key"] == "config-key"

    def test_api_key_falls_back_to_env_var(self, fake_chat_openai, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "env-fallback")
        config = OpenDataSciConfig(provider="openai")  # type: ignore[arg-type]
        create_openai_model(config)
        assert fake_chat_openai["api_key"] == "env-fallback"

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_openai", raising=False)
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ValueError, match="langchain-openai"):
                create_openai_model(OpenDataSciConfig(provider="openai"))  # type: ignore[arg-type]


class TestCreateOpenAISecondaryModel:
    def test_uses_resolved_secondary_model(self, fake_chat_openai) -> None:
        config = OpenDataSciConfig(provider="openai", secondary_model="gpt-mini-test")  # type: ignore[arg-type]
        create_openai_secondary_model(config)
        assert fake_chat_openai["model"] == "gpt-mini-test"

    def test_temperature_is_zero(self, fake_chat_openai) -> None:
        create_openai_secondary_model(OpenDataSciConfig(provider="openai"))  # type: ignore[arg-type]
        assert fake_chat_openai["temperature"] == 0

    def test_max_tokens_is_capped_low(self, fake_chat_openai) -> None:
        create_openai_secondary_model(OpenDataSciConfig(provider="openai"))  # type: ignore[arg-type]
        assert fake_chat_openai["max_tokens"] == 1000

    def test_prompt_cache_key_also_set_on_secondary_model(self, fake_chat_openai) -> None:
        create_openai_secondary_model(OpenDataSciConfig(provider="openai"))  # type: ignore[arg-type]
        assert fake_chat_openai["model_kwargs"] == {"prompt_cache_key": _PROMPT_CACHE_KEY}

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_openai", raising=False)
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ValueError, match="langchain-openai"):
                create_openai_secondary_model(OpenDataSciConfig(provider="openai"))  # type: ignore[arg-type]


class TestCachedSystemPrompt:
    def test_returns_prompt_unchanged(self) -> None:
        # OpenAI caches >= 1024-token prefixes automatically; the prompt is passed through.
        assert cached_system_prompt("hello world") == "hello world"

    def test_empty_string_returned_as_is(self) -> None:
        assert cached_system_prompt("") == ""
