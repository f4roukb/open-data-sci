"""Unit tests for opendatasci.models.google factory functions."""


import sys
from unittest.mock import MagicMock, patch

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.google import (
    cached_system_prompt,
    create_gemini_model,
    create_gemini_secondary_model,
    create_vertexai_model,
    create_vertexai_secondary_model,
)


@pytest.fixture
def fake_genai(monkeypatch):
    """Insert a fake ``langchain_google_genai`` module with a recording ``ChatGoogleGenerativeAI``."""
    captured: dict = {}

    def _ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ChatGoogleGenerativeAI", **kwargs)

    fake_module = MagicMock(name="langchain_google_genai_stub")
    fake_module.ChatGoogleGenerativeAI = _ctor
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    return captured


@pytest.fixture
def fake_vertexai(monkeypatch):
    """Insert a fake ``langchain_google_vertexai`` module with a recording ``ChatVertexAI``."""
    captured: dict = {}

    def _ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ChatVertexAI", **kwargs)

    fake_module = MagicMock(name="langchain_google_vertexai_stub")
    fake_module.ChatVertexAI = _ctor
    monkeypatch.setitem(sys.modules, "langchain_google_vertexai", fake_module)
    return captured


class TestCreateGeminiPrimaryModel:
    def test_uses_resolved_model(self, fake_genai) -> None:
        config = OpenDataSciConfig(provider="gemini", model="gemini-test")  # type: ignore[arg-type]
        create_gemini_model(config)
        assert fake_genai["model"] == "gemini-test"

    def test_falls_back_to_provider_default(self, fake_genai) -> None:
        config = OpenDataSciConfig(provider="gemini")  # type: ignore[arg-type]
        create_gemini_model(config)
        assert fake_genai["model"] == config.model

    def test_temperature_propagated(self, fake_genai) -> None:
        config = OpenDataSciConfig(provider="gemini", temperature=0.7)  # type: ignore[arg-type]
        create_gemini_model(config)
        assert fake_genai["temperature"] == 0.7

    def test_api_key_from_config_preferred_over_env(self, fake_genai, monkeypatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        config = OpenDataSciConfig(provider="gemini", google_api_key="config-key")  # type: ignore[arg-type]
        create_gemini_model(config)
        assert fake_genai["google_api_key"] == "config-key"

    def test_api_key_falls_back_to_env_var(self, fake_genai, monkeypatch) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "env-fallback")
        config = OpenDataSciConfig(provider="gemini")  # type: ignore[arg-type]
        create_gemini_model(config)
        assert fake_genai["google_api_key"] == "env-fallback"

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_google_genai", raising=False)
        with patch.dict(sys.modules, {"langchain_google_genai": None}):
            with pytest.raises(ValueError, match="langchain-google-genai"):
                create_gemini_model(OpenDataSciConfig(provider="gemini"))  # type: ignore[arg-type]


class TestCreateGeminiSecondaryModel:
    def test_uses_resolved_secondary_model(self, fake_genai) -> None:
        config = OpenDataSciConfig(provider="gemini", secondary_model="gemini-mini-test")  # type: ignore[arg-type]
        create_gemini_secondary_model(config)
        assert fake_genai["model"] == "gemini-mini-test"

    def test_temperature_is_zero(self, fake_genai) -> None:
        create_gemini_secondary_model(OpenDataSciConfig(provider="gemini"))  # type: ignore[arg-type]
        assert fake_genai["temperature"] == 0

    def test_max_output_tokens_capped(self, fake_genai) -> None:
        create_gemini_secondary_model(OpenDataSciConfig(provider="gemini"))  # type: ignore[arg-type]
        assert fake_genai["max_output_tokens"] == 1000

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_google_genai", raising=False)
        with patch.dict(sys.modules, {"langchain_google_genai": None}):
            with pytest.raises(ValueError, match="langchain-google-genai"):
                create_gemini_secondary_model(OpenDataSciConfig(provider="gemini"))  # type: ignore[arg-type]


class TestCreateVertexAIPrimaryModel:
    def test_uses_resolved_model(self, fake_vertexai) -> None:
        config = OpenDataSciConfig(provider="vertexai", model="gemini-vertex-test")  # type: ignore[arg-type]
        create_vertexai_model(config)
        assert fake_vertexai["model"] == "gemini-vertex-test"

    def test_falls_back_to_provider_default(self, fake_vertexai) -> None:
        config = OpenDataSciConfig(provider="vertexai")  # type: ignore[arg-type]
        create_vertexai_model(config)
        assert fake_vertexai["model"] == config.model

    def test_temperature_propagated(self, fake_vertexai) -> None:
        config = OpenDataSciConfig(provider="vertexai", temperature=0.3)  # type: ignore[arg-type]
        create_vertexai_model(config)
        assert fake_vertexai["temperature"] == 0.3

    def test_location_from_config_field(self, fake_vertexai, monkeypatch) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        config = OpenDataSciConfig(provider="vertexai", google_cloud_location="europe-west4")  # type: ignore[arg-type]
        create_vertexai_model(config)
        assert fake_vertexai["location"] == "europe-west4"

    def test_location_falls_back_to_env_var(self, fake_vertexai, monkeypatch) -> None:
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-west1")
        config = OpenDataSciConfig(provider="vertexai")  # type: ignore[arg-type]
        create_vertexai_model(config)
        assert fake_vertexai["location"] == "us-west1"

    def test_location_default_is_none(self, fake_vertexai, monkeypatch) -> None:
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        config = OpenDataSciConfig(provider="vertexai")  # type: ignore[arg-type]
        create_vertexai_model(config)
        assert fake_vertexai["location"] is None

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_google_vertexai", raising=False)
        with patch.dict(sys.modules, {"langchain_google_vertexai": None}):
            with pytest.raises(ValueError, match="langchain-google-vertexai"):
                create_vertexai_model(OpenDataSciConfig(provider="vertexai"))  # type: ignore[arg-type]


class TestCreateVertexAISecondaryModel:
    def test_uses_resolved_secondary_model(self, fake_vertexai) -> None:
        config = OpenDataSciConfig(provider="vertexai", secondary_model="gemini-flash-test")  # type: ignore[arg-type]
        create_vertexai_secondary_model(config)
        assert fake_vertexai["model"] == "gemini-flash-test"

    def test_temperature_is_zero(self, fake_vertexai) -> None:
        create_vertexai_secondary_model(OpenDataSciConfig(provider="vertexai"))  # type: ignore[arg-type]
        assert fake_vertexai["temperature"] == 0

    def test_max_output_tokens_capped(self, fake_vertexai) -> None:
        create_vertexai_secondary_model(OpenDataSciConfig(provider="vertexai"))  # type: ignore[arg-type]
        assert fake_vertexai["max_output_tokens"] == 1000

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_google_vertexai", raising=False)
        with patch.dict(sys.modules, {"langchain_google_vertexai": None}):
            with pytest.raises(ValueError, match="langchain-google-vertexai"):
                create_vertexai_secondary_model(OpenDataSciConfig(provider="vertexai"))  # type: ignore[arg-type]


class TestCachedSystemPrompt:
    def test_returns_prompt_unchanged(self) -> None:
        assert cached_system_prompt("hello world") == "hello world"

    def test_empty_string_returned_as_is(self) -> None:
        assert cached_system_prompt("") == ""
