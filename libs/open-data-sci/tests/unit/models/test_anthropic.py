"""Unit tests for opendatasci.models.anthropic factory functions."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models.anthropic import (
    cached_system_prompt,
    create_anthropic_model,
    create_anthropic_secondary_model,
)


@pytest.fixture
def fake_chat_anthropic(monkeypatch):
    """Insert a fake ``langchain_anthropic`` module with a recording ``ChatAnthropic``."""
    captured: dict = {}

    def _ctor(**kwargs):
        captured.update(kwargs)
        return MagicMock(name="ChatAnthropic", **kwargs)

    fake_module = MagicMock(name="langchain_anthropic_stub")
    fake_module.ChatAnthropic = _ctor
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_module)
    return captured


class TestCreateAnthropicPrimaryModel:
    def test_uses_resolved_model(self, fake_chat_anthropic) -> None:
        config = OpenDataSciConfig(provider="anthropic", model="claude-test-1")  # type: ignore[arg-type]
        create_anthropic_model(config)
        assert fake_chat_anthropic["model"] == "claude-test-1"

    def test_falls_back_to_provider_default_when_model_unset(self, fake_chat_anthropic) -> None:
        config = OpenDataSciConfig(provider="anthropic")  # type: ignore[arg-type]
        create_anthropic_model(config)
        assert fake_chat_anthropic["model"] == config.model

    def test_non_adaptive_models_run_without_thinking(self, fake_chat_anthropic) -> None:
        # Models without adaptive thinking (e.g. haiku-4-5) run without a
        # thinking config and receive the configured temperature.
        config = OpenDataSciConfig(
            provider="anthropic", model="claude-haiku-4-5", primary_temperature=0.3
        )  # type: ignore[arg-type]
        create_anthropic_model(config)
        assert "thinking" not in fake_chat_anthropic
        assert fake_chat_anthropic["temperature"] == 0.3

    @pytest.mark.parametrize(
        "model_id",
        ["claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-4-8"],
    )
    def test_modern_models_use_adaptive_thinking(self, fake_chat_anthropic, model_id) -> None:
        # Claude 4.6+ / Sonnet 5: adaptive thinking, no sampling params.
        config = OpenDataSciConfig(provider="anthropic", model=model_id)  # type: ignore[arg-type]
        create_anthropic_model(config)
        assert fake_chat_anthropic["thinking"] == {"type": "adaptive"}
        assert "temperature" not in fake_chat_anthropic

    def test_api_key_from_config_preferred_over_env(self, fake_chat_anthropic, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = OpenDataSciConfig(provider="anthropic", anthropic_api_key="config-key")  # type: ignore[arg-type]
        create_anthropic_model(config)
        assert fake_chat_anthropic["api_key"] == "config-key"

    def test_api_key_falls_back_to_env_var(self, fake_chat_anthropic, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-fallback")
        config = OpenDataSciConfig(provider="anthropic")  # type: ignore[arg-type]
        create_anthropic_model(config)
        assert fake_chat_anthropic["api_key"] == "env-fallback"

    def test_max_tokens_set_for_thinking(self, fake_chat_anthropic) -> None:
        create_anthropic_model(OpenDataSciConfig(provider="anthropic"))  # type: ignore[arg-type]
        # Large max_tokens budget gives adaptive thinking room to work.
        assert fake_chat_anthropic["max_tokens"] == 16000

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        # ``None`` in sys.modules forces ``import`` to raise ImportError.
        monkeypatch.delitem(sys.modules, "langchain_anthropic", raising=False)
        with patch.dict(sys.modules, {"langchain_anthropic": None}):
            with pytest.raises(ValueError, match="langchain-anthropic"):
                create_anthropic_model(OpenDataSciConfig(provider="anthropic"))  # type: ignore[arg-type]


class TestCreateAnthropicSecondaryModel:
    def test_uses_resolved_secondary_model(self, fake_chat_anthropic) -> None:
        config = OpenDataSciConfig(provider="anthropic", secondary_model="haiku-test")  # type: ignore[arg-type]
        create_anthropic_secondary_model(config)
        assert fake_chat_anthropic["model"] == "haiku-test"

    def test_temperature_is_zero(self, fake_chat_anthropic) -> None:
        create_anthropic_secondary_model(OpenDataSciConfig(provider="anthropic"))  # type: ignore[arg-type]
        assert fake_chat_anthropic["temperature"] == 0

    def test_no_thinking_block(self, fake_chat_anthropic) -> None:
        # Secondary model is for auxiliary tasks; extended thinking must be disabled.
        create_anthropic_secondary_model(OpenDataSciConfig(provider="anthropic"))  # type: ignore[arg-type]
        assert "thinking" not in fake_chat_anthropic

    def test_max_tokens_is_capped_low(self, fake_chat_anthropic) -> None:
        create_anthropic_secondary_model(OpenDataSciConfig(provider="anthropic"))  # type: ignore[arg-type]
        assert fake_chat_anthropic["max_tokens"] == 1000

    def test_missing_package_raises_llm_provider_error(self, monkeypatch) -> None:
        monkeypatch.delitem(sys.modules, "langchain_anthropic", raising=False)
        with patch.dict(sys.modules, {"langchain_anthropic": None}):
            with pytest.raises(ValueError, match="langchain-anthropic"):
                create_anthropic_secondary_model(OpenDataSciConfig(provider="anthropic"))  # type: ignore[arg-type]


class TestCachedSystemPrompt:
    def test_wraps_in_ephemeral_cache_breakpoint(self) -> None:
        # Anthropic uses explicit cache_control breakpoints rather than automatic caching.
        assert cached_system_prompt("hello") == [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
        ]
