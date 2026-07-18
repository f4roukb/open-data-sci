"""Component tests: per-provider model factories (``opendatasci.models.*``).

Each provider module exposes ``create_*_model`` / ``create_*_secondary_model``
factories plus a ``cached_system_prompt`` helper.  These tests exercise:

* the real construction path where the provider SDK is a hard dependency
  (``langchain_openai`` → openai / azure / openai-compatible), asserting the
  exact parameters the factory wires into the client;
* the construction path for optional SDKs (gemini / vertexai / bedrock) via a
  fake module injected into ``sys.modules``, asserting the kwargs contract;
* the missing-dependency path, asserting the actionable ``ValueError``;
* the ``cached_system_prompt`` contract per provider.
"""

import importlib
import sys
from types import ModuleType
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

from opendatasci.configs import OpenDataSciConfig
from opendatasci.models import aws, google, microsoft, openai
from opendatasci.models import local as local_models


@pytest.fixture
def openai_config() -> OpenDataSciConfig:
    return OpenDataSciConfig(
        provider="openai",
        model="gpt-4o",
        secondary_provider="openai",
        secondary_model="gpt-4o-mini",
        openai_api_key="sk-test",
        temperature=0.3,
    )


class _RecordingModel:
    """Stands in for an optional provider's chat-model class; records kwargs."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _fake_module(name: str, **attrs: object) -> ModuleType:
    mod = ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIFactory:
    def test_primary_model_parameters(self, openai_config) -> None:
        model = openai.create_openai_model(openai_config)
        assert model.model_name == "gpt-4o"
        assert model.temperature == 0.3
        assert model.openai_api_key.get_secret_value() == "sk-test"
        assert model.reasoning_effort == "medium"
        assert model.model_kwargs["prompt_cache_key"] == "open-data-sci-system-v1"

    def test_secondary_model_is_cheap_and_deterministic(self, openai_config) -> None:
        model = openai.create_openai_secondary_model(openai_config)
        assert model.model_name == "gpt-4o-mini"
        assert model.temperature == 0
        assert model.max_tokens == 1000
        assert model.model_kwargs["prompt_cache_key"] == "open-data-sci-system-v1"

    @pytest.mark.parametrize(
        "factory", [openai.create_openai_model, openai.create_openai_secondary_model]
    )
    def test_missing_dependency_raises_value_error(self, factory, openai_config) -> None:
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ValueError, match="langchain-openai is not installed"):
                factory(openai_config)

    def test_cached_system_prompt_is_identity(self) -> None:
        assert openai.cached_system_prompt("sys prompt") == "sys prompt"


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------


@pytest.fixture
def azure_config() -> OpenDataSciConfig:
    return OpenDataSciConfig(
        provider="azure",
        model="my-deployment",
        secondary_provider="azure",
        secondary_model="my-mini-deployment",
        azure_api_key="azure-key",
        azure_endpoint="https://example.openai.azure.com",
        temperature=0.4,
    )


class TestAzureFactory:
    def test_primary_model_parameters(self, azure_config) -> None:
        model = microsoft.create_azure_model(azure_config)
        assert model.deployment_name == "my-deployment"
        assert model.azure_endpoint == "https://example.openai.azure.com"
        assert model.temperature == 0.4
        assert model.model_kwargs["prompt_cache_key"] == "open-data-sci-system-v1"

    def test_secondary_model_is_cheap_and_deterministic(self, azure_config) -> None:
        model = microsoft.create_azure_secondary_model(azure_config)
        assert model.deployment_name == "my-mini-deployment"
        assert model.temperature == 0
        assert model.max_tokens == 1000

    @pytest.mark.parametrize(
        "factory", [microsoft.create_azure_model, microsoft.create_azure_secondary_model]
    )
    def test_missing_endpoint_raises_value_error(self, factory, azure_config) -> None:
        config = azure_config.model_copy(update={"azure_endpoint": None})
        with pytest.raises(ValueError, match="Azure OpenAI endpoint is not configured"):
            factory(config)

    @pytest.mark.parametrize(
        "factory", [microsoft.create_azure_model, microsoft.create_azure_secondary_model]
    )
    def test_missing_dependency_raises_value_error(self, factory, azure_config) -> None:
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ValueError, match="langchain-openai is not installed"):
                factory(azure_config)

    def test_cached_system_prompt_is_identity(self) -> None:
        assert microsoft.cached_system_prompt("sys prompt") == "sys prompt"


# ---------------------------------------------------------------------------
# Local / self-hosted (Ollama + OpenAI-compatible servers)
# ---------------------------------------------------------------------------


@pytest.fixture
def compat_config() -> OpenDataSciConfig:
    return OpenDataSciConfig(
        provider="openai_compatible_server",
        model="llama3",
        secondary_provider="openai_compatible_server",
        secondary_model="llama3-small",
        openai_api_key=None,
        llm_server_base_url=None,
        temperature=0.2,
    )


class TestOpenAICompatibleFactory:
    def test_primary_defaults_base_url_and_api_key(self, compat_config) -> None:
        model = local_models.create_openai_compatible_model(compat_config)
        assert model.model_name == "llama3"
        assert model.openai_api_base == "http://localhost:8000/v1"
        assert model.openai_api_key.get_secret_value() == "EMPTY"
        assert model.temperature == 0.2

    def test_primary_uses_configured_base_url_and_key(self, compat_config) -> None:
        config = compat_config.model_copy(
            update={"llm_server_base_url": "http://gpu-box:9000/v1", "openai_api_key": "k"}
        )
        model = local_models.create_openai_compatible_model(config)
        assert model.openai_api_base == "http://gpu-box:9000/v1"
        assert model.openai_api_key.get_secret_value() == "k"

    def test_secondary_model_is_cheap_and_deterministic(self, compat_config) -> None:
        model = local_models.create_openai_compatible_secondary_model(compat_config)
        assert model.model_name == "llama3-small"
        assert model.temperature == 0
        assert model.max_tokens == 1000
        assert model.openai_api_base == "http://localhost:8000/v1"

    @pytest.mark.parametrize(
        "factory",
        [
            local_models.create_openai_compatible_model,
            local_models.create_openai_compatible_secondary_model,
        ],
    )
    def test_missing_dependency_raises_value_error(self, factory, compat_config) -> None:
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ValueError, match="langchain-openai is not installed"):
                factory(compat_config)

    def test_cached_system_prompt_is_identity(self) -> None:
        assert local_models.cached_system_prompt("sys prompt") == "sys prompt"


@pytest.fixture
def ollama_config() -> OpenDataSciConfig:
    return OpenDataSciConfig(
        provider="ollama",
        model="llama3",
        secondary_provider="ollama",
        secondary_model="llama3-small",
        llm_server_base_url=None,
        temperature=0.2,
    )


class TestOllamaFactory:
    @pytest.mark.parametrize(
        "factory",
        [local_models.create_ollama_model, local_models.create_ollama_secondary_model],
    )
    def test_missing_dependency_raises_value_error(self, factory, ollama_config) -> None:
        with patch.dict(sys.modules, {"langchain_ollama": None}):
            with pytest.raises(ValueError, match="langchain-ollama is not installed"):
                factory(ollama_config)

    def test_primary_model_kwargs(self, ollama_config) -> None:
        fake = _fake_module("langchain_ollama", ChatOllama=_RecordingModel)
        with patch.dict(sys.modules, {"langchain_ollama": fake}):
            model = local_models.create_ollama_model(ollama_config)
        assert model.kwargs["model"] == "llama3"
        assert model.kwargs["temperature"] == 0.2

    def test_secondary_model_kwargs(self, ollama_config) -> None:
        fake = _fake_module("langchain_ollama", ChatOllama=_RecordingModel)
        with patch.dict(sys.modules, {"langchain_ollama": fake}):
            model = local_models.create_ollama_secondary_model(ollama_config)
        assert model.kwargs["model"] == "llama3-small"
        assert model.kwargs["temperature"] == 0
        assert model.kwargs["num_predict"] == 1000


# ---------------------------------------------------------------------------
# Google (Gemini API + Vertex AI)
# ---------------------------------------------------------------------------


@pytest.fixture
def gemini_config() -> OpenDataSciConfig:
    return OpenDataSciConfig(
        provider="gemini",
        model="gemini-2.5-pro",
        secondary_provider="gemini",
        secondary_model="gemini-2.5-flash",
        google_api_key="g-key",
        google_cloud_project="my-project",
        google_cloud_location="europe-west1",
        temperature=0.5,
    )


class TestGeminiFactory:
    @pytest.mark.parametrize(
        "factory", [google.create_gemini_model, google.create_gemini_secondary_model]
    )
    def test_missing_dependency_raises_value_error(self, factory, gemini_config) -> None:
        with patch.dict(sys.modules, {"langchain_google_genai": None}):
            with pytest.raises(ValueError, match="langchain-google-genai is not installed"):
                factory(gemini_config)

    def test_primary_model_kwargs(self, gemini_config) -> None:
        fake = _fake_module("langchain_google_genai", ChatGoogleGenerativeAI=_RecordingModel)
        with patch.dict(sys.modules, {"langchain_google_genai": fake}):
            model = google.create_gemini_model(gemini_config)
        assert model.kwargs == {
            "model": "gemini-2.5-pro",
            "google_api_key": "g-key",
            "temperature": 0.5,
        }

    def test_secondary_model_kwargs(self, gemini_config) -> None:
        fake = _fake_module("langchain_google_genai", ChatGoogleGenerativeAI=_RecordingModel)
        with patch.dict(sys.modules, {"langchain_google_genai": fake}):
            model = google.create_gemini_secondary_model(gemini_config)
        assert model.kwargs == {
            "model": "gemini-2.5-flash",
            "google_api_key": "g-key",
            "temperature": 0,
            "max_output_tokens": 1000,
        }

    def test_cached_system_prompt_is_identity(self) -> None:
        assert google.cached_system_prompt("sys prompt") == "sys prompt"


class TestVertexAIFactory:
    @pytest.mark.parametrize(
        "factory", [google.create_vertexai_model, google.create_vertexai_secondary_model]
    )
    def test_missing_dependency_raises_value_error(self, factory, gemini_config) -> None:
        with patch.dict(sys.modules, {"langchain_google_vertexai": None}):
            with pytest.raises(ValueError, match="langchain-google-vertexai is not installed"):
                factory(gemini_config)

    def test_primary_model_kwargs(self, gemini_config) -> None:
        fake = _fake_module("langchain_google_vertexai", ChatVertexAI=_RecordingModel)
        with patch.dict(sys.modules, {"langchain_google_vertexai": fake}):
            model = google.create_vertexai_model(gemini_config)
        assert model.kwargs == {
            "model": "gemini-2.5-pro",
            "project": "my-project",
            "location": "europe-west1",
            "temperature": 0.5,
        }

    def test_secondary_model_kwargs(self, gemini_config) -> None:
        fake = _fake_module("langchain_google_vertexai", ChatVertexAI=_RecordingModel)
        with patch.dict(sys.modules, {"langchain_google_vertexai": fake}):
            model = google.create_vertexai_secondary_model(gemini_config)
        assert model.kwargs["model"] == "gemini-2.5-flash"
        assert model.kwargs["temperature"] == 0
        assert model.kwargs["max_output_tokens"] == 1000


# ---------------------------------------------------------------------------
# AWS Bedrock
# ---------------------------------------------------------------------------


@pytest.fixture
def bedrock_config() -> OpenDataSciConfig:
    return OpenDataSciConfig(
        provider="bedrock",
        model="anthropic.claude-sonnet-4-6",
        secondary_provider="bedrock",
        secondary_model="anthropic.claude-haiku",
        aws_region="eu-west-1",
    )


def _chunk(usage: dict | None) -> ChatGenerationChunk:
    return ChatGenerationChunk(
        message=AIMessageChunk(content="hi", usage_metadata=usage)
    )


class _FakeBedrockBase:
    """Fake ``ChatBedrockConverse``: records kwargs and streams scripted chunks."""

    scripted_chunks: list[ChatGenerationChunk] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield from self.scripted_chunks


@pytest.fixture
def aws_with_fake_sdk():
    """Reload ``opendatasci.models.aws`` against a fake ``langchain_aws``.

    The module resolves its Bedrock base class at import time, so covering the
    installed-SDK branch requires a reload with the fake in ``sys.modules``.
    A second reload afterwards restores the module to the real environment.
    """
    fake = _fake_module("langchain_aws", ChatBedrockConverse=_FakeBedrockBase)
    with patch.dict(sys.modules, {"langchain_aws": fake}):
        reloaded = importlib.reload(aws)
        yield reloaded
    importlib.reload(aws)


class TestBedrockFactory:
    def test_missing_dependency_raises_value_error(self, bedrock_config) -> None:
        if aws._BedrockBase is not None:
            pytest.skip("langchain-aws installed in this environment")
        with pytest.raises(ValueError, match="langchain-aws is not installed"):
            aws.create_bedrock_model(bedrock_config)
        with pytest.raises(ValueError, match="langchain-aws is not installed"):
            aws.create_bedrock_secondary_model(bedrock_config)

    def test_primary_model_kwargs(self, aws_with_fake_sdk, bedrock_config) -> None:
        model = aws_with_fake_sdk.create_bedrock_model(bedrock_config)
        assert model.kwargs["model"] == "anthropic.claude-sonnet-4-6"
        assert model.kwargs["region_name"] == "eu-west-1"
        assert model.kwargs["temperature"] == 1  # extended thinking requires temperature 1
        assert model.kwargs["max_tokens"] == 16000
        assert model.kwargs["additional_model_request_fields"] == {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        }
        assert model.kwargs["disable_streaming"] is False

    def test_secondary_model_kwargs(self, aws_with_fake_sdk, bedrock_config) -> None:
        model = aws_with_fake_sdk.create_bedrock_secondary_model(bedrock_config)
        assert model.kwargs["model"] == "anthropic.claude-haiku"
        assert model.kwargs["temperature"] == 0
        assert model.kwargs["max_tokens"] == 1000

    def test_stream_strips_list_valued_usage_fields(
        self, aws_with_fake_sdk, bedrock_config
    ) -> None:
        dirty = _chunk(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cacheDetails": [{"cacheType": "default"}],
            }
        )
        _FakeBedrockBase.scripted_chunks = [dirty]
        try:
            model = aws_with_fake_sdk.create_bedrock_model(bedrock_config)
            chunks = list(model._stream([HumanMessage(content="q")]))
        finally:
            _FakeBedrockBase.scripted_chunks = []
        assert len(chunks) == 1
        cleaned = chunks[0].message.usage_metadata
        assert "cacheDetails" not in cleaned
        assert cleaned["input_tokens"] == 10
        assert cleaned["total_tokens"] == 15


class TestStripListUsageFields:
    def test_chunk_without_lists_is_returned_unchanged(self) -> None:
        chunk = _chunk({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})
        assert aws._strip_list_usage_fields(chunk) is chunk

    def test_chunk_without_usage_metadata_is_returned_unchanged(self) -> None:
        chunk = _chunk(None)
        assert aws._strip_list_usage_fields(chunk) is chunk

    def test_list_fields_are_removed_and_ints_preserved(self) -> None:
        chunk = _chunk(
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "cacheDetails": [],
            }
        )
        cleaned = aws._strip_list_usage_fields(chunk)
        assert cleaned is not chunk
        assert cleaned.message.usage_metadata == {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }
        assert cleaned.message.content == "hi"


class TestBedrockCachedSystemPrompt:
    def test_wraps_prompt_with_cache_point(self) -> None:
        assert aws.cached_system_prompt("sys") == [
            {"type": "text", "text": "sys"},
            {"cachePoint": {"type": "default"}},
        ]
