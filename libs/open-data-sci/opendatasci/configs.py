"""Configuration for OpenDataSci."""

from pathlib import Path
from types import MappingProxyType
from typing import List

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from opendatasci.models.providers import Provider
from opendatasci.skills.local import (
    _BUILTIN_DOMAINS_DIRECTORY as _DEFAULT_BUILTIN_DOMAINS_DIRECTORY,
)
from opendatasci.skills.local import _BUILTIN_SKILLS_DIRECTORY as _DEFAULT_BUILTIN_SKILLS_DIRECTORY
from opendatasci.tools.mcp import MCPServerSpec

DEFAULT_MODEL: MappingProxyType[Provider, str] = MappingProxyType(
    {
        Provider.ANTHROPIC: "claude-sonnet-5",
        Provider.OPENAI: "gpt-5.6-sol",
        Provider.BEDROCK: "us.anthropic.claude-sonnet-5",
        Provider.GEMINI: "gemini-3.5-flash",
        Provider.VERTEXAI: "gemini-3.5-flash",
        Provider.AZURE: "gpt-5.6-sol",
        Provider.OLLAMA: "qwen3.5:9b",
        Provider.OPENAI_COMPATIBLE_SERVER: "Qwen/Qwen3.5-4B",
    }
)

DEFAULT_SECONDARY_MODEL: MappingProxyType[Provider, str] = MappingProxyType(
    {
        Provider.ANTHROPIC: "claude-haiku-4-5",
        Provider.OPENAI: "gpt-5.6-luna",
        Provider.BEDROCK: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        Provider.GEMINI: "gemini-3.1-flash-lite",
        Provider.VERTEXAI: "gemini-3.1-flash-lite",
        Provider.AZURE: "gpt-5.6-luna",
        Provider.OLLAMA: "qwen3.5:9b",
        Provider.OPENAI_COMPATIBLE_SERVER: "Qwen/Qwen3.5-4B",
    }
)

# Maps a provider to the OpenDataSciConfig field that holds its API key.
# Providers that use cloud-native auth (bedrock, vertexai, ollama) have no
# key field, so switching to them never requires an API key check.
PROVIDER_KEY_FIELD: MappingProxyType[Provider, str | None] = MappingProxyType(
    {
        Provider.ANTHROPIC: "anthropic_api_key",
        Provider.OPENAI: "openai_api_key",
        Provider.GEMINI: "google_api_key",
        Provider.AZURE: "azure_api_key",
        Provider.OPENAI_COMPATIBLE_SERVER: "openai_api_key",
        Provider.BEDROCK: None,
        Provider.VERTEXAI: None,
        Provider.OLLAMA: None,
    }
)


class OpenDataSciConfig(BaseSettings):
    """Configuration for OpenDataSci.

    All fields can be set via environment variables (names shown in parentheses)
    or a ``.env`` file.  Pass an instance to :func:`create_agent` or
    :class:`Agent` to apply custom settings.

    Attributes:
        provider:        LLM provider for the primary model.  One of
                         ``"anthropic"``, ``"openai"``, ``"bedrock"``,
                         ``"gemini"``, ``"vertexai"``, ``"azure"``,
                         ``"ollama"``, ``"openai_compatible_server"`` (any
                         self-hosted OpenAI-compatible server, e.g. vLLM).
        model:           Provider-specific model identifier.  Falls back to a
                         sensible default per provider when not set.
        secondary_provider: LLM provider for the secondary (auxiliary) model.
                         Defaults to ``provider`` when not set.  Set to a
                         different provider to mix backends — e.g. Anthropic
                         for the primary model and OpenAI for summarisation.
        secondary_model: Model identifier for lightweight tasks such as memory
                         summarisation.  Falls back to a sensible default per
                         provider when not set.
        anthropic_api_key: API key for Anthropic (``ANTHROPIC_API_KEY``).
        openai_api_key:    API key for OpenAI and OpenAI-compatible servers
                           (``OPENAI_API_KEY``).
        google_api_key:    API key for Google Gemini (``GOOGLE_API_KEY``).
        azure_api_key:     API key for Azure OpenAI (``AZURE_OPENAI_API_KEY``).
                           Omit when using service-principal auth instead.
        aws_region:        AWS region for Bedrock (``REGION``).  Defaults to
                         ``"us-east-1"`` when not set.
        google_cloud_project:  GCP project ID for Vertex AI
                           (``GOOGLE_CLOUD_PROJECT``).
        google_cloud_location: Vertex AI region / location
                           (``GOOGLE_CLOUD_LOCATION``).
        azure_endpoint:    Azure OpenAI resource endpoint URL
                           (``AZURE_OPENAI_ENDPOINT``).  Required when
                           ``provider`` is ``"azure"``.
        azure_api_version: Azure OpenAI API version
                           (``AZURE_OPENAI_API_VERSION``).  Defaults to
                           ``"2025-01-01-preview"``.
        llm_server_base_url: Base URL for self-hosted providers
                           (``LLM_SERVER_BASE_URL``).  Required for
                           ``"ollama"`` and ``"openai_compatible_server"``;
                           falls back to ``http://localhost:11434`` and
                           ``http://localhost:8000/v1`` respectively when
                           not set.
        temperature:     LLM sampling temperature.  Not sent to Anthropic and
                         Bedrock models that use adaptive thinking (Claude
                         4.6+), which reject explicit sampling parameters.
        name:            Display name of the agent.  Defaults to ``"Sai"``.
        mcp_servers: MCP servers the agent may connect to (``MCP_SERVERS``).
                         Only the ``http`` and ``sse`` transports are
                         supported; each server's tools are (re)discovered
                         regularly rather than once at startup.
        skills_directory: Path to a directory of custom skill files
                         (``SKILLS_DIRECTORY``).  Loaded in addition to the
                         built-in skills; custom skills override built-ins of
                         the same name.
        builtin_skills_directory: Path to the built-in skills bundled with
                         the package (``BUILTIN_SKILLS_DIRECTORY``).  Override
                         only if you need to replace the defaults entirely.
        skill_domains_directory: Path to a directory of custom skill
                         domains (``SKILL_DOMAINS_DIRECTORY``).  Loaded in
                         addition to the built-in skill domains; custom
                         domains override built-ins of the same name.
        builtin_skill_domains_directory: Path to the built-in skill
                         domains bundled with the package
                         (``BUILTIN_SKILL_DOMAINS_DIRECTORY``).  Override
                         only if you need to replace the defaults entirely.
        worker_timeout_seconds: Maximum seconds to wait for all spawned
                         workers to finish.  ``None`` disables the timeout.
                         Defaults to ``300.0`` (5 minutes).
        midturn_compaction_threshold: Token count after which the agent's
                         context is compacted mid-turn (during a single turn's
                         reasoning/acting loop).  Must be strictly positive.
                         Defaults to ``96000``.
        local_code_exec_timeout: Maximum seconds allowed for a single
                         code-execution run in the local sandbox
                         (``CODE_EXEC_TIMEOUT``).  Defaults to
                         ``1800`` (30 minutes).
    Cloud authentication (environment variables read by the underlying SDKs):

    **AWS Bedrock** — boto3 credential chain (pick one):

    - Long-lived IAM key: ``AWS_ACCESS_KEY_ID`` + ``AWS_SECRET_ACCESS_KEY``
    - Temporary STS credentials: add ``AWS_SESSION_TOKEN`` to the above.
    - EC2 / ECS / Lambda: credentials are fetched automatically from the
      instance metadata service; no env vars required.

    **Google Vertex AI** — Application Default Credentials chain (pick one):

    - Service account JSON key: set ``GOOGLE_APPLICATION_CREDENTIALS`` to
      the path of the key file; also set ``GOOGLE_CLOUD_PROJECT``.
    - User credentials: run ``gcloud auth application-default login``.
    - Cloud Run / GCE / GKE: credentials are fetched automatically;
      ``GOOGLE_CLOUD_PROJECT`` is still required.

    **Azure OpenAI** — API key *or* service principal (not both):

    - API key: ``AZURE_OPENAI_API_KEY``
    - Service principal (requires ``pip install 'open-data-sci[azure]'``):
      set ``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID``, and
      ``AZURE_CLIENT_SECRET``.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        populate_by_name=True,
        env_ignore_empty=True,
        env_file=".env",
        extra="ignore",
    )

    # ── Model selection ───────────────────────────────────────────────────────
    provider: Provider = Field(default=Provider.ANTHROPIC, alias="PROVIDER")
    model: str = Field(default=DEFAULT_MODEL[Provider.ANTHROPIC], alias="MODEL")
    secondary_provider: Provider = Field(default=Provider.ANTHROPIC, alias="SECONDARY_PROVIDER")
    secondary_model: str = Field(
        default=DEFAULT_SECONDARY_MODEL[Provider.ANTHROPIC], alias="SECONDARY_MODEL"
    )

    # ── Per-provider API keys (loaded from env via alias) ─────────────────────
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    azure_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")

    # ── Cloud region / location ───────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1", alias="REGION")
    google_cloud_project: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str | None = Field(default=None, alias="GOOGLE_CLOUD_LOCATION")

    # ── Azure-specific ────────────────────────────────────────────────────────
    azure_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_api_version: str = Field(default="2025-01-01-preview", alias="AZURE_OPENAI_API_VERSION")

    # ── Self-hosted endpoint (Ollama / OpenAI-compatible server) ──────────────
    llm_server_base_url: str | None = Field(default=None, alias="LLM_SERVER_BASE_URL")

    # ── Sampling & reasoning ──────────────────────────────────────────────────
    temperature: float = Field(default=0.0, alias="TEMPERATURE")

    # ── Agent Customization ───────────────────────────────────────────────────────
    name: str = Field(default="Sai", alias="NAME")

    # ── MCP ───────────────────────────────────────────────────────────
    mcp_servers: List[MCPServerSpec] = Field(default_factory=list, alias="MCP_SERVERS")

    # ── Context management ───────────────────────────────────────────────────────
    midturn_compaction_threshold: int = Field(
        default=96000,
        alias="MIDTURN_COMPACTION_THRESHOLD",
        gt=0,
        description="Token count after which the agent's context is compacted mid-turn. Only applies in execution mode.",
    )

    # ── Skills ───────────────────────────────────────────────────────────────────
    skills_directory: Path | None = Field(
        default=None,
        alias="SKILLS_DIRECTORY",
    )
    builtin_skills_directory: Path = Field(
        default=_DEFAULT_BUILTIN_SKILLS_DIRECTORY,
        alias="BUILTIN_SKILLS_DIRECTORY",
    )
    skill_domains_directory: Path | None = Field(
        default=None,
        alias="SKILL_DOMAINS_DIRECTORY",
    )
    builtin_skill_domains_directory: Path = Field(
        default=_DEFAULT_BUILTIN_DOMAINS_DIRECTORY,
        alias="BUILTIN_SKILL_DOMAINS_DIRECTORY",
    )

    # ── Worker configuration ───────────────────────────────────────────────────────
    worker_timeout_seconds: float | None = Field(
        default=300.0,
        alias="WORKER_TIMEOUT_SECONDS",
    )

    # ── Sandbox ───────────────────────────────────────────────────────────────────
    local_code_exec_timeout: int = Field(
        default=1800,  # 30 minutes
        alias="CODE_EXEC_TIMEOUT",
        gt=0,
        description="Maximum seconds allowed for a single local code-execution run.",
    )

    @model_validator(mode="after")
    def _validate_providers(self) -> "OpenDataSciConfig":
        if self.provider not in DEFAULT_MODEL:
            supported = ", ".join(f"'{p}'" for p in sorted(DEFAULT_MODEL))
            raise ValueError(
                f"Unknown provider '{self.provider}'. Supported providers: {supported}."
            )
        if self.secondary_provider is not None and self.secondary_provider not in DEFAULT_MODEL:
            supported = ", ".join(f"'{p}'" for p in sorted(DEFAULT_MODEL))
            raise ValueError(
                f"Unknown secondary model provider '{self.secondary_provider}'. "
                f"Supported providers: {supported}."
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OpenDataSciConfig":
        """Load an ``OpenDataSciConfig`` from a YAML file.

        The file must be a YAML mapping whose keys match ``OpenDataSciConfig``
        field names.  Unknown keys are ignored, so config files written for
        other versions keep loading.  Environment variables are still applied
        for any field not present in the file.

        Raises:
            ImportError: If PyYAML is not installed.
            ValueError: If the file does not contain a mapping.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. Install it with: pip install pyyaml"
            ) from exc

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"YAML config at '{path}' must be a mapping (key: value pairs), "
                f"got {type(data).__name__}"
            )

        # Ignore unknown keys so config files written for other versions
        # (e.g. containing removed fields) keep loading.
        valid_fields = set(cls.model_fields.keys())
        known = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**known)
