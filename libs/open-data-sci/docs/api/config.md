# OpenDataSciConfig

`OpenDataSciConfig` is the single configuration object accepted by `create_agent` and `Agent`. Every field can be set via:

1. Direct constructor argument
2. Environment variable (see the alias column below)
3. `.env` file in the current working directory (loaded automatically)
4. A YAML config file via `OpenDataSciConfig.from_yaml(path)`

TUI flags always take precedence over all of the above.

## Quick reference

```python
from opendatasci import OpenDataSciConfig

# Defaults — uses Anthropic with claude-sonnet-5
config = OpenDataSciConfig()

# OpenAI with custom temperature
config = OpenDataSciConfig(
    provider="openai",
    model="gpt-5.6-sol",
    openai_api_key="sk-...",
    primary_temperature=0.2,
)

# Mixed providers — Anthropic for the primary model, OpenAI for summarisation
config = OpenDataSciConfig(
    provider="anthropic",
    secondary_provider="openai",
    secondary_model="gpt-5.6-luna",
    openai_api_key="sk-...",
)

# Load from YAML
config = OpenDataSciConfig.from_yaml("opendatasci_config.yaml")
```

## Field reference

### Model selection

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `provider` | `PROVIDER` | `"anthropic"` | LLM provider for the primary model |
| `model` | `MODEL` | *(provider default)* | Primary model identifier |
| `secondary_provider` | `SECONDARY_PROVIDER` | `"anthropic"` | Provider for the secondary model |
| `secondary_model` | `SECONDARY_MODEL` | *(provider default)* | Secondary model for lightweight tasks |

### API keys

| Field | Env var | Description |
|-------|---------|-------------|
| `anthropic_api_key` | `ANTHROPIC_API_KEY` | Anthropic API key |
| `openai_api_key` | `OPENAI_API_KEY` | OpenAI / vLLM API key |
| `google_api_key` | `GOOGLE_API_KEY` | Google Gemini API key |
| `azure_api_key` | `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |

### Cloud / endpoint settings

| Field | Env var | Description |
|-------|---------|-------------|
| `aws_region` | `REGION` | AWS region for Bedrock |
| `google_cloud_project` | `GOOGLE_CLOUD_PROJECT` | GCP project ID for Vertex AI |
| `google_cloud_location` | `GOOGLE_CLOUD_LOCATION` | Vertex AI region |
| `azure_endpoint` | `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint URL |
| `azure_api_version` | `AZURE_OPENAI_API_VERSION` | Azure API version (default: `2025-01-01-preview`) |
| `llm_server_base_url` | `LLM_SERVER_BASE_URL` | Custom endpoint for Ollama / OpenAI-compatible server |

### Sampling & reasoning

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `primary_temperature` | `PRIMARY_TEMPERATURE` | `0.0` | LLM sampling temperature for the primary model (not sent to Claude 4.6+ / Sonnet 5 models) |

### Agent behaviour

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `name` | `NAME` | `"Sai"` | Agent display name, injected into all system prompts |
| `autocompaction_threshold` | `AUTOCOMPACTION_THRESHOLD` | `96000` | Token count after which the agent's context is compacted mid-turn |
| `worker_timeout_seconds` | `WORKER_TIMEOUT_SECONDS` | `300.0` | Max seconds for all spawned workers to finish (`null` = no timeout) |

### Skills

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `skills_directory` | `SKILLS_DIRECTORY` | `None` | Path to a user-defined skills directory |
| `builtin_skills_directory` | `BUILTIN_SKILLS_DIRECTORY` | *(bundled)* | Path to the built-in skills directory |

### Sandbox

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `local_code_exec_timeout` | `CODE_EXEC_TIMEOUT` | `1800` | Max seconds for a single local sandbox execution |

### MCP

| Field | Env var | Default | Description |
|-------|---------|---------|-------------|
| `mcp_servers` | `MCP_SERVERS` | `[]` | MCP tool server URLs |

---

## Default models per provider

| Provider | Primary model | Secondary model |
|----------|--------------|-----------------|
| `anthropic` | `claude-sonnet-5` | `claude-haiku-4-5` |
| `openai` | `gpt-5.6-sol` | `gpt-5.6-luna` |
| `bedrock` | `us.anthropic.claude-sonnet-5` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| `gemini` | `gemini-3.5-flash` | `gemini-3.1-flash-lite` |
| `vertexai` | `gemini-3.5-flash` | `gemini-3.1-flash-lite` |
| `azure` | `gpt-5.6-sol` | `gpt-5.6-luna` |
| `ollama` | `qwen3.5:9b` | `qwen3.5:9b` |
| `openai_compatible_server` | `Qwen/Qwen3.5-4B` | `Qwen/Qwen3.5-4B` |

---

## YAML config file

```yaml
# opendatasci_config.yaml
provider: anthropic
model: claude-sonnet-5

secondary_provider: openai
secondary_model: gpt-5.6-luna

primary_temperature: 0.1

name: Sai

worker_timeout_seconds: 600
autocompaction_threshold: 80000
```

```python
config = OpenDataSciConfig.from_yaml("opendatasci_config.yaml")
```

Unknown keys in the YAML file are ignored, so config files written for other versions keep loading.

---

## API reference

::: opendatasci.configs.OpenDataSciConfig
    options:
      show_root_heading: true
      show_source: false
      members:
        - from_yaml
