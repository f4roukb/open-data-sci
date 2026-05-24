# Getting Started

## Installation

```bash
pip install open-data-sci
```

**Requirements:** Python 3.11

### Provider extras

The default installation includes the Anthropic and OpenAI clients. Install additional extras to unlock other providers:

| Extra | Provider |
|-------|----------|
| `open-data-sci[aws]` | AWS Bedrock |
| `open-data-sci[gemini]` | Google Gemini (AI Studio) |
| `open-data-sci[gcp]` | Google Vertex AI |
| `open-data-sci[azure]` | Azure OpenAI |
| `open-data-sci[ollama]` | Ollama (local models) |
| `open-data-sci[vllm]` | vLLM (self-hosted) |

### Capability extras

```bash
pip install "open-data-sci[jax]"   # Deep learning — JAX, Flax, Optax
```

The `[jax]` extra is required for the built-in **Deep Learning** skill. Combine extras freely:

```bash
pip install "open-data-sci[aws,gemini,jax]"
```

---

## Choosing a provider

OpenDataSci works with every major LLM provider. Select one via the `--provider` TUI flag or the `provider` field in `OpenDataSciConfig`.

| Provider | `--provider` | Default model | Auth |
|----------|-------------|---------------|------|
| Anthropic *(default)* | `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-5.5` | `OPENAI_API_KEY` |
| AWS Bedrock | `bedrock` | `us.anthropic.claude-sonnet-4-6` | boto3 credential chain |
| Google Gemini | `gemini` | `gemini-2.5-pro` | `GOOGLE_API_KEY` |
| Google Vertex AI | `vertexai` | `gemini-2.5-pro` | Application Default Credentials |
| Azure OpenAI | `azure` | `gpt-4o` | `AZURE_OPENAI_API_KEY` or service principal |
| Ollama | `ollama` | `llama3.2:3b` | none (local server) |
| vLLM | `vllm` | `meta-llama/Llama-3.2-3B-Instruct` | none (self-hosted) |

Run `opendatasci --list-providers` to print this table at any time.

### Authentication

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...
    opendatasci data.csv
    ```

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY=sk-...
    opendatasci data.csv --provider openai
    ```

=== "AWS Bedrock"

    ```bash
    # Long-lived IAM key
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export REGION=us-east-1
    opendatasci data.csv --provider bedrock

    # Or use an IAM role / EC2 instance profile — no env vars needed
    ```

=== "Google Gemini"

    ```bash
    export GOOGLE_API_KEY=AIza...
    opendatasci data.csv --provider gemini
    ```

=== "Google Vertex AI"

    ```bash
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT=my-project
    export GOOGLE_CLOUD_LOCATION=us-central1
    opendatasci data.csv --provider vertexai
    ```

=== "Azure OpenAI"

    ```bash
    export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
    export AZURE_OPENAI_API_KEY=...
    opendatasci data.csv --provider azure --model gpt-4o
    ```

=== "Ollama"

    ```bash
    # Start Ollama first: ollama serve
    opendatasci data.csv --provider ollama --model llama3.2:3b
    ```

=== "vLLM"

    ```bash
    # Start vLLM first: vllm serve meta-llama/Llama-3.2-3B-Instruct
    opendatasci data.csv --provider vllm --model meta-llama/Llama-3.2-3B-Instruct
    ```

---

## TUI quick start

```bash
# Analyse a single file with the default Anthropic provider
opendatasci data.csv

# Load an entire directory of data files
opendatasci ./my-project/

# Change model
opendatasci data.csv --provider openai --model gpt-4o

# Mix providers — heavy primary model, lightweight secondary
opendatasci data.csv --provider anthropic --secondary-provider openai --secondary-model gpt-5.4-mini

# Colour-blind-safe theme
opendatasci data.csv --theme accessible

# Load settings from a YAML file (TUI flags override individual fields)
opendatasci data.csv --config opendatasci_config.yaml
```

### All TUI options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `anthropic` | LLM provider for the primary model |
| `--model` | *(provider default)* | Primary model identifier |
| `--secondary-provider` | *(same as `--provider`)* | Provider for the secondary model |
| `--secondary-model` | *(provider default)* | Secondary model for lightweight tasks |
| `--api-key` | *(env var)* | API key for the primary provider |
| `--theme` | `default` | Colour theme: `default`, `accessible`, `light`, `solarized`, `dracula` |
| `--config` | | Path to a YAML config file |
| `--list-providers` | | Print all providers and default models, then exit |
| `--version` | | Print the installed version, then exit |

---

## Python SDK quick start

The Python API is async-first. Every public method that touches the network is a coroutine or an async generator.

### Minimal example

```python
import asyncio
from opendatasci import create_agent

async def main() -> None:
    async with create_agent("sales.csv") as agent:
        async for event in agent.astream("What is the average revenue by region?"):
            if event.type == "token":
                print(event.content, end="", flush=True)
            elif event.type == "response":
                print()  # newline after final answer

asyncio.run(main())
```

### With a custom provider

```python
from opendatasci import create_agent, OpenDataSciConfig

config = OpenDataSciConfig(
    provider="openai",
    model="gpt-4o",
    openai_api_key="sk-...",
    temperature=0.2,
)

async with create_agent("data.parquet", config=config) as agent:
    async for event in agent.astream("Train a gradient-boosting model on the target column."):
        ...
```

### Consuming stream events

`agent.astream()` yields [`AgentStreamEvent`](api/types.md) objects. Each event has a `type` and `content` string, plus optional `metadata`.

```python
async for event in agent.astream(query):
    match event.type:
        case "token":
            # Incremental response text
            print(event.content, end="", flush=True)
        case "reasoning":
            # Extended-thinking token (Anthropic / Bedrock only)
            pass
        case "tool_call":
            print(f"\n[tool] {event.content}")
        case "tool_result":
            pass
        case "worker_done":
            idx = event.metadata["worker_idx"]
            ok = event.metadata["success"]
            print(f"\n[worker {idx}] {'ok' if ok else 'failed'}")
        case "input_required":
            # Agent needs a choice from the user — resume by calling astream() again
            choice = input(event.content + " ")
            async for follow_up in agent.astream(choice):
                pass  # handle follow_up events as usual
        case "response":
            # Final assembled answer — end of turn
            print()
        case "error":
            print(f"\nError: {event.content}")
```

---

## Configuration file

Pass `--config path/to/file.yaml` to the TUI or use `OpenDataSciConfig.from_yaml()` in the SDK:

```yaml
# opendatasci_config.yaml
provider: anthropic
model: claude-sonnet-4-6

secondary_provider: openai
secondary_model: gpt-4o-mini

temperature: 0.1
thinking_budget: 8000

extra_web_domains:
  - arxiv.org
  - huggingface.co

worker_timeout_seconds: 600
midturn_compaction_threshold: 80000
```

```python
from opendatasci import OpenDataSciConfig

config = OpenDataSciConfig.from_yaml("opendatasci_config.yaml")
```

---

## Workspace structure

OpenDataSci reads from and writes to a **workspace** — a local directory containing your data files.

```
my-project/
├── data.csv
├── data2.parquet
└── .opendatasci/          # managed by OpenDataSci
    ├── mcp.json           # MCP tool server URLs (optional)
    └── plans/             # persisted agent plans (auto-managed)
```

### MCP tool servers

Add external MCP servers by creating `.opendatasci/mcp.json`:

```json
{
  "servers": [
    { "url": "http://localhost:3000/mcp" }
  ]
}
```

Or set `mcp_servers` in `OpenDataSciConfig`.

### Skills

Create `.opendatasci/skills/` in your workspace and add Markdown files describing domain-specific methodology. The agent loads these automatically and applies them as additional expertise.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI / vLLM API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource URL |
| `REGION` | Cloud region (Bedrock) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region |
| `LLM_SERVER_BASE_URL` | Custom endpoint (Ollama / vLLM) |
| `SKILLS_DIRECTORY` | Path to a user-defined skills directory |
| `BUILTIN_SKILLS_DIRECTORY` | Override the bundled built-in skills directory |
| `CODE_EXEC_TIMEOUT` | Max seconds for one sandbox execution (default: `1800`) |

A `.env` file in the current working directory is loaded automatically on startup.
