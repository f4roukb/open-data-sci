# Getting Started

## Installation

```bash
pip install open-data-sci
```

**Requirements:**

- Python 3.12
- macOS or Linux (Windows is not supported)

### System dependencies

The sandbox that runs agent-generated code shells out to native binaries that `pip` cannot install. Install them with your OS package manager before using the agent:

```bash
# macOS
brew install ripgrep

# Linux (Debian/Ubuntu)
sudo apt-get install -y bubblewrap socat ripgrep

# Linux (Fedora)
sudo dnf install -y bubblewrap socat ripgrep

# Linux (Arch)
sudo pacman -S --noconfirm bubblewrap socat ripgrep
```

### Provider extras

The default installation includes the Anthropic and OpenAI clients. Install additional extras to unlock other providers:

| Extra | Provider |
|-------|----------|
| `open-data-sci[aws]` | AWS Bedrock |
| `open-data-sci[gemini]` | Google Gemini (AI Studio) |
| `open-data-sci[gcp]` | Google Vertex AI |
| `open-data-sci[azure]` | Azure OpenAI |
| `open-data-sci[ollama]` | Ollama (local models) |

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
| Anthropic *(default)* | `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-5.6-sol` | `OPENAI_API_KEY` |
| AWS Bedrock | `bedrock` | `us.anthropic.claude-sonnet-5` | boto3 credential chain |
| Google Gemini | `gemini` | `gemini-3.5-flash` | `GOOGLE_API_KEY` |
| Google Vertex AI | `vertexai` | `gemini-3.5-flash` | Application Default Credentials |
| Azure OpenAI | `azure` | `gpt-5.6-sol` | `AZURE_OPENAI_API_KEY` or service principal |
| Ollama | `ollama` | `qwen3.5:9b` | none (local server) |
| OpenAI-compatible server (e.g. vLLM) | `openai_compatible_server` | `Qwen/Qwen3.5-4B` | none (self-hosted) |

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
    opendatasci data.csv --provider azure --model gpt-5.6-sol
    ```

=== "Ollama"

    ```bash
    # Start Ollama first: ollama serve
    opendatasci data.csv --provider ollama --model qwen3.5:9b
    ```

=== "OpenAI-compatible server"

    ```bash
    # Start any OpenAI-compatible server first, e.g. vLLM:
    # vllm serve Qwen/Qwen3.5-4B
    opendatasci data.csv --provider openai_compatible_server --model Qwen/Qwen3.5-4B
    ```

---

## TUI quick start

```bash
# Analyse a single file with the default Anthropic provider
opendatasci data.csv

# Load an entire directory of data files
opendatasci ./my-project/

# Change model
opendatasci data.csv --provider openai --model gpt-5.6-sol

# Mix providers — heavy primary model, lightweight secondary
opendatasci data.csv --provider anthropic --secondary-provider openai --secondary-model gpt-5.6-luna

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

### Slash commands

Inside a running session, type `/` for autocomplete or `@` to attach a file. Available commands:

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/models` | Show the primary and secondary model in use |
| `/themes` | List available colour themes (selected at launch with `--theme`) |
| `/ls-workspace` | List files in the workspace |
| `/compact` | Summarize and compress the conversation history |
| `/clear` | Clear all conversation context, including the plan (preserves session variables) |
| `/reset` | Reset the agent session and reload data from disk |
| `/stop` | Stop the running agent (future messages pick up where it left off) |
| `/cancel-message` | Cancel the most recently queued message |
| `/cancel-all-messages` | Cancel all messages queued while the agent was busy |
| `/exit` | Exit OpenDataSci |

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
    model="gpt-5.6-sol",
    openai_api_key="sk-...",
    temperature=0.2,
)

async with create_agent("data.parquet", config=config) as agent:
    async for event in agent.astream("Train a gradient-boosting model on the target column."):
        ...
```

### Consuming stream events

`agent.astream()` yields typed dataclass events — `TokenEvent`, `ToolCallEvent`,
`WorkerDoneEvent`, `InputRequiredEvent`, `ApprovalRequiredEvent`, `ResponseEvent`,
and more. Each has a `type` class variable (so `event.type == "token"` still
works) plus its own strongly-typed fields — no generic `metadata` dict to dig
through. See [Events & Types](api/types.md) for the complete reference.

```python
from opendatasci.streaming import (
    InputRequiredEvent, ResponseEvent, TokenEvent, ToolCallEvent, WorkerDoneEvent,
)

async for event in agent.astream(query):
    if isinstance(event, TokenEvent):
        # Incremental response text
        print(event.content, end="", flush=True)

    elif isinstance(event, ToolCallEvent):
        print(f"\n[tool] {event.tool}")

    elif isinstance(event, WorkerDoneEvent):
        print(f"\n[worker {event.worker_idx}] {'ok' if event.success else 'failed'}")

    elif isinstance(event, InputRequiredEvent):
        # Agent needs a choice from the user — resume by calling astream() again
        choice = input(event.content + " ")
        async for follow_up in agent.astream(choice):
            pass  # handle follow_up events as usual

    elif isinstance(event, ResponseEvent):
        # Final assembled answer — end of turn
        print()
```

---

## Configuration file

Pass `--config path/to/file.yaml` to the TUI or use `OpenDataSciConfig.from_yaml()` in the SDK:

```yaml
# opendatasci_config.yaml
provider: anthropic
model: claude-sonnet-5

secondary_provider: openai
secondary_model: gpt-5.6-luna

temperature: 0.1

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
└── .opendatasci/           # managed by OpenDataSci
    ├── mcp.json            # MCP tool server URLs (optional)
    ├── skills/             # custom skill files (optional, see below)
    ├── plans/              # persisted agent plans (auto-managed)
    ├── dataset_notes/      # per-dataset notes carried across sessions
    ├── dataset_profiling/  # per-dataset profile cards carried across sessions
    ├── artifacts/          # tables, plots, and models the agent writes during a run
    └── session.json        # session state (auto-managed)
```

`dataset_notes/` and `dataset_profiling/` are what make project memory persistent:
the agent writes what it learns about a dataset here, keyed by dataset path, and
reads it back at the start of future sessions in the same workspace.

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

Read directly by `OpenDataSciConfig` (see [OpenDataSciConfig](api/config.md) for the full field list):

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI / OpenAI-compatible server API key |
| `GOOGLE_API_KEY` | Google Gemini (AI Studio) API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version (default: `2025-01-01-preview`) |
| `REGION` | AWS region for Bedrock (default: `us-east-1`) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region |
| `LLM_SERVER_BASE_URL` | Custom endpoint (Ollama / OpenAI-compatible server) |
| `NAME` | Display name of the agent (default: `Sai`) |
| `MCP_SERVERS` | JSON array of MCP server URLs, alternative to `.opendatasci/mcp.json` (e.g. `["http://localhost:3000/mcp"]`) |
| `EXTRA_FETCH_DOMAINS` | JSON array of additional hostnames the agent's `fetch_url` tool may retrieve |
| `SKILLS_DIRECTORY` | Path to a user-defined skills directory |
| `BUILTIN_SKILLS_DIRECTORY` | Override the bundled built-in skills directory |
| `SKILL_DOMAINS_DIRECTORY` | Path to a user-defined skill domains directory |
| `BUILTIN_SKILL_DOMAINS_DIRECTORY` | Override the bundled built-in skill domains directory |
| `WORKER_TIMEOUT_SECONDS` | Max seconds to wait for spawned workers to finish (default: `300`) |
| `MIDTURN_COMPACTION_THRESHOLD` | Token count that triggers mid-turn context compaction (default: `96000`) |
| `CODE_EXEC_TIMEOUT` | Max seconds for one sandbox execution (default: `1800`) |

Read by the underlying cloud SDKs, not by OpenDataSci itself — set whichever match your provider and auth method:

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Long-lived IAM key for Bedrock |
| `AWS_SESSION_TOKEN` | Add alongside the above for temporary STS credentials |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to a service-account JSON key for Vertex AI |
| `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` / `AZURE_CLIENT_SECRET` | Service-principal auth for Azure OpenAI (requires `pip install 'open-data-sci[azure]'`) |

A `.env` file in the current working directory is loaded automatically on startup.
