# OpenDataSci

A production-grade AI agent for data science and machine learning. See the [project README](../../README.md) for an overview, benchmark results, and feature descriptions.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [TUI Reference](#tui-reference)
- [Slash Commands](#slash-commands)
- [File Attachments](#file-attachments)
- [Key Bindings](#key-bindings)
- [Themes](#themes)
- [Python SDK](#python-sdk)
- [Models](#models)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)

---

## Installation

```bash
pip install open-data-sci
```

**Requirements:** Python 3.12

### Provider extras

Install optional extras to unlock additional LLM providers:

```bash
pip install "open-data-sci[aws]"       # AWS Bedrock
pip install "open-data-sci[gemini]"    # Google Gemini (AI Studio)
pip install "open-data-sci[gcp]"       # Google Vertex AI
pip install "open-data-sci[azure]"     # Azure OpenAI
pip install "open-data-sci[ollama]"    # Ollama (local models)
pip install "open-data-sci[vllm]"      # vLLM (self-hosted)
```

### Capability extras

```bash
pip install "open-data-sci[jax]"       # Deep learning — JAX, Flax, Optax
```

The `[jax]` extra is required to use the **Deep Learning** skill. Without it, the agent's sandboxed Python environment has no training framework available.

Multiple extras can be combined:

```bash
pip install "open-data-sci[aws,gemini,jax]"
```

---

## Quick Start

### Basic setup

Set your API key and point OpenDataSci at your data:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
opendatasci data.csv
```

A `.env` file in the working directory is loaded automatically, so you can also place it there:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

To use a different provider, pass `--provider`:

```bash
opendatasci data.csv --provider openai --api-key sk-...
opendatasci data.csv --provider ollama --model llama3.2:3b   # local, no key needed
```

### Setup with a config file

For a reusable configuration across projects, create a YAML file and pass it with `--config`. TUI flags always take precedence over values in the file.

```yaml
# datasci.yaml
provider: anthropic
model: claude-sonnet-4-6
secondary_provider: openai
secondary_model: gpt-4o-mini
temperature: 0.1
thinking_budget: 8000
```

```bash
opendatasci data.csv --config datasci.yaml
```

Annotated config files for every supported provider are available in [`examples/configs/`](examples/configs/).

### Python SDK

```python
from opendatasci import create_agent

async with create_agent("data.csv") as agent:
    async for event in agent.astream("Summarise this dataset and train a model on the target column."):
        print(event)
```

### More examples

The [`examples/`](examples/README.md) directory covers TUI walkthroughs, batch scripts, Jupyter notebooks, and annotated config files across every supported provider.

---

## TUI Reference

```
opendatasci PATH [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | Data file or directory to load into the workspace |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `anthropic` | LLM provider for the primary model. Choices: `anthropic`, `openai`, `bedrock`, `gemini`, `vertexai`, `azure`, `ollama`, `vllm` |
| `--model` | *(provider default)* | Primary model name — provider-specific identifier. Omit to use the provider's default (see [Models](#models)) |
| `--secondary-provider` | *(same as `--provider`)* | Provider for the secondary (auxiliary) model — may differ from `--provider` |
| `--secondary-model` | *(provider default)* | Secondary model name for lightweight tasks (summarisation, etc.) |
| `--api-key` | *(env var)* | API key for the primary provider. Falls back to the standard env var for the selected provider |
| `--theme` | `default` | Colour palette. Choices: `default`, `accessible`, `light`, `solarized`, `dracula`. Run `/themes` inside the TUI for descriptions |
| `--debug` | `false` | Enable debug output — writes a detailed `opendatasci_debug.log` |
| `--config` | *(none)* | Path to a YAML file containing `OpenDataSciConfig` fields; explicit TUI flags take precedence |
| `--list-providers` | | Print all supported providers and their default models, then exit |
| `--version` | | Print the installed version, then exit |

### Examples

```bash
# Minimal — analyse a single file with the default Anthropic provider
opendatasci data.xlsx

# Switch provider and primary model
opendatasci data.csv --provider openai --model gpt-4o

# Bedrock with a region
REGION=us-west-2 opendatasci ./project/ --provider bedrock

# Colour-blind safe theme
opendatasci data.parquet --theme accessible

# Mix providers — heavy model on one, lightweight secondary on another
opendatasci data.csv --provider anthropic --secondary-provider openai --secondary-model gpt-5.4-mini

# See all available providers
opendatasci --list-providers
```

---

## Slash Commands

Type `/` in the input box to trigger autocomplete. All commands are available at any time.

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation context (preserves session variables and loaded data) |
| `/compact` | Summarise and compress conversation history to free up context |
| `/help` | Show all available commands |
| `/ls-workspace` | List all files currently in the workspace |
| `/models` | Show the primary and secondary model in use |
| `/reset` | Reset the agent session and reload data from disk |
| `/stop` | Stop the currently running agent turn (future messages resume from where it left off) |
| `/themes` | List available colour themes with descriptions |
| `/exit` | Quit OpenDataSci |

---

## File Attachments

Attach files or code snippets to any message using the `@` prefix:

```
@path/to/file.py                      # attach an entire file
@path/to/notebook.ipynb:L10-L40      # attach a specific line range
```

The agent sees the attached content as structured context inline with your message. Paths are resolved relative to your current working directory.

---

## Key Bindings

| Key | Action |
|-----|--------|
| `Ctrl+C` (×2) | Quit |
| `Ctrl+D` | Quit |
| `Ctrl+R` | Reset session |
| `Ctrl+L` | Clear conversation |
| `Escape` | Focus input box |
| `Tab` / `Shift+Tab` | Cycle through autocomplete suggestions |
| `↑` / `↓` | Navigate input history or autocomplete |

---

## Themes

Select a theme at launch with `--theme`. Run `/themes` inside the TUI to see descriptions.

| Name | Description |
|------|-------------|
| `default` | Dark background with muted accents (built-in default) |
| `accessible` | Okabe-Ito palette — colour-blind safe |
| `light` | Light background with dark text |
| `solarized` | Solarized Dark by Ethan Schoonover |
| `dracula` | Dracula — vivid pastels on near-black |

---

## Python SDK

The async-first Python API gives full programmatic control over the agent.

### Basic usage

```python
from opendatasci import create_agent

async with create_agent("sales.xlsx") as agent:
    async for event in agent.astream("What is the average revenue by region?"):
        print(event)
```

### Custom provider and model

```python
from opendatasci import OpenDataSciConfig, create_agent

config = OpenDataSciConfig(
    provider="openai",
    model="gpt-4o",
    openai_api_key="sk-...",
    temperature=0.2,
)

async with create_agent("data.parquet", config=config) as agent:
    async for event in agent.astream("Train a gradient boosting model on the target column."):
        print(event)
```

### `OpenDataSciConfig` reference

| Field | Description |
|-------|-------------|
| `provider` | LLM provider (`"anthropic"`, `"openai"`, `"bedrock"`, `"gemini"`, `"vertexai"`, `"azure"`, `"ollama"`, `"vllm"`) |
| `model` | Primary model identifier — omit to use the provider default |
| `secondary_provider` | Provider for the lightweight secondary model — defaults to the primary provider |
| `secondary_model` | Secondary model identifier — omit to use the provider default |
| `anthropic_api_key` | Anthropic API key (env: `ANTHROPIC_API_KEY`) |
| `openai_api_key` | OpenAI / vLLM API key (env: `OPENAI_API_KEY`) |
| `google_api_key` | Google Gemini API key (env: `GOOGLE_API_KEY`) |
| `azure_api_key` | Azure OpenAI API key (env: `AZURE_OPENAI_API_KEY`) |
| `aws_region` | AWS region for Bedrock (env: `REGION`) |
| `google_cloud_project` | GCP project ID for Vertex AI (env: `GOOGLE_CLOUD_PROJECT`) |
| `google_cloud_location` | Vertex AI region (env: `GOOGLE_CLOUD_LOCATION`) |
| `azure_endpoint` | Azure OpenAI resource endpoint URL (env: `AZURE_OPENAI_ENDPOINT`) |
| `llm_server_base_url` | Custom API base URL — required for `ollama` and `vllm` (env: `LLM_SERVER_BASE_URL`) |

---

## Models

OpenDataSci supports every major LLM provider. Pass `--provider` to the TUI or set it in `OpenDataSciConfig`.

| Provider | Flag | Extra required | Default model |
|----------|------|----------------|---------------|
| Anthropic | `anthropic` | *(none — default)* | `claude-sonnet-4-6` |
| OpenAI | `openai` | *(none)* | `gpt-5.5` |
| AWS Bedrock | `bedrock` | `open-data-sci[aws]` | `us.anthropic.claude-sonnet-4-6` |
| Google Gemini | `gemini` | `open-data-sci[gemini]` | `gemini-2.5-pro` |
| Google Vertex AI | `vertexai` | `open-data-sci[gcp]` | `gemini-2.5-pro` |
| Azure OpenAI | `azure` | `open-data-sci[azure]` | `gpt-4o` |
| Ollama | `ollama` | `open-data-sci[ollama]` | `llama3.2:3b` |
| vLLM | `vllm` | `open-data-sci[vllm]` | `meta-llama/Llama-3.2-3B-Instruct` |

Pass `--list-providers` to print this table from the TUI at any time.

---

## Configuration

### Workspace files

Place these files inside your workspace's `.opendatasci/` directory:

| Path | Purpose |
|------|---------|
| `.opendatasci/mcp.json` | MCP server URLs — connects the agent to external tool servers |
| `.opendatasci/plans/` | Persisted plan files — auto-managed, one file per planning session |

### Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for the Anthropic provider |
| `OPENAI_API_KEY` | API key for the OpenAI provider |
| `REGION` | Cloud provider region |
| `LLM_SERVER_BASE_URL` | Custom API base URL — used by `ollama` and `vllm` providers |
| `SKILLS_DIRECTORY` | Path to a directory of user-defined skill files (overrides none by default) |
| `BUILTIN_SKILLS_DIRECTORY` | Path to the built-in skills directory (defaults to the bundled `resources/skills`) |
| `CODE_EXEC_TIMEOUT` | Max seconds for a single sandboxed code execution (default: `1800`) |

A `.env` file in the working directory is loaded automatically at startup.
