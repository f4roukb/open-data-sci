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
- [Custom Skills](#custom-skills)
- [Environment Variables](#environment-variables)

---

## Installation

```bash
pip install open-data-sci
```

**Requirements:**
- Python: 3.12
- Platform: macOS or Linux (Windows is not supported)

### System dependencies

The sandbox that runs model-generated code shells out to native binaries that `pip` cannot install. Install them with your OS package manager before using the agent:

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

If you've cloned the repository, `make install-system-dependencies` runs the right command for your platform automatically.

Additionally, install the [GitHub CLI](https://cli.github.com) (`gh`) to use the built-in **`github.com`** skill (`execute_cli_command` shells out to it for read-oriented GitHub lookups):

```bash
# macOS
brew install gh

# Linux (Debian/Ubuntu)
sudo apt install gh

# Linux (Fedora)
sudo dnf install gh

# Linux (Arch)
sudo pacman -S github-cli
```

`gh` runs unauthenticated inside the sandbox (the sandbox denies read access to `~/.config/gh`, so it can't inherit a host `gh auth login` session) — fine for public read-only lookups, subject to GitHub's unauthenticated rate limits.

### Provider extras

Install optional extras to unlock additional LLM providers:

```bash
pip install "open-data-sci[aws]"       # AWS Bedrock
pip install "open-data-sci[gemini]"    # Google Gemini (AI Studio)
pip install "open-data-sci[gcp]"       # Google Vertex AI
pip install "open-data-sci[azure]"     # Azure OpenAI
pip install "open-data-sci[ollama]"    # Ollama (local models)
```

### Capability extras

```bash
pip install "open-data-sci[deep-learning]" # Deep learning on the host — PyTorch, JAX, Transformers, Sentence-Transformers
pip install "open-data-sci[finance]" # Finance data — yfinance
```

The `[deep-learning]` extra — deep learning directly on the host, for machines with a GPU or NPU — is required to use the **Deep Learning** skill; without it, the agent's sandboxed Python environment has no training framework available. The `[finance]` extra is required to use the **`finance.yahoo.com`** skill.

> **GPU access inside the sandbox is opt-in, and it's a real host-kernel exposure.** When a `[deep-learning]` package (`torch`, `jax`, `transformers`, `sentence-transformers`) is installed, the sandbox bind-mounts the host's GPU compute device nodes (`/dev/nvidia*`, `/dev/dri/renderD*` on Linux) so those frameworks can actually use the GPU — otherwise sandboxed code has no path to accelerator hardware at all. This is a materially different risk than the sandbox's filesystem/network isolation: it hands sandboxed code direct `ioctl` access to the host kernel's GPU driver (GPU driver ioctl surfaces have a real CVE history), and there's no GPU-equivalent of the CPU/memory resource limits the sandbox otherwise enforces. A warning is logged whenever this activates. See the module docstring in `opendatasci/sandbox/srt.py` for the full detail. macOS/Metal passthrough and NPU passthrough are not verified — see that docstring for current status. Uninstall the `[deep-learning]` packages to disable this entirely.

Multiple extras can be combined:

```bash
pip install "open-data-sci[aws,gemini,deep-learning,finance]"
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
opendatasci data.csv --provider ollama --model qwen3.5:9b   # local, no key needed
```

### Setup with a config file

For a reusable configuration across projects, create a YAML file and pass it with `--config`. TUI flags always take precedence over values in the file.

```yaml
# datasci.yaml
provider: anthropic
model: claude-sonnet-5
secondary_provider: openai
secondary_model: gpt-5.6-luna
temperature: 0.1
```

```bash
opendatasci data.csv --config datasci.yaml
```

Annotated config files for every supported provider are available in [`examples/configs/`](examples/configs/).

### Python SDK

```python
from opendatasci import create_agent, Invocation

async with create_agent("data.csv") as agent:
    invocation = Invocation.from_text("Summarise this dataset and train a model on the target column.")
    async for event in agent.astream(invocation):
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
| `--provider` | `anthropic` | LLM provider for the primary model. Choices: `anthropic`, `openai`, `bedrock`, `gemini`, `vertexai`, `azure`, `ollama`, `openai_compatible_server` |
| `--model` | *(provider default)* | Primary model name — provider-specific identifier. Omit to use the provider's default (see [Models](#models)) |
| `--secondary-provider` | *(same as `--provider`)* | Provider for the secondary (auxiliary) model — may differ from `--provider` |
| `--secondary-model` | *(provider default)* | Secondary model name for lightweight tasks (summarisation, etc.) |
| `--api-key` | *(env var)* | API key for the primary provider. Falls back to the standard env var for the selected provider |
| `--theme` | `default` | Colour palette. Choices: `default`, `accessible`, `light`, `solarized`, `dracula`. Run `/themes` inside the TUI for descriptions |
| `--config` | *(none)* | Path to a YAML file containing `OpenDataSciConfig` fields; explicit TUI flags take precedence |
| `--list-providers` | | Print all supported providers and their default models, then exit |
| `--version` | | Print the installed version, then exit |

### Examples

```bash
# Minimal — analyse a single file with the default Anthropic provider
opendatasci data.xlsx

# Switch provider and primary model
opendatasci data.csv --provider openai --model gpt-5.6-sol

# Bedrock with a region
REGION=us-west-2 opendatasci ./project/ --provider bedrock

# Colour-blind safe theme
opendatasci data.parquet --theme accessible

# Mix providers — heavy model on one, lightweight secondary on another
opendatasci data.csv --provider anthropic --secondary-provider openai --secondary-model gpt-5.6-luna

# See all available providers
opendatasci --list-providers
```

---

## Slash Commands

Type `/` in the input box to trigger autocomplete. All commands are available at any time.

| Command | Description |
|---------|-------------|
| `/cancel-all-messages` | Cancel all messages queued while the agent was busy |
| `/cancel-message` | Cancel the most recently queued message |
| `/clear` | Clear conversation context (workspace files are untouched) |
| `/compact` | Summarise and compress conversation history to free up context |
| `/help` | Show all available commands |
| `/ls-workspace` | List all files currently in the workspace |
| `/models` | Show the primary and secondary model in use |
| `/reset` | Reset the agent session and reload data from disk |
| `/stop` | Stop the currently running agent turn (future messages resume from where it left off) |
| `/themes` | List available colour themes with descriptions |
| `/exit` | Quit OpenDataSci |

Sending a message while the agent is still working doesn't reject it — it's pinned above the input box as a queued message and run automatically, in order, once the agent finishes (unless the agent is waiting on your answer to a question). Use `/cancel-message` or `/cancel-all-messages` to discard queued messages instead of waiting for them to run.

When the agent schedules work in the background (e.g. concurrent worker agents running an ensemble sweep), a **Background** line in the header shows which tasks are still running and their latest self-reported progress. You don't need to check back manually — as soon as a background task finishes, the agent picks it up and continues on its own.

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
    model="gpt-5.6-sol",
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
| `provider` | LLM provider (`"anthropic"`, `"openai"`, `"bedrock"`, `"gemini"`, `"vertexai"`, `"azure"`, `"ollama"`, `"openai_compatible_server"`) |
| `model` | Primary model identifier — omit to use the provider default |
| `secondary_provider` | Provider for the lightweight secondary model — defaults to the primary provider |
| `secondary_model` | Secondary model identifier — omit to use the provider default |
| `anthropic_api_key` | Anthropic API key (env: `ANTHROPIC_API_KEY`) |
| `openai_api_key` | OpenAI / OpenAI-compatible server API key (env: `OPENAI_API_KEY`) |
| `google_api_key` | Google Gemini API key (env: `GOOGLE_API_KEY`) |
| `azure_api_key` | Azure OpenAI API key (env: `AZURE_OPENAI_API_KEY`) |
| `aws_region` | AWS region for Bedrock (env: `REGION`) |
| `google_cloud_project` | GCP project ID for Vertex AI (env: `GOOGLE_CLOUD_PROJECT`) |
| `google_cloud_location` | Vertex AI region (env: `GOOGLE_CLOUD_LOCATION`) |
| `azure_endpoint` | Azure OpenAI resource endpoint URL (env: `AZURE_OPENAI_ENDPOINT`) |
| `azure_api_version` | Azure OpenAI API version — defaults to `2025-01-01-preview` (env: `AZURE_OPENAI_API_VERSION`) |
| `llm_server_base_url` | Custom API base URL — required for `ollama` and `openai_compatible_server` (env: `LLM_SERVER_BASE_URL`) |
| `temperature` | Sampling temperature — not sent to Claude 4.6+ / Sonnet 5 models, which use adaptive thinking (env: `TEMPERATURE`) |
| `name` | Display name for the agent — defaults to `"Sai"` (env: `NAME`) |
| `mcp_servers` | List of MCP server URLs the agent may connect to (env: `MCP_SERVERS`) |
| `skills_directory` | Path to a directory of custom skill files loaded in addition to built-ins (env: `SKILLS_DIRECTORY`) |
| `builtin_skills_directory` | Path to the built-in skills directory — override only to replace defaults entirely (env: `BUILTIN_SKILLS_DIRECTORY`) |
| `worker_timeout_seconds` | Max seconds to wait for spawned workers to finish — `null` disables the timeout, default `300` (env: `WORKER_TIMEOUT_SECONDS`) |
| `midturn_compaction_threshold` | Token count at which context is compacted mid-turn — default `96000` (env: `MIDTURN_COMPACTION_THRESHOLD`) |
| `local_code_exec_timeout` | Max seconds for a single sandboxed code-execution run — default `1800` (env: `CODE_EXEC_TIMEOUT`) |

---

## Models

OpenDataSci supports every major LLM provider. Pass `--provider` to the TUI or set it in `OpenDataSciConfig`.

| Provider | Flag | Extra required | Default model |
|----------|------|----------------|---------------|
| Anthropic | `anthropic` | *(none — default)* | `claude-sonnet-5` |
| OpenAI | `openai` | *(none)* | `gpt-5.6-sol` |
| OpenAI-compatible server (e.g. vLLM) | `openai_compatible_server` | *(none)* | `Qwen/Qwen3.5-4B` |
| AWS Bedrock | `bedrock` | `open-data-sci[aws]` | `us.anthropic.claude-sonnet-5` |
| Google Gemini | `gemini` | `open-data-sci[gemini]` | `gemini-3.5-flash` |
| Google Vertex AI | `vertexai` | `open-data-sci[gcp]` | `gemini-3.5-flash` |
| Azure OpenAI | `azure` | `open-data-sci[azure]` | `gpt-5.6-sol` |
| Ollama | `ollama` | `open-data-sci[ollama]` | `qwen3.5:9b` |

Pass `--list-providers` to print this table from the TUI at any time.

---

## Configuration

### Workspace files

Place these files inside your workspace's `.opendatasci/` directory:

| Path | Purpose |
|------|---------|
| `.opendatasci/mcp.json` | MCP server definitions — connects the agent to external tool servers |
| `.opendatasci/plans/` | Persisted plan files — auto-managed, one file per planning session |

`mcp.json` uses the same convention as Cursor:

```json
{
  "mcpServers": {
    "my-server":   { "url": "http://localhost:8080" },
    "another":     { "url": "http://localhost:9000" }
  }
}
```

---

## Custom Skills

Skills are Markdown (or YAML/JSON) files that give the agent a specialised persona and instruction set. OpenDataSci ships several built-in skills; you can add your own at the workspace level or point the agent at any directory you choose.

### Workspace skills (recommended)

Place skill files inside `.opendatasci/skills/` in your workspace directory — they are picked up automatically, no configuration needed:

```
<workspace>/
└── .opendatasci/
    ├── skills/
    │   ├── my_skill.md              # standalone skill named "my_skill"
    │   └── my_domain/
    │       └── specialist.md        # domain-scoped skill: "my_domain::specialist"
    └── skill_domains/
        └── my_domain/
            └── manifest.md          # optional domain manifest
```

Subdirectories create *domain-scoped* skills. The agent refers to them with the `domain::skill` naming convention (e.g. `my_domain::specialist`).

### File format

Skill files must be `.md`. The filename stem becomes the skill name and the file body is the prompt content. Files with any other extension are silently ignored.

```markdown
<!-- .opendatasci/skills/forecasting.md -->
You are a time-series forecasting specialist. When analysing data, always...
```

### Global skills directory

To share skills across workspaces, set `SKILLS_DIRECTORY` in your environment or `.env` file:

```bash
SKILLS_DIRECTORY=/home/user/my-skills
```

This directory is scanned *in addition to* the workspace `.opendatasci/skills/` directory and the built-in skills. When two sources define a skill with the same name, the later source wins (built-ins → workspace → `SKILLS_DIRECTORY`).

You can also pass `skills_directory` directly when using the Python SDK:

```python
from opendatasci import OpenDataSciConfig, create_agent

config = OpenDataSciConfig(skills_directory="/home/user/my-skills")

async with create_agent("data.csv", config=config) as agent:
    ...
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | API key for the Anthropic provider |
| `OPENAI_API_KEY` | API key for the OpenAI / OpenAI-compatible server provider |
| `GOOGLE_API_KEY` | API key for the Google Gemini provider |
| `AZURE_OPENAI_API_KEY` | API key for the Azure OpenAI provider |
| `REGION` | AWS region for Bedrock |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID for Vertex AI |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region / location |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint URL |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version (default: `2025-01-01-preview`) |
| `LLM_SERVER_BASE_URL` | Custom API base URL — used by `ollama` and `openai_compatible_server` providers |
| `TEMPERATURE` | LLM sampling temperature |
| `MCP_SERVERS` | Comma-separated list of MCP server URLs |
| `SKILLS_DIRECTORY` | Path to a directory of user-defined skill files |
| `BUILTIN_SKILLS_DIRECTORY` | Path to the built-in skills directory (defaults to the bundled skills) |
| `WORKER_TIMEOUT_SECONDS` | Max seconds to wait for spawned workers (default: `300`) |
| `MIDTURN_COMPACTION_THRESHOLD` | Token count at which context is compacted mid-turn (default: `96000`) |
| `CODE_EXEC_TIMEOUT` | Max seconds for a single sandboxed code execution (default: `1800`) |

A `.env` file in the working directory is loaded automatically at startup.
