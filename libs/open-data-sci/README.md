# OpenDataSci

[![PyPI version](https://img.shields.io/pypi/v/open-data-sci.svg)](https://pypi.org/project/open-data-sci/)
[![Python versions](https://img.shields.io/pypi/pyversions/open-data-sci.svg)](https://pypi.org/project/open-data-sci/)
[![License](https://img.shields.io/pypi/l/open-data-sci.svg)](https://pypi.org/project/open-data-sci/)

A production-grade AI agent for data science and machine learning — run it as an interactive terminal app, or embed it as an async Python SDK in your own service.

- **Every major LLM provider** — Anthropic, OpenAI, AWS Bedrock, Google Gemini/Vertex AI, Azure OpenAI, Ollama, or any OpenAI-compatible server ([details](#models))
- **Sandboxed code execution** — the agent writes and runs real Python and CLI code against your data, isolated from the host
- **Zero-config first run** — point it at a file or directory and an interactive wizard handles the rest; every setting is also scriptable through YAML, environment variables, or the Python SDK
- **Extensible** — connect [MCP servers](#mcp-servers) for extra tools, or add [custom skills](#custom-skills) to specialise the agent for a domain
- **Cloud-portable architecture** — every stateful dependency (workspace, sandbox, memory, sessions, background tasks) sits behind an abstract interface, so a multi-tenant deployment is a matter of swapping implementations rather than rewriting the agent ([details](#cloud-portability))

Full documentation, including the complete Python API reference: https://opendatasci.readthedocs.io/en/latest/

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [First Launch: The Setup Wizard](#first-launch-the-setup-wizard)
- [TUI Reference](#tui-reference)
- [Slash Commands](#slash-commands)
- [The `/config` Panel](#the-config-panel)
- [File Attachments](#file-attachments)
- [Python SDK](#python-sdk)
- [Cloud Portability](#cloud-portability)
- [Models](#models)
- [MCP Servers](#mcp-servers)
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

**You don't need to configure anything before running the TUI.** `opendatasci` alone launches an interactive setup wizard the first time it runs — see [First Launch](#first-launch-the-setup-wizard). The steps below matter if you want everything ready ahead of time (scripted installs, containers, CI), or if you're embedding OpenDataSci as a library rather than running the TUI.

### System dependencies

The sandbox that runs model-generated code shells out to native binaries that `pip` cannot install: `ripgrep` everywhere, plus `bubblewrap` and `socat` on Linux. **The TUI detects a missing dependency on first launch and offers to install it for you** — you only need to install them by hand if you're setting things up ahead of time (scripted installs, containers, CI) or skip that step in the wizard. See the documentation for the exact command per platform, including the optional GitHub CLI (`gh`) install needed for the built-in **`github.com`** skill.

### Provider extras

Install optional extras to unlock additional LLM providers:

```bash
pip install "open-data-sci[aws]"       # AWS Bedrock
pip install "open-data-sci[gemini]"    # Google Gemini (AI Studio)
pip install "open-data-sci[gcp]"       # Google Vertex AI
pip install "open-data-sci[azure]"     # Azure OpenAI
pip install "open-data-sci[ollama]"    # Ollama (local models)
```

Anthropic, OpenAI, and any OpenAI-compatible server (e.g. vLLM) work with no extra.

### Capability extras

```bash
pip install "open-data-sci[deep-learning]" # Deep learning on the host — PyTorch, JAX, Transformers, Sentence-Transformers
pip install "open-data-sci[finance]"       # Finance data — yfinance
```

The `[deep-learning]` extra — deep learning directly on the host, for machines with a GPU or NPU — is required to use the **Deep Learning** skill; without it, the agent's sandboxed Python environment has no training framework available. The `[finance]` extra is required to use the **`finance.yahoo.com`** skill.

> **GPU access inside the sandbox is opt-in, and it's a real host-kernel exposure.** Installing a `[deep-learning]` package makes the sandbox bind-mount the host's accelerator device nodes so those frameworks can actually use the hardware — a materially different risk than the sandbox's filesystem/network isolation, since it hands sandboxed code direct access to the host kernel's GPU driver. A warning is logged whenever this activates; uninstall the `[deep-learning]` packages to disable it entirely. See the documentation for the full breakdown (device nodes, per-platform coverage, and the underlying risk).

Multiple extras can be combined:

```bash
pip install "open-data-sci[aws,gemini,deep-learning,finance]"
```

---

## Quick Start

Point OpenDataSci at your data — nothing else required:

```bash
opendatasci data.csv
```

On a first run, a wizard walks you through picking a provider/model and entering whatever secret it needs (e.g. an API key) — see [First Launch](#first-launch-the-setup-wizard). Everything you enter is remembered for next time, so this only happens once per machine, not once per project.

If you already have an API key set (as an environment variable or in a `.env` file in the working directory), the wizard skips straight past that field:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

```bash
opendatasci data.csv
```

Everything — provider, model, secondary model, theme — can also be changed after launch without restarting, from the `/config` panel (alias `/settings`; see [Slash Commands](#slash-commands)).

### Setup with a config file

For a reusable configuration across projects, create a YAML file and pass it with `--config`. Whatever the file sets is used as-is; anything it leaves out (including provider/model selection, which then falls back to the wizard) is still picked up interactively.

```yaml
# datasci.yaml
provider: anthropic
model: claude-sonnet-5
secondary_provider: openai
secondary_model: gpt-5.6-luna
primary_temperature: 0.1
```

```bash
opendatasci data.csv --config datasci.yaml
```

An annotated config file ships for every supported provider.

### Quick start with the Python SDK

```python
from opendatasci import create_agent, Invocation

async with create_agent("data.csv") as agent:
    invocation = Invocation.from_text("Summarise this dataset and train a model on the target column.")
    async for event in agent.astream(invocation):
        print(event)
```

There's no wizard here — the SDK is not the TUI, so provide `config=OpenDataSciConfig(...)` (or set env vars) up front. See [Python SDK](#python-sdk) below for more, including custom providers.

---

## First Launch: The Setup Wizard

The very first time you run `opendatasci` (or any time it detects nothing was resolved from `--config`/env), it runs you through up to three steps before handing control to the chat. Every step is skipped automatically once it's already satisfied, so a second launch on the same machine is typically instant.

### 1. System dependencies check

If a required sandbox binary (`ripgrep`, and on Linux `bubblewrap`/`socat`) isn't installed, a one-time screen explains what's missing and offers to install it for you with your OS package manager (you may be prompted for your password). Decline, and it shows the exact command to run yourself, then lets you continue anyway — the sandbox is only needed once the agent actually executes code, so this step never blocks you from reaching the chat.

**Setting this up manually ahead of time (see [System dependencies](#system-dependencies)) makes the wizard skip this step entirely** — useful for scripted installs, Docker images, or CI, where there's no one at the keyboard to answer the prompt.

### 2. Provider & model selection

A short, linear flow (theme, then whichever of primary/secondary provider and model aren't already set) — one choice per screen, arrow keys to pick, no back button. Set any of these non-interactively with `--config` (see [Setup with a config file](#setup-with-a-config-file)); the wizard only asks about whatever the file leaves unresolved.

### 3. Provider secrets

Whatever the chosen provider still needs — an API key, an Azure endpoint, a GCP project ID — is collected one field at a time. Each value is saved as you enter it (to `~/.opendatasci/secrets/api.yaml`), so quitting partway through doesn't lose what you've already typed, and it won't be asked again on a later launch. Environment variables and `.env` always take precedence over this saved file, so a value you export or add to `.env` later overrides whatever the wizard remembered.

Everything the wizard sets can be changed afterwards, live, from the `/config` panel — see [The `/config` panel](#the-config-panel).

---

## TUI Reference

```
opendatasci PATH [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATH` | Data file or directory to load into the workspace. Defaults to the current directory when omitted |

### Options

| Flag | Description |
|------|-------------|
| `--config FILE` | Path to a YAML file containing `OpenDataSciConfig` fields. Fields it sets are used as-is; anything it doesn't set (including theme, which it never sets) is picked interactively on startup |
| `--version` | Print the installed version, then exit |

Provider, model, secondary provider/model, theme, and API keys are set through `--config`, environment variables/`.env`, or the interactive wizard/`/config` panel. If you're scripting a launch and want it to never prompt, use `--config` (and make sure any secrets it needs are in the environment).

### Examples

```bash
# Minimal — analyse a single file, wizard fills in whatever's missing
opendatasci data.xlsx

# Fully non-interactive, everything resolved from the file + env
opendatasci data.csv --config datasci.yaml

# Bedrock, credentials from the environment, model/provider from the file
REGION=us-west-2 opendatasci ./project/ --config examples/configs/config_bedrock.yaml
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
| `/config` (alias `/settings`) | Open the [configuration panel](#the-config-panel) — theme, models, providers, MCP servers, and more |
| `/help` | Show all available commands |
| `/ls-workspace` | List all files currently in the workspace |
| `/models` | Jump straight into `/config`'s Models section (primary/secondary provider and model, primary temperature) |
| `/reset` | Reset the agent session and reload data from disk |
| `/exit` | Quit OpenDataSci |

Switching provider or model from `/config` rebuilds the agent in the background; if the new provider/model fails to start (e.g. a missing API key), the error is reported and your current session keeps running untouched.

Sending a message while the agent is still working doesn't reject it — it's pinned above the input box as a queued message and run automatically, in order, once the agent finishes (unless the agent is waiting on your answer to a question). Use `/cancel-message` or `/cancel-all-messages` to discard queued messages instead of waiting for them to run.

When the agent schedules work in the background (e.g. concurrent worker agents running an ensemble sweep), a **Background** line in the header shows which tasks are still running. You don't need to check back manually — as soon as a background task finishes, the agent picks it up and continues on its own.

---

## The `/config` Panel

Run `/config` (or `/settings` — same command, either name works) to open a navigable menu covering every setting the agent needs, organised into sections:

| Section | What's in it |
|---------|--------------|
| **Display** | Theme; Tips (toggle the rotating footer hints) |
| **Integrations** | **MCP Servers** — add, verify, or remove [MCP servers](#mcp-servers) the agent can call, either by loading candidates from an `mcp.json` file or entering one manually (name, URL, transport, headers); **Custom skills** — point at a folder of [custom skills](#custom-skills) |
| **Models** | Grouped under **Primary Model** (provider, model, sampling temperature) and **Secondary Model** (provider, model) — picking a new provider resets its paired model to that provider's default, and the model choices offered depend on whichever provider is currently selected |
| **Personalization** | Agent display name |
| **Subagents** | Worker timeout (max seconds a spawned worker may run) |

Navigate with arrow keys and Enter, back out a level with Escape. Changing a model or provider applies immediately in the background — a failed switch (bad key, unreachable server) leaves your current session running untouched and reports the error instead.

---

## File Attachments

Attach files or code snippets to any message using the `@` prefix:

```
@path/to/file.py                      # attach an entire file
```

While typing, matching files are discovered relative to your current working directory. The message sent to the agent carries a reference to the resolved absolute path rather than the file's content — the agent reads the file itself using its own file-reading tool.

---

## Python SDK

The async-first Python API gives full programmatic control over the agent, independent of the TUI — this is what you reach for to embed OpenDataSci in a script, a service, a desktop app, or a notebook.

### Basic usage

```python
from opendatasci import Invocation, create_agent

async with create_agent("sales.xlsx") as agent:
    async for event in agent.astream(Invocation.from_text("What is the average revenue by region?")):
        print(event)
```

### Custom provider and model

```python
from opendatasci import Invocation, OpenDataSciConfig, create_agent

config = OpenDataSciConfig(
    provider="openai",
    model="gpt-5.6-sol",
    openai_api_key="sk-...",
    primary_temperature=0.2,
)

async with create_agent("data.parquet", config=config) as agent:
    async for event in agent.astream(Invocation.from_text("Train a gradient boosting model on the target column.")):
        print(event)
```

There's no wizard here — `OpenDataSciConfig` is a plain `pydantic-settings` model that never prompts for anything, so code built on the SDK directly is responsible for supplying whatever the chosen provider needs, same as any other library. See the documentation for the complete field-by-field reference (every field, its environment variable alias, and its default), plus patterns for headless batch processing, long-lived sessions, and multi-tenant deployment.

---

## Cloud Portability

Every stateful dependency OpenDataSci relies on — where it stores data, where it runs code, where it keeps memory — sits behind an abstract interface (workspace, sandbox, project memory, session mapping, conversation checkpoints, background tasks, skill registry, human approval), and the local backend shipped today is just one implementation of each. None of this is enabled out of the box — the shipped implementations are all local. Swap in a cloud-infrastructure-backed implementation of the same interface (e.g. an S3-compatible object store for the workspace, Firecracker microVMs for sandboxing, Valkey for session state) and the agent keeps working unchanged, which is what makes moving OpenDataSci into a multi-tenant or distributed deployment a matter of configuration and infrastructure choice, not a rewrite. See the documentation for the full list of interfaces and their recommended cloud implementations.

---

## Models

OpenDataSci supports every major LLM provider. Pass `provider`/`model` in your `--config` YAML or `OpenDataSciConfig`, or pick them from the setup wizard / `/config` → Models.

| Provider | Value | Extra required | Default model |
|----------|-------|-----------------|---------------|
| Anthropic | `anthropic` | *(none — default)* | `claude-sonnet-5` |
| OpenAI | `openai` | *(none)* | `gpt-5.6-sol` |
| OpenAI-compatible server (e.g. vLLM) | `openai_compatible_server` | *(none)* | `Qwen/Qwen3.5-4B` |
| AWS Bedrock | `bedrock` | `open-data-sci[aws]` | `us.anthropic.claude-sonnet-5` |
| Google Gemini | `gemini` | `open-data-sci[gemini]` | `gemini-3.5-flash` |
| Google Vertex AI | `vertexai` | `open-data-sci[gcp]` | `gemini-3.5-flash` |
| Azure OpenAI | `azure` | `open-data-sci[azure]` | `gpt-5.6-sol` |
| Ollama | `ollama` | `open-data-sci[ollama]` | `qwen3.5:9b` |

---

## MCP Servers

Connect the agent to external [Model Context Protocol](https://modelcontextprotocol.io) servers to give it additional tools. Only the two remote transports are supported — `http` (Streamable HTTP) and `sse` (Server-Sent Events) — a server reachable only via stdio (`command`/`args`) is out of scope, since OpenDataSci never launches a child process to talk to one.

### From the TUI

The easiest path: `/config` → Integrations → MCP Servers. Load candidate servers from an existing `mcp.json` file (pick which to add — these are added as-is, with no connectivity check), or add one manually (name, URL, transport, headers) — a manually-added server is tested and must connect successfully before it's kept.

Tools are (re)discovered from every configured server at the start of each turn, not just once at startup, so enabling/disabling tools on the server side takes effect without restarting OpenDataSci.

### Via the SDK

```python
from opendatasci import OpenDataSciConfig, create_agent
from opendatasci.tools.mcp import MCPServerSpec, MCPTransport

config = OpenDataSciConfig(
    mcp_servers=[
        MCPServerSpec(name="my-server", url="http://localhost:8080", transport=MCPTransport.HTTP),
    ]
)

async with create_agent("data.csv", config=config) as agent:
    ...
```

---

## Custom Skills

Skills are Markdown files that give the agent a specialised persona and instruction set. OpenDataSci ships several built-in skills; you can add your own at the workspace level or point the agent at any directory you choose.

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

To share skills across workspaces, set `SKILLS_DIRECTORY` in your environment or `.env` file — or point `/config` → Integrations → Custom skills at it live from inside the TUI:

```bash
SKILLS_DIRECTORY=/home/user/my-skills
```

This directory is scanned *in addition to* the workspace `.opendatasci/skills/` directory and the built-in skills. When two sources define a skill with the same name, the later source wins (built-ins → workspace → `SKILLS_DIRECTORY`). Skill *domains* have the equivalent `SKILL_DOMAINS_DIRECTORY`.

You can also pass `skills_directory` directly when using the Python SDK:

```python
from opendatasci import OpenDataSciConfig, create_agent

config = OpenDataSciConfig(skills_directory="/home/user/my-skills")

async with create_agent("data.csv", config=config) as agent:
    ...
```

---

## Environment Variables

Each variable below is also settable as a field on `OpenDataSciConfig` directly in the Python SDK — use whichever fits your setup.

| Variable | Description |
|----------|-------------|
| `PROVIDER` | LLM provider for the primary model (default: `anthropic`) |
| `MODEL` | Primary model identifier (default: provider default) |
| `SECONDARY_PROVIDER` | Provider for the secondary model (default: `anthropic`, independent of `PROVIDER`) |
| `SECONDARY_MODEL` | Secondary model for lightweight tasks (default: provider default) |
| `NAME` | Agent display name, injected into all system prompts (default: `Sai`) |
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
| `PRIMARY_TEMPERATURE` | LLM sampling temperature for the primary model |
| `MCP_SERVERS` | MCP server definitions — see [MCP Servers](#mcp-servers) |
| `SKILLS_DIRECTORY` | Path to a directory of user-defined skill files |
| `BUILTIN_SKILLS_DIRECTORY` | Path to the built-in skills directory (defaults to the bundled skills) |
| `SKILL_DOMAINS_DIRECTORY` | Path to a directory of user-defined skill domains |
| `BUILTIN_SKILL_DOMAINS_DIRECTORY` | Path to the built-in skill domains directory (defaults to the bundled domains) |
| `WORKER_TIMEOUT_SECONDS` | Max seconds to wait for spawned workers (default: `300`) |
| `AUTOCOMPACTION_THRESHOLD` | Token count at which context is compacted mid-turn (default: `96000`) |
| `CODE_EXEC_TIMEOUT` | Max seconds for a single sandboxed code execution (default: `1800`) |

A `.env` file in the working directory is loaded automatically at startup. A `--config` YAML file's fields take precedence over environment variables and `.env`, which in turn take precedence over whatever the setup wizard has saved to `~/.opendatasci/settings/global.yaml` and `~/.opendatasci/secrets/api.yaml`.
