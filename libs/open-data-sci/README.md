# OpenDataSci

A production-grade AI agent for data science and machine learning. See the [project README](../../README.md) for an overview, benchmark results, and feature descriptions.

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [First Launch: The Setup Wizard](#first-launch-the-setup-wizard)
- [TUI Reference](#tui-reference)
- [Slash Commands](#slash-commands)
- [The `/config` Panel](#the-config-panel)
- [File Attachments](#file-attachments)
- [Key Bindings](#key-bindings)
- [Themes](#themes)
- [Python SDK](#python-sdk)
- [Embedding OpenDataSci in Your Own App](#embedding-opendatasci-in-your-own-app)
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

The sandbox that runs model-generated code shells out to native binaries that `pip` cannot install: `ripgrep` everywhere, plus `bubblewrap` and `socat` on Linux. **The TUI detects a missing dependency on first launch and offers to install it for you** (see below) — you only need to do this by hand if you're setting things up ahead of time or skip that step in the wizard:

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

Additionally, install the [GitHub CLI](https://cli.github.com) (`gh`) to use the built-in **`github.com`** skill (`execute_cli_command` shells out to it for read-oriented GitHub lookups). This one isn't checked by the setup wizard — install it yourself if you want that skill:

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

Anthropic, OpenAI, and any OpenAI-compatible server (e.g. vLLM) work with no extra.

### Capability extras

```bash
pip install "open-data-sci[deep-learning]" # Deep learning on the host — PyTorch, JAX, Transformers, Sentence-Transformers
pip install "open-data-sci[finance]"       # Finance data — yfinance
```

The `[deep-learning]` extra — deep learning directly on the host, for machines with a GPU or NPU — is required to use the **Deep Learning** skill; without it, the agent's sandboxed Python environment has no training framework available. The `[finance]` extra is required to use the **`finance.yahoo.com`** skill.

> **GPU access inside the sandbox is opt-in, and it's a real host-kernel exposure.** When a `[deep-learning]` package (`torch`, `jax`, `transformers`, `sentence-transformers`) is installed, the sandbox bind-mounts the host's GPU compute device nodes (`/dev/nvidia*`, `/dev/dri/renderD*` on Linux) so those frameworks can actually use the GPU — otherwise sandboxed code has no path to accelerator hardware at all. This is a materially different risk than the sandbox's filesystem/network isolation: it hands sandboxed code direct `ioctl` access to the host kernel's GPU driver (GPU driver ioctl surfaces have a real CVE history), and there's no GPU-equivalent of the CPU/memory resource limits the sandbox otherwise enforces. A warning is logged whenever this activates. See the module docstring in `opendatasci/sandbox/srt.py` for the full detail. macOS/Metal passthrough and NPU passthrough are not verified — see that docstring for current status. Uninstall the `[deep-learning]` packages to disable this entirely.

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

Everything — provider, model, secondary model, theme — can also be changed after launch without restarting, from the `/config` panel (see [Slash Commands](#slash-commands)).

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

Annotated config files for every supported provider are available in [`examples/configs/`](examples/configs/).

### Python SDK

```python
from opendatasci import create_agent, Invocation

async with create_agent("data.csv") as agent:
    invocation = Invocation.from_text("Summarise this dataset and train a model on the target column.")
    async for event in agent.astream(invocation):
        print(event)
```

There's no wizard here — the SDK is not the TUI, so provide `config=OpenDataSciConfig(...)` (or set env vars) up front. See [Embedding OpenDataSci in Your Own App](#embedding-opendatasci-in-your-own-app).

### More examples

The [`examples/`](examples/README.md) directory covers TUI walkthroughs, batch scripts, Jupyter notebooks, and annotated config files across every supported provider.

---

## First Launch: The Setup Wizard

The very first time you run `opendatasci` (or any time it detects nothing was resolved from `--config`/env), it runs you through up to three steps before handing control to the chat. Every step is skipped automatically once it's already satisfied, so a second launch on the same machine is typically instant.

### 1. System dependencies check

If a required sandbox binary (`ripgrep`, and on Linux `bubblewrap`/`socat`) isn't installed, a one-time screen explains what's missing and offers to install it for you with your OS package manager (you may be prompted for your password). Decline, and it shows the exact command to run yourself, then lets you continue anyway — the sandbox is only needed once the agent actually executes code, so this step never blocks you from reaching the chat.

**Setting this up manually ahead of time (see [System dependencies](#system-dependencies)) makes the wizard skip this step entirely** — useful for scripted installs, Docker images, or CI, where there's no one at the keyboard to answer the prompt.

### 2. Provider & model selection

A short, linear flow (theme, then whichever of primary/secondary provider and model aren't already set) — one choice per screen, arrow keys to pick, no back button. Set any of these non-interactively with `--config` (see [Setup with a config file](#setup-with-a-config-file)); the wizard only asks about whatever the file leaves unresolved.

### 3. Provider secrets

Whatever the chosen provider still needs — an API key, an Azure endpoint, a GCP project ID — is collected one field at a time. Each value is saved as you enter it (to `~/.opendatasci/config.yaml`), so quitting partway through doesn't lose what you've already typed, and it won't be asked again on a later launch. Environment variables and `.env` always take precedence over this saved file, so a value you export or add to `.env` later overrides whatever the wizard remembered.

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
| `--list-providers` | Print all supported providers and their default models, then exit |
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

# See all available providers and their default models
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
| **Display** | Theme (see [Themes](#themes)); Tips (toggle the rotating footer hints) |
| **Integrations** | **MCP Servers** — add, verify, or remove [MCP servers](#mcp-servers) the agent can call, either by loading candidates from an `mcp.json` file or entering one manually (name, URL, transport, headers); **Skills directory** — point at a folder of [custom skills](#custom-skills) |
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

The agent sees the attached content as structured context inline with your message. Paths are resolved relative to your current working directory.

---

## Key Bindings

| Key | Action |
|-----|--------|
| `Ctrl+C` | Stop the running agent turn; press again while idle to quit |
| `Ctrl+R` | Reset session |
| `Ctrl+L` | Clear conversation |
| `Escape` | Focus input box; step back a level in `/config` |
| `Tab` | Cycle `@file` and `/command` completions |
| `↑` / `↓` | Navigate input history or completion suggestions |

---

## Themes

Pick a theme in the setup wizard, or switch live any time from `/config` → Display → Theme — no restart required.

| Name | Description |
|------|-------------|
| `default` | Dark background with muted accents (built-in default) |
| `accessible` | Okabe-Ito palette — colour-blind safe |
| `light` | Light background with dark text |
| `solarized` | Solarized Dark by Ethan Schoonover |
| `dracula` | Dracula — vivid pastels on near-black |

---

## Python SDK

The async-first Python API gives full programmatic control over the agent, independent of the TUI — this is what you reach for to embed OpenDataSci in a script, a service, a desktop app, or a notebook.

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
    primary_temperature=0.2,
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
| `primary_temperature` | Sampling temperature for the primary model — not sent to Claude 4.6+ / Sonnet 5 models, which use adaptive thinking (env: `PRIMARY_TEMPERATURE`) |
| `name` | Display name for the agent — defaults to `"Sai"` (env: `NAME`) |
| `mcp_servers` | MCP servers the agent may connect to — see [MCP Servers](#mcp-servers) (env: `MCP_SERVERS`) |
| `skills_directory` | Path to a directory of custom skill files loaded in addition to built-ins (env: `SKILLS_DIRECTORY`) |
| `builtin_skills_directory` | Path to the built-in skills directory — override only to replace defaults entirely (env: `BUILTIN_SKILLS_DIRECTORY`) |
| `skill_domains_directory` | Path to a directory of custom skill domains, loaded in addition to built-ins (env: `SKILL_DOMAINS_DIRECTORY`) |
| `builtin_skill_domains_directory` | Path to the built-in skill domains directory — override only to replace defaults entirely (env: `BUILTIN_SKILL_DOMAINS_DIRECTORY`) |
| `worker_timeout_seconds` | Max seconds to wait for spawned workers to finish — `null` disables the timeout, default `300` (env: `WORKER_TIMEOUT_SECONDS`) |
| `autocompaction_threshold` | Token count at which context is compacted mid-turn — default `96000` (env: `AUTOCOMPACTION_THRESHOLD`) |
| `local_code_exec_timeout` | Max seconds for a single sandboxed code-execution run — default `1800` (env: `CODE_EXEC_TIMEOUT`) |

Note that `OpenDataSciConfig` itself never prompts for anything — it's a plain `pydantic-settings` model. The setup wizard is a TUI-only affair (`opendatasci/_tui/`); code built on the SDK directly is responsible for supplying whatever the chosen provider needs, same as any other library.

---

## Embedding OpenDataSci in Your Own App

The same `create_agent`/`astream` pattern the TUI is built on works unattended — no terminal, no human answering prompts — which is the shape you want for a desktop app's backend, a batch job, an API service, or a cloud deployment.

### Headless batch processing

The pattern below (trimmed from [`examples/scripts/020_script_anthropic.py`](examples/scripts/020_script_anthropic.py) — see that file, plus its [OpenAI-compatible-server](examples/scripts/021_script_openai_compatible_server.py) and [Bedrock](examples/scripts/022_script_bedrock.py) variants, for the runnable version) drives the agent over a batch of files with no TUI at all — suitable for a scheduled job, a CI pipeline, or a worker process behind an API:

```python
import asyncio
from pathlib import Path

from opendatasci import Invocation, OpenDataSciConfig, create_agent

async def analyse(csv_path: Path, config: OpenDataSciConfig) -> str:
    final = ""
    async with create_agent(str(csv_path), config=config) as agent:
        async for event in agent.astream(Invocation.from_text("Summarise this dataset.")):
            if event.type == "response":
                final = event.content
            elif event.type == "error":
                raise RuntimeError(event.content)
    return final

async def main() -> None:
    # No terminal, no prompts — every value the provider needs must be
    # supplied here or via the environment before this runs.
    config = OpenDataSciConfig(provider="anthropic", primary_temperature=0.1)
    for path in Path("data").glob("*.csv"):
        report = await analyse(path, config)
        Path("reports", path.with_suffix(".report.txt").name).write_text(report)

asyncio.run(main())
```

Swap `config` for any other provider — e.g. `OpenDataSciConfig(provider="openai_compatible_server", model="Qwen/Qwen3.5-4B", llm_server_base_url="http://gpu-box:8000/v1")` to point at a self-hosted vLLM server with no external API key at all, a natural fit for an on-prem or air-gapped deployment.

### Long-lived sessions (a desktop app or notebook)

Keep the agent alive across multiple calls — each becomes a follow-up turn in the same conversation, sharing sandbox/session state — using `AsyncExitStack` instead of a single `async with` block:

```python
from contextlib import AsyncExitStack
from opendatasci import Invocation, create_agent

stack = AsyncExitStack()
agent = await stack.enter_async_context(create_agent("data.csv"))

async for event in agent.astream(Invocation.from_text("Profile this dataset.")):
    ...  # handle events (e.g. forward "token" events to a UI as they stream)

async for event in agent.astream(Invocation.from_text("Now train a baseline model.")):
    ...  # second turn, same session — the agent still has the first turn's context

await stack.aclose()  # tears down the sandbox and any open connections
```

This is the shape a desktop app's backend or a long-running notebook kernel wants: one agent instance per user session, driven by whatever UI events (button clicks, chat input) your app already has, forwarding `agent.astream()`'s event stream to your own renderer instead of a terminal. See [`examples/notebooks/`](examples/notebooks/) for a full worked example (dataset profiling → model training → SHAP interpretation across several cells/turns).

### Cloud / multi-tenant deployment notes

- **Configuration is entirely explicit** — `OpenDataSciConfig` reads only `__init__` kwargs, environment variables, and `.env`; nothing about it assumes an interactive terminal, so it's safe to construct per-request or per-tenant in a server process.
- **Secrets belong to your deployment's own secret manager**, not `.env` — pass them as `OpenDataSciConfig(...)` kwargs sourced from wherever your platform already keeps them (env injected by the orchestrator, a secrets API, etc.).
- **Sandboxed code execution needs the same [system dependencies](#system-dependencies)** (`ripgrep`, and on Linux `bubblewrap`/`socat`) baked into your container image — there's no wizard to fall back on in a headless deployment, so install them at build time.
- **`agent.astream()`'s event stream** (`token`/`response`/`error`, plus tool-call and background-task events) is the integration surface for a custom frontend — pipe it into a WebSocket, an SSE endpoint, or your desktop app's own message-passing, rather than trying to reuse any `_tui`-internal code (that package is private and not part of the public API).
- See [`examples/configs/`](examples/configs/) for a ready-made `OpenDataSciConfig` per provider to adapt into your deployment's own config-loading path.

---

## Cloud Portability

Every stateful dependency OpenDataSci relies on — where it stores data, where it runs code, where it keeps memory — sits behind an abstract interface, and the local backend shipped today is just one implementation of each. Swap in a cloud-infrastructure-backed implementation of the same interface and the agent keeps working unchanged, which is what makes moving OpenDataSci into a multi-tenant or distributed deployment a matter of configuration and infrastructure choice, not a rewrite.

### Dependencies and their interfaces

| Dependency | Utility | Abstraction | Shipped Implementation | Recommended Cloud Implementation |
|---|---|---|---|---|
| Workspace | The dataset files and other workspace artifacts | `BaseWorkspace` | Local directory on disk | Object store (e.g., S3) |
| Code execution | Running the agent's sandboxed Python and CLI executions | `BaseSandbox` | Local OS sandbox | Firecracker microVMs |
| Project memory | Dataset profiles, notes, and session plans | `BaseContextStore` | Local project directory | MongoDB |
| Session-to-thread mapping | The session-to-thread mapping | `BaseSessionManager` | Local session file | Redis |
| Conversation checkpoints | Conversation checkpoint state | `BaseCheckpointSaver` | In-memory saver | Managed Postgres |
| Background tasks | Running and tracking background tasks | `BackgroundTaskManagerBase` | In-process async tasks | Celery (with a Redis or SQS broker) |
| Skill registry | Skill and skill-domain files shared across an agent fleet | `BaseSkillStore` | Local skill files | Object store (e.g., S3) |
| Human approval channel | Collecting the user's approve/reject decision for guarded actions in a headless deployment | `HumanApprovalBaseManager` | TUI prompt | A hosted approval workflow (e.g., a Slack app) |

None of this is enabled out of the box — the shipped implementations are all local. Cloud portability here means the architecture doesn't stand in the way: swapping in a cloud-backed implementation of one of these interfaces doesn't require touching the agent logic that depends on it.

---

## Models

OpenDataSci supports every major LLM provider. Pass `provider`/`model` in your `--config` YAML or `OpenDataSciConfig`, or pick them from the setup wizard / `/config` → Providers.

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

Run `opendatasci --list-providers` to print this table from the CLI at any time.

---

## MCP Servers

Connect the agent to external [Model Context Protocol](https://modelcontextprotocol.io) servers to give it additional tools. Only the two remote transports are supported — `http` (Streamable HTTP) and `sse` (Server-Sent Events) — a server reachable only via stdio (`command`/`args`) is out of scope, since OpenDataSci never launches a child process to talk to one.

### From the TUI

The easiest path: `/config` → Integrations → MCP Servers. Load candidate servers from an existing `mcp.json` file (pick which to add), or add one manually (name, URL, transport, headers) — either way, the server is verified reachable before being kept.

### `.opendatasci/mcp.json`

Place this inside your workspace's `.opendatasci/` directory to have it picked up automatically. The format mirrors Cursor/VS Code's convention:

```json
{
  "mcpServers": {
    "my-server": {
      "url": "http://localhost:8080",
      "type": "http",
      "headers": { "Authorization": "Bearer ..." }
    },
    "another": { "url": "http://localhost:9000", "type": "sse" }
  }
}
```

`type` defaults to `"http"` and `headers` defaults to `{}` when omitted. Tools are (re)discovered from every configured server at the start of each turn, not just once at startup, so enabling/disabling tools on the server side takes effect without restarting OpenDataSci.

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

To share skills across workspaces, set `SKILLS_DIRECTORY` in your environment or `.env` file — or point `/config` → Integrations → Skills directory at it live from inside the TUI:

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
| `PRIMARY_TEMPERATURE` | LLM sampling temperature for the primary model |
| `MCP_SERVERS` | MCP server definitions — see [MCP Servers](#mcp-servers) |
| `SKILLS_DIRECTORY` | Path to a directory of user-defined skill files |
| `BUILTIN_SKILLS_DIRECTORY` | Path to the built-in skills directory (defaults to the bundled skills) |
| `SKILL_DOMAINS_DIRECTORY` | Path to a directory of user-defined skill domains |
| `BUILTIN_SKILL_DOMAINS_DIRECTORY` | Path to the built-in skill domains directory (defaults to the bundled domains) |
| `WORKER_TIMEOUT_SECONDS` | Max seconds to wait for spawned workers (default: `300`) |
| `AUTOCOMPACTION_THRESHOLD` | Token count at which context is compacted mid-turn (default: `96000`) |
| `CODE_EXEC_TIMEOUT` | Max seconds for a single sandboxed code execution (default: `1800`) |

A `.env` file in the working directory is loaded automatically at startup. Anything set here (or exported directly) always overrides both a `--config` YAML file's corresponding field and whatever the setup wizard has saved to `~/.opendatasci/config.yaml`.
