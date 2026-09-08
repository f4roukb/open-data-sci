# Getting Started

## Installation

```bash
pip install open-data-sci
```

**Requirements:**

- Python 3.12
- macOS or Linux (Windows is not supported)

### System dependencies

The sandbox that runs agent-generated code shells out to native binaries that `pip` cannot install: `ripgrep` everywhere, plus `bubblewrap` and `socat` on Linux. **The TUI detects a missing dependency on first launch and offers to install it for you** — you only need to do this by hand if you're setting things up ahead of time (scripted installs, containers, CI) or decline that step in the wizard. If you've cloned the repository, `make install-system-dependencies` runs the right command for your platform automatically:

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

### Sandbox filesystem access

Sandboxed code can write only inside the active workspace. For reads, the sandbox denies every top-level dotfile/dotdir directly under your home directory (`~/.ssh`, `~/.aws`, `~/.config`, `~/.netrc`, `~/.config/gh`, etc.) by naming convention, plus a few macOS-specific credential stores (Keychain, browser/WebKit cookie jars) — not an exhaustive hand-picked list, so it also catches locations that weren't explicitly enumerated. It's a heuristic, not a guarantee: don't rely on it as the sole protection for anything you can't afford to leak. If your Python toolchain (pyenv/uv/rye) happens to live under one of those directories, only the minimal path down to the interpreter itself is left readable — everything else nearby (e.g. a Linux credential keyring next to a `uv`-managed interpreter) stays denied.

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
pip install "open-data-sci[deep-learning]" # Deep learning on the host — PyTorch, JAX, Transformers, Sentence-Transformers
pip install "open-data-sci[finance]" # Finance data — yfinance
```

The `[deep-learning]` extra — deep learning directly on the host, for machines with a GPU or NPU — is required for the built-in **Deep Learning** skill; `[finance]` for the built-in **`finance.yahoo.com`** skill. Combine extras freely:

> **GPU access inside the sandbox is opt-in, and it's a real host-kernel exposure.** When a `[deep-learning]` package (`torch`, `jax`, `transformers`, `sentence-transformers`) is installed, the sandbox bind-mounts the host's accelerator device nodes (`/dev/nvidia*`, `/dev/dri/renderD*`, `/dev/dxg` for WSL2, and `/dev/accel/*` for NPUs on Linux) so those frameworks can actually use the hardware — otherwise sandboxed code has no path to accelerator hardware at all. This is a materially different risk than the sandbox's filesystem/network isolation: it hands sandboxed code direct `ioctl` access to the host kernel's GPU driver (GPU driver ioctl surfaces have a real CVE history), and there's no GPU-equivalent of the CPU/memory resource limits the sandbox otherwise enforces. A warning is logged whenever this activates. See the module docstring in `opendatasci/sandbox/srt.py` for the full detail — deep learning on macOS runs on CPU only, with no accelerator passthrough. Uninstall the `[deep-learning]` packages to disable this entirely.

```bash
pip install "open-data-sci[aws,gemini,deep-learning,finance]"
```

---

## Choosing a provider

OpenDataSci works with every major LLM provider. Choose one by passing a `--config`
file that sets the `provider` field in `OpenDataSciConfig` (see
[`examples/configs/`](../examples/configs/) for an annotated file per provider) —
or leave it unset and the TUI's onboarding wizard prompts for a choice interactively
on first launch.

| Provider | `provider` value | Default model | Auth |
|----------|-------------------|---------------|------|
| Anthropic *(default)* | `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `gpt-5.6-sol` | `OPENAI_API_KEY` |
| AWS Bedrock | `bedrock` | `us.anthropic.claude-sonnet-5` | boto3 credential chain |
| Google Gemini | `gemini` | `gemini-3.5-flash` | `GOOGLE_API_KEY` |
| Google Vertex AI | `vertexai` | `gemini-3.5-flash` | Application Default Credentials |
| Azure OpenAI | `azure` | `gpt-5.6-sol` | `AZURE_OPENAI_API_KEY` or service principal |
| Ollama | `ollama` | `qwen3.5:9b` | none (local server) |
| OpenAI-compatible server (e.g. vLLM) | `openai_compatible_server` | `Qwen/Qwen3.5-4B` | none (self-hosted) |

### Authentication

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY=sk-ant-...
    opendatasci data.csv
    ```

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY=sk-...
    opendatasci data.csv --config examples/configs/config_openai.yaml
    ```

=== "AWS Bedrock"

    ```bash
    # Long-lived IAM key
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export REGION=us-east-1
    opendatasci data.csv --config examples/configs/config_bedrock.yaml

    # Or use an IAM role / EC2 instance profile — no env vars needed
    ```

=== "Google Gemini"

    ```bash
    export GOOGLE_API_KEY=AIza...
    opendatasci data.csv --config examples/configs/config_gemini.yaml
    ```

=== "Google Vertex AI"

    ```bash
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT=my-project
    export GOOGLE_CLOUD_LOCATION=us-central1
    opendatasci data.csv --config examples/configs/config_vertexai.yaml
    ```

=== "Azure OpenAI"

    ```bash
    export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
    export AZURE_OPENAI_API_KEY=...
    opendatasci data.csv --config examples/configs/config_azure.yaml
    ```

=== "Ollama"

    ```bash
    # Start Ollama first: ollama serve
    opendatasci data.csv --config examples/configs/config_ollama.yaml
    ```

=== "OpenAI-compatible server"

    ```bash
    # Start any OpenAI-compatible server first, e.g. vLLM:
    # vllm serve Qwen/Qwen3.5-4B
    opendatasci data.csv --config examples/configs/config_openai_compatible_server.yaml
    ```

---

## TUI quick start

```bash
# Analyse a single file with the default Anthropic provider
opendatasci data.csv

# Load an entire directory of data files
opendatasci ./my-project/

# Change provider/model, and mix providers (heavy primary, lightweight secondary) —
# set provider/model/secondary_provider/secondary_model in the config file instead
opendatasci data.csv --config opendatasci_config.yaml
```

Anything a config file doesn't set — including the colour theme — is picked
interactively by the TUI's onboarding wizard on first launch, and can be changed
later from `/config` (or `/settings`) without relaunching.

### All TUI options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | | Path to a YAML config file |
| `--version` | | Print the installed version, then exit |

---

## Key bindings

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
| `dark (colorblind)` | Dark background, Okabe-Ito colour-blind safe palette (built-in default) |
| `dark` | Dark background with muted blue accents |
| `light` | Light background with dark text |
| `light (colorblind)` | Light background, Okabe-Ito colour-blind safe palette |

---

## Python SDK quick start

The Python API is async-first. Every public method that touches the network is a coroutine or an async generator.

### Minimal example

```python
import asyncio
from opendatasci import create_agent, Invocation

async def main() -> None:
    async with create_agent("sales.csv") as agent:
        invocation = Invocation.from_text("What is the average revenue by region?")
        async for event in agent.astream(invocation):
            if event.type == "token":
                print(event.content, end="", flush=True)
            elif event.type == "response":
                print()  # newline after final answer

asyncio.run(main())
```

### With a custom provider

```python
from opendatasci import create_agent, Invocation, OpenDataSciConfig

config = OpenDataSciConfig(
    provider="openai",
    model="gpt-5.6-sol",
    openai_api_key="sk-...",
    primary_temperature=0.2,
)

async with create_agent("data.parquet", config=config) as agent:
    async for event in agent.astream(Invocation.from_text("Train a gradient-boosting model on the target column.")):
        ...
```

### Consuming stream events

`agent.astream()` takes a single `Invocation` or a `list[Invocation]` and yields [`AgentStreamEvent`](api/types.md) objects. A list is not multiple separate requests — every item is folded into one turn, producing exactly one response. Each event has a `type` and `content` string, plus optional `metadata`.

```python
async for event in agent.astream(invocation):
    match event.type:
        case "token":
            # Incremental response text
            print(event.content, end="", flush=True)
        case "reasoning":
            # Thinking token (Anthropic / Bedrock only)
            pass
        case "tool_call":
            print(f"\n[tool] {event.content}")
        case "tool_result":
            pass
        case "task_done":
            idx = event.metadata["task_idx"]
            ok = event.metadata["success"]
            print(f"\n[worker {idx}] {'ok' if ok else 'failed'}")
        case "input_required":
            # Agent needs a choice from the user — resume with resume_with_input()
            choice = input(event.content + " ")
            async for follow_up in agent.resume_with_input(choice):
                pass  # handle follow_up events as usual
        case "approval_required":
            # Agent needs yes/no approval before running a command — resume with resume_with_approval()
            answer = input(event.content + " Allow? (y/n): ")
            async for follow_up in agent.resume_with_approval(answer.strip().lower().startswith("y")):
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
model: claude-sonnet-5

secondary_provider: openai
secondary_model: gpt-5.6-luna

primary_temperature: 0.1

worker_timeout_seconds: 600
autocompaction_threshold: 80000
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
└── .opendatasci/            # managed by OpenDataSci
    ├── session.json         # session-to-thread mapping (auto-managed)
    ├── plans/                # persisted agent plans (auto-managed)
    ├── dataset_notes/        # persisted dataset notes (auto-managed)
    ├── dataset_profiling/    # persisted dataset profile cards (auto-managed)
    ├── skills/               # optional — custom skill files (see below)
    └── skill_domains/        # optional — custom skill-domain manifests
```

### MCP tool servers

MCP servers aren't configured per-workspace. Add them either:

- Globally, via `~/.opendatasci/integrations/mcp.json` (also editable live from `/config` → Integrations → MCP Servers in the TUI), using the Cursor/VS Code `mcp.json` convention:

  ```json
  {
    "mcpServers": {
      "my-server": {
        "url": "http://localhost:3000/mcp",
        "type": "http",
        "headers": { "Authorization": "Bearer ..." }
      }
    }
  }
  ```

- Or by setting `mcp_servers` directly in `OpenDataSciConfig`.

Only the `http` and `sse` transports are supported.

### Skills

Create `.opendatasci/skills/` in your workspace and add Markdown files describing domain-specific methodology. The agent loads these automatically and applies them as additional expertise.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `PROVIDER` | LLM provider for the primary model (default: `anthropic`) |
| `MODEL` | Primary model identifier (default: provider default) |
| `SECONDARY_PROVIDER` | Provider for the secondary model (default: `anthropic`) |
| `SECONDARY_MODEL` | Secondary model for lightweight tasks (default: provider default) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI / OpenAI-compatible server API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource URL |
| `AZURE_OPENAI_API_VERSION` | Azure OpenAI API version (default: `2025-01-01-preview`) |
| `REGION` | Cloud region (Bedrock, default: `us-east-1`) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (Vertex AI) |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region |
| `LLM_SERVER_BASE_URL` | Custom endpoint (Ollama / OpenAI-compatible server) |
| `PRIMARY_TEMPERATURE` | Sampling temperature for the primary model (default: `0.0`) |
| `NAME` | Display name for the agent (default: `Sai`) |
| `MCP_SERVERS` | MCP servers the agent may connect to |
| `SKILLS_DIRECTORY` | Path to a directory of custom skill files, loaded in addition to built-ins |
| `BUILTIN_SKILLS_DIRECTORY` | Override the bundled built-in skills directory |
| `SKILL_DOMAINS_DIRECTORY` | Path to a directory of custom skill domains, loaded in addition to built-ins |
| `BUILTIN_SKILL_DOMAINS_DIRECTORY` | Override the bundled built-in skill domains directory |
| `WORKER_TIMEOUT_SECONDS` | Max seconds to wait for spawned workers to finish (default: `300`) |
| `AUTOCOMPACTION_THRESHOLD` | Token count at which context is compacted mid-turn (default: `96000`) |
| `CODE_EXEC_TIMEOUT` | Max seconds for one sandbox execution (default: `1800`) |

A `.env` file in the current working directory is loaded automatically on startup.
