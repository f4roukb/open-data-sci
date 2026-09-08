# create_agent

`create_agent` is the recommended entry point for the Python SDK. It constructs a fully-wired [`Agent`](agent.md) from a local file or directory path, wiring up the workspace, sandbox factory, skill store, and persistence stores automatically.

## Usage

```python
from opendatasci import Invocation, create_agent

async with create_agent("data.csv") as agent:
    async for event in agent.astream(Invocation.from_text("What is the average revenue by region?")):
        if event.type == "token":
            print(event.content, end="", flush=True)
```

The function returns an `Agent` that must be used as an async context manager. The sandbox is created on `__aenter__` and released on `__aexit__`.

## With a custom config

```python
from opendatasci import create_agent, Invocation, OpenDataSciConfig

config = OpenDataSciConfig(
    provider="openai",
    model="gpt-5.6-sol",
    openai_api_key="sk-...",
)

async with create_agent("/data/sales.parquet", config=config) as agent:
    async for event in agent.astream(Invocation.from_text("Train a gradient-boosting classifier.")):
        ...
```

## Embedding OpenDataSci in your own app

The same `create_agent`/`astream` pattern the TUI is built on works unattended — no terminal, no human answering prompts — which is the shape you want for a desktop app's backend, a batch job, an API service, or a cloud deployment.

### Headless batch processing

This pattern drives the agent over a batch of files with no TUI at all — suitable for a scheduled job, a CI pipeline, or a worker process behind an API:

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

This is the shape a desktop app's backend or a long-running notebook kernel wants: one agent instance per user session, driven by whatever UI events (button clicks, chat input) your app already has, forwarding `agent.astream()`'s event stream to your own renderer instead of a terminal — for instance, a notebook that profiles a dataset, trains a model, and runs a SHAP interpretation across several cells, each a follow-up turn in the same session.

### Cloud / multi-tenant deployment notes

- **Configuration is entirely explicit** — `OpenDataSciConfig` reads only `__init__` kwargs, environment variables, and `.env`; nothing about it assumes an interactive terminal, so it's safe to construct per-request or per-tenant in a server process.
- **Secrets belong to your deployment's own secret manager**, not `.env` — pass them as `OpenDataSciConfig(...)` kwargs sourced from wherever your platform already keeps them (env injected by the orchestrator, a secrets API, etc.).
- **Sandboxed code execution needs the same system dependencies** (`ripgrep`, and on Linux `bubblewrap`/`socat`) baked into your container image — there's no wizard to fall back on in a headless deployment, so install them at build time.
- **`agent.astream()`'s event stream** (`token`/`response`/`error`, plus tool-call and background-task events) is the integration surface for a custom frontend — pipe it into a WebSocket, an SSE endpoint, or your desktop app's own message-passing, rather than trying to reuse any `_tui`-internal code (that package is private and not part of the public API).
- A ready-made `OpenDataSciConfig` ships for each provider to adapt into your deployment's own config-loading path.

## Reference

::: opendatasci.agents.agents_factory.create_agent
    options:
      show_root_heading: true
      show_source: true
