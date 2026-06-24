# API Reference

This section documents the full public Python API for `opendatasci`.

## What you can do with it

- **Run the agent** — build and converse with an agent via `create_agent()` or `Agent` directly.
- **Configure it** — choose a provider, model, and behaviour via `OpenDataSciConfig`.
- **Stream its output** — consume typed events (`AgentStreamEvent`) as the agent thinks, calls tools, and answers.
- **Point it at your data** — `LocalWorkspace` (or your own `BaseWorkspace` backend) tells the agent where to find files.
- **Inspect and manage memory** — `ChatTurnContext` and friends let you reason about what the agent remembers.
- **Bring your own execution backend** — implement `BaseSandbox` / `BaseSandboxFactory` to run agent code somewhere other than the bundled local sandbox.
- **Embed the terminal experience** — `OpenDataSciTuiService` is the layer the TUI is built on, for embedding it elsewhere.

## Public API summary

All of the following are importable directly from `opendatasci`:

```python
from opendatasci import (
    create_agent,           # Factory: build a fully-wired agent from a path
    Agent,       # The agent class itself
    OpenDataSciConfig,      # Configuration dataclass
    LocalWorkspace,         # Filesystem-backed workspace
    AgentStreamEvent,       # Streaming event dataclass
    SandboxExecResult,      # Code execution result dataclass
    ChatTurnContext,        # Assembled per-turn context (recap messages + ongoing turn messages)
)
```

## Pages

| Page | What it covers |
|------|---------------|
| [create_agent](open_data_sci.md) | `create_agent()` — the recommended way to build an agent |
| [Agent](agent.md) | Full agent class: `astream`, `rewind_turn`, `compact_chat_history`, … |
| [OpenDataSciConfig](config.md) | All configuration fields and environment variable mappings |
| [TUI Service](session_manager.md) | `OpenDataSciTuiService` — service layer used by the terminal UI |
| [Memory](memory.md) | `ChatTurnContext`, `ChatTurnSummary`, message provenance tagging |
| [Workspace](workbench.md) | `BaseWorkspace`, `LocalWorkspace` |
| [Sandbox & Execution](session.md) | `BaseSandbox`, `SandboxExecResult`, TUI command allowlist |
| [Events & Types](types.md) | `AgentStreamEvent` — all event types explained |
