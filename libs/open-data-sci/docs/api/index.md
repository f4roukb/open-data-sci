# API Reference

This section documents the full public Python API for `opendatasci`.

## Package structure

```
opendatasci/
├── __init__.py           # Public re-exports: create_agent, Agent,
│                         # OpenDataSciConfig, LocalWorkspace, AgentStreamEvent,
│                         # SandboxExecResult, ChatMemory
├── configs.py            # OpenDataSciConfig — all settings in one dataclass
├── agents/
│   ├── agents.py         # Agent, ConcurrentWorkerAgent
│   ├── agents_factory.py # create_agent() convenience factory
│   └── chat_memory.py    # ChatMemory, ChatHistoryCompactor, TurnSummarizer
├── workspace/
│   ├── base.py           # BaseWorkspace (ABC)
│   └── local.py          # LocalWorkspace
├── sandbox/
│   └── base.py           # BaseSandbox, BaseSandboxFactory, SandboxExecResult
├── streaming/
│   └── events.py         # AgentStreamEvent
├── context/
│   └── base.py           # BaseContextStore
└── _tui/
    └── service.py        # OpenDataSciTuiService
```

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
    ChatMemory,             # Rolling conversation memory
)
```

## Pages

| Page | What it covers |
|------|---------------|
| [create_agent](open_data_sci.md) | `create_agent()` — the recommended way to build an agent |
| [Agent](agent.md) | Full agent class: `astream`, `rewind_turn`, `compact_chat_history`, … |
| [OpenDataSciConfig](config.md) | All configuration fields and environment variable mappings |
| [TUI Service](session_manager.md) | `OpenDataSciTuiService` — service layer used by the terminal UI |
| [Memory](memory.md) | `ChatMemory`, `ChatHistoryCompactor` |
| [Workspace](workbench.md) | `BaseWorkspace`, `LocalWorkspace` |
| [Sandbox & Execution](session.md) | `BaseSandbox`, `SandboxExecResult`, TUI command allowlist |
| [Events & Types](types.md) | `AgentStreamEvent` — all event types explained |
