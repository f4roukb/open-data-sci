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
    Agent,                  # The agent class itself
    OpenDataSciConfig,      # Configuration dataclass
    LocalWorkspace,         # Filesystem-backed workspace
    AgentStreamEvent,       # Streaming event dataclass
    SandboxExecResult,      # Code execution result dataclass
    ChatTurnContext,        # Assembled per-turn context (compaction + summaries + ongoing turn messages)
)
```

## Pages

| Page | What it covers |
|------|---------------|
| [create_agent](open_data_sci.md) | `create_agent()` — the recommended way to build an agent |
| [Agent](agent.md) | Full agent class: `astream`, `rewind_turn`, `compact_chat_history`, `BaseSessionManager`, … |
| [OpenDataSciConfig](config.md) | All configuration fields and environment variable mappings |
| [TUI Service](session_manager.md) | `OpenDataSciTuiService` — service layer used by the terminal UI |
| [Memory](memory.md) | `ChatTurnContext`, `ChatTurnSummary`, `ChatHistoryCompaction`, message provenance tagging |
| [Workspace](workbench.md) | `BaseWorkspace`, `LocalWorkspace` |
| [Skills](skills.md) | `Skill`, `SkillDomain`, `BaseSkillStore`, `LocalSkillStore` |
| [Sandbox & Execution](session.md) | `BaseSandbox`, `SandboxExecResult`, TUI command allowlist |
| [Background Tasks](tasks.md) | `BackgroundTaskManagerBase`, `BackgroundTaskManager`, `BackgroundTaskRecord`, `BackgroundTaskStatus` — the data model behind background-scheduled worker subtasks |
| [Events & Types](types.md) | `AgentStreamEvent` — all event types explained |

## Cloud portability

Every stateful dependency OpenDataSci relies on — where it stores data, where it runs code, where it keeps memory — sits behind an abstract interface, and the local backend shipped today is just one implementation of each. Swap in a cloud-infrastructure-backed implementation of the same interface and the agent keeps working unchanged, which is what makes moving OpenDataSci into a multi-tenant or distributed deployment a matter of configuration and infrastructure choice, not a rewrite.

| Dependency | Utility | Abstraction | Shipped Implementation | Open-Source Option | Popular Cloud-Provider Option |
|---|---|---|---|---|---|
| Workspace | The dataset files and other workspace artifacts | `BaseWorkspace` | Local directory on disk | Object store (e.g., Ceph, MinIO) | S3, GCS, or Azure Blob Storage |
| Code execution | Running the agent's sandboxed Python and CLI executions | `BaseSandbox` | Local OS sandbox | gVisor | Fargate |
| Project memory | Dataset profiles, notes, and session plans | `BaseContextStore` | Local project directory | MongoDB | MongoDB Atlas |
| Session-to-thread mapping | The session-to-thread mapping | `BaseSessionManager` | Local session file | Valkey | ElastiCache |
| Conversation checkpoints | Conversation checkpoint state | `BaseCheckpointSaver` (LangGraph) | In-memory saver | MongoDB | MongoDB Atlas |
| Background tasks | Running and tracking background tasks | `BackgroundTaskManagerBase` | In-process async tasks | Celery + Valkey | Celery + SQS |
| Skill store | Skill and skill-domain files shared across an agent fleet | `BaseSkillStore` | Local skill files | Object store (e.g., Ceph, MinIO) | S3, GCS, or Azure Blob Storage |
| Human approval channel | Collecting the user's approve/reject decision for guarded actions in a headless deployment | `HumanApprovalBaseManager` | TUI prompt | | |

None of this is enabled out of the box — the shipped implementations are all local. Cloud portability here means the architecture doesn't stand in the way: swapping in a cloud-backed implementation of one of these interfaces doesn't require touching the agent logic that depends on it. Each row offers one open-source path and one popular cloud-provider path, so a deployment can commit to one or mix them per dependency. Where it fits, the same technology carries across rows instead of pulling in something new per dependency: MongoDB covers both project memory and checkpoints, Valkey/ElastiCache covers both session mapping and the task queue, and the same object store covers both workspace and skill registry.
