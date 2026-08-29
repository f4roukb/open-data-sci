# Background Tasks

The agent can delegate work to concurrent sub-agents via its internal `task` tool. By default (`run_mode="foreground"`) this blocks until every subtask finishes and returns the combined result — visible in the stream as [`ToolCallEvent`/`SubagentEvent`/`TaskDoneEvent`](types.md). For long-running subtasks (heavy training runs, large-scale processing) the agent can instead schedule work in the background (`run_mode="background"`): each subtask becomes its own tracked task and the tool call returns immediately with a task ID, so the agent can keep the conversation moving and check back on it later.

This background-scheduling layer is backed by `opendatasci.tasks`.

## Data model

- **`AgentTaskRecord`** — a point-in-time snapshot of one background task: its `summary`, `status`, timestamps, any `AgentTaskProgressReport`s recorded against it, and its `result` or `error` once it reaches a terminal state.
- **`AgentTaskStatus`** — `running`, `completed`, `failed`, or `cancelled`.
- **`AgentTaskProgressUpdate`** / **`AgentTaskProgressReport`** — an incremental progress checkpoint (what's done, what's ongoing, blockers, and an ETA) that can be appended to `AgentTaskRecord.progress`, in call order, so a caller can see how a long-running task is progressing without waiting for it to finish. A worker running in the background is given a `report_progress` tool, bound to its own `task_id`, that populates these as it works.

## Task manager

**`AgentTaskManagerBase`** is the abstract interface a task-tracking backend implements: create a record and schedule work against it (`submit_task`), read one record (`get_task`) or all of them (`list_tasks`), request cancellation (`cancel_task`), write to a record (`upsert_record`, the primitive every mutation goes through — including `push_task_progress`, which appends an `AgentTaskProgressReport`), and await completions (`listen_task_updates`, an async iterator — not a poll — that yields each task exactly once as it reaches a terminal status).

Alongside `listen_task_updates()`, the manager also exposes a second, independent way to retrieve completions: `gather_task_updates()` returns and clears every completed record collected since the last call, and `has_task_updates()` is a cheap, non-blocking peek at whether gathering would return anything. Where `listen_task_updates()` is a notification stream (for a caller that wants to know *that* something finished, e.g. to show a UI message), the gather methods are a content buffer (for a caller that wants the record itself, on its own schedule). The two are independent — gathering does not consume the listener's stream, so each can have its own consumer without racing.

The bundled **`LocalAgentTaskManager`** runs submitted work as in-process `asyncio` tasks, keeping records for the lifetime of the manager instance (i.e. the agent session). When constructed with an `output_root`, it also publishes each completed task's result to disk at `<output_root>/<task_id>.md` — by default `.opendatasci/workers/outputs/<task_id>.md` — so the result survives past the manager's in-memory, session-scoped lifetime. Publishing is skipped when no `output_root` is configured.

The agent constructs its own `LocalAgentTaskManager` internally and exposes it as `agent.task_manager`, so a caller driving the agent (the TUI, or a hosted-service equivalent) can consume `listen_task_updates()` itself to learn about background completions without polling. It is not currently an injectable constructor argument of `Agent` the way `sandbox_factory` or `session_manager` are — a future storage backend (DB/cloud-backed) would implement the same `AgentTaskManagerBase` interface, though task *execution* stays in-process for now.

## Reference

::: opendatasci.tasks.base.AgentTaskStatus
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.AgentTaskProgressUpdate
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.AgentTaskProgressReport
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.AgentTaskRecord
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.AgentTaskManagerBase
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.local.LocalAgentTaskManager
    options:
      show_root_heading: true
      show_source: false
