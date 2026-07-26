# Background Tasks

The agent can delegate work to concurrent sub-agents via its internal `task` tool. By default (`run_mode="foreground"`) this blocks until every subtask finishes and returns the combined result — visible in the stream as [`ToolCallEvent`/`SubagentEvent`/`TaskDoneEvent`](types.md). For long-running subtasks (heavy training runs, large-scale processing) the agent can instead schedule work in the background (`run_mode="background"`): each subtask becomes its own tracked task and the tool call returns immediately with a task ID, so the agent can keep the conversation moving and check back on it later.

This background-scheduling layer is backed by `opendatasci.tasks`.

## Data model

- **`TaskRecord`** — a point-in-time snapshot of one background task: its `summary`, `status`, timestamps, any `TaskProgressReport`s recorded against it, and its `result` or `error` once it reaches a terminal state.
- **`TaskStatus`** — `running`, `completed`, `failed`, or `cancelled`.
- **`TaskProgressUpdate`** / **`TaskProgressReport`** — while a background worker runs, it can report incremental progress (what's done, what's ongoing, blockers, and an ETA). Each report is appended to `TaskRecord.progress` in call order, so a caller can see how a long-running task is progressing without waiting for it to finish.

## Task manager

**`BaseTaskManager`** is the abstract interface a task-tracking backend implements: create a record and schedule work against it (`submit_task`), read one record (`get_task`) or all of them (`list_tasks`), and request cancellation (`cancel_task`). There is no method here for *writing* to a task's record — a task manager exposes reading tasks, not mutating them. `submit_task` only hands the scheduled work its `task_id`, not the record itself.

The bundled **`LocalTaskManager`** runs submitted work as in-process `asyncio` tasks, keeping records for the lifetime of the manager instance (i.e. the agent session). When constructed with an `output_root`, it also publishes each completed task's result to disk at `<output_root>/<task_id>.md` — by default `.opendatasci/workers/outputs/<task_id>.md` — so the result survives past the manager's in-memory, session-scoped lifetime. Publishing is skipped when no `output_root` is configured.

The agent constructs its own `LocalTaskManager` internally as part of its tool set; it is not currently an injectable constructor argument of `Agent` the way `sandbox_factory` or `session_manager` are.

## Progress reporting

Publishing progress is a concern of whoever runs the work, not of the task manager — see [`WorkerAgent.report_progress`](agent.md#opendatasci.agents.agents.WorkerAgent.report_progress). Its default implementation fetches the record via `get_task` and mutates it in place, which works because `LocalTaskManager`'s records are shared, in-process objects — mutating the fetched record *is* publishing the update. A distributed `WorkerAgent` subclass running against a remote task manager can't rely on that object-identity trick, since a remote worker has no live reference into wherever the task manager actually keeps its records; it should override `report_progress` to publish through whatever channel that task manager reads from instead (a queue, object storage, a database row).

## Reference

::: opendatasci.tasks.base.TaskStatus
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.TaskProgressUpdate
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.TaskProgressReport
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.TaskRecord
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.BaseTaskManager
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.local.LocalTaskManager
    options:
      show_root_heading: true
      show_source: false
