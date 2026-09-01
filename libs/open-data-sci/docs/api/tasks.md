# Background Tasks

The agent can delegate work to concurrent sub-agents via its internal `task` tool. Each spawned worker runs in its own sandbox, so code and CLI-command executions across workers are genuinely parallelized, not just interleaved. By default (`run_mode="foreground"`) this blocks until every subtask finishes and returns the combined result — visible in the stream as [`ToolCallEvent`/`SubagentEvent`/`TaskDoneEvent`](types.md). For long-running subtasks (heavy training runs, large-scale processing) the agent can instead schedule work in the background (`run_mode="background"`): each subtask becomes its own tracked task and the tool call returns immediately with a task ID, so the agent can keep the conversation moving and check back on it later.

This background-scheduling layer is backed by `opendatasci.tasks`.

## Data model

- **`BackgroundTaskRecord`** — a point-in-time snapshot of one background task: its `summary`, `status`, timestamps, any `BackgroundTaskProgressReport`s recorded against it, and its `result` or `error` once it reaches a terminal state.
- **`BackgroundTaskStatus`** — `running`, `completed`, `failed`, or `cancelled`.
- **`BackgroundTaskProgressUpdate`** / **`BackgroundTaskProgressReport`** — an incremental progress checkpoint (what's done, what's ongoing, blockers, and an ETA) that can be appended to `BackgroundTaskRecord.progress`, in call order, so a caller can see how a long-running task is progressing without waiting for it to finish. A worker running in the background is given a `report_progress` tool, bound to its own `task_id`, that populates these as it works.

## Task manager

`agent.task_manager` is a **`BackgroundTaskManagerBase`** — the interface for querying and observing background work from outside the agent:

| Method | Use it to |
|---|---|
| `get_task(task_id)` | Look up one task's current record. |
| `list_tasks()` | List every tracked task. |
| `cancel_task(task_id)` | Request cancellation (best-effort). |
| `listen_task_updates()` | `async for task_id, update_id in agent.task_manager.listen_task_updates():` — yields a task ID as soon as an update (e.g. a completion) is recorded against it. Use this to show a notification or trigger your own follow-up logic without polling. |

While a turn is already in progress, the agent drains its own task manager automatically as work completes, so a result can change what it does next within the same turn rather than sitting unused until the turn ends. Starting a *new* turn to deliver a result — when the agent is otherwise idle, or once the current turn wraps up — is the driving caller's job: the bundled TUI does this for you by watching `listen_task_updates()` and kicking off a turn as soon as one is warranted, so if you're using the TUI you never need to think about this at all.

**Bringing your own task tracking.** `BackgroundTaskManagerBase` is an abstract interface, not a hardwired implementation. The bundled **`BackgroundTaskManager`** runs tasks in-process via `asyncio` (pass `output_root` to its constructor to also persist each result to `<output_root>/<task_id>.md`), but a deployment that already tracks work elsewhere — a database, a job queue, a message broker — can back `agent.task_manager` with its own implementation instead, as long as it satisfies the same interface. Driving a turn's start is always the caller's job regardless of implementation: build an `Invocation` tagged `origin=MessageOrigin.TASK` for a finished task's result and pass it to `astream()` alongside (or instead of) user text — see [Agent](agent.md).

## Reference

::: opendatasci.tasks.base.BackgroundTaskStatus
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.BackgroundTaskProgressUpdate
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.BackgroundTaskProgressReport
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.BackgroundTaskRecord
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.BackgroundTaskManagerBase
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.local.BackgroundTaskManager
    options:
      show_root_heading: true
      show_source: false
