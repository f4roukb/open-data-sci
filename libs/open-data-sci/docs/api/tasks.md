# Background Tasks

The agent can delegate work to concurrent sub-agents via its internal `task` tool. Each spawned worker runs in its own sandbox, so code and CLI-command executions across workers are genuinely parallelized, not just interleaved. By default (`run_mode="foreground"`) this blocks until every subtask finishes and returns the combined result — visible in the stream as [`ToolCallEvent`/`SubagentEvent`/`TaskDoneEvent`](types.md). For long-running subtasks (heavy training runs, large-scale processing) the agent can instead schedule work in the background (`run_mode="background"`): each subtask becomes its own tracked task and the tool call returns immediately with a task ID, so the agent can keep the conversation moving and check back on it later.

This background-scheduling layer is backed by `opendatasci.tasks`.

## Data model

- **`BackgroundTaskRecord`** — a point-in-time snapshot of one background task: its `summary`, `status`, timestamps, any `BackgroundTaskProgressReport`s recorded against it, its plain-text `activity` log (one entry per tool call the worker makes — see [Activity log](#activity-log) below), and its `result` or `error` once it reaches a terminal state.
- **`BackgroundTaskStatus`** — `running`, `completed`, `failed`, or `cancelled`.
- **`BackgroundTaskProgressUpdate`** / **`BackgroundTaskProgressReport`** — an incremental progress checkpoint (what's done, what's ongoing, blockers, and an ETA) that can be appended to `BackgroundTaskRecord.progress`, in call order, so a caller can see how a long-running task is progressing without waiting for it to finish. A worker running in the background is given a `report_progress` tool, bound to its own `task_id`, that populates these as it works.
- **`BackgroundTaskUpdate`** / **`BackgroundTaskUpdateKind`** — one event pushed against a task: `completed` (the task reached a terminal state — `status`/`result`/`error` are populated) or `progress` (a monitor's regex matched the task's activity — `monitor_id`/`pattern`/`matched_text` are populated; see [Monitoring task activity](#monitoring-task-activity)). `BackgroundTaskUpdate.to_message()` renders either kind as the `TaskMessage` fed back to the model.

## Task manager

`agent.task_manager` is a **`BackgroundTaskManagerBase`** — the interface for querying and observing background work from outside the agent:

| Method | Use it to |
|---|---|
| `get_task(task_id)` | Look up one task's current record. |
| `list_tasks()` | List every tracked task. |
| `cancel_task(task_id)` | Request cancellation (best-effort). |
| `push_activity(task_id, entry)` | Append one plain-text entry to a task's activity log (used internally by the `task` tool as a worker's tool calls complete). |
| `monitor_task(task_id, regex_patterns)` | Register one monitor per pattern against a task's activity log; returns the new monitor IDs. See [Monitoring task activity](#monitoring-task-activity). |
| `stop_monitoring_task(task_id=..., monitor_ids=...)` | Remove monitors, by owning task, by explicit ID, or both (to disable specific monitors on one task). Raises `ValueError` — naming the offending ID(s) — for an unknown task, an unknown monitor, or a monitor that doesn't belong to the given task. |
| `list_task_monitors(task_id)` | Return `{monitor_id: pattern}` for a task's currently active monitors. |
| `record_task_update(task_id, kind, ...)` | The single write path behind both delivery mechanisms below — store a `BackgroundTaskUpdate` and notify both the doorbell and the content buffer. Most callers use `push_activity`/`monitor_task` rather than this directly. |
| `listen_task_updates()` | `async for task_id, update_id in agent.task_manager.listen_task_updates():` — yields as soon as an update (a completion or a monitor match) is recorded against a task. Use this to show a notification or trigger your own follow-up logic without polling. Single-consumer: each update is delivered exactly once. |
| `pull_task_updates()` / `has_task_updates()` | Non-blocking drain of updates collected since the last pull — independent of `listen_task_updates()`, so each can have its own consumer without racing. |

While a turn is already in progress, the agent drains its own task manager automatically as work completes, so a result can change what it does next within the same turn rather than sitting unused until the turn ends. Starting a *new* turn to deliver a result — when the agent is otherwise idle, or once the current turn wraps up — is the driving caller's job: the bundled TUI does this for you by watching `listen_task_updates()` and kicking off a turn as soon as one is warranted, so if you're using the TUI you never need to think about this at all.

**Bringing your own task tracking.** `BackgroundTaskManagerBase` is an abstract interface, not a hardwired implementation. The bundled **`BackgroundTaskManager`** runs tasks in-process via `asyncio` (pass `output_root` to its constructor to also persist each result to `<output_root>/<task_id>.md`), but a deployment that already tracks work elsewhere — a database, a job queue, a message broker — can back `agent.task_manager` with its own implementation instead, as long as it satisfies the same interface. Driving a turn's start is always the caller's job regardless of implementation: build an `Invocation` tagged `origin=MessageOrigin.TASK` for a finished task's result and pass it to `astream()` alongside (or instead of) user text — see [Agent](agent.md).

## Activity log

Every background task accumulates a plain-text `activity` log on its `BackgroundTaskRecord`: one entry per tool call the worker makes (`tool: <name>\nresult: <output>`), appended as each call completes. It's capped at 200 entries and 32KB per entry (oldest entries/characters dropped first) to bound memory on a long-running worker, and it's readable through the `check_task` tool without any extra setup.

## Monitoring task activity

Rather than relying on a worker to proactively narrate its own progress, the scheduling agent can register a **monitor** — a regex checked against every future activity entry on a task. `monitor_task(task_id, regex_patterns)` registers one monitor per pattern and returns their `monitor_id`s; each monitor keeps firing on every matching entry for as long as it stays registered (it does not stop after the first match), and firing produces a `BackgroundTaskUpdateKind.PROGRESS` update delivered through the same channel a completion uses. Two monitors never share an ID, even when they watch the same pattern on different tasks (or twice on the same task).

Matching happens against the full, untruncated entry — never the 32KB-truncated copy that gets persisted to `activity` — so a match placed past the truncation cutoff is never missed. Only the newly-appended entry is scanned each time, never a task's prior activity, so a monitor is never re-triggered by text it has already seen; if a single entry contains multiple matches, every one of them produces its own update.

Use `stop_monitoring_task` to remove monitors once they're no longer needed — by `task_id` (every monitor on that task), by `monitor_ids` (those specific monitors, regardless of which task(s) they belong to), or by both together (to disable specific monitors on a task that has several, without touching its others). `list_task_monitors(task_id)` and the `check_task`/`list_tasks` tools (the latter via `show_monitors=True`) surface a task's currently active monitors as `Monitoring logs:\n- Monitor(<id>) Matches regex: <pattern>` text, so the agent can see what's already being watched before adding more.

These are exposed to the agent as the `monitor_task` and `stop_monitoring_task` tools, alongside `check_task`/`list_tasks`/`stop_task` — see `opendatasci.tools.tasks.create_task_management_tools`.

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

::: opendatasci.tasks.base.BackgroundTaskUpdateKind
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.tasks.base.BackgroundTaskUpdate
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
