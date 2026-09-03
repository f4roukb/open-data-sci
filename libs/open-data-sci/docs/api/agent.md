# Agent

`Agent` is the core conversational AI agent. It orchestrates LLM calls, tool execution, concurrent workers, and rolling memory automatically, so you only need to send queries and consume the resulting stream.

## Lifecycle

The agent must be used as an async context manager. The sandbox is created on entry and closed on exit:

```python
from opendatasci import Invocation, create_agent

async with create_agent("data.csv") as agent:
    async for event in agent.astream(Invocation.from_text("Describe this dataset")):
        print(event)
```

For advanced use cases where you need to control each dependency explicitly, construct the agent directly:

```python
from opendatasci.agents.agents import Agent
from opendatasci.workspace.local import LocalWorkspace
from opendatasci import Invocation, OpenDataSciConfig

workspace = LocalWorkspace("./data/")
config = OpenDataSciConfig(provider="anthropic")

async with Agent(workspace=workspace, config=config) as agent:
    async for event in agent.astream(Invocation.from_text("Analyse sales trends")):
        ...
```

## Streaming events

`agent.astream()` takes a single `Invocation` or a `list[Invocation]` and is an async generator that yields [`AgentStreamEvent`](types.md) objects as the agent works. A list is not multiple separate requests — every item is folded into one turn, producing exactly one response. See the [Events & Types](types.md) page for the full event taxonomy.

### Handling an interrupt

Some tools pause the agent and ask the user something before continuing — a free-text or multiple-choice question, or a yes/no command-approval request. While paused, `astream()` cannot be used to start a new turn; resume with the dedicated method matching the event you received instead:

```python
async for event in agent.astream(invocation):
    if event.type == "input_required":
        choice = input(f"{event.content} [{', '.join(event.choices)}]: ")
        async for follow_up in agent.resume_with_input(choice):
            # process follow_up events as usual
            ...
    elif event.type == "approval_required":
        answer = input(f"{event.content} Allow? (y/n): ")
        async for follow_up in agent.resume_with_approval(answer.strip().lower().startswith("y")):
            ...
    elif event.type == "token":
        print(event.content, end="", flush=True)
    elif event.type == "response":
        print()
```

## Managing conversation history

| Method | Description |
|--------|-------------|
| `clear_chat_history()` | Remove all messages and rolling memory summaries. The sandbox instance itself is untouched. |
| `rewind_turn()` | Remove only the last turn (user message + agent response) from the conversation. |
| `compact_chat_history()` | Fold all turn summaries and any existing compaction into a single `ChatHistoryCompaction` record. Returns the compaction text. Use this instead of `clear_chat_history` when you want to preserve context across a long session. |

```python
# After many turns, compact instead of clearing:
summary = await agent.compact_chat_history()
print("Compacted:", summary)
```

## Reference

::: opendatasci.agents.agents.Agent
    options:
      show_root_heading: true
      show_source: false
      members:
        - astream
        - resume_with_input
        - resume_with_approval
        - is_user_input_required
        - rewind_turn
        - clear_chat_history
        - compact_chat_history

---

## Session manager

A session manager tracks the mapping from a session to its conversation threads in the graph checkpointer. Clearing the conversation (`clear_chat_history`) creates a new thread, so no prior state is visible to the LLM on the next turn.

The default `LocalSessionManager` persists this mapping to `.opendatasci/session.json` inside the workspace. For cloud or multi-process deployments — where two agent instances must share a conversation — replace it with a custom `BaseSessionManager` backed by shared storage:

```python
import uuid
from opendatasci.session.session_manager import BaseSessionManager

class RedisSessionManager(BaseSessionManager):
    def get_or_create_thread(self) -> uuid.UUID: ...
    def create_thread(self) -> uuid.UUID: ...
    def get_current_thread(self) -> uuid.UUID: ...

from opendatasci.agents.agents import Agent
from opendatasci import LocalWorkspace, OpenDataSciConfig

async with Agent(
    workspace=LocalWorkspace("data/"),
    session_manager=RedisSessionManager(),
    config=OpenDataSciConfig(),
) as agent:
    ...
```

::: opendatasci.session.session_manager.BaseSessionManager
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.session.session_manager.LocalSessionManager
    options:
      show_root_heading: true
      show_source: false

---

## WorkerAgent

`WorkerAgent` is the sub-agent spawned internally when OpenDataSci delegates subtasks to concurrent workers. You do not normally construct this directly.

Workers run concurrently by design, each in its own sandbox, so their code and CLI-command executions each spawn a dedicated subprocess — the compute-bound work itself is genuinely parallelized across simultaneous executions, not just interleaved.

A subtask can run in one of two modes. In the foreground, it runs to completion and returns its result immediately, blocking the conversation until it's done — the right choice when OpenDataSci needs the result before it can proceed. In the background, it's scheduled instead: the tool call returns right away with a task ID, and OpenDataSci keeps the conversation moving while the work continues — the right choice for long-running work like heavy training runs or large-scale processing. `agent.task_manager` exposes the full task-tracking API for background subtasks, including `listen_task_updates()` for learning about completions without polling — see [Background Tasks](tasks.md).

A background task's result reaches the model automatically if it finishes while the agent is already working on something else. If it finishes while the agent is idle, delivering it is up to whoever is driving the agent — the bundled TUI does this for you — see [Background Tasks](tasks.md) for how.

::: opendatasci.agents.workers.WorkerAgent
    options:
      show_root_heading: true
      show_source: false
      members:
        - ainvoke
