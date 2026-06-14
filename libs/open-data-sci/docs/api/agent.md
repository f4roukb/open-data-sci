# Agent

`Agent` is the core conversational AI agent. It wraps a LangGraph state machine that orchestrates LLM calls, tool execution, concurrent workers, and rolling memory.

## Lifecycle

The agent must be used as an async context manager. The sandbox is created on entry and closed on exit:

```python
from opendatasci import create_agent

async with create_agent("data.csv") as agent:
    async for event in agent.astream("Describe this dataset"):
        print(event)
```

For advanced use cases where you need to control each dependency explicitly, construct the agent directly:

```python
from opendatasci.agents.agents import Agent
from opendatasci.workspace.local import LocalWorkspace
from opendatasci import OpenDataSciConfig

workspace = LocalWorkspace("./data/")
config = OpenDataSciConfig(provider="anthropic")

async with Agent(workspace=workspace, config=config) as agent:
    async for event in agent.astream("Analyse sales trends"):
        ...
```

## Streaming events

`agent.astream()` is an async generator that yields [`AgentStreamEvent`](types.md) objects as the agent works. See the [Events & Types](types.md) page for the full event taxonomy.

### Handling an interrupt

Some tools pause the agent and ask the user to pick an option (for example, to confirm a destructive operation). When this happens an `AgentStreamEvent` with `type="input_required"` is yielded. Resume the agent by calling `astream()` again with the user's answer:

```python
async for event in agent.astream(query):
    if event.type == "input_required":
        choice = input(f"{event.content} [{', '.join(event.metadata['choices'])}]: ")
        async for follow_up in agent.astream(choice):
            # process follow_up events as usual
            ...
    elif event.type == "token":
        print(event.content, end="", flush=True)
    elif event.type == "response":
        print()
```

## Managing conversation history

| Method | Description |
|--------|-------------|
| `clear_chat_history()` | Remove all messages and rolling memory summaries. Preserves sandbox state. |
| `rewind_turn()` | Remove only the last turn (user message + agent response) from the graph state. |
| `compact_chat_history()` | Use the LLM to summarise old turns, then discard them. Returns the summary text. Use this instead of `clear_chat_history` when you want to keep context across a long session without blowing up the context window. |

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
        - rewind_turn
        - clear_chat_history
        - compact_chat_history
        - graph

---

## ConcurrentWorkerAgent

`ConcurrentWorkerAgent` is the sub-agent spawned internally when the orchestrator delegates subtasks to concurrent workers. You do not normally construct this directly.

::: opendatasci.agents.agents.ConcurrentWorkerAgent
    options:
      show_root_heading: true
      show_source: false
      members:
        - run
