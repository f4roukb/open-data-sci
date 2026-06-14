# Memory

OpenDataSci maintains two layers of conversation memory:

1. **`ChatMemory`** — a rolling window of turn summaries injected into every system prompt. Keeps the agent aware of recent history without blowing up the context window.
2. **`ChatHistoryCompactor`** — compacts raw LangChain message history by asking the LLM to summarise older turns. Called by `Agent.compact_chat_history()`.

## ChatMemory

`ChatMemory` is a dataclass returned by `ChatMemoryBuilder.build()` at the start of each turn. It carries two things:

- **`messages`** — the chat history ready to feed the agent, with a leading memory `SystemMessage` prepended when there is recalled context.
- **`turn_summaries`** — the updated rolling list of `TurnSummaryRecord` objects to write back to agent state.

The agent manages `ChatMemory` internally via `ChatMemoryBuilder`. After every turn, `TurnSummarizer` uses the secondary LLM to generate a structured summary, which is scheduled in the background and flushed at the start of the next turn.

```python
from opendatasci.agents.chat_memory import ChatMemory, TurnSummaryRecord

# ChatMemory is a plain dataclass — the agent creates and consumes it internally.
# Inspect it from graph state if needed:
snapshot = agent.graph.get_state({"configurable": {"thread_id": session_id}})
turn_summaries: list[TurnSummaryRecord] = snapshot.values.get("turn_summaries", [])
for record in turn_summaries:
    print(record.format())
```

## ChatHistoryCompactor

`ChatHistoryCompactor` works on the raw LangChain message list stored in the LangGraph state. It:

1. Validates the history (must be complete turns — no dangling messages).
2. Summarises all turns except the most recent `cutoff` turns.
3. Returns a new message list with older turns replaced by a single `SystemMessage` containing the summary.

`Agent.compact_chat_history()` wraps this compactor and applies the result back to graph state automatically.

```python
from opendatasci.agents.chat_memory import ChatHistoryCompactor

compactor = ChatHistoryCompactor(llm=my_llm)

# Compact, keeping the most recent 2 turns verbatim
new_messages = await compactor.compact(messages, cutoff=2)
```

## Reference

::: opendatasci.agents.chat_memory.ChatMemory
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.agents.chat_memory.ChatHistoryCompactor
    options:
      show_root_heading: true
      show_source: false
      members:
        - compact
