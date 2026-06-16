# Memory

OpenDataSci maintains two layers of conversation memory:

1. **`PreparedHistory`** — a dataclass assembled at the start of each turn. It carries the trimmed message list and any recalled context (preamble + rolling turn summaries) as plain text ready for the system prompt.
2. **`ChatHistoryCompactor`** — compacts raw LangChain message history by asking the LLM to summarise older turns. Called by `Agent.compact_chat_history()`.

## PreparedHistory

`PreparedHistory` is a dataclass returned by `ChatHistoryBuilder.build()` at the start of each turn. It carries three things:

- **`messages`** — Human/AI/Tool messages only (no `SystemMessage`). Mid-turn compaction is applied when the ongoing turn exceeds the token budget.
- **`memory_text`** — Rendered recall context (preamble + turn summaries) as a plain string for the system prompt. `None` when there is nothing to recall.
- **`turn_summaries`** — the updated rolling list of `TurnSummaryRecord` objects to write back to agent state.

The agent manages `PreparedHistory` internally via `ChatHistoryBuilder`. After every turn, `TurnSummarizer` uses the secondary LLM to generate a structured summary, which is scheduled in the background and flushed at the start of the next turn.

```python
from opendatasci.agents.chat_memory import PreparedHistory, TurnSummaryRecord

# PreparedHistory is a plain dataclass — the agent creates and consumes it internally.
# Inspect turn summaries from graph state if needed:
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

::: opendatasci.agents.chat_memory.PreparedHistory
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
