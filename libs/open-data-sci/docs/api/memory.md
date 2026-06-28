# Memory

OpenDataSci keeps conversation history as rolling turn summaries rather than raw messages. Completed turns are summarised and discarded; only the current in-progress turn is kept in full.

## Concepts

| Class | What it represents |
|-------|-------------------|
| `ChatTurnSummary` | A compressed record of a single completed turn. |
| `ChatHistoryCompaction` | A further-folded narrative produced by `Agent.compact_chat_history()`. |
| `ChatTurnContext` | The assembled messages fed to the LLM at the start of each turn — summaries, compaction recall, current plan, and the ongoing turn. |

## Inspecting memory from graph state

Turn summaries and the compaction record live in `AgentState` and can be read from a graph snapshot:

```python
snapshot = agent.graph.get_state({"configurable": {"thread_id": session_id}})
turn_summaries = snapshot.values.get("turn_summaries", [])
for record in turn_summaries:
    print(record.to_content())

compaction = snapshot.values.get("chat_history_compaction")
if compaction is not None:
    print(compaction.to_content())
```

## Message types

`opendatasci.memory.messages` defines typed LangChain message subclasses. Each carries a `created_at` timestamp. The principal user-facing type is `UserMessage`, which marks a message as originating from the user and flags interrupt replies via `is_input_on_interrupt`.

| Class | Base | Purpose |
|-------|------|---------|
| `UserMessage` | `HumanMessage` | A message from the user. |
| `HarnessMessage` | `HumanMessage` | A message constructed internally by the harness. |
| `SummaryMessage` | `HumanMessage` | A harness message carrying a turn-summary recall block. |
| `PlanMessage` | `HumanMessage` | A harness message carrying the current session plan. |
| `AgentMessage` | `AIMessage` | A message produced by the LLM agent. |

## Reference

::: opendatasci.memory.messages.UserMessage
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.messages.HarnessMessage
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.messages.SummaryMessage
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.messages.PlanMessage
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.messages.AgentMessage
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.chat_memory.ChatTurnContext
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.chat_memory.ChatTurnSummary
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.chat_memory.ChatHistoryCompaction
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.turn_memory.AgentLoopCompactor
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.memory.turn_memory.TurnRewinder
    options:
      show_root_heading: true
      show_source: false
