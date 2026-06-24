# Memory

OpenDataSci maintains conversation memory in three parts:

1. **Message provenance tagging** — every `HumanMessage` sent to the LLM carries a `<message_metadata>` tag (origin + timestamp), stamped once at creation and rendered fresh for each call.
2. **`ChatTurnContext`** — a dataclass assembled at the start of each turn. Its `messages` are the recall messages (rolling turn summaries, then the current plan, each its own `HumanMessage`, oldest first) followed by the ongoing turn's own messages.
3. **Explicit compaction** — `Agent.compact_chat_history()` folds the rolling turn summaries into a single, denser compaction summary, which is just another (numberless) entry in the same rolling window.

Completed turns are never kept as raw LangChain messages: the agent's internal message state is scoped to the ongoing turn only, and every other completed turn lives on exclusively as a rolling summary.

## Message provenance tagging

Every `HumanMessage` that reaches the LLM is expected to start with:

```
<message_metadata><origin>user</origin><timestamp>2024-06-01T12:00:00+00:00</timestamp></message_metadata>
```

`additional_kwargs` on a `HumanMessage` should only ever be read or written through `HumanMessageMetadata` (in `opendatasci.agents._chat_messages`, internal to the agent) — never as a raw dict — so its shape stays consistent everywhere. Its fields (`origin`, `timestamp`, `is_input_on_interrupt`) default to "unset" (`None`/`False`), so producers can tell what's actually been set and fill in only what's missing rather than overwriting it. `ChatMessageOrigin` is `USER` (the actual end user), `HARNESS` (plans, summaries, and other synthesized content), or `UNSPECIFIED` (a render-time fallback for anything that was never stamped). `HumanMessageMetadata` itself implements `LLMDigestibleMixin`: `to_content()` renders the tag above from its own fields.

This is a two-step process, deliberately split so a timestamp is generated exactly once:

- **`stamp_chat_message_metadata(message, origin, timestamp=None)`** — called once, at the point a `HumanMessage` is created (e.g. `Agent._prepare_user_message`, the worker's task message). Parses the message's current `HumanMessageMetadata`; if `origin`/`timestamp` are both already set, returns the message unchanged. Otherwise fills in only the missing ones and writes the result back to `additional_kwargs` (via `HumanMessageMetadata.attach_to`) — never into `content` — so it travels with the message through state/checkpoints untouched.
- **`render_messages_for_llm(messages)`** — called once per LLM call, inside `ChatHistoryBuilder.build()` (and directly in `AgentNode` when no `ChatHistoryBuilder` is configured, e.g. for worker agents). Reads each message's `HumanMessageMetadata` and bakes `to_content()`'s tag into a *transient* copy of its content. Never mutates the original, and the rendered copy is never written back to state — `AgentState.messages` always holds the original, untagged content.

## ChatTurnContext

`ChatTurnContext` is a dataclass assembled internally at the start of each turn. It carries two things:

- **`messages`** — recall messages first (one `HumanMessage` per rolling turn summary, then the current plan if any), then the current turn's own Human/AI/Tool messages — already rendered for the LLM (see above). Never contains `SystemMessage`s.
- **`turn_summaries`** — the updated rolling list of `ChatTurnSummary` records to write back to agent state.

The agent creates and consumes `ChatTurnContext` internally; you don't construct it yourself. Inspect turn summaries from graph state if needed:

```python
snapshot = agent.graph.get_state({"configurable": {"thread_id": session_id}})
turn_summaries = snapshot.values.get("turn_summaries", [])
for record in turn_summaries:
    print(record.to_content())
```

## ChatTurnSummary and LLMDigestibleMixin

`ChatTurnSummary` implements `LLMDigestibleMixin` — any class that defines `to_content() -> str` to render itself as LLM-facing text. `Plan` (the session plan, in `opendatasci.context.plans`) implements the same mixin.

`ChatTurnSummary.turn` is `int | None`. A regular per-turn summary has a turn number; a **compaction summary** — produced by `ChatHistoryCompactor` — has `turn=None` and folds many turns into one record (its full text lives in `agent`, with `user`/`actions` left empty). A compaction summary is otherwise an ordinary entry in `turn_summaries`: it ages out of the rolling window via the exact same FIFO eviction as any other summary, once enough new turns push it out.

## Explicit compaction

`Agent.compact_chat_history()` has the same effect on the conversation as `clear_chat_history()` — the ongoing turn's raw messages are wiped — except `turn_summaries` ends up holding one compaction record instead of being emptied:

1. `ChatHistoryCompactor.compact(turn_summaries)` folds every entry in `turn_summaries` (which may already include an older compaction summary) into one new compaction summary via the LLM.
2. The ongoing turn's messages are wiped, mirroring `clear_chat_history()`.
3. `turn_summaries` is replaced with `[compaction_summary]`.

```python
new_summary_text = await agent.compact_chat_history()
```

## Reference

::: opendatasci.agents.chat_memory.ChatTurnContext
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.agents.chat_memory.ChatTurnSummary
    options:
      show_root_heading: true
      show_source: false
