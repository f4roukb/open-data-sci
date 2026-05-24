# Events & Types

## Stream event types

`Agent.astream()` yields strongly-typed event objects, all subclasses of
`BaseAgentStreamEvent`.  Import them from `opendatasci.streaming`.

```python
from opendatasci.streaming import (
    BaseAgentStreamEvent,
    ReasoningEvent,
    TokenEvent,
    ToolCallEvent,
    ToolCommunicationEvent,
    ToolResultEvent,
    WorkerDoneEvent,
    SubagentEvent,
    InputRequiredEvent,
    UsageEvent,
    ResponseEvent,
    ErrorEvent,
)
```

Each class has a `type` class variable (matching the table below) so
existing `event.type == "token"` comparisons continue to work alongside
`isinstance` checks.

### Event reference

| Class | `type` | Key fields | Description |
|-------|--------|------------|-------------|
| `ReasoningEvent` | `"reasoning"` | `content` | Extended-thinking token (Anthropic / Bedrock only) |
| `TokenEvent` | `"token"` | `content` | Incremental response text |
| `ToolCallEvent` | `"tool_call"` | `tool`, `tool_call_id`, `label`, `icon`, `display`, `summary`, `worker_summaries` | Agent is invoking a tool |
| `ToolCommunicationEvent` | `"tool_communication"` | `content`, `tool_call_id`, `tool_name` | In-progress status from a long-running tool |
| `ToolResultEvent` | `"tool_result"` | `content`, `tool_call_id`, `is_error` | Tool returned its result |
| `WorkerDoneEvent` | `"worker_done"` | `worker_idx`, `success` | A parallel worker finished |
| `SubagentEvent` | `"subagent_event"` | `content`, `worker_idx`, `event_type`, `success`, `summary` | Lifecycle event from inside a running worker |
| `InputRequiredEvent` | `"input_required"` | `content`, `choices` | Agent paused; call `astream(answer)` to resume |
| `UsageEvent` | `"usage"` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens` | Token usage update (fields are `int \| None`) |
| `ResponseEvent` | `"response"` | `content` | End-of-turn marker with the full assembled response |
| `ErrorEvent` | `"error"` | `content` | Unrecoverable error |

`AgentStreamEvent` is exported as a union type alias of all the above for
use in type annotations.

### Handling each event type

```python
from opendatasci.streaming import (
    InputRequiredEvent, ResponseEvent, TokenEvent, ToolCallEvent,
    UsageEvent, WorkerDoneEvent, ErrorEvent,
)

async for event in agent.astream("Analyse this dataset"):
    if isinstance(event, TokenEvent):
        print(event.content, end="", flush=True)

    elif isinstance(event, ToolCallEvent):
        print(f"\n→ {event.label} …")

    elif isinstance(event, WorkerDoneEvent):
        status = "done" if event.success else "failed"
        print(f"\n[worker {event.worker_idx}] {status}")

    elif isinstance(event, InputRequiredEvent):
        ans = input(f"{event.content} ({'/'.join(event.choices)}): ")
        async for follow_up in agent.astream(ans):
            ...

    elif isinstance(event, UsageEvent):
        print(f"tokens: {event.input_tokens} in / {event.output_tokens} out")

    elif isinstance(event, ResponseEvent):
        print()

    elif isinstance(event, ErrorEvent):
        print(f"\nError: {event.content}")
```

## Reference

::: opendatasci.streaming.events
    options:
      show_root_heading: false
      show_source: false
      members:
        - BaseAgentStreamEvent
        - ReasoningEvent
        - TokenEvent
        - ToolCallEvent
        - ToolCommunicationEvent
        - ToolResultEvent
        - WorkerDoneEvent
        - SubagentEvent
        - InputRequiredEvent
        - UsageEvent
        - ResponseEvent
        - ErrorEvent
