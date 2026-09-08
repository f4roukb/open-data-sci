# TUI Service

`OpenDataSciTuiService` is the service layer that sits between the terminal UI (`CLIController`) and the underlying `Agent` / `BaseSandbox`. It owns both the agent and the sandbox for the lifetime of a terminal session.

You do not normally instantiate this class directly — the TUI creates it for you. It is documented here for integrators who want to embed the OpenDataSci TUI in a custom terminal application.

## Overview

```python
from opendatasci._tui.service import OpenDataSciTuiService
from opendatasci import create_agent, Invocation, OpenDataSciConfig
from pathlib import Path

config = OpenDataSciConfig()

async with create_agent("data.csv", config=config) as agent:
    service = OpenDataSciTuiService(
        agent=agent,
        sandbox=agent._sandbox,
        workspace_path=Path(agent._workspace.get_reference()),
    )
    async for event in service.astream(Invocation.from_text("Describe this dataset")):
        print(event)
```

## Key responsibilities

- **`astream(invocation)`** — delegates to the agent and yields `AgentStreamEvent` objects (`invocation` is an `Invocation` or `list[Invocation]`, never a plain string)
- **`resume_with_input(answer)`** — resumes a pending question/choice prompt with the user's answer
- **`resume_with_approval(approved)`** — resumes a pending command-approval prompt with the user's decision
- **`is_user_input_required()`** — `True` iff the agent is paused awaiting the user's answer to a pending question or approval request
- **`rewind_turn()`** — removes the last turn from the conversation history
- **`reset_session()`** — resets the sandbox's execution session and clears the agent's context (history, turn summaries, compaction, active skills, mode flags, pending interrupts, and the persisted plan)
- **`clear_context()`** — clears the agent's context (history, turn summaries, compaction, active skills, mode flags, pending interrupts, and the persisted plan) without touching the sandbox
- **`compact_chat_history()`** — calls the agent's LLM-based compaction and returns the summary
- **`get_workspace_files()`** — returns filenames visible in the workspace (used by `/ls-workspace`)
- **`task_manager`** — property exposing the agent's `BackgroundTaskManagerBase` (see [Background Tasks](tasks.md))
- **`close()`** — releases sandbox resources (e.g. stops Docker containers); also called automatically on `__aexit__`

## Reference

::: opendatasci._tui.service.OpenDataSciTuiService
    options:
      show_root_heading: true
      show_source: false
      filters: ["!^_"]
