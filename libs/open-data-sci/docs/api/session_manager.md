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
- **`reset_session()`** — clears the sandbox's execution history and the agent's conversation history
- **`clear_context()`** — clears only the conversation history
- **`compact_chat_history()`** — calls the agent's LLM-based compaction and returns the summary
- **`get_workspace_files()`** — returns filenames visible in the workspace (used by `/ls-workspace`)

## Reference

::: opendatasci._tui.service.OpenDataSciTuiService
    options:
      show_root_heading: true
      show_source: false
      filters: ["!^_"]
