# TUI Service

`OpenDataSciTuiService` is the service layer that sits between the terminal UI (`CLIController`) and the underlying `Agent` / `BaseSandbox`. It owns both the agent and the sandbox for the lifetime of a terminal session.

You do not normally instantiate this class directly — the TUI creates it for you. It is documented here for integrators who want to embed the OpenDataSci TUI in a custom terminal application.

## Overview

```python
from opendatasci._tui.service import OpenDataSciTuiService
from opendatasci.agents.agents import Agent
from opendatasci.sandbox.srt import SRTSandboxFactory
from opendatasci.workspace.local import LocalWorkspace
from opendatasci import OpenDataSciConfig
from pathlib import Path

workspace = LocalWorkspace("data.csv")
config = OpenDataSciConfig()
sandbox_factory = SRTSandboxFactory()

# Acquire the sandbox manually
async with sandbox_factory.create(workspace_path=Path(workspace.get_reference())) as sandbox:
    agent = Agent(workspace=workspace, config=config)
    async with agent:
        service = OpenDataSciTuiService(
            agent=agent,
            sandbox=sandbox,
            workspace_path=Path(workspace.get_reference()),
        )
        async for event in service.astream("Describe this dataset"):
            print(event)
```

## Key responsibilities

- **`astream(query)`** — delegates to the agent and yields `AgentStreamEvent` objects
- **`reset_session()`** — resets both the sandbox execution state and the agent's conversation history
- **`clear_context()`** — clears only the conversation history (sandbox variables survive)
- **`compact_chat_history()`** — calls the agent's LLM-based compaction and returns the summary
- **`get_workspace_files()`** — returns filenames visible in the workspace (used by `/ls-workspace`)

## Reference

::: opendatasci._tui.service.OpenDataSciTuiService
    options:
      show_root_heading: true
      show_source: false
      filters: ["!^_"]
