# Workspace

A **workspace** is the data container the agent operates on. It provides a backend-agnostic reference that the agent uses to locate datasets and derive store paths.

The only built-in implementation is `LocalWorkspace`, which is backed by a local filesystem directory.

## LocalWorkspace

`LocalWorkspace` accepts either a file path or a directory path:

- **File path** — the parent directory becomes the workspace root.
- **Directory path** — that directory is the workspace root.

```python
from opendatasci import LocalWorkspace

# Single file — workspace root is the parent directory
ws = LocalWorkspace("data/sales.csv")

# Directory — workspace root is the directory itself
ws = LocalWorkspace("data/")

# Both raise FileNotFoundError if the path does not exist
```

`LocalWorkspace` is passed to `Agent` directly when building a custom agent setup. `create_agent()` creates one automatically from the `path` argument.

```python
from opendatasci.agents.agents import Agent
from opendatasci import LocalWorkspace, OpenDataSciConfig

workspace = LocalWorkspace("./project/")
async with Agent(workspace=workspace, config=OpenDataSciConfig()) as agent:
    ...
```

## Custom workspaces

Subclass `BaseWorkspace` to implement a custom backend (S3, GCS, a database, etc.). The agent only calls `get_reference()`, so implementations only need to return a string identifier that the rest of the stack understands.

```python
from opendatasci.workspace.base import BaseWorkspace

class S3Workspace(BaseWorkspace):
    def __init__(self, bucket: str, prefix: str) -> None:
        self._ref = f"s3://{bucket}/{prefix}"

    def get_reference(self) -> str:
        return self._ref
```

## Reference

::: opendatasci.workspace.base.BaseWorkspace
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.workspace.local.LocalWorkspace
    options:
      show_root_heading: true
      show_source: false
