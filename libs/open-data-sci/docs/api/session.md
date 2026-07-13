# Sandbox & Execution

The **sandbox** is an isolated Python execution environment that the agent uses to run code. All data manipulation, model training, and chart generation happen inside the sandbox — your Python environment is never modified.

The default sandbox is backed by [SRT (Sandbox Runtime)](https://github.com/dreadnode/sandbox-runtime), a lightweight subprocess-based executor. Each agent session gets its own sandbox instance; state (variables, DataFrames, trained models) persists across turns within a session and is cleared on `/reset`.

## SandboxExecResult

Every code execution returns a `SandboxExecResult`:

```python
from opendatasci import SandboxExecResult

result: SandboxExecResult = await sandbox.execute("print(1 + 1)")

result.success    # bool — True if no exception was raised
result.stdout     # str  — captured stdout
result.output     # Any  — the return value of the last expression (if any)
result.error      # str | None — exception traceback on failure
result.code       # str  — the code that was executed
```

## TUI command allowlist

The agent can also run read-only shell commands via `sandbox.execute_cli()`. Only the commands in the allowlist below are permitted. Shell operators `||`, `;`, backticks, `$(`, `>`, and `<` are always rejected; `|` (pipe) and `&&` are allowed.

| Category | Commands |
|----------|----------|
| Directory | `ls`, `dir`, `pwd`, `tree` |
| File viewing | `cat`, `head`, `tail`, `file`, `stat` |
| Search | `grep`, `find`, `which` |
| Text processing | `cut`, `sort`, `uniq`, `wc`, `awk`, `sed`, `tr`, `strings` |
| Comparison / checksums | `diff`, `cmp`, `md5sum`, `sha256sum`, `shasum` |
| Structured / binary | `jq`, `xxd`, `od` |
| Output / info | `echo`, `printf`, `date`, `uname`, `printenv`, `env` |
| Archive inspection | `unzip`, `tar`, `zip` |

## Custom sandbox

Subclass `BaseSandbox` and `BaseSandboxFactory` to implement a remote or containerised sandbox:

```python
from opendatasci.sandbox.base import BaseSandbox, BaseSandboxFactory, SandboxExecResult
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

class MyRemoteSandbox(BaseSandbox):
    async def execute(self, code: str) -> SandboxExecResult:
        # send code to a remote executor …
        ...

    async def execute_cli(self, command: str) -> SandboxExecResult: ...
    def reset(self) -> None: ...
    # close() is optional — override it to release external resources
    # (e.g. a remote connection); the default implementation is a no-op.

class MyRemoteSandboxFactory(BaseSandboxFactory):
    @asynccontextmanager
    async def create(self, workspace_path: Path | None = None) -> AsyncIterator[MyRemoteSandbox]:
        sandbox = MyRemoteSandbox()
        try:
            yield sandbox
        finally:
            await sandbox.close()
```

Pass the factory to `Agent`:

```python
from opendatasci.agents.agents import Agent
from opendatasci import LocalWorkspace, OpenDataSciConfig

async with Agent(
    workspace=LocalWorkspace("data.csv"),
    sandbox_factory=MyRemoteSandboxFactory(),
    config=OpenDataSciConfig(),
) as agent:
    ...
```

## Reference

::: opendatasci.sandbox.base.SandboxExecResult
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.sandbox.base.BaseSandbox
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.sandbox.base.BaseSandboxFactory
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.sandbox.base.validate_cli_command
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.sandbox.srt.SRTSandbox
    options:
      show_root_heading: true
      show_source: false

---

::: opendatasci.sandbox.srt.SRTSandboxFactory
    options:
      show_root_heading: true
      show_source: false
