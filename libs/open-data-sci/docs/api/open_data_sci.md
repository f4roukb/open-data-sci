# create_agent

`create_agent` is the recommended entry point for the Python SDK. It constructs a fully-wired [`Agent`](agent.md) from a local file or directory path, wiring up the workspace, sandbox factory, skill store, and persistence stores automatically.

## Usage

```python
from opendatasci import create_agent

async with create_agent("data.csv") as agent:
    async for event in agent.astream("What is the average revenue by region?"):
        if event.type == "token":
            print(event.content, end="", flush=True)
```

The function returns an `Agent` that must be used as an async context manager. The sandbox is created on `__aenter__` and released on `__aexit__`.

## With a custom config

```python
from opendatasci import create_agent, OpenDataSciConfig

config = OpenDataSciConfig(
    provider="openai",
    model="gpt-4o",
    openai_api_key="sk-...",
)

async with create_agent("/data/sales.parquet", config=config) as agent:
    async for event in agent.astream("Train a gradient-boosting classifier."):
        ...
```

## Reference

::: opendatasci.agents.agents_factory.create_agent
    options:
      show_root_heading: true
      show_source: true
