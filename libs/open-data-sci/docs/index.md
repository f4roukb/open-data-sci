# OpenDataSci

**OpenDataSci is an autonomous AI agent for data science and machine learning.** Point it at a CSV, Parquet, Excel file, or a whole directory of data — tell it what you need in plain English. It plans, writes and executes Python in an isolated sandbox, self-reviews its own work, and iterates until it gets there. No notebooks to babysit, no boilerplate EDA to write by hand.

It runs the same way whether you drive it from the terminal or embed it in your own Python service, and it works with any major LLM provider — pick the one that matches your infrastructure, compliance boundary, or budget.

## Key features

| | |
|--|--|
| **Full workflow, one agent** | EDA, cleaning, feature engineering, modelling, evaluation, visualisation, reporting |
| **Sandboxed code execution** | Every line of generated Python runs in an isolated sandbox — no side effects on your machine |
| **Self-review** | Significant steps are reviewed and revised before the agent moves on, instead of surfacing every wrong turn to you |
| **Concurrent workers** | Up to 3 sub-agents run in parallel for tasks like model comparisons or hyperparameter sweeps |
| **Persistent project memory** | Dataset notes and profiles are written to disk and carried into future sessions, so re-analysing a dataset doesn't start from zero |
| **Any major LLM provider** | Anthropic, OpenAI, AWS Bedrock, Google Gemini, Vertex AI, Azure OpenAI, Ollama, or any OpenAI-compatible server (e.g. vLLM) — mix a primary and a secondary model freely |
| **Skills system** | Drop Markdown skill files into `.opendatasci/skills/` to inject your own domain methodology; ships with Data Science, Machine Learning, Deep Learning, Quantitative Analysis, Competitive Data Science, and Data Science Education skills built in |
| **MCP tool servers** | Connect the agent to external tool servers — internal databases, proprietary APIs — via `.opendatasci/mcp.json` |
| **Rich TUI** | A terminal interface with live tool calls, streaming output, themes, and slash commands |
| **Python SDK** | An async-first API to embed the same agent in scripts, services, or notebooks |

## Quick links

- [Getting Started](getting-started.md) — installation, choosing a provider, TUI and SDK walkthroughs
- [API Reference](api/index.md) — full Python SDK documentation
- [GitHub](https://github.com/f4roukb/open-data-sci)

## Quick start

Every example below uses the default provider (Anthropic). Swap in `--provider openai`, `--provider gemini`, `--provider ollama`, or any of the [other supported providers](getting-started.md#choosing-a-provider) — the rest of the workflow doesn't change.

=== "TUI"

    ```bash
    pip install open-data-sci
    export ANTHROPIC_API_KEY=sk-ant-...
    opendatasci data.csv
    ```

=== "Python SDK"

    ```python
    import asyncio
    from opendatasci import create_agent

    async def main() -> None:
        async with create_agent("data.csv") as agent:
            async for event in agent.astream("Summarise this dataset."):
                if event.type == "token":
                    print(event.content, end="", flush=True)

    asyncio.run(main())
    ```

Continue with [Getting Started](getting-started.md) for installation requirements, every provider's auth setup, the full TUI flag reference, and the SDK's streaming event types.
