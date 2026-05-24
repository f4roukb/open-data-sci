# OpenDataSci

**OpenDataSci** is a production-grade AI agent for data science and machine learning. Point it at a CSV, Parquet, Excel, or any other data file and ask questions in plain English — it writes Python, executes it in an isolated sandbox, and returns answers, charts, and models.

## Key features

- **Conversational analysis** — multi-turn sessions with full context; follow-up questions build on previous answers
- **Sandboxed code execution** — all Python runs in an isolated SRT sandbox; no side-effects to your environment
- **Parallel workers** — complex tasks (e.g. "compare five models") are split across parallel sub-agents automatically
- **Every major LLM provider** — Anthropic, OpenAI, AWS Bedrock, Google Gemini, Vertex AI, Azure OpenAI, Ollama, vLLM
- **Extended thinking** — long-horizon reasoning via Anthropic and Bedrock's extended-thinking mode
- **Skills system** — drop Markdown skill files into `.opendatasci/skills/` to inject domain methodology
- **MCP tool servers** — connect the agent to external tool servers via `.opendatasci/mcp.json`

## Quick links

- [Getting Started](getting-started.md) — installation, providers, first steps
- [API Reference](api/index.md) — full Python SDK documentation
- [GitHub](https://github.com/f4roukb/open-data-sci)

## Quick start

=== "TUI"

    ```bash
    pip install open-data-sci
    export ANTHROPIC_API_KEY=sk-ant-...
    opendatasci data.csv
    ```

=== "Python SDK"

    ```python
    from opendatasci import create_agent

    async with create_agent("data.csv") as agent:
        async for event in agent.astream("Summarise this dataset."):
            if event.type == "token":
                print(event.content, end="", flush=True)
    ```
