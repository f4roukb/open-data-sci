# Changelog

All notable changes to OpenDataSci will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-21

Initial public release.

### Added

- **Autonomous agent** — plans, writes Python, executes it in an isolated sandbox, inspects output, and iterates without human-in-the-loop prompting.
- **Sandboxed execution** — code runs inside a native OS sandbox (macOS sandbox-exec / Linux bubblewrap) via `sandbox-runtime`, preventing filesystem and network escapes.
- **Full ML/DS library surface** — Polars, Pandas, DuckDB, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna, Prophet, SHAP, PyOD, UMAP, matplotlib, seaborn, plotly, and more. Deep learning (JAX, Flax, Optax) is available via the `[jax]` extra.
- **Multi-provider LLM support** — Anthropic Claude, OpenAI, Google Gemini (AI Studio & Vertex AI), Amazon Bedrock, Azure OpenAI, Ollama, and OpenAI-compatible servers (e.g., vLLM server).
- **Extended thinking** — reasoning tokens are extracted and streamed as a dedicated event type for models that expose chain-of-thought.
- **Domain skills** — focused methodology prompts for Data Science, Machine Learning, Deep Learning, Quantitative Analysis, Competitive Data Science, and Education; custom project-level skills via `.opendatasci/skills/`.
- **Injectable skill store** — `Agent` accepts an optional `skill_store` parameter (`BaseSkillStore`) to supply a custom skill source; shared across the orchestrator and all spawned workers. Defaults to `LocalSkillStore` pointed at `<workspace>/.opendatasci/skills/`.
- **Plan mode** — agent commits to a structured multi-step plan before executing; plans are persisted to `.opendatasci/plans/`.
- **Self-review mode** — dedicated review pass catches and corrects mistakes before results are returned.
- **Code verification** — a secondary LLM critiques generated code before execution, catching logical errors independently of the main model.
- **Concurrent workers** — up to 3 concurrent worker agents for ensembling, hyperparameter search, or experiment runs; each worker can be pre-loaded with a domain skill and optionally granted web access.
- **Web search and URL fetching** — agent can search the web and retrieve URLs; configurable domain allowlist via `OpenDataSciConfig`.
- **Interactive user questions** — agent can pause mid-turn to ask the user a multiple-choice question and block until an answer is received.
- **Dataset profiling** — automatic dataset profiling with hash-based result caching to avoid redundant recomputation.
- **Tool output redaction** — tool arguments and outputs are automatically redacted from context beyond a configurable window, keeping long sessions within model limits.
- **Streaming terminal UI** — token-level streaming with a polished `rich`/`textual` interface.
- **TUI** (`opendatasci`) — point it at a data file or directory, with full provider and model configuration flags.
- **Python SDK** — async `create_agent` factory and `OpenDataSciConfig` for embedding the agent in your own applications.
- **MCP server integration** — connect to external Model Context Protocol servers via `OpenDataSciConfig.mcp_servers`.
- **Session memory** — conversation history and sandbox state are maintained across turns within a session; history can be compacted to a summary to reclaim context space.
- **Context summarisation** — automatic background compression of long conversation history to stay within model context limits.
- **Workspace loading** — load a single file or an entire directory as the agent's working dataset.

[Unreleased]: https://github.com/f4roukb/open-data-sci/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/f4roukb/open-data-sci/releases/tag/v0.1.0
