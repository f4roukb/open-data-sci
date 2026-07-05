# Changelog

All notable changes to OpenDataSci will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-05

### Added

- **Human-in-the-loop command approval** — `execute_cli_command` (main agent only) gains a `request_approval` flag: before a potentially impactful command runs, `HumanApprovalManager` generates a plain-language impact assessment (description plus a heads-up when the command could harm the user's device or active work) and pauses the graph via a LangGraph interrupt until the user answers. Declined commands return a message steering the agent toward safer alternatives. New `opendatasci.human_inputs` package with the `HumanApprovalBaseManager` extension point.
- **`ApprovalRequiredEvent` stream event** — SDK consumers receive the pending command, its description, and heads-up, and resume by calling `astream` again with `"yes"`/`"no"`. The TUI renders it as an interactive approval prompt (arrow-key selection, Enter to confirm, Esc to decline).
- **Typed chat messages with provenance** — new `opendatasci/memory/` package with LangChain message subclasses (`UserMessage`, `HarnessMessage`, `SummaryMessage`, `PlanMessage`, `AgentMessage`, `AgentToAgentMessage`), each carrying a `created_at` timestamp and rendering itself with provenance metadata tags (user vs. harness vs. interrupt resume) via `RenderableMessageMixin`. A single `render_messages_for_llm` chokepoint tags messages before they reach the LLM without touching stored state.
- **Chat history compaction record** — dedicated `ChatHistoryCompaction` dataclass (`compacted_at`, `timespan`, `content`) stored in `AgentState.chat_history_compaction`; `compact_chat_history()` folds the previous compaction, all turn summaries, and the current turn into a single record, and the compaction is dropped from context automatically once the turn-summary window fills up.
- **Session plans as context** — new `Plan` context object (`context/plans.py`) that renders into a recall message stamped with its creation timestamp, so committed plans survive history compaction.
- **TUI message queue** — messages submitted while the agent is busy are queued (`PendingMessageQueue`), shown in a pending panel, and drained turn-by-turn; new `/cancel-message` and `/cancel-all-messages` commands.

### Changed

- **Turn-scoped agent state** — `AgentState.messages` now holds only the in-progress turn; completed turns are folded into `turn_summaries` (window raised from 3 to 10, capped at 25) instead of accumulating raw messages indefinitely.
- **TUI theming via CSS variables** — theme palettes are exported as Textual CSS variables, so every registered theme (including the color-blind-safe `visible` palette) restyles the entire app; the separate `styles_visible.tcss` stylesheet is gone.
- **TUI interrupt handling** — Ctrl+C and Esc during a running turn now stop the agent (like `/stop`) instead of quitting; quitting remains a deliberate double Ctrl+C while idle.
- **TUI auto-scroll** — the message pane pins to the bottom with a releasable anchor: new content keeps the view pinned until the user scrolls up, and scrolling back down re-engages it.
- **Sandbox command timeout** — default raised from 30 minutes to 12 hours to accommodate long-running training and hyperparameter-search jobs.
- **Quieter tool display** — low-signal tools (`list_python_libs`, `read_dataset_info`, `profile_dataset`, `list_workspace_files`, `fetch_url`, `verify_python_code`) no longer clutter the TUI transcript.
- **Database connectivity** — swapped `connectorx` for `duckdb`.
- **`/compact`** no longer echoes the generated summary back; it simply confirms completion.

### Fixed

- Command approval now uses the secondary model for the impact assessment: the primary model has extended thinking enabled, which Anthropic rejects in combination with the forced `tool_choice` that structured output requires, so every assessment call failed and the approval prompt was silently skipped. Assessment failures now also fail closed — the approval prompt still appears, showing the raw command with a fallback warning, instead of letting the command run unapproved.
- Compaction no longer masquerades as a `ChatTurnSummary` with `turn=None`; it is modeled and stored explicitly, and `clear_chat_history()` now also resets it.
- `verify_python_code` sends its review request with the correct message role for provider compatibility.
- Agent replies are wrapped in `AgentMessage` so they carry timestamps like every other message.
- `get_message_text_content()` strips whitespace from all text parts.

### Removed

- Dead `prompts/message_templates.py` module and unused message-utility helpers (`get_last_turn_messages`, `get_ongoing_turn_messages`, `render_turns`).

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

[Unreleased]: https://github.com/f4roukb/open-data-sci/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/f4roukb/open-data-sci/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/f4roukb/open-data-sci/releases/tag/v0.1.0
