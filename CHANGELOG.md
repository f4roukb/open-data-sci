# Changelog

All notable changes to OpenDataSci will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-18

### Added

- **Human-in-the-loop command approval** — `execute_cli_command` (main agent only) gains a `request_approval` flag: before a potentially impactful command runs, `HumanApprovalManager` generates a plain-language impact assessment (description plus a heads-up when the command could harm the user's device or active work) and pauses the graph via a LangGraph interrupt until the user answers. Declined commands return a message steering the agent toward safer alternatives. New `opendatasci.human_inputs` package with the `HumanApprovalBaseManager` extension point.
- **`ApprovalRequiredEvent` stream event** — SDK consumers receive the pending command, its description, and heads-up, and resume by calling `astream` again with `"yes"`/`"no"`. The TUI renders it as an interactive approval prompt (arrow-key selection, Enter to confirm, Esc to decline).
- **Typed chat messages with provenance** — new `opendatasci/memory/` package with LangChain message subclasses (`UserMessage`, `HarnessMessage`, `SummaryMessage`, `PlanMessage`, `AgentMessage`, `AgentToAgentMessage`), each carrying a `created_at` timestamp and rendering itself with provenance metadata tags (user vs. harness vs. interrupt resume) via `RenderableMessageMixin`. A single `render_messages_for_llm` chokepoint tags messages before they reach the LLM without touching stored state.
- **Chat history compaction record** — dedicated `ChatHistoryCompaction` dataclass (`compacted_at`, `timespan`, `content`) stored in `AgentState.chat_history_compaction`; `compact_chat_history()` folds the previous compaction, all turn summaries, and the current turn into a single record, and the compaction is dropped from context automatically once the turn-summary window fills up.
- **Session plans as context** — new `Plan` context object (`context/plans.py`) that renders into a recall message stamped with its creation timestamp, so committed plans survive history compaction.
- **TUI message queue** — messages submitted while the agent is busy are queued (`PendingMessageQueue`), shown in a pending panel, and drained turn-by-turn; new `/cancel-message` and `/cancel-all-messages` commands.
- **Skill domains** — a new layer above skills: a `SkillDomain` is the map of a broad task domain (which skills exist under it and when each applies) without itself carrying task-execution know-how. The `load_skill` tool now accepts `skill_domain_name` alongside `skill_name`, loading either or both independently (each replaces its own predecessor); a new `list_skills` tool reports the available skill domains and standalone skills. New built-in domains: `competitive_data_science`, `data_science`, `data_science_education`, `deep_learning`, `machine_learning`, `quantitative_analysis`. Skill names belonging to a domain are qualified as `"<domain>::<skill>"`. `OpenDataSciConfig` gains `skill_domains_directory` / `builtin_skill_domains_directory` (`SKILL_DOMAINS_DIRECTORY` / `BUILTIN_SKILL_DOMAINS_DIRECTORY`), loaded with the same precedence as skills.
- **`opendatasci.session` package** — `BaseSessionManager` / `LocalSessionManager` track a session's conversation threads in the graph checkpointer, decoupling thread identity from the `Agent` so a stateless deployment can supply its own storage-backed manager. `Agent` accepts an optional `session_manager`, defaulting to a file-backed manager persisted at `.opendatasci/session.json`.

### Changed

- **Default models refreshed across all providers** — Anthropic `claude-sonnet-4-6` → `claude-sonnet-5`, OpenAI `gpt-5.5` → `gpt-5.6-sol` (secondary `gpt-5.4-mini` → `gpt-5.6-luna`), Bedrock `us.anthropic.claude-sonnet-4-6` → `us.anthropic.claude-sonnet-5`, Gemini / Vertex AI `gemini-2.5-pro` → `gemini-3.5-flash` (secondary `gemini-2.5-flash` → `gemini-3.1-flash-lite`), Azure OpenAI `gpt-4o` → `gpt-5.6-sol` (secondary `gpt-4o-mini` → `gpt-5.6-luna`), Ollama `llama3.2:3b` → `qwen3.5:9b`, OpenAI-compatible server `meta-llama/Llama-3.2-3B-Instruct` → `Qwen/Qwen3.5-4B` (sized to fit a 16 GB GPU at batch size 1 under vLLM's default bf16).
- **Adaptive thinking for Claude 4.6+** — the Anthropic and Bedrock clients now send `thinking: {"type": "adaptive"}` for Claude 4.6+ models (Opus 4.6/4.7/4.8, Sonnet 4.6, Sonnet 5) and omit explicit sampling parameters, which Opus 4.7+ and Sonnet 5 reject with a 400. Models without adaptive thinking (e.g. `claude-haiku-4-5`) run without a thinking config and receive the configured `temperature`.
- The `/models` TUI command now formats single-version Claude IDs (e.g. `claude-sonnet-5`) correctly.
- **`OpenDataSciConfig.from_yaml` ignores unknown keys** instead of raising `ValueError`, so config files written for other versions keep loading.
- **Tools rewritten as classes** — every `@tool`-decorated function tool (`list_python_libs`, `verify_python_code`, `enter_plan_mode`/`exit_plan_mode`, `enter_self_review_mode`/`exit_self_review_mode`, `load_skill`/`list_skills`, `ask_user_mcq`, `web_search`/`fetch_url`, worker/dataset-info/workspace tools) is now a `pydantic`-backed class deriving from new `OpenDataSciBaseTool` (async) or `OpenDataSciSyncTool` (sync, `Command`-returning) base classes in `opendatasci/tools/base.py`, instead of a closure-based factory function. Behavior is unchanged; this consolidates argument validation (including the camelCase fix below) in one place.
- **Turn-scoped agent state** — `AgentState.messages` now holds only the in-progress turn; completed turns are folded into `turn_summaries` (window raised from 3 to 10, capped at 25) instead of accumulating raw messages indefinitely.
- **TUI theming via CSS variables** — theme palettes are exported as Textual CSS variables, so every registered theme (including the color-blind-safe `visible` palette) restyles the entire app; the separate `styles_visible.tcss` stylesheet is gone.
- **TUI interrupt handling** — Ctrl+C and Esc during a running turn now stop the agent (like `/stop`) instead of quitting; quitting remains a deliberate double Ctrl+C while idle.
- **TUI auto-scroll** — the message pane pins to the bottom with a releasable anchor: new content keeps the view pinned until the user scrolls up, and scrolling back down re-engages it.
- **Sandbox command timeout** — default raised from 30 minutes to 12 hours to accommodate long-running training and hyperparameter-search jobs.
- **Quieter tool display** — low-signal tools (`list_python_libs`, `read_dataset_info`, `profile_dataset`, `list_workspace_files`, `fetch_url`, `verify_python_code`) no longer clutter the TUI transcript.
- **Database connectivity** — swapped `connectorx` for `duckdb`.
- **`/compact`** no longer echoes the generated summary back; it simply confirms completion.
- **`/clear`** now starts a brand-new checkpointer thread instead of patching the existing one, so it fully drops conversation history, turn summaries, compaction, active skills/skill domains, mode flags, and any pending interrupt in one step; it also cancels pending turn-summarization tasks and deletes the session's persisted plan. A turn still streaming when `/clear` is invoked is stopped first so it can't write the cleared conversation back on completion.
- `midturn_compaction_threshold` default raised from 80,000 to 96,000 tokens.

### Fixed

- **camelCase tool-call arguments no longer silently dropped** — some models emit camelCase keys (e.g. `requestApproval`) for multi-word snake_case parameters despite the tool schema advertising the snake_case name. LangChain's `BaseTool._parse_input` re-derives the validated dict by intersecting field names against the *original* input's keys, so a value that only validated via a camelCase alias was silently filtered out. All tools now normalize dict keys from camelCase to snake_case before validation via a new `_CamelCaseArgsMixin`.
- Command approval now uses the secondary model for the impact assessment: the primary model has extended thinking enabled, which Anthropic rejects in combination with the forced `tool_choice` that structured output requires, so every assessment call failed and the approval prompt was silently skipped. Assessment failures now also fail closed — the approval prompt still appears, showing the raw command with a fallback warning, instead of letting the command run unapproved.
- Compaction no longer masquerades as a `ChatTurnSummary` with `turn=None`; it is modeled and stored explicitly, and `clear_chat_history()` now also resets it.
- The agent graph no longer errors when a maintenance `update_state` (compact, rewind) empties `messages` mid-route.
- `verify_python_code` sends its review request with the correct message role for provider compatibility.
- Agent replies are wrapped in `AgentMessage` so they carry timestamps like every other message.
- `get_message_text_content()` strips whitespace from all text parts.

### Removed

- `thinking_budget` config field (env: `THINKING_BUDGET`) — no supported Anthropic or Bedrock model uses a fixed thinking-token budget anymore; Claude 4.6+ models manage thinking depth automatically via adaptive thinking. Config files that still contain the key load fine; the value is ignored.
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
