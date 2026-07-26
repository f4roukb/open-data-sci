# Changelog

All notable changes to OpenDataSci will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Six new built-in skill domains for curated site navigation** — one domain per external site, rather than a single umbrella `web` domain, since each site accrues enough distinct aspects to warrant its own set of skills:
  - `kaggle.com` — `competitions` (Data/Leaderboard/Code/Discussion tabs, submission format), `datasets` (Data Card, licensing, usability score), `prior_editions_research` (finding and vetting past winning approaches).
  - `arxiv.org` — `searching` (category/date scoping, cross-listings), `reading_papers` (abstract page first, revision history, code/checkpoint availability), `credibility` (weighing Comments field, citation count, and version history given the lack of peer review).
  - `huggingface.co` — `leaderboards` (which Spaces leaderboard to check per model category, filtering for hardware fit, translating an entry into a memory/compute estimate).
  - `github.com` — `repository_reconnaissance`, `issues_and_prs`, `code_and_repo_search`, covering GitHub interaction through the `gh` CLI (the agent has no browser). These teach which `gh` subcommand group to reach for and why, not exact flags/JSON field names — those drift across `gh` versions, so the skills point at `gh <command> --help` as the authority on current syntax.
  - `paperswithcode.com` — `sota_leaderboards` (task/dataset-scoped leaderboards linking SOTA claims to runnable code, complementing `arxiv.org`).
  - `finance.yahoo.com` — `yfinance_basics` (fetching single/multi-ticker price history, pulling fundamentals and corporate actions, and the reliability caveats of an unofficial API); requires the new `[finance]` extra.
- **`gh` (GitHub CLI) is now runnable via `execute_cli_command`** — added to the sandbox's `ALLOWED_CLI_COMMANDS`, backing the new `github.com` skill. Since the allowlist only checks the binary name (not the subcommand), this trusts callers to stick to read-oriented usage (`view`/`list`/`diff`/`search`/`api` GET) rather than write subcommands, consistent with the existing trust placed in e.g. `tar`/`zip`. Two changes make this safe by construction rather than by convention alone:
  - `SRTSandbox.execute_cli` now runs under a separate sandbox config (`_make_cli_config`) that allows network access scoped to GitHub's own hosts only (`github.com`, `api.github.com`, `codeload.github.com`, `objects.githubusercontent.com`, `raw.githubusercontent.com`); Python code execution (`execute`) is untouched and remains fully network-isolated, as does every other allowlisted CLI command.
  - `~/.config/gh` was added to the sandbox's denied-read paths, so a sandboxed `gh` call can't inherit a host `gh auth login` session — it runs unauthenticated, subject to GitHub's public rate limits.
  - Installing `gh` itself is a new system-dependency step (`brew install gh` / `apt install gh` / `dnf install gh` / `pacman -S github-cli`), documented alongside the existing ripgrep/bubblewrap/socat requirements; it is optional, only needed for the `github.com` skill.
- **`finance` extra** (`pip install "open-data-sci[finance]"`) — installs `yfinance`, backing the new `finance.yahoo.com` skill.
- **Background (non-blocking) worker scheduling** — the `task` tool (see rename below) gains a `run_mode` argument (`RunMode.FOREGROUND`, the default, or `RunMode.BACKGROUND`): background mode submits each subtask as its own tracked task (one `task_id` per subtask, not one per `task` call) and returns immediately instead of blocking on completion. New main-agent-only tools back this: `check_task` (single task's status, timestamps, progress, and result/error, as JSON) and `list_tasks` (a table of tasks filtered by status, `{RUNNING}` by default); `cancel_task` (best-effort cancellation) is unchanged. New `opendatasci.tasks` package (`BaseTaskManager` interface, `LocalTaskManager` in-process `asyncio`-backed implementation, `TaskRecord`, `TaskStatus`, `TaskProgressReport`/`TaskProgressUpdate`) tracks scheduled work for the lifetime of the session.
- **Worker progress reporting** — a background worker gains a dedicated, worker-only tool for reporting what's done, what's ongoing, blockers, and an ETA. Publishing the report is not something `BaseTaskManager` exposes: new `WorkerAgent.report_progress` classmethod owns it instead — its default implementation fetches the record via `get_task_record` and mutates it in place (sufficient for `LocalTaskManager`'s in-process, shared records), and a distributed `WorkerAgent` subclass can override it to publish through whatever channel a remote task manager actually reads from. Reports are surfaced through `check_task`. `WORKER_SYSTEM_PROMPT` now instructs workers to report progress as they go, described behaviorally rather than by tool name.
- **Background task results published to disk** — on completion, `LocalTaskManager` writes the result to `.opendatasci/workers/outputs/<task_id>.md`, so it survives past the in-memory `TaskRecord`'s session-scoped lifetime. Skipped when no output root is configured.

### Changed

- **`[jax]` extra renamed and expanded to `[deep-learning]`** (`pip install "open-data-sci[deep-learning]"`) — now installs PyTorch and Transformers/Sentence-Transformers alongside the existing JAX/Flax/Optax stack, so the **Deep Learning** skill's sandboxed environment isn't limited to a single framework. **Breaking:** `pip install "open-data-sci[jax]"` no longer works; use `[deep-learning]` instead.
- **`fetch_url` no longer restricted to an allowlisted set of domains** — the tool now fetches any http(s) URL. `OpenDataSciConfig.extra_web_domains` / `override_web_domains` (env: `EXTRA_FETCH_DOMAINS`) and `create_web_tools`'s domain arguments are removed accordingly.
- **`enter_plan_mode` and `enter_self_review_mode` merged into a single `switch_agentic_mode` tool** — takes a `mode` argument (`AgenticMode.PLAN` / `AgenticMode.SELF_REVIEW`) and refuses to switch while another mode is already active, regardless of which one. All mode-lifecycle tools (`switch_agentic_mode`, `exit_plan_mode`, `exit_self_review_mode`) now live together in a new `opendatasci/tools/modes.py`, replacing `tools/planning.py` and `tools/critic.py`. The "when to use Plan Mode vs. Self-Review Mode" guidance that used to live in the two entry tools' descriptions moved to the `# Modes` section of the main system prompt, since the tool now just points there.
- **`tools/factory.py`'s `create_agent_tools` split into `create_execution_mode_tools`, `create_plan_mode_tools`, and `create_self_review_mode_tools`** — the latter two derive their tool lists from the first, dropping `task` and `switch_agentic_mode` and keeping only the exit tool matching that mode. Previously both `exit_plan_mode` and `exit_self_review_mode` were bound to the LLM in *both* modes; each is now only visible to the model in its own mode.
- **`spawn_workers` tool renamed to `task`** (`ToolName.SPAWN_WORKERS` → `ToolName.TASK`), reflecting the new background-scheduling and parallel-execution options above; `ToolName` also gains `CHECK_TASK` and `LIST_TASKS`. **Breaking:** any code referencing the tool by its old name (`"spawn_workers"`, `ToolName.SPAWN_WORKERS`) needs updating.
- **`WorkerTask` is now `TaskTool.TaskDetails`**, nested under the tool class instead of module-level; `opendatasci.tools` now exports `TaskTool` in place of `WorkerTask`. **Breaking:** `from opendatasci.tools import WorkerTask` no longer works.
- **`tools/workers.py` merged into `tools/tasks.py`** — task creation (`task`) and task management (`check_task`/`list_tasks`/`cancel_task`) are all task tools, so they now live together. `tools/factory.py`'s `create_worker_tools()` is renamed `create_task_tools()` and returns only the `task` tool; the former `create_task_tools()` (management tools) is renamed `create_task_management_tools()`. Both take the same shared `task_manager: BaseTaskManager` so `check_task`/`list_tasks`/`cancel_task` can see tasks scheduled by `task`. **Breaking:** existing callers of `create_worker_tools()`/`create_task_tools()` need updating to the new names.
- **`BaseTaskManager`/`LocalTaskManager` methods renamed for clarity**: `submit` → `submit_task`, `list` → `list_tasks`, `cancel` → `cancel_task`. **Breaking:** existing callers need updating to the new names.
- **`BaseTaskManager` never had a `report_progress` and never will** — a task manager exposes reading tasks (`get_task_record`/`list_tasks`), not writing to them; `submit_task`'s `work` callable only receives the `task_id`, not the record. Publishing progress is a concern of whatever runs the work — see `WorkerAgent.report_progress` above.
- **`OpenDataSciSyncTool` removed; all tools now extend `OpenDataSciBaseTool`** — the sync/async split saved nothing (LangGraph mostly invokes tools via `ainvoke`, so a `_run`-only tool paid for a thread-pool hop it didn't need) while doubling the base classes to maintain. The formerly-sync, state-mutating tools (`switch_agentic_mode`, `exit_plan_mode`, `exit_self_review_mode`, `load_skill`, `list_skills`, `list_workspace_files`) now implement `_arun` directly. The camelCase-argument-normalizing mixin (`_CamelCaseArgsMixin`) was folded directly into `OpenDataSciBaseTool` rather than kept as a separate mixin, since every tool needs it.

## [0.2.1] - 2026-07-21

### Fixed

- **Streamed usage totals no longer double-count cached tokens** — Anthropic sends usage on both the `message_start` chunk (authoritative input/cache totals) and the `message_delta` chunk (repeats those totals alongside the final `output_tokens`); `langchain_anthropic` merges chunks by summing matching usage fields, so the final `AIMessage.usage_metadata` double-counted input/cache tokens for streamed calls. The stream processor now reads raw chunks directly, taking input/cache totals from the first usage-bearing chunk and the output total from the last one.
- CI: fixed the TestPyPI publish workflow (executable bit on publish/install scripts, dev-build version suffix generation) so pre-release packages publish correctly.

### Added

- Component test coverage for the TUI app/widgets, model factories, and sandbox runner.

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
- A tool call that raises instead of returning a `ToolMessage` no longer leaves its ephemeral block's spinner running forever in the TUI — `on_tool_error` events are now handled and produce an error `ToolResultEvent`.
- The command-approval prompt now states its purpose ("I need your approval to run a bash script") before showing the command, and the heads-up warning is condensed to a single line instead of a separate labeled block.

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
