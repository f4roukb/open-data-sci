<div align="center">
  <img src="resources/logo/logo-light.png" alt="OpenDataSci" width="640" />
</div>

<div align="center">
  <p>
    <a href="#what-it-does">What it does</a> ·
    <a href="#why-not-a-general-purpose-ai-tool">Why not a general-purpose AI tool?</a> ·
    <a href="#supported-llm-providers">Models</a> ·
    <a href="libs/open-data-sci/README.md">Full Reference</a>
  </p>
</div>

**OpenDataSci is a truly autonomous AI agent purpose-built for data science and machine learning.** Point it at a workspace or file, tell it what you need, and it plans following scentific methdology, writes and executes code, checks its own work when needed, and iterates fast until it gets it right. **No data science knowledge required.**

---

## Benchmark

**OpenDataSci: AUC 0.95069. Top-30% finish among 3k+ teams and 36k+ submissions.** ([Kaggle Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/leaderboard?tab=public&search=farouk+boukil))

The task was to predict whether an F1 driver will pit on the next lap. Pit stops are rare events, making class imbalance a core challenge. The right call on any given lap depends on many domain-specific factors. For example, tyre degradation, race position, competitor strategy, safety car windows, and dozens of interacting variables that require careful feature engineering and proper temporal handling to exploit.

OpenDataSci was given the raw competition data and a single plain-English instruction. No domain hints, prompt tuning, or human guidance. It explored the data, engineered features, trained and validated models, tuned hyperparameters, and created a submission on its own.

_In comparison, the winner scored AUC 0.95503 with 195 submissions!_

---

## What it does

Most "AI for data" tools turn you into the bottleneck. Every experiment starts with re-explaining your data from scratch. Every output needs to be verified by someone who already knows data science. Every wrong turn costs a full cycle: prompt, wait, review, correct, repeat. You need domain knowledge just to catch what the tool got wrong. And the moment you close the session, every observation, every insight, every learned quirk of your dataset is gone.

**OpenDataSci is the expert you need.** It plans rigorously, executes, and catches its own mistakes before they reach you. When it goes in the wrong direction, it self-corrects. Every insight it uncovers is persisted and carried forward across sessions, so the next experiment starts smarter than the last. You set the goal. It does the work.

| | |
|--|--|
| **Full workflow** | EDA, cleaning, feature engineering, modelling, evaluation, visualisation, reporting |
| **Real code execution** | Full Python in a native OS sandbox |
| **Built-in DS methodology** | Leakage prevention, proper evaluation, causality awareness |
| **Self-review** | Every significant step is reviewed and revised before moving forward |
| **Parallel experimentation** | Up to 3 concurrent worker agents for ensemble runs, hyperparameter sweeps, strategy comparisons |
| **Persistent project memory** | Data schema, profiles, and notes accumulate across sessions |
| **Safe by default** | Sandboxed execution: everything runs safely inside your workspace |
| **Human-in-the-loop** | At genuine decision forks that impact your intended goal, it pauses and asks, then gets on with it |
| **Specialized Skills** | Data Science, Machine Learning, Deep Learning, Quantitative Analysis, Competitive DS, Education |
| **Extensible** | Drop Markdown skill files into `.opendatasci/skills/` to inject your own domain knowledge |
| **Web access** | Searches for papers, docs, and library changelogs mid-analysis |
| **MCP-ready** | Connect any MCP-compatible tool server: internal databases, custom APIs, proprietary sources |

---

## Supported LLM providers

OpenDataSci supports every major cloud provider and fully self-hosted deployments. Use your existing infrastructure, stay within your compliance boundary, or keep costs low with a local model.

- Anthropic
- OpenAI
- AWS Bedrock
- Google Gemini (AI Studio)
- Google Vertex AI
- Azure OpenAI
- Ollama (local)
- vLLM (self-hosted)

You can take it a step further and mix providers within a single session: one model for heavy reasoning, another for lightweight tasks like summarisation.

---

## Built-in ML library surface

No setup friction. OpenDataSci ships with the complete stack a practitioner would need.

| Domain | Libraries |
|--------|-----------|
| DataFrames | Polars, Pandas, DuckDB, ConnectorX, PyArrow |
| File formats | Excel (openpyxl, xlrd, fastexcel), Parquet, HDF5, JSON, XML |
| Classical ML | scikit-learn, LightGBM, CatBoost, XGBoost, statsmodels |
| Deep learning | JAX, Flax, Optax |
| AutoML / tuning | Optuna |
| Forecasting | Prophet, ARIMA, ETS |
| Interpretability | SHAP |
| Anomaly detection | PyOD |
| Imbalanced data | imbalanced-learn |
| Dimensionality reduction | UMAP |
| Validation | pandera |
| Visualisation | matplotlib, seaborn, plotly |
| Numerics | NumPy, SciPy, SymPy, Numba |

---

## Documentation

The full documentation is available at [open-data-sci.readthedocs.io](https://open-data-sci.readthedocs.io/), covering getting started, the Python SDK API reference, and configuration.

---

## Setup

Full installation and configuration instructions are in the [library README](libs/open-data-sci/README.md), including provider setup, environment variables, TUI flags, slash commands, key bindings, and the Python SDK reference.

---

## Examples

The [examples directory](libs/open-data-sci/examples/README.md) covers every supported provider across three usage patterns:

- **TUI walkthroughs** — interactive sessions with slash commands, file attachments, and keyboard shortcuts
- **Batch scripts** — run the agent autonomously with no human in the loop
- **Jupyter notebooks** — end-to-end ML workflows with the agent kept alive across cells
- **YAML config files** — annotated provider configurations ready to drop in

---

<div align="center">
  <sub>Licensed under Apache 2.0 · Copyright 2026 Farouk Boukil</sub>
</div>
