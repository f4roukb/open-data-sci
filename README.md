<div align="center">
  <img src="resources/logo/logo-light.png" alt="OpenDataSci" width="70%" />
</div>

**OpenDataSci is a truly autonomous AI agent purpose-built for data science and machine learning.** Point it at a dataset (single file or directory), tell it what you need: it plans with scientific rigor, writes and executes code, self-reviews its progress, and iterates fast until it gets it right. **No data science knowledge required.**

<div align="center">
  <img src="resources/demo/run-open-data-sci-fast.gif" alt="OpenDataSci demo" width="95%" />
</div>

---

## Contents

- [Benchmark](#benchmark)
- [What does OpenDataSci do?](#what-it-does)
- [For data scientists](#for-data-scientists)
- [Supported LLM providers](#supported-llm-providers)
- [Built-in ML library surface](#built-in-ml-library-surface)
- [Setup](#setup)
- [Examples](#examples)
- [Documentation](#documentation)

---

## Benchmark

**OpenDataSci v0.1.0 scored AUC 0.95069 -- Top-30% finish among 3k+ teams and 36k+ submissions.** ([Kaggle Playground Series S6E5](https://www.kaggle.com/competitions/playground-series-s6e5/leaderboard?tab=public&search=farouk+boukil))

The task was to predict whether an F1 driver will pit on the next lap. Pit stops are rare, making class imbalance a core challenge. The right call depends on dozens of interacting variables that require careful feature engineering and proper temporal handling.

The winner scored AUC 0.95503 across 195 submissions. That marginal gap relative to OpenDataSci's one-shot resolution cost a month of full-time work: a dozen model families, deep learning, notebooks with up to 400 hand-engineered features, AutoML sweeps across 4 libraries, AutoFE tools that either failed or timed out, and 186 out-of-fold models to ensemble. Claude was used throughout, yet repeatedly had to be talked out of giving up early and kept regenerating entire notebooks for marginal impact.

OpenDataSci needed one instruction: try to win the competition. Given only the zipped data, with no domain hints, prompt tuning, or human guidance, it explored the data, engineered features, tuned models, built diverse ensembles, and created a submission.

Find the winner's full writeup [here](https://www.kaggle.com/competitions/playground-series-s6e5/writeups/1st-place-by-the-skin-of-my-teeth).

---

## What does OpenDataSci do?

Most "AI for data" tools turn you into the bottleneck. Every experiment starts with re-explaining your data from scratch. Every output still needs a data scientist to verify. Every wrong turn costs a full cycle: prompt, wait, review, correct, repeat. And the moment you close the session, every insight and learned quirk of your dataset is gone.

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

## For data scientists

If you already know what you're doing, **OpenDataSci removes the friction that eats your time**: boilerplate EDA, repetitive feature engineering cycles, juggling notebooks across experiments. You stay focused on what actually requires your judgment, like improving business metrics.

Use it as a first-pass analyst: let it explore the data, surface what matters, and run the baseline while you think about strategy. Spin up parallel experiments without managing multiple environments. Inject your domain knowledge via skill files and have it applied consistently across every run. When you want to take the wheel, take it. OpenDataSci hands off cleanly.

The benchmark above was a from-scratch run with no expert guidance. With yours, it will certainly further!

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
| Deep learning | JAX, Flax, Optax (optional) |
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

The full documentation is available at [opendatasci.readthedocs.io](https://opendatasci.readthedocs.io/en/latest/), covering getting started, the Python SDK API reference, and configuration.

---

## Setup

Full installation and configuration instructions are in the [library README](libs/open-data-sci/README.md), including provider setup, environment variables, TUI flags, slash commands, key bindings, and the Python SDK reference.

---

## Examples

The [examples directory](libs/open-data-sci/examples/README.md) covers every supported provider across three usage patterns:

- **TUI walkthroughs**: interactive sessions with slash commands, file attachments, and keyboard shortcuts
- **Batch scripts**: run the agent autonomously with no human in the loop
- **Jupyter notebooks**: end-to-end ML workflows with the agent kept alive across cells
- **YAML config files**: annotated provider configurations ready to drop in

---

<div align="center">
  <sub>Licensed under Apache 2.0 · Copyright 2026 Farouk Boukil</sub>
</div>
