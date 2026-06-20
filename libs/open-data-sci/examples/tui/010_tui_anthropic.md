# 01 — Exploring data from the TUI

The OpenDataSci TUI is a full-screen terminal UI. It is the fastest way to go from a raw
data file to an analysis — no code required, no notebook to spin up.

## Setup

Set your API key in the environment:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Launching

Pass a file, or a directory to load every data file inside it at once:

```bash
opendatasci sales.csv
opendatasci ./data/                        # all CSV, Excel, Parquet, … files in the folder
opendatasci sales.csv --provider openai    # use GPT instead of Claude
```

The TUI opens with a scrollable conversation panel above and an input bar at the bottom.
Type a question and press **Enter**.

---

## A realistic session

Below is a multi-turn exchange. Everything after `>` is what you type; the rest is
what OpenDataSci returns (abbreviated).

---

### Turn 1 — understand the data

```
> What does this dataset contain? Give me the column types, row count, and flag
  anything that looks off — nulls, suspicious values, heavy skew.
```

OpenDataSci loads the file, profiles the schema, computes descriptive statistics for every
column, checks for nulls and outliers, and returns a structured summary. You don't need
to specify the format; it infers it from the file extension.

---

### Turn 2 — dig into a trend

```
> Which product categories grew fastest quarter-over-quarter?
  Show a breakdown with % change and highlight any outliers.
```

OpenDataSci writes Python to compute QoQ growth, executes it in an isolated sandbox, and
reports back the numbers. If it generates a chart it saves it to the workspace and
tells you the path.

---

### Turn 3 — drill down on something surprising

```
> The Electronics Q3 number looks odd. Is it driven by a few large orders or
  a broad volume increase?
```

The full conversation stays in context, so OpenDataSci remembers the previous analysis.
No need to re-describe the data or the earlier findings.

---

### Turn 4 — get a deliverable

```
> Write a 5-bullet executive summary of the main findings for a Monday morning standup.
```

---

## Slash commands

Type `/` and press **Tab** to autocomplete. All commands take effect immediately.

| Command | What it does |
|---------|--------------|
| `/compact` | Summarises the conversation and replaces it with a compressed version — use this when sessions get long instead of losing context |
| `/reset` | Clears sandbox state and conversation; data is reloaded fresh from disk |
| `/clear` | Clears conversation history but keeps sandbox variables (DataFrames, models, etc.) |
| `/ls-workspace` | Lists every file in the current workspace |
| `/models` | Shows which primary model and secondary model are in use |
| `/themes` | Lists available colour themes and marks the active one |
| `/help` | Prints all available commands with descriptions |
| `/stop` | Interrupts a running agent turn |
| `/exit` | Quit |

**When to use `/compact` vs `/reset`:** Use `/compact` to free up context while
keeping the thread going — OpenDataSci summarises what happened and continues. Use `/reset`
when you want a completely clean start (different question, fresh sandbox state).

---

## Attaching files inline

Prefix any path with `@` to paste its content into your message. Tab-completion works
after `@`, so you can browse your filesystem without typing full paths.

```
> @src/etl_pipeline.py What does this script do and are there any obvious bugs?
```

Line-range syntax keeps context focused on the relevant section:

```
> @src/etl_pipeline.py:L45-L80 Can you simplify this section?
```

Works for `.py`, `.sql`, `.md`, `.ipynb`, and plain text files.

---

## Switching providers

```bash
# Anthropic Claude (default)
opendatasci sales.csv

# OpenAI GPT
opendatasci sales.csv --provider openai --model gpt-4o

# Google Gemini
opendatasci sales.csv --provider gemini --model gemini-2.5-pro

# Local model via Ollama  (no API key needed)
opendatasci sales.csv --provider ollama --model llama3.2:3b

# Self-hosted OpenAI-compatible server (e.g. vLLM)
opendatasci sales.csv --provider openai_compatible_server --model meta-llama/Llama-3.2-3B-Instruct
```

AWS Bedrock, Azure OpenAI, and Google Vertex AI are also supported —
see [Getting Started](../docs/getting-started.md#choosing-a-provider) for their auth setup.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Submit question |
| `Ctrl+R` | Reset session |
| `Ctrl+L` | Clear conversation |
| `Ctrl+C` (twice) | Quit |
| `Esc` | Focus input / cancel a pending choice |

---

## Tips

**Long analyses:** OpenDataSci runs to completion even if the terminal is idle.
Come back and type a follow-up when you're ready.

**Concurrent workers:** For tasks that benefit from concurrency, OpenDataSci may spawn worker agents
automatically. You'll see their output stream in as each one finishes.

**Project-level skills:** Drop a `.md` file into `.opendatasci/skills/` in your project
directory to give OpenDataSci domain-specific methodology for that project only
(e.g. a file describing your company's KPI definitions or preferred modelling approach).
