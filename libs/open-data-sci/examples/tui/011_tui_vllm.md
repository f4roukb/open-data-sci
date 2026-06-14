# 011 — Exploring data from the TUI with vLLM

vLLM is a self-hosted inference server. It runs entirely on your hardware —
no API key, no data sent to a third party, and no per-token cost beyond electricity.
A modern GPU (or a recent Apple Silicon Mac) is required.

## When to choose vLLM

- Data that cannot leave your machine (PII, trade secrets, regulated data)
- Offline or air-gapped environments
- High-volume workloads where cloud API costs add up
- Experimentation with open-weight models

For cloud inference without managing servers, see
[`012_tui_bedrock.md`](012_tui_bedrock.md) (AWS Bedrock) or
[`010_tui_anthropic.md`](010_tui_anthropic.md) (Anthropic API).

---

## Setup

### 1 — Install vLLM and the provider extra

```bash
pip install vllm
pip install "open-data-sci[vllm]"
```

### 2 — Start the vLLM server

vLLM exposes an OpenAI-compatible HTTP API. Start it with the model you want
before launching OpenDataSci:

```bash
# Llama 3.2 3B — fits in ~8 GB VRAM, fast on a single consumer GPU
vllm serve meta-llama/Llama-3.2-3B-Instruct

# Larger model — better reasoning, needs more VRAM
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Quantized model — lower VRAM at some quality cost
vllm serve meta-llama/Llama-3.2-3B-Instruct --quantization awq
```

The server listens on `http://localhost:8000/v1` by default.
Set `LLM_SERVER_BASE_URL` only if you change the port or host:

```bash
export LLM_SERVER_BASE_URL=http://localhost:9000/v1   # non-default port
```

### 3 — Download models from Hugging Face

Some models require accepting a licence on the Hugging Face hub before
downloading. Log in once with:

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## Launching

```bash
# Default vLLM model (meta-llama/Llama-3.2-3B-Instruct)
opendatasci sales.csv --provider vllm

# Choose a different model — must match what the running server is serving
opendatasci sales.csv --provider vllm --model meta-llama/Llama-3.1-8B-Instruct

# Custom server URL
LLM_SERVER_BASE_URL=http://192.168.1.10:8000/v1 opendatasci sales.csv --provider vllm

# Load config from file
opendatasci sales.csv --config examples/config_vllm.yaml
```

---

## A realistic session

Everything after `>` is what you type. The session flow is the same as with
cloud providers; only the setup differs.

### Turn 1 — understand the data

```
> What does this dataset contain? Give me the column types, row count, and flag
  anything that looks off — nulls, suspicious values, heavy skew.
```

### Turn 2 — explore

```
> Which product categories grew fastest quarter-over-quarter?
  Show a breakdown with % change and highlight any outliers.
```

### Turn 3 — deliver

```
> Write a 5-bullet executive summary of the main findings for a Monday morning standup.
```

---

## Slash commands

| Command | What it does |
|---------|--------------|
| `/compact` | Summarise and compress the conversation to free context |
| `/reset` | Clear sandbox state and reload data from disk |
| `/clear` | Clear conversation history, keep sandbox variables |
| `/ls-workspace` | List every file in the workspace |
| `/models` | Show primary and secondary model in use |
| `/stop` | Interrupt a running agent turn |
| `/exit` | Quit |

---

## Tips for local models

**Context window:** Smaller open-weight models often have shorter context windows
(4 K–8 K tokens) than frontier cloud models. Use `/compact` earlier than you would
with Anthropic or OpenAI to avoid running out of context.

**Reasoning quality:** A 3B model will handle straightforward EDA and summaries well
but may struggle with complex multi-step ML pipelines. Upgrade to a 70B+ model (e.g.
`meta-llama/Llama-3.3-70B-Instruct`) for harder problems — but check VRAM requirements.

**Secondary model:** The secondary model (used for memory summarisation) defaults to
the same model as the primary. This is fine for local setups but doubles VRAM usage
if both are running concurrently. Pass `--secondary-provider anthropic` to offload
lightweight tasks to a cloud model if you prefer.

**Keyboard shortcuts:** identical to [`010_tui_anthropic.md`](010_tui_anthropic.md).
