# 011 — Exploring data from the TUI with an OpenAI-compatible server

OpenDataSci supports any self-hosted inference server that exposes an
OpenAI-compatible HTTP API — vLLM, LM Studio, llama.cpp's server,
text-generation-inference, and others. It runs entirely on your hardware —
no API key, no data sent to a third party, and no per-token cost beyond
electricity. A modern GPU (or a recent Apple Silicon Mac) is required for
most servers.

This walkthrough uses vLLM as the example server; swap in any other
OpenAI-compatible server and the `--provider openai_compatible_server` flag
still applies.

## When to choose a self-hosted server

- Data that cannot leave your machine (PII, trade secrets, regulated data)
- Offline or air-gapped environments
- High-volume workloads where cloud API costs add up
- Experimentation with open-weight models

For cloud inference without managing servers, see
[`012_tui_bedrock.md`](012_tui_bedrock.md) (AWS Bedrock) or
[`010_tui_anthropic.md`](010_tui_anthropic.md) (Anthropic API).

---

## Setup (vLLM example)

### 1 — Install vLLM

```bash
pip install vllm
```

### 2 — Start the vLLM server

vLLM exposes an OpenAI-compatible HTTP API. Start it with the model you want
before launching OpenDataSci:

```bash
# Qwen 3.5 4B — fits a 16 GB GPU at batch size 1, fast on a single consumer GPU
vllm serve Qwen/Qwen3.5-4B

# Larger model — better reasoning, needs more VRAM (~24 GB at bf16)
vllm serve Qwen/Qwen3.5-9B --port 8000

# Quantized model — lower VRAM at some quality cost
vllm serve Qwen/Qwen3.5-4B --quantization awq
```

The server listens on `http://localhost:8000/v1` by default.
Set `LLM_SERVER_BASE_URL` only if you change the port or host, or if you're
pointing at a different OpenAI-compatible server entirely:

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
# Default model (Qwen/Qwen3.5-4B)
opendatasci sales.csv --provider openai_compatible_server

# Choose a different model — must match what the running server is serving
opendatasci sales.csv --provider openai_compatible_server --model Qwen/Qwen3.5-9B

# Custom server URL
LLM_SERVER_BASE_URL=http://192.168.1.10:8000/v1 opendatasci sales.csv --provider openai_compatible_server

# Load config from file
opendatasci sales.csv --config examples/configs/config_openai_compatible_server.yaml
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
| `/cancel-all-messages` | Cancel all messages queued while the agent was busy |
| `/cancel-message` | Cancel the most recently queued message |
| `/compact` | Summarise and compress the conversation to free context |
| `/reset` | Clear sandbox state and reload data from disk |
| `/clear` | Clear conversation history; the sandbox is untouched |
| `/ls-workspace` | List every file in the workspace |
| `/models` | Show primary and secondary model in use |
| `/stop` | Interrupt a running agent turn |
| `/exit` | Quit |

---

## Tips for local models

**Context window:** Qwen 3.5 models support a 256K context window, but serving the
full window costs VRAM — vLLM caps `--max-model-len` to what fits on your GPU. If
you serve a reduced window, use `/compact` earlier than you would with Anthropic or
OpenAI to avoid running out of context.

**Reasoning quality:** A 4B model will handle straightforward EDA and summaries well
but may struggle with complex multi-step ML pipelines. Upgrade to a larger model
(e.g. `Qwen/Qwen3.5-27B`) for harder problems — but check VRAM requirements.

**Secondary model:** The secondary model (used for memory summarisation) defaults to
the same model as the primary. This is fine for local setups but doubles VRAM usage
if both are running concurrently. Pass `--secondary-provider anthropic` to offload
lightweight tasks to a cloud model if you prefer.

**Keyboard shortcuts:** identical to [`010_tui_anthropic.md`](010_tui_anthropic.md).
