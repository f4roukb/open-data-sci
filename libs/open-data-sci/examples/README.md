# Examples

Patterns for working with OpenDataSci across every supported LLM provider.

## Prerequisites

- Python 3.11+
- An API key or server running for the provider you choose

```bash
pip install open-data-sci
```

Provider extras (only needed for non-default providers):

```bash
pip install "open-data-sci[aws]"      # Bedrock
pip install "open-data-sci[gemini]"   # Gemini AI Studio
pip install "open-data-sci[gcp]"      # Vertex AI
pip install "open-data-sci[azure]"    # Azure OpenAI
pip install "open-data-sci[ollama]"   # Ollama
```

---

## TUI

Full-screen terminal UI — ask questions in plain English, no code required.

| File | Provider | Notes |
|------|----------|-------|
| [`tui/010_tui_anthropic.md`](tui/010_tui_anthropic.md) | Anthropic | Full walkthrough: session flow, slash commands, file attachments, keyboard shortcuts |
| [`tui/011_tui_openai_compatible_server.md`](tui/011_tui_openai_compatible_server.md) | OpenAI-compatible server | Self-hosted, no API key; covers vLLM server setup and local-model tips |
| [`tui/012_tui_bedrock.md`](tui/012_tui_bedrock.md) | AWS Bedrock | Managed inference via IAM; covers credentials, model access, and cross-region IDs |

---

## Batch scripts

Run the agent autonomously — no TUI, no human in the loop. Generates synthetic
sales CSVs, analyses each, and writes plain-text reports to `./reports/`.

| File | Provider |
|------|----------|
| [`scripts/020_script_anthropic.py`](scripts/020_script_anthropic.py) | Anthropic |
| [`scripts/021_script_openai_compatible_server.py`](scripts/021_script_openai_compatible_server.py) | OpenAI-compatible server |
| [`scripts/022_script_bedrock.py`](scripts/022_script_bedrock.py) | AWS Bedrock |

Key pattern (identical across all variants):

```python
async with create_agent(path, config=config) as agent:
    async for event in agent.astream(prompt):
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "response":
            final = event.content
```

---

## Jupyter notebooks

End-to-end supervised ML: profile a churn dataset, train classifiers, interpret
with SHAP. Uses `AsyncExitStack` to keep the agent alive across cells so each
cell is a follow-up turn in the same conversation.

| File | Provider |
|------|----------|
| [`notebooks/030_notebook_anthropic.ipynb`](notebooks/030_notebook_anthropic.ipynb) | Anthropic |
| [`notebooks/031_notebook_openai_compatible_server.ipynb`](notebooks/031_notebook_openai_compatible_server.ipynb) | OpenAI-compatible server |
| [`notebooks/032_notebook_bedrock.ipynb`](notebooks/032_notebook_bedrock.ipynb) | AWS Bedrock |

---

## Config files

Annotated `OpenDataSciConfig` YAML files — pass to any example with `--config`.
TUI flags override values set here.

| File | Provider | Auth |
|------|----------|------|
| [`configs/config_anthropic.yaml`](configs/config_anthropic.yaml) | Anthropic | `ANTHROPIC_API_KEY` |
| [`configs/config_openai.yaml`](configs/config_openai.yaml) | OpenAI | `OPENAI_API_KEY` |
| [`configs/config_bedrock.yaml`](configs/config_bedrock.yaml) | AWS Bedrock | boto3 credential chain |
| [`configs/config_gemini.yaml`](configs/config_gemini.yaml) | Google Gemini (AI Studio) | `GOOGLE_API_KEY` |
| [`configs/config_vertexai.yaml`](configs/config_vertexai.yaml) | Google Vertex AI | Application Default Credentials |
| [`configs/config_azure.yaml`](configs/config_azure.yaml) | Azure OpenAI | `AZURE_OPENAI_API_KEY` or service principal |
| [`configs/config_ollama.yaml`](configs/config_ollama.yaml) | Ollama | *(none — local server)* |
| [`configs/config_openai_compatible_server.yaml`](configs/config_openai_compatible_server.yaml) | OpenAI-compatible server | *(none — local server)* |

Usage:

```bash
opendatasci data.csv --config examples/configs/config_gemini.yaml
opendatasci data.csv --config examples/configs/config_vertexai.yaml --model gemini-2.5-pro
```

See the [full provider list and default models](../README.md#models) for all
supported options.
