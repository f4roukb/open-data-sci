# 012 — Exploring data from the TUI with AWS Bedrock

AWS Bedrock is a managed inference service that lets you run frontier models
(Anthropic Claude, Meta Llama, Mistral, and others) without managing API keys
separately — access is controlled through AWS IAM, so it integrates naturally
into existing enterprise AWS infrastructure.

## When to choose Bedrock

- Your organisation already runs on AWS and prefers consolidated billing and IAM
- Compliance requirements mandate AWS-region data residency
- You need cross-region inference with automatic failover
- Your team manages access via IAM roles rather than shared API keys

For direct Anthropic API access, see [`010_tui_anthropic.md`](010_tui_anthropic.md).

---

## Setup

### 1 — Enable model access in the AWS console

Bedrock requires you to explicitly opt in to each model family.
In the AWS console: **Bedrock → Model access → Manage model access** → enable
the Anthropic Claude models (or whichever family you plan to use) → **Save changes**.

This step is per-account and per-region. Access is typically granted within seconds.

### 2 — Install the provider extra

```bash
pip install "open-data-sci[aws]"
```

### 3 — Configure AWS credentials

Bedrock uses the standard boto3 credential chain. Any of the following works:

**Environment variables (quickest for local use):**

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...    # only for temporary STS credentials
```

**AWS credentials file (`~/.aws/credentials`):**

```ini
[default]
aws_access_key_id     = AKIA...
aws_secret_access_key = ...
```

**IAM role (EC2, ECS, Lambda, SageMaker):**
No credentials needed — they are fetched automatically from the instance
metadata service.

**AWS Vault / SSO:**

```bash
aws-vault exec my-profile -- opendatasci sales.csv --provider bedrock
# or
aws sso login && opendatasci sales.csv --provider bedrock
```

---

## Launching

```bash
# Default region (us-east-1) and default model
REGION=us-east-1 opendatasci sales.csv --provider bedrock

# Explicit region and model
REGION=us-west-2 opendatasci sales.csv --provider bedrock \
  --model us.anthropic.claude-sonnet-4-6

# Load config from file (region set inside the file)
opendatasci sales.csv --config examples/config_bedrock.yaml
```

### Cross-region inference model IDs

Bedrock cross-region inference prefixes the model ID with the inference profile
region (`us.`, `eu.`, `ap.`). Always use the prefixed form:

| Profile | Model ID |
|---------|----------|
| US | `us.anthropic.claude-sonnet-4-6` |
| US | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| EU | `eu.anthropic.claude-sonnet-4-6` |
| AP | `ap.anthropic.claude-sonnet-4-6` |

Pass `--list-providers` to print the full default model table at any time.

---

## A realistic session

The TUI experience is identical to [`010_tui_anthropic.md`](010_tui_anthropic.md);
only the setup and launch command differ.

### Turn 1 — understand the data

```
> What does this dataset contain? Give me the column types, row count, and flag
  anything that looks off — nulls, suspicious values, heavy skew.
```

### Turn 2 — dig into a trend

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
| `/clear` | Clear conversation history, keep sandbox variables |
| `/ls-workspace` | List every file in the workspace |
| `/models` | Show primary and secondary model in use |
| `/stop` | Interrupt a running agent turn |
| `/exit` | Quit |

---

## Tips

**Region selection:** Use `REGION=us-east-1` for the widest model availability.
Not all models are available in every region — the Bedrock console shows coverage.

**IAM permissions:** The calling principal needs the `bedrock:InvokeModel` and
`bedrock:InvokeModelWithResponseStream` actions on the models you plan to use.
A minimal IAM policy:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.*"
}
```

**Keyboard shortcuts:** identical to [`010_tui_anthropic.md`](010_tui_anthropic.md).
