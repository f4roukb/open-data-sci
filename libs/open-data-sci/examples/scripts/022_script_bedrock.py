"""022 — Batch analysis script with AWS Bedrock.

Identical workflow to 020_script_anthropic.py but driven by AWS Bedrock instead
of the Anthropic API. Credentials are resolved through the standard boto3 chain
(environment variables, ~/.aws/credentials, or an IAM role).

Prerequisites:
    pip install "open-data-sci[aws]"

    # Enable model access in the AWS Bedrock console, then authenticate:
    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=...
    export REGION=us-east-1   # or whichever region has model access

Usage:
    python 022_script_bedrock.py

The script generates three synthetic monthly sales CSVs, processes each with
OpenDataSci, and writes reports to ./reports/.
"""


import asyncio
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from opendatasci import OpenDataSciConfig, create_agent

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ── Synthetic data ─────────────────────────────────────────────────────────────

def _write_sales_csv(path: Path, month: str, rng: np.random.Generator) -> None:
    """Write a plausible monthly sales CSV to *path*."""
    n = int(rng.integers(300, 600))
    df = pd.DataFrame({
        "date":     pd.date_range(f"2025-{month}-01", periods=n, freq="h").strftime("%Y-%m-%d"),
        "product":  rng.choice(["Widget A", "Widget B", "Gadget Pro", "Service Plan"], n),
        "region":   rng.choice(["EMEA", "APAC", "AMER"], n, p=[0.4, 0.35, 0.25]),
        "revenue":  rng.uniform(50, 5_000, n).round(2),
        "units":    rng.integers(1, 50, n),
        "returned": rng.choice([0, 1], n, p=[0.9, 0.1]),
    })
    path.write_text(df.to_csv(index=False))


# ── Analysis ───────────────────────────────────────────────────────────────────

_PROMPT = """\
Analyse this monthly sales dataset and write a concise report with:
1. Total revenue and units sold, broken down by product and by region.
2. Return rate and its estimated revenue impact.
3. The top-performing product-region combination.
4. One concrete recommendation for next month, grounded in the data.
Keep the report under 300 words.
"""


async def analyse(csv_path: Path, config: OpenDataSciConfig, output_dir: Path) -> None:
    log.info("→ %s", csv_path.name)

    report_path = output_dir / csv_path.with_suffix(".report.txt").name

    final = ""
    async with create_agent(str(csv_path), config=config) as agent:
        async for event in agent.astream(_PROMPT):
            if event.type == "token":
                print(event.content, end="", flush=True)
            elif event.type == "response":
                final = event.content
                print()
            elif event.type == "error":
                log.error("Agent error for %s: %s", csv_path.name, event.content)
                return

    if final:
        header = f"# {csv_path.stem}  —  generated {datetime.now():%Y-%m-%d %H:%M}\n\n"
        report_path.write_text(header + final)
        log.info("   report  → %s", report_path)


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    data_dir   = Path("data")
    output_dir = Path("reports")
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)
    for month in ("01", "02", "03"):
        p = data_dir / f"sales_2025_{month}.csv"
        if not p.exists():
            _write_sales_csv(p, month, rng)
            log.info("Created %s", p)

    # aws_region falls back to the REGION env var, then "us-east-1".
    # The cross-region inference model ID prefix (us., eu., ap.) must match
    # the region you are authenticated against.
    config = OpenDataSciConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-4-6",
        temperature=0.1,
    )

    csv_files = sorted(data_dir.glob("*.csv"))
    log.info("Analysing %d file(s) via Bedrock (%s)", len(csv_files), config.model)

    for path in csv_files:
        await analyse(path, config, output_dir)

    log.info("Done. Reports in %s/", output_dir)


if __name__ == "__main__":
    asyncio.run(main())
