# finance.yahoo.com Skill Domain

Curated entry point into pulling market data via `finance.yahoo.com` — in practice this means the `yfinance` Python library, which wraps Yahoo Finance's unofficial API. Requires the `[finance]` extra (`pip install "open-data-sci[finance]"`). Load the skill here before writing any code that fetches prices or fundamentals for a ticker.

## yfinance Basics

- skill: finance.yahoo.com::yfinance_basics

The 20% of `yfinance` that covers 80% of tasks: fetching single/multi-ticker price history, pulling fundamentals and corporate actions, and the reliability caveats (inconsistent `.info` coverage, silent failures on bad symbols, unofficial-API throttling) that come with an unofficial API; load before pulling any Yahoo Finance data.
