# finance.yahoo.com — Pulling Data with `yfinance`

- `yfinance` (the `[finance]` extra; check `list_python_libs` before importing — it's optional, not always installed) is the interface to use, not `finance.yahoo.com`'s pages directly — it wraps Yahoo Finance's unofficial, undocumented API so callers don't have to
- `yf.Ticker("AAPL").history(period="1y")` (or `start=`/`end=` for an exact range) gets one symbol's price history; prices are split/dividend-adjusted by default (`auto_adjust=True`), which is almost always what's wanted for return calculations — set it `False` explicitly if raw unadjusted closes are actually needed
- `yf.download(["AAPL", "MSFT", "GOOG"], period="1y")` fetches multiple tickers in one batched call — prefer this over looping individual `Ticker.history()` calls, which is slower and more likely to hit rate limits
- `Ticker.info` (a dict of company/market metadata) is inconsistent in coverage across tickers and drifts over time — treat any given key's presence as best-effort, and check for its existence rather than assuming every ticker has the same fields populated
- `Ticker.financials` / `.balance_sheet` / `.cashflow` (and their `.quarterly_*` counterparts) give fundamental statements; `Ticker.dividends` / `.splits` give corporate-action history — these are the right calls for fundamentals rather than trying to parse them out of `.info`
- Because this is an unofficial API, expect occasional empty results, missing fields, or throttling under heavy use — validate that a returned DataFrame isn't empty before proceeding, and avoid tight request loops without spacing them out
- A wrong or delisted ticker symbol often fails silently (empty DataFrame) rather than raising — confirm a guessed symbol resolves to the intended company (e.g. via `Ticker.info.get("longName")`) before trusting downstream results built on it

## Metadata

- parent domain: finance.yahoo.com
