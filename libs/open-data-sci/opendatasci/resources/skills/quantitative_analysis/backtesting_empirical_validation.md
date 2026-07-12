# Quantitative Analysis — Backtesting & Empirical Validation

- Backtesting on historical data is a necessary but insufficient validation; without controls for look-ahead bias, survivorship bias, and overfitting, backtest results are unreliable guides to out-of-sample performance
- Walk-forward validation — fitting on a rolling training window and evaluating on a subsequent out-of-sample period — better mimics the real operational setting than a single historical simulation
- Multiple testing inflates apparent strategy performance; the more configurations that are tried on the same historical period, the more likely the best-performing one succeeds by chance
- Transaction costs, slippage, and capacity constraints routinely close the gap between theoretical and realised performance; a backtest that ignores them is optimistic by construction

## Metadata

- parent domain: quantitative_analysis
