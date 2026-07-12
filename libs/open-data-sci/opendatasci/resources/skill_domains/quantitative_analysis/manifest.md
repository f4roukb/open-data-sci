# Quantitative Analysis Skill Domain

A playbook for quantitative and financial/statistical analysis: formulating the problem precisely, grounding it in mathematical and statistical foundations, handling time series and signals, quantifying risk and uncertainty, casting problems as optimisation, validating empirically, and communicating results with appropriate precision. Load the skill that matches the stage of the analysis.

## Problem Formulation

- skill: quantitative_analysis::problem_formulation

Translating a real-world question into a precise mathematical statement, distinguishing estimation, prediction, and optimisation problems, and treating constraints as part of the objective; load before any modelling begins.

## Mathematical & Statistical Foundations

- skill: quantitative_analysis::mathematical_statistical_foundations

Deriving results from first principles, knowing where linearity and distributional assumptions hold or break, checking stationarity, and accounting for heavy tails; load when choosing or justifying the mathematical machinery behind an analysis.

## Time Series & Signal Analysis

- skill: quantitative_analysis::time_series_signal_analysis

Reading autocorrelation structure, avoiding spurious regression via cointegration, decomposing seasonality and trend, modelling volatility clustering, and choosing the forecast horizon deliberately; load when working with time-indexed or signal data.

## Risk & Uncertainty Quantification

- skill: quantitative_analysis::risk_uncertainty_quantification

Reporting uncertainty bounds, running scenario analysis and stress tests, accounting for model risk, choosing between VaR and CVaR, and propagating uncertainty via Monte Carlo simulation; load when a result needs to be expressed with its risk or uncertainty.

## Optimisation

- skill: quantitative_analysis::optimisation

Recognising problem structure (convex/non-convex, constrained/unconstrained, continuous/integer), the guarantees convexity provides, numerical stability, and the shared logic of regularisation across optimisation and statistics; load when a problem is cast or solved as an optimisation.

## Backtesting & Empirical Validation

- skill: quantitative_analysis::backtesting_empirical_validation

Guarding against look-ahead and survivorship bias, using walk-forward validation, correcting for multiple testing, and accounting for transaction costs and capacity constraints; load when validating a strategy or model against historical data.

## Communicating Quantitative Results

- skill: quantitative_analysis::communicating_results

Matching numerical precision to input precision, documenting assumptions explicitly, running sensitivity analysis, and distinguishing statistical from practical significance; load when writing up or presenting quantitative results.
