# Quantitative Analysis Skill Domain

A playbook for quantitative and financial/statistical analysis: formulating the problem precisely, grounding it in mathematical and statistical foundations, handling time series and signals, quantifying risk and uncertainty, casting problems as optimisation, validating empirically, and communicating results with appropriate precision. Load the skill that matches the stage of the analysis.

## Problem Formulation

### Metadata
- skill_domain_name: quantitative_analysis
- skill_name: problem_formulation

### Description
Translating a real-world question into a precise mathematical statement, distinguishing estimation, prediction, and optimisation problems, and treating constraints as part of the objective; load before any modelling begins.

## Mathematical & Statistical Foundations

### Metadata
- skill_domain_name: quantitative_analysis
- skill_name: mathematical_statistical_foundations

### Description
Deriving results from first principles, knowing where linearity and distributional assumptions hold or break, checking stationarity, and accounting for heavy tails; load when choosing or justifying the mathematical machinery behind an analysis.

## Time Series & Signal Analysis

### Metadata
- skill_domain_name: quantitative_analysis
- skill_name: time_series_signal_analysis

### Description
Reading autocorrelation structure, avoiding spurious regression via cointegration, decomposing seasonality and trend, modelling volatility clustering, and choosing the forecast horizon deliberately; load when working with time-indexed or signal data.

## Risk & Uncertainty Quantification

### Metadata
- skill_domain_name: quantitative_analysis
- skill_name: risk_uncertainty_quantification

### Description
Reporting uncertainty bounds, running scenario analysis and stress tests, accounting for model risk, choosing between VaR and CVaR, and propagating uncertainty via Monte Carlo simulation; load when a result needs to be expressed with its risk or uncertainty.

## Optimisation

### Metadata
- skill_domain_name: quantitative_analysis
- skill_name: optimisation

### Description
Recognising problem structure (convex/non-convex, constrained/unconstrained, continuous/integer), the guarantees convexity provides, numerical stability, and the shared logic of regularisation across optimisation and statistics; load when a problem is cast or solved as an optimisation.

## Backtesting & Empirical Validation

### Metadata
- skill_domain_name: quantitative_analysis
- skill_name: backtesting_empirical_validation

### Description
Guarding against look-ahead and survivorship bias, using walk-forward validation, correcting for multiple testing, and accounting for transaction costs and capacity constraints; load when validating a strategy or model against historical data.

## Communicating Quantitative Results

### Metadata
- skill_domain_name: quantitative_analysis
- skill_name: communicating_results

### Description
Matching numerical precision to input precision, documenting assumptions explicitly, running sensitivity analysis, and distinguishing statistical from practical significance; load when writing up or presenting quantitative results.
