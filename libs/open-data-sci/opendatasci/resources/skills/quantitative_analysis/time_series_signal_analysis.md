# Quantitative Analysis — Time Series & Signal Analysis

- Autocorrelation structure (ACF/PACF plots, Ljung-Box tests) should inform model choice before fitting; ignoring it leads to invalid standard errors and spurious relationships
- Spurious regression between integrated time series is a well-documented failure mode; cointegration analysis is the appropriate tool for modelling long-run relationships between non-stationary series
- Seasonality, trend, and irregular components are often better handled explicitly than absorbed into a single model; decomposition clarifies what each component contributes
- Volatility clustering — the empirical regularity that large moves tend to follow large moves — is a persistent feature of financial and economic time series that standard models ignore; GARCH-family models are the standard treatment
- Choosing the forecast horizon deliberately matters: the right model for one-step-ahead forecasting is often not the right model for long-horizon forecasting

## Part of

- `quantitative_analysis` — the quantitative analysis skill domain this belongs to; load it for the full map of skills and when to reach for each.
