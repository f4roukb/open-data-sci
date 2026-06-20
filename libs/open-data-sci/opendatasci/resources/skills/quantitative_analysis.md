# Quantitative Analysis Skill

**Problem Formulation**
- Translating a real-world question into a precise mathematical statement is the first and often most consequential step; ambiguity in the objective propagates into every downstream modelling choice
- Distinguishing between estimation problems (what is the value of some unknown quantity?), prediction problems (what will happen?), and optimisation problems (what should we do?) shapes the entire analytical approach
- Constraints and feasibility requirements are as important as the objective; an optimal solution that violates real-world constraints is not a solution

**Mathematical & Statistical Foundations**
- Results derived from first principles are more defensible than results produced by black-box procedures; being able to trace a conclusion back to its assumptions is essential for knowing when to trust it
- Linearity assumptions are convenient but often wrong; understanding where a linear approximation holds and where it breaks down is a core skill
- Probability distributions carry specific assumptions about the data-generating process; selecting a distribution because it fits historical data is not the same as selecting one because it reflects the underlying mechanism
- Stationarity assumptions underlie much of classical time series analysis; checking for non-stationarity (unit roots, structural breaks, regime changes) before applying methods that assume it prevents misleading inference
- Heavy tails and extreme events are often the quantities that matter most in risk-sensitive applications; thin-tail distributional assumptions systematically underestimate tail risk

**Time Series & Signal Analysis**
- Autocorrelation structure (ACF/PACF plots, Ljung-Box tests) should inform model choice before fitting; ignoring it leads to invalid standard errors and spurious relationships
- Spurious regression between integrated time series is a well-documented failure mode; cointegration analysis is the appropriate tool for modelling long-run relationships between non-stationary series
- Seasonality, trend, and irregular components are often better handled explicitly than absorbed into a single model; decomposition clarifies what each component contributes
- Volatility clustering — the empirical regularity that large moves tend to follow large moves — is a persistent feature of financial and economic time series that standard models ignore; GARCH-family models are the standard treatment
- Choosing the forecast horizon deliberately matters: the right model for one-step-ahead forecasting is often not the right model for long-horizon forecasting

**Risk & Uncertainty Quantification**
- Point estimates without uncertainty bounds are incomplete; the width of the confidence or credible interval is often more decision-relevant than the point itself
- Scenario analysis and stress testing complement statistical risk measures by exploring tail outcomes that may not be well-represented in historical data
- Model risk — the risk that the model itself is wrong — is a distinct and often underappreciated source of uncertainty; comparing results across plausible alternative models is a useful guard
- Tail risk measures (VaR, CVaR/Expected Shortfall) answer different questions: VaR describes a threshold, CVaR describes what to expect when that threshold is breached — for many purposes CVaR is the more informative measure
- Monte Carlo simulation provides a flexible framework for propagating uncertainty through complex models; the quality of the output depends entirely on the quality of the input distribution assumptions

**Optimisation**
- Many quantitative problems can be cast as optimisation; recognising the structure (convex vs. non-convex, constrained vs. unconstrained, continuous vs. integer) determines what solvers are applicable and what guarantees are available
- Convex problems have the substantial advantage that local optima are global optima; non-convex problems may require heuristics, multiple starting points, or relaxations
- Numerical stability matters: poorly conditioned problems can produce results that look precise but are sensitive to small perturbations in inputs or solver tolerances
- In practice, regularisation in optimisation and regularisation in statistics are the same idea expressed in different languages — both bias a solution toward simpler structure in exchange for reduced variance

**Backtesting & Empirical Validation**
- Backtesting on historical data is a necessary but insufficient validation; without controls for look-ahead bias, survivorship bias, and overfitting, backtest results are unreliable guides to out-of-sample performance
- Walk-forward validation — fitting on a rolling training window and evaluating on a subsequent out-of-sample period — better mimics the real operational setting than a single historical simulation
- Multiple testing inflates apparent strategy performance; the more configurations that are tried on the same historical period, the more likely the best-performing one succeeds by chance
- Transaction costs, slippage, and capacity constraints routinely close the gap between theoretical and realised performance; a backtest that ignores them is optimistic by construction

**Communicating Quantitative Results**
- Numerical precision in outputs should match the precision of the inputs; reporting eight decimal places from a model estimated on noisy data implies false certainty
- Assumptions deserve explicit documentation: a result is only as valid as the assumptions that support it, and readers need to be able to evaluate those assumptions for themselves
- Sensitivity analysis — showing how conclusions change as key inputs or assumptions vary — is often more valuable than a single point result, particularly when inputs are uncertain or contested
- The practical significance of a quantitative result (does the difference matter for the decision at hand?) is distinct from its statistical significance and should always be addressed