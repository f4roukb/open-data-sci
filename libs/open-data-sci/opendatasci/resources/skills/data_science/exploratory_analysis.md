# Data Science — Exploratory Analysis

- Data rarely arrives in the shape expected; profiling shape, dtypes, missing value rates, cardinality, and basic descriptive stats early tends to surface surprises before they become silent errors
- Duplicate rows, near-constant columns, and unexpected nulls can distort aggregations and models in ways that are hard to trace later
- For numeric distributions, tools like histograms and box plots reveal skew, modality, and outlier structure; for categoricals, frequency distributions expose long tails and rare levels worth knowing about before modelling
- Correlation analysis (linear and monotonic) helps map feature relationships; high pairwise correlation can matter for model interpretability even when it doesn't hurt predictive accuracy
- Time-indexed data rewards a chronological view before any aggregation — gaps, seasonality, trend breaks, and data collection artefacts tend to show up immediately and change how the data should be handled

## Metadata

- parent domain: data_science
