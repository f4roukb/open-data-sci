# Data Science — Data Quality & Preparation

- Understanding *why* data is missing (structurally absent, randomly missing, or missing in a way correlated with the outcome) shapes the right response — imputation, exclusion, or flagging as a separate signal
- Outliers deserve investigation before any treatment; distinguishing measurement error from genuinely extreme values is consequential — removing real extremes can mask the most interesting signal
- Joining datasets is a common source of silent row inflation or key loss; checking counts and cardinality before and after a join is a lightweight step that often catches real problems
- Encoding choices interact with the model: ordered features carry ordinal meaning, nominal features don't — treating them the same can introduce spurious relationships
- Scaling matters for methods sensitive to feature magnitude and is irrelevant for others; knowing which is which avoids unnecessary transformation

## Part of

- `data_science` — the general data science skill domain this belongs to; load it for the full map of skills and when to reach for each.
