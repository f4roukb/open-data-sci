# Machine Learning — Problem Framing

- Before writing any code, the prediction task is worth thinking through carefully: what exactly is being predicted, at what point in time, using what information, and with what tolerance for different kinds of errors
- Label quality often determines the ceiling on model performance more than architecture does — severe imbalance, label noise, and ambiguous labelling criteria are worth surfacing early
- The data modality (tabular, text, image, time series, graph) and available volume should inform the range of approaches worth considering
- Leakage — features computed using information that wouldn't be available at prediction time — is the most common source of over-optimistic evaluation results; it's worth tracing feature provenance carefully before trusting any metrics

## Part of

- `machine_learning` — the machine learning skill domain this belongs to; load it for the full map of skills and when to reach for each.
