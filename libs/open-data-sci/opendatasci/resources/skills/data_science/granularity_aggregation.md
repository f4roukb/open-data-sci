# Data Science — Granularity & Aggregation

- "What does one row represent?" is one of the most important questions to establish early — grain mismatches between datasets are a frequent source of silent errors when joining or comparing
- Aggregation choices (sum vs. mean vs. median, weekly vs. monthly, per-user vs. per-event) embed analytical decisions that change what the numbers mean; making them explicit prevents misinterpretation
- Aggregating too early can destroy signal; aggregating at the wrong level can introduce it artificially

## Metadata

- parent domain: data_science
