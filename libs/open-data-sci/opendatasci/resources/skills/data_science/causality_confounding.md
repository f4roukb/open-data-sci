# Data Science — Causality & Confounding

- Correlation between two variables rarely tells you which causes which, or whether a third variable drives both — most real-world datasets are observational and can't establish causation without additional assumptions or experimental design
- Confounders — variables that influence both the feature and the outcome — can make a spurious relationship look real or mask a genuine one; identifying and controlling for them is central to any analysis aimed at understanding what to do, not just what happened
- Selection bias and survivorship bias are pervasive: the data available is often not a random sample of the population of interest (e.g., only active customers, only completed transactions, only surviving products); the conclusions drawn are only as valid as that sample
- Simpson's paradox is surprisingly common: a trend visible in aggregate can reverse when broken down by a subgroup — always worth checking whether aggregate results hold across meaningful partitions before drawing conclusions

## Metadata

- parent domain: data_science
