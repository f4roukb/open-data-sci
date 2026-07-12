# Competitive Data Science — Phase 1: Exploratory Data Analysis

**Planning**
- A productive EDA phase has an explicit question list (what one row represents, how train and test differ in distribution, where missingness sits, what natural groupings exist, what the target looks like across subgroups) rather than open-ended browsing
- Time-boxing prevents the common failure mode of polishing plots while the competition clock runs; the goal is enough understanding to design validation and a first feature set, not exhaustive characterisation
- The artefacts worth producing from EDA — a data dictionary, a list of suspect columns, a hypothesis list for feature engineering, a clear picture of train/test differences — feed directly into the next two phases

**Knowledge & Information**
- "What does one row represent?" is one of the most consequential questions to settle early; grain mismatches between datasets or between train and test are a frequent source of silent errors when joining or comparing
- Profiling shape, dtypes, missing-value rates, cardinality, descriptive statistics, and target distribution before modelling surfaces most data quirks worth knowing
- Inspecting train and test feature distributions side by side, including missingness patterns and category-level coverage, reveals distribution shift and features whose meaning differs across the split
- Adversarial validation — training a binary classifier to distinguish train from test — quantifies distribution shift; an AUC well above 0.5 means random CV will overestimate test performance, and the most predictive features in that classifier are the suspect ones
- For time-indexed data, a chronological view reveals gaps, seasonality, trend breaks, and regime changes before they distort downstream work
- Duplicate rows, near-constant columns, suspect nulls, and outliers can distort aggregations and model training in ways that are difficult to trace later
- The target's marginal distribution (class balance, skew, heavy tails, zero inflation) informs both loss choice and metric interpretation; rare positives in particular shape sampling and threshold strategies

**Tricks**
- Plotting target by every feature (binned for numeric, level-by-level for categorical) is a cheap, high-information scan for non-linearity, monotonicity, and useful interactions
- Computing target statistics across categorical levels directly identifies the strongest column-level predictors and seeds the first round of target-encoded features
- Inspecting the most and least frequent values per column quickly surfaces encoded missingness sentinels, unit or currency changes, and high-cardinality leakage indicators
- A short "data dictionary" file summarising column meaning, grain, and observed quirks pays for itself many times over when feature engineering ramps up
- Visualising row-level NaN patterns (e.g. as a sorted boolean matrix) often reveals structural missingness tied to entity type, time period, or recording source

## Part of

- `competitive_data_science` — the phased competition playbook this phase belongs to; load it for the full campaign overview, phase sequencing, and the cross-phase loop-back wiring.
