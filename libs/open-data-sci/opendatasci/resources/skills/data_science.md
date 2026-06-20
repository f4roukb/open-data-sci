# Data Science Skill

**Framing the Problem**
- Vague requests benefit from being grounded in a concrete hypothesis or target metric before touching any data — the question shapes every downstream choice
- Understanding what a "good" answer looks like up front (decision to be made, threshold for action, audience) prevents wasted analysis
- Distinguishing exploratory work (generating hypotheses) from confirmatory work (testing them) matters — mixing the two silently inflates false discovery rates

**Exploratory Analysis**
- Data rarely arrives in the shape expected; profiling shape, dtypes, missing value rates, cardinality, and basic descriptive stats early tends to surface surprises before they become silent errors
- Duplicate rows, near-constant columns, and unexpected nulls can distort aggregations and models in ways that are hard to trace later
- For numeric distributions, tools like histograms and box plots reveal skew, modality, and outlier structure; for categoricals, frequency distributions expose long tails and rare levels worth knowing about before modelling
- Correlation analysis (linear and monotonic) helps map feature relationships; high pairwise correlation can matter for model interpretability even when it doesn't hurt predictive accuracy
- Time-indexed data rewards a chronological view before any aggregation — gaps, seasonality, trend breaks, and data collection artefacts tend to show up immediately and change how the data should be handled

**Data Quality & Preparation**
- Understanding *why* data is missing (structurally absent, randomly missing, or missing in a way correlated with the outcome) shapes the right response — imputation, exclusion, or flagging as a separate signal
- Outliers deserve investigation before any treatment; distinguishing measurement error from genuinely extreme values is consequential — removing real extremes can mask the most interesting signal
- Joining datasets is a common source of silent row inflation or key loss; checking counts and cardinality before and after a join is a lightweight step that often catches real problems
- Encoding choices interact with the model: ordered features carry ordinal meaning, nominal features don't — treating them the same can introduce spurious relationships
- Scaling matters for methods sensitive to feature magnitude and is irrelevant for others; knowing which is which avoids unnecessary transformation

**Causality & Confounding**
- Correlation between two variables rarely tells you which causes which, or whether a third variable drives both — most real-world datasets are observational and can't establish causation without additional assumptions or experimental design
- Confounders — variables that influence both the feature and the outcome — can make a spurious relationship look real or mask a genuine one; identifying and controlling for them is central to any analysis aimed at understanding what to do, not just what happened
- Selection bias and survivorship bias are pervasive: the data available is often not a random sample of the population of interest (e.g., only active customers, only completed transactions, only surviving products); the conclusions drawn are only as valid as that sample
- Simpson's paradox is surprisingly common: a trend visible in aggregate can reverse when broken down by a subgroup — always worth checking whether aggregate results hold across meaningful partitions before drawing conclusions

**Granularity & Aggregation**
- "What does one row represent?" is one of the most important questions to establish early — grain mismatches between datasets are a frequent source of silent errors when joining or comparing
- Aggregation choices (sum vs. mean vs. median, weekly vs. monthly, per-user vs. per-event) embed analytical decisions that change what the numbers mean; making them explicit prevents misinterpretation
- Aggregating too early can destroy signal; aggregating at the wrong level can introduce it artificially

**Statistical Testing**
- Sample size and statistical power deserve attention before interpreting null results — a study that fails to detect an effect may simply be underpowered; estimating the sample size needed to detect a meaningful effect size is a useful sanity check
- The right test depends on data structure, distribution, and the independence assumptions that can actually be defended — parametric tests have assumptions worth checking before applying
- Non-parametric alternatives trade statistical power for fewer assumptions; the right tradeoff depends on sample size and how badly assumptions are violated
- Testing many hypotheses simultaneously inflates false positives in ways that compound quickly; multiple comparison correction adjusts for this but also reduces sensitivity — the right correction depends on whether you're guarding against any false positive or controlling the proportion of false discoveries
- P-values and effect sizes answer different questions: significance says whether an effect is detectable given sample size; effect size says whether it's large enough to matter — both are needed to judge a result
- Confidence intervals communicate uncertainty more directly than p-values alone and translate more naturally into business language

**Modeling & Evaluation**
- A naive baseline (mean/mode predictor, last-value carry-forward, a simple rule) makes the value of a model concrete — sometimes the baseline is surprisingly hard to beat, which is itself informative
- Model choice should be driven by the problem structure, not by a default preference for a particular algorithm: consider the full spectrum from linear models (interpretable, well-regularised) through tree-based ensembles (robust to feature scale, capture interactions) to neural networks (high capacity, need volume and tuning) — the right family depends on signal strength, data volume, interpretability needs, and the nature of the decision boundary or regression surface
- Metric choice should reflect the real objective and data distribution; accuracy misleads on imbalanced problems, RMSE penalises large errors disproportionately, percentage-based errors behave badly near zero — each metric embeds assumptions worth making explicit
- Split strategy encodes assumptions about how the model will be used: stratified splits for class balance, time-ordered splits when temporal structure exists, group-aware splits when rows share an entity; the wrong strategy produces optimistic numbers that don't hold
- Evaluating on a single held-out set can be noisy; cross-validation spread gives a better picture of how stable performance is across different data slices
- Slicing metrics by relevant subgroups or prediction ranges often reveals where a model underperforms in ways aggregate numbers conceal
- When combining multiple models, diversity of predictions is what drives ensemble gains — architecturally similar models (e.g., XGBoost, LightGBM, and CatBoost) are likely to produce highly correlated outputs and make the same mistakes, so blending them yields little improvement; meaningful gains come from ensembling models from different families (tree-based, linear, neural) or models trained on different feature sets or subsets of the data

**Communicating Findings**
- Estimates with uncertainty (intervals, spread across folds) are more useful than point values — they convey how stable a result is and what confidence is warranted
- Leading with the key finding in plain language, then following with technical detail, tends to serve mixed audiences better than leading with methodology
- Caveats, assumptions, and plausible alternative explanations are part of a rigorous analysis, not afterthoughts — surfacing them early builds rather than undermines credibility
- If the data isn't sufficient for a strong conclusion, saying so clearly — and describing what additional data, sample size, or experimental design would change that — is itself a useful output
""".strip()