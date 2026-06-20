# Competitive Data Science Skill

A phased playbook for high-stakes data science competitions, built around incremental delivery. Each phase opens with planning so direction is set before work begins, and the high-leverage phases close with a review so signals from the current phase loop back into earlier decisions when warranted. The wiring at the end of the document spells out the most common loop-backs explicitly.

The phases describe what tends to matter at each stage, the knowledge that informs the choices, and the tricks that experienced competitors rely on. They do not prescribe a single path — flexibility to adapt the order, skip a step, or revisit a phase is part of the playbook.

## Phase 0 — Reconnaissance & Harness

**Planning**
- The shape of the campaign — how many days for baseline, EDA, feature work, model development, ensembling, final-week consolidation — is set in this phase; without an explicit cadence, single phases tend to absorb disproportionate time
- The deliverable from this phase is a working end-to-end pipeline (load → split → train → predict → submission file) and an experiment log; later phases iterate on individual steps without rebuilding the harness
- Compute and time budgets influence model choice as much as data does; framing them up front avoids late discovery that the chosen architecture is untrainable in the available window
- Teaming, when permitted, multiplies compute, diversity of ideas, and the volume of experiments that can be run in parallel — many winning solutions are ensemble products of multiple collaborators' independent pipelines; the decision is best made early because team dynamics and shared infrastructure benefit from time to develop, while solo competition remains viable and sometimes preferable when fast iteration matters more than diversity and coordination overhead would slow decisions down

**Knowledge & Information**
- The scoring metric is part of the modelling problem rather than its preamble; non-standard metrics (MAP@K, RMSLE, weighted log-loss, F-beta, Quadratic Weighted Kappa, AUC-PR variants) typically reward bespoke loss surrogates, custom objectives, or post-processing rather than naive optimisation of a generic objective
- Sponsoring organisations and prior editions of the competition often reveal which signal sources the data was constructed to expose and which were intentionally redacted; reading past winners' writeups on the same platform compresses weeks of independent exploration into hours of reading
- Public discussion forums concentrate the highest-density information in the first days of a competition — data quirks, evaluation edge cases, label issues, leaks — and disproportionately benefit those who read them first
- Competitions with similar data modality or problem type on the same hosting platform frequently share winning patterns (specific feature families, model choices, post-processing tricks) that transfer with light adaptation
- The submission format (column order, required precision, header expectations, prediction range) is part of the contract; mismatches cost ranked submissions
- If the dataset derives from a public source, the original documentation, schema descriptions, and domain context often resolve ambiguities the competition description leaves open

**Tricks**
- A "v0" submission that returns a constant (target mean, modal class, sample submission unchanged) validates the I/O pipeline end-to-end before any model exists and locks in the public LB anchor for later comparisons
- A versioned experiment log capturing CV score, public LB score, brief change description, feature set, and model class is the difference between knowing what worked and guessing — its value compounds with every submission
- Pre-committing to file naming conventions for predictions, OOF arrays, and submission files (e.g. `oof_<model>_<seed>.npy`, `sub_<model>_<date>.csv`) removes friction when many experiments coexist
- Fixing random seeds for every stochastic component from the start ensures later comparisons reflect genuine improvements rather than variance

## Phase 1 — Exploratory Data Analysis

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

## Phase 2 — Validation Strategy

**Planning**
- The validation scheme is the single most leverage-laden decision in a competition and is best designed before serious modelling begins; a strong local CV that correlates tightly with leaderboard score is more valuable than any single model improvement
- The plan starts from how the test set was constructed (time cutoff, geographic split, entity holdout, draw from a different distribution) and replicates that structure in CV
- Persistence of fold assignments across the campaign ensures every model's OOF predictions are directly comparable downstream

**Knowledge & Information**
- CV splits should mirror the test split: time-ordered for temporal holdouts, group-aware when rows share an identity (user, device, session, location), stratified for rare outcomes, nested combinations when several conditions apply
- The public leaderboard is a noisy signal on a small sample; private leaderboards routinely reshuffle relative to public — trusting a robust local CV over public-LB chasing is the default of strong competitors, though when the public sample is large, the CV-LB correlation has been demonstrably tight across submissions, or the test split is known to be drawn from the same distribution as train, LB carries genuine information worth weighing alongside CV
- The gap between local CV and public LB across submissions is itself a diagnostic: a consistent offset is acceptable, an inconsistent one signals that the CV scheme does not reflect the test distribution
- Repeated CV (multiple seeds, multiple shuffles) reduces noise in the validation estimate at the cost of compute and is worth running for final model selection rather than every iteration
- Out-of-fold (OOF) predictions are a free byproduct of CV that enables stacking, post-processing calibration, and error analysis — saving them by default removes friction later
- Adversarial validation results from Phase 1 directly inform CV design: when a feature separates train from test, validating on a fold drawn from the train distribution will not reflect how the model performs on the test distribution
- The public LB is computed on a small fraction of the test set (often 20–50%); its score variance is large enough that small public-LB movements between submissions often reflect noise rather than improvement

**Tricks**
- Constructing a fold structure that explicitly mimics observed train/test differences (e.g. using the most recent time slice as the validation fold when the test set is the future) is more reliable than relying on stratification alone
- A small "blend holdout" — a fold reserved for selecting ensemble weights and never used during base model training or hyperparameter search — preserves the integrity of ensembling decisions
- Persisting fold assignments to disk and re-using them across every model in the campaign keeps OOF predictions directly comparable and enables clean stacking later
- Sample-weighted CV, where validation weights reflect the test distribution (e.g. up-weighting recent observations under temporal shift), can produce CV scores that track LB more tightly than equal-weight CV

**Review** (close before moving on)
- Does the CV scheme replicate the test set construction? If not, redesign before any feature or model work proceeds
- Is the CV variance across folds small enough that meaningful improvements will be distinguishable from noise?
- Has at least one baseline submission anchored the CV-LB correspondence? If not, defer further work until it has
- Have OOF predictions and fold assignments been persisted so they can be reused throughout the campaign?

## Phase 3 — Baseline Model

**Planning**
- The point of the baseline is to compress the end-to-end pipeline into something runnable — data load, validation split, training, prediction, submission file — before optimising any single step
- A baseline submitted within the first one or two days establishes the floor and reveals pipeline bugs while they are still cheap to fix
- The baseline doubles as a measurement device: the value of every later improvement is expressed relative to this anchor

**Knowledge & Information**
- A naive baseline (target mean, mode, last-value carry-forward, simple rule keyed off the most predictive column) is the floor against which all subsequent complexity is measured — sometimes it is surprisingly hard to beat, which is itself a strong signal about the problem
- For tabular data, a gradient-boosting model with sensible defaults trained on raw features is the natural first real baseline; for text, TF-IDF with logistic regression or a small distilled transformer; for image, a pre-trained backbone with a linear head; for time series, a seasonal-naive or simple boosted-tree lag model
- Submitting the baseline locks in the CV-LB correspondence and provides the reference point for every later experiment
- The baseline's per-fold variance characterises noise floor — improvements smaller than this variance are unlikely to be real

**Tricks**
- Logging baseline metrics across all CV folds (mean, std, per-fold scores) characterises stability and informs how much variance later improvements need to overcome
- Storing OOF predictions and feature importances from the baseline produces an immediate map of which features matter and where the model is uncertain — both feed directly into Phase 4
- A baseline ablation (one feature removed at a time, scored against CV) is cheap and surfaces leakage candidates and dead features before serious feature work begins

## Phase 4 — Feature Engineering

**Planning**
- A feature plan starts from explicit hypotheses about signal sources (relational structure, temporal context, interactions, domain knowledge) rather than mechanical generation of every possible transformation
- Features cheap to compute and individually testable (each evaluated against the same CV scheme) make iteration fast and attribution clear
- A per-iteration feature budget — add N features, evaluate, prune — prevents accumulation of dead weight and keeps the feature set interpretable
- The feature engineering plan is the longest phase in most competitions; structuring it as a sequence of small, evaluable batches makes progress visible and prevents the search from going stale

**Knowledge & Information**
- Competition datasets frequently reward entity-level aggregations: group-by statistics (mean, std, min, max, count, nunique, skew, median) computed across users, sessions, locations, or time windows encode relational structure that row-level features miss
- Target encoding (mean of the target per category level, smoothed against the global mean) is consistently effective for high-cardinality categoricals but must be computed within each CV fold to avoid leakage
- Lag features, rolling statistics, expanding-window aggregations, exponentially weighted means, and time-since-event features form the core vocabulary for temporal datasets; their window sizes are hyperparameters worth searching
- Interaction features (products, ratios, differences, polynomial terms) between top-importance columns often outperform any single transformation; the cheap heuristic is to interact the top-K features from the baseline's importance ranking
- Frequency encoding (count of occurrences of each category level) is a near-free, leakage-safe alternative to one-hot for high-cardinality columns
- Cyclic encodings (sine/cosine of hour, day-of-week, month) preserve continuity at the boundaries that integer encoding does not natively express; the benefit is most pronounced for linear and neural models, while tree-based models can recover the modular structure through multiple splits on the integer-encoded feature when enough data is available
- For text, character-level and word-level n-grams, length statistics, sentiment scores, and pre-trained sentence embeddings each capture different aspects of the signal and ensemble well
- For image, traditional descriptors (HOG, colour histograms, LBP) and pre-trained backbone embeddings complement each other when compute is constrained
- Categorical features with cardinality in the thousands to millions typically respond better to target encoding, hashing, or learned embeddings than to one-hot; the right choice depends on the model family and the data volume
- Feature provenance matters: any feature computed using information unavailable at prediction time silently inflates CV and LB scores — tracing the construction of every feature against the data timeline is the only defence

**Tricks**
- Permutation importance and SHAP values on a trained baseline give a more reliable feature ranking than impurity-based importance, which biases toward high-cardinality features; impurity-based importance remains useful as a near-zero-cost first pass, particularly when many features need a directional ranking quickly or when permutation/SHAP would be prohibitively expensive
- Target encoding with K-fold nested inside the outer CV is the standard leak-safe pattern; failing to nest is the single most common source of silent CV-LB gaps in competition pipelines
- Forward feature selection by greedy CV gain is expensive but surfaces the truly load-bearing subset; backward elimination starting from the full feature set is faster and often sufficient
- A "control" feature — pure random noise added to the feature set — calibrates how much importance is attributable to chance; any real feature ranking below it is a candidate for pruning
- Re-running adversarial validation after adding new features detects features that encode the train/test split itself
- Storing the feature engineering as a pure transformation function (fit on train, applied to any split) prevents training/inference skew and makes leak-safety easier to audit

**Review** (close before moving on)
- Does the CV improvement from new features hold on a freshly seeded split? If not, suspect overfitting to fold structure — return to Phase 2 to evaluate CV variance and possibly redesign
- Has adversarial validation been re-run after the new feature set? If new features separate train from test more easily than before, those features encode the split — return to Phase 1 to inspect them and consider removal
- Are feature importances dominated by a single feature with implausibly high signal? Suspect leakage — trace the feature's construction against the data timeline before trusting any downstream metric
- Is the CV-LB gap consistent with the pre-feature-work baseline? A sudden divergence is a leak signal or a CV scheme problem — if persistent, return to Phase 2
- Has redundancy been pruned? Highly correlated features add noise without value and slow training; a brief correlation review at the end of the phase pays off in later iteration speed

## Phase 5 — Model Development

**Planning**
- The model plan covers a portfolio of model families to evaluate (linear, tree-based, neural) with a budget per family, rather than a single architecture to perfect
- Building several distinct base models — even when individually weaker than the strongest — pays compounding returns at the ensembling phase
- Iteration speed dominates progress in this phase; running on a representative stratified subsample first and full data once the architecture is settled cuts wall-clock cost substantially without losing directional signal

**Knowledge & Information**
- Model selection follows problem structure and data characteristics rather than a habitual preference for any one family; the choice is a design decision deserving the same rigour as any other modelling choice
- For tabular data, gradient-boosted decision trees are the most consistent top performer across small-to-medium datasets; differences across implementations in handling of categoricals, missing values, and split criteria occasionally swing one toward better performance on a given dataset, and at very large scale or in domains with rich high-cardinality interactions (CTR prediction, recommendation, certain industrial datasets) neural approaches can match or surpass them
- Linear and logistic regression with engineered features can be primary contenders in low-data regimes, when interactions are well-captured by hand-crafted features, or under strict interpretability constraints; they also serve as complementary ensemble components, as a sanity check on whether non-linear models add value, and as the standard meta-learner in stacking
- Neural architectures on tabular data (MLP, TabNet, FT-Transformer, NODE) are competitive at scale and increasingly close the gap to boosted trees on small-to-medium datasets when learned categorical embeddings or cross-feature interactions carry signal; their primary value in many competitions is diversity contribution to the ensemble, but in data regimes where they match or beat boosted trees they belong as a primary candidate rather than only as an ensemble component
- For text, fine-tuned transformer checkpoints (BERT-family, DeBERTa, RoBERTa, ELECTRA, distilled variants) lift performance substantially over feature-based baselines at meaningful compute cost on most natural-language tasks; on very short, highly structured, or heavily label-noisy text, TF-IDF or hashed n-grams with a linear classifier can match or outperform transformers at a fraction of the cost
- For image, pre-trained backbones (EfficientNet, ConvNeXt, ViT, Swin) via transfer learning are the standard entry point; augmentation design, head architecture, and training schedule often move the score more than swapping backbones of similar capacity, while in tasks where the backbone's inductive bias aligns particularly well with the data (fine-grained classification, medical imaging, satellite imagery, dense prediction) the backbone choice itself can be the dominant factor
- For time series with many parallel series, gradient boosting on lag features competes with and often beats dedicated forecasting architectures (LSTM, Temporal Fusion Transformer); the dedicated architectures win when complex exogenous structure or long-range dependencies dominate
- Test-time augmentation (TTA) — averaging predictions over augmented copies of each test instance — produces small but consistent gains in image tasks and sometimes in text and tabular
- Models train on the train split and select hyperparameters on the validation fold; the test set is never seen by any model selection process

**Tricks**
- Training the same model with several random seeds and averaging predictions is the cheapest, most reliable way to reduce variance and improve score
- For boosted trees, early stopping against the validation fold within each CV split removes the n_estimators hyperparameter from the search and acts as the primary regulariser
- Pseudo-labelling — training on high-confidence test predictions, then re-training — can add meaningful gains when the test set is large relative to train, but risks amplifying mistakes if confidence calibration is poor
- Saving OOF and test predictions from every meaningful model run feeds Phase 7 directly; ensembling later without these arrays forces re-running expensive trainings
- Knowledge distillation (training a smaller or differently-architected student on the soft predictions of a strong teacher) produces useful diversity for ensembling when the student family is genuinely different from the teacher's
- For any neural training, starting with a low number of epochs (one or two), monitoring train and validation curves, and continuing only when results warrant it avoids wasted compute on unpromising architectures and surfaces data or pipeline issues early

## Phase 6 — Hyperparameter Tuning

**Planning**
- A tuning budget — number of trials, wall-clock cap — set before search starts prevents the common failure mode of tuning expanding to fill all available time with diminishing returns past the first 50–100 trials per model
- The search strategy matches the budget: Bayesian optimisation for large continuous spaces, random search for moderate spaces, grid search for small or discrete spaces where exact coverage and reproducibility of the search matter more than efficiency
- Running the search on a representative stratified subsample and validating the winner on the full dataset is the standard speed-cost tradeoff

**Knowledge & Information**
- The same CV scheme used for evaluation should be used for tuning to keep estimates consistent and comparable
- Reporting the distribution of CV scores across configurations (not only the best) characterises sensitivity and reveals whether the winner is a stable optimum or a lucky tail draw
- For boosted trees, the high-leverage hyperparameters are typically learning rate, number of leaves / max depth, min child weight / min data in leaf, L1/L2 regularisation, feature/row subsampling fractions, and early-stopping rounds
- For neural networks, learning rate and learning-rate schedule, batch size, optimiser choice, weight decay, dropout, and augmentation strength dominate; architecture changes often matter less than these
- Successive halving and Hyperband prune unpromising configurations early and routinely cut search cost by 3–10× while typically preserving the winning configuration; the pruning can occasionally drop late-bloomers whose early loss is high but whose converged optimum is strong, so a full evaluation budget on a small confirmation slate is a reasonable hedge when search outcomes are surprising
- Winning hyperparameters from past competitions on the same data modality often transfer as strong defaults and reduce the search to narrow refinement

**Tricks**
- Persisting the full search history (every trial's parameters, score, and intermediate state) enables resuming after interruptions, inspecting parameter importance, and warm-starting future searches in the same competition
- A coarse-to-fine schedule — a wide search with few trials, then a narrow search around the best region — is more efficient than a single broad search
- Tuning multiple models in parallel using independent studies, then ensembling, often yields more total signal than exhaustively tuning a single model

## Phase 7 — Ensembling & Stacking

**Planning**
- The ensembling plan starts from the set of diverse base models built across Phase 5 rather than from squeezing a final percentage point out of any single model
- A simple weighted average is the natural first ensemble; stacking is the next step when base model errors are uncorrelated enough to support a meta-learner
- Ensemble selection and weighting are performed on a holdout that no base model has seen — typically a reserved blend fold or out-of-fold predictions — to avoid overfitting the blend

**Knowledge & Information**
- Diversity of predictions drives ensemble gains; two moderately strong models with uncorrelated errors outperform two strong models that make the same mistakes
- Ensembling models from the same family (multiple gradient-boosting implementations, multiple variants of the same neural architecture) produces highly correlated predictions, but their differences in growth policy (leaf-wise vs. level-wise), categorical handling, regularisation, or initialisation are large enough that small but consistent gains over the single best member are common — in tight competitions where every fraction of a metric point matters, within-family blends are worth keeping in the ensemble even when their marginal lift is modest
- The largest gains typically come from combining genuinely different families (gradient boosting + neural network + linear) or models trained on substantially different feature sets or data samples
- Stacking with a simple meta-learner (logistic regression, ridge, light boosted tree with few leaves) on out-of-fold predictions captures systematic differences between base models with low overfitting risk on the blend; more expressive meta-learners can still be appropriate when base models are many and their interactions are non-trivial, provided the meta-learner is itself validated on a fold disjoint from the one used to train it
- Weight optimisation via constrained optimisers (Nelder-Mead, simplex methods, gradient-based solvers under a simplex constraint) on held-out CV often improves over uniform averaging when base models have meaningfully different strengths; validating the optimised weights on a separate fold guards against overfitting the blend
- Geometric mean (averaging in log-space) and rank-averaging are alternatives to arithmetic mean that work better when predictions have heterogeneous scale or are used as rankings rather than probabilities

**Tricks**
- Adding a poorly tuned, low-capacity model from a different family (a small MLP alongside boosted trees) often improves the ensemble despite being individually weaker
- Correlation matrices of OOF predictions across base models reveal which models contribute genuine diversity and which are redundant; pruning redundant models stabilises the blend without hurting score
- Training base models on different feature subsets (full, A, B) and ensembling produces cheap diversity without new architectures
- Capping prediction range to the empirical target range, or clipping outliers, can yield small but reliable gains under metrics that penalise extreme errors
- Multi-level stacking (a second-stage stacker over first-stage stacker outputs) has won mature, ensemble-heavy competitions where many strong and architecturally varied base models exist, but the marginal return diminishes quickly with each added stage and the complexity overhead is high — single-level stacking with diverse base models captures most of the available signal in most settings, and the additional level is worth the cost mainly when the ensemble is already large and well-validated

**Review** (close before moving on)
- Does the ensemble CV exceed the best single-model CV by a margin that survives across seeds? If not, the ensemble is not contributing — return to Phase 5 to build genuinely different models, or to Phase 4 to construct alternative feature subsets
- Are the base model OOF predictions correlated above ~0.95? Diversity is insufficient — return to Phase 5 (different family) or Phase 4 (different feature subset)
- Is the public LB improvement consistent with the CV improvement? If not, the blend is overfitting OOF — return to Phase 2 to inspect CV structure
- Has the meta-learner been validated on a fold disjoint from the one used to fit base models? If not, the stacking estimate is optimistic

## Phase 8 — Final Submission Selection

**Planning**
- Most platforms allow two final submissions for private leaderboard evaluation; diversification — typically one best-CV submission and one best-LB submission — reduces the chance of a catastrophic private-LB reshuffle
- Final candidates should be prepared and validated 24–48 hours before the deadline; reserving a tested pipeline for final-day use prevents last-minute breakage from costing the competition
- The final day is for selection, sanity checks, and re-running the submission pipeline end-to-end — not for new architectures or feature ideas

**Knowledge & Information**
- The private leaderboard frequently reshuffles relative to public; a position-conservative final selection hedges against either signal being misleading
- Confidence in CV over LB increases when the CV scheme provably replicates the test split, when CV variance is small, and when the CV-LB gap has been consistent across many submissions
- Some late-stage strategies — clipping predictions to a tighter range, blending with a constant, applying a learned calibration, threshold optimisation on OOF for classification under non-standard metrics — yield small but reliable gains
- Threshold optimisation under metrics like F-beta or Quadratic Weighted Kappa is performed on OOF predictions and applied to the final test predictions
- The shake-up risk between public and private leaderboards is higher in competitions with small test sets, severe class imbalance, or distribution shift between train and test — adjusting final-selection conservatism accordingly is part of the strategy

**Tricks**
- A "safety" submission constructed as the average of the top-N CV submissions is often more robust than picking any single submission and frequently outperforms it on private LB
- Comparing the prediction distribution of the final submission against the training target distribution catches obvious calibration mistakes (overly confident, biased toward one class) before they cost the leaderboard
- A short pre-submission checklist (correct rows, correct column order, no NaN, range plausible, file size sane, file opens in the platform's preview) catches the most embarrassing mistakes
- Test-time augmentation and multi-seed inference averaging for image and neural pipelines apply naturally at the final stage and tend to nudge the score upward without changing any other component

## Phase Wiring

When signals from a later phase challenge the assumptions of an earlier one, looping back is part of the playbook rather than a sign of failure. The most common patterns:

- **Consistent CV-LB gap that diverges after introducing new features** → return to Phase 4's review; if unresolved, return to Phase 2 to redesign the CV scheme
- **Adversarial validation easily separates train from test** → return to Phase 1 to inspect the responsible features and to Phase 4 to engineer around them (or remove them)
- **A single feature dominates importance with implausibly high signal** → return to Phase 4 to trace feature provenance for leakage before trusting any downstream metric
- **Ensemble does not improve over the best single model** → return to Phase 5 to build genuinely different model families, or to Phase 4 to construct alternative feature subsets
- **Hyperparameter tuning yields large CV gains that do not appear on LB** → return to Phase 2 (CV scheme may not reflect test) or Phase 4 (features may be CV-specific)
- **Final-week consolidation reveals brittleness in the pipeline** → reserve compute for stability over additional features and return to Phase 0 to harden the harness
- **Mid-competition shared insight from the discussion forum changes the picture** (e.g. a documented leak, a previously unknown grouping structure) → revisit Phase 1 to incorporate the new understanding, then re-validate Phase 2 and Phase 4 in light of it
- **Distribution shift detected when comparing final-submission predictions to the training target distribution** → return to Phase 1 (re-examine differences) and Phase 2 (verify CV scheme reflects them)
- **CV variance per fold is large enough to obscure improvements** → return to Phase 2 to consider repeated CV, more folds, or a fold structure better aligned with the test distribution

The loops are bounded by time: late in the campaign, the cost of redesigning validation or pruning load-bearing features is high, and Phase 8's diversification across CV and LB is the pragmatic hedge against decisions that can no longer be unwound.