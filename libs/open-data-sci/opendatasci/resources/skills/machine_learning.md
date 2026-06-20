# Machine Learning Skill

**Problem Framing**
- Before writing any code, the prediction task is worth thinking through carefully: what exactly is being predicted, at what point in time, using what information, and with what tolerance for different kinds of errors
- Label quality often determines the ceiling on model performance more than architecture does — severe imbalance, label noise, and ambiguous labelling criteria are worth surfacing early
- The data modality (tabular, text, image, time series, graph) and available volume should inform the range of approaches worth considering
- Leakage — features computed using information that wouldn't be available at prediction time — is the most common source of over-optimistic evaluation results; it's worth tracing feature provenance carefully before trusting any metrics

**Splitting Strategy**
- The split strategy encodes assumptions about the real-world prediction setting; getting it wrong produces metrics that don't generalise
- Temporal data requires time-ordered splits to avoid the model seeing the future during training — random shuffling of chronological data is a quiet but serious source of leakage
- When rows share an entity (user, customer, session, location), entity-aware splits prevent the model from memorising entities it will never see again
- Stratified splits preserve class distribution across folds and are particularly important when classes are rare
- The test set should be treated as a one-time evaluation; tuning against it invalidates it as an unbiased estimate

**Feature Engineering & Selection**
- Preprocessing transformations need to be fit only on training data and applied consistently to every split; fitting on the full dataset before splitting leaks distributional information into evaluation
- For tabular data, useful transformations include datetime decomposition, lag and rolling features, interaction terms, and outlier treatment — the right choices are domain-dependent
- High-cardinality categoricals benefit from encoding strategies that don't naively explode dimensionality; the right approach depends on the model family and available data volume
- Feature magnitude matters for some model families and is irrelevant for others — scaling decisions should match the model's sensitivity
- More features is not always better: irrelevant or redundant features add noise and variance, can hurt distance-based models, and make models harder to interpret and debug; feature selection (variance-based filtering, correlation-based pruning, importance-based selection) is often worth doing before fitting complex models
- Training/inference skew — features computed differently at training time versus inference time — is a common source of silent degradation

**Model Selection & Complexity**
- A minimal baseline establishes the floor: what performance is achievable with no model at all, or with a trivial rule? This makes the value of subsequent complexity concrete and measurable
- Increasing model complexity should be driven by measured improvement on held-out data, not by a prior assumption that a more powerful model will help — underfitting is a real failure mode too
- Model selection should follow the problem structure and data characteristics, not a habitual preference for any particular algorithm family — the choice of model is itself a design decision that deserves the same rigour as any other modelling choice
- For tabular data, a principled progression starts with linear or logistic regression as an interpretable, regularisable baseline; tree-based ensembles (random forests, gradient-boosting variants such as LightGBM, XGBoost, CatBoost, or scikit-learn's HistGradientBoosting) are the natural next step when non-linearity and feature interactions are evident and data volume supports them; neural architectures (MLP, TabNet, FT-Transformer) make sense when data volume is large and the iteration budget supports their tuning overhead — moving up the complexity ladder only when simpler models demonstrably fall short
- For text and NLP tasks, TF-IDF or bag-of-words features with a linear classifier is a strong, cheap baseline; pre-trained transformer models (BERT-family, smaller distilled variants, domain-specific checkpoints) substantially raise the performance ceiling but carry far higher compute costs — calibrate the choice to the task scale and available resources
- For time-series and sequential data, statistical methods (ARIMA, ETS, Theta, Prophet) are often sufficient and interpretable when the series is short, labelled features are scarce, or seasonality and trend dominate; ML approaches (tree ensembles with lag features, LSTM, Temporal Fusion Transformer) add value when exogenous signals, many parallel series, or complex non-linear dependencies are present
- For image and spatial data, pre-trained convolutional or vision-transformer backbones via transfer learning are the standard entry point due to transfer efficiency; simpler feature-based approaches (HOG, colour statistics, patch descriptors) remain viable when labels are very scarce or compute is tightly constrained
- Model families make different assumptions: linear models assume additive, globally linear effects and are easy to interpret, debug, and regularise; tree-based ensembles capture interactions and non-linearities without explicit feature engineering and are relatively robust to irrelevant features and scale differences; neural networks learn representations end-to-end but require substantially more data, hyperparameter effort, and iteration cycles to realise their potential — these tradeoffs should be reasoned through explicitly for each problem
- The right level of complexity depends on data volume, feature richness, signal-to-noise ratio, and interpretability requirements; many real problems are solved well by simple models, and complexity beyond what the signal supports hurts rather than helps
- Ensemble and stacking strategies can narrow the gap between model families and are most effective when combining architecturally distinct model families (e.g., gradient boosting + neural network + linear model) or models trained on different feature sets or data samples — ensembling similar frameworks (e.g., XGBoost, LightGBM, CatBoost) produces highly correlated predictions and therefore little diversity benefit

**Hyperparameter Tuning**
- The search strategy should match the budget: exhaustive search is only feasible for small spaces; random search and Bayesian optimisation cover large spaces more efficiently
- The wall-clock cost of a search is the product of configurations evaluated and the cost of each evaluation; on large datasets both factors deserve explicit management — run the search on a carefully constructed representative subsample (stratified by class and any structural variable such as time period or entity) to cut per-evaluation time dramatically while preserving most of the directional signal; techniques like successive halving and Hyperband go further by pruning unpromising configurations early rather than running every trial to completion; always re-train or re-evaluate the winning configuration on the full dataset before committing to it
- The same cross-validation strategy used for model evaluation should be used during tuning to keep estimates consistent
- Reporting the distribution of CV scores across configurations, not just the best, gives a clearer picture of sensitivity to hyperparameter choices
- All stochastic components should have fixed random seeds to ensure results are stable across runs

**Overfitting & Regularisation**
- The gap between training and validation performance is the primary diagnostic for overfitting; a model that performs well on training data but poorly on held-out data has memorised rather than generalised
- Overfitting risk increases with model capacity relative to data volume — more parameters, more trees, deeper networks all have higher capacity and require more data or more regularisation to generalise
- Regularisation techniques (penalising model complexity, limiting depth, adding noise during training, early stopping) are the primary tools for closing the train/val gap; the right form depends on the model family
- Learning curves (performance as a function of training set size) are a useful diagnostic: poor performance that improves with more data suggests a data problem; a persistent train/val gap that doesn't close suggests a regularisation or capacity problem
- Underfitting — where even training performance is poor — points in the opposite direction: the model may lack the capacity or the features to capture the signal

**Evaluation & Diagnostics**
- Aggregate metrics can hide a lot; slicing by relevant subgroups, prediction ranges, or time periods often reveals where a model underperforms in ways the headline number conceals
- Error analysis — directly examining mispredictions (false positives, false negatives, worst residuals, confused classes) — is often more informative than metrics alone; patterns in where a model fails point directly at what to fix
- For classification, the decision threshold affects the precision-recall tradeoff and should be chosen deliberately based on the relative cost of false positives versus false negatives, not left at a default
- Probability calibration matters when predicted scores are used as actual probability estimates rather than just rankings
- Residual analysis for regression surfaces systematic patterns — heteroscedasticity, non-linearity, outlier influence — that aggregate error metrics don't capture
- Cross-validation spread (mean ± std across folds) characterises how stable performance is, not just what the best-case number is

**Class Imbalance**
- Class imbalance affects both what the model learns and how performance is measured; addressing only one side gives a misleading picture
- Threshold adjustment and class weighting are low-cost interventions that often recover substantial minority-class performance without resampling
- Resampling techniques change the training distribution and should only ever be applied within the training fold — contaminating validation or test data with synthetic samples invalidates evaluation
- Extreme imbalance changes the problem framing: precision at a given recall threshold or anomaly detection approaches may be more appropriate than standard classification evaluation

**Interpretability**
- Understanding what a model has learned is distinct from understanding why it performs well; feature importance methods address the former
- Global importances summarise the model's overall behaviour across the dataset; local explanations explain individual predictions — both are useful but answer different questions
- Impurity-based importance measures are known to favour high-cardinality features and can be misleading; permutation-based importance and gradient-based attribution methods are generally more reliable
- Interpretability requirements should inform model choice early; post-hoc approximations of black-box models have their own failure modes