# Machine Learning Skill Domain

The core practice of building and evaluating predictive models: framing the prediction task, splitting data soundly, engineering and selecting features, choosing model complexity, tuning hyperparameters, diagnosing overfitting, evaluating rigorously, handling class imbalance, and interpreting what a model has learned. Load the skill that matches the stage of the modelling workflow you're in.

## Problem Framing

### Metadata
- skill_domain_name: machine_learning
- skill_name: problem_framing

### Description
Defining the prediction task precisely, assessing label quality, matching approach to data modality and volume, and tracing feature provenance for leakage; load before writing any modelling code.

## Splitting Strategy

### Metadata
- skill_domain_name: machine_learning
- skill_name: splitting_strategy

### Description
Choosing time-ordered, entity-aware, or stratified splits to match the real-world prediction setting, and treating the test set as a one-time evaluation; load before any training or validation begins.

## Feature Engineering & Selection

### Metadata
- skill_domain_name: machine_learning
- skill_name: feature_engineering_selection

### Description
Fitting preprocessing only on training data, tabular transformation families, encoding high-cardinality categoricals, scaling decisions, feature selection, and avoiding training/inference skew; load while building or revising the feature set.

## Model Selection & Complexity

### Metadata
- skill_domain_name: machine_learning
- skill_name: model_selection_complexity

### Description
Establishing a baseline, matching model family to data modality (tabular, text, time series, image) and volume, the assumptions each family makes, and when ensembling across diverse families pays off; load when choosing or escalating model complexity.

## Hyperparameter Tuning

### Metadata
- skill_domain_name: machine_learning
- skill_name: hyperparameter_tuning

### Description
Matching search strategy to budget, cutting search cost via representative subsamples and early pruning, keeping CV consistent between tuning and evaluation, and fixing seeds; load once a model is ready to be tuned.

## Overfitting & Regularisation

### Metadata
- skill_domain_name: machine_learning
- skill_name: overfitting_regularisation

### Description
Reading the train/validation gap, matching regularisation technique to model family, and using learning curves to distinguish data-limited from capacity-limited regimes; load when diagnosing over- or under-fitting.

## Evaluation & Diagnostics

### Metadata
- skill_domain_name: machine_learning
- skill_name: evaluation_diagnostics

### Description
Slicing metrics by subgroup, doing direct error analysis, choosing classification thresholds deliberately, calibration, residual analysis, and reading CV spread; load when evaluating a trained model beyond the headline metric.

## Class Imbalance

### Metadata
- skill_domain_name: machine_learning
- skill_name: class_imbalance

### Description
Threshold adjustment, class weighting, fold-safe resampling, and reframing extreme-imbalance problems; load whenever the target classes are meaningfully imbalanced.

## Interpretability

### Metadata
- skill_domain_name: machine_learning
- skill_name: interpretability

### Description
Global versus local explanations, the pitfalls of impurity-based importance versus permutation and gradient-based methods, and factoring interpretability into model choice; load when a model's behaviour needs to be explained.
