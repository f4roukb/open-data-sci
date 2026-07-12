# Machine Learning Skill Domain

The core practice of building and evaluating predictive models: framing the prediction task, splitting data soundly, engineering and selecting features, choosing model complexity, tuning hyperparameters, diagnosing overfitting, evaluating rigorously, handling class imbalance, and interpreting what a model has learned. Load the skill that matches the stage of the modelling workflow you're in.

## Problem Framing

- skill: machine_learning::problem_framing

Defining the prediction task precisely, assessing label quality, matching approach to data modality and volume, and tracing feature provenance for leakage; load before writing any modelling code.

## Splitting Strategy

- skill: machine_learning::splitting_strategy

Choosing time-ordered, entity-aware, or stratified splits to match the real-world prediction setting, and treating the test set as a one-time evaluation; load before any training or validation begins.

## Feature Engineering & Selection

- skill: machine_learning::feature_engineering_selection

Fitting preprocessing only on training data, tabular transformation families, encoding high-cardinality categoricals, scaling decisions, feature selection, and avoiding training/inference skew; load while building or revising the feature set.

## Model Selection & Complexity

- skill: machine_learning::model_selection_complexity

Establishing a baseline, matching model family to data modality (tabular, text, time series, image) and volume, the assumptions each family makes, and when ensembling across diverse families pays off; load when choosing or escalating model complexity.

## Hyperparameter Tuning

- skill: machine_learning::hyperparameter_tuning

Matching search strategy to budget, cutting search cost via representative subsamples and early pruning, keeping CV consistent between tuning and evaluation, and fixing seeds; load once a model is ready to be tuned.

## Overfitting & Regularisation

- skill: machine_learning::overfitting_regularisation

Reading the train/validation gap, matching regularisation technique to model family, and using learning curves to distinguish data-limited from capacity-limited regimes; load when diagnosing over- or under-fitting.

## Evaluation & Diagnostics

- skill: machine_learning::evaluation_diagnostics

Slicing metrics by subgroup, doing direct error analysis, choosing classification thresholds deliberately, calibration, residual analysis, and reading CV spread; load when evaluating a trained model beyond the headline metric.

## Class Imbalance

- skill: machine_learning::class_imbalance

Threshold adjustment, class weighting, fold-safe resampling, and reframing extreme-imbalance problems; load whenever the target classes are meaningfully imbalanced.

## Interpretability

- skill: machine_learning::interpretability

Global versus local explanations, the pitfalls of impurity-based importance versus permutation and gradient-based methods, and factoring interpretability into model choice; load when a model's behaviour needs to be explained.
