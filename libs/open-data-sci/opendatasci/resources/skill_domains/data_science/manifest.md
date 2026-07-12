# Data Science Skill Domain

The general practice of turning a question into an analysis: framing it precisely, exploring and preparing the data, reasoning about causality, testing hypotheses rigorously, building and evaluating models, and communicating the result. The skills below cover the full arc from a vague request to a defensible finding; load the one that matches the stage of work in front of you.

## Framing the Problem

- skill: data_science::framing_the_problem

Grounding a vague request in a concrete hypothesis or target metric, defining what a "good" answer looks like, and separating exploratory from confirmatory work; load at the very start of any analysis, before touching data.

## Exploratory Analysis

- skill: data_science::exploratory_analysis

Profiling shape, dtypes, missingness, and cardinality; reading distributions and correlations; and treating time-indexed data chronologically; load once the question is framed and before any modelling or cleaning decisions are made.

## Data Quality & Preparation

- skill: data_science::data_quality_preparation

Diagnosing why data is missing, investigating outliers before treating them, checking joins for row inflation or key loss, and matching encoding and scaling choices to the model; load while cleaning and preparing data for analysis or modelling.

## Causality & Confounding

- skill: data_science::causality_confounding

Distinguishing correlation from causation, identifying confounders, watching for selection and survivorship bias, and checking for Simpson's paradox; load whenever an analysis aims to inform a decision rather than just describe what happened.

## Granularity & Aggregation

- skill: data_science::granularity_aggregation

Establishing what one row represents, making aggregation choices explicit, and avoiding aggregating too early or at the wrong level; load whenever joining, comparing, or summarising datasets.

## Statistical Testing

- skill: data_science::statistical_testing

Sample size and power, choosing the right parametric or non-parametric test, correcting for multiple comparisons, and reading p-values, effect sizes, and confidence intervals together; load whenever a claim needs to be tested rigorously.

## Modeling & Evaluation

- skill: data_science::modeling_evaluation

Establishing a naive baseline, choosing a model family driven by problem structure, selecting metrics and split strategies that reflect the real objective, and using diverse models for ensemble gains; load once the analysis moves into building and evaluating a model.

## Communicating Findings

- skill: data_science::communicating_findings

Reporting estimates with uncertainty, leading with the finding before the methodology, surfacing caveats and assumptions, and being explicit when the data doesn't support a strong conclusion; load when writing up or presenting results.
