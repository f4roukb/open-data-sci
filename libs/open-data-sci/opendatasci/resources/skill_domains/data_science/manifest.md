# Data Science Skill Domain

The general practice of turning a question into an analysis: framing it precisely, exploring and preparing the data, reasoning about causality, testing hypotheses rigorously, building and evaluating models, and communicating the result. The skills below cover the full arc from a vague request to a defensible finding; load the one that matches the stage of work in front of you.

## Framing the Problem

### Metadata
- skill_domain_name: data_science
- skill_name: framing_the_problem

### Description
Grounding a vague request in a concrete hypothesis or target metric, defining what a "good" answer looks like, and separating exploratory from confirmatory work; load at the very start of any analysis, before touching data.

## Exploratory Analysis

### Metadata
- skill_domain_name: data_science
- skill_name: exploratory_analysis

### Description
Profiling shape, dtypes, missingness, and cardinality; reading distributions and correlations; and treating time-indexed data chronologically; load once the question is framed and before any modelling or cleaning decisions are made.

## Data Quality & Preparation

### Metadata
- skill_domain_name: data_science
- skill_name: data_quality_preparation

### Description
Diagnosing why data is missing, investigating outliers before treating them, checking joins for row inflation or key loss, and matching encoding and scaling choices to the model; load while cleaning and preparing data for analysis or modelling.

## Causality & Confounding

### Metadata
- skill_domain_name: data_science
- skill_name: causality_confounding

### Description
Distinguishing correlation from causation, identifying confounders, watching for selection and survivorship bias, and checking for Simpson's paradox; load whenever an analysis aims to inform a decision rather than just describe what happened.

## Granularity & Aggregation

### Metadata
- skill_domain_name: data_science
- skill_name: granularity_aggregation

### Description
Establishing what one row represents, making aggregation choices explicit, and avoiding aggregating too early or at the wrong level; load whenever joining, comparing, or summarising datasets.

## Statistical Testing

### Metadata
- skill_domain_name: data_science
- skill_name: statistical_testing

### Description
Sample size and power, choosing the right parametric or non-parametric test, correcting for multiple comparisons, and reading p-values, effect sizes, and confidence intervals together; load whenever a claim needs to be tested rigorously.

## Modeling & Evaluation

### Metadata
- skill_domain_name: data_science
- skill_name: modeling_evaluation

### Description
Establishing a naive baseline, choosing a model family driven by problem structure, selecting metrics and split strategies that reflect the real objective, and using diverse models for ensemble gains; load once the analysis moves into building and evaluating a model.

## Communicating Findings

### Metadata
- skill_domain_name: data_science
- skill_name: communicating_findings

### Description
Reporting estimates with uncertainty, leading with the finding before the methodology, surfacing caveats and assumptions, and being explicit when the data doesn't support a strong conclusion; load when writing up or presenting results.
