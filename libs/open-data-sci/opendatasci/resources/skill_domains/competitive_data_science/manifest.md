# Competitive Data Science Skill

A phased playbook for high-stakes data science competitions, built around incremental delivery. Each phase opens with planning so direction is set before work begins, and the high-leverage phases close with a review so signals from the current phase loop back into earlier decisions when warranted. The wiring at the end of the document spells out the most common loop-backs explicitly.

The phases describe what tends to matter at each stage, the knowledge that informs the choices, and the tricks that experienced competitors rely on. They do not prescribe a single path — flexibility to adapt the order, skip a step, or revisit a phase is part of the playbook.

Each phase below is detailed in its own skill file. Load the file for a phase when you are actively working that phase — its Planning / Knowledge & Information / Tricks (and, where present, Review) content is the substance; what follows here is only the pointer.

## Phase 0 — Reconnaissance & Harness

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: reconnaissance

### Description
Campaign planning (timeline, compute/hardware/team budgets), pre-competition research (metric quirks, prior editions, forums, submission format), and harness tricks (v0 submission, experiment log, naming conventions, seed fixing); load at the very start of a competition, before any pipeline exists.

## Phase 1 — Exploratory Data Analysis

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: eda

### Description
The question list that should drive EDA, profiling and adversarial-validation techniques for spotting train/test shift, and tricks for surfacing quirks fast; load once the harness is in place and before designing validation or features.

## Phase 2 — Validation Strategy

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: validation

### Description
Designing a CV scheme that mirrors how the test set was built, reading the CV–LB relationship, and a phase-close review checklist; load before any serious modelling begins, and revisit whenever CV and LB disagree.

## Phase 3 — Baseline Model

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: baseline

### Description
What a naive and a first real baseline look like per data modality, why submitting one early matters, and tricks for logging metrics and running ablations; load right after validation is settled, before feature or model work.

## Phase 4 — Feature Engineering

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: feature_engineering

### Description
The vocabulary of feature families (aggregations, target/frequency/cyclic encoding, lags, interactions, modality-specific features), leak-safe construction patterns, and a phase-close review checklist; load once a baseline exists and feature iteration begins.

## Phase 5 — Model Development

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: model_development

### Description
How to pick model families per data modality and scale, and training tricks (seed averaging, early stopping, pseudo-labelling, distillation); load once a feature set is in place and a portfolio of base models is being built.

## Phase 6 — Hyperparameter Tuning

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: hyperparameter_tuning

### Description
Setting a tuning budget and search strategy, the high-leverage hyperparameters per model family, and tricks like coarse-to-fine search and successive halving; load once base models exist and are ready to be tuned.

## Phase 7 — Ensembling & Stacking

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: ensembling

### Description
Blending and stacking diverse base models, weight optimisation, diversity diagnostics, and a phase-close review checklist; load once multiple base models (and their OOF predictions) are available.

## Phase 8 — Final Submission Selection

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: final_submission

### Description
Choosing conservative final submissions under leaderboard shake-up risk, late-stage calibration tricks, and the pre-submission checklist; load in the final 24–48 hours of a competition.

## Phase Wiring

### Metadata
- skill_domain_name: competitive_data_science
- skill_name: phase_wiring

### Description
The most common loop-back patterns between phases (e.g. a CV-LB gap after new features, adversarial validation catching a leak, an ensemble that doesn't help) and which phase to return to for each; load whenever a signal from the current phase casts doubt on an earlier decision, or to check for loop-backs after closing any phase's review.
