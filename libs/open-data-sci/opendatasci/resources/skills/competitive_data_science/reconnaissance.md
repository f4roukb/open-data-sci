# Competitive Data Science — Phase 0: Reconnaissance & Harness

**Planning**
- The shape of the campaign — how many days for baseline, EDA, feature work, model development, ensembling, final-week consolidation — is set in this phase; without an explicit cadence, single phases tend to absorb disproportionate time
- The deliverable from this phase is a working end-to-end pipeline (load → split → train → predict → submission file) and an experiment log; later phases iterate on individual steps without rebuilding the harness
- Compute and time budgets influence model choice as much as data does; framing them up front avoids late discovery that the chosen architecture is untrainable in the available window
- Those budgets start from the actual machine: checking the available hardware resources (CPU cores, free RAM, GPU and VRAM, free disk) before the campaign plan is set grounds every later choice — an architecture or tuning plan sized for a GPU workstation is unexecutable in the available window on a modest CPU box
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

## Part of

- `competitive_data_science` — the phased competition playbook this phase belongs to; load it for the full campaign overview, phase sequencing, and the cross-phase loop-back wiring.
