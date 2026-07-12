# Competitive Data Science — Phase 8: Final Submission Selection

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

## Metadata

- parent domain: competitive_data_science
