# `eta_surrogate.pkl` provenance

This directory (`cps_coordination/models/`) is gitignored — the model binaries
themselves are local-filesystem artifacts, not tracked in git. This file is
explicitly whitelisted (`.gitignore`) so the swap below survives even if the
local files are lost, regenerated, or the machine changes.

## Current production model (as of 2026-08-12)

`eta_surrogate.pkl` / `surrogate_feature_selection.yaml` — the **naive-feature
candidate**, promoted to production during the stall-rate investigation (see
`.claude/plans/stall_rate_investigation.md`, "Step 0"). Adds `naive_eta_remaining`
(a physical straight-line-at-cruise-speed floor) as an input feature, fixing a
diagnosed bug where the prior model sometimes predicted *below* the physical
floor (`cps_coordination/coordination/eta_surrogate.py`'s own docstring).

| | previous production (`*_pre_naive_feature_fix`) | current production |
|---|---|---|
| R² | 0.9908 | 0.9919 |
| MAE | 59.32s | 54.05s (−9%) |
| RMSE | 76.35s | 71.43s (−6%) |

Same source data for both (1.72M rows / 100k episodes), same selection
hyperparameters/random_state — a clean matched comparison, not a different
training run.

## Files in this directory

- `eta_surrogate.pkl`, `surrogate_feature_selection.yaml` — **current
  production**, the naive-feature candidate above.
- `eta_surrogate_pre_naive_feature_fix.pkl`,
  `surrogate_feature_selection_pre_naive_feature_fix.yaml` — backup of the
  model that was production before 2026-08-12. Kept for rollback / comparison,
  not referenced by any config.

(The original `*_naive_feature_candidate.*` files — byte-identical to current
production, verified via `sha1sum` at promotion time — were removed
2026-08-12 as redundant now that the swap is complete and documented above.)

## Regenerating

Trained via `cps_coordination/testing/train_surrogate.py` (see that script and
`cps_coordination/coordination/eta_surrogate.py` for the training pipeline and
feature list). Any future retrain that changes the feature set or fixes another
bug should update this file with the same before/after comparison table, and
re-run `cps_coordination/testing/validate_cps_pipeline.py`'s full regression
suite before promoting.
