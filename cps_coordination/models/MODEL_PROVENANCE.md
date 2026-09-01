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

**Caveat (added 2026-09-01, found during thesis V&V):** the table above uses
`select_surrogate_features.py`'s scout CV, which defaults to `n_estimators=50`.
The model that actually shipped (`eta_surrogate.pkl`) is trained via
`train_surrogate.py` with `n_estimators=15` (same memory-vs-accuracy tradeoff
as the DTG sampler, see `chapter4_more_results.md`'s Production Feature Sets
section) — so the table's "current production" row does **not** reflect the
shipped model's actual accuracy. It's still a valid, matched comparison of the
*feature decision* (`naive_eta_remaining` in vs. out, both at 50 trees), just
not the deployed model's held-out performance.

Re-running `validate_surrogate.py`'s held-out CV directly against the shipped
`eta_surrogate.pkl` (its real 15-tree config, same data, same
`GroupKFold(n_splits=5, random_state=42)`) gives:

| | shipped model (15 trees, real held-out CV) |
|---|---|
| R² | 0.9908 |
| MAE | 57.1s |
| RMSE | 76.4s |

This is the number that should be cited as the ETA surrogate's actual
predictive accuracy — reproducible via `python
cps_coordination/testing/validate_surrogate.py --skip-condition3`.

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
