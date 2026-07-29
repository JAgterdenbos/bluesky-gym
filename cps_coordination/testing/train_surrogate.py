"""
cps_coordination/testing/train_surrogate.py
--------------------------------------------
Routine (re)training script for the self-describing ETASurrogate.

Loads the feature-set/target-transform decision from a selection artifact
produced once by ``select_surrogate_features.py`` and does a single final
``ExtraTreesRegressor`` fit on the full dataset — no per-run feature
importance reduction, no transform-comparison CV. See
``select_surrogate_features.py``'s module docstring for why that decision
was pulled out into its own script (it barely changes run to run, so paying
for ~26 extra tree-ensemble fits on every retrain was pure waste).

Pipeline
--------
1.  Load the selection artifact (feature columns to keep, target transform,
    lag_steps/window used when that decision was made).
2.  Load parquet data, filter successful episodes, compute steps_to_go.
3.  Derive exact IAF anchors from PathPlanningGoalEnv constants (no estimation).
4.  Engineer static geometric + lag features (using the selection's lag
    params unless overridden).
5.  Build the full 14-column feature matrix; slice to the selected columns.
6.  Optionally (``--report-cv``): GroupKFold CV on the final pipeline, purely
    for an informational OOF metrics printout — skipped by default.
7.  Fit the final model on the full reduced dataset with the winning transform.
8.  Package into ETASurrogate via ETASurrogate.from_training() and save.

Usage
-----
  # 1. One-time (or occasional) feature/transform selection:
  python cps_coordination/testing/select_surrogate_features.py \\
      path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet

  # 2. Routine training, reusing that selection:
  python cps_coordination/testing/train_surrogate.py \\
      path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet \\
      --output cps_coordination/models/eta_surrogate.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME
from cps_coordination.coordination.eta_surrogate import ETASurrogate, TRANSFORMS
from cps_coordination.testing.surrogate_data import (
    build_feature_matrix,
    et_metrics,
    prepare_modelling_features,
)

_DEFAULT_OUTPUT = Path("cps_coordination/models/eta_surrogate.pkl")
_DEFAULT_SELECTION = Path("cps_coordination/models/surrogate_feature_selection.yaml")


# ---------------------------------------------------------------------------
# Selection artifact
# ---------------------------------------------------------------------------

def load_selection(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"No feature-selection artifact at {path}. Run "
            f"select_surrogate_features.py first (see this script's module docstring)."
        )
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Optional diagnostic CV — reports expected OOF metrics for the exact
# pipeline about to be shipped (fixed features + fixed transform), unlike
# the old script's CV which ran on the full un-reduced feature set.
# ---------------------------------------------------------------------------

def report_cv(
    X_reduced: np.ndarray,
    y: np.ndarray,
    groups,
    fwd_fn,
    inv_fn,
    n_splits: int,
    et_params: dict,
) -> None:
    gkf = GroupKFold(n_splits=n_splits)
    oof_y: List[np.ndarray] = []
    oof_p: List[np.ndarray] = []

    print(f"\n{n_splits}-Fold Group Cross-Validation (final feature set + transform)")
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_reduced, groups=groups)):
        model = ExtraTreesRegressor(**et_params).fit(X_reduced[tr_idx], fwd_fn(y[tr_idx]))
        y_pred = inv_fn(model.predict(X_reduced[va_idx]))
        oof_y.append(y[va_idx])
        oof_p.append(y_pred)
        m = et_metrics(y[va_idx], y_pred)
        print(f"  Fold {fold_idx + 1}/{n_splits}: R2={m['R2']:.4f}  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}")

    m_oof = et_metrics(np.concatenate(oof_y), np.concatenate(oof_p))
    print(f"  OOF: R2={m_oof['R2']:.4f}  MAE={m_oof['MAE']:.2f} steps  RMSE={m_oof['RMSE']:.2f} steps")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and serialise an ETASurrogate using a pre-computed feature/transform selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("data", type=Path, help="Path to rollout parquet (or CSV) file.")
    p.add_argument("--selection", type=Path, default=_DEFAULT_SELECTION,
                   help="Feature-selection YAML produced by select_surrogate_features.py.")
    p.add_argument("--output", "-o", type=Path, default=_DEFAULT_OUTPUT,
                   help="Destination path for the serialised ETASurrogate.")
    p.add_argument("--n-estimators", type=int, default=15,
                   help="Matches the DTG sampler's actual deployed choice (not its "
                        "100-tree code default) — the paper's ET deployment "
                        "memory-cost trade-off: 15 vs 100 estimators took DTG's "
                        "footprint from ~40GB to ~1.0-1.5GB for a small accuracy cost.")
    p.add_argument("--max-depth", type=int, default=15)
    p.add_argument("--min-samples-leaf", type=int, default=10)
    p.add_argument("--lag-steps", type=int, default=None,
                   help="Step lag for delta_atd. Defaults to the value recorded "
                        "in the selection artifact.")
    p.add_argument("--window", type=int, default=None,
                   help="Rolling window for cumabs_cte/heading_volatility. Defaults "
                        "to the value recorded in the selection artifact.")
    p.add_argument("--target", choices=["steps", "seconds"], default=None,
                   help="Regression target. Defaults to the value recorded in the "
                        "selection artifact ('steps' if absent, for selections made "
                        "before this flag existed). 'seconds' (continuous time_to_go) "
                        "forces --sim-dt to 1.0 regardless of the flag below, since "
                        "the model then predicts seconds directly.")
    p.add_argument("--sim-dt", type=float, default=float(ACTION_TIME),
                   help="Seconds per unit of the model's predicted step count. "
                        "This is ACTION_TIME (the env's decision-step interval), "
                        "NOT bluesky's 5s physics tick — the surrogate is trained "
                        "against rollout data whose 'step' column increments once "
                        "per env.step() call (see surrogate_data.py). Ignored "
                        "(forced to 1.0) when --target=seconds.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--report-cv", action="store_true",
                   help="Also run a GroupKFold CV on the final pipeline for an "
                        "informational OOF metrics printout (adds n_splits extra fits).")
    p.add_argument("--report-cv-n-splits", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    selection = load_selection(args.selection)
    lag_steps = args.lag_steps if args.lag_steps is not None else selection["lag_steps"]
    window = args.window if args.window is not None else selection["window"]
    if lag_steps != selection["lag_steps"] or window != selection["window"]:
        print(
            f"  Note: lag_steps/window ({lag_steps}/{window}) differ from the "
            f"selection artifact's ({selection['lag_steps']}/{selection['window']}) — "
            "the feature/transform choice may no longer be optimal for this data."
        )

    col_indices = selection["col_indices"]
    feature_names_kept = selection["feature_names_kept"]
    transform_name = selection["transform_name"]
    fwd_fn, inv_fn = TRANSFORMS[transform_name]

    target = args.target if args.target is not None else selection.get("target", "steps")
    if target == "seconds":
        sim_dt = 1.0
        print(f"  Target: 'seconds' (continuous time_to_go) -> forcing sim_dt=1.0 "
              f"(model predicts seconds directly; --sim-dt={args.sim_dt} ignored)")
    else:
        sim_dt = args.sim_dt

    et_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=args.random_state,
    )

    print(f"Loading data from: {args.data}")
    _raw_df, model_df, iaf_ref, runway_encoder, all_runways = prepare_modelling_features(
        args.data, lag_steps, window
    )
    print(f"  {len(model_df):,} modelling rows  |  {model_df['episode'].nunique():,} episodes  |  {len(all_runways)} runways")

    X_full, y_full, _feat_names = build_feature_matrix(model_df, runway_encoder, target=target)
    X_reduced = X_full[:, col_indices]
    print(f"  Reduced feature matrix: {X_reduced.shape}  (features: {feature_names_kept})")
    print(f"  Target: {target!r}  |  Target transform: {transform_name!r}")

    if args.report_cv:
        report_cv(
            X_reduced, y_full, model_df["episode"], fwd_fn, inv_fn,
            args.report_cv_n_splits, et_params,
        )

    print(f"\nFitting final model on {len(y_full):,} samples ...")
    model = ExtraTreesRegressor(**et_params).fit(X_reduced, fwd_fn(y_full))
    print("  Done.")

    needs_lag = any(
        name in feature_names_kept
        for name in ["delta_atd", "cumabs_cte", "heading_volatility"]
    )

    surrogate = ETASurrogate.from_training(
        model=model,
        iaf_ref=iaf_ref,
        runway_encoder=runway_encoder,
        feature_names=feature_names_kept,
        col_indices=col_indices,
        needs_lag=needs_lag,
        lag_steps=lag_steps,
        window=window,
        sim_dt=sim_dt,
        transform_name=transform_name,
        inv_transform=inv_fn,
        target=target,
    )

    print(f"\nSurrogate summary:")
    print(f"  {surrogate!r}")
    print(f"  Needs lag features: {surrogate._needs_lag}")
    print(f"  State layout      : {surrogate.state_layout}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    surrogate.save(args.output)
    print(f"\nSurrogate saved -> {args.output.resolve()}")

    loaded = ETASurrogate.load(args.output)
    test_state = np.array([0.3, -0.4, 120.0, 45.0])
    lag = np.zeros(3)
    test_rwy = all_runways[0]
    eta = loaded.predict_eta(test_state, test_rwy, current_sim_time=0.0, lag_features=lag)
    print(f"  Round-trip check: predict_eta({test_rwy!r}) = {eta:.1f} s  [OK]")


if __name__ == "__main__":
    main()
