"""
cps_coordination/testing/select_surrogate_features.py
--------------------------------------------------------
One-time (or occasional) analysis: decide which of the 14 canonical
ETASurrogate features to keep and which target transform to use, and save
that decision to a small YAML side-car artifact.

Why this is a separate script
------------------------------
``train_surrogate.py`` used to re-derive this decision from scratch on every
retrain (a scout ``ExtraTreesRegressor`` fit for importance reduction, plus a
5-fold x 3-transform CV sweep) — ~26 extra tree-ensemble fits that produced
the same answer run after run, since the surviving feature set and winning
transform are properties of the *feature engineering + data distribution*,
not something that needs re-deciding every time the production model is
refit on updated data. Run this script once (or whenever the feature
engineering changes, or you want to sanity-check the decision still holds on
new data); ``train_surrogate.py`` then just loads the result and does a
single final fit.

Uses cheaper trees than the production model by default (``--n-estimators``
default 50 vs. the production default of 200) since ranking importances and
comparing transforms doesn't need full production precision.

Usage
-----
  python cps_coordination/testing/select_surrogate_features.py \\
      path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet \\
      --output cps_coordination/models/surrogate_feature_selection.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from cps_coordination.coordination.eta_surrogate import TRANSFORMS, _LAG_COL_START, _N_LAG
from cps_coordination.testing.surrogate_data import (
    build_feature_matrix,
    et_metrics,
    prepare_modelling_features,
)

_DEFAULT_OUTPUT = Path("cps_coordination/models/surrogate_feature_selection.yaml")


# ---------------------------------------------------------------------------
# Feature reduction
# ---------------------------------------------------------------------------

def reduce_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    threshold: float,
    et_params: dict,
) -> Tuple[List[int], List[str]]:
    """Drop features below an importance threshold using a scout ET fit.

    Lag features (``delta_atd``/``cumabs_cte``/``heading_volatility``) share
    a semantic group and are kept or dropped together, by their summed
    importance.
    """
    scout = ExtraTreesRegressor(**et_params).fit(X_train, y_train)
    importances = scout.feature_importances_
    mask = importances >= threshold

    lag_idx = list(range(_LAG_COL_START, _LAG_COL_START + _N_LAG))
    if lag_idx:
        keep_lag = importances[lag_idx].sum() >= threshold
        for i in lag_idx:
            mask[i] = keep_lag

    col_indices = [i for i, keep in enumerate(mask) if keep]
    kept_names = [feature_names[i] for i in col_indices]
    dropped = [feature_names[i] for i in range(len(feature_names)) if not mask[i]]

    print(
        f"  Feature reduction: {len(col_indices)}/{len(feature_names)} kept"
        + (f"  |  dropped: {dropped}" if dropped else "  |  none dropped")
    )
    return col_indices, kept_names


# ---------------------------------------------------------------------------
# Transform selection
# ---------------------------------------------------------------------------

def select_best_transform(
    X_reduced: np.ndarray,
    y: np.ndarray,
    groups,
    n_splits: int,
    sig_threshold_pct: float,
    et_params: dict,
) -> Tuple[str, Dict[str, float]]:
    """Pick identity / log1p / sqrt by GroupKFold OOF RMSE."""
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n  {'Transform':<12} {'R2':>10} {'MAE':>10} {'RMSE':>10}")
    print("  " + "-" * 46)

    results = []
    for name, (fwd, inv) in TRANSFORMS.items():
        oof_y: List[np.ndarray] = []
        oof_p: List[np.ndarray] = []
        for tr_idx, va_idx in gkf.split(X_reduced, groups=groups):
            m = ExtraTreesRegressor(**et_params).fit(X_reduced[tr_idx], fwd(y[tr_idx]))
            oof_y.append(y[va_idx])
            oof_p.append(inv(m.predict(X_reduced[va_idx])))

        metrics = et_metrics(np.concatenate(oof_y), np.concatenate(oof_p))
        results.append({"name": name, "metrics": metrics})
        print(f"  {name:<12} {metrics['R2']:>10.4f} {metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f}")

    identity_rmse = results[0]["metrics"]["RMSE"]
    best = min(results, key=lambda r: r["metrics"]["RMSE"])
    improvement = 100.0 * (identity_rmse - best["metrics"]["RMSE"]) / identity_rmse

    print(f"\n  Improvement of {best['name']!r} over identity: {improvement:+.2f}%")
    winner = best if improvement >= sig_threshold_pct else results[0]
    decision = "significant" if improvement >= sig_threshold_pct else f"below threshold ({sig_threshold_pct:.1f}%)"
    print(f"  Decision: {decision} — using [{winner['name']}]")

    return winner["name"], winner["metrics"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-time feature-importance + target-transform selection for ETASurrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("data", type=Path, help="Path to rollout parquet (or CSV) file.")
    p.add_argument("--output", "-o", type=Path, default=_DEFAULT_OUTPUT,
                   help="Destination path for the selection YAML.")
    p.add_argument("--target", choices=["steps", "seconds"], default="steps",
                   help="Regression target: 'steps' (steps_to_go, current production) "
                        "or 'seconds' (continuous time_to_go, Finding 1's candidate).")
    p.add_argument("--n-splits", type=int, default=5,
                   help="GroupKFold folds for transform-selection CV.")
    p.add_argument("--importance-threshold", type=float, default=0.01,
                   help="Feature importance threshold for reduction (scout model).")
    p.add_argument("--sig-threshold-pct", type=float, default=1.5,
                   help="Minimum RMSE improvement (%%) over identity to adopt a transform.")
    p.add_argument("--n-estimators", type=int, default=50,
                   help="Trees for the scout/comparison fits — cheaper than the "
                        "production model since only relative ranking matters here.")
    p.add_argument("--max-depth", type=int, default=15)
    p.add_argument("--min-samples-leaf", type=int, default=10)
    p.add_argument("--lag-steps", type=int, default=5, help="Step lag for delta_atd.")
    p.add_argument("--window", type=int, default=10,
                   help="Rolling window for cumabs_cte and heading_volatility.")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    et_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=args.random_state,
    )

    print(f"Loading data from: {args.data}")
    _raw_df, model_df, _iaf_ref, runway_encoder, all_runways = prepare_modelling_features(
        args.data, args.lag_steps, args.window
    )
    print(f"  {len(model_df):,} modelling rows  |  {model_df['episode'].nunique():,} episodes  |  {len(all_runways)} runways")

    X_full, y_full, feat_names = build_feature_matrix(model_df, runway_encoder, target=args.target)
    print(f"  Feature matrix: {X_full.shape}  (target={args.target!r})")

    print("\nRunning feature reduction (scout fit) ...")
    col_indices, kept_names = reduce_features(
        X_full, y_full, feat_names, args.importance_threshold, et_params
    )
    X_reduced = X_full[:, col_indices]

    print("\nSelecting best target transform ...")
    transform_name, tf_metrics = select_best_transform(
        X_reduced, y_full,
        groups=model_df["episode"],
        n_splits=args.n_splits,
        sig_threshold_pct=args.sig_threshold_pct,
        et_params=et_params,
    )

    artifact = {
        "source_data": str(args.data),
        "n_rows": int(len(model_df)),
        "n_episodes": int(model_df["episode"].nunique()),
        "target": args.target,
        "feature_names_kept": kept_names,
        "col_indices": col_indices,
        "transform_name": transform_name,
        "transform_metrics": tf_metrics,
        "lag_steps": args.lag_steps,
        "window": args.window,
        "importance_threshold": args.importance_threshold,
        "sig_threshold_pct": args.sig_threshold_pct,
        "selection_n_estimators": args.n_estimators,
        "random_state": args.random_state,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(artifact, f, sort_keys=False)

    print(f"\nSelection saved -> {args.output.resolve()}")
    print(f"  Features kept ({len(kept_names)}): {kept_names}")
    print(f"  Transform: {transform_name!r}")


if __name__ == "__main__":
    main()
