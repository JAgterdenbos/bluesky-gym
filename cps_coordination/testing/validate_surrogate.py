"""
cps_coordination/testing/validate_surrogate.py
-------------------------------------------------
Single validation gate for ETASurrogate accuracy work
(see .claude/eta_surrogate_accuracy_plan.md, Phase B). Thin orchestration
only -- no new feature engineering: reuses surrogate_data.py for the
canonical feature pipeline, surrogate_analyse.py's diagnostic plots, and
diagnose_success_rate.py's condition 3 for the true end-to-end gate.

Two parts, run in order:

  1. Held-out cross-validation replicating the surrogate's exact shipped
     recipe (feature columns, target -- ``steps`` vs continuous ``seconds``
     -- target transform, lag params, ET hyperparameters -- all read off the
     loaded ``ETASurrogate`` itself, not re-specified here). The serialized
     .pkl is trained on 100% of the data and has no OOF predictions of its
     own, so this is the closest honest held-out proxy for it. Predictions
     (transformed-space) are inverse-transformed and scaled by
     ``surrogate.sim_dt`` -- exactly how production's ``predict_eta`` does it
     -- then compared against the continuous ``time_to_go`` ground truth
     (never ``steps_to_go*ACTION_TIME`` on both sides), so the Finding-1
     label-noise gap is visible instead of hidden by using the same biased
     conversion on both sides.
  2. ``diagnose_success_rate.py`` condition 3 (single-aircraft,
     tta_mode="solo", real surrogate, zero separation pressure) run against
     the actual serialized artifact through the real simulator -- the
     authoritative, deployment-realistic gate.

Held-out numbers from (1) are never sufficient on their own: the whole
reason this investigation started is that held-out metrics already looked
reasonable while deployment error stayed high (Finding 1). Condition 3 is
the actual gate every candidate must clear.

Usage
-----
  python cps_coordination/testing/validate_surrogate.py \\
      --surrogate cps_coordination/models/eta_surrogate.pkl \\
      path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet

  # Held-out metrics only, skip the simulator-driven condition-3 gate:
  python cps_coordination/testing/validate_surrogate.py --skip-condition3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from cps_coordination.coordination.eta_surrogate import (
    ETASurrogate,
    TRANSFORMS,
    cartesian_to_polar,
)
from cps_coordination.testing.surrogate_data import (
    build_feature_matrix,
    et_metrics,
    prepare_modelling_features,
)
from cps_coordination.testing.surrogate_analyse import (
    _apply_style,
    _tolerance_ratio_str,
    plot_error_by_horizon,
    plot_error_map,
    plot_feature_vs_error,
    plot_residual_analysis,
)

_DEFAULT_SURROGATE = Path("cps_coordination/models/eta_surrogate.pkl")
_DEFAULT_DATA = Path(
    "path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet"
)
_DEFAULT_OUT_DIR = Path(__file__).parent.parent / "figures" / "validate"


# ---------------------------------------------------------------------------
# 1. Held-out CV on the exact shipped recipe
# ---------------------------------------------------------------------------

def _et_params_from(model: ExtraTreesRegressor) -> dict:
    return dict(
        n_estimators=model.n_estimators,
        max_depth=model.max_depth,
        min_samples_leaf=model.min_samples_leaf,
        max_features=model.max_features,
        n_jobs=-1,
        random_state=model.random_state,
    )


def run_holdout_cv(
    data_path: Path,
    surrogate: ETASurrogate,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fresh GroupKFold CV replicating the surrogate's exact recipe.

    Uses the model's own self-described feature columns, target (``steps`` vs
    ``seconds`` -- ``surrogate._target``), target transform, lag params, and
    ET hyperparameters -- read directly off *surrogate* -- via
    surrogate_data.py's shared pipeline. No independent feature engineering
    happens here.

    Returns ``(oof_df, y_true_sec, y_pred_sec)`` where ``y_true_sec`` is the
    continuous ``time_to_go`` ground truth (regardless of which target the
    model itself was fit on) and ``y_pred_sec`` is the model's raw output,
    inverse-transformed and scaled by ``surrogate.sim_dt`` -- exactly how
    production's ``predict_eta`` converts it (``sim_dt=ACTION_TIME`` for a
    ``steps`` target, ``sim_dt=1.0`` for a ``seconds`` target that already
    predicts seconds directly).
    """
    et_params = _et_params_from(surrogate._model)
    fwd_fn, inv_fn = TRANSFORMS[surrogate._transform_name]
    col_indices = surrogate._feature_col_indices

    _raw_df, model_df, _iaf_ref, runway_encoder, _all_runways = prepare_modelling_features(
        data_path, surrogate._lag_steps, surrogate._window,
    )
    X_full, y_fit, _names = build_feature_matrix(
        model_df, runway_encoder, target=surrogate._target
    )
    X_reduced = X_full[:, col_indices]
    unit = "steps" if surrogate._target == "steps" else "s"

    gkf = GroupKFold(n_splits=n_splits)
    oof_idx: List[np.ndarray] = []
    oof_pred: List[np.ndarray] = []

    print(f"\n{n_splits}-Fold Group Cross-Validation (exact shipped recipe: "
          f"{len(col_indices)} features, target={surrogate._target!r}, "
          f"transform={surrogate._transform_name!r})")
    for fold_idx, (tr_idx, va_idx) in enumerate(
        gkf.split(X_reduced, groups=model_df["episode"])
    ):
        model = ExtraTreesRegressor(**et_params).fit(
            X_reduced[tr_idx], fwd_fn(y_fit[tr_idx])
        )
        y_pred = inv_fn(model.predict(X_reduced[va_idx]))
        oof_idx.append(va_idx)
        oof_pred.append(y_pred)
        m = et_metrics(y_fit[va_idx], y_pred)
        print(f"  Fold {fold_idx + 1}/{n_splits}: R2={m['R2']:.4f}  "
              f"MAE={m['MAE']:.2f} {unit}  RMSE={m['RMSE']:.2f} {unit}")

    order = np.concatenate(oof_idx)
    y_pred_oof = np.concatenate(oof_pred)
    oof_df = model_df.iloc[order].reset_index(drop=True)

    y_true_sec = oof_df["time_to_go"].to_numpy(dtype=float)
    y_pred_sec = y_pred_oof * surrogate.sim_dt
    return oof_df, y_true_sec, y_pred_sec


def print_per_runway_metrics(
    df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    print("\nPer-runway held-out metrics (seconds, vs continuous time_to_go)")
    print(f"{'Runway':<8} {'R2':>8} {'MAE':>10} {'RMSE':>10}")
    print("-" * 40)
    for rwy in sorted(df["runway"].unique()):
        mask = (df["runway"] == rwy).to_numpy()
        m = et_metrics(y_true[mask], y_pred[mask])
        print(f"{rwy:<8} {m['R2']:>8.4f} {m['MAE']:>10.1f} {m['RMSE']:>10.1f}")


# ---------------------------------------------------------------------------
# 2. End-to-end gate -- diagnose_success_rate.py condition 3
# ---------------------------------------------------------------------------

def run_condition_3_gate(surrogate: ETASurrogate, n_episodes: int) -> list:
    """Run the real frozen worker through diagnose_success_rate.py's
    condition 3: multi-agent env, tta_mode="solo", real surrogate, N=1,
    zero separation pressure. The authoritative end-to-end check -- held-out
    metrics alone are not sufficient (Finding 1: they looked fine once while
    deployment error stayed high).
    """
    from cps_coordination.testing.diagnose_success_rate import (
        _make_experiment,
        _summarize,
        run_multi_agent_condition,
    )

    experiment = _make_experiment(k_cps=0, mode="static", runways=None)
    model = experiment.make_model(experiment._make_multi_agent_env(1))
    records = run_multi_agent_condition(
        experiment, model, n_episodes, n_aircraft=1,
        tta_mode="solo", surrogate=surrogate, k_cps=0,
    )
    _summarize(
        "Condition 3 (validate_surrogate gate): multi-agent, solo TTA, "
        "real ETASurrogate, N=1, zero separation pressure",
        records,
    )
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation gate for a serialized ETASurrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", nargs="?", type=Path, default=_DEFAULT_DATA,
                         help="Path to rollout parquet (or CSV) for the held-out CV.")
    parser.add_argument("--surrogate", type=Path, default=_DEFAULT_SURROGATE,
                         help="Path to the serialized ETASurrogate to validate.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=30,
                         help="Episodes for the condition-3 end-to-end gate.")
    parser.add_argument("--skip-condition3", action="store_true",
                         help="Skip the simulator-driven condition-3 gate; "
                              "held-out metrics only.")
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    args = parser.parse_args()

    print(f"Loading surrogate from: {args.surrogate}")
    surrogate = ETASurrogate.load(args.surrogate)
    print(f"  {surrogate!r}")

    print(f"\nLoading data from: {args.data}")
    oof_df, y_true_sec, y_pred_sec = run_holdout_cv(args.data, surrogate, args.n_splits)

    metrics = et_metrics(y_true_sec, y_pred_sec)
    print(f"\nHeld-out seconds metrics (model output x sim_dt={surrogate.sim_dt} "
          f"vs continuous time_to_go ground truth):")
    print(f"  R2={metrics['R2']:.4f}  MAE={metrics['MAE']:.1f}s  RMSE={metrics['RMSE']:.1f}s")
    print(f"  MAE = {_tolerance_ratio_str(metrics['MAE'])}")
    print_per_runway_metrics(oof_df, y_true_sec, y_pred_sec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()
    r_arr, theta_arr = cartesian_to_polar(
        oof_df["x"].to_numpy(dtype=float), oof_df["y"].to_numpy(dtype=float)
    )
    oof_df["r"] = r_arr
    oof_df["theta"] = theta_arr

    print("\nGenerating diagnostic figures ...")
    plot_error_map(oof_df, y_true_sec, y_pred_sec, args.out_dir)
    plot_residual_analysis(y_true_sec, y_pred_sec, args.out_dir)
    plot_error_by_horizon(y_true_sec, y_pred_sec, args.out_dir)
    plot_feature_vs_error(oof_df, y_true_sec, y_pred_sec, args.out_dir)
    print(f"  Figures saved to {args.out_dir.resolve()}")

    if args.skip_condition3:
        print("\n--skip-condition3 set: held-out metrics only, no end-to-end gate run.")
        return

    print(f"\nRunning condition-3 end-to-end gate ({args.episodes} episodes) ...")
    run_condition_3_gate(surrogate, args.episodes)


if __name__ == "__main__":
    main()
