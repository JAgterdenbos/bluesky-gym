"""
cps_coordination/testing/train_surrogate.py
--------------------------------------------
End-to-end training script for the self-describing ETASurrogate.

Runs the full champion feature-engineering and model-selection pipeline —
identical in methodology to ``surrogate_analyse.py`` — and serialises a
production-ready :class:`ETASurrogate` that carries its own feature recipe
(IAF anchors, surviving column indices, target transform).

Pipeline
--------
1.  Load parquet data, filter successful episodes, compute steps_to_go.
2.  Derive exact IAF anchors from PathPlanningGoalEnv constants (no estimation).
3.  Engineer static geometric features (ATD, CTE, heading_error, r_sq).
4.  Engineer lag features (delta_atd, cumabs_cte, heading_volatility).
5.  Build the full 13-column feature matrix; encode runway as integer.
6.  5-fold group cross-validation (fold-isolated lag + reduction)
    → report OOF metrics for baseline inspection.
7.  Global feature reduction via scout ExtraTreesRegressor.
8.  Select best target transform (identity / log1p / sqrt) by OOF RMSE.
9.  Fit final model on the full reduced dataset with the winning transform.
10. Package into ETASurrogate via ETASurrogate.from_training().
11. Save to disk with joblib.

Usage
-----
  python cps_coordination/testing/train_surrogate.py \\
      path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet \\
      --output cps_coordination/models/eta_surrogate.pkl

  # All options:
  python cps_coordination/testing/train_surrogate.py data.parquet \\
      --output models/eta_surrogate.pkl \\
      --n-splits 5 \\
      --importance-threshold 0.01 \\
      --sig-threshold-pct 1.5 \\
      --n-estimators 200 \\
      --max-depth 15 \\
      --min-samples-leaf 10 \\
      --lag-steps 5 \\
      --window 10 \\
      --sim-dt 5.0 \\
      --random-state 42
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from bluesky_gym.envs.common.functions import get_point_at_distance
from bluesky_gym.envs.pathplanning_goal_env import (
    FAF_DISTANCE,
    IAF_DISTANCE,
    MAX_DISTANCE,
    MAX_TIME as _ENV_MAX_TIME,
    NM2KM,
    RUNWAYS_SCHIPHOL_FAF,
    SCHIPHOL,
)
from cps_coordination.coordination.eta_surrogate import (
    ALL_FEATURE_NAMES,
    ETASurrogate,
    TRANSFORMS,
    _LAG_COL_START,
    cartesian_to_polar,
    decompose_heading,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TIME: float = float(_ENV_MAX_TIME)   # seconds (6 h), from env definition

_DEFAULT_OUTPUT = Path("cps_coordination/models/eta_surrogate.pkl")


# ---------------------------------------------------------------------------
# Geo helpers (pure numpy — no bluesky init required)
# ---------------------------------------------------------------------------

def _kwikqdrdist(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """Flat-earth bearing [deg, 0=N clockwise] and distance [nm] from A to B.

    Matches the BlueSky ``kwikqdrdist`` implementation exactly so coordinate
    values are identical to those produced by the env at runtime.
    """
    re = 6_371_000.0  # Earth radius [m]
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    cavelat = np.cos(np.radians(0.5 * (lat1 + lat2)))
    dangle = np.sqrt(dlat ** 2 + (dlon * cavelat) ** 2)
    dist_nm = re * dangle / 1852.0
    qdr_deg = float(np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360)
    return qdr_deg, float(dist_nm)


# ---------------------------------------------------------------------------
# 1. Data loading and preparation
# ---------------------------------------------------------------------------

def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load parquet (or CSV), filter successful episodes, compute steps_to_go.

    Parameters
    ----------
    path : Path
        Parquet or CSV file path.

    Returns
    -------
    pd.DataFrame
        Full dataset (including terminal states where steps_to_go == 0).
        ``t`` column is un-normalised to seconds.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = (
        pd.read_parquet(path, engine="pyarrow")
        if path.suffix == ".parquet"
        else pd.read_csv(path)
    )

    if "is_success" in df.columns:
        df = df[df["is_success"]].copy()
    else:
        df = df.copy()

    # Un-normalise time to seconds
    df["t"] = df["t"] * _MAX_TIME

    # steps_to_go: 0 at the terminal state (IAF crossing)
    df["steps_to_go"] = (
        df.groupby("episode")["step"].transform("max") - df["step"]
    )

    return df


# ---------------------------------------------------------------------------
# 2. IAF reference — derived from PathPlanningGoalEnv constants
# ---------------------------------------------------------------------------

def compute_iaf_reference_from_env(
    runways: List[str],
) -> Dict[str, Tuple[float, float, float]]:
    """Compute exact IAF positions and approach headings from env constants.

    Uses the same geometry as ``PathPlanningGoalEnv._compute_goal_vector`` so
    training features are identical to the values used at env runtime.  No
    estimation from rollout data is needed.

    The IAF for each runway is placed ``FAF_DISTANCE + IAF_DISTANCE`` km behind
    the runway threshold at bearing ``track - 180°``.  Its normalised (x, y)
    position is computed relative to Schiphol, matching the env's observation
    encoding.  The approach heading is the runway track converted to bearing
    radians.

    Parameters
    ----------
    runways : list[str]
        Runway identifiers present in the training data.

    Returns
    -------
    dict
        ``{runway_id: (iaf_x, iaf_y, approach_heading_rad)}``.
        ``iaf_x``, ``iaf_y`` are normalised by ``MAX_DISTANCE`` (same scale as
        the ``x``, ``y`` columns in the training parquet).
    """
    iaf_ref: Dict[str, Tuple[float, float, float]] = {}

    for rwy_id in runways:
        rwy_info = RUNWAYS_SCHIPHOL_FAF[rwy_id]

        # IAF position: FAF_DISTANCE + IAF_DISTANCE km behind the runway at
        # bearing (track - 180°), matching _compute_goal_vector exactly.
        iaf_lat, iaf_lon = get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE + IAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        # Convert to the env's normalised (x, y) coordinate frame.
        qdr_deg, dis_nm = _kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], iaf_lat, iaf_lon)
        qdr_rad = np.radians(qdr_deg)
        dis_km  = dis_nm * NM2KM

        iaf_x = float(np.sin(qdr_rad) * (dis_km / MAX_DISTANCE))
        iaf_y = float(np.cos(qdr_rad) * (dis_km / MAX_DISTANCE))

        # Approach heading: aircraft approaches the runway flying track direction.
        approach_heading_rad = float(np.radians(rwy_info["track"]))

        iaf_ref[rwy_id] = (iaf_x, iaf_y, approach_heading_rad)

    return iaf_ref


# ---------------------------------------------------------------------------
# 3. Static geometric feature engineering
# ---------------------------------------------------------------------------

def engineer_geometric_features(
    df: pd.DataFrame,
    iaf_ref: Dict[str, Tuple[float, float, float]],
) -> pd.DataFrame:
    """Attach IAF-relative geometric features, fully vectorised.

    New columns: ``r_sq``, ``along_track_dist``, ``cross_track_error``,
    ``heading_error``.

    Parameters
    ----------
    df : pd.DataFrame
        Modelling rows (steps_to_go > 0).
    iaf_ref : dict
        ``{runway_id: (iaf_x, iaf_y, approach_heading_rad)}``, as returned by
        :func:`compute_iaf_reference_from_env`.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["r_sq"] = df["x"] ** 2 + df["y"] ** 2

    missing = set(df["runway"].unique()) - set(iaf_ref.keys())
    if missing:
        warnings.warn(
            f"IAF reference missing for runways {missing}; "
            "geometric features will be NaN for those rows.",
            stacklevel=2,
        )

    iaf_x = df["runway"].map({r: v[0] for r, v in iaf_ref.items()}).to_numpy(dtype=float)
    iaf_y = df["runway"].map({r: v[1] for r, v in iaf_ref.items()}).to_numpy(dtype=float)
    ah    = df["runway"].map({r: v[2] for r, v in iaf_ref.items()}).to_numpy(dtype=float)

    dx = df["x"].to_numpy(dtype=float) - iaf_x
    dy = df["y"].to_numpy(dtype=float) - iaf_y
    sin_ah = np.sin(ah)
    cos_ah = np.cos(ah)

    df["along_track_dist"] = dx * sin_ah + dy * cos_ah
    df["cross_track_error"] = dx * cos_ah - dy * sin_ah
    df["heading_error"] = (df["heading"].to_numpy(dtype=float) - ah + np.pi) % (
        2.0 * np.pi
    ) - np.pi

    return df


# ---------------------------------------------------------------------------
# 4. Lag feature engineering
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    lag_steps: int = 5,
    window: int = 10,
) -> pd.DataFrame:
    """Append macro-historical lag features, grouped strictly by episode.

    Mirrors the ``add_lag_features`` implementation in ``surrogate_analyse.py``
    so training and analysis results are directly comparable.

    New columns: ``delta_atd``, ``cumabs_cte``, ``heading_volatility``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``[along_track_dist, cross_track_error, heading, episode, step]``.
    lag_steps : int
        Step lag for ``delta_atd`` (default 5).
    window : int
        Rolling window for ``cumabs_cte`` and ``heading_volatility`` (default 10).

    Returns
    -------
    pd.DataFrame
        Original row order preserved.
    """
    df = df.copy()
    orig_order = df.index

    df = df.sort_values(["episode", "step"])
    grouped = df.groupby("episode", sort=False)

    df["delta_atd"] = grouped["along_track_dist"].transform(
        lambda s: (s - s.shift(lag_steps)).bfill()
    )
    df["cumabs_cte"] = grouped["cross_track_error"].transform(
        lambda s: s.abs().rolling(window, min_periods=1).sum()
    )

    def _heading_volatility(s: pd.Series) -> pd.Series:
        diff = s.diff().bfill()
        diff = (diff + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [−π, π]
        return diff.abs().rolling(window, min_periods=1).mean()

    df["heading_volatility"] = grouped["heading"].transform(_heading_volatility)

    return df.loc[orig_order]


# ---------------------------------------------------------------------------
# 5. Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    runway_encoder: LabelEncoder,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Assemble the full 13-column feature matrix in canonical order.

    Uses ``cartesian_to_polar`` and ``decompose_heading`` from
    ``eta_surrogate`` to guarantee training / inference consistency.
    Heading values from parquet are in bearing radians; they are converted
    to degrees before passing to ``decompose_heading`` (which expects degrees).

    Parameters
    ----------
    df : pd.DataFrame
        Modelling rows with all engineered columns present.
    runway_encoder : LabelEncoder
        Fitted encoder for runway identifiers.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 13)
    y : np.ndarray, shape (n_samples,)
        Target = ``steps_to_go``.
    feature_names : list[str]
        ``ALL_FEATURE_NAMES`` (the canonical 13-column list).
    """
    x_arr = df["x"].to_numpy(dtype=float)
    y_arr = df["y"].to_numpy(dtype=float)
    heading_rad = df["heading"].to_numpy(dtype=float)   # bearing radians in parquet
    heading_deg = np.rad2deg(heading_rad)

    r_arr, theta_arr = cartesian_to_polar(x_arr, y_arr)
    sin_psi, cos_psi = decompose_heading(heading_deg)

    rwy_codes = runway_encoder.transform(df["runway"]).astype(float)
    elapsed = df["step"].to_numpy(dtype=float)
    r_sq = df["r_sq"].to_numpy(dtype=float)
    atd = df["along_track_dist"].to_numpy(dtype=float)
    cte = df["cross_track_error"].to_numpy(dtype=float)
    h_err = df["heading_error"].to_numpy(dtype=float)
    d_atd = df["delta_atd"].to_numpy(dtype=float)
    c_cte = df["cumabs_cte"].to_numpy(dtype=float)
    h_vol = df["heading_volatility"].to_numpy(dtype=float)

    X = np.column_stack([
        r_arr, theta_arr, rwy_codes, elapsed,
        sin_psi, cos_psi, r_sq,
        atd, cte, h_err,
        d_atd, c_cte, h_vol,
    ])
    y = df["steps_to_go"].to_numpy(dtype=float)

    return X, y, list(ALL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# 6. Feature reduction
# ---------------------------------------------------------------------------

def reduce_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    threshold: float,
    et_params: dict,
) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """Drop features below an importance threshold using a preliminary ET fit.

    Returns
    -------
    X_reduced : np.ndarray, shape (n_train, n_kept)
    mask : np.ndarray, shape (n_features,)  boolean
    col_indices : list[int]
        Indices of surviving columns within the full 13-column vector.
    kept_names : list[str]
    """
    scout = ExtraTreesRegressor(**et_params).fit(X_train, y_train)
    importances = scout.feature_importances_
    mask = importances >= threshold

    # Lag features share a semantic group; drop or keep all three together
    lag_idx = list(range(_LAG_COL_START, len(feature_names)))
    if lag_idx:
        lag_total_imp = importances[lag_idx].sum()
        keep_lag = lag_total_imp >= threshold
        for i in lag_idx:
            mask[i] = keep_lag

    col_indices = [i for i, keep in enumerate(mask) if keep]
    kept_names = [feature_names[i] for i in col_indices]
    dropped = [feature_names[i] for i in range(len(feature_names)) if not mask[i]]

    print(
        f"    Feature reduction: {mask.sum()}/{len(feature_names)} kept"
        + (f"  |  dropped: {dropped}" if dropped else "  |  none dropped")
    )
    return X_train[:, mask], mask, col_indices, kept_names


# ---------------------------------------------------------------------------
# 7. Cross-validation (for reporting only — fold-isolated pipeline)
# ---------------------------------------------------------------------------

def _et_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R²":   float(r2_score(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def cross_validate(
    full_df: pd.DataFrame,
    iaf_ref: Dict[str, Tuple[float, float, float]],
    runway_encoder: LabelEncoder,
    n_splits: int,
    lag_steps: int,
    window: int,
    importance_threshold: float,
    et_params: dict,
) -> None:
    """5-fold group cross-validation.

    IAF anchors are derived from env constants (exact, same across all folds).
    Lag features are still computed per-fold on the train split only, so the
    val fold receives features built from its own episode history.

    Prints per-fold and OOF metrics.  Diagnostic reporting only — no artefacts
    are returned for use in the final model.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_y_true: List[np.ndarray] = []
    oof_y_pred: List[np.ndarray] = []

    print(f"\n{n_splits}-Fold Group Cross-Validation (identity transform, baseline)")
    print(f"  Total rows: {len(full_df):,}  |  episodes: {full_df['episode'].nunique():,}")

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(full_df, groups=full_df["episode"])
    ):
        raw_train = full_df.iloc[train_idx]
        raw_val   = full_df.iloc[val_idx]

        train_m = raw_train[raw_train["steps_to_go"] > 0].dropna(subset=["steps_to_go"])
        val_m   = raw_val[raw_val["steps_to_go"] > 0].dropna(subset=["steps_to_go"])

        # IAF reference is exact (from env constants) — no fold isolation needed.
        train_m = engineer_geometric_features(train_m, iaf_ref)
        val_m   = engineer_geometric_features(val_m, iaf_ref)
        train_m = add_lag_features(train_m, lag_steps, window)
        val_m   = add_lag_features(val_m, lag_steps, window)

        X_tr, y_tr, feat_names = build_feature_matrix(train_m, runway_encoder)
        X_va, y_va, _          = build_feature_matrix(val_m, runway_encoder)

        print(f"\n  Fold {fold_idx + 1}/{n_splits} ...", end="  ")
        _, mask, _, _ = reduce_features(X_tr, y_tr, feat_names, importance_threshold, et_params)
        X_tr_r = X_tr[:, mask]
        X_va_r = X_va[:, mask]

        model  = ExtraTreesRegressor(**et_params).fit(X_tr_r, y_tr)
        y_pred = model.predict(X_va_r)

        oof_y_true.append(y_va)
        oof_y_pred.append(y_pred)

        m = _et_metrics(y_va, y_pred)
        print(
            f"  Fold {fold_idx + 1}: "
            f"R²={m['R²']:.4f}  MAE={m['MAE']:.2f} steps  RMSE={m['RMSE']:.2f} steps"
        )

    y_oof      = np.concatenate(oof_y_true)
    y_pred_oof = np.concatenate(oof_y_pred)
    m_oof      = _et_metrics(y_oof, y_pred_oof)

    print(f"\n  OOF Summary (baseline)")
    print(f"  {'Metric':<8} {'Value':>10}")
    print("  " + "-" * 20)
    print(f"  {'R²':<8} {m_oof['R²']:>10.4f}")
    print(f"  {'MAE':<8} {m_oof['MAE']:>10.2f} steps")
    print(f"  {'RMSE':<8} {m_oof['RMSE']:>10.2f} steps")


# ---------------------------------------------------------------------------
# 8. Transform selection
# ---------------------------------------------------------------------------

def select_best_transform(
    X_reduced: np.ndarray,
    y: np.ndarray,
    all_episodes: pd.Series,
    n_splits: int,
    sig_threshold_pct: float,
    et_params: dict,
) -> Tuple[str, object, object, Dict[str, float]]:
    """Select the best target transform (identity / log1p / sqrt) by OOF RMSE.

    Runs cross-validation on the already-reduced feature matrix so no
    additional feature selection is needed per fold.

    Parameters
    ----------
    X_reduced : np.ndarray, shape (n, n_selected)
    y : np.ndarray, shape (n,)
        Target in original (step) space.
    all_episodes : pd.Series
        Episode IDs for each row (used as groups for GroupKFold).
    n_splits : int
    sig_threshold_pct : float
        Minimum RMSE improvement (% over identity) to adopt a non-identity
        transform.
    et_params : dict

    Returns
    -------
    transform_name : str
    fwd_fn : Callable
    inv_fn : Callable
    metrics : dict
    """
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n  {'Transform':<12} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
    print("  " + "-" * 46)

    results = []
    for name, (fwd, inv) in TRANSFORMS.items():
        oof_y: List[np.ndarray] = []
        oof_p: List[np.ndarray] = []
        for tr_idx, va_idx in gkf.split(X_reduced, groups=all_episodes):
            m = ExtraTreesRegressor(**et_params).fit(X_reduced[tr_idx], fwd(y[tr_idx]))
            oof_y.append(y[va_idx])
            oof_p.append(inv(m.predict(X_reduced[va_idx])))

        y_oof = np.concatenate(oof_y)
        p_oof = np.concatenate(oof_p)
        metrics = _et_metrics(y_oof, p_oof)
        results.append({"name": name, "fwd": fwd, "inv": inv, "metrics": metrics})
        print(
            f"  {name:<12} {metrics['R²']:>10.4f} "
            f"{metrics['MAE']:>10.2f} {metrics['RMSE']:>10.2f}"
        )

    identity_rmse = results[0]["metrics"]["RMSE"]
    best = min(results, key=lambda r: r["metrics"]["RMSE"])
    improvement = 100.0 * (identity_rmse - best["metrics"]["RMSE"]) / identity_rmse

    print(f"\n  Improvement of {best['name']!r} over identity: {improvement:+.2f}%")
    winner = best if improvement >= sig_threshold_pct else results[0]
    if improvement >= sig_threshold_pct:
        print(f"  Decision: significant — using [{winner['name']}]")
    else:
        print(f"  Decision: below threshold ({sig_threshold_pct:.1f}%) — using [identity]")

    return winner["name"], winner["fwd"], winner["inv"], winner["metrics"]


# ---------------------------------------------------------------------------
# 9. Final model fit + surrogate assembly
# ---------------------------------------------------------------------------

def fit_and_assemble(
    X_reduced: np.ndarray,
    y: np.ndarray,
    fwd_fn: object,
    inv_fn: object,
    transform_name: str,
    col_indices: List[int],
    feature_names_kept: List[str],
    iaf_ref_dict: Dict[str, Tuple[float, float, float]],
    runway_encoder: LabelEncoder,
    lag_steps: int,
    window: int,
    sim_dt: float,
    et_params: dict,
) -> ETASurrogate:
    """Fit the final model and package it into an ETASurrogate.

    Parameters
    ----------
    X_reduced : np.ndarray, shape (n, n_selected)
        Full dataset, reduced feature matrix.
    y : np.ndarray, shape (n,)
        Targets in original step space.
    fwd_fn, inv_fn : Callable
        Forward / inverse transform functions (module-level, picklable).
    transform_name : str
    col_indices : list[int]
        Surviving column indices into the full 13-column vector.
    feature_names_kept : list[str]
    iaf_ref_dict : dict
    runway_encoder : LabelEncoder
    lag_steps, window : int
    sim_dt : float
    et_params : dict

    Returns
    -------
    ETASurrogate
    """
    print(f"\nFitting final model on {len(y):,} samples ...")
    model = ExtraTreesRegressor(**et_params).fit(X_reduced, fwd_fn(y))
    print("  Done.")

    needs_lag = any(
        name in feature_names_kept
        for name in ["delta_atd", "cumabs_cte", "heading_volatility"]
    )

    surrogate = ETASurrogate.from_training(
        model=model,
        iaf_ref=iaf_ref_dict,
        runway_encoder=runway_encoder,
        feature_names=feature_names_kept,
        col_indices=col_indices,
        needs_lag=needs_lag,
        lag_steps=lag_steps,
        window=window,
        sim_dt=sim_dt,
        transform_name=transform_name,
        inv_transform=inv_fn,
    )
    return surrogate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train and serialise an ETASurrogate from rollout parquet data. "
            "The feature set is determined entirely by importance-based reduction "
            "of the full 13-column champion feature set."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "data",
        type=Path,
        help="Path to rollout parquet (or CSV) file.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination path for the serialised ETASurrogate.",
    )
    p.add_argument("--n-splits", type=int, default=5,
                   help="Number of GroupKFold folds for cross-validation.")
    p.add_argument("--importance-threshold", type=float, default=0.01,
                   help="Feature importance threshold for reduction (scout model).")
    p.add_argument("--sig-threshold-pct", type=float, default=1.5,
                   help="Minimum RMSE improvement (%%) over identity to adopt a transform.")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=15)
    p.add_argument("--min-samples-leaf", type=int, default=10)
    p.add_argument("--lag-steps", type=int, default=5,
                   help="Step lag for delta_atd.")
    p.add_argument("--window", type=int, default=10,
                   help="Rolling window for cumabs_cte and heading_volatility.")
    p.add_argument("--sim-dt", type=float, default=5.0,
                   help="Simulation timestep in seconds.")
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

    # ── 1. Load ─────────────────────────────────────────────────────────
    print(f"Loading data from: {args.data}")
    df = load_and_prepare(args.data)
    all_runways = sorted(df["runway"].unique())
    print(
        f"  {(df['steps_to_go'] > 0).sum():,} modelling rows  ·  "
        f"{df['episode'].nunique():,} episodes  ·  "
        f"{len(all_runways)} runways"
    )
    print(f"  Runways: {all_runways}")

    # ── Runway encoder ───────────────────────────────────────────────────
    runway_encoder = LabelEncoder().fit(all_runways)

    # ── 2. IAF reference — exact values from env constants ───────────────
    print("\nDeriving IAF reference from PathPlanningGoalEnv constants ...")
    iaf_ref = compute_iaf_reference_from_env(all_runways)
    for rwy, (ix, iy, ah) in iaf_ref.items():
        print(f"  {rwy}: iaf=({ix:.4f}, {iy:.4f}), approach_hdg={np.rad2deg(ah):.1f}°")

    # ── 3–4. Feature engineering (full dataset for final model) ──────────
    print("\nEngineering features ...")
    model_df = df[df["steps_to_go"] > 0].dropna(subset=["steps_to_go"]).copy()
    model_df = engineer_geometric_features(model_df, iaf_ref)
    model_df = add_lag_features(model_df, args.lag_steps, args.window)
    print(f"  Modelling rows after engineering: {len(model_df):,}")

    # ── 5. Build full feature matrix ─────────────────────────────────────
    X_full, y_full, feat_names = build_feature_matrix(model_df, runway_encoder)
    print(f"  Feature matrix: {X_full.shape}")

    # ── 6. Cross-validation (baseline reporting) ─────────────────────────
    cross_validate(
        df, iaf_ref, runway_encoder,
        args.n_splits, args.lag_steps, args.window,
        args.importance_threshold, et_params,
    )

    # ── 7. Global feature reduction ───────────────────────────────────────
    print("\nRunning global feature reduction ...")
    X_reduced, _, col_indices, feature_names_kept = reduce_features(
        X_full, y_full, feat_names, args.importance_threshold, et_params
    )
    print(f"  Reduced feature matrix: {X_reduced.shape}")
    print(f"  Surviving features: {feature_names_kept}")

    # ── 8. Transform selection ────────────────────────────────────────────
    print("\nSelecting best target transform ...")
    transform_name, fwd_fn, inv_fn, tf_metrics = select_best_transform(
        X_reduced, y_full,
        all_episodes=model_df["episode"],
        n_splits=args.n_splits,
        sig_threshold_pct=args.sig_threshold_pct,
        et_params=et_params,
    )
    print(
        f"\n  Winner: {transform_name!r}  "
        f"R²={tf_metrics['R²']:.4f}  "
        f"MAE={tf_metrics['MAE']:.2f} steps  "
        f"RMSE={tf_metrics['RMSE']:.2f} steps"
    )

    # ── 9. Fit final model + assemble surrogate ───────────────────────────
    surrogate = fit_and_assemble(
        X_reduced=X_reduced,
        y=y_full,
        fwd_fn=fwd_fn,
        inv_fn=inv_fn,
        transform_name=transform_name,
        col_indices=col_indices,
        feature_names_kept=feature_names_kept,
        iaf_ref_dict=iaf_ref,
        runway_encoder=runway_encoder,
        lag_steps=args.lag_steps,
        window=args.window,
        sim_dt=args.sim_dt,
        et_params=et_params,
    )

    print(f"\nSurrogate summary:")
    print(f"  {surrogate!r}")
    print(f"  Needs lag features: {surrogate._needs_lag}")
    print(f"  State layout      : {surrogate.state_layout}")

    # ── 10. Save ──────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    surrogate.save(args.output)
    print(f"\nSurrogate saved → {args.output.resolve()}")

    # ── Quick round-trip check ────────────────────────────────────────────
    loaded = ETASurrogate.load(args.output)
    test_state = np.array([0.3, -0.4, 120.0, 45.0])
    lag = np.zeros(3)
    test_rwy = all_runways[0]
    eta = loaded.predict_eta(test_state, test_rwy, current_sim_time=0.0, lag_features=lag)
    print(f"  Round-trip check: predict_eta({test_rwy!r}) = {eta:.1f} s  ✓")


if __name__ == "__main__":
    main()
