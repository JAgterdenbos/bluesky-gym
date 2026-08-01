"""
cps_coordination/testing/surrogate_data.py
--------------------------------------------
Shared data-loading / feature-engineering pipeline for the ETASurrogate
training tools (:mod:`select_surrogate_features` and :mod:`train_surrogate`).

Kept in one place so the two scripts can never silently drift apart on how
raw rollout parquet rows become the canonical 14-column feature matrix —
whatever feature-importance/transform decision ``select_surrogate_features``
makes is guaranteed to be reproducible by ``train_surrogate`` because both
call the exact same functions.

Not used by ``surrogate_analyse.py``, which intentionally keeps its own
independent (plotting-oriented) copy of this pipeline — left untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    SPEED,
)
from cps_coordination.coordination.eta_surrogate import (
    ALL_FEATURE_NAMES,
    cartesian_to_polar,
    decompose_heading,
)

MAX_TIME: float = float(_ENV_MAX_TIME)   # seconds (6 h), from env definition


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

    Returns the full dataset (including terminal states where
    ``steps_to_go == 0``); ``t`` is un-normalised to seconds.
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

    df["t"] = df["t"] * MAX_TIME
    df["steps_to_go"] = (
        df.groupby("episode")["step"].transform("max") - df["step"]
    )
    df["time_to_go"] = df.groupby("episode")["t"].transform("max") - df["t"]
    return df


# ---------------------------------------------------------------------------
# 2. IAF reference — derived from PathPlanningGoalEnv constants
# ---------------------------------------------------------------------------

def compute_iaf_reference_from_env(
    runways: List[str],
) -> Dict[str, Tuple[float, float, float]]:
    """Compute exact IAF positions and approach headings from env constants.

    Uses the same geometry as ``PathPlanningGoalEnv._compute_goal_vector`` so
    training features are identical to the values produced by the env at
    runtime. No estimation from rollout data is needed.

    Returns ``{runway_id: (iaf_x, iaf_y, approach_heading_rad)}`` where
    ``iaf_x``/``iaf_y`` are normalised by ``MAX_DISTANCE`` (same scale as the
    ``x``, ``y`` columns in the training parquet).
    """
    iaf_ref: Dict[str, Tuple[float, float, float]] = {}

    for rwy_id in runways:
        rwy_info = RUNWAYS_SCHIPHOL_FAF[rwy_id]

        iaf_lat, iaf_lon = get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE + IAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        qdr_deg, dis_nm = _kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], iaf_lat, iaf_lon)
        qdr_rad = np.radians(qdr_deg)
        dis_km = dis_nm * NM2KM

        iaf_x = float(np.sin(qdr_rad) * (dis_km / MAX_DISTANCE))
        iaf_y = float(np.cos(qdr_rad) * (dis_km / MAX_DISTANCE))
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
    ``heading_error``, ``naive_eta_remaining``.
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
    # Physical lower bound on remaining flight time (seconds) -- same
    # straight-line-at-cruise-speed formula as eta_surrogate.py's inference-time
    # computation and coordination_baseline._estimate_naive_eta.
    df["naive_eta_remaining"] = np.hypot(dx, dy) * MAX_DISTANCE * 1000.0 / SPEED

    return df


def engineer_target_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``remaining_time_budget`` — the goal-conditioned policy's own
    active temporal target minus elapsed time, i.e. ``rta * MAX_TIME - t``.

    Mirrors the equivalent inference-time signal (``info["goal_vector"][2]``)
    that ``CPSManager`` already has available; see Finding 2 of the
    ETASurrogate accuracy plan. Requires ``rta`` and ``t`` (seconds) columns.
    """
    df = df.copy()
    df["remaining_time_budget"] = (df["rta"] * MAX_TIME) - df["t"]
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

    New columns: ``delta_atd``, ``cumabs_cte``, ``heading_volatility``.
    Original row order is preserved.
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
        diff = (diff + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
        return diff.abs().rolling(window, min_periods=1).mean()

    df["heading_volatility"] = grouped["heading"].transform(_heading_volatility)

    return df.loc[orig_order]


# ---------------------------------------------------------------------------
# 5. Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    runway_encoder: LabelEncoder,
    target: Literal["steps", "seconds"] = "steps",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Assemble the full 15-column feature matrix in canonical order.

    Uses ``cartesian_to_polar``/``decompose_heading`` from ``eta_surrogate``
    to guarantee training/inference consistency. Heading values from parquet
    are in bearing radians; converted to degrees before ``decompose_heading``
    (which expects degrees).

    ``target`` selects ``y``: ``"steps"`` (default, current behavior) uses
    ``steps_to_go``; ``"seconds"`` uses the continuous ``time_to_go``.

    Returns ``(X, y, feature_names)`` — ``X`` shape ``(n, 15)``, columns 14
    and 15 being ``remaining_time_budget`` (requires
    ``engineer_target_time_feature`` to have run on ``df``) and
    ``naive_eta_remaining`` (requires ``engineer_geometric_features``), and
    ``feature_names`` is ``ALL_FEATURE_NAMES``.
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
    rem_budget = df["remaining_time_budget"].to_numpy(dtype=float)
    naive_remaining = df["naive_eta_remaining"].to_numpy(dtype=float)

    X = np.column_stack([
        r_arr, theta_arr, rwy_codes, elapsed,
        sin_psi, cos_psi, r_sq,
        atd, cte, h_err,
        d_atd, c_cte, h_vol,
        rem_budget, naive_remaining,
    ])
    y = (
        df["steps_to_go"].to_numpy(dtype=float)
        if target == "steps"
        else df["time_to_go"].to_numpy(dtype=float)
    )

    return X, y, list(ALL_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def et_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def prepare_modelling_features(
    data_path: Path,
    lag_steps: int,
    window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float, float]], LabelEncoder, List[str]]:
    """Run steps 1-5 end to end: load, derive IAF ref, engineer features.

    Returns ``(raw_df, model_df, iaf_ref, runway_encoder, all_runways)``.
    ``raw_df`` (unfiltered by ``steps_to_go > 0``) is returned too since
    fold-isolated diagnostics need to re-derive per-fold features from it.
    """
    raw_df = load_and_prepare(data_path)
    all_runways = sorted(raw_df["runway"].unique())
    runway_encoder = LabelEncoder().fit(all_runways)
    iaf_ref = compute_iaf_reference_from_env(all_runways)

    model_df = raw_df[raw_df["steps_to_go"] > 0].dropna(subset=["steps_to_go"]).copy()
    model_df = engineer_geometric_features(model_df, iaf_ref)
    model_df = add_lag_features(model_df, lag_steps, window)
    model_df = engineer_target_time_feature(model_df)

    return raw_df, model_df, iaf_ref, runway_encoder, all_runways
