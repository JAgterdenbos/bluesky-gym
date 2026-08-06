"""
cps_coordination/scripts/cps_metrics_offline.py
--------------------------------------------------
Recompute CPS coordination metrics (Gamma, Gamma_r, C_sep, Delta epsilon vs.
static/uncoordinated, R_rec, rho_ripple) plus spatial tortuosity/entropy/KL
divergence from logged Parquet telemetry (roadmap step 8), decoupled from
collection time so metric definitions can be revised without re-running M
episodes.

Reuses the three pure aggregate-metric helper functions from
``cps_coordination.experiments.metrics`` (not ``coordination_baseline`` --
the BlueSky-heavy experiment class -- see that module's own docstring for
the Phase D.4 split rationale). Note ``cps_coordination/__init__.py``
eagerly imports ``CPSCoordinationExperiment`` (and therefore bluesky) for
any ``cps_coordination.*`` import regardless of submodule, so this script
is not actually import-light in practice despite reaching into the leaner
module -- a pre-existing package-structure gap, not something introduced
by the Phase D.4 split.

Usage
-----
  python cps_coordination/scripts/cps_metrics_offline.py --save-path experiments/cps_eval/manual_run
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from cps_coordination.experiments.metrics import (
    _compute_separation_compliance,
    _compute_throughput,
    _lag1_autocorrelation,
)
from path_planning.rta.testing.spatial_visitation_analysis import (
    build_heatmap,
    compute_information_metrics,
    compute_tortuosity,
)

_DEFAULT_RECAT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "cps_base.yaml")
)


def load_recat_matrix(config_path: str = _DEFAULT_RECAT_CONFIG) -> Dict[str, Dict[str, float]]:
    """Load the RECAT-EU matrix from ``cps_base.yaml``'s ``recat_eu`` key.

    Deliberately duplicates the few lines of
    ``CPSCoordinationExperiment._load_recat_matrix`` rather than
    instantiating a full experiment just to read a YAML file — this script's
    entire point is to avoid the heavy bluesky/gym/SB3 import chain that
    class pulls in.
    """
    if os.path.exists(config_path):
        with open(config_path) as fh:
            data = yaml.safe_load(fh) or {}
        matrix = data.get("recat_eu", {})
        if matrix:
            return {
                lead: {trail: float(v) for trail, v in row.items()}
                for lead, row in matrix.items()
            }
    cats = ["A", "B", "C", "D", "E", "F"]
    return {lead: {trail: 90.0 for trail in cats} for lead in cats}


def load_telemetry(save_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two Parquet streams written by ``run_cps_eval.py``."""
    aircraft_df = pd.read_parquet(os.path.join(save_path, "cps_eval_aircraft.parquet"))
    separation_path = os.path.join(save_path, "cps_eval_separation.parquet")
    separation_df = (
        pd.read_parquet(separation_path)
        if os.path.exists(separation_path)
        else pd.DataFrame(columns=["episode_id", "runway_id", "acid_lead", "acid_trail",
                                    "gap_actual_s", "required_sep_s"])
    )
    return aircraft_df, separation_df


def recompute_separation_compliance(
    separation_df: pd.DataFrame, tolerance_s: float = 5.0
) -> float:
    """C_sep from precomputed per-pair gaps — recomputable at any tolerance
    without re-deriving pairs from landing times."""
    if separation_df.empty:
        return float("nan")
    compliant = separation_df["gap_actual_s"] >= (separation_df["required_sep_s"] - tolerance_s)
    return float(compliant.mean())


def recompute_metrics(
    aircraft_df: pd.DataFrame,
    separation_df: pd.DataFrame,
    recat_matrix: Dict[str, Dict[str, float]],
    sep_tolerance_s: float = 5.0,
    rta_tolerance_s: float = 60.0,  # δ_t (Eq. recovery_rate) — matches
                                    # coordination_baseline.RTA_TOLERANCE_SEC;
                                    # hardcoded rather than imported to keep
                                    # this script's lightweight import chain
                                    # (see module docstring).
) -> Dict[str, Any]:
    """Mirror of ``CPSMetricsReporter.compute_aggregate_metrics``,
    reading from logged Parquet DataFrames instead of in-memory
    ``_EpisodeRecord`` objects."""
    if aircraft_df.empty:
        return {"error": "no records"}

    n_aircraft = len(aircraft_df)
    success_rate = float(aircraft_df["success"].mean())

    successful = aircraft_df[aircraft_df["success"]]
    # Pooled by runway_id alone: correct for throughput (a landing count over
    # the total observed time span), but NOT valid for separation-compliance
    # pairwise gaps -- see landing_times_by_rwy_episode below.
    landing_times_by_rwy: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
    # Separation compliance must never compare landing times across
    # different episodes' independent simulation clocks -- each episode
    # restarts its clock near zero at env.reset(), so "consecutive landing
    # pair" is only meaningful within a single (runway, episode).
    landing_times_by_rwy_episode: Dict[Tuple[str, int], List[Tuple[float, str]]] = (
        defaultdict(list)
    )
    for _, row in successful.iterrows():
        landing_times_by_rwy[row["runway_id"]].append(
            (float(row["actual_landing_time"]), row["acid"])
        )
        landing_times_by_rwy_episode[(row["runway_id"], row["episode_id"])].append(
            (float(row["actual_landing_time"]), row["acid"])
        )

    total_time_s = float(successful["actual_landing_time"].max()) if not successful.empty else 3600.0
    window_h = max(total_time_s / 3600.0, 1e-6)
    gamma, gamma_r = _compute_throughput(landing_times_by_rwy, window_h)

    wake_cats = dict(zip(aircraft_df["acid"], aircraft_df["wake_cat"]))
    c_sep_from_pairs = recompute_separation_compliance(separation_df, sep_tolerance_s)
    # Cross-check against the from-scratch derivation used at collection
    # time (same landing-time-sorted-pairs logic, episode-scoped) — should
    # agree exactly since both come from the same underlying landings.
    c_sep_from_landings = _compute_separation_compliance(
        landing_times_by_rwy_episode, wake_cats, recat_matrix, tolerance_s=sep_tolerance_s,
    )

    # Two distinct Δε comparisons -- see metrics.py::compute_aggregate_metrics
    # for the full rationale. `vs_static` is the literal Eq. tracking_degradation
    # (RQ2.2); `vs_uncoordinated` is a secondary, honestly-labelled reference
    # against an uncoordinated run under the same frozen Worker, NOT Groot et
    # al.'s published data. `rta_error_static` is only present in telemetry
    # collected after the static-TTA addition -- older Parquet files (e.g.
    # a sweep collected pre-addition) simply have no `vs_static` metric.
    if "rta_error_static" in aircraft_df:
        delta_eps_static_values = (
            aircraft_df["rta_error_cps"].abs() - aircraft_df["rta_error_static"].abs()
        ).dropna()
        delta_epsilon_vs_static = (
            float(delta_eps_static_values.mean()) if len(delta_eps_static_values) else float("nan")
        )
    else:
        delta_epsilon_vs_static = float("nan")
    delta_eps_uncoord_values = (
        aircraft_df["rta_error_cps"].abs() - aircraft_df["rta_error_solo"].abs()
    ).dropna()
    delta_epsilon_vs_uncoordinated = (
        float(delta_eps_uncoord_values.mean()) if len(delta_eps_uncoord_values) else float("nan")
    )

    # --- R_rec (Eq. recovery_rate): M_update = mid-trajectory-updated
    # aircraft; recovered = landed within rta_tolerance_s of that TTA.
    updated = aircraft_df[aircraft_df["tta_updated_mid_trajectory"]]
    r_rec = (
        float((updated["rta_error_cps"].abs() <= rta_tolerance_s).mean())
        if len(updated) else float("nan")
    )

    sorted_success = successful.sort_values("actual_landing_time")
    rho_ripple = _lag1_autocorrelation(list(sorted_success["rta_error_cps"]))

    # --- Stall metrics (pre-Step-10 audit §1.3) -- mirrors
    # CPSMetricsReporter.compute_aggregate_metrics's split.
    # `stall_rate` (bare plateau-detection rate) is kept only as a
    # diagnostic/before-after comparison value, NOT the headline risk
    # metric -- see `stall_unrecovered`/`stall_recovery_rate` below.
    if "stall_detected" in aircraft_df:
        stall_detected_col = aircraft_df["stall_detected"]
        success_col = aircraft_df["success"]
        n_stall_detected = int(stall_detected_col.sum())
        n_stall_recovered = int((stall_detected_col & success_col).sum())
        n_stall_unrecovered = int((stall_detected_col & ~success_col).sum())
        stall_rate = n_stall_detected / n_aircraft
        stall_recovered = n_stall_recovered / n_aircraft
        stall_unrecovered = n_stall_unrecovered / n_aircraft
        stall_recovery_rate = (
            n_stall_recovered / n_stall_detected if n_stall_detected > 0 else float("nan")
        )
    else:
        stall_rate = stall_recovered = stall_unrecovered = stall_recovery_rate = float("nan")

    return {
        "n_episodes": int(aircraft_df["episode_id"].nunique()),
        "n_aircraft": n_aircraft,
        "success_rate": round(success_rate, 4),
        "gamma": round(gamma, 4),
        "gamma_r": {rwy: round(v, 4) for rwy, v in gamma_r.items()},
        "c_sep": round(c_sep_from_pairs, 4) if not np.isnan(c_sep_from_pairs) else "nan",
        "c_sep_from_landings_crosscheck": (
            round(float(c_sep_from_landings), 4) if not np.isnan(c_sep_from_landings) else "nan"
        ),
        "delta_epsilon_vs_static": (
            round(delta_epsilon_vs_static, 4) if not np.isnan(delta_epsilon_vs_static) else "nan"
        ),
        "delta_epsilon_vs_uncoordinated": (
            round(delta_epsilon_vs_uncoordinated, 4)
            if not np.isnan(delta_epsilon_vs_uncoordinated) else "nan"
        ),
        "r_rec": round(r_rec, 4) if not np.isnan(r_rec) else "nan",
        "rho_ripple": round(rho_ripple, 4) if not np.isnan(rho_ripple) else "nan",
        "stall_unrecovered": round(stall_unrecovered, 4) if not np.isnan(stall_unrecovered) else "nan",
        "stall_recovery_rate": round(stall_recovery_rate, 4) if not np.isnan(stall_recovery_rate) else "nan",
        "stall_recovered": round(stall_recovered, 4) if not np.isnan(stall_recovered) else "nan",
        "stall_rate": round(stall_rate, 4) if not np.isnan(stall_rate) else "nan",  # diagnostic only
    }


def explode_trajectories(aircraft_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format point cloud: one row per (episode_id, acid, trajectory point).

    ``episode`` is a compound ``f"{episode_id}_{acid}"`` key so
    :func:`compute_tortuosity` (which groups strictly by an ``"episode"``
    column) groups by individual aircraft trajectory, not by whole
    multi-aircraft episode.
    """
    if aircraft_df.empty:
        return pd.DataFrame(columns=["episode", "x", "y"])

    df = aircraft_df[["episode_id", "acid", "traj_x", "traj_y"]].copy()
    df["episode"] = df["episode_id"].astype(str) + "_" + df["acid"].astype(str)
    df = df.explode(["traj_x", "traj_y"], ignore_index=True)
    df = df.dropna(subset=["traj_x", "traj_y"])
    df = df.rename(columns={"traj_x": "x", "traj_y": "y"})
    return df[["episode", "x", "y"]].astype({"x": float, "y": float})


def recompute_spatial_metrics(aircraft_df: pd.DataFrame, bins: int = 200) -> Dict[str, Any]:
    """Tortuosity + Shannon entropy of the successful-aircraft trajectory
    point cloud, via the ported ``spatial_visitation_analysis.py`` functions."""
    successful = aircraft_df[aircraft_df["success"]]
    point_cloud = explode_trajectories(successful)
    if point_cloud.empty:
        return {"tortuosity_mean": "nan", "entropy_bits": "nan"}

    tortuosity_mean = compute_tortuosity(point_cloud)
    _, _, _, _, _, H = build_heatmap(point_cloud, bins=bins)
    entropy_bits, _, _ = compute_information_metrics(H)

    return {
        "tortuosity_mean": round(float(tortuosity_mean), 4),
        "entropy_bits": round(float(entropy_bits), 4),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute CPS coordination metrics offline from logged Parquet telemetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--save-path", type=str, required=True,
                   help="Directory containing cps_eval_aircraft.parquet / cps_eval_separation.parquet.")
    p.add_argument("--sep-tolerance-s", type=float, default=5.0, help="C_sep tolerance (seconds).")
    p.add_argument("--rta-tolerance-s", type=float, default=60.0,
                   help="R_rec tolerance (seconds) — Eq. recovery_rate's delta_t.")
    p.add_argument("--bins", type=int, default=200, help="2D histogram bins for the spatial heatmap.")
    p.add_argument("--recat-config", type=str, default=_DEFAULT_RECAT_CONFIG,
                   help="Path to cps_base.yaml (for the recat_eu matrix).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    aircraft_df, separation_df = load_telemetry(args.save_path)
    recat_matrix = load_recat_matrix(args.recat_config)

    metrics = recompute_metrics(
        aircraft_df, separation_df, recat_matrix,
        sep_tolerance_s=args.sep_tolerance_s, rta_tolerance_s=args.rta_tolerance_s,
    )
    spatial = recompute_spatial_metrics(aircraft_df, bins=args.bins)
    combined = {**metrics, **spatial}

    print("\n--- Offline CPS Coordination Metrics ---")
    for k, v in combined.items():
        print(f"  {k:<32}: {v}")

    out_path = os.path.join(args.save_path, "cps_metrics_offline.yaml")
    with open(out_path, "w") as fh:
        yaml.dump(combined, fh, default_flow_style=False, sort_keys=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
