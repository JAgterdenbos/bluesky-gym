"""
cps_coordination/experiments/metrics.py
----------------------------------------
Aggregate-metric computation, stdout reporting, and disk logging for CPS
coordination evaluation runs — split out of ``CPSCoordinationExperiment``
(pre-Step-10 audit, Phase D.4) so the experiment class owns only the
episode-running/CPS-coordination loop, and this module owns turning a list
of per-aircraft :class:`~cps_coordination.experiments.coordination_baseline._EpisodeRecord`
into the metric table, console printout, and CSV/YAML artifacts.

See ``CPSCoordinationExperiment.evaluate()`` for the only production call
site (construct one :class:`CPSMetricsReporter` per evaluation run and call
``compute_aggregate_metrics`` → ``print_metrics`` → ``save_logs`` in
sequence).

Deliberately does NOT import anything from ``bluesky_gym.envs`` (which
transitively imports ``bluesky`` itself) — ``cps_metrics_offline.py``
reuses this module's helper functions specifically to stay BlueSky-free
(pure Parquet + YAML), so keep this module importable standalone.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

if TYPE_CHECKING:
    from cps_coordination.experiments.coordination_baseline import _EpisodeRecord

# δ_t, the RTA operational tolerance (Eq. recovery_rate / eq:per_group_sr in
# methodology.tex) — same constant that governs the worker's own on-time
# reward and is_success check, NOT the separation-compliance gap tolerance.
# Equal to ``bluesky_gym.envs.pathplanning_goal_env.RTA_TOLERANCE * MAX_TIME``
# (1 minute) -- hardcoded rather than imported since that module pulls in
# bluesky itself (see module docstring) and RTA_TOLERANCE is defined there
# as ``1 * 60 / MAX_TIME``, so multiplying back by MAX_TIME always yields
# exactly 60.0 regardless of MAX_TIME's value.
RTA_TOLERANCE_SEC = 60.0


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────


def _lag1_autocorrelation(series: List[float]) -> float:
    """Compute lag-1 autocorrelation of *series* (ρ_ripple).

    Returns ``float('nan')`` if the series has fewer than 2 elements or
    zero variance, matching standard behaviour for undefined autocorrelation.

    Implements: ρ_1 = Cov(x_t, x_{t-1}) / Var(x_t)  (Pearson).

    Parameters
    ----------
    series : List[float]
        Ordered sequence of values (e.g. per-aircraft RTA errors within
        an episode, sorted by landing time).

    Returns
    -------
    float
        Lag-1 autocorrelation in ``[-1, 1]``, or NaN.
    """
    if len(series) < 2:
        return float("nan")
    arr = np.asarray(series, dtype=float)
    if np.std(arr) == 0.0:
        return float("nan")
    # Pearson correlation between consecutive pairs
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def _compute_separation_compliance(
    landing_times: Dict[Tuple[str, int], List[Tuple[float, str]]],
    wake_cats: Dict[str, str],
    recat_matrix: Dict[str, Dict[str, float]],
    tolerance_s: float = 5.0,
) -> float:
    """Compute C_sep: fraction of consecutive pairs meeting RECAT-EU separation.

    Within each ``(runway_id, episode_id)`` group, consecutive landings
    (sorted by time) are checked against the RECAT-EU matrix. A pair is
    *compliant* if the observed gap is ≥ (required_separation − tolerance_s).

    Grouping by ``(runway_id, episode_id)`` — not ``runway_id`` alone — is
    required: ``actual_landing_time`` is local to each episode's own
    simulation clock (every episode's clock restarts near zero at
    ``env.reset()``), so pooling landing times across independent episodes
    under a shared runway key would treat unrelated aircraft from different
    simulation runs as if they were sequential landings on one timeline.

    Parameters
    ----------
    landing_times : Dict[Tuple[str, int], List[Tuple[float, str]]]
        ``{(runway_id, episode_id): [(landing_time, acid), ...]}`` —
        unsorted within each group is fine.
    wake_cats : Dict[str, str]
        ``{acid: wake_turbulence_category}`` mapping.
    recat_matrix : Dict[str, Dict[str, float]]
        RECAT-EU separation matrix (seconds).
    tolerance_s : float
        Compliance slack in seconds (default 5 s).

    Returns
    -------
    float
        C_sep ∈ [0, 1].
    """
    n_pairs = 0
    n_compliant = 0

    for _group_key, times_and_acids in landing_times.items():
        if len(times_and_acids) < 2:
            continue
        sorted_pairs: List[Tuple[float, str]] = sorted(
            times_and_acids, key=lambda x: x[0]  # type: ignore[index]
        )
        for i in range(1, len(sorted_pairs)):
            t_prev, acid_prev = sorted_pairs[i - 1]  # type: ignore[misc]
            t_curr, acid_curr = sorted_pairs[i]       # type: ignore[misc]
            gap = t_curr - t_prev
            lead_cat = wake_cats.get(acid_prev, "C")
            trail_cat = wake_cats.get(acid_curr, "C")
            required = recat_matrix.get(lead_cat, {}).get(trail_cat, 90.0)
            n_pairs += 1
            if gap >= (required - tolerance_s):
                n_compliant += 1

    return (n_compliant / n_pairs) if n_pairs > 0 else float("nan")


def _compute_throughput(
    landing_times: Dict[str, List[Tuple[float, str]]],
    window_h: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute total throughput Γ and per-runway throughput Γ_r.

    Parameters
    ----------
    landing_times : Dict[str, List[Tuple[float, str]]]
        ``{runway_id: [(landing_time_s, acid), ...]}``
    window_h : float
        Observation window in hours.

    Returns
    -------
    gamma : float
        Total landings per hour.
    gamma_r : Dict[str, float]
        Per-runway landings per hour.
    """
    total = sum(len(v) for v in landing_times.values())
    gamma = total / window_h

    gamma_r: Dict[str, float] = {
        rwy: len(lts) / window_h for rwy, lts in landing_times.items()
    }
    return gamma, gamma_r


# ──────────────────────────────────────────────────────────────────────────────
# CPSMetricsReporter
# ──────────────────────────────────────────────────────────────────────────────


class CPSMetricsReporter:
    """Computes, prints, and saves aggregate CPS coordination metrics.

    Stateless aside from the tolerances/output path fixed at construction —
    ``compute_aggregate_metrics`` takes the full per-aircraft record list
    and RECAT-EU matrix as arguments and returns a plain metric dict, so it
    can be called standalone (e.g. from a regression test building a small
    synthetic record list) without a full ``CPSCoordinationExperiment``.

    Parameters
    ----------
    save_path : str, optional
        Output directory for :meth:`save_logs`. Only required if
        :meth:`save_logs` is actually called (``compute_aggregate_metrics``/
        ``print_metrics`` don't need it).
    separation_tolerance_s : float
        C_sep compliance slack, seconds (default 5.0, matches the prior
        ``cps_eval.separation_tolerance_s`` config default).
    rta_tolerance_s : float
        δ_t for R_rec's on-time check, seconds (default ``RTA_TOLERANCE_SEC``).
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        separation_tolerance_s: float = 5.0,
        rta_tolerance_s: float = RTA_TOLERANCE_SEC,
    ) -> None:
        self.save_path = save_path
        self.separation_tolerance_s = separation_tolerance_s
        self.rta_tolerance_s = rta_tolerance_s

    def compute_aggregate_metrics(
        self,
        records: List["_EpisodeRecord"],
        recat_matrix: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Compute all CPS metrics from the full set of episode records.

        Metrics computed
        ----------------
        gamma           : Total throughput (landings/hour).
        gamma_r         : Per-runway throughput (landings/hour).
        c_sep           : Separation compliance fraction.
        delta_epsilon_vs_static : Tracking degradation (Eq. tracking_degradation,
                          RQ2.2's literal metric): mean |RTA_error_CPS| −
                          |RTA_error_static|, dynamic replanning vs. the same
                          greedy-scheduled TTA assigned once and frozen.
        delta_epsilon_vs_uncoordinated : Secondary metric: mean |RTA_error_CPS| −
                          |RTA_error_solo|, CPS-coordinated vs. an uncoordinated
                          reference run under the identical frozen Worker. NOT
                          Groot et al.'s published data -- a locally-generated
                          reference, kept for internal comparison only.
        r_rec           : Recovery success rate.
        rho_ripple      : Delay ripple index (lag-1 autocorrelation of RTA errors).
        stall_unrecovered : Fraction of aircraft flagged stalled AND that never
                          landed -- the actually-costly subset, and the headline
                          risk metric to report alongside success_rate (pre-Step-10
                          audit §1.3; replaces bare stall_rate as the headline).
        stall_recovery_rate : Of aircraft flagged stalled, the fraction that still
                          landed successfully -- mitigation-effectiveness
                          diagnostic (compare before/after CPSModelConfig.
                          enable_stall_detection or the Test A/B/C ablations).
        stall_recovered : Fraction of all aircraft flagged stalled AND successful
                          (the complement of stall_unrecovered within stall_rate).
        stall_rate      : Fraction of aircraft flagged as stalled by CPSManager
                          (distance-to-IAF not shrinking over a rolling window --
                          see cps_manager.py's STALL_WINDOW_S). Diagnostic ONLY --
                          answers "did progress plateau," not "did it fail" (an
                          aircraft can legitimately stall during path-stretching
                          and still converge). Kept for before/after ablation
                          comparisons; not the headline risk metric.
        n_episodes      : Total episodes evaluated.
        n_aircraft      : Total aircraft evaluated.
        success_rate    : Fraction of successful landings.

        Parameters
        ----------
        records : List[_EpisodeRecord]
            All per-aircraft records from all evaluation episodes.
        recat_matrix : Dict[str, Dict[str, float]]
            RECAT-EU separation matrix for C_sep calculation.

        Returns
        -------
        Dict[str, Any]
            Mapping of metric name → value.
        """
        if not records:
            return {"error": "no records collected"}

        n_aircraft = len(records)
        success_rate = sum(r.success for r in records) / n_aircraft

        # --- Throughput ---
        # Approximate: count landings over total sim time span. Pooled across
        # episodes by runway_id alone is correct here -- throughput is just a
        # landing count over the total observed time span, not a pairwise
        # adjacent-landing comparison, so it isn't sensitive to different
        # episodes' clocks being independent (unlike separation compliance
        # below, which is).
        landing_times_by_rwy: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        # Separation compliance must never compare landing times across
        # different episodes' independent simulation clocks -- each episode
        # restarts its clock near zero at env.reset(), so "consecutive
        # landing pair" is only meaningful within a single (runway, episode).
        landing_times_by_rwy_episode: Dict[Tuple[str, int], List[Tuple[float, str]]] = (
            defaultdict(list)
        )
        for rec in records:
            if rec.success:
                landing_times_by_rwy[rec.runway_id].append(
                    (rec.actual_landing_time, rec.acid)
                )
                landing_times_by_rwy_episode[(rec.runway_id, rec.episode_id)].append(
                    (rec.actual_landing_time, rec.acid)
                )

        total_time_s = max(
            (rec.actual_landing_time for rec in records if rec.success),
            default=3600.0,
        )
        window_h = max(total_time_s / 3600.0, 1e-6)
        gamma, gamma_r = _compute_throughput(landing_times_by_rwy, window_h)

        # --- Separation compliance ---
        wake_cats = {rec.acid: rec.wake_cat for rec in records}
        c_sep = _compute_separation_compliance(
            landing_times_by_rwy_episode,
            wake_cats,
            recat_matrix,
            tolerance_s=self.separation_tolerance_s,
        )

        # --- Tracking degradation Δε ---
        # Two distinct comparisons (see docstring above): the literal
        # Eq. tracking_degradation (cps vs. static-TTA, RQ2.2's actual
        # question about the cost of replanning) and a secondary,
        # honestly-labelled uncoordinated-reference comparison (cps vs.
        # solo) that is NOT Groot et al.'s published data.
        delta_eps_static_values = [
            abs(rec.rta_error_cps) - abs(rec.rta_error_static)
            for rec in records
            if not np.isnan(rec.rta_error_static)
        ]
        delta_epsilon_vs_static = (
            float(np.mean(delta_eps_static_values)) if delta_eps_static_values else float("nan")
        )
        delta_eps_uncoord_values = [
            abs(rec.rta_error_cps) - abs(rec.rta_error_solo)
            for rec in records
            if not np.isnan(rec.rta_error_solo)
        ]
        delta_epsilon_vs_uncoordinated = (
            float(np.mean(delta_eps_uncoord_values)) if delta_eps_uncoord_values else float("nan")
        )

        # --- Recovery success rate R_rec (Eq. recovery_rate) ---
        # M_update = aircraft that received a genuine mid-trajectory TTA
        # update (not just their initial assignment); recovered = landed
        # within delta_t of that TTA despite the update.
        updated_records = [rec for rec in records if rec.tta_updated_mid_trajectory]
        r_rec = (
            sum(
                1 for rec in updated_records
                if not np.isnan(rec.rta_error_cps) and abs(rec.rta_error_cps) <= self.rta_tolerance_s
            ) / len(updated_records)
            if updated_records
            else float("nan")
        )

        # --- Delay ripple index ρ_ripple ---
        # Sort records by landing time to form the arrival sequence
        sorted_records = sorted(
            (rec for rec in records if rec.success),
            key=lambda r: r.actual_landing_time,
        )
        rta_error_sequence = [rec.rta_error_cps for rec in sorted_records]
        rho_ripple = _lag1_autocorrelation(rta_error_sequence)

        # --- Stall metrics (pre-Step-10 audit §1.3) ---
        # `stall_detected` ("did distance-to-IAF plateau for >=30 min") answers
        # a different question than "did this aircraft fail to land" -- an
        # aircraft can legitimately stall during path-stretching (Assumption
        # 1's designed behaviour) and still converge afterward. `stall_rate`
        # is kept below purely as a diagnostic/before-after comparison value
        # (e.g. against `enable_stall_detection` ablations); it is NOT the
        # headline risk metric -- that's `stall_unrecovered` (the actually-
        # costly subset: stalled AND never landed) alongside `success_rate`,
        # with `stall_recovery_rate` as the mitigation-effectiveness
        # diagnostic. See print_metrics / cps_metrics_offline.py's
        # recompute_metrics for the same split applied to logged telemetry.
        n_stall_detected = sum(rec.stall_detected for rec in records)
        n_stall_recovered = sum(rec.stall_detected and rec.success for rec in records)
        n_stall_unrecovered = sum(rec.stall_detected and not rec.success for rec in records)
        stall_rate = n_stall_detected / n_aircraft
        stall_recovered = n_stall_recovered / n_aircraft
        stall_unrecovered = n_stall_unrecovered / n_aircraft
        stall_recovery_rate = (
            n_stall_recovered / n_stall_detected if n_stall_detected > 0 else float("nan")
        )

        return {
            "n_episodes": len(set(r.episode_id for r in records)),
            "n_aircraft": n_aircraft,
            "success_rate": round(success_rate, 4),
            "gamma": round(gamma, 4),
            "gamma_r": {rwy: round(v, 4) for rwy, v in gamma_r.items()},
            "c_sep": round(float(c_sep), 4) if not np.isnan(c_sep) else "nan",
            "delta_epsilon_vs_static": (
                round(delta_epsilon_vs_static, 4) if not np.isnan(delta_epsilon_vs_static) else "nan"
            ),
            "delta_epsilon_vs_uncoordinated": (
                round(delta_epsilon_vs_uncoordinated, 4)
                if not np.isnan(delta_epsilon_vs_uncoordinated) else "nan"
            ),
            "r_rec": round(r_rec, 4) if not np.isnan(r_rec) else "nan",
            "rho_ripple": round(rho_ripple, 4) if not np.isnan(rho_ripple) else "nan",
            "stall_unrecovered": round(stall_unrecovered, 4),
            "stall_recovery_rate": (
                round(stall_recovery_rate, 4) if not np.isnan(stall_recovery_rate) else "nan"
            ),
            "stall_recovered": round(stall_recovered, 4),
            "stall_rate": round(stall_rate, 4),  # diagnostic only -- see comment above
        }

    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print the aggregate metric table to stdout."""
        print("\n--- CPS Coordination Metrics ---")
        print(f"  Episodes evaluated   : {metrics.get('n_episodes')}")
        print(f"  Aircraft evaluated   : {metrics.get('n_aircraft')}")
        print(f"  Success rate         : {metrics.get('success_rate', 'n/a'):.2%}")
        print(f"  Throughput Γ         : {metrics.get('gamma', 'n/a')} ac/h")
        print(f"  Per-runway Γ_r       : {metrics.get('gamma_r', {})}")
        print(f"  Sep. compliance C_sep: {metrics.get('c_sep', 'n/a')}")
        print(f"  Δε vs. static TTA    : {metrics.get('delta_epsilon_vs_static', 'n/a')} s "
              "(Eq. tracking_degradation, RQ2.2)")
        print(f"  Δε vs. uncoordinated : {metrics.get('delta_epsilon_vs_uncoordinated', 'n/a')} s "
              "(secondary, NOT Groot et al.'s data)")
        print(f"  Recovery rate R_rec  : {metrics.get('r_rec', 'n/a')}")
        print(f"  Ripple index ρ_ripple: {metrics.get('rho_ripple', 'n/a')}")
        print(f"  Stall unrecovered    : {metrics.get('stall_unrecovered', 'n/a')} "
              "(stalled AND never landed -- the costly subset)")
        print(f"  Stall recovery rate  : {metrics.get('stall_recovery_rate', 'n/a')} "
              "(of stalled aircraft, fraction that still landed)")
        print(f"  Stall detected (diag): {metrics.get('stall_rate', 'n/a')} "
              "(plateau flag only, NOT a failure rate -- see stall_unrecovered)")
        print()

    def save_logs(
        self,
        records: List["_EpisodeRecord"],
        metrics: Dict[str, Any],
    ) -> None:
        """Write per-aircraft CSV log and aggregate YAML metrics to disk.

        Outputs
        -------
        ``<save_path>/cps_eval_log.csv``    — one row per aircraft record.
        ``<save_path>/cps_metrics.yaml``    — aggregate metric dict.
        """
        if self.save_path is None:
            raise ValueError("CPSMetricsReporter.save_logs requires save_path to be set")
        os.makedirs(self.save_path, exist_ok=True)

        # Per-aircraft CSV
        csv_path = os.path.join(self.save_path, "cps_eval_log.csv")
        csv_fields = [
            "episode_id", "acid", "flight_id", "runway_id", "wake_cat", "assigned_tta",
            "actual_landing_time", "rta_error_cps", "rta_error_static", "rta_error_solo",
            "tta_updated_mid_trajectory", "success",
        ]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            for rec in records:
                writer.writerow(
                    {
                        "episode_id": rec.episode_id,
                        "acid": rec.acid,
                        "flight_id": rec.flight_id,
                        "runway_id": rec.runway_id,
                        "wake_cat": rec.wake_cat,
                        "assigned_tta": rec.assigned_tta,
                        "actual_landing_time": rec.actual_landing_time,
                        "rta_error_cps": rec.rta_error_cps,
                        "rta_error_static": rec.rta_error_static,
                        "rta_error_solo": rec.rta_error_solo,
                        "tta_updated_mid_trajectory": rec.tta_updated_mid_trajectory,
                        "success": rec.success,
                    }
                )
        print(f"Episode log saved → {csv_path}")

        # Aggregate metrics YAML
        yaml_path = os.path.join(self.save_path, "cps_metrics.yaml")
        with open(yaml_path, "w") as fh:
            yaml.dump(
                {"timestamp": datetime.now().isoformat(), **metrics},
                fh,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"Aggregate metrics saved → {yaml_path}")
