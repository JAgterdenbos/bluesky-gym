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


def _grouped_lag1_autocorrelation(
    group_ids: np.ndarray,
    order_keys: np.ndarray,
    values: np.ndarray,
) -> Tuple[float, float]:
    """Mean and sample std, across groups, of per-group lag-1
    autocorrelations (ρ_ripple), fully vectorized.

    Equivalent to sorting each group by ``order_keys``, computing the
    Pearson lag-1 autocorrelation once per group, then averaging the
    non-NaN results — but via a single lexsort + ``np.add.reduceat`` pass
    instead of a Python-level loop over groups (episode counts run into the
    thousands across a sweep, so this is evaluated at that scale).

    Lag-1 pairs are only formed between adjacent elements that share the
    same ``group_ids`` entry after sorting — a pair straddling two
    different groups (e.g. two episodes) is dropped, matching the
    per-episode grouping this replaces.

    Parameters
    ----------
    group_ids : np.ndarray
        Grouping key per element (e.g. episode_id).
    order_keys : np.ndarray
        Within-group ordering key (e.g. actual_landing_time).
    values : np.ndarray
        Values to autocorrelate (e.g. rta_error_cps).

    Returns
    -------
    Tuple[float, float]
        (mean, std) of per-group Pearson lag-1 autocorrelations in
        ``[-1, 1]``. Both NaN if no group yields a non-degenerate pair;
        std alone is NaN with fewer than 2 such groups.
    """
    if len(values) < 2:
        return float("nan"), float("nan")

    order = np.lexsort((order_keys, group_ids))
    g = np.asarray(group_ids)[order]
    v = np.asarray(values, dtype=float)[order]

    same_group = g[:-1] == g[1:]
    if not np.any(same_group):
        return float("nan"), float("nan")

    x = v[:-1][same_group]
    y = v[1:][same_group]
    g_pairs = g[:-1][same_group]  # already sorted, contiguous per group

    _, starts = np.unique(g_pairs, return_index=True)
    counts = np.diff(np.append(starts, len(x)))

    # Mean-centered Pearson formula (what a per-group np.corrcoef call
    # actually computes), not the algebraically-equivalent-but-unstable
    # raw sum-of-squares formula: the latter subtracts two large near-equal
    # sums for a (near-)constant group and can round to a spuriously
    # nonzero variance. Centering reduces that roundoff from eps scale to
    # eps^2 scale, but even centering isn't a full guarantee: ``sum/count``
    # doesn't always round back to exactly the constant value, and since
    # that rounding bias is then *uniform* across every element of a
    # constant group, it can still cancel to exactly rho=+-1 instead of
    # NaN (verified: reproducible with real float64 values, not just
    # theoretical).
    sum_x = np.add.reduceat(x, starts)
    sum_y = np.add.reduceat(y, starts)
    mean_x = np.repeat(sum_x / counts, counts)
    mean_y = np.repeat(sum_y / counts, counts)
    dx = x - mean_x
    dy = y - mean_y

    cov = np.add.reduceat(dx * dy, starts)
    # Clip residual negative roundoff (variance is never negative) so a
    # near-zero-variance group reliably falls into the denom == 0 (=> NaN)
    # branch below rather than two negative roundoff errors multiplying
    # into a spurious positive denom.
    var_x = np.maximum(np.add.reduceat(dx * dx, starts), 0.0)
    var_y = np.maximum(np.add.reduceat(dy * dy, starts), 0.0)
    denom = np.sqrt(var_x * var_y)

    # Belt-and-braces exact check: a group's x (or y) pair-values are
    # mathematically constant iff every element equals the group's first
    # element -- a direct equality comparison, immune to the subtraction/
    # cancellation error the arithmetic above is still subject to. Forces
    # true zero-variance groups to NaN regardless of what floating-point
    # roundoff the centered formula computed for them.
    first_x = np.repeat(x[starts], counts)
    first_y = np.repeat(y[starts], counts)
    x_constant = np.add.reduceat((x != first_x).astype(np.int64), starts) == 0
    y_constant = np.add.reduceat((y != first_y).astype(np.int64), starts) == 0
    degenerate = x_constant | y_constant

    with np.errstate(invalid="ignore", divide="ignore"):
        per_group_rho = np.where((denom > 0) & ~degenerate, cov / denom, np.nan)

    valid = per_group_rho[~np.isnan(per_group_rho)]
    if valid.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if valid.size >= 2 else float("nan")
    return mean, std


def _episode_ratio_mean_std(
    episode_ids: np.ndarray,
    indicator: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Mean and sample std, across episodes, of each episode's mean of
    ``indicator`` (optionally restricted to ``mask``).

    Backs the ``<metric>_std`` companion figures for the pooled/global
    ratio metrics (Δε, R_rec, stall metrics) — the episode-to-episode
    variance behind an aggregate that's otherwise reported as a single
    pooled number. Does NOT change any primary metric's value; callers
    keep computing their pooled aggregate exactly as before and only
    derive this std from the same per-episode partition. Vectorized via
    sort + ``np.add.reduceat`` (same idiom as
    :func:`_grouped_lag1_autocorrelation`) instead of a Python-level loop
    over episodes.

    Parameters
    ----------
    episode_ids : np.ndarray
        Episode id per record.
    indicator : np.ndarray
        Per-record value to average within each episode (e.g. an on-time
        0/1 indicator, or a per-record Δε value).
    mask : np.ndarray, optional
        Boolean population selector (e.g. "received a mid-trajectory TTA
        update", or "flagged stalled"). Records outside the mask don't
        contribute to any episode's ratio at all — unlike a NaN left in
        ``indicator``, which would still count toward that episode's
        denominator under a plain per-episode mean.

    Returns
    -------
    Tuple[float, float]
        (mean, std) of per-episode means. Both NaN if no episode has a
        masked record; std alone is NaN with fewer than 2 valid episodes.
    """
    episode_ids = np.asarray(episode_ids)
    indicator = np.asarray(indicator, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        episode_ids = episode_ids[mask]
        indicator = indicator[mask]
    if len(indicator) == 0:
        return float("nan"), float("nan")

    order = np.argsort(episode_ids, kind="stable")
    eids = episode_ids[order]
    vals = indicator[order]
    _, starts = np.unique(eids, return_index=True)
    sums = np.add.reduceat(vals, starts)
    counts = np.diff(np.append(starts, len(vals)))
    with np.errstate(invalid="ignore"):
        per_episode_ratio = sums / counts

    valid = per_episode_ratio[~np.isnan(per_episode_ratio)]
    if valid.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if valid.size >= 2 else float("nan")
    return mean, std


def _compute_separation_compliance(
    landing_times: Dict[Tuple[str, int], List[Tuple[float, str]]],
    wake_cats: Dict[Tuple[int, str], str],
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
    wake_cats : Dict[Tuple[int, str], str]
        ``{(episode_id, acid): wake_turbulence_category}`` mapping. Keyed by
        ``(episode_id, acid)`` rather than ``acid`` alone since acids are
        reused across episode slots (same collision shape as the slot-
        recycling bug) — currently harmless since ``wake_cat`` is hardcoded
        to ``"D"`` everywhere it's constructed, but load-bearing once wake-
        category diversity is introduced.
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

    for group_key, times_and_acids in landing_times.items():
        _runway_id, episode_id = group_key
        if len(times_and_acids) < 2:
            continue
        sorted_pairs: List[Tuple[float, str]] = sorted(
            times_and_acids, key=lambda x: x[0]  # type: ignore[index]
        )
        for i in range(1, len(sorted_pairs)):
            t_prev, acid_prev = sorted_pairs[i - 1]  # type: ignore[misc]
            t_curr, acid_curr = sorted_pairs[i]       # type: ignore[misc]
            gap = t_curr - t_prev
            lead_cat = wake_cats.get((episode_id, acid_prev), "D")
            trail_cat = wake_cats.get((episode_id, acid_curr), "D")
            required = recat_matrix.get(lead_cat, {}).get(trail_cat, 90.0)
            n_pairs += 1
            if gap >= (required - tolerance_s):
                n_compliant += 1

    return (n_compliant / n_pairs) if n_pairs > 0 else float("nan")


def _compute_separation_compliance_counts_by_episode(
    landing_times: Dict[Tuple[str, int], List[Tuple[float, str]]],
    wake_cats: Dict[Tuple[int, str], str],
    recat_matrix: Dict[str, Dict[str, float]],
    tolerance_s: float = 5.0,
) -> Dict[int, Tuple[int, int]]:
    """Per-episode ``(n_compliant, n_pairs)`` counts, pooling across
    runways within each episode -- backs C_sep's ``c_sep_std`` companion
    figure (episode-to-episode variance in separation compliance).

    Deliberately a standalone loop rather than a refactor of
    :func:`_compute_separation_compliance` to share this logic — keeps
    that already-tested primary-value function's code path completely
    untouched, at the cost of walking the same pairs twice when both the
    pooled C_sep and this per-episode breakdown are needed (this runs in
    offline reporting, not the simulation hot path, so the duplicate pass
    is cheap). See that function's docstring for the grouping rationale.
    """
    counts: Dict[int, List[int]] = defaultdict(lambda: [0, 0])  # [n_compliant, n_pairs]

    for group_key, times_and_acids in landing_times.items():
        _runway_id, episode_id = group_key
        if len(times_and_acids) < 2:
            continue
        sorted_pairs: List[Tuple[float, str]] = sorted(
            times_and_acids, key=lambda x: x[0]  # type: ignore[index]
        )
        for i in range(1, len(sorted_pairs)):
            t_prev, acid_prev = sorted_pairs[i - 1]  # type: ignore[misc]
            t_curr, acid_curr = sorted_pairs[i]       # type: ignore[misc]
            gap = t_curr - t_prev
            lead_cat = wake_cats.get((episode_id, acid_prev), "D")
            trail_cat = wake_cats.get((episode_id, acid_curr), "D")
            required = recat_matrix.get(lead_cat, {}).get(trail_cat, 90.0)
            counts[episode_id][1] += 1
            if gap >= (required - tolerance_s):
                counts[episode_id][0] += 1

    return {episode_id: (c[0], c[1]) for episode_id, c in counts.items()}


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

        Every metric above except ``n_episodes``/``n_aircraft``/``success_rate``
        also has a ``<metric>_std`` companion key: the sample standard
        deviation (``ddof=1``) of that metric's value across episodes --
        episode-to-episode variance, not a standard error of the mean.
        NaN with fewer than 2 valid episode observations. Does not change
        any primary ``<metric>`` value above.

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
        episode_ids_all = np.array([rec.episode_id for rec in records])

        # --- Throughput ---
        # Landing COUNTS can be pooled across episodes by runway_id alone --
        # unlike separation compliance below, a raw count isn't sensitive to
        # different episodes' clocks being independent. But the elapsed-time
        # DENOMINATOR is: actual_landing_time resets near zero at every
        # episode's env.reset(), so a naive max() over all pooled records only
        # recovers the single largest episode's own span, not the true total
        # elapsed time across all episodes -- summed below instead (bug found
        # 2026-08-08; this comment previously incorrectly extended the
        # pooling argument to the denominator too).
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

        episode_success_counts: Dict[int, int] = defaultdict(int)
        episode_max_landing_s: Dict[int, float] = {}
        for rec in records:
            if rec.success:
                episode_success_counts[rec.episode_id] += 1
                if rec.actual_landing_time > episode_max_landing_s.get(rec.episode_id, 0.0):
                    episode_max_landing_s[rec.episode_id] = rec.actual_landing_time
        total_time_s = sum(episode_max_landing_s.values()) if episode_max_landing_s else 3600.0
        window_h = max(total_time_s / 3600.0, 1e-6)
        gamma, gamma_r = _compute_throughput(landing_times_by_rwy, window_h)

        # gamma_std: dispersion of each episode's OWN landings/hour ratio --
        # a genuinely different quantity from gamma itself (a pooled
        # total/total-window ratio, not a mean of per-episode ratios), kept
        # purely as an episode-to-episode variance diagnostic alongside the
        # unchanged primary gamma value.
        per_episode_gamma = [
            episode_success_counts[ep] / max(span_s / 3600.0, 1e-6)
            for ep, span_s in episode_max_landing_s.items()
        ]
        gamma_std = (
            float(np.std(per_episode_gamma, ddof=1)) if len(per_episode_gamma) >= 2 else float("nan")
        )

        # --- Separation compliance ---
        # Keyed by (episode_id, acid), not acid alone -- see
        # _compute_separation_compliance's docstring for the collision
        # rationale (acids are reused across episode slots).
        wake_cats = {(rec.episode_id, rec.acid): rec.wake_cat for rec in records}
        c_sep = _compute_separation_compliance(
            landing_times_by_rwy_episode,
            wake_cats,
            recat_matrix,
            tolerance_s=self.separation_tolerance_s,
        )
        per_episode_c_sep_counts = _compute_separation_compliance_counts_by_episode(
            landing_times_by_rwy_episode, wake_cats, recat_matrix,
            tolerance_s=self.separation_tolerance_s,
        )
        per_episode_c_sep = [c / p for c, p in per_episode_c_sep_counts.values() if p > 0]
        c_sep_std = (
            float(np.std(per_episode_c_sep, ddof=1)) if len(per_episode_c_sep) >= 2 else float("nan")
        )

        # --- Tracking degradation Δε ---
        # Two distinct comparisons (see docstring above): the literal
        # Eq. tracking_degradation (cps vs. static-TTA, RQ2.2's actual
        # question about the cost of replanning) and a secondary,
        # honestly-labelled uncoordinated-reference comparison (cps vs.
        # solo) that is NOT Groot et al.'s published data.
        #
        # Masks require BOTH rta_error_cps and rta_error_static/solo to be
        # non-NaN, not just the static/solo side: the cps/static/solo
        # passes are three causally independent env rollouts under the
        # same seed (coordination_baseline.py's three-pass design), so
        # rta_error_cps can be NaN (that pass's aircraft never got a TTA
        # assignment) on a record where rta_error_static is perfectly
        # valid, or vice versa. A stray NaN on the cps side wasn't
        # filtered here before, which would have silently poisoned the
        # whole pooled np.mean() to NaN for every record via a single
        # NaN row (the dataset currently on disk has zero such rows --
        # checked directly -- so this was latent, not something that
        # actually corrupted any published number).
        cps_arr = np.array([rec.rta_error_cps for rec in records])
        static_arr = np.array([rec.rta_error_static for rec in records])
        solo_arr = np.array([rec.rta_error_solo for rec in records])
        static_mask = ~np.isnan(cps_arr) & ~np.isnan(static_arr)
        solo_mask = ~np.isnan(cps_arr) & ~np.isnan(solo_arr)

        delta_eps_static_all = np.abs(cps_arr) - np.abs(static_arr)
        delta_eps_static_values = delta_eps_static_all[static_mask]
        delta_epsilon_vs_static = (
            float(np.mean(delta_eps_static_values)) if delta_eps_static_values.size else float("nan")
        )
        _, delta_epsilon_vs_static_std = _episode_ratio_mean_std(
            episode_ids_all, delta_eps_static_all, static_mask,
        )
        delta_eps_uncoord_all = np.abs(cps_arr) - np.abs(solo_arr)
        delta_eps_uncoord_values = delta_eps_uncoord_all[solo_mask]
        delta_epsilon_vs_uncoordinated = (
            float(np.mean(delta_eps_uncoord_values)) if delta_eps_uncoord_values.size else float("nan")
        )
        _, delta_epsilon_vs_uncoordinated_std = _episode_ratio_mean_std(
            episode_ids_all, delta_eps_uncoord_all, solo_mask,
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
        _, r_rec_std = _episode_ratio_mean_std(
            episode_ids_all,
            np.array([
                1.0 if (not np.isnan(rec.rta_error_cps) and abs(rec.rta_error_cps) <= self.rta_tolerance_s)
                else 0.0
                for rec in records
            ]),
            np.array([rec.tta_updated_mid_trajectory for rec in records]),
        )

        # --- Delay ripple index ρ_ripple ---
        # Lag-1 autocorrelation must be computed WITHIN each episode's own
        # arrival sequence, not pooled across episodes -- a lag-1 pair must
        # never straddle two independent simulation runs' clocks (same bug
        # class as throughput's total_time_s, fixed 2026-08-08; mirrors the
        # landing_times_by_rwy_episode episode-scoping pattern above).
        # Vectorized via _grouped_lag1_autocorrelation (lexsort + reduceat)
        # instead of a Python-level loop over episodes.
        successful_records = [rec for rec in records if rec.success]
        rho_ripple, rho_ripple_std = (
            _grouped_lag1_autocorrelation(
                np.array([rec.episode_id for rec in successful_records]),
                np.array([rec.actual_landing_time for rec in successful_records]),
                np.array([rec.rta_error_cps for rec in successful_records]),
            )
            if successful_records
            else (float("nan"), float("nan"))
        )

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
        stall_detected_arr = np.array([rec.stall_detected for rec in records], dtype=float)
        success_arr = np.array([rec.success for rec in records], dtype=float)
        stall_mask = np.array([rec.stall_detected for rec in records], dtype=bool)
        _, stall_rate_std = _episode_ratio_mean_std(episode_ids_all, stall_detected_arr)
        _, stall_recovered_std = _episode_ratio_mean_std(episode_ids_all, stall_detected_arr * success_arr)
        _, stall_unrecovered_std = _episode_ratio_mean_std(
            episode_ids_all, stall_detected_arr * (1.0 - success_arr)
        )
        _, stall_recovery_rate_std = _episode_ratio_mean_std(episode_ids_all, success_arr, stall_mask)

        def _std_or_nan(x: float) -> Any:
            return round(x, 4) if not np.isnan(x) else "nan"

        return {
            "n_episodes": len(set(r.episode_id for r in records)),
            "n_aircraft": n_aircraft,
            "success_rate": round(success_rate, 4),
            "gamma": round(gamma, 4),
            "gamma_std": _std_or_nan(gamma_std),
            "gamma_r": {rwy: round(v, 4) for rwy, v in gamma_r.items()},
            "c_sep": round(float(c_sep), 4) if not np.isnan(c_sep) else "nan",
            "c_sep_std": _std_or_nan(c_sep_std),
            "delta_epsilon_vs_static": (
                round(delta_epsilon_vs_static, 4) if not np.isnan(delta_epsilon_vs_static) else "nan"
            ),
            "delta_epsilon_vs_static_std": _std_or_nan(delta_epsilon_vs_static_std),
            "delta_epsilon_vs_uncoordinated": (
                round(delta_epsilon_vs_uncoordinated, 4)
                if not np.isnan(delta_epsilon_vs_uncoordinated) else "nan"
            ),
            "delta_epsilon_vs_uncoordinated_std": _std_or_nan(delta_epsilon_vs_uncoordinated_std),
            "r_rec": round(r_rec, 4) if not np.isnan(r_rec) else "nan",
            "r_rec_std": _std_or_nan(r_rec_std),
            "rho_ripple": round(rho_ripple, 4) if not np.isnan(rho_ripple) else "nan",
            "rho_ripple_std": _std_or_nan(rho_ripple_std),
            "stall_unrecovered": round(stall_unrecovered, 4),
            "stall_unrecovered_std": _std_or_nan(stall_unrecovered_std),
            "stall_recovery_rate": (
                round(stall_recovery_rate, 4) if not np.isnan(stall_recovery_rate) else "nan"
            ),
            "stall_recovery_rate_std": _std_or_nan(stall_recovery_rate_std),
            "stall_recovered": round(stall_recovered, 4),
            "stall_recovered_std": _std_or_nan(stall_recovered_std),
            "stall_rate": round(stall_rate, 4),  # diagnostic only -- see comment above
            "stall_rate_std": _std_or_nan(stall_rate_std),
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
