"""
scratchpad/phase4_detector_comparison.py
--------------------------------------------
Phase 4 design step: replay several candidate stall-detector formulations
against the rich per-cycle trace collected by phase4_detector_range_trace.py
(t, x, y, runway_id, tta, eta per aircraft per cycle at cap=50), and score
each against ground-truth outcome (success / death_cause).

Ground-truth proxy: an aircraft "genuinely failed" if success == False. This
is imperfect (some non-stall failures exist, e.g. RTA-tolerance misses -- see
the parent capacity-sweep plan's death_cause/stall_detected coincidence
finding) but is the best available label without hand-annotating trajectories.

For each candidate:
  - flag_rate: fraction of aircraft ever flagged
  - precision: of flagged aircraft, fraction that actually failed
  - recall:    of aircraft that actually failed, fraction that got flagged
  - median flag-point distance-to-IAF (km) -- sanity check against Phase 1's
    finding that the CURRENT detector flags aircraft ~140-150km out.

Usage: uv run python scratchpad/phase4_detector_comparison.py
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from cps_coordination.coordination.eta_surrogate import ETASurrogate

MAX_DISTANCE = 300.0
ACTION_TIME = 120.0
STALL_WINDOW_S = 1800.0
STALL_PROGRESS_EPS_KM = 5.0

ROOT = "experiments/cps_eval/stall_window_grounding_20260813"


# ---------------------------------------------------------------------------
# Candidate detectors. Each takes (t, dist_km, tta, eta) arrays (aligned,
# ascending t) for one flight and returns the flag step index, or None.
# ---------------------------------------------------------------------------

def detector_baseline(t, dist_km, tta, eta) -> Optional[int]:
    """Current production rule: best-distance-ever, fixed 1800s window, 5km eps."""
    best = math.inf
    best_t = t[0]
    for i in range(len(t)):
        if dist_km[i] < best - STALL_PROGRESS_EPS_KM:
            best = dist_km[i]
            best_t = t[i]
            continue
        if t[i] - best_t >= STALL_WINDOW_S:
            return i
    return None


def detector_schedule_relative(t, dist_km, tta, eta) -> Optional[int]:
    """Window extended by the aircraft's own currently-imposed CPS delay
    (tta - eta): effective_window = STALL_WINDOW_S + max(0, tta[i] - eta[i]).
    """
    best = math.inf
    best_t = t[0]
    for i in range(len(t)):
        if dist_km[i] < best - STALL_PROGRESS_EPS_KM:
            best = dist_km[i]
            best_t = t[i]
            continue
        imposed_delay = 0.0 if math.isnan(tta[i]) else max(0.0, tta[i] - eta[i])
        window = STALL_WINDOW_S + imposed_delay
        if t[i] - best_t >= window:
            return i
    return None


def detector_fixed_3600(t, dist_km, tta, eta) -> Optional[int]:
    """Naive recalibration: double the fixed window to 3600s (flight-scale),
    no schedule-awareness -- cheap baseline-improvement comparator."""
    best = math.inf
    best_t = t[0]
    for i in range(len(t)):
        if dist_km[i] < best - STALL_PROGRESS_EPS_KM:
            best = dist_km[i]
            best_t = t[i]
            continue
        if t[i] - best_t >= 3600.0:
            return i
    return None


def detector_own_flight_relative(t, dist_km, tta, eta) -> Optional[int]:
    """Window = max(STALL_WINDOW_S, 0.5 * elapsed-time-so-far-in-this-flight)
    -- adapts to each flight's own pace without needing tta/eta at all."""
    best = math.inf
    best_t = t[0]
    t0 = t[0]
    for i in range(len(t)):
        if dist_km[i] < best - STALL_PROGRESS_EPS_KM:
            best = dist_km[i]
            best_t = t[i]
            continue
        elapsed = t[i] - t0
        window = max(STALL_WINDOW_S, 0.5 * elapsed)
        if t[i] - best_t >= window:
            return i
    return None


def detector_velocity_ema(t, dist_km, tta, eta, alpha: float = 0.3) -> Optional[int]:
    """EMA-smoothed distance-rate: flag when the smoothed rate stays
    persistently >= -eps_rate (not shrinking meaningfully) for the window.
    Different signal type (rate, not best-ever-vs-window) per the
    'progress-rate/velocity-based' candidate."""
    if len(t) < 2:
        return None
    eps_rate = STALL_PROGRESS_EPS_KM / STALL_WINDOW_S  # km/s threshold
    ema_rate = 0.0
    bad_since = None
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            continue
        rate = (dist_km[i] - dist_km[i - 1]) / dt  # negative = approaching
        ema_rate = alpha * rate + (1 - alpha) * ema_rate
        if ema_rate >= -eps_rate:
            if bad_since is None:
                bad_since = t[i]
            elif t[i] - bad_since >= STALL_WINDOW_S:
                return i
        else:
            bad_since = None
    return None


def detector_probation(t, dist_km, tta, eta, recovery_window_s: float = 900.0, recovery_eps_km: float = 10.0) -> Optional[int]:
    """Same best-distance-ever/1800s-window trigger as baseline, but NOT
    sticky: once flagged, keep watching -- if distance beats the at-flag
    value by recovery_eps_km (2x STALL_PROGRESS_EPS_KM) within
    recovery_window_s, un-flag and resume normal best-ever tracking (eligible
    to be re-flagged later on a genuine subsequent plateau). Returns the
    step of the LAST flag still in effect at trajectory end, or None if
    never flagged or fully recovered by the end -- this directly targets
    Phase 1's finding that ~99-100% of currently-flagged aircraft go on to
    make real further progress, by giving the mechanism a way to notice that
    recovery instead of freezing the aircraft out for the rest of the flight
    the instant it first crosses the threshold.
    """
    best = math.inf
    best_t = t[0]
    flagged_at: Optional[int] = None
    flagged_dist = None
    for i in range(len(t)):
        if flagged_at is not None:
            if dist_km[i] < flagged_dist - recovery_eps_km:
                flagged_at = None  # recovered -- resume normal tracking
                best = dist_km[i]
                best_t = t[i]
                continue
            if t[i] - t[flagged_at] >= recovery_window_s:
                return flagged_at  # never recovered in time -- stays flagged for good
            continue
        if dist_km[i] < best - STALL_PROGRESS_EPS_KM:
            best = dist_km[i]
            best_t = t[i]
            continue
        if t[i] - best_t >= STALL_WINDOW_S:
            flagged_at = i
            flagged_dist = dist_km[i]
    return flagged_at


def _make_probation(recovery_window_s, recovery_eps_km, reflag_window_s):
    """Like detector_probation, but the re-flag window after a recovery is
    reflag_window_s (not a full fresh STALL_WINDOW_S) -- a genuinely-failing
    aircraft that shows one ambiguous transient dip should be caught again
    quickly, not effectively get a second full 30-min grace period."""
    def fn(t, dist_km, tta, eta):
        best = math.inf
        best_t = t[0]
        flagged_at = None
        flagged_dist = None
        window = STALL_WINDOW_S
        for i in range(len(t)):
            if flagged_at is not None:
                if dist_km[i] < flagged_dist - recovery_eps_km:
                    flagged_at = None
                    best = dist_km[i]
                    best_t = t[i]
                    window = reflag_window_s
                    continue
                if t[i] - t[flagged_at] >= recovery_window_s:
                    return flagged_at
                continue
            if dist_km[i] < best - STALL_PROGRESS_EPS_KM:
                best = dist_km[i]
                best_t = t[i]
                continue
            if t[i] - best_t >= window:
                flagged_at = i
                flagged_dist = dist_km[i]
        return flagged_at
    return fn


CANDIDATES: Dict[str, Callable] = {
    "A_baseline": detector_baseline,
    "B_schedule_relative": detector_schedule_relative,
    "C_fixed_3600s": detector_fixed_3600,
    "D_own_flight_relative": detector_own_flight_relative,
    "E_velocity_ema": detector_velocity_ema,
    "I_probation_300_10_reflag900": _make_probation(300.0, 10.0, 900.0),
    "K_probation_120_10_reflag900": _make_probation(120.0, 10.0, 900.0),
    "L_probation_120_5_reflag900": _make_probation(120.0, 5.0, 900.0),
    "M_probation_120_10_reflag300": _make_probation(120.0, 10.0, 300.0),
    "N_probation_120_5_reflag300": _make_probation(120.0, 5.0, 300.0),
    "O_probation_120_15_reflag900": _make_probation(120.0, 15.0, 900.0),
    "P_probation_120_10_reflag1800": _make_probation(120.0, 10.0, 1800.0),
}


def main() -> None:
    surrogate = ETASurrogate.load("cps_coordination/models/eta_surrogate.pkl")
    iaf_ref = surrogate._iaf_ref

    trace = pd.read_parquet(f"{ROOT}/phase4_detector_trace_cap50.parquet")
    outcome = pd.read_parquet(f"{ROOT}/phase4_detector_outcome_cap50.parquet")

    # cap=50 == total_arrivals_per_episode does NOT guarantee acid uniqueness
    # within an episode: aircraft can still land/die well before
    # spawn_window_s elapses, freeing their slot for a same-episode refill
    # (confirmed: 68/1500 outcome rows are (episode_id, acid) duplicates).
    # Disambiguate via local rank instead of a shared explicit key:
    #  - outcome_df's row order is termination order, which for a FIXED
    #    (episode_id, acid) equals occupancy/spawn order (a slot can only be
    #    refilled after its previous occupant departs) -- so cumcount() on
    #    the as-loaded (unsorted) frame recovers spawn-order rank.
    #  - trace_df carries `spawn_time` directly; sorting by it within each
    #    (episode_id, acid) group and taking cumcount() recovers the same
    #    rank independently.
    outcome = outcome.copy()
    outcome["local_rank"] = outcome.groupby(["episode_id", "acid"]).cumcount()
    outcome_idx = outcome.set_index(["episode_id", "acid", "local_rank"])

    trace = trace.sort_values(["episode_id", "acid", "spawn_time", "t"])
    trace["local_rank"] = (
        trace.groupby(["episode_id", "acid"])["spawn_time"]
        .rank(method="dense").astype(int) - 1
    )

    flights = {}
    for (ep, acid, rank), g in trace.groupby(["episode_id", "acid", "local_rank"]):
        t = g["t"].to_numpy(dtype=float)
        x = g["x"].to_numpy(dtype=float)
        y = g["y"].to_numpy(dtype=float)
        rwy = g["runway_id"].to_numpy()
        tta = g["tta"].to_numpy(dtype=float)
        eta = g["eta"].to_numpy(dtype=float)
        dist_km = np.array([
            math.hypot(iaf_ref.get(r, (0.0, 0.0))[0] - xi, iaf_ref.get(r, (0.0, 0.0))[1] - yi) * MAX_DISTANCE
            if r in iaf_ref else np.nan
            for r, xi, yi in zip(rwy, x, y)
        ])
        flights[(ep, acid, rank)] = (t, dist_km, tta, eta)

    results = []
    for name, fn in CANDIDATES.items():
        flagged = 0
        flag_dists = []
        tp = fp = fn_ = tn = 0
        for key, (t, dist_km, tta, eta) in flights.items():
            if np.isnan(dist_km).any() or len(t) < 2:
                continue
            flag_step = fn(t, dist_km, tta, eta)
            failed = key in outcome_idx.index and not bool(outcome_idx.loc[key, "success"])
            if flag_step is not None:
                flagged += 1
                flag_dists.append(dist_km[flag_step])
                if failed:
                    tp += 1
                else:
                    fp += 1
            else:
                if failed:
                    fn_ += 1
                else:
                    tn += 1
        n = tp + fp + fn_ + tn
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn_) if (tp + fn_) else float("nan")
        results.append({
            "detector": name, "n": n, "flagged": flagged,
            "flag_rate": flagged / n if n else float("nan"),
            "precision": precision, "recall": recall,
            "median_flag_dist_km": float(np.median(flag_dists)) if flag_dists else float("nan"),
            "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
        })

    res_df = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print(res_df.to_string(index=False))


if __name__ == "__main__":
    main()
