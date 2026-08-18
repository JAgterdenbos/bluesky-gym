"""
Vectorization correctness + performance gate for
``cps_coordination/coordination/cps_manager.py``.

Covers the hot per-decision-step loop identified in
``.claude/plans/phase3_cps_coordination_plan.md``'s "Vectorization &
Optimization" section:

  1. ``CPSManager._assign_runways_dynamic``'s eligibility/rank loop
     (previously a triple-nested pure-Python loop over aircraft x runway x
     aircraft, now a broadcast over ``(n, n, r)`` boolean arrays).
  2. ``CPSManager._apply_k_cps_constraint``'s O(n) sortedness short-circuit
     (see below).

Finding 1's loop is checked against a standalone "legacy" reference (a
literal transcription of the pre-vectorization code) on a synthetic fixture
(n=10 aircraft, r=12 runways, k_cps=3), asserting the vectorized rewrite is
a bit-identical restatement, not an approximation. A capped repeated-call
benchmark then confirms the rewrite is actually faster, not just equivalent.

A gate used to cover a *different*, since-removed fairness-weighted version
of ``_apply_k_cps_constraint`` (Finding 3 of the original audit). That
method was removed entirely 2026-08-12 (see
``.claude/plans/stall_rate_investigation.md``) after being found to never
win against plain FCFS at any tested ``fairness_weight > 0``. A fairness-
free reintroduction of the *method itself* (not that removed version)
followed on 2026-08-18 (see
``.claude/plans/cps_static_mode_k_cps_design.md``) using an
earliest-ETA-in-window selection rule instead, which is a no-op on the
method's always-pre-sorted real input. Finding 2 below covers the resulting
O(n) short-circuit added on top of that (skip the O(n·(2k+1)) sweep
entirely when the input is already sorted) -- a numpy-vectorized rewrite of
the sweep itself was also benchmarked and found 6-10x *slower* than the
plain Python loop at production window widths (same conclusion as the
original, now-removed fairness-weighted sweep's own vectorization
attempt), so that path was not pursued further.

Run: python cps_coordination/testing/test_vectorization_performance.py
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Tuple

import numpy as np

from bluesky_gym.envs.pathplanning_goal_env import ALL_RUNWAYS
from cps_coordination.coordination.cps_manager import AircraftState, CPSManager

N_AIRCRAFT = 10
N_RUNWAYS = 12
K_CPS = 3
N_BENCHMARK_REPS = 1000

# Small, internally-consistent RECAT-EU-shaped matrix -- values don't need
# to match cps_base.yaml exactly, just exercise multiple wake categories.
_RECAT_MATRIX: Dict[str, Dict[str, float]] = {
    "A": {"A": 120, "B": 100, "C": 80, "D": 120, "E": 140, "F": 180},
    "B": {"A": 80, "B": 80, "C": 80, "D": 100, "E": 120, "F": 140},
    "C": {"A": 80, "B": 80, "C": 80, "D": 80, "E": 100, "F": 120},
    "D": {"A": 80, "B": 80, "C": 80, "D": 80, "E": 80, "F": 100},
    "E": {"A": 80, "B": 80, "C": 80, "D": 80, "E": 80, "F": 80},
    "F": {"A": 80, "B": 80, "C": 80, "D": 80, "E": 80, "F": 80},
}


def _make_manager(**kwargs) -> CPSManager:
    return CPSManager(
        k_cps=K_CPS,
        recat_matrix=_RECAT_MATRIX,
        available_runways=ALL_RUNWAYS[:N_RUNWAYS],
        **kwargs,
    )


# ------------------------------------------------------------------ #
# Finding 1: _assign_runways_dynamic
# ------------------------------------------------------------------ #


class _StubSurrogate:
    """Returns a fixed, precomputed ``(n, r)`` ETA matrix regardless of
    input -- isolates the eligibility/rank loop from the (already-
    vectorized, see Finding 4 of the audit) surrogate call itself."""

    def __init__(self, eta_matrix: np.ndarray) -> None:
        self._eta_matrix = eta_matrix

    def predict_eta_fleet_all_runways(
        self, states, runways, current_time, lag_features, target_time_budget=None
    ) -> np.ndarray:
        return self._eta_matrix


def _make_dynamic_fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    runways = ALL_RUNWAYS[:N_RUNWAYS]
    fleet = [
        AircraftState(
            acid=f"AC{i:03d}",
            state=np.zeros(5),
            runway_id=runways[i % len(runways)],
            eta=0.0,
            wake_cat="D",
            spawn_time=0.0,
        )
        for i in range(N_AIRCRAFT)
    ]
    eta_matrix = rng.uniform(500.0, 5000.0, size=(N_AIRCRAFT, N_RUNWAYS))
    return fleet, eta_matrix, runways


def _legacy_assign_runways_dynamic(
    fleet: List[AircraftState], eta_matrix: np.ndarray, runways: List[str], k: int,
    current_time: float, final_approach_lock_s: float, reassignment_hysteresis_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Literal transcription of the pre-vectorization triple-nested loop,
    plus the final-approach lock and reassignment hysteresis added to the
    real method after the original vectorization (see
    FINAL_APPROACH_LOCK_S/REASSIGNMENT_HYSTERESIS_S's docstrings in
    cps_manager.py) -- kept in sync here so this stays a genuine parity
    check rather than silently drifting into testing a stale reference."""
    n = len(fleet)
    rwy_index = {r: j for j, r in enumerate(runways)}
    current_col = np.array([rwy_index.get(ac.runway_id, 0) for ac in fleet])
    current_eta = eta_matrix[np.arange(n), current_col]
    fcfs_rank = np.argsort(np.argsort(current_eta))

    best_rwy_idx = np.empty(n, dtype=int)
    for i, ac in enumerate(fleet):
        eligible: List[int] = []
        for j, rwy in enumerate(runways):
            other_etas = [
                eta_matrix[i2, j]
                for i2, ac2 in enumerate(fleet)
                if i2 != i and ac2.runway_id == rwy
            ]
            sigma_r = sum(1 for e in other_etas if e < eta_matrix[i, j])
            if abs(sigma_r - int(fcfs_rank[i])) <= k:
                eligible.append(j)

        # Final-approach lock: force back to the current runway regardless
        # of what the fairness window would otherwise allow.
        if current_eta[i] - current_time <= final_approach_lock_s:
            eligible = [int(current_col[i])]

        if not eligible:
            eligible = [rwy_index.get(ac.runway_id, int(np.argmin(eta_matrix[i])))]
        candidate = min(eligible, key=lambda j: eta_matrix[i, j])

        # Reassignment hysteresis: only switch away from a (window-)eligible
        # current runway if the candidate beats it by more than the margin.
        if int(current_col[i]) in eligible:
            gain = eta_matrix[i, current_col[i]] - eta_matrix[i, candidate]
            if gain < reassignment_hysteresis_s:
                candidate = int(current_col[i])

        best_rwy_idx[i] = candidate
    return best_rwy_idx, fcfs_rank


def check_assign_runways_dynamic_parity() -> bool:
    ok = True
    fleet, eta_matrix, runways = _make_dynamic_fixture()
    manager = _make_manager(runway_assignment_mode="dynamic")

    legacy_best_rwy_idx, legacy_fcfs_rank = _legacy_assign_runways_dynamic(
        fleet, eta_matrix, runways, K_CPS,
        current_time=0.0,
        final_approach_lock_s=manager.FINAL_APPROACH_LOCK_S,
        reassignment_hysteresis_s=manager.REASSIGNMENT_HYSTERESIS_S,
    )

    manager._fleet = copy.deepcopy(fleet)
    manager._fleet_index = {ac.acid: i for i, ac in enumerate(manager._fleet)}
    manager._assign_runways_dynamic(_StubSurrogate(eta_matrix), current_time=0.0, lag_features=None)

    rwy_index = {r: j for j, r in enumerate(runways)}
    vec_best_rwy_idx = np.array([rwy_index[ac.runway_id] for ac in manager._fleet])
    vec_fcfs_rank = np.array([ac.fcfs_rank for ac in manager._fleet])
    vec_eta = np.array([ac.eta for ac in manager._fleet])
    expected_eta = eta_matrix[np.arange(N_AIRCRAFT), legacy_best_rwy_idx]

    if not np.array_equal(vec_best_rwy_idx, legacy_best_rwy_idx):
        print(f"FAIL: best_rwy_idx mismatch\n  legacy={legacy_best_rwy_idx}\n  vector={vec_best_rwy_idx}")
        ok = False
    if not np.array_equal(vec_fcfs_rank, legacy_fcfs_rank):
        print(f"FAIL: fcfs_rank mismatch\n  legacy={legacy_fcfs_rank}\n  vector={vec_fcfs_rank}")
        ok = False
    if not np.array_equal(vec_eta, expected_eta):
        print(f"FAIL: eta mismatch\n  expected={expected_eta}\n  vector={vec_eta}")
        ok = False

    if ok:
        print(
            f"PASS: _assign_runways_dynamic vectorized rewrite is bit-identical to "
            f"legacy on n={N_AIRCRAFT}, r={N_RUNWAYS}, k={K_CPS} fixture"
        )
    return ok


def benchmark_assign_runways_dynamic() -> bool:
    fleet, eta_matrix, runways = _make_dynamic_fixture()
    surrogate = _StubSurrogate(eta_matrix)
    manager = _make_manager(runway_assignment_mode="dynamic")

    legacy_fleets = [copy.deepcopy(fleet) for _ in range(N_BENCHMARK_REPS)]
    t0 = time.perf_counter()
    for f in legacy_fleets:
        _legacy_assign_runways_dynamic(
            f, eta_matrix, runways, K_CPS,
            current_time=0.0,
            final_approach_lock_s=manager.FINAL_APPROACH_LOCK_S,
            reassignment_hysteresis_s=manager.REASSIGNMENT_HYSTERESIS_S,
        )
    legacy_elapsed = time.perf_counter() - t0

    vec_fleets = [copy.deepcopy(fleet) for _ in range(N_BENCHMARK_REPS)]
    t0 = time.perf_counter()
    for f in vec_fleets:
        manager._fleet = f
        manager._fleet_index = {ac.acid: i for i, ac in enumerate(f)}
        manager._assign_runways_dynamic(surrogate, current_time=0.0, lag_features=None)
    vector_elapsed = time.perf_counter() - t0

    speedup = legacy_elapsed / vector_elapsed if vector_elapsed > 0 else float("inf")
    print(
        f"_assign_runways_dynamic x{N_BENCHMARK_REPS}: "
        f"legacy={legacy_elapsed*1e3:.2f}ms  vectorized={vector_elapsed*1e3:.2f}ms  "
        f"speedup={speedup:.2f}x"
    )
    ok = vector_elapsed < legacy_elapsed
    print("PASS" if ok else "FAIL", ": vectorized rewrite is faster than legacy")
    return ok


# ------------------------------------------------------------------ #
# Finding 2: _apply_k_cps_constraint's O(n) sortedness short-circuit
# ------------------------------------------------------------------ #

K_CPS_BENCH_N = 50   # matches cps_scale_10k.yaml's max_concurrent_aircraft
K_CPS_BENCH_K = 3    # matches the production k_cps sweep's largest value


def _make_k_cps_fixture(n: int, seed: int) -> List[AircraftState]:
    """Already ascending-(eta, acid)-sorted, matching the real call site's
    invariant (:meth:`CPSManager._replan` always passes ``_fcfs_order()``'s
    -- optionally stall-filtered, order-preserving -- output)."""
    rng = np.random.default_rng(seed)
    etas = np.sort(rng.uniform(0.0, 3000.0, size=n))
    return [
        AircraftState(
            acid=f"AC{i:04d}", state=np.zeros(5), runway_id="27",
            eta=float(etas[i]), wake_cat="D",
        )
        for i in range(n)
    ]


def _legacy_apply_k_cps_constraint_full_sweep(
    fcfs_order: List[AircraftState], k: int,
) -> List[AircraftState]:
    """Literal transcription of the O(n·(2k+1)) sweep without the
    sortedness short-circuit -- the pre-optimization reference this finding
    benchmarks against."""
    if k == 0:
        return list(fcfs_order)
    n = len(fcfs_order)
    scheduled_mask = [False] * n
    scheduled: List[AircraftState] = []
    for pos in range(n):
        window_lo = max(0, pos - k)
        window_hi = min(pos + k, n - 1)
        idxs = [i for i in range(window_lo, window_hi + 1) if not scheduled_mask[i]]
        if not idxs:
            idxs = [i for i in range(n) if not scheduled_mask[i]]
        best_idx = idxs[0]
        for idx in idxs[1:]:
            ac, best_ac = fcfs_order[idx], fcfs_order[best_idx]
            if (ac.eta, ac.acid) < (best_ac.eta, best_ac.acid):
                best_idx = idx
        scheduled_mask[best_idx] = True
        scheduled.append(fcfs_order[best_idx])
    return scheduled


def check_apply_k_cps_constraint_parity() -> bool:
    ok = True
    manager = _make_manager(runway_assignment_mode="static")
    for n, k in [(5, 2), (10, 3), (35, 3), (50, 3), (50, 1)]:
        for seed in range(50):
            fixture = _make_k_cps_fixture(n, seed)
            manager.k_cps = k
            legacy = [ac.acid for ac in _legacy_apply_k_cps_constraint_full_sweep(fixture, k)]
            actual = [ac.acid for ac in manager._apply_k_cps_constraint(fixture)]
            if actual != legacy:
                print(f"FAIL: n={n} k={k} seed={seed}\n  legacy={legacy}\n  actual={actual}")
                ok = False
    if ok:
        print(
            "PASS: _apply_k_cps_constraint's short-circuit is bit-identical to the full "
            "sweep across n in {5,10,35,50}, k in {1,2,3}, 50 random seeds each"
        )
    return ok


def benchmark_apply_k_cps_constraint() -> bool:
    fixture = _make_k_cps_fixture(K_CPS_BENCH_N, seed=0)
    manager = _make_manager(runway_assignment_mode="static")
    manager.k_cps = K_CPS_BENCH_K

    t0 = time.perf_counter()
    for _ in range(N_BENCHMARK_REPS):
        _legacy_apply_k_cps_constraint_full_sweep(fixture, K_CPS_BENCH_K)
    legacy_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N_BENCHMARK_REPS):
        manager._apply_k_cps_constraint(fixture)
    shortcircuit_elapsed = time.perf_counter() - t0

    speedup = legacy_elapsed / shortcircuit_elapsed if shortcircuit_elapsed > 0 else float("inf")
    print(
        f"_apply_k_cps_constraint x{N_BENCHMARK_REPS} (n={K_CPS_BENCH_N}, k={K_CPS_BENCH_K}): "
        f"full_sweep={legacy_elapsed*1e3:.2f}ms  short_circuit={shortcircuit_elapsed*1e3:.2f}ms  "
        f"speedup={speedup:.2f}x"
    )
    ok = shortcircuit_elapsed < legacy_elapsed
    print("PASS" if ok else "FAIL", ": short-circuit is faster than the full sweep")
    return ok


if __name__ == "__main__":
    passed_parity_1 = check_assign_runways_dynamic_parity()
    passed_perf_1 = benchmark_assign_runways_dynamic()
    passed_parity_2 = check_apply_k_cps_constraint_parity()
    passed_perf_2 = benchmark_apply_k_cps_constraint()
    raise SystemExit(
        0 if (passed_parity_1 and passed_perf_1 and passed_parity_2 and passed_perf_2) else 1
    )
