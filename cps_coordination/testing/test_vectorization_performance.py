"""
Vectorization correctness + performance gate for
``cps_coordination/coordination/cps_manager.py``.

Covers the two hot per-decision-step loops identified in
``.claude/plans/phase3_cps_coordination_plan.md``'s "Vectorization &
Optimization" section:

  1. ``CPSManager._assign_runways_dynamic``'s eligibility/rank loop
     (previously a triple-nested pure-Python loop over aircraft x runway x
     aircraft, now a broadcast over ``(n, n, r)`` boolean arrays).
  2. ``CPSManager._apply_k_cps_constraint``'s per-position candidate window
     (previously one ``_tta_for``/``_slack_penalty`` Python call per
     candidate, now a vectorized batch per position; the outer per-position
     loop stays sequential -- ``runway_last`` is genuinely mutated between
     positions).

Each loop is checked against a standalone "legacy" reference (a literal
transcription of the pre-vectorization code) on the same synthetic fixture
(n=10 aircraft, r=12 runways, k_cps=3), asserting the vectorized rewrite is
a bit-identical restatement, not an approximation. A capped repeated-call
benchmark then confirms the rewrite is actually faster, not just equivalent.

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
            wake_cat="C",
            spawn_time=0.0,
        )
        for i in range(N_AIRCRAFT)
    ]
    eta_matrix = rng.uniform(500.0, 5000.0, size=(N_AIRCRAFT, N_RUNWAYS))
    return fleet, eta_matrix, runways


def _legacy_assign_runways_dynamic(
    fleet: List[AircraftState], eta_matrix: np.ndarray, runways: List[str], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Literal transcription of the pre-vectorization triple-nested loop."""
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
        if not eligible:
            eligible = [rwy_index.get(ac.runway_id, int(np.argmin(eta_matrix[i])))]
        best_rwy_idx[i] = min(eligible, key=lambda j: eta_matrix[i, j])
    return best_rwy_idx, fcfs_rank


def check_assign_runways_dynamic_parity() -> bool:
    ok = True
    fleet, eta_matrix, runways = _make_dynamic_fixture()

    legacy_best_rwy_idx, legacy_fcfs_rank = _legacy_assign_runways_dynamic(
        fleet, eta_matrix, runways, K_CPS
    )

    manager = _make_manager(runway_assignment_mode="dynamic")
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

    legacy_fleets = [copy.deepcopy(fleet) for _ in range(N_BENCHMARK_REPS)]
    t0 = time.perf_counter()
    for f in legacy_fleets:
        _legacy_assign_runways_dynamic(f, eta_matrix, runways, K_CPS)
    legacy_elapsed = time.perf_counter() - t0

    manager = _make_manager(runway_assignment_mode="dynamic")
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
# Finding 3: _apply_k_cps_constraint
# ------------------------------------------------------------------ #


def _make_k_cps_fixture(seed: int = 1):
    rng = np.random.default_rng(seed)
    runways = ALL_RUNWAYS[:3]  # force real contention within a small n=10 fleet
    wake_cats = list(_RECAT_MATRIX.keys())
    etas = sorted(rng.uniform(1000.0, 6000.0, size=N_AIRCRAFT))
    fcfs_order = [
        AircraftState(
            acid=f"AC{i:03d}",
            state=np.zeros(5),
            runway_id=runways[i % len(runways)],
            eta=float(etas[i]),
            wake_cat=str(rng.choice(wake_cats)),
            spawn_time=0.0,
        )
        for i in range(N_AIRCRAFT)
    ]
    # Cross-cycle committed state: one genuine "different aircraft departed
    # last cycle" seed (has_prev branch), one exact self-match (the aircraft
    # sees its own prior commit -- must NOT self-separate), one runway with
    # no committed entry at all (no-prev branch).
    runway_last_committed = {
        runways[0]: (900.0, "D", "GHOST-PREV"),
        runways[1]: (950.0, "E", fcfs_order[1].acid),
    }
    return fcfs_order, runways, runway_last_committed


def _legacy_apply_k_cps_constraint(
    manager: CPSManager, fcfs_order: List[AircraftState], current_time: float
) -> List[AircraftState]:
    """Literal transcription of the pre-vectorization outer/inner loop,
    still calling the (unchanged) real ``_tta_for``/``_slack_penalty``."""
    if manager.k_cps == 0 or manager.fairness_weight <= 0.0:
        return list(fcfs_order)

    n = len(fcfs_order)
    k = manager.k_cps
    scheduled_mask = [False] * n
    scheduled: List[AircraftState] = []
    runway_last: Dict[str, AircraftState] = {}

    for pos in range(n):
        window_lo = max(0, pos - k)
        window_hi = min(pos + k, n - 1)
        eligible = [i for i in range(window_lo, window_hi + 1) if not scheduled_mask[i]]
        if not eligible:
            eligible = [i for i in range(n) if not scheduled_mask[i]]

        best_idx = best_cost = None
        best_tta = 0.0
        for idx in eligible:
            ac = fcfs_order[idx]
            tta_if_here = manager._tta_for(ac, runway_last)
            imposed_delay = max(0.0, tta_if_here - ac.eta)
            cost = imposed_delay - manager.fairness_weight * manager._slack_penalty(ac, current_time)
            if best_cost is None or cost < best_cost:
                best_idx, best_cost, best_tta = idx, cost, tta_if_here

        chosen = fcfs_order[best_idx]  # type: ignore[index]
        scheduled_mask[best_idx] = True
        scheduled.append(chosen)
        chosen.tta = best_tta
        runway_last[chosen.runway_id] = chosen

    return scheduled


def check_apply_k_cps_constraint_parity() -> bool:
    ok = True
    fcfs_order, _runways, runway_last_committed = _make_k_cps_fixture()
    current_time = 1000.0

    legacy_manager = _make_manager(fairness_weight=0.5)
    legacy_manager._runway_last_committed = dict(runway_last_committed)
    legacy_order = copy.deepcopy(fcfs_order)
    legacy_scheduled = _legacy_apply_k_cps_constraint(legacy_manager, legacy_order, current_time)

    vector_manager = _make_manager(fairness_weight=0.5)
    vector_manager._runway_last_committed = dict(runway_last_committed)
    vector_order = copy.deepcopy(fcfs_order)
    vector_scheduled = vector_manager._apply_k_cps_constraint(vector_order, current_time)

    legacy_acids = [ac.acid for ac in legacy_scheduled]
    vector_acids = [ac.acid for ac in vector_scheduled]
    legacy_ttas = np.array([ac.tta for ac in legacy_scheduled])
    vector_ttas = np.array([ac.tta for ac in vector_scheduled])

    if legacy_acids != vector_acids:
        print(f"FAIL: scheduled order mismatch\n  legacy={legacy_acids}\n  vector={vector_acids}")
        ok = False
    if not np.array_equal(legacy_ttas, vector_ttas):
        print(f"FAIL: tta mismatch\n  legacy={legacy_ttas}\n  vector={vector_ttas}")
        ok = False

    if ok:
        print(
            f"PASS: _apply_k_cps_constraint vectorized rewrite is bit-identical to "
            f"legacy on n={N_AIRCRAFT}, k={K_CPS}, fairness_weight=0.5 fixture"
        )
    return ok


def benchmark_apply_k_cps_constraint() -> bool:
    fcfs_order, _runways, runway_last_committed = _make_k_cps_fixture()
    current_time = 1000.0

    legacy_manager = _make_manager(fairness_weight=0.5)
    legacy_orders = [copy.deepcopy(fcfs_order) for _ in range(N_BENCHMARK_REPS)]
    t0 = time.perf_counter()
    for order in legacy_orders:
        legacy_manager._runway_last_committed = dict(runway_last_committed)
        _legacy_apply_k_cps_constraint(legacy_manager, order, current_time)
    legacy_elapsed = time.perf_counter() - t0

    vector_manager = _make_manager(fairness_weight=0.5)
    vector_orders = [copy.deepcopy(fcfs_order) for _ in range(N_BENCHMARK_REPS)]
    t0 = time.perf_counter()
    for order in vector_orders:
        vector_manager._runway_last_committed = dict(runway_last_committed)
        vector_manager._apply_k_cps_constraint(order, current_time)
    vector_elapsed = time.perf_counter() - t0

    speedup = legacy_elapsed / vector_elapsed if vector_elapsed > 0 else float("inf")
    print(
        f"_apply_k_cps_constraint x{N_BENCHMARK_REPS}: "
        f"legacy={legacy_elapsed*1e3:.2f}ms  vectorized={vector_elapsed*1e3:.2f}ms  "
        f"speedup={speedup:.2f}x"
    )
    ok = vector_elapsed < legacy_elapsed
    print("PASS" if ok else "FAIL", ": vectorized rewrite is faster than legacy")
    return ok


if __name__ == "__main__":
    passed_parity_1 = check_assign_runways_dynamic_parity()
    passed_parity_3 = check_apply_k_cps_constraint_parity()
    passed_perf_1 = benchmark_assign_runways_dynamic()
    passed_perf_3 = benchmark_apply_k_cps_constraint()
    raise SystemExit(
        0 if (passed_parity_1 and passed_parity_3 and passed_perf_1 and passed_perf_3) else 1
    )
