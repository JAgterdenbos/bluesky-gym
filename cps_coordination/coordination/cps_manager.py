"""
cps_coordination/coordination/cps_manager.py
---------------------------------------------
CPSManager: Hierarchical Constrained Position Shifting sequence manager.

Implements the k-CPS coordination algorithm described in the project
methodology:

  1. Maintain and update an ordered fleet of N_a incoming aircraft.
  2. Compute the FCFS reference sequence by ascending absolute ETA
     (t + T̂_i), where T̂_i is predicted by the ETASurrogate.
  3. Apply the k-CPS constraint: no aircraft may shift more than k
     positions from its FCFS rank.
  4. Run the greedy forward scheduler:
       TTA_i = max(ETA_i, TTA_{i-1} + ΔT_sep)
     where ΔT_sep is read from the RECAT-EU wake turbulence matrix.
  5. Support 'static' (sector-committed IAFs) and 'dynamic'
     (minimum-feasible-TTA selection) runway assignment modes.
  6. Re-evaluate every delta_t_plan simulation seconds; propagate
     goal updates when the shift exceeds delta_update seconds.

Optional TrajectoryBuffer
-------------------------
When a :class:`~cps_coordination.coordination.trajectory_buffer.TrajectoryBuffer`
is supplied at construction, ``update_fleet`` pushes each aircraft's
current position and heading into the buffer every step and retrieves
lag features (delta_atd, cumabs_cte, heading_volatility) to pass to the
ETASurrogate.  Without a buffer, lag features default to zeros, which
yields gracefully degraded (but functional) ETA predictions if the
surrogate was trained with lag features.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, get_args, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cps_coordination.coordination.eta_surrogate import ETASurrogate
    from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Aircraft state record
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AircraftState:
    """Lightweight record for a single aircraft tracked by CPSManager.

    Attributes
    ----------
    acid : str
        Aircraft callsign / unique identifier.
    state : np.ndarray
        Current observation feature vector fed to the ETASurrogate.
        Expected layout: ``[x, y, elapsed_steps, heading_deg_bearing]``.
    runway_id : str
        Currently assigned (or candidate) landing runway identifier.
    eta : float
        Absolute estimated arrival time at the IAF (seconds), refreshed
        each planning cycle by the ETASurrogate.
    tta : float
        Target Time of Arrival assigned by the greedy forward scheduler.
        Initialised to ``math.inf`` as a sentinel; set finite by
        :meth:`CPSManager._greedy_schedule`.
    fcfs_rank : int
        0-indexed position in the current FCFS reference sequence.
    wake_cat : str
        RECAT-EU wake turbulence category letter (``"A"``–``"F"``).
    """

    acid: str
    state: np.ndarray
    runway_id: str
    eta: float
    tta: float = math.inf
    fcfs_rank: int = 0
    wake_cat: str = "C"


# ──────────────────────────────────────────────────────────────────────────────
# CPSManager
# ──────────────────────────────────────────────────────────────────────────────

RunwayAssignmentMode = Literal["static", "dynamic"]


class CPSManager:
    """Hierarchical k-CPS sequence manager and TTA assigner.

    Parameters
    ----------
    k_cps : int
        Maximum positional shift from the FCFS rank.
        ``k_cps=0`` degenerates to pure FCFS.
    recat_matrix : Dict[str, Dict[str, float]]
        RECAT-EU time-based separation matrix.
        ``recat_matrix[leading_cat][trailing_cat]`` → seconds.
    runway_assignment_mode : RunwayAssignmentMode
        ``"static"`` — runway committed at sector entry and never changed.
        ``"dynamic"`` — re-selects the runway yielding the minimum
        feasible TTA each replanning cycle.
    delta_t_plan : int
        Replanning interval in simulation seconds.
    delta_update : float
        Minimum TTA change (seconds) that triggers a goal notification.
    available_runways : List[str], optional
        Candidate runways (required for dynamic mode).
    trajectory_buffer : TrajectoryBuffer, optional
        Per-aircraft rolling history used to compute lag features
        (delta_atd, cumabs_cte, heading_volatility) for the surrogate.
        When ``None``, lag features are implicitly zero.
    """

    __name__ = "CPSManager"
    _valid_modes = set(get_args(RunwayAssignmentMode))

    def __init__(
        self,
        k_cps: int,
        recat_matrix: Dict[str, Dict[str, float]],
        runway_assignment_mode: RunwayAssignmentMode = "dynamic",
        delta_t_plan: int = 60,
        delta_update: float = 1.0,
        available_runways: Optional[List[str]] = None,
        trajectory_buffer: Optional["TrajectoryBuffer"] = None,
    ) -> None:
        if runway_assignment_mode not in self._valid_modes:
            raise ValueError(
                f"runway_assignment_mode must be one of {self._valid_modes!r}, "
                f"got {runway_assignment_mode!r}"
            )
        self.k_cps = k_cps
        self.recat_matrix = recat_matrix
        self.runway_assignment_mode: RunwayAssignmentMode = runway_assignment_mode
        self.delta_t_plan = delta_t_plan
        self.delta_update = delta_update
        self.available_runways: List[str] = available_runways or []
        self._trajectory_buffer = trajectory_buffer

        self._fleet: List[AircraftState] = []
        self._fleet_index: Dict[str, int] = {}
        self._prev_ttas: Dict[str, float] = {}
        self._last_plan_time: float = -math.inf

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_fleet(
        self,
        aircraft: List[AircraftState],
        current_time: float,
        surrogate: Optional["ETASurrogate"] = None,
    ) -> List[str]:
        """Ingest a new fleet snapshot and run a replan when due.

        Trajectory buffer update
        ~~~~~~~~~~~~~~~~~~~~~~~~
        If a ``trajectory_buffer`` was supplied, this method:

        1. Evicts callsigns no longer in the fleet (departed aircraft).
        2. Pushes each active aircraft's ``(x, y, heading_rad)`` into the
           buffer so lag features accumulate step-by-step.
        3. Retrieves the current lag feature matrix ``(n, 3)`` for the
           surrogate.

        ETA refresh strategy
        ~~~~~~~~~~~~~~~~~~~~
        *Dynamic mode*: ``_assign_runways_dynamic`` evaluates every
        (aircraft × runway) pair in one vectorised call and writes both
        ``runway_id`` and ``eta`` — no separate ETA refresh needed.

        *Static mode*: ETAs are refreshed once per planning cycle,
        immediately before ``_replan`` reads them.

        Parameters
        ----------
        aircraft : List[AircraftState]
            Current snapshot of all active aircraft.
        current_time : float
            Current simulation clock in seconds.
        surrogate : ETASurrogate, optional
            ETA prediction model.

        Returns
        -------
        List[str]
            Callsigns whose TTA changed by more than ``delta_update`` seconds.
        """
        old_acids = {ac.acid for ac in self._fleet}
        self._fleet = aircraft
        self._fleet_index = {ac.acid: i for i, ac in enumerate(aircraft)}
        new_acids = {ac.acid for ac in aircraft}

        # Update trajectory buffer
        if self._trajectory_buffer is not None:
            for acid in old_acids - new_acids:
                self._trajectory_buffer.evict(acid)
            for ac in self._fleet:
                # state[3] is heading_deg_bearing → convert to radians for buffer
                self._trajectory_buffer.push(
                    ac.acid,
                    float(ac.state[0]),
                    float(ac.state[1]),
                    float(np.deg2rad(ac.state[3])),
                )

        # Compute lag features for this step
        lag_features = self._get_lag_features(surrogate)

        if self.runway_assignment_mode == "dynamic":
            self._assign_runways_dynamic(surrogate, current_time, lag_features)

        if current_time - self._last_plan_time >= self.delta_t_plan:
            if surrogate is not None and self.runway_assignment_mode != "dynamic":
                self._refresh_etas(surrogate, current_time, lag_features)
            self._replan(current_time)
            self._last_plan_time = current_time

        return self._collect_changed_targets()

    def get_tta(self, acid: str) -> Optional[float]:
        """Return the current TTA for *acid*, or ``None`` if unknown.

        O(1) lookup via ``_fleet_index``.
        """
        idx = self._fleet_index.get(acid)
        if idx is None:
            return None
        tta = self._fleet[idx].tta
        return None if math.isinf(tta) else tta

    def get_sequence(self) -> List[AircraftState]:
        """Return the scheduled sequence sorted by ascending TTA."""
        return sorted(self._fleet, key=lambda a: a.tta)

    def reset(self) -> None:
        """Clear internal state for the start of a new episode."""
        self._fleet = []
        self._fleet_index = {}
        self._prev_ttas = {}
        self._last_plan_time = -math.inf
        if self._trajectory_buffer is not None:
            self._trajectory_buffer.reset()

    # ------------------------------------------------------------------ #
    # Core scheduling pipeline
    # ------------------------------------------------------------------ #

    def _replan(self, _current_time: float) -> None:
        """Full replanning pass: FCFS → k-CPS permutation → greedy TTAs."""
        if not self._fleet:
            return
        fcfs_order = self._fcfs_order()
        for rank, ac in enumerate(fcfs_order):
            ac.fcfs_rank = rank
        optimised = self._apply_k_cps_constraint(fcfs_order)
        self._greedy_schedule(optimised)

    def _fcfs_order(self) -> List[AircraftState]:
        """Sort fleet by ascending absolute ETA → FCFS reference sequence."""
        return sorted(self._fleet, key=lambda a: (a.eta, a.acid))

    def _apply_k_cps_constraint(
        self, fcfs_order: List[AircraftState]
    ) -> List[AircraftState]:
        """Return a permutation that satisfies the k-CPS window.

        Strategy — greedy forward sweep:
          At scheduling position ``pos``, pick the eligible aircraft
          (FCFS rank in ``[pos−k, pos+k]``) with the earliest ETA.
          Ties are broken by FCFS rank for stability.

        For ``k_cps=0`` this is pure FCFS.  For ``k_cps ≥ n−1`` the
        constraint is inactive and the scheduler is free to optimise fully.

        Parameters
        ----------
        fcfs_order : List[AircraftState]

        Returns
        -------
        List[AircraftState]
        """
        if self.k_cps == 0:
            return list(fcfs_order)

        n = len(fcfs_order)
        k = self.k_cps
        heap: list = []
        scheduled_mask = [False] * n
        scheduled: List[AircraftState] = []
        next_to_add = 0

        for pos in range(n):
            while next_to_add <= min(pos + k, n - 1):
                ac = fcfs_order[next_to_add]
                heapq.heappush(heap, (ac.eta, ac.fcfs_rank, next_to_add))
                next_to_add += 1

            best_idx: Optional[int] = None
            while heap:
                _, _, idx = heap[0]
                if scheduled_mask[idx] or idx < pos - k:
                    heapq.heappop(heap)
                    continue
                best_idx = idx
                heapq.heappop(heap)
                break

            if best_idx is None:
                best_idx = next(i for i in range(n) if not scheduled_mask[i])

            scheduled_mask[best_idx] = True
            scheduled.append(fcfs_order[best_idx])

        return scheduled

    def _greedy_schedule(self, sequence: List[AircraftState]) -> None:
        """Assign TTAs via the greedy forward rule with per-runway tracking.

        For each aircraft in the k-CPS-constrained sequence:

          TTA_i = max(ETA_i, TTA_{prev_on_same_runway} + ΔT_sep(prev, i))

        Each runway is tracked independently.

        Parameters
        ----------
        sequence : List[AircraftState]
        """
        runway_last: Dict[str, AircraftState] = {}

        for ac in sequence:
            rwy = ac.runway_id
            if rwy not in runway_last:
                ac.tta = ac.eta
            else:
                prev = runway_last[rwy]
                sep = self._get_separation(prev.wake_cat, ac.wake_cat)
                ac.tta = max(ac.eta, prev.tta + sep)
            runway_last[rwy] = ac

    def _get_separation(self, leading_cat: str, trailing_cat: str) -> float:
        """Look up RECAT-EU minimum time separation (seconds).

        Falls back to 90 s (conservative Category C/C default) when the
        combination is absent from the matrix.
        """
        return float(
            self.recat_matrix.get(leading_cat, {}).get(trailing_cat, 90.0)
        )

    # ------------------------------------------------------------------ #
    # Dynamic runway assignment
    # ------------------------------------------------------------------ #

    def _assign_runways_dynamic(
        self,
        surrogate: Optional["ETASurrogate"],
        current_time: float,
        lag_features: Optional[np.ndarray],
    ) -> None:
        """Re-select each aircraft's runway to minimise its predicted ETA.

        Evaluates every (aircraft × runway) pair in a single vectorised
        model call, then takes the argmin over the runway axis.

        Parameters
        ----------
        surrogate : ETASurrogate, optional
        current_time : float
        lag_features : np.ndarray, shape (n, 3), optional
        """
        if surrogate is None or not self.available_runways or not self._fleet:
            return

        states = np.stack([ac.state for ac in self._fleet])  # (n, 4)
        eta_matrix = surrogate.predict_eta_fleet_all_runways(
            states, self.available_runways, current_time, lag_features
        )  # (n, r)

        best_rwy_idx = np.argmin(eta_matrix, axis=1)  # (n,)
        for i, ac in enumerate(self._fleet):
            j = int(best_rwy_idx[i])
            ac.runway_id = self.available_runways[j]
            ac.eta = float(eta_matrix[i, j])

    # ------------------------------------------------------------------ #
    # ETA refresh + change-detection
    # ------------------------------------------------------------------ #

    def _get_lag_features(
        self,
        surrogate: Optional["ETASurrogate"],
    ) -> Optional[np.ndarray]:
        """Compute lag features for the full fleet from the trajectory buffer.

        Returns ``None`` when no buffer is present or when the surrogate
        has no IAF reference (e.g. not yet loaded).
        """
        if self._trajectory_buffer is None or not self._fleet:
            return None
        if surrogate is None or not surrogate._iaf_ref:
            return None

        iaf_data = [
            surrogate._iaf_ref.get(ac.runway_id, (0.0, 0.0, 0.0))
            for ac in self._fleet
        ]
        iaf_xs = np.array([d[0] for d in iaf_data])
        iaf_ys = np.array([d[1] for d in iaf_data])
        iaf_ahs = np.array([d[2] for d in iaf_data])

        return self._trajectory_buffer.get_lag_features_batch(
            [ac.acid for ac in self._fleet], iaf_xs, iaf_ys, iaf_ahs
        )

    def _refresh_etas(
        self,
        surrogate: "ETASurrogate",
        current_time: float,
        lag_features: Optional[np.ndarray],
    ) -> None:
        """Re-estimate ETA for every aircraft in one vectorised call.

        Parameters
        ----------
        surrogate : ETASurrogate
        current_time : float
        lag_features : np.ndarray, shape (n, 3), optional
        """
        if not self._fleet:
            return
        states = np.stack([ac.state for ac in self._fleet])       # (n, 4)
        runway_ids = [ac.runway_id for ac in self._fleet]
        etas = surrogate.predict_eta_fleet(
            states, runway_ids, current_time, lag_features
        )  # (n,)
        for i, ac in enumerate(self._fleet):
            ac.eta = float(etas[i])

    def _collect_changed_targets(self) -> List[str]:
        """Return callsigns whose TTA shifted by more than ``delta_update`` s.

        Also evicts departed aircraft from ``_prev_ttas`` so a later flight
        reusing the same callsign always triggers a fresh notification.
        """
        active_acids = {ac.acid for ac in self._fleet}
        stale = [acid for acid in self._prev_ttas if acid not in active_acids]
        for acid in stale:
            del self._prev_ttas[acid]

        changed: List[str] = []
        for ac in self._fleet:
            if math.isinf(ac.tta):
                continue
            prev = self._prev_ttas.get(ac.acid)
            if prev is None or abs(ac.tta - prev) > self.delta_update:
                changed.append(ac.acid)
                self._prev_ttas[ac.acid] = ac.tta
        return changed
