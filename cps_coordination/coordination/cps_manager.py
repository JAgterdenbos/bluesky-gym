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
     (minimum-feasible-TTA selection, subject to the same k-CPS fairness
     bound on the candidate runway's projected position) runway assignment
     modes.
  6. Re-evaluate every delta_t_plan simulation seconds; propagate
     goal updates when the shift exceeds delta_update seconds.
  7. Detect aircraft whose distance to their IAF hasn't shrunk over a
     rolling window ("stalled") and freeze their ETA/runway rather than
     continuing to re-target them every cycle -- see STALL_WINDOW_S /
     is_stalled() and the class docstring for why this exists.

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

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, get_args, TYPE_CHECKING

import numpy as np

from bluesky_gym.envs.pathplanning_goal_env import MAX_DISTANCE

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
        Expected layout: ``[x, y, elapsed_steps, heading_deg_bearing,
        remaining_time_budget]``. The 5th element (Finding 2's
        goal-conditioned active temporal target minus elapsed time) is
        computed by ``_build_fleet`` -- unlike lag features, it needs no
        rolling buffer, so it travels here rather than as a separate
        manager-level argument. ``CPSManager`` splits it back out before
        calling the surrogate, mirroring how lag features are passed
        separately (see ``_refresh_etas``/``_assign_runways_dynamic``).
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
        ``"dynamic"`` — re-selects, every step, the runway yielding the
        minimum feasible TTA among those satisfying the k-CPS fairness
        bound (see :meth:`_assign_runways_dynamic`).
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
    enable_stall_detection : bool
        If True (default), freeze a flagged-stalled aircraft's ETA/runway
        instead of continuing to re-target it every cycle (see the
        STALL_WINDOW_S class docstring above). Detection itself
        (``is_stalled``) always runs regardless of this flag, so
        ``stall_detected`` telemetry is available either way -- set this to
        False to reproduce the pre-mitigation behaviour exactly, e.g. for a
        before/after ablation comparison.
    fairness_weight : float
        Weight (w2) on the slack-protection term in
        :meth:`_apply_k_cps_constraint`'s cost rule. ``0.0`` (default)
        reproduces exact FCFS ordering -- provably total-delay-optimal
        under wake-homogeneity (§2.2), so this is a safe, opt-in
        generalisation. ``> 0.0`` biases the k-CPS window's greedy
        selection to prioritise aircraft with less remaining slack (and
        any already-``is_stalled``-flagged aircraft) for earlier, lower-
        imposed-delay scheduling positions, at the cost of imposing more
        delay on higher-slack aircraft instead.
    enable_cross_cycle_runway_seeding : bool
        If True (default), :meth:`_tta_for` seeds a runway's "previous
        scheduled aircraft" from ``self._runway_last_committed`` (persisted
        across replanning cycles) whenever the current cycle's own sequence
        doesn't contain an earlier aircraft on that runway. Set False
        (Test C, pre-Step-10 audit §1.2) to isolate the surrogate-feature
        feedback loop (§1.1) from this second, structurally distinct
        cross-cycle ratcheting channel.
    """

    __name__ = "CPSManager"
    _valid_modes = set(get_args(RunwayAssignmentMode))

    # Stall detection: an aircraft that hasn't beaten its best-ever distance
    # to its IAF by at least STALL_PROGRESS_EPS_KM within the last
    # STALL_WINDOW_S seconds is flagged as "stalled" and its ETA/runway stop
    # being refreshed (see _update_stall_tracking). This exists because every
    # replanning cycle re-derives ETA/TTA from the aircraft's *current*
    # position -- an aircraft that isn't making progress (e.g. stuck in a
    # holding loop for reasons unrelated to CPS, confirmed by direct position
    # tracing) gets a perpetually-refreshed "just barely feasible from here"
    # target instead of a fixed one, so the RTA-error signal never
    # accumulates enough divergence to register as "late" -- it looks
    # achievable at every single snapshot even though the aircraft never
    # converges. A feasibility-margin sweep (0-1800s) was tried first and
    # empirically falsified: it doesn't rescue stalled aircraft and actively
    # pushes previously-converging ones into the same failure by making
    # every target later. Freezing the last good ETA/TTA once stalled at
    # least stops the CPS layer from perpetuating an unbounded spiral; it
    # does not (and cannot, since the worker policy is frozen) force the
    # aircraft to actually converge.
    #
    # STALL_WINDOW_S=1800 (30 min) rather than a short window: a genuinely
    # circling aircraft's distance-to-IAF oscillates by tens of km within
    # any few-hundred-second span (confirmed empirically), so a short window
    # sees spurious "new bests" from that noise alone and never fires. 30
    # minutes is long enough to average out that noise while still catching
    # the failure with most of a 6h episode left to matter.
    STALL_WINDOW_S = 1800.0
    STALL_PROGRESS_EPS_KM = 5.0

    # Fairness-weighted k-CPS selection (_apply_k_cps_constraint, §2.3 of the
    # pre-Step-10 audit): reference scale (seconds) for slack_penalty --
    # matches CPSEnvKwargsConfig's own documented ~20-minute path-stretching
    # delay-absorption capacity for the frozen worker (see
    # experiments/config.py's reduced_wake_separation docstring), so an
    # aircraft's slack_penalty saturates to 0 once it has at least that much
    # margin before its own predicted arrival, and grows as margin shrinks
    # below it. STALL_SLACK_PENALTY_BOOST_S adds a fixed extra boost for any
    # aircraft already flagged is_stalled, on top of its margin-derived
    # penalty (see _slack_penalty).
    SLACK_REFERENCE_S = 1200.0
    STALL_SLACK_PENALTY_BOOST_S = 1200.0

    def __init__(
        self,
        k_cps: int,
        recat_matrix: Dict[str, Dict[str, float]],
        runway_assignment_mode: RunwayAssignmentMode = "dynamic",
        delta_t_plan: int = 60,
        delta_update: float = 1.0,
        available_runways: Optional[List[str]] = None,
        trajectory_buffer: Optional["TrajectoryBuffer"] = None,
        enable_stall_detection: bool = True,
        fairness_weight: float = 0.0,
        enable_cross_cycle_runway_seeding: bool = True,
    ) -> None:
        if runway_assignment_mode not in self._valid_modes:
            raise ValueError(
                f"runway_assignment_mode must be one of {self._valid_modes!r}, "
                f"got {runway_assignment_mode!r}"
            )
        self.k_cps = k_cps
        # w2 in _apply_k_cps_constraint's cost rule (§2.3): 0.0 (default)
        # short-circuits the k-CPS permutation to exact FCFS -- provably the
        # total-delay-optimal schedule under today's wake-homogeneity
        # assumption (§2.2) -- so this is a strict, opt-in generalisation of
        # the pre-fix no-op behaviour, not a behaviour change at the default.
        self.fairness_weight = fairness_weight
        # Test C (pre-Step-10 audit §1.2): isolates the surrogate-feature
        # feedback loop (§1.1) from the *second*, structurally distinct
        # compounding channel in _tta_for -- _runway_last_committed persisting
        # a runway's last TTA across replanning cycles even once its owning
        # aircraft has left the active fleet. True (default) reproduces
        # today's behaviour exactly; set False (typically on a single-
        # aircraft runway, so there is no other aircraft to separate
        # against) to confirm whether the surrogate-feature loop alone is
        # sufficient to produce a stall, independent of this ratchet.
        self.enable_cross_cycle_runway_seeding = enable_cross_cycle_runway_seeding
        # Detection/tracking (_update_stall_tracking, is_stalled) always
        # runs regardless of this flag, so `stall_detected` is always
        # available in telemetry for an ablation comparison -- this flag
        # only controls whether _refresh_etas/_assign_runways_dynamic
        # actually freeze a flagged aircraft's ETA/runway.
        self.enable_stall_detection = enable_stall_detection
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
        # Last (tta, wake_cat, acid) committed per runway, persisted across
        # replanning cycles even after the responsible aircraft leaves the
        # active fleet — see _greedy_schedule. acid is tracked so a still-
        # active aircraft that remains the sole occupant of its runway
        # across consecutive replanning cycles is never separated against
        # its own prior commit (see _greedy_schedule's self-separation note).
        self._runway_last_committed: Dict[str, Tuple[float, str, str]] = {}
        # Stall detection: acid -> best (smallest) distance-to-IAF ever seen
        # and the sim time it was achieved at; acid -> flagged once stalled.
        self._best_distance_km: Dict[str, float] = {}
        self._best_distance_time: Dict[str, float] = {}
        self._stalled_acids: set = set()
        # Snapshot of eta/runway_id taken the cycle an acid is first flagged
        # stalled, re-applied every subsequent cycle (see update_fleet) so
        # the frozen target actually stays fixed rather than drifting with
        # the caller's freshly-reseeded per-cycle naive ETA.
        self._frozen_eta: Dict[str, float] = {}
        self._frozen_runway: Dict[str, str] = {}
        self._frozen_tta: Dict[str, float] = {}

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

        # Re-pin already-stalled aircraft to their frozen eta/runway. This
        # has to happen here, not just via the "don't overwrite" skips in
        # _refresh_etas/_assign_runways_dynamic below: the caller
        # (CPSCoordinationExperiment._build_fleet) constructs a brand-new
        # AircraftState every cycle with a freshly-recomputed naive-ETA
        # placeholder, so merely not touching `aircraft` here would still
        # leave a moving (still-drifting-with-position) value in place --
        # confirmed empirically (a first version of this without the
        # re-pin never actually stopped an already-flagged aircraft's
        # eta/tta from changing cycle to cycle).
        if self.enable_stall_detection:
            for ac in self._fleet:
                if ac.acid in self._stalled_acids and ac.acid in self._frozen_eta:
                    ac.eta = self._frozen_eta[ac.acid]
                    ac.runway_id = self._frozen_runway[ac.acid]
                    ac.tta = self._frozen_tta[ac.acid]

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

        for acid in old_acids - new_acids:
            self._best_distance_km.pop(acid, None)
            self._best_distance_time.pop(acid, None)
            self._stalled_acids.discard(acid)
            self._frozen_eta.pop(acid, None)
            self._frozen_runway.pop(acid, None)
            self._frozen_tta.pop(acid, None)
        self._update_stall_tracking(surrogate, current_time)

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
        self._runway_last_committed = {}
        self._best_distance_km = {}
        self._best_distance_time = {}
        self._stalled_acids = set()
        self._frozen_eta = {}
        self._frozen_runway = {}
        self._frozen_tta = {}
        if self._trajectory_buffer is not None:
            self._trajectory_buffer.reset()

    def is_stalled(self, acid: str) -> bool:
        """Return whether *acid* has been flagged as stalled this episode.

        Once flagged, an aircraft stays flagged for the rest of the episode
        (see :meth:`_update_stall_tracking`) -- there is no un-stalling,
        since the point is to stop chasing a target that already proved
        unreachable, not to re-engage the same feedback loop the moment
        distance happens to dip briefly.
        """
        return acid in self._stalled_acids

    # ------------------------------------------------------------------ #
    # Core scheduling pipeline
    # ------------------------------------------------------------------ #

    def _replan(self, current_time: float) -> None:
        """Full replanning pass: FCFS → k-CPS permutation → greedy TTAs.

        Stalled aircraft (see :meth:`_update_stall_tracking`) are excluded
        from the k-CPS permutation and greedy scheduling entirely, not just
        from having their own TTA touched: a frozen-but-never-landing
        aircraft that stayed in the sequence would force every *other*,
        still-converging aircraft on the same runway to keep separating
        against it too, jamming the runway around a landing that never
        happens. It would also re-trigger the exact ratcheting pattern
        :meth:`_greedy_schedule` already guards against for a single
        self-separating aircraft, except pairwise between two stalled
        aircraft that keep swapping FCFS order relative to each other's
        ever-growing prior TTA (confirmed empirically: freezing eta alone,
        without this exclusion, still produced unbounded TTA growth for two
        simultaneously-stalled aircraft separating against each other every
        cycle). fcfs_rank is still assigned to every aircraft for telemetry
        consistency; only participation in the runway timeline is excluded.
        """
        if not self._fleet:
            return
        fcfs_order = self._fcfs_order()
        for rank, ac in enumerate(fcfs_order):
            ac.fcfs_rank = rank
        active = (
            [ac for ac in fcfs_order if ac.acid not in self._stalled_acids]
            if self.enable_stall_detection
            else fcfs_order
        )
        optimised = self._apply_k_cps_constraint(active, current_time)
        self._greedy_schedule(optimised)

    def _fcfs_order(self) -> List[AircraftState]:
        """Sort fleet by ascending absolute ETA → FCFS reference sequence."""
        return sorted(self._fleet, key=lambda a: (a.eta, a.acid))

    def _apply_k_cps_constraint(
        self, fcfs_order: List[AircraftState], current_time: float = 0.0,
    ) -> List[AircraftState]:
        """Return a permutation that satisfies the k-CPS window.

        ``fairness_weight == 0.0`` (default): returns ``fcfs_order``
        unchanged. §2.1 of the pre-Step-10 audit proves the pre-fix
        heap-based "earliest ETA in window" selection is *always* a no-op
        identity permutation on FCFS-sorted input (regardless of ``k_cps``),
        and §2.2 proves that identity permutation is provably total-delay-
        optimal under today's wake-homogeneity assumption. Short-circuiting
        here reproduces that exactly, by construction, rather than relying
        on the cost rule below to happen to reduce to the same output.

        ``fairness_weight > 0.0``: greedy forward sweep over scheduling
        positions ``pos = 0..n-1``. At each position, among the eligible
        window ``{idx ∈ [pos-k, min(pos+k, n-1)] : not yet scheduled}``,
        selects the candidate minimising

            cost(idx) = imposed_delay(idx) − fairness_weight · slack_penalty(idx)

        where ``imposed_delay(idx)`` is the runway-contention delay this
        candidate would incur if scheduled at ``pos`` (mirroring
        :meth:`_greedy_schedule`'s own per-runway ``max(eta, prev_tta+sep)``
        rule via a virtual, not-yet-committed copy of that tracking state —
        see :meth:`_tta_for`), and ``slack_penalty(idx)`` (see
        :meth:`_slack_penalty`) is large for aircraft with little remaining
        margin before their own predicted arrival (or already flagged
        ``is_stalled``).

        The slack term is *subtracted*, not added: a myopic per-position
        argmin that instead ADDS a "badness" score for fragile aircraft
        would systematically lose them the cheap, low-delay positions early
        in the sweep (since a low-penalty candidate always looks cheaper at
        a tied delay) and defer them to the most-delayed leftover slots —
        exactly backwards from the intent stated in §2.3 ("prefer to impose
        the runway-contention delay on the aircraft with the most slack to
        absorb it safely"). Subtracting the (non-negative) slack_penalty
        makes a fragile candidate's cost *lower*, so it wins the position
        currently being filled — and since the sweep fills positions in
        increasing-imposed-delay order, that means fragile aircraft
        preferentially claim the earliest, cheapest slots instead of being
        left for the most-delayed ones.

        Complexity: O(n·(2k+1)) — every eligible candidate's cost is
        evaluated directly each position; no heap, since a non-monotonic
        cost function can't reuse the sorted-input shortcut the old
        heap-based approach relied on.

        Parameters
        ----------
        fcfs_order : List[AircraftState]
        current_time : float
            Current simulation clock (seconds) — used by
            :meth:`_slack_penalty` to derive each candidate's remaining
            margin before its own predicted arrival. Unused when
            ``fairness_weight == 0.0``.

        Returns
        -------
        List[AircraftState]
        """
        if self.k_cps == 0 or self.fairness_weight <= 0.0:
            return list(fcfs_order)

        n = len(fcfs_order)
        k = self.k_cps
        scheduled_mask = [False] * n
        scheduled: List[AircraftState] = []
        runway_last: Dict[str, AircraftState] = {}  # virtual -- mirrors _greedy_schedule

        for pos in range(n):
            window_lo = max(0, pos - k)
            window_hi = min(pos + k, n - 1)
            eligible = [i for i in range(window_lo, window_hi + 1) if not scheduled_mask[i]]
            if not eligible:
                # An earlier position can pull an index out of a *later*
                # position's own window before that position is reached
                # (the window shifts every step, it isn't reserved) -- rare,
                # but the old heap-based code had the identical fallback for
                # the identical reason: fall back to any remaining
                # unscheduled index. n - pos unscheduled indices always
                # remain at this point, so this is never itself empty.
                eligible = [i for i in range(n) if not scheduled_mask[i]]

            best_idx: Optional[int] = None
            best_cost: Optional[float] = None
            best_tta: float = 0.0
            for idx in eligible:
                ac = fcfs_order[idx]
                tta_if_here = self._tta_for(ac, runway_last)
                imposed_delay = max(0.0, tta_if_here - ac.eta)
                cost = imposed_delay - self.fairness_weight * self._slack_penalty(ac, current_time)
                if best_cost is None or cost < best_cost:
                    best_idx, best_cost, best_tta = idx, cost, tta_if_here

            chosen = fcfs_order[best_idx]  # type: ignore[arg-type]
            scheduled_mask[best_idx] = True
            scheduled.append(chosen)
            # Commit the virtual TTA now so later positions on the same
            # runway see it via `runway_last`, exactly mirroring
            # _greedy_schedule's own incremental state. _greedy_schedule
            # re-derives (and overwrites) the identical value when it
            # processes `scheduled` afterward, since it walks the same
            # order through the same _tta_for rule from the same starting
            # state (self._runway_last_committed, untouched until then).
            chosen.tta = best_tta
            runway_last[chosen.runway_id] = chosen

        return scheduled

    def _slack_penalty(self, ac: AircraftState, current_time: float) -> float:
        """Non-negative "fragility" score for the k-CPS fairness cost rule.

        Monotonically decreasing in the aircraft's own remaining margin
        (``eta - current_time``, a lighter-weight stand-in for the
        surrogate's ``naive_eta_remaining`` feature that avoids threading
        IAF-reference access into the k-CPS selection path for what is an
        ablation knob — both represent the same physical quantity, remaining
        time before this aircraft's own earliest feasible arrival): saturates
        to 0 once margin reaches ``SLACK_REFERENCE_S`` (the frozen worker's
        documented ~20-minute delay-absorption capacity), and grows linearly
        as margin shrinks below it. Any aircraft already flagged
        ``is_stalled`` gets a fixed additional boost, so the k-CPS layer
        actively avoids compounding a fragile aircraft's delay (§2.3).
        """
        margin = ac.eta - current_time
        penalty = max(0.0, self.SLACK_REFERENCE_S - margin)
        if ac.acid in self._stalled_acids:
            penalty += self.STALL_SLACK_PENALTY_BOOST_S
        return penalty

    def _tta_for(self, ac: AircraftState, runway_last: Dict[str, AircraftState]) -> float:
        """Compute what ``ac.tta`` would be if scheduled next on its own
        runway, given a per-runway "last scheduled aircraft" tracker.

        Shared by :meth:`_greedy_schedule` (the real, committing pass) and
        :meth:`_apply_k_cps_constraint` (a virtual, non-committing pass that
        evaluates candidates before an order is finalised) so the two can
        never disagree — see :meth:`_apply_k_cps_constraint`'s docstring.

        That seeded value (from ``self._runway_last_committed``, persisted
        across replanning cycles) is only a genuine "previous, already-
        departed aircraft" when its acid differs from ``ac``'s own. When the
        same aircraft remains the sole occupant of a runway across
        consecutive replanning cycles (the common case, since
        ``delta_t_plan`` is typically equal to ``ACTION_TIME`` — every
        decision step replans), it would otherwise read back its own prior
        commit and separate against itself, ratcheting its tta upward by
        ``sep`` every cycle indefinitely (confirmed bug: this alone produced
        multi-thousand-second tta/landing-time divergence for a single
        aircraft with no second aircraft ever involved). In that self-match
        case the tta is just the aircraft's own current eta — no other
        aircraft to separate from.

        When ``self.enable_cross_cycle_runway_seeding`` is False (Test C,
        §1.2), this cross-cycle seed is skipped entirely -- each replanning
        cycle only separates against aircraft *currently* in ``sequence``,
        isolating the surrogate-feature feedback loop (§1.1) from this
        second, structurally distinct ratcheting channel.
        """
        rwy = ac.runway_id
        if rwy in runway_last:
            prev = runway_last[rwy]
            sep = self._get_separation(prev.wake_cat, ac.wake_cat)
            return max(ac.eta, prev.tta + sep)
        if self.enable_cross_cycle_runway_seeding and rwy in self._runway_last_committed:
            prev_tta, prev_wake_cat, prev_acid = self._runway_last_committed[rwy]
            if prev_acid == ac.acid:
                return ac.eta
            sep = self._get_separation(prev_wake_cat, ac.wake_cat)
            return max(ac.eta, prev_tta + sep)
        return ac.eta

    def _greedy_schedule(self, sequence: List[AircraftState]) -> None:
        """Assign TTAs via the greedy forward rule with per-runway tracking.

        For each aircraft in the k-CPS-constrained sequence:

          TTA_i = max(ETA_i, TTA_{prev_on_same_runway} + ΔT_sep(prev, i))

        Each runway is tracked independently — see :meth:`_tta_for` for the
        per-aircraft rule (shared with :meth:`_apply_k_cps_constraint`'s
        virtual evaluation pass) and the cross-cycle seeding/self-match
        notes. Once every aircraft in *sequence* has been assigned a tta,
        the final per-runway state is committed to
        ``self._runway_last_committed`` for the next replanning cycle.

        Parameters
        ----------
        sequence : List[AircraftState]
        """
        runway_last: Dict[str, AircraftState] = {}

        for ac in sequence:
            ac.tta = self._tta_for(ac, runway_last)
            runway_last[ac.runway_id] = ac

        for rwy, ac in runway_last.items():
            self._runway_last_committed[rwy] = (ac.tta, ac.wake_cat, ac.acid)

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
        """Re-select each aircraft's runway to minimise its predicted ETA,
        subject to the k-CPS fairness constraint (Eq. dynamic_runway_assignment
        in the thesis methodology):

            r_i* = argmin_r TTA_i^(r)  s.t.  |sigma_i^(r) - sigma_i^FCFS| <= k

        A candidate runway is only eligible if reassigning aircraft i to it
        would not shift i's position, relative to the FCFS reference
        sequence, by more than ``k_cps`` places — this is what "balances
        runway load without relaxing fairness constraints" means. Evaluates
        every (aircraft × runway) pair in a single vectorised model call,
        then restricts the argmin to the fairness-eligible runways.

        ``sigma_i^FCFS`` is the aircraft's rank (ascending predicted ETA)
        across the whole fleet, using each aircraft's ETA on its *own
        current* runway — the same FCFS reference used everywhere else
        (Eq. absolute_eta), computed here from this call's freshly predicted
        ``eta_matrix`` rather than the (possibly stale/naive-placeholder)
        ``ac.eta`` attribute, since dynamic mode is this method's sole
        source of ETA refresh (see :meth:`update_fleet`'s docstring).

        ``sigma_i^(r)`` is the rank aircraft i would hold, by predicted ETA
        on candidate runway r, among the *other* aircraft currently assigned
        to r (their existing, pre-reassignment ``runway_id``). This is a
        per-aircraft approximation rather than a jointly optimal
        simultaneous reassignment — consistent with
        :meth:`_apply_k_cps_constraint`'s own greedy, non-exhaustive
        resolution of the same kind of parallel combinatorial constraint.

        If no candidate runway satisfies the fairness bound (e.g. ``k_cps=0``
        with a busy runway), the aircraft's current runway is kept rather
        than leaving the assignment undefined — mirroring
        :meth:`_apply_k_cps_constraint`'s own fallback.

        Parameters
        ----------
        surrogate : ETASurrogate, optional
        current_time : float
        lag_features : np.ndarray, shape (n, 3), optional
        """
        if surrogate is None or not self.available_runways or not self._fleet:
            return

        states = np.stack([ac.state for ac in self._fleet])  # (n, 5)
        target_time_budget = states[:, 4] if states.shape[1] > 4 else None
        eta_matrix = surrogate.predict_eta_fleet_all_runways(
            states, self.available_runways, current_time, lag_features,
            target_time_budget=target_time_budget,
        )  # (n, r)

        n = len(self._fleet)
        k = self.k_cps
        rwy_index = {r: j for j, r in enumerate(self.available_runways)}
        current_col = np.array(
            [rwy_index.get(ac.runway_id, 0) for ac in self._fleet]
        )
        current_eta = eta_matrix[np.arange(n), current_col]
        # sigma_i^FCFS: 0-indexed rank by ascending ETA-on-current-runway.
        fcfs_rank = np.argsort(np.argsort(current_eta))
        for i, ac in enumerate(self._fleet):
            ac.fcfs_rank = int(fcfs_rank[i])

        best_rwy_idx = np.empty(n, dtype=int)
        for i, ac in enumerate(self._fleet):
            eligible: List[int] = []
            for j, rwy in enumerate(self.available_runways):
                other_etas = [
                    eta_matrix[i2, j]
                    for i2, ac2 in enumerate(self._fleet)
                    if i2 != i and ac2.runway_id == rwy
                ]
                sigma_r = sum(1 for e in other_etas if e < eta_matrix[i, j])
                if abs(sigma_r - int(fcfs_rank[i])) <= k:
                    eligible.append(j)
            if not eligible:
                eligible = [
                    rwy_index.get(ac.runway_id, int(np.argmin(eta_matrix[i])))
                ]
            best_rwy_idx[i] = min(eligible, key=lambda j: eta_matrix[i, j])

        for i, ac in enumerate(self._fleet):
            if self.enable_stall_detection and ac.acid in self._stalled_acids:
                continue  # frozen: don't chase a moving target (see is_stalled)
            j = int(best_rwy_idx[i])
            ac.runway_id = self.available_runways[j]
            ac.eta = float(eta_matrix[i, j])

    # ------------------------------------------------------------------ #
    # Stall detection
    # ------------------------------------------------------------------ #

    def _update_stall_tracking(
        self,
        surrogate: Optional["ETASurrogate"],
        current_time: float,
    ) -> None:
        """Flag aircraft that haven't beaten their best-ever distance to the
        IAF in the last ``STALL_WINDOW_S`` seconds.

        Uses each aircraft's own current ``(x, y)`` state and its assigned
        runway's IAF reference point -- a physical quantity, independent of
        the surrogate's ETA prediction (which is itself the thing being
        distorted by the feedback loop this is meant to break, so it can't
        be used as the convergence signal).

        Deliberately a "no new best in N seconds" test rather than a
        windowed delta: a genuinely holding/circling aircraft's distance
        oscillates by tens of km within any short window (confirmed via
        direct position tracing on the seed=1000 diagnostic scenario) even
        though it never gets meaningfully closer over the long run, so a
        simple start-vs-end-of-window comparison intermittently reads as
        "progress" purely from that noise and never fires. Tracking the
        best distance ever achieved and requiring a fresh improvement within
        the window is immune to that -- oscillation can produce a new best by
        luck, but can't sustain "no new best" for the full window the way
        genuine non-convergence does.

        Once flagged, an acid stays in ``_stalled_acids`` for the rest of
        the episode (see :meth:`is_stalled`).
        """
        if surrogate is None or not surrogate._iaf_ref or not self._fleet:
            return
        for ac in self._fleet:
            if ac.acid in self._stalled_acids:
                continue
            iaf = surrogate._iaf_ref.get(ac.runway_id)
            if iaf is None:
                continue
            iaf_x, iaf_y, _ = iaf
            dist_km = math.hypot(iaf_x - ac.state[0], iaf_y - ac.state[1]) * MAX_DISTANCE

            best = self._best_distance_km.get(ac.acid)
            if best is None or dist_km < best - self.STALL_PROGRESS_EPS_KM:
                self._best_distance_km[ac.acid] = dist_km
                self._best_distance_time[ac.acid] = current_time
                continue

            stagnant_for = current_time - self._best_distance_time.get(ac.acid, current_time)
            if stagnant_for >= self.STALL_WINDOW_S:
                self._stalled_acids.add(ac.acid)
                # Snapshot the current eta/runway/tta as the frozen target --
                # re-applied every subsequent cycle in update_fleet. tta
                # falls back to eta if not yet finite (no committed TTA yet).
                self._frozen_eta[ac.acid] = ac.eta
                self._frozen_runway[ac.acid] = ac.runway_id
                self._frozen_tta[ac.acid] = ac.tta if math.isfinite(ac.tta) else ac.eta

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
        states = np.stack([ac.state for ac in self._fleet])       # (n, 5)
        target_time_budget = states[:, 4] if states.shape[1] > 4 else None
        runway_ids = [ac.runway_id for ac in self._fleet]
        etas = surrogate.predict_eta_fleet(
            states, runway_ids, current_time, lag_features,
            target_time_budget=target_time_budget,
        )  # (n,)
        for i, ac in enumerate(self._fleet):
            if self.enable_stall_detection and ac.acid in self._stalled_acids:
                continue  # frozen: don't chase a moving target (see is_stalled)
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
