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

from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME, MAX_DISTANCE

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
        Aircraft callsign / unique identifier. NOTE: under a rolling-
        arrival-stream env config, this string is reused across different
        physical aircraft that occupy the same slot at different times in
        an episode -- see ``spawn_time`` below.
    spawn_time : float
        Absolute (episode-clock) instant this physical aircraft spawned
        into its slot. Unlike ``acid`` (slot-derived, reused per-slot),
        this is unique per physical occupancy, so :meth:`CPSManager.
        update_fleet` diffs on ``(acid, spawn_time)`` rather than ``acid``
        alone to detect a same-step slot refill (a fresh aircraft spawning
        into a slot the same env.step() call that its predecessor
        departed it) -- a plain acid set-diff would silently miss that
        case and leak the departed aircraft's bookkeeping into the new
        occupant.
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
    wake_cat: str = "D"
    spawn_time: float = 0.0


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
    enable_cross_cycle_runway_seeding : bool
        If True (default), :meth:`_tta_for` seeds a runway's "previous
        scheduled aircraft" from ``self._runway_last_committed`` (persisted
        across replanning cycles) whenever the current cycle's own sequence
        doesn't contain an earlier aircraft on that runway. Set False
        (Test C, pre-Step-10 audit §1.2) to isolate the surrogate-feature
        feedback loop (§1.1) from this second, structurally distinct
        cross-cycle ratcheting channel.
    reassignment_hysteresis_s : float
        Margin (seconds) a candidate runway must beat the current runway's
        predicted ETA by before :meth:`_assign_runways_dynamic` switches to
        it -- see the ``REASSIGNMENT_HYSTERESIS_S`` class-constant docstring
        below for the full rationale. Defaults to that class constant
        (today's exact behaviour, unchanged for every existing caller).
        Exposed as a constructor param (rather than only the class constant)
        so ``scripts/run_batch_eval.py``'s ``--reassignment-hysteresis-s``
        flag can sweep it per the concurrency-cap/reassignment-guard-timing
        resweep (``.claude/plans/concurrency_cap_and_reassignment_guard_resweep.md``)
        without mutating shared class state. Must be a non-negative multiple
        of ``ACTION_TIME / 2`` -- enforced at construction time, since the
        value's entire physical meaning ("N half-control-cycles' worth of
        predicted-ETA gain") depends on it, and the resweep grid's 0.5x
        candidate needs half-cycle granularity.
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

    # Probation (non-sticky re-check) for a newly-flagged aircraft: rather
    # than finalizing a stall the instant STALL_WINDOW_S elapses, give the
    # aircraft one more RECOVERY_WINDOW_S to prove the flag was premature --
    # if its distance beats the at-flag value by RECOVERY_PROGRESS_EPS_KM
    # within that window, un-flag and resume normal tracking instead of
    # finalizing. Added after the capacity-sweep investigation
    # (.claude/plans/stall_rate_investigation.md, 2026-08-12) found the
    # original sticky rule flags aircraft ~140-150km from their IAF that go
    # on to make real further progress ~99-100% of the time -- STALL_WINDOW_S
    # was calibrated against episode length, not per-flight length (a typical
    # flight is only 23-31 steps / 46-62 minutes, so a fixed 30-minute
    # no-improvement window is on the order of an entire flight, not a small
    # sub-window of a long one), and every simple recalibration of the window
    # alone (schedule-relative, fixed-larger, own-flight-relative,
    # velocity-based) topped out around 0.42 precision without giving up most
    # of the recall a real detector needs. An offline multi-candidate replay
    # against real trajectories showed this probation shape hits >=0.85
    # precision at RECOVERY_WINDOW_S=ACTION_TIME (one control cycle) and
    # RECOVERY_PROGRESS_EPS_KM=STALL_PROGRESS_EPS_KM, cutting false-positive
    # freezes ~10x versus the sticky rule at comparable recall -- see that
    # plan doc's Phase 4 comparison table for the full parameter sweep this
    # was chosen from. False positives matter beyond mislabeling: a flagged
    # aircraft is excluded from _replan's runway timeline for the rest of its
    # flight (see _replan's `active` filter), which also removes it from
    # separation-compliance accounting for other aircraft sharing its runway.
    RECOVERY_WINDOW_S = ACTION_TIME
    RECOVERY_PROGRESS_EPS_KM = STALL_PROGRESS_EPS_KM
    # Once an aircraft has needed probation once (shown genuinely ambiguous
    # behaviour), it's on a shorter leash for the rest of the flight: a
    # second no-improvement stretch only needs to reach REFLAG_WINDOW_S
    # (not a fresh STALL_WINDOW_S) before probation re-triggers. Chosen
    # empirically alongside the two constants above (the same sweep's
    # dominant variant): 300s beat both a full-length and a same-length
    # (900s) re-arm window on every metric (precision, recall, and FP count)
    # simultaneously.
    REFLAG_WINDOW_S = 300.0

    # _assign_runways_dynamic's argmin-predicted-ETA rule had no guard
    # against reassigning an aircraft that is already close to IAF-crossing
    # on its current runway -- the frozen worker policy is only trained up
    # to the IAF (ETASurrogate.predict_eta predicts time-to-IAF-crossing,
    # not time-to-touchdown), so a reassignment issued this close to the
    # IAF leaves it no remaining control cycles in which to actually
    # redirect toward a different IAF/runway. Confirmed empirically (see
    # .claude/plans/max_concurrent_aircraft_capacity_sweep.md's capacity
    # sweep investigation, 2026-08-12): this produced two distinct
    # "wrong_runway" death populations -- one where the aircraft, unable to
    # converge on its new target, eventually triggered stall detection
    # (~16,000s dwell, frozen after STALL_WINDOW_S of no progress), and a
    # second, smaller population with *normal* (non-stalled) flight
    # durations that simply crossed their original runway's sink because
    # the reassignment came too late to act on.
    #
    # An aircraft within FINAL_APPROACH_LOCK_S of its current-runway ETA is
    # now excluded from reassignment entirely (forced to keep its current
    # runway). This is a control-cadence margin, NOT a schedule-slack
    # concept -- deliberately NOT set to the frozen worker's ~20-minute
    # (1200s) delay-absorption capacity, which would lock out 25-40% of a
    # typical ~3,000-4,700s flight (median
    # true per-aircraft dwell measured in the same investigation) and
    # neuter dynamic reassignment almost entirely. Set to 2 x ACTION_TIME
    # instead: the minimum margin guaranteeing at least one full env.step
    # cycle for a newly-assigned target to actually influence the frozen
    # worker's action before IAF-crossing would otherwise occur, plus one
    # cycle of buffer for the fleet snapshot/replan cadence (delta_t_plan
    # is itself typically configured equal to ACTION_TIME, e.g. both 120s
    # in cps_scale_10k.yaml).
    FINAL_APPROACH_LOCK_S = 2.0 * ACTION_TIME

    # _assign_runways_dynamic's argmin-predicted-ETA rule also had no
    # hysteresis: on every replanning cycle it re-picks the pure argmin
    # over eligible runways, so two runways with near-tied predicted ETA
    # cause the choice to flip back and forth purely on ETA-prediction
    # noise, cycle to cycle. Confirmed empirically (see
    # .claude/plans/max_concurrent_aircraft_capacity_sweep.md, 2026-08-12):
    # ~8,950 actual runway switches over 30 episodes x 50 concurrent
    # aircraft -- ~6 per aircraft, across only 2 runways -- with a median
    # predicted-ETA "gain" of just ~112s, spread throughout the flight
    # rather than concentrated near landing (a distinct, much larger effect
    # than the rare late-reassignment case FINAL_APPROACH_LOCK_S targets).
    # A candidate must now beat the current runway's predicted ETA by more
    # than REASSIGNMENT_HYSTERESIS_S to be worth switching to. Expressed as
    # an integer multiple of ACTION_TIME (the fixed physical control-step
    # size), not delta_t_plan (a CPS-specific replanning-cadence config
    # value that happens to equal ACTION_TIME in cps_scale_10k.yaml but
    # isn't guaranteed to), so the margin keeps one fixed physical meaning
    # regardless of how delta_t_plan is configured: a candidate must save
    # at least this many full control cycles' worth of time to be worth the
    # disruption of retargeting the frozen worker mid-flight. Aircraft
    # whose current runway is NOT itself eligible under the k-CPS fairness
    # window are exempt from this margin -- that is a genuine
    # fairness-forced move, not a discretionary optimization. Set to
    # 2 x ACTION_TIME (matching FINAL_APPROACH_LOCK_S's own multiple, so
    # both margins share one physical reading -- "two control cycles is the
    # threshold below which a decision isn't worth disrupting the frozen
    # worker's committed path for"). See the capacity-sweep plan doc for
    # the 1x-vs-2x empirical comparison this was chosen from.
    #
    # This class constant is now only the *default* -- see the
    # `reassignment_hysteresis_s` constructor param below, added for the
    # first documented sensitivity sweep of this constant
    # (.claude/plans/concurrency_cap_and_reassignment_guard_resweep.md,
    # 2026-08-20), which sweeps {0.5,1,2,3} x ACTION_TIME. Any caller not
    # passing that param gets this exact value, unchanged.
    REASSIGNMENT_HYSTERESIS_S = 2.0 * ACTION_TIME

    # _assign_runways_dynamic's cost function (see docstring) had no
    # load-balancing term: the argmin over eligible runways optimized raw
    # single-aircraft predicted ETA in isolation, with sigma_matrix/
    # eligible acting only as a fairness-position bound, never an occupancy
    # penalty. Confirmed at M=2,000 production scale (Vector 9,
    # .claude/plans/phase3_cps_coordination_plan.md) to concentrate 90-94%
    # of dynamic-mode stalling on one runway (18R) at k1/k3, vs its
    # ~44-48% baseline traffic share -- static mode splits 50/50 at every
    # k, so this is algorithmic, not geometric. A mean-centered,
    # self-excluding occupancy-count penalty was implemented and swept at
    # diagnostic scale (.claude/plans/cps_runway_load_balancing_fix.md) and
    # found to make the choice-split *worse*, not better, at every tested
    # weight in the production-shaped harness -- removed entirely rather
    # than kept "just in case" (same precedent as fairness_weight's
    # removal). A queueing-delay-based replacement term
    # (QUEUE_DELAY_WEIGHT_S/queue_delay_weight_s/queue_delay_penalty) was
    # also implemented, swept, and removed 2026-08-15 for the same reason:
    # a real split improvement, but a mechanistically-confirmed oscillation
    # cost (magnitude mismatch with REASSIGNMENT_HYSTERESIS_S) that a
    # bounded variant couldn't separate from the win either. See
    # .claude/plans/cps_runway_queue_delay_fix.md for the full investigation,
    # including a more promising but still-unimplemented candidate (a fixed
    # raw-ETA-advantage offset, paused pending root-cause investigation).

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
        enable_cross_cycle_runway_seeding: bool = True,
        log_reassignment_events: bool = False,
        reassignment_hysteresis_s: float = REASSIGNMENT_HYSTERESIS_S,
    ) -> None:
        if runway_assignment_mode not in self._valid_modes:
            raise ValueError(
                f"runway_assignment_mode must be one of {self._valid_modes!r}, "
                f"got {runway_assignment_mode!r}"
            )
        _half_action_time = ACTION_TIME / 2.0
        if reassignment_hysteresis_s < 0 or (
            abs(reassignment_hysteresis_s % _half_action_time) > 1e-6
            and abs(reassignment_hysteresis_s % _half_action_time - _half_action_time) > 1e-6
        ):
            raise ValueError(
                f"reassignment_hysteresis_s must be a non-negative multiple of "
                f"ACTION_TIME/2 ({_half_action_time}s) -- its physical meaning is 'N "
                f"half-control-cycles' worth of predicted-ETA gain', and the resweep grid's "
                f"0.5x candidate (see concurrency_cap_and_reassignment_guard_resweep.md) "
                f"needs half-cycle granularity, not just whole multiples of ACTION_TIME "
                f"itself; got {reassignment_hysteresis_s!r}"
            )
        self.reassignment_hysteresis_s = reassignment_hysteresis_s
        self.k_cps = k_cps
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
        # Probation state (see RECOVERY_WINDOW_S docstring): acid -> (time,
        # distance_km) at the moment it crossed the no-improvement window,
        # for acids currently awaiting a recovery-or-finalize decision.
        # acid -> the effective no-improvement window to use going forward
        # (STALL_WINDOW_S until a first probation/recovery, REFLAG_WINDOW_S
        # after) -- absent means "still on the original STALL_WINDOW_S".
        self._probation_since: Dict[str, Tuple[float, float]] = {}
        self._stall_window_override: Dict[str, float] = {}
        # Snapshot of eta/runway_id taken the cycle an acid is first flagged
        # stalled, re-applied every subsequent cycle (see update_fleet) so
        # the frozen target actually stays fixed rather than drifting with
        # the caller's freshly-reseeded per-cycle naive ETA.
        self._frozen_eta: Dict[str, float] = {}
        self._frozen_runway: Dict[str, str] = {}
        self._frozen_tta: Dict[str, float] = {}
        # Diagnostic-only (Vector 9, .claude/plans/phase3_cps_coordination_plan.md):
        # per-decision-cycle record of _assign_runways_dynamic's eligibility/
        # choice for every aircraft, appended only when log_reassignment_events
        # is True (default off -- zero overhead/behavior change for every
        # existing caller). Drained via drain_reassignment_log().
        self.log_reassignment_events = log_reassignment_events
        self._reassignment_events: List[dict] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def drain_reassignment_log(self) -> List[dict]:
        """Return and clear all reassignment-event records logged so far.

        Only populated when ``log_reassignment_events=True`` was passed to
        the constructor. Callers should drain once per episode (this manager
        has no episode concept of its own) and tag the returned rows with
        their own episode_id/k_cps/mode before writing to telemetry.
        """
        events, self._reassignment_events = self._reassignment_events, []
        return events

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
        # Diff on (acid, spawn_time), not acid alone: under a rolling-
        # arrival-stream config, MultiAgentPathPlanningGoalEnv refills a
        # freed slot within the same env.step() call that terminated its
        # previous occupant, so the acid string is never actually absent
        # between two consecutive update_fleet() calls -- a plain acid
        # set-diff would silently miss the swap and leak the departed
        # aircraft's stall-tracking state into the new occupant (see
        # AircraftState.spawn_time's docstring). spawn_time strictly
        # increases across a same-step swap, so it disambiguates them.
        old_spawn_time = {ac.acid: ac.spawn_time for ac in self._fleet}
        self._fleet = aircraft
        self._fleet_index = {ac.acid: i for i, ac in enumerate(aircraft)}
        new_spawn_time = {ac.acid: ac.spawn_time for ac in aircraft}
        stale_acids = {
            acid for acid, spawn_time in old_spawn_time.items()
            if new_spawn_time.get(acid) != spawn_time
        }

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
            for acid in stale_acids:
                self._trajectory_buffer.evict(acid)
            for ac in self._fleet:
                # state[3] is heading_deg_bearing → convert to radians for buffer
                self._trajectory_buffer.push(
                    ac.acid,
                    float(ac.state[0]),
                    float(ac.state[1]),
                    float(np.deg2rad(ac.state[3])),
                )

        for acid in stale_acids:
            self._best_distance_km.pop(acid, None)
            self._best_distance_time.pop(acid, None)
            self._stalled_acids.discard(acid)
            self._probation_since.pop(acid, None)
            self._stall_window_override.pop(acid, None)
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
        self._probation_since = {}
        self._stall_window_override = {}
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
        """Full replanning pass: FCFS → k-CPS reorder → greedy TTAs.

        Both runway-assignment modes now run a k-bounded reorder
        (:meth:`_apply_k_cps_constraint`) on the FCFS sequence before greedy
        scheduling: no aircraft may shift more than ``k_cps`` positions from
        its FCFS rank, and within that window the earliest-ETA candidate is
        always chosen. This is complementary to, not a replacement for,
        dynamic mode's own separate ``k_cps``-bounded runway-reassignment
        eligibility window in :meth:`_assign_runways_dynamic`: that
        mechanism decides *which runway* an aircraft lands on; this one
        decides *what order* it lands in among aircraft sharing whichever
        runway it ends up on.

        An earlier fairness-weighted version of this same reordering step
        (biasing low-slack aircraft toward earlier positions) was removed
        after ``.claude/plans/stall_rate_investigation.md`` (2026-08-12)
        found it never won against plain FCFS at any tested
        ``fairness_weight > 0``, in either runway-assignment mode, at any
        congestion level. The fairness-free version reintroduced here
        (``.claude/plans/cps_static_mode_k_cps_design.md``) is an *exact*
        identity permutation on FCFS-sorted input, for any ``k_cps`` and any
        separation matrix (provable by induction, not merely
        wake-homogeneity-dependent): earliest-ETA-in-window always resolves
        to the aircraft already at the current position when the input is
        pre-sorted ascending by ETA. See the plan doc for the full proof and
        for why an earlier draft's delay-cost-minimizing selection rule was
        rejected -- it is not equivalent to this one and is not a no-op.

        Stalled aircraft (see :meth:`_update_stall_tracking`) are excluded
        from greedy scheduling entirely, not just from having their own TTA
        touched: a frozen-but-never-landing
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
        active = self._apply_k_cps_constraint(active)
        self._greedy_schedule(active)

    def _fcfs_order(self) -> List[AircraftState]:
        """Sort fleet by ascending absolute ETA → FCFS reference sequence."""
        return sorted(self._fleet, key=lambda a: (a.eta, a.acid))

    def _apply_k_cps_constraint(
        self, fcfs_order: List[AircraftState],
    ) -> List[AircraftState]:
        """k-bounded reorder: no aircraft may shift more than k positions
        from its FCFS rank (both runway-assignment modes).

        Greedy forward sweep over positions 0..n-1. At each position, among
        unscheduled candidates in [pos-k, pos+k], selects the one with the
        earliest ETA (ties broken by acid, matching :meth:`_fcfs_order`'s
        own tie-break) -- i.e. the literal k-CPS definition from this
        module's own docstring, not a delay-cost simulation.

        An earlier draft of this method (see
        .claude/plans/cps_static_mode_k_cps_design.md §2's original text,
        superseded by this implementation) instead picked the candidate
        minimizing its own imposed runway-contention delay. That rule is
        NOT equivalent to earliest-ETA-in-window and is not a no-op: with a
        wide ETA gap in the window (e.g. etas [100, 105, 110, 500, 510] on
        one runway, k=2), it lets the far-future aircraft (eta=500) jump
        into an early slot ahead of the near-term ones purely because doing
        so costs *that aircraft* zero delay -- pushing the near-term
        aircraft later and increasing total delay relative to FCFS. That
        contradicts FCFS's own optimality (classical EDD-optimality: with
        equal separations, ascending-ETA order minimizes total/maximum
        delay on a single server) -- the bug was in the greedy heuristic's
        search, not in that optimality claim.

        Earliest-ETA-in-window has no such failure mode: by induction, on
        FCFS-sorted input the earliest-ETA candidate among any unscheduled
        window is always exactly the aircraft at the current position, for
        any k and any separation matrix (wake-homogeneity not even
        required) -- so this is an *exact* identity permutation on
        FCFS-sorted input, not merely an empirically-expected one.

        Performance: the sole call site (:meth:`_replan`) always passes
        ``fcfs_order`` already ascending-(eta, acid)-sorted, so the
        induction proof above applies unconditionally there -- an O(n)
        sortedness check below short-circuits the O(n·(2k+1)) sweep in that
        case, falling back to the real sweep only if that assumption is
        ever violated (e.g. a future caller passing unsorted input). This
        was benchmarked as a ~6x speedup at production scale (n=50, k=3:
        18.8us -> 3.1us/call) with 800-trial parity verification against
        the full sweep. A numpy-vectorized rewrite of the sweep itself was
        also benchmarked and rejected -- 6-10x *slower* than the plain
        Python loop at this window width (~7 elements at k_cps=3), matching
        this file's other vectorization finding for the analogous
        now-removed fairness-weighted sweep: numpy's fixed per-call
        dispatch overhead dominates at this scale. See
        cps_coordination/testing/test_vectorization_performance.py.
        """
        if self.k_cps == 0:
            return list(fcfs_order)

        n = len(fcfs_order)
        if all(
            (fcfs_order[i].eta, fcfs_order[i].acid)
            <= (fcfs_order[i + 1].eta, fcfs_order[i + 1].acid)
            for i in range(n - 1)
        ):
            return list(fcfs_order)

        k = self.k_cps
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

    def _tta_for(self, ac: AircraftState, runway_last: Dict[str, AircraftState]) -> float:
        """Compute what ``ac.tta`` would be if scheduled next on its own
        runway, given a per-runway "last scheduled aircraft" tracker.

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
        per-aircraft rule and the cross-cycle seeding/self-match notes. Once
        every aircraft in *sequence* has been assigned a tta,
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
        simultaneous reassignment.

        If no candidate runway satisfies the fairness bound (e.g. ``k_cps=0``
        with a busy runway), the aircraft's current runway is kept rather
        than leaving the assignment undefined.

        Parameters
        ----------
        surrogate : ETASurrogate, optional
        current_time : float
        lag_features : np.ndarray, shape (n, 3), optional
        """
        if surrogate is None or not self.available_runways or not self._fleet:
            return

        # Stalled aircraft are frozen (see is_stalled/"don't chase a moving
        # target" below) and must never have their runway/eta overwritten --
        # but leaving them in this method's own eta_matrix/sigma_matrix
        # computation is also wrong even though their *own* assignment ends
        # up protected: a stalled aircraft that keeps being re-evaluated
        # every tick (a) pollutes reassignment-event telemetry for as long
        # as it remains airborne (which can be hours past its own
        # STALL_WINDOW_S, since nothing here stops it from continuing to
        # appear), and (b) keeps counting as a `member` occupying its runway
        # in sigma_matrix's congestion-rank computation, distorting the
        # fairness eligibility every *other*, still-progressing aircraft is
        # evaluated against. Excluding stalled acids from this method's
        # fleet view entirely (not just the final write-back) fixes both.
        if self.enable_stall_detection and self._stalled_acids:
            active_indices = [
                i for i, ac in enumerate(self._fleet)
                if ac.acid not in self._stalled_acids
            ]
            active_fleet = [self._fleet[i] for i in active_indices]
            # lag_features rows are in self._fleet's order (computed by the
            # caller, update_fleet, before this method runs) -- must be
            # subset with the same indices to stay aligned with active_fleet.
            if lag_features is not None:
                lag_features = lag_features[active_indices]
        else:
            active_fleet = self._fleet
        if not active_fleet:
            return

        states = np.stack([ac.state for ac in active_fleet])  # (n, 5)
        target_time_budget = states[:, 4] if states.shape[1] > 4 else None
        eta_matrix = surrogate.predict_eta_fleet_all_runways(
            states, self.available_runways, current_time, lag_features,
            target_time_budget=target_time_budget,
        )  # (n, r)

        n = len(active_fleet)
        k = self.k_cps
        rwy_index = {r: j for j, r in enumerate(self.available_runways)}
        current_col = np.array(
            [rwy_index.get(ac.runway_id, 0) for ac in active_fleet]
        )
        current_eta = eta_matrix[np.arange(n), current_col]
        # sigma_i^FCFS: 0-indexed rank by ascending ETA-on-current-runway.
        fcfs_rank = np.argsort(np.argsort(current_eta))
        for i, ac in enumerate(active_fleet):
            ac.fcfs_rank = int(fcfs_rank[i])

        # Vectorized eligibility/rank computation -- a literal, order-
        # independent restatement of the nested-loop form above (see
        # `.claude/plans/phase3_cps_coordination_plan.md`'s "Vectorization &
        # Optimization" section for the derivation this mirrors exactly).
        n_rwy = len(self.available_runways)
        # member[i2, j]: is aircraft i2 currently assigned to candidate
        # runway j? (current_col already encodes ac2.runway_id via rwy_index,
        # consistent with current_eta/fcfs_rank above.)
        member = current_col[:, None] == np.arange(n_rwy)[None, :]  # (n, r)
        # less[i2, i, j]: aircraft i2's ETA on runway j strictly less than
        # aircraft i's ETA on runway j. less[i, i, j] is always False (an
        # element is never < itself), so this subsumes the original loop's
        # `i2 != i` guard without needing it explicitly.
        less = eta_matrix[:, None, :] < eta_matrix[None, :, :]  # (n, n, r)
        # sigma_matrix[i, j]: rank aircraft i would hold on runway j among
        # the *other* aircraft currently assigned to j.
        sigma_matrix = (less & member[:, None, :]).sum(axis=0)  # (n, r)

        eligible = np.abs(sigma_matrix - fcfs_rank[:, None]) <= k  # (n, r)

        # Final-approach lock: an aircraft within FINAL_APPROACH_LOCK_S of
        # landing on its current runway is excluded from reassignment --
        # forced back to its current runway regardless of what the fairness
        # window would otherwise allow (see FINAL_APPROACH_LOCK_S docstring
        # above).
        locked = (current_eta - current_time) <= self.FINAL_APPROACH_LOCK_S  # (n,)
        if locked.any():
            eligible[locked] = False
            eligible[locked, current_col[locked]] = True

        no_eligible = ~eligible.any(axis=1)
        if no_eligible.any():
            runway_found = np.array(
                [ac.runway_id in rwy_index for ac in active_fleet]
            )
            argmin_eta = np.argmin(eta_matrix, axis=1)
            fallback_col = np.where(runway_found, current_col, argmin_eta)
            eligible[no_eligible, fallback_col[no_eligible]] = True

        # np.argmin, like the original min(eligible, key=...), returns the
        # first (lowest-j) occurrence of the minimum on ties.
        masked_eta = np.where(eligible, eta_matrix, np.inf)
        best_rwy_idx = np.argmin(masked_eta, axis=1)

        # Reassignment hysteresis -- see REASSIGNMENT_HYSTERESIS_S's
        # docstring above for the full rationale and supporting evidence.
        current_is_eligible = eligible[np.arange(n), current_col]
        stay_is_close_enough = (
            eta_matrix[np.arange(n), current_col] - masked_eta[np.arange(n), best_rwy_idx]
        ) < self.reassignment_hysteresis_s
        keep_current = current_is_eligible & stay_is_close_enough
        best_rwy_idx = np.where(keep_current, current_col, best_rwy_idx)

        if self.log_reassignment_events:
            for i, ac in enumerate(active_fleet):
                c = int(current_col[i])
                j = int(best_rwy_idx[i])
                self._reassignment_events.append({
                    "current_time": float(current_time),
                    "acid": ac.acid,
                    "current_runway": self.available_runways[c],
                    "fcfs_rank": int(fcfs_rank[i]),
                    "sigma_current": int(sigma_matrix[i, c]),
                    "eligible_runways": ",".join(
                        self.available_runways[jj] for jj in range(n_rwy) if eligible[i, jj]
                    ),
                    "chosen_runway": self.available_runways[j],
                    "switched": bool(j != c),
                    "eta_gap_s": float(current_eta[i] - eta_matrix[i, j]),
                    # Always False now: stalled acids are excluded from
                    # active_fleet above, so they never reach this loop.
                    # Column kept for telemetry schema stability.
                    "stalled_excluded": False,
                    "sigma_per_runway": ",".join(
                        f"{self.available_runways[jj]}:{int(sigma_matrix[i, jj])}"
                        for jj in range(n_rwy)
                    ),
                    "eta_per_runway": ",".join(
                        f"{self.available_runways[jj]}:{eta_matrix[i, jj]:.1f}"
                        for jj in range(n_rwy)
                    ),
                    "x": float(states[i, 0]),
                    "y": float(states[i, 1]),
                })

        # Stalled acids are already excluded from active_fleet above (see
        # comment near its construction), so every remaining aircraft here
        # is safe to write back -- no "don't chase a moving target" check
        # needed at this point, unlike before that exclusion existed.
        for i, ac in enumerate(active_fleet):
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
        IAF in the last ``STALL_WINDOW_S`` (or ``REFLAG_WINDOW_S``, see
        below) seconds -- subject to a probation re-check before finalizing.

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

        Probation: crossing the window no longer finalizes a stall
        immediately. The acid instead enters ``_probation_since`` for up to
        ``RECOVERY_WINDOW_S`` -- if its distance beats the at-flag value by
        ``RECOVERY_PROGRESS_EPS_KM`` within that window, it's released back
        to normal tracking (with its required window shortened to
        ``REFLAG_WINDOW_S`` from then on, since it's already shown once that
        it needed the benefit of the doubt); otherwise it's finalized exactly
        as before. See ``RECOVERY_WINDOW_S``'s class docstring for why (found
        empirically: the sticky version flags aircraft that go on to make
        real further progress ~99-100% of the time).

        Once *finalized*, an acid stays in ``_stalled_acids`` for the rest of
        the episode (see :meth:`is_stalled`) -- probation only defers that
        decision by up to ``RECOVERY_WINDOW_S``, it doesn't remove it.
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

            probation = self._probation_since.get(ac.acid)
            if probation is not None:
                flagged_at_time, flagged_at_dist = probation
                if dist_km < flagged_at_dist - self.RECOVERY_PROGRESS_EPS_KM:
                    # Recovered: release from probation, resume normal
                    # tracking with a shortened window from here on.
                    del self._probation_since[ac.acid]
                    self._stall_window_override[ac.acid] = self.REFLAG_WINDOW_S
                    self._best_distance_km[ac.acid] = dist_km
                    self._best_distance_time[ac.acid] = current_time
                    continue
                if current_time - flagged_at_time >= self.RECOVERY_WINDOW_S:
                    self._finalize_stall(ac)
                continue

            best = self._best_distance_km.get(ac.acid)
            if best is None or dist_km < best - self.STALL_PROGRESS_EPS_KM:
                self._best_distance_km[ac.acid] = dist_km
                self._best_distance_time[ac.acid] = current_time
                continue

            window = self._stall_window_override.get(ac.acid, self.STALL_WINDOW_S)
            stagnant_for = current_time - self._best_distance_time.get(ac.acid, current_time)
            if stagnant_for >= window:
                self._probation_since[ac.acid] = (current_time, dist_km)

    def _finalize_stall(self, ac: AircraftState) -> None:
        """Commit a stall past probation: flag ``ac`` and snapshot its
        eta/runway/tta as the frozen target, re-applied every subsequent
        cycle in :meth:`update_fleet`. tta falls back to eta if not yet
        finite (no committed TTA yet).
        """
        self._stalled_acids.add(ac.acid)
        self._probation_since.pop(ac.acid, None)
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
