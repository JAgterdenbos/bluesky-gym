"""
Multi-aircraft generalisation of ``PathPlanningGoalEnv`` for Phase III CPS
coordination evaluation.

Why this file exists
---------------------
BlueSky (``import bluesky as bs``) makes ``bs.traf``/``bs.scr`` process-wide
singletons (``bluesky/core/entity.py::EntityMeta`` — ``__init__`` only runs
once per process; every later construction returns the same instance). The
single-agent ``PathPlanningGoalEnv`` always creates a single hardcoded
aircraft ``"kl001"`` at traffic-array index ``0`` and reads/writes
``bs.traf.<field>[0]`` everywhere. Spinning up N independent
``PathPlanningGoalEnv`` instances to simulate "N aircraft" (as the earlier
CPS evaluation code did) makes every instance silently fight over that same
shared slot. This module instead manages many aircraft, each with a unique
callsign, inside **one** shared BlueSky simulation.

API shape
---------
This is intentionally *not* a standard single-agent ``gym.Env``. A CPS
coordinator needs to see every active aircraft at once each decision step
and call the frozen worker policy in a single batched ``model.predict()``
call, so the runtime contract is::

    obs, info_list = env.reset(seed=...)
    ...
    obs, rewards, terminated, truncated, info_list = env.step(actions)

where every value in ``obs`` (``"observation"``, ``"achieved_goal"``,
``"desired_goal"``) has a leading batch axis of size ``len(info_list)``, and
``actions`` must be a ``(len(info_list), 2)`` array whose rows correspond
*positionally* to the array most recently returned by ``reset()``/``step()``
(each row also carries an ``"acid"``/``"slot"`` key so a caller can build
``AircraftState`` records without needing a separate index/slot argument).

**Important — ``step()``'s return is a transition record, not a ready-made
next observation.** ``obs``/``rewards``/``terminated``/``truncated``/
``info_list`` all describe the slots that were active *going into* that
``step()`` call (useful for logging/replay: this is what happened to each
aircraft that acted this step). Internally, any slot that just terminated is
immediately despawned and — if more arrivals remain — replaced with a freshly
spawned aircraft in the same slot, updating ``env.active_slots`` for the
*next* call. That means the array you're holding right after ``step()`` can
already be stale (wrong aircraft, or even wrong batch size never applies here
since replacement is 1-for-1, but the *content* is wrong) for driving the next
decision. Always call :meth:`get_active_batch` to fetch the current
``(obs, info_list)`` for ``env.active_slots`` before the next
``model.predict()`` — the CPS integration loop needs to call it anyway, since
it also refreshes ``desired_goal``/``current_runway`` after any
``set_tta``/``set_runway`` calls made that step.

Only ``action_mode="hdg"`` is supported (the mode all trained workers use) —
``"wpt"`` mode's "run inner sim ticks until this aircraft's own waypoint is
reached" semantics don't batch cleanly across aircraft with different
reach times, and no trained worker needs it here.

Unlike the single-agent env, this env never samples its own RTA/goal time —
``CPSManager`` (backed by ``ETASurrogate``) is the sole source of temporal
goals in a CPS-coordinated evaluation, so there is no ``rta_sampler``
parameter here. A spawned aircraft's ``desired_goal`` starts with only the
spatial (x, y) IAF target populated (t=0 placeholder); the CPS integration
loop calls :meth:`set_tta` once it has computed that aircraft's target time
of arrival (typically within one decision step of spawning), and
:meth:`get_active_batch` lets the caller re-fetch a fresh ``(obs, info_list)``
after any :meth:`set_tta`/:meth:`set_runway` calls, before invoking the
frozen worker policy.

All spatial/temporal constants and the runway/IAF geometry are imported
verbatim from :mod:`bluesky_gym.envs.pathplanning_goal_env` — they must not
be redefined, since the frozen SAC worker was trained against those exact
normalisations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.path import Path

import bluesky as bs
from bluesky.traffic import Route
from bluesky_gym.envs.common.screen_dummy import ScreenDummy
import bluesky_gym.envs.common.functions as fn

import gymnasium as gym
from gymnasium import spaces

from bluesky_gym.envs.pathplanning_goal_env import (
    SCHIPHOL,
    NM2KM,
    NM2M,
    RUNWAYS_SCHIPHOL_FAF,
    ALL_RUNWAYS,
    OVERLAPPING_RUNWAYS,
    FAF_DISTANCE,
    IAF_DISTANCE,
    MAX_DISTANCE,
    MAX_TIME,
    SPEED,
    ALTITUDE,
    SIM_DT,
    ACTION_TIME,
    ACTION_FREQUENCY,
    RTA_TOLERANCE,
    POPULATION_WEIGHT,
    PATH_LENGTH_WEIGHT,
)


# Kill-switch between the two _check_terminal implementations, both kept in
# this file: True (default) uses _check_terminal_batched (one vectorized
# numpy call per tick across all active aircraft, ~2.8x faster -- see
# cps_coordination/testing/prototype_numpy_path_check.py for the validation
# this was ported from); False uses the original per-slot, per-shape
# matplotlib.path.Path.intersects_path loop (_check_terminal). Both were
# verified bit-for-bit equivalent via validate_multiagent_env.py and a
# metrics-identical M=30/8-combo compare -- toggle this if you need to A/B
# them again, or as a quick rollback without reverting code.
USE_NUMPY_TERMINAL_CHECK = True


# --------------------------------------------------------------------- #
# Batched terminal-condition geometry (segment crossing OR full enclosure,
# matching matplotlib.path.Path.intersects_path's default filled=True
# semantics -- reverse-engineered and validated to bit-exact agreement
# against real matplotlib across 100k+ random + adversarial cases and
# 2000+ batched multi-shape/multi-aircraft ticks; see
# cps_coordination/testing/prototype_numpy_path_check.py for the full
# derivation/validation this was ported from). Tests one aircraft-tick's
# tiny movement segment against ALL of a runway pool's SINK/RESTRICT
# shapes in one vectorized call instead of one matplotlib call per shape
# per aircraft -- see _check_terminal_batched below for how the result
# maps back to per-aircraft death_cause/reward, preserving the exact
# sink > restrict > other-runway priority order _check_terminal used.
# --------------------------------------------------------------------- #

def _cross_batch(ox, oy, ux, uy, vx, vy):
    return (ux - ox) * (vy - oy) - (uy - oy) * (vx - ox)


def _on_segment_batch(ux, uy, vx, vy, wx, wy):
    return (np.minimum(ux, vx) <= wx) & (wx <= np.maximum(ux, vx)) & \
           (np.minimum(uy, vy) <= wy) & (wy <= np.maximum(uy, vy))


def _segment_hits_batch(p1s: np.ndarray, p2s: np.ndarray, ax, ay, bx, by,
                         boundaries: np.ndarray) -> np.ndarray:
    """(n_agents, 2) x edges -> (n_agents, n_groups) segment-crossing bool,
    reduced per shape group via np.logical_or.reduceat."""
    p1x, p1y = p1s[:, None, 0], p1s[:, None, 1]
    p2x, p2y = p2s[:, None, 0], p2s[:, None, 1]

    d1 = _cross_batch(ax, ay, bx, by, p1x, p1y)
    d2 = _cross_batch(ax, ay, bx, by, p2x, p2y)
    d3 = _cross_batch(p1x, p1y, p2x, p2y, ax, ay)
    d4 = _cross_batch(p1x, p1y, p2x, p2y, bx, by)

    proper = (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))) & \
             (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))

    touching = (
        ((d1 == 0) & _on_segment_batch(ax, ay, bx, by, p1x, p1y)) |
        ((d2 == 0) & _on_segment_batch(ax, ay, bx, by, p2x, p2y)) |
        ((d3 == 0) & _on_segment_batch(p1x, p1y, p2x, p2y, ax, ay)) |
        ((d4 == 0) & _on_segment_batch(p1x, p1y, p2x, p2y, bx, by))
    )
    edge_hits = proper | touching
    return np.logical_or.reduceat(edge_hits, boundaries, axis=1)


def _point_in_polygon_crossings_batch(pts: np.ndarray, ax, ay, bx, by) -> np.ndarray:
    """Ray-cast crossing test (pts: (n_pts, 2)) against closing (wraparound-
    included) edges; caller reduces+parity-checks per shape group."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = (bx - ax) * (y - ay) / (by - ay) + ax
    return ((ay > y) != (by > y)) & (x < x_intersect)


def _filled_hits_batch(
    p1s: np.ndarray, p2s: np.ndarray,
    open_ax, open_ay, open_bx, open_by, open_boundaries: np.ndarray,
    close_ax, close_ay, close_bx, close_by, close_boundaries: np.ndarray,
) -> np.ndarray:
    """(n_agents, n_groups) bool: segment crossing OR both query endpoints
    enclosed -- the exact behavior of
    ``shape_path.intersects_path(query_path)`` (matplotlib's default
    filled=True), for every active aircraft against every shape in one call."""
    seg_hits = _segment_hits_batch(p1s, p2s, open_ax, open_ay, open_bx, open_by, open_boundaries)

    n_agents = p1s.shape[0]
    stacked = np.concatenate([p1s, p2s], axis=0)  # (2*n_agents, 2)
    crossings = _point_in_polygon_crossings_batch(stacked, close_ax, close_ay, close_bx, close_by)
    counts = np.add.reduceat(crossings.astype(np.int64), close_boundaries, axis=1)
    inside = (counts % 2) == 1
    p1_in, p2_in = inside[:n_agents], inside[n_agents:]

    return seg_hits | (p1_in & p2_in)


class MultiAgentPathPlanningGoalEnv(gym.Env):
    """Multi-aircraft, single-BlueSky-instance version of ``PathPlanningGoalEnv``.

    Parameters
    ----------
    runways : list[str] | None
        Pool of runway IDs to sample from. Defaults to all 12 runways.
    max_concurrent_aircraft : int
        Maximum number of aircraft simultaneously active ("slots").
    n_aircraft_total : int
        Total number of arrivals to generate over the episode. A new
        aircraft is spawned into any freed slot once its scheduled arrival
        time (see ``spawn_window_s``) has elapsed, until this many have
        been spawned. The episode ends once this many have spawned *and*
        terminated.
    spawn_window_s : float
        Width (seconds) of the arrival window. Each arrival's scheduled
        spawn time is drawn i.i.d. ``Uniform(0, spawn_window_s)`` at
        ``reset()`` (sorted, so arrivals are offered in ascending time
        order); an arrival spawns into a free slot once its scheduled time
        has elapsed *and* a slot is free (the schedule is a lower bound on
        spawn time, not an exact instant — a slot may still be occupied).
        Default ``0.0`` reproduces the original behavior exactly: every
        arrival is eligible from time zero, so the initial wave spawns
        immediately at ``reset()`` and any freed slot refills on the next
        decision step.
    """

    metadata = {"render_modes": [], "render_fps": 0}

    def __init__(
        self,
        render_mode=None,
        runways: list | None = None,
        action_mode: str = "hdg",
        max_concurrent_aircraft: int = 5,
        n_aircraft_total: int = 20,
        spawn_window_s: float = 0.0,
    ):
        if action_mode != "hdg":
            raise NotImplementedError(
                "MultiAgentPathPlanningGoalEnv only supports action_mode='hdg' "
                "(the mode all trained workers use); 'wpt' mode's per-aircraft "
                "until-waypoint-reached loop does not batch across aircraft."
            )
        assert render_mode is None, "MultiAgentPathPlanningGoalEnv is headless-only."

        self.runways = runways if runways is not None else ALL_RUNWAYS
        self.action_mode = action_mode
        self.render_mode = None

        self.max_concurrent_aircraft = max_concurrent_aircraft
        self.n_aircraft_total = n_aircraft_total
        self.spawn_window_s = spawn_window_s

        # Per-slot single-aircraft space (documentation/registry contract) —
        # runtime obs/action arrays stack this along a leading batch axis.
        obs_shape = (3,)
        goal_shape = (3,)
        act_shape = (2,)
        goal_space = spaces.Box(-1.5, 1.5, shape=goal_shape, dtype=np.float64)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-1.5, 1.5, shape=obs_shape, dtype=np.float64),
            "achieved_goal": goal_space,
            "desired_goal": goal_space,
        })
        self.action_space = spaces.Box(-1, 1, shape=act_shape, dtype=np.float64)

        # ── bluesky init (once; bs.traf/bs.scr are process-wide singletons) ──
        bs.init(mode="sim", detached=True)
        bs.scr = ScreenDummy()
        bs.stack.stack(f"DT {SIM_DT};FF")

        self.pop_array = np.genfromtxt(
            "bluesky_gym/envs/data/population_1km.csv", delimiter=" "
        )
        self.x_array = np.genfromtxt("bluesky_gym/envs/data/x_array.csv", delimiter=" ")
        self.y_array = np.genfromtxt("bluesky_gym/envs/data/y_array.csv", delimiter=" ")
        self.x_max = np.max(self.x_array)
        self.y_max = np.max(self.y_array)
        self.cell_size = 1000
        self.projection_size = 30

        self.population_weight = POPULATION_WEIGHT
        self.path_length_weight = PATH_LENGTH_WEIGHT

        # SINK{rwy}/RESTRICT{rwy} shape coordinates are static for the whole
        # process lifetime once registered below -- cache their Path objects
        # instead of rebuilding one from scratch on every _check_terminal call
        # (was ~1.3M Path()/np.reshape() constructions per M=5 profiling run).
        self._shape_path_cache: Dict[str, Path] = {}

        # Batched terminal-check geometry (segment+enclosure across all
        # runways' shapes in one call, see _check_terminal_batched) --
        # built lazily on first use, same reasoning as _shape_path_cache
        # above: shape coordinates aren't guaranteed registered in
        # bs.tools.areafilter.basic_shapes until after the POLYLINE stack
        # commands below are actually processed.
        self._batch_shapes_built = False

        # Bound ONCE here rather than branched on every tick in step()'s
        # inner loop (up to ACTION_FREQUENCY=24x per step() call) -- the
        # closest Python equivalent of a C preprocessor #ifdef/#else
        # selecting which implementation gets compiled in, since Python has
        # no compile-time macros: pick the method reference once, call it
        # unconditionally thereafter. See USE_NUMPY_TERMINAL_CHECK above.
        self._check_terminal_impl = (
            self._check_terminal_batched if USE_NUMPY_TERMINAL_CHECK else self._check_terminal_looped
        )

        self._set_terminal_conditions(self.runways)

        self._reset_slot_state()
        self._acid_to_idx: Dict[str, int] = {}
        self._active_slots: List[int] = []

    @property
    def active_slots(self) -> List[int]:
        """Slots active as of the last reset()/step() call, in the exact
        row order of the obs/info batch that call returned."""
        return list(self._active_slots)

    def get_active_batch(self) -> Tuple[dict, List[dict]]:
        """Recompute ``(obs, info_list)`` for the currently active slots.

        Call this after :meth:`set_tta`/:meth:`set_runway` (both change
        ``desired_goal``/``current_runway``, which only ``info_list`` and
        ``obs`` reflect once recomputed), and always call it after
        :meth:`step` before the next ``model.predict()`` — see the module
        docstring for why ``step()``'s own return isn't safe to reuse there
        once any slot has despawned/respawned.
        """
        slots = self._active_slots
        obs = self._get_obs_batched(slots)
        info_list = [self._get_info(slot) for slot in slots]
        return obs, info_list

    # ------------------------------------------------------------------ #
    # Slot bookkeeping
    # ------------------------------------------------------------------ #

    def _reset_slot_state(self) -> None:
        n = self.max_concurrent_aircraft
        self.acid_slots: List[Optional[str]] = [None] * n
        self.current_runway: List[Optional[str]] = [None] * n
        self.goal_vector: List[Optional[np.ndarray]] = [None] * n
        self.non_overlapping_runways: List[List[str]] = [[] for _ in range(n)]
        self.prev_lat: List[float] = [0.0] * n
        self.prev_lon: List[float] = [0.0] * n
        self.simt: List[float] = [0.0] * n
        self.has_tta: List[bool] = [False] * n
        self.death_cause: List[Optional[str]] = [None] * n
        self.step_reward: List[float] = [0.0] * n
        self.segment_reward: List[float] = [0.0] * n
        self.total_reward: List[float] = [0.0] * n
        self.average_noise: List[float] = [0.0] * n
        self.average_path: List[float] = [0.0] * n
        self._slot_spawn_time: List[float] = [0.0] * n
        self._n_spawned = 0
        self._n_terminated = 0

    def _reindex_all(self) -> None:
        """Full rebuild of acid->traffic-index map. Required after any delete()
        since deletions shift every later aircraft's index down by one."""
        self._acid_to_idx = {a: i for i, a in enumerate(bs.traf.id)}

    def is_episode_done(self) -> bool:
        return self._n_terminated >= self.n_aircraft_total and not self._active_slots

    # ------------------------------------------------------------------ #
    # Core API (non-standard, batched — see module docstring)
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        bs.traf.reset()

        self._reset_slot_state()
        self._acid_to_idx = {}

        self._episode_time = 0.0
        self._next_arrival_idx = 0
        if self.spawn_window_s > 0:
            self._spawn_schedule = np.sort(
                self.np_random.uniform(0.0, self.spawn_window_s, size=self.n_aircraft_total)
            )
        else:
            # Every arrival eligible from time zero — reproduces the
            # original single-wave-then-instant-refill behavior exactly.
            self._spawn_schedule = np.zeros(self.n_aircraft_total)

        self._maybe_spawn_scheduled_arrivals()

        self._active_slots = [s for s in range(self.max_concurrent_aircraft)
                               if self.acid_slots[s] is not None]

        obs = self._get_obs_batched(self._active_slots)
        info_list = [self._get_info(slot) for slot in self._active_slots]
        return obs, info_list

    def step(self, actions: np.ndarray):
        slots = self._active_slots
        actions = np.asarray(actions, dtype=np.float64)
        assert actions.shape[0] == len(slots), (
            f"actions batch size {actions.shape[0]} must match the active-slot "
            f"count {len(slots)} from the previous reset()/step() call"
        )

        idxs = {slot: self._acid_to_idx[self.acid_slots[slot]] for slot in slots}

        for slot in slots:
            self.step_reward[slot] = 0.0
            self.segment_reward[slot] = 0.0

        self._set_action_batched(slots, idxs, actions)

        frozen_obs: Dict[int, dict] = {}
        frozen_info: Dict[int, dict] = {}
        terminated = {slot: False for slot in slots}
        truncated = {slot: False for slot in slots}
        still_active = set(slots)

        for _ in range(ACTION_FREQUENCY):
            if not still_active:
                break
            bs.sim.step()
            active_list = list(still_active)
            for slot in active_list:
                idx = idxs[slot]
                self.simt[slot] += bs.sim.simdt
                self._update_reward(slot, idx)

            term_trunc = self._check_terminal_impl(active_list, idxs)
            for slot in active_list:
                term, trunc = term_trunc[slot]
                if term or trunc:
                    idx = idxs[slot]
                    terminated[slot] = term
                    truncated[slot] = trunc
                    frozen_obs[slot] = self._get_obs_single(slot, idx)
                    frozen_info[slot] = self._get_info(slot)
                    still_active.discard(slot)

        for slot in still_active:
            idx = idxs[slot]
            frozen_obs[slot] = self._get_obs_single(slot, idx)
            frozen_info[slot] = self._get_info(slot)

        obs_batched = self._stack_obs([frozen_obs[slot] for slot in slots])
        rewards = np.array([self.segment_reward[slot] for slot in slots])
        term_arr = np.array([terminated[slot] for slot in slots])
        trunc_arr = np.array([truncated[slot] for slot in slots])
        info_list = [frozen_info[slot] for slot in slots]

        self._episode_time += ACTION_TIME
        self._finalize_step(slots, terminated, truncated, idxs)

        return obs_batched, rewards, term_arr, trunc_arr, info_list

    def render(self):
        pass

    def close(self):
        bs.traf.reset()
        self._reset_slot_state()
        self._acid_to_idx = {}
        self._active_slots = []

    # ------------------------------------------------------------------ #
    # Lifecycle: spawn / despawn
    # ------------------------------------------------------------------ #

    def _spawn_into_slot(self, slot: int) -> None:
        acid = f"AC{slot:03d}"
        spawn_lat, spawn_lon, spawn_heading = self._get_spawn()
        bs.traf.cre(acid, "a320", spawn_lat, spawn_lon, spawn_heading, ALTITUDE, SPEED)
        idx = bs.traf.ntraf - 1
        self._acid_to_idx[acid] = idx

        acrte = Route._routes.get(acid)
        acrte.delrte(idx)

        bs.traf.ap.setdest(idx, "EHAM")
        bs.traf.ap.setLNAV(idx, True)
        bs.traf.ap.route[idx].addwptMode(idx, "FLYOVER")

        self.acid_slots[slot] = acid
        self.current_runway[slot] = self.np_random.choice(self.runways)
        self.non_overlapping_runways[slot] = self._compute_non_overlapping_runways(
            self.current_runway[slot]
        )
        self.prev_lat[slot] = float(bs.traf.lat[idx])
        self.prev_lon[slot] = float(bs.traf.lon[idx])
        self.simt[slot] = 0.0
        self._slot_spawn_time[slot] = self._episode_time
        self.has_tta[slot] = False
        self.death_cause[slot] = None
        self.step_reward[slot] = 0.0
        self.segment_reward[slot] = 0.0
        self.total_reward[slot] = 0.0
        self.average_noise[slot] = 0.0
        self.average_path[slot] = 0.0

        self.goal_vector[slot] = self._compute_goal_vector(self.current_runway[slot])
        self._n_spawned += 1

    def _finalize_step(
        self,
        prev_slots: List[int],
        terminated: Dict[int, bool],
        truncated: Dict[int, bool],
        idxs: Dict[int, int],
    ) -> None:
        done_slots = [s for s in prev_slots if terminated[s] or truncated[s]]
        if done_slots:
            del_idxs = sorted(idxs[s] for s in done_slots)
            bs.traf.delete(np.array(del_idxs, dtype=int))
            for slot in done_slots:
                self.acid_slots[slot] = None
                self._n_terminated += 1
            self._reindex_all()

        self._maybe_spawn_scheduled_arrivals()

        self._active_slots = [
            s for s in range(self.max_concurrent_aircraft) if self.acid_slots[s] is not None
        ]

    def _maybe_spawn_scheduled_arrivals(self) -> None:
        """Spawn any arrivals whose scheduled time has elapsed into free slots.

        ``_spawn_schedule`` (sorted ascending, built in :meth:`reset`) is a
        lower bound on each arrival's spawn time, not an exact instant — an
        arrival only spawns once its scheduled time has elapsed *and* a
        slot is free. Free slots are filled lowest-index first, which
        reproduces today's deterministic slot-fill order exactly when
        ``spawn_window_s == 0`` (every scheduled time is 0, so this reduces
        to "fill every free slot immediately").
        """
        free_slots = [
            s for s in range(self.max_concurrent_aircraft) if self.acid_slots[s] is None
        ]
        for slot in free_slots:
            if self._next_arrival_idx >= self.n_aircraft_total:
                break
            if self._spawn_schedule[self._next_arrival_idx] > self._episode_time:
                break
            self._spawn_into_slot(slot)
            self._next_arrival_idx += 1

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _get_obs_single(self, slot: int, idx: int) -> dict:
        brg, dis = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[idx], bs.traf.lon[idx]
        )
        brg = np.radians(brg)
        dis = dis * NM2KM / MAX_DISTANCE

        x = np.sin(brg) * dis
        y = np.cos(brg) * dis
        t = self.simt[slot] / MAX_TIME

        obs_vec = np.array([x, y, t], dtype=np.float64)
        t_achieved = t if self.has_tta[slot] else 0.0
        achieved_vec = np.array([x, y, t_achieved], dtype=np.float64)

        return {
            "observation": obs_vec,
            "achieved_goal": achieved_vec,
            "desired_goal": self.goal_vector[slot].copy(),
        }

    def _get_obs_batched(self, slots: List[int]) -> dict:
        rows = []
        for slot in slots:
            idx = self._acid_to_idx[self.acid_slots[slot]]
            rows.append(self._get_obs_single(slot, idx))
        return self._stack_obs(rows)

    @staticmethod
    def _stack_obs(rows: List[dict]) -> dict:
        if not rows:
            return {
                "observation": np.zeros((0, 3), dtype=np.float64),
                "achieved_goal": np.zeros((0, 3), dtype=np.float64),
                "desired_goal": np.zeros((0, 3), dtype=np.float64),
            }
        return {
            key: np.stack([row[key] for row in rows], axis=0)
            for key in ("observation", "achieved_goal", "desired_goal")
        }

    def _compute_goal_vector(self, runway: str) -> np.ndarray:
        """Spatial (x, y) IAF target for *runway*. The temporal component (t)
        is left at 0.0 here — CPSManager is the sole source of TTAs, injected
        separately via :meth:`set_tta`."""
        rwy_info = RUNWAYS_SCHIPHOL_FAF[runway]

        iaf_lat, iaf_lon = fn.get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE + IAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        goal_brg, goal_dis = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], iaf_lat, iaf_lon)
        goal_brg = np.radians(goal_brg)
        goal_dis = goal_dis * NM2KM / MAX_DISTANCE

        goal_x = np.sin(goal_brg) * goal_dis
        goal_y = np.cos(goal_brg) * goal_dis

        return np.array([goal_x, goal_y, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Dynamic runway steering (CPS integration hook)
    # ------------------------------------------------------------------ #

    def set_runway(self, slot: int, runway_id: str) -> None:
        """Re-target slot's success/terminal-condition runway mid-episode.

        Called by the CPS integration loop whenever CPSManager's
        dynamically-assigned runway differs from what this env currently
        has recorded for the slot. Only the spatial/success-check target
        changes here; the CPS-assigned TTA is injected separately into
        ``desired_goal[..., -1]`` by the caller.
        """
        if self.acid_slots[slot] is None or runway_id == self.current_runway[slot]:
            return
        self.current_runway[slot] = runway_id
        self.non_overlapping_runways[slot] = self._compute_non_overlapping_runways(runway_id)
        new_goal = self._compute_goal_vector(runway_id)
        # Preserve the CPS-assigned temporal target (t component) if one has
        # already been injected; only the spatial (x, y) target moves here.
        new_goal[2] = self.goal_vector[slot][2]
        self.goal_vector[slot] = new_goal

    def set_tta(self, slot: int, tta_seconds: float) -> None:
        """Inject CPSManager's assigned target time of arrival for *slot*.

        Called by the CPS integration loop for every aircraft whose TTA
        ``CPSManager.update_fleet`` reports as changed (including on first
        assignment right after spawn). Only the temporal (t) component of
        ``desired_goal`` is touched; the spatial target is set separately by
        :meth:`set_runway`.

        ``tta_seconds`` is CPSManager's absolute target on the episode's
        global clock (necessarily so — RECAT-EU sequencing compares TTAs
        across aircraft with different spawn times on one shared timeline).
        Every temporal quantity in this env's own observation/goal space is
        *local*, zeroed at this slot's own spawn instant (matching the
        frozen worker's training convention), so the absolute TTA must be
        converted to "seconds since this aircraft's spawn" before being
        written into ``goal_vector``. For any slot spawned at global time 0
        (every slot under the default ``spawn_window_s=0``, and always true
        for a slot's very first occupant) this is a no-op: local == global.
        """
        if self.acid_slots[slot] is None:
            return
        local_tta = tta_seconds - self._slot_spawn_time[slot]
        self.goal_vector[slot][2] = local_tta / MAX_TIME
        self.has_tta[slot] = True

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #

    def _update_reward(self, slot: int, idx: int) -> None:
        path_length = self._get_path_length(idx) * self.path_length_weight
        population_exposure = self._get_population_exposure(idx) * self.population_weight
        self.average_path[slot] += path_length
        self.average_noise[slot] += population_exposure

        tick_reward = path_length + population_exposure
        self.step_reward[slot] += tick_reward
        self.segment_reward[slot] += tick_reward

    def _get_path_length(self, idx: int) -> float:
        return bs.traf.tas[idx] * SIM_DT / 1852.0

    def _get_population_exposure(self, idx: int) -> float:
        brg, dist = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], bs.traf.lat[idx], bs.traf.lon[idx])
        x = np.sin(np.radians(brg)) * dist * NM2M
        y = np.cos(np.radians(brg)) * dist * NM2M
        z = bs.traf.alt[idx]

        x_index_min = int(((x + self.x_max) / self.cell_size) - self.projection_size)
        x_index_max = int(((x + self.x_max) / self.cell_size) + self.projection_size)
        y_index_min = int(((self.y_max - y) / self.cell_size) - self.projection_size)
        y_index_max = int(((self.y_max - y) / self.cell_size) + self.projection_size)

        distance2 = (
            (self.x_array[y_index_min:y_index_max, x_index_min:x_index_max] - x) ** 2
            + (self.y_array[y_index_min:y_index_max, x_index_min:x_index_max] - y) ** 2
            + z ** 2
        )
        return np.sum(
            self.pop_array[y_index_min:y_index_max, x_index_min:x_index_max] / distance2
        )

    # ------------------------------------------------------------------ #
    # Terminal conditions
    # ------------------------------------------------------------------ #

    def _rta_penalty_mult(self, slot: int) -> float:
        if not self.has_tta[slot]:
            return 1.0
        abs_x = abs(self.goal_vector[slot][2] - (self.simt[slot] / MAX_TIME))
        if abs_x <= RTA_TOLERANCE:
            return 1.0 - (abs_x / RTA_TOLERANCE) ** 2
        return 0.0

    def _get_shape_path(self, name: str) -> Path:
        cached = self._shape_path_cache.get(name)
        if cached is None:
            coords = bs.tools.areafilter.basic_shapes[name].coordinates
            cached = Path(np.reshape(coords, (-1, 2)))
            self._shape_path_cache[name] = cached
        return cached

    def _check_terminal(self, slot: int, idx: int) -> Tuple[bool, bool]:
        lat, lon = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
        line_ac = Path(np.array([[self.prev_lat[slot], self.prev_lon[slot]], [lat, lon]]))
        self.prev_lat[slot] = lat
        self.prev_lon[slot] = lon

        rwy = self.current_runway[slot]
        target_sink = self._get_shape_path(f"SINK{rwy}")
        if target_sink.intersects_path(line_ac):
            self.segment_reward[slot] += 10.0 * self._rta_penalty_mult(slot)
            self.death_cause[slot] = "success"
            return True, False

        target_restrict = self._get_shape_path(f"RESTRICT{rwy}")
        if target_restrict.intersects_path(line_ac):
            self.segment_reward[slot] += -1.0
            self.death_cause[slot] = "restrict"
            return True, False

        for other_rwy in self.non_overlapping_runways[slot]:
            other_sink = self._get_shape_path(f"SINK{other_rwy}")
            if other_sink.intersects_path(line_ac):
                self.segment_reward[slot] += -1.0
                self.death_cause[slot] = "wrong_runway"
                return True, False

            other_restrict = self._get_shape_path(f"RESTRICT{other_rwy}")
            if other_restrict.intersects_path(line_ac):
                self.segment_reward[slot] += -1.0
                self.death_cause[slot] = "restrict"
                return True, False

        if self.simt[slot] >= MAX_TIME:
            self.segment_reward[slot] += -1.0
            self.death_cause[slot] = "timeout"
            return False, True

        dis_origin = bs.tools.geo.kwikdist(SCHIPHOL[0], SCHIPHOL[1], lat, lon) * NM2KM
        if dis_origin > MAX_DISTANCE * 1.05:
            self.segment_reward[slot] += -1.0
            self.death_cause[slot] = "out_of_bounds"
            return False, True

        return False, False

    def _check_terminal_looped(
        self, slots: List[int], idxs: Dict[int, int]
    ) -> Dict[int, Tuple[bool, bool]]:
        """Same interface as _check_terminal_batched (one call per tick,
        covering every active slot) but delegates to the original per-slot,
        per-shape matplotlib.path.Path.intersects_path loop -- the USE_NUMPY_
        TERMINAL_CHECK=False path, kept for A/B comparison and rollback."""
        return {slot: self._check_terminal(slot, idxs[slot]) for slot in slots}

    def _build_batch_shapes(self) -> None:
        """One-time (lazy, like _get_shape_path) construction of the
        vectorized geometry inputs for _check_terminal_batched: every
        SINK{rwy}/RESTRICT{rwy} shape's edges flattened into one array per
        test type (segment-crossing edges, point-in-polygon closing edges),
        plus a per-runway priority-ordered list mirroring _check_terminal's
        own sink -> restrict -> other-runway short-circuit order (built from
        _compute_non_overlapping_runways, the same deterministic function
        that populates self.non_overlapping_runways per slot -- so the order
        here is guaranteed identical to what a per-slot call would use)."""
        if self._batch_shapes_built:
            return

        shape_names: List[str] = []
        for rwy in self.runways:
            shape_names.append(f"SINK{rwy}")
            shape_names.append(f"RESTRICT{rwy}")
        shape_index = {name: i for i, name in enumerate(shape_names)}
        polys = [self._get_shape_path(name).vertices for name in shape_names]

        open_starts, open_ends, open_group_ids = [], [], []
        for gi, poly in enumerate(polys):
            open_starts.append(poly[:-1])
            open_ends.append(poly[1:])
        open_edge_starts = np.concatenate(open_starts, axis=0)
        open_edge_ends = np.concatenate(open_ends, axis=0)
        open_counts = np.array([len(p) - 1 for p in polys])
        open_boundaries = np.concatenate([[0], np.cumsum(open_counts)[:-1]])

        close_starts, close_ends = [], []
        for poly in polys:
            close_starts.append(poly)
            close_ends.append(np.roll(poly, -1, axis=0))
        close_edge_starts = np.concatenate(close_starts, axis=0)
        close_edge_ends = np.concatenate(close_ends, axis=0)
        close_counts = np.array([len(p) for p in polys])
        close_boundaries = np.concatenate([[0], np.cumsum(close_counts)[:-1]])

        self._batch_open_ax = open_edge_starts[None, :, 0]
        self._batch_open_ay = open_edge_starts[None, :, 1]
        self._batch_open_bx = open_edge_ends[None, :, 0]
        self._batch_open_by = open_edge_ends[None, :, 1]
        self._batch_open_boundaries = open_boundaries

        self._batch_close_ax = close_edge_starts[None, :, 0]
        self._batch_close_ay = close_edge_starts[None, :, 1]
        self._batch_close_bx = close_edge_ends[None, :, 0]
        self._batch_close_by = close_edge_ends[None, :, 1]
        self._batch_close_boundaries = close_boundaries

        self._priority_order_by_runway: Dict[str, List[Tuple[int, str]]] = {}
        for rwy in self.runways:
            order = [
                (shape_index[f"SINK{rwy}"], "success"),
                (shape_index[f"RESTRICT{rwy}"], "restrict"),
            ]
            for other_rwy in self._compute_non_overlapping_runways(rwy):
                order.append((shape_index[f"SINK{other_rwy}"], "wrong_runway"))
                order.append((shape_index[f"RESTRICT{other_rwy}"], "restrict"))
            self._priority_order_by_runway[rwy] = order

        self._batch_shapes_built = True

    def _check_terminal_batched(
        self, slots: List[int], idxs: Dict[int, int]
    ) -> Dict[int, Tuple[bool, bool]]:
        """Batched replacement for calling _check_terminal once per slot:
        tests every active aircraft's tick movement against every runway's
        SINK/RESTRICT shape in ONE vectorized call (see _filled_hits_batch),
        then walks each aircraft's own priority-ordered shape list (built in
        _build_batch_shapes) to pick the first hit -- identical short-circuit
        semantics to _check_terminal's sequential if-return chain, just with
        the underlying per-shape test batched instead of looped."""
        self._build_batch_shapes()

        n = len(slots)
        p1s = np.empty((n, 2))
        p2s = np.empty((n, 2))
        for i, slot in enumerate(slots):
            idx = idxs[slot]
            lat, lon = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
            p1s[i] = (self.prev_lat[slot], self.prev_lon[slot])
            p2s[i] = (lat, lon)
            self.prev_lat[slot] = lat
            self.prev_lon[slot] = lon

        hits = _filled_hits_batch(
            p1s, p2s,
            self._batch_open_ax, self._batch_open_ay, self._batch_open_bx, self._batch_open_by,
            self._batch_open_boundaries,
            self._batch_close_ax, self._batch_close_ay, self._batch_close_bx, self._batch_close_by,
            self._batch_close_boundaries,
        )

        results: Dict[int, Tuple[bool, bool]] = {}
        for i, slot in enumerate(slots):
            rwy = self.current_runway[slot]
            winner = None
            for shape_idx, label in self._priority_order_by_runway[rwy]:
                if hits[i, shape_idx]:
                    winner = label
                    break

            if winner == "success":
                self.segment_reward[slot] += 10.0 * self._rta_penalty_mult(slot)
                self.death_cause[slot] = "success"
                results[slot] = (True, False)
                continue
            if winner == "restrict":
                self.segment_reward[slot] += -1.0
                self.death_cause[slot] = "restrict"
                results[slot] = (True, False)
                continue
            if winner == "wrong_runway":
                self.segment_reward[slot] += -1.0
                self.death_cause[slot] = "wrong_runway"
                results[slot] = (True, False)
                continue

            if self.simt[slot] >= MAX_TIME:
                self.segment_reward[slot] += -1.0
                self.death_cause[slot] = "timeout"
                results[slot] = (False, True)
                continue

            lat, lon = p2s[i]
            dis_origin = bs.tools.geo.kwikdist(SCHIPHOL[0], SCHIPHOL[1], lat, lon) * NM2KM
            if dis_origin > MAX_DISTANCE * 1.05:
                self.segment_reward[slot] += -1.0
                self.death_cause[slot] = "out_of_bounds"
                results[slot] = (False, True)
                continue

            results[slot] = (False, False)

        return results

    # ------------------------------------------------------------------ #
    # Action
    # ------------------------------------------------------------------ #

    def _set_action_batched(
        self, slots: List[int], idxs: Dict[int, int], actions: np.ndarray
    ) -> None:
        bearings = np.rad2deg(np.arctan2(actions[:, 0], actions[:, 1]))
        for local_i, slot in enumerate(slots):
            bs.traf.ap.selhdgcmd(idxs[slot], float(bearings[local_i]))

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    def _get_info(self, slot: int) -> dict:
        goal_vector = self.goal_vector[slot]
        on_time = (
            abs(goal_vector[2] - (self.simt[slot] / MAX_TIME)) <= RTA_TOLERANCE
            if self.has_tta[slot] else True
        )
        correct_runway = self.death_cause[slot] == "success"
        is_success = on_time and correct_runway

        acid = self.acid_slots[slot]
        idx = self._acid_to_idx.get(acid) if acid is not None else None
        hdg = float(np.radians(bs.traf.hdg[idx])) if idx is not None else 0.0

        return {
            "slot": slot,
            "acid": acid,
            "is_success": is_success,
            "death_cause": self.death_cause[slot],
            "sim_time": self.simt[slot],
            # Absolute (global-clock) spawn instant of this slot's current
            # aircraft. "sim_time" above is local (elapsed since spawn, for
            # the frozen worker's own per-episode timing/rewards) -- a
            # caller comparing landing times *across* aircraft with
            # different spawn offsets (throughput, separation compliance,
            # RTA error against a CPSManager-assigned absolute TTA) needs
            # spawn_time + sim_time, not sim_time alone.
            "spawn_time": self._slot_spawn_time[slot],
            "step_reward": self.step_reward[slot],
            "total_reward": self.total_reward[slot],
            "average_path_rew": self.average_path[slot],
            "average_noise_rew": self.average_noise[slot],
            "population_weight": self.population_weight,
            "path_length_weight": self.path_length_weight,
            "current_runway": self.current_runway[slot],
            "goal_vector": goal_vector.tolist(),
            "on_time": on_time,
            "correct_runway": correct_runway,
            "heading": hdg,
        }

    # ------------------------------------------------------------------ #
    # Spawn / geometry helpers (reused verbatim from PathPlanningGoalEnv)
    # ------------------------------------------------------------------ #

    def _get_spawn(self):
        """Spawn at a fixed edge radius (random bearing only).

        Deliberately diverges from ``PathPlanningGoalEnv._get_spawn`` (which
        draws spawn *distance* uniformly across the whole annulus — intentional
        domain randomization for training the frozen worker across variable
        distance-to-go). For CPS coordination evaluation, aircraft should
        model arrivals entering at a fixed sector/TMA boundary, so distance is
        pinned at a fixed 0.9 * MAX_DISTANCE radius (a bit inside the old
        0.95 * MAX_DISTANCE ceiling, with headroom below the
        1.05 * MAX_DISTANCE out_of_bounds threshold) rather than resampled.
        """
        spawn_bearing = self.np_random.uniform(0, 360)
        spawn_distance = 0.9 * MAX_DISTANCE
        spawn_lat, spawn_lon = fn.get_point_at_distance(
            SCHIPHOL[0], SCHIPHOL[1], spawn_distance, spawn_bearing
        )
        spawn_heading = (spawn_bearing + 180 + 360) % 360
        return spawn_lat, spawn_lon, spawn_heading

    def _compute_non_overlapping_runways(self, current_runway: str) -> List[str]:
        runway_overlaps = OVERLAPPING_RUNWAYS.get(current_runway, [])
        return [
            rwy for rwy in self.runways
            if rwy != current_runway and rwy not in runway_overlaps
        ]

    def _set_terminal_conditions(self, runway_list) -> None:
        for rwy in runway_list:
            num_points = 36
            rwy_info = RUNWAYS_SCHIPHOL_FAF[rwy]

            faf_lat, faf_lon = fn.get_point_at_distance(
                rwy_info["lat"], rwy_info["lon"], FAF_DISTANCE, rwy_info["track"] - 180
            )

            cw_bound = ((rwy_info["track"] - 180 + 360) % 360) + (60 / 2)
            ccw_bound = ((rwy_info["track"] - 180 + 360) % 360) - (60 / 2)
            angles = np.linspace(cw_bound, ccw_bound, num_points)
            lat_iaf, lon_iaf = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, angles)

            command = f"POLYLINE SINK{rwy}"
            for i in range(len(lat_iaf)):
                command += f" {lat_iaf[i]} {lon_iaf[i]}"
            bs.stack.stack(command)
            bs.stack.stack(
                f"POLYLINE RESTRICT{rwy} {lat_iaf[0]} {lon_iaf[0]} "
                f"{faf_lat} {faf_lon} {lat_iaf[-1]} {lon_iaf[-1]}"
            )
            bs.stack.stack(f"COLOR RESTRICT{rwy} red")
