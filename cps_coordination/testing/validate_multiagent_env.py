"""
Regression gates for Phase III roadmap steps 1-2.

Step 1: at ``max_concurrent_aircraft=1, n_aircraft_total=1``,
``MultiAgentPathPlanningGoalEnv`` must reproduce ``PathPlanningGoalEnv-v0``
bit-for-bit given an *identical spawn point* and action sequence. This
isolates "did the single-aircraft stepping/reward/termination logic get
generalised correctly" from "does multi-aircraft slot/index bookkeeping
work" before any real concurrency is introduced.

Both envs' ``_get_spawn`` are monkeypatched to a fixed value for this check
only — production code is untouched. This is necessary because the two envs'
``_get_spawn`` are deliberately *not* identical: ``PathPlanningGoalEnv``
spawns at a uniformly-random distance (intentional domain randomization for
training the frozen worker across variable distance-to-go, see
``docs/paper/Thesis_Paper_draft.pdf``'s DTG-based training rationale), while
``MultiAgentPathPlanningGoalEnv`` was changed to spawn at a fixed edge radius
(0.95 * MAX_DISTANCE) for CPS coordination evaluation, modelling aircraft
entering at a fixed sector/TMA boundary rather than the worker's training
distribution. Pinning both to the same spawn point for this check re-isolates
the invariant it was designed for (stepping mechanics) from that now-
intentional spawn-distribution divergence, rather than comparing two
different random distributions and calling any mismatch a bug.

Step 2: at ``max_concurrent_aircraft=2, n_aircraft_total=3`` (so a mid-episode
``bs.traf.delete()`` and a respawn into the freed slot are both exercised),
confirm the acid->traffic-index remap survives the delete and that no
aircraft's observation ever "jumps" to another aircraft's position (which is
what index-confusion after a delete would look like), and that every
terminated aircraft's outcome is attributed to the correct slot/acid.

Run: python cps_coordination/testing/validate_multiagent_env.py
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

import bluesky_gym
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.multi_agent_pathplanning_env import MultiAgentPathPlanningGoalEnv
from bluesky_gym.envs.pathplanning_goal_env import SCHIPHOL

SEED = 42
ACTION_SEED = 123
MAX_STEPS = 300  # generous upper bound; episodes end well before MAX_TIME/ACTION_TIME

# Fixed spawn point for the step-1 bit-for-bit check (see module docstring for
# why both envs' _get_spawn are monkeypatched to this rather than relying on
# matching seeds to independently produce the same draw).
_FIXED_SPAWN_BEARING = 45.0
_FIXED_SPAWN_DISTANCE_KM = 200.0
_FIXED_SPAWN_LAT, _FIXED_SPAWN_LON = fn.get_point_at_distance(
    SCHIPHOL[0], SCHIPHOL[1], _FIXED_SPAWN_DISTANCE_KM, _FIXED_SPAWN_BEARING
)
_FIXED_SPAWN_HEADING = (_FIXED_SPAWN_BEARING + 180) % 360


def _fixed_spawn():
    """Stub matching ``_get_spawn``'s ``(lat, lon, heading)`` return, with no
    ``np_random`` draws — keeps the subsequent runway-choice draw in sync
    between the two envs, since neither env's real ``_get_spawn`` consumes
    the same number of random draws any more (single-agent draws 2 — bearing
    and distance; multi-agent now draws only 1 — bearing, distance fixed)."""
    return _FIXED_SPAWN_LAT, _FIXED_SPAWN_LON, _FIXED_SPAWN_HEADING

# Max plausible per-decision-step displacement in normalised (x, y) units:
# aircraft ground speed is on the order of SPEED=125 m/s, ACTION_TIME=120s
# per decision step -> ~15km, divided by MAX_DISTANCE=300km -> 0.05 per
# component. Genuine index-confusion bugs produce jumps between two
# independently-spawned aircraft's positions (typically >0.2), so this
# threshold has ample margin without being so loose it'd miss a real bug.
JUMP_THRESHOLD = 0.15


def _run_single_agent(seed: int, actions: np.ndarray):
    bluesky_gym.register_envs()
    env = gym.make("PathPlanningGoalEnv-v0").unwrapped
    env._get_spawn = _fixed_spawn
    obs, info = env.reset(seed=seed)
    trace = [(obs, 0.0, False, False, info)]
    for i in range(MAX_STEPS):
        obs, reward, terminated, truncated, info = env.step(actions[i])
        trace.append((obs, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    env.close()
    return trace


def _run_multi_agent(seed: int, actions: np.ndarray):
    env = MultiAgentPathPlanningGoalEnv(max_concurrent_aircraft=1, n_aircraft_total=1)
    env._get_spawn = _fixed_spawn
    obs_batched, info_list = env.reset(seed=seed)
    trace = [(obs_batched, 0.0, False, False, info_list[0])]
    for i in range(MAX_STEPS):
        action = actions[i].reshape(1, 2)
        obs_batched, rewards, terminated, truncated, info_list = env.step(action)
        trace.append((obs_batched, float(rewards[0]), bool(terminated[0]), bool(truncated[0]), info_list[0]))
        if terminated[0] or truncated[0]:
            break
    env.close()
    return trace


def _stack_single_obs(obs: dict) -> dict:
    """Give the single-agent obs dict a leading batch axis of 1 for comparison."""
    return {k: np.asarray(v)[None, :] for k, v in obs.items()}


def compare() -> bool:
    rng = np.random.default_rng(ACTION_SEED)
    actions = rng.uniform(-1, 1, size=(MAX_STEPS, 2))

    single_trace = _run_single_agent(SEED, actions)
    multi_trace = _run_multi_agent(SEED, actions)

    ok = True
    if len(single_trace) != len(multi_trace):
        print(f"FAIL: trace length mismatch — single={len(single_trace)} multi={len(multi_trace)}")
        ok = False

    n = min(len(single_trace), len(multi_trace))
    info_keys_to_compare = [
        "is_success", "death_cause", "sim_time", "current_runway",
        "on_time", "correct_runway",
    ]

    for i in range(n):
        s_obs, s_rew, s_term, s_trunc, s_info = single_trace[i]
        m_obs, m_rew, m_term, m_trunc, m_info = multi_trace[i]

        s_obs_b = _stack_single_obs(s_obs)
        for key in ("observation", "achieved_goal", "desired_goal"):
            if not np.allclose(s_obs_b[key], m_obs[key], atol=1e-9):
                print(f"FAIL step {i}: obs['{key}'] mismatch: single={s_obs_b[key]} multi={m_obs[key]}")
                ok = False

        if not math_close(s_rew, m_rew):
            print(f"FAIL step {i}: reward mismatch: single={s_rew} multi={m_rew}")
            ok = False

        if s_term != m_term or s_trunc != m_trunc:
            print(f"FAIL step {i}: terminated/truncated mismatch: "
                  f"single=({s_term},{s_trunc}) multi=({m_term},{m_trunc})")
            ok = False

        for key in info_keys_to_compare:
            if s_info.get(key) != m_info.get(key):
                print(f"FAIL step {i}: info['{key}'] mismatch: single={s_info.get(key)} multi={m_info.get(key)}")
                ok = False

    if ok:
        print(f"PASS: {n} steps matched bit-for-bit "
              f"(seed={SEED}, action_seed={ACTION_SEED}, "
              f"final death_cause={single_trace[-1][4].get('death_cause')})")
    return ok


def math_close(a: float, b: float, atol: float = 1e-9) -> bool:
    return abs(a - b) <= atol


def _run_index_bookkeeping_check(seed: int, max_concurrent: int, n_total: int, max_steps: int):
    """Drive N aircraft with random actions, tracking per-slot position
    continuity across step() calls (using the get_active_batch() contract —
    never step()'s own stale return — for the next action batch).

    Returns (violations, max_jump, outcomes, n_steps_run, n_distinct_acids).
    """
    env = MultiAgentPathPlanningGoalEnv(max_concurrent_aircraft=max_concurrent, n_aircraft_total=n_total)
    obs, info_list = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 777)

    last_obs_by_slot: dict[int, np.ndarray] = {}
    just_terminated_slots: set[int] = set()
    outcomes: list[dict] = []
    violations: list[tuple] = []
    distinct_acids: set[str] = set()
    max_jump = 0.0

    step_i = 0
    while step_i < max_steps and len(info_list) > 0:
        for row, info in enumerate(info_list):
            slot = info["slot"]
            distinct_acids.add(info["acid"])
            xy = obs["observation"][row][:2].copy()
            if slot in last_obs_by_slot and slot not in just_terminated_slots:
                jump = float(np.linalg.norm(xy - last_obs_by_slot[slot]))
                max_jump = max(max_jump, jump)
                if jump > JUMP_THRESHOLD:
                    violations.append((step_i, slot, info["acid"], jump))
            last_obs_by_slot[slot] = xy
            just_terminated_slots.discard(slot)

        actions = rng.uniform(-1, 1, size=(len(info_list), 2))
        _obs_terminal, _rewards, terminated, truncated, info_terminal = env.step(actions)

        for row, info in enumerate(info_terminal):
            if terminated[row] or truncated[row]:
                slot = info["slot"]
                just_terminated_slots.add(slot)
                outcomes.append({
                    "acid": info["acid"], "slot": slot,
                    "death_cause": info["death_cause"], "step": step_i,
                })

        # Mandatory: step()'s own return can be stale once a slot has
        # despawned/respawned — always refresh via get_active_batch().
        obs, info_list = env.get_active_batch()
        step_i += 1

    env.close()
    return violations, max_jump, outcomes, step_i, len(distinct_acids)


def check_index_bookkeeping() -> bool:
    max_concurrent, n_total, max_steps = 2, 3, 400
    violations, max_jump, outcomes, steps_run, n_distinct = _run_index_bookkeeping_check(
        SEED, max_concurrent, n_total, max_steps
    )

    ok = True
    if violations:
        print(f"FAIL: {len(violations)} position-continuity violations "
              f"(index confusion after delete()?): {violations[:5]}")
        ok = False

    # Note: acid strings are stable *per slot* by design (f"AC{slot:03d}"),
    # so a respawn into a freed slot intentionally reuses the same acid —
    # n_distinct is expected to be <= max_concurrent, not n_total. Distinct
    # spawn/termination *events* are what len(outcomes) checks below.
    if n_distinct > max_concurrent:
        print(f"FAIL: saw {n_distinct} distinct acid strings, more than "
              f"max_concurrent_aircraft={max_concurrent} — acid reuse-per-slot invariant broken")
        ok = False

    if len(outcomes) != n_total:
        print(f"FAIL: expected {n_total} terminal outcomes, got {len(outcomes)}: {outcomes}")
        ok = False

    respawn_happened = any(
        outcomes[i]["slot"] == outcomes[j]["slot"]
        for i in range(len(outcomes)) for j in range(i + 1, len(outcomes))
    )
    if not respawn_happened:
        print("FAIL: no slot was reused across a termination — test didn't exercise a respawn "
              "(the exact scenario roadmap step 2 needs covered)")
        ok = False

    if ok:
        print(f"PASS: N=2 index bookkeeping — {steps_run} steps, {n_distinct} distinct aircraft, "
              f"max position jump between consecutive obs for the same slot = {max_jump:.4f} "
              f"(threshold {JUMP_THRESHOLD}), outcomes={outcomes}")
    return ok


if __name__ == "__main__":
    passed_step1 = compare()
    passed_step2 = check_index_bookkeeping()
    raise SystemExit(0 if (passed_step1 and passed_step2) else 1)
