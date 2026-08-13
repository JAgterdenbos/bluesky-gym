"""
cps_coordination/scripts/measure_solo_flight_time_fixed_radius.py
---------------------------------------------------------------------
Measures the frozen worker's true zero-coordination flight time, spawning at
the SAME fixed radius (0.9 * MAX_DISTANCE) the CPS multi-agent env actually
uses -- NOT PathPlanningGoalEnv-v0's own training distribution, which samples
spawn distance uniformly across the whole annulus
(`bluesky_gym/envs/pathplanning_goal_env.py::_get_spawn`) and is therefore not
a fair "zero-coordination" reference for a Little's Law comparison against
CPS-coordinated dwell time (`multi_agent_pathplanning_env.py::_get_spawn`
spawns at a fixed radius, random bearing only -- confirmed intentional
divergence, see that method's own docstring).

Built for `.claude/plans/max_concurrent_aircraft_capacity_sweep.md`'s "Round 7"
(rigorous, non-circular Little's Law steady-state floor for the
`max_concurrent_aircraft` capacity sweep) -- an earlier pass in that round used
the mismatched training-distribution flight_time and got a materially wrong,
too-low floor; this script is the fix.

Reuses `diagnose_success_rate.py`'s exact model/RTA-sampler loading pattern
(`_find_pretrained_run_id`, `_load_matching_rta_sampler`) rather than
re-deriving it.

Usage
-----
  python cps_coordination/scripts/measure_solo_flight_time_fixed_radius.py \\
      --episodes 200 --runways 18R 27
"""

from __future__ import annotations

import argparse
import types

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import bluesky_gym
import bluesky_gym.envs.pathplanning_goal_env as ppg
from bluesky_gym.envs.common import functions as fn
from bluesky_gym.envs.pathplanning_goal_env import MAX_DISTANCE, PathPlanningGoalEnv

from cps_coordination.scripts.diagnose_success_rate import (
    _find_pretrained_run_id,
    _load_matching_rta_sampler,
)

CPS_SPAWN_FRAC = 0.9  # matches multi_agent_pathplanning_env.py::_get_spawn exactly


def _pin_spawn_radius(env: PathPlanningGoalEnv) -> None:
    """Monkey-patch env._get_spawn to the CPS env's exact fixed-radius formula
    (random bearing, distance pinned to 0.9*MAX_DISTANCE) instead of the
    single-agent training distribution's uniform-annulus sampling."""

    def patched(self):
        spawn_bearing = self.np_random.uniform(0, 360)
        spawn_distance = CPS_SPAWN_FRAC * MAX_DISTANCE
        spawn_lat, spawn_lon = fn.get_point_at_distance(
            ppg.SCHIPHOL[0], ppg.SCHIPHOL[1], spawn_distance, spawn_bearing
        )
        spawn_heading = (spawn_bearing + 180 + 360) % 360
        return spawn_lat, spawn_lon, spawn_heading

    env._get_spawn = types.MethodType(patched, env)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=5000)
    p.add_argument("--runways", type=str, nargs="*", default=["18R", "27"])
    p.add_argument("--total-arrivals-per-episode", type=int, default=50,
                   help="For the printed L_floor only -- matches the density this "
                        "sweep is sized for.")
    p.add_argument("--spawn-window-s", type=float, default=2400.0,
                   help="For the printed L_floor only.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bluesky_gym.register_envs()

    run_id = _find_pretrained_run_id()
    model_path = f"experiments/PathPlanningGoalEnv-v0/SAC/models/{run_id}/final_model"

    rta_sampler = _load_matching_rta_sampler(model_path + ".zip")
    env = gym.make(
        "PathPlanningGoalEnv-v0", rta_sampler=rta_sampler, runways=args.runways,
    ).unwrapped
    assert isinstance(env, PathPlanningGoalEnv)
    _pin_spawn_radius(env)

    model = SAC.load(model_path, env=env)

    flight_times = []
    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed_base + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        if info.get("is_success", False):
            successes += 1
            flight_times.append(info["sim_time"] / 60.0)
    env.close()

    ft = np.array(flight_times)
    print(f"run_id={run_id}, runways={args.runways}, spawn_radius={CPS_SPAWN_FRAC}*MAX_DISTANCE")
    print(f"n_episodes={args.episodes}, n_success={successes} ({successes/args.episodes:.1%})")
    print(f"flight_time (min): mean={ft.mean():.2f} median={np.median(ft):.2f} "
          f"std={ft.std(ddof=1):.2f} min={ft.min():.2f} max={ft.max():.2f}")
    w_min_s = ft.mean() * 60
    print(f"flight_time (s): mean={w_min_s:.1f}")

    lam_nominal = args.total_arrivals_per_episode / (args.spawn_window_s / 3600)
    l_floor = lam_nominal * (w_min_s / 3600)
    print(f"\nL_floor = lambda_nominal({lam_nominal:.1f} ac/h) * W_min({w_min_s:.0f}s) "
          f"= {l_floor:.2f} aircraft")


if __name__ == "__main__":
    main()
