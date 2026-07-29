"""
cps_coordination/testing/analyze_condition3_horizon.py
---------------------------------------------------------
Phase E (see .claude/eta_surrogate_accuracy_plan.md, Verification step 3):
bucket condition 3's per-aircraft ETA prediction error by time-since-first-
TTA-assignment, to test Finding 2's hypothesis that the `remaining_time_budget`
feature -- self-consistently informative only from the first TTA assignment
onward -- should concentrate any accuracy gain late-flight while leaving
early-flight error roughly flat.

Condition 3 (validate_surrogate.py's gate: multi-agent env, tta_mode="solo",
N=1, zero separation pressure) assigns a TTA every decision step starting at
each aircraft's own spawn step (see coordination_baseline.py::_run_episode,
the `tta_mode == "solo"` branch sets `env.set_tta` unconditionally every
step for every active aircraft) -- so "time since first TTA assignment" ==
elapsed time since spawn for that aircraft. Since condition 3 is N=1 with no
rolling arrivals, the loop's own `sim_time` (0 at reset, +ACTION_TIME per
step) already *is* that elapsed time -- no new assignment-time bookkeeping
needed, just per-step prediction capture instead of the terminal-only
`rta_error_cps` that `_EpisodeRecord` stores.

Bucketing is by *fraction* of that episode's own total flight duration
(elapsed / landing_time), not raw seconds, so "early-flight" vs "late-flight"
are comparable across episodes of different lengths.

Usage
-----
  python cps_coordination/testing/analyze_condition3_horizon.py \\
      --baseline cps_coordination/models/eta_surrogate.pkl \\
      --candidate cps_coordination/models/eta_surrogate_combined_candidate.pkl \\
      --episodes 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME
from cps_coordination.coordination.cps_manager import CPSManager
from cps_coordination.coordination.eta_surrogate import ETASurrogate
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.coordination_baseline import CPSCoordinationExperiment
from cps_coordination.testing.diagnose_success_rate import _make_experiment

SEED_BASE = 1000
_BUCKET_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0001]  # +epsilon so fraction==1.0 lands in the last bucket
_BUCKET_LABELS = ["0-25%", "25-50%", "50-75%", "75-100%"]


def run_condition3_horizon(
    experiment: CPSCoordinationExperiment,
    model,
    n_episodes: int,
    surrogate: ETASurrogate,
) -> np.ndarray:
    """Run N single-aircraft, solo-TTA, zero-separation episodes (condition
    3's exact setup), recording (elapsed_since_spawn, predicted_eta) at every
    decision step. Once each episode's true landing time is known at
    termination, converts to (fraction_of_flight_elapsed, abs_error) rows.

    Mirrors diagnose_success_rate.py's _run_episode_verbose loop (same
    fleet-building / cps_manager.update_fleet / set_tta calls for tta_mode=
    "solo") but captures structured per-step data instead of printing.

    Returns an (n_steps, 2) array of columns [fraction_elapsed, abs_error_s].
    """
    env = experiment._make_multi_agent_env(1)
    recat_matrix = experiment._load_recat_matrix()
    runways = experiment.cfg.env.env_kwargs.runways
    mode = experiment.cfg.model.runway_assignment_mode

    rows: List[Tuple[float, float]] = []
    for ep in range(n_episodes):
        cps_manager = CPSManager(
            k_cps=0,
            recat_matrix=recat_matrix,
            runway_assignment_mode=mode,
            delta_t_plan=120,
            delta_update=1.0,
            available_runways=runways,
            trajectory_buffer=TrajectoryBuffer(),
        )
        obs, info_list = env.reset(seed=SEED_BASE + ep)
        sim_time = 0.0
        steps: List[Tuple[float, float]] = []  # (elapsed, predicted_eta)

        while not env.is_episode_done():
            fleet = experiment._build_fleet(obs, info_list, sim_time)
            acid_to_slot = {info["acid"]: info["slot"] for info in info_list}

            cps_manager.update_fleet(aircraft=fleet, current_time=sim_time, surrogate=surrogate)
            for ac in fleet:
                slot = acid_to_slot[ac.acid]
                if ac.runway_id != env.current_runway[slot]:
                    env.set_runway(slot, ac.runway_id)
                env.set_tta(slot, ac.eta)
                steps.append((sim_time, ac.eta))

            obs, info_list = env.get_active_batch()
            actions, _ = model.predict(obs, deterministic=True)
            _obs_t, _rew, terminated, truncated, info_terminal = env.step(actions)

            for row, info in enumerate(info_terminal):
                if terminated[row] or truncated[row]:
                    landing_time = float(info.get("spawn_time", 0.0)) + float(
                        info.get("sim_time", sim_time)
                    )
                    for elapsed, predicted_eta in steps:
                        fraction = elapsed / landing_time if landing_time > 0 else 0.0
                        rows.append((fraction, abs(predicted_eta - landing_time)))

            sim_time += ACTION_TIME
            obs, info_list = env.get_active_batch()

    env.close()
    return np.array(rows)


def print_bucket_table(label: str, data: np.ndarray) -> None:
    print(f"\n=== {label} — error by fraction-of-flight-elapsed ===")
    print(f"{'bucket':<10}{'n':>8}{'mean_err_s':>14}{'median_err_s':>14}")
    fractions, errors = data[:, 0], data[:, 1]
    for lo, hi, name in zip(_BUCKET_EDGES[:-1], _BUCKET_EDGES[1:], _BUCKET_LABELS):
        mask = (fractions >= lo) & (fractions < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"{name:<10}{n:>8}{'--':>14}{'--':>14}")
            continue
        bucket_errs = errors[mask]
        print(f"{name:<10}{n:>8}{bucket_errs.mean():>14.1f}{np.median(bucket_errs):>14.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=Path("cps_coordination/models/eta_surrogate.pkl"))
    p.add_argument("--candidate", type=Path,
                   default=Path("cps_coordination/models/eta_surrogate_combined_candidate.pkl"))
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()

    experiment = _make_experiment(k_cps=0, mode="static", runways=None)
    model = experiment.make_model(experiment._make_multi_agent_env(1))

    for label, path in [("Baseline (production)", args.baseline), ("Combined candidate", args.candidate)]:
        print(f"\nLoading surrogate from: {path}")
        surrogate = ETASurrogate.load(path)
        data = run_condition3_horizon(experiment, model, args.episodes, surrogate)
        print_bucket_table(label, data)


if __name__ == "__main__":
    main()
