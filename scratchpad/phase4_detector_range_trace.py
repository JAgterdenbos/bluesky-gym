"""
scratchpad/phase4_detector_range_trace.py
--------------------------------------------
Phase 4 design step of .claude/plans/stall_rate_investigation.md: collect one
rich per-cycle trace (t, x, y, runway_id, tta, eta) for every aircraft at
cap=50, then replay several CANDIDATE stall-detector formulations against it
offline and score each against ground-truth outcome (success / death_cause).

Collection: monkeypatches CPSManager._update_stall_tracking to log the raw
per-cycle state alongside the real (unmodified) detector, so the actually-
logged stall_detected/success/death_cause in this run's own _EpisodeRecords
serve as the "current baseline" arm for free, and every candidate below can
be replayed against literally the same trajectories for a fair comparison.

Run with: uv run python scratchpad/phase4_detector_range_trace.py
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bluesky_gym.experiment.config import ExperimentConfig, SessionConfig

from cps_coordination.coordination.cps_manager import CPSManager
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.config import CPSEnvConfig, CPSEnvKwargsConfig, CPSModelConfig
from cps_coordination.experiments.coordination_baseline import CPSCoordinationExperiment

RUN_ID = "20260615_095840"
ETA_SURROGATE_PATH = "cps_coordination/models/eta_surrogate.pkl"
CAP = 50
TOTAL_ARRIVALS_PER_EPISODE = 50
SPAWN_WINDOW_S = 2400.0
DELTA_T_PLAN = 120
DELTA_UPDATE = 1.0
K_CPS = 3
MODE = "dynamic"
RUNWAYS = ["18R", "27"]
EPISODES = 30
SEED_BASE = 1000
ACTION_TIME = 120.0
MAX_DISTANCE = 300.0

_current_episode_id = -1
rows: List[Tuple] = []  # (ep, acid, t, x, y, runway_id, tta, eta, spawn_time)


def _wrap_replan(original):
    """Hook _replan (not _update_stall_tracking) so tta reflects THIS cycle's
    freshly-committed value. _update_stall_tracking runs before _replan in
    update_fleet, AND the external caller (coordination_baseline._build_fleet)
    constructs a brand-new AircraftState every cycle (tta defaults to
    math.inf) before update_fleet even starts -- so a hook on
    _update_stall_tracking only ever sees inf/frozen-stalled tta, never a
    genuine live in-progress value (confirmed empirically: first version of
    this script logged tta as NaN 87% of the time, with the remaining 13%
    exactly equal to eta -- the stalled-aircraft freeze re-pin, not a real
    schedule). Hooking _replan's exit point instead captures the tta that
    _greedy_schedule just committed for this cycle, at the same current_time
    _update_stall_tracking used moments earlier in the same update_fleet call.
    """
    def wrapped(self: CPSManager, current_time: float):
        result = original(self, current_time)
        for ac in self._fleet:
            rows.append((
                _current_episode_id, ac.acid, current_time,
                float(ac.state[0]), float(ac.state[1]), ac.runway_id,
                float(ac.tta) if math.isfinite(ac.tta) else float("nan"),
                float(ac.eta), float(ac.spawn_time),
            ))
        return result
    return wrapped


def main() -> None:
    CPSManager._replan = _wrap_replan(CPSManager._replan)

    cfg = ExperimentConfig(
        model=CPSModelConfig(delta_t_plan=DELTA_T_PLAN, delta_update=DELTA_UPDATE, eta_surrogate_path=ETA_SURROGATE_PATH),
        session=SessionConfig(pretrained_run_id=RUN_ID, eval_episodes=EPISODES, do_train=False),
        env=CPSEnvConfig(env_kwargs=CPSEnvKwargsConfig(runways=RUNWAYS)),
    )
    experiment = CPSCoordinationExperiment(cfg)
    recat_matrix = experiment._load_recat_matrix()
    surrogate = experiment._build_surrogate()
    print(f"eta_surrogate: {'loaded' if surrogate else 'none'}")

    env = experiment._make_multi_agent_env(CAP, n_aircraft_total=TOTAL_ARRIVALS_PER_EPISODE, spawn_window_s=SPAWN_WINDOW_S)
    model = experiment.make_model(env)

    cps_manager = CPSManager(
        k_cps=K_CPS, recat_matrix=recat_matrix, runway_assignment_mode=MODE,
        delta_t_plan=DELTA_T_PLAN, delta_update=DELTA_UPDATE, available_runways=list(RUNWAYS),
        trajectory_buffer=TrajectoryBuffer(),
        enable_cross_cycle_runway_seeding=False,
    )

    global _current_episode_id
    all_records = []
    for ep_idx in range(EPISODES):
        _current_episode_id = ep_idx
        ep_seed = SEED_BASE + ep_idx
        cps_records = experiment._run_episode(
            env=env, model=model, cps_manager=cps_manager, surrogate=surrogate,
            deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="cps",
            track_trajectory=False,
        )
        cps_manager.reset()
        all_records.extend(cps_records)
        print(f"  [{ep_idx + 1}/{EPISODES}] episodes traced")

    trace_df = pd.DataFrame(rows, columns=["episode_id", "acid", "t", "x", "y", "runway_id", "tta", "eta", "spawn_time"])
    outcome_df = pd.DataFrame([
        {"episode_id": r.episode_id, "acid": r.acid, "success": r.success,
         "death_cause": r.death_cause, "stall_detected_baseline": r.stall_detected}
        for r in all_records
    ])

    out_dir = "experiments/cps_eval/stall_window_grounding_20260813"
    trace_df.to_parquet(f"{out_dir}/phase4_detector_trace_cap50.parquet")
    outcome_df.to_parquet(f"{out_dir}/phase4_detector_outcome_cap50.parquet")
    print(f"\nWrote {len(trace_df)} trace rows, {len(outcome_df)} outcome rows to {out_dir}/")


if __name__ == "__main__":
    main()
