"""
scratchpad/phase2_slack_and_churn_trace.py
--------------------------------------------
Phase 2 of .claude/plans/stall_rate_investigation.md: instrumented single run at
cap=50 (the worst case) tracing (a) the k-CPS slack_penalty term's saturation
and (b) dynamic-reassignment churn, both joined to that same run's own
stall_detected/success outcome -- to test hypotheses 2 and 3 (does slack_penalty
lose differentiating power under congestion? does residual post-fix churn still
correlate with stalling?).

Non-invasive: monkeypatches CPSManager._apply_k_cps_constraint and
CPSManager._assign_runways_dynamic to log alongside the real computation,
without editing the tracked cps_manager.py source -- mirrors the parent
investigation's established trace_reassignments.py precedent, so no
validate_cps_pipeline.py re-run is required for this step.

Join key note: uses (episode_id, acid) only, NOT (episode_id, acid, spawn_time)
-- safe ONLY because this script is hardcoded to cap=50 with
total_arrivals_per_episode=50 (cap == total arrivals means no slot in a
rolling-arrival-stream episode is ever refilled with a second physical
aircraft, so acid is already unique within an episode here). Do not reuse this
join key at a lower cap without adding spawn_time disambiguation (see
AircraftState.spawn_time's docstring in cps_manager.py).

Usage
-----
  uv run python scratchpad/phase2_slack_and_churn_trace.py
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bluesky_gym.envs.pathplanning_goal_env import ALL_RUNWAYS
from bluesky_gym.experiment.config import ExperimentConfig, SessionConfig

from cps_coordination.coordination.cps_manager import AircraftState, CPSManager
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.config import CPSEnvConfig, CPSEnvKwargsConfig, CPSModelConfig
from cps_coordination.experiments.coordination_baseline import CPSCoordinationExperiment

RUN_ID = "20260615_095840"
ETA_SURROGATE_PATH = "cps_coordination/models/eta_surrogate.pkl"  # corrected model (Step 0)
CAP = 50
TOTAL_ARRIVALS_PER_EPISODE = 50  # == CAP: guarantees no acid reuse within an episode
SPAWN_WINDOW_S = 2400.0
DELTA_T_PLAN = 120
DELTA_UPDATE = 1.0
K_CPS = 3
MODE = "dynamic"
FAIRNESS_WEIGHT = 0.5
RUNWAYS = ["18R", "27"]
EPISODES = 30
SEED_BASE = 1000

# --- instrumentation state (module-level, tagged with the current episode) ---
_current_episode_id = -1
slack_rows: List[Tuple[int, str, float, float, float, bool]] = []  # (ep, acid, t, margin, penalty, is_stalled)
switch_counts: Dict[Tuple[int, str], int] = defaultdict(int)


def _wrap_apply_k_cps_constraint(original):
    def wrapped(self: CPSManager, fcfs_order: List[AircraftState], current_time: float = 0.0):
        if self.k_cps != 0 and self.fairness_weight > 0.0:
            # Mirrors _apply_k_cps_constraint's own inlined slack_penalty
            # formula exactly (see its "Matches _slack_penalty exactly" note)
            # -- the standalone _slack_penalty method is NOT on this call
            # path at runtime, so hooking it alone would log nothing.
            for ac in fcfs_order:
                margin = ac.eta - current_time
                penalty = max(0.0, self.SLACK_REFERENCE_S - margin)
                is_stalled = ac.acid in self._stalled_acids
                if is_stalled:
                    penalty += self.STALL_SLACK_PENALTY_BOOST_S
                slack_rows.append((_current_episode_id, ac.acid, current_time, margin, penalty, is_stalled))
        return original(self, fcfs_order, current_time)
    return wrapped


def _wrap_assign_runways_dynamic(original):
    def wrapped(self: CPSManager, surrogate, current_time, lag_features):
        before = {ac.acid: ac.runway_id for ac in self._fleet}
        result = original(self, surrogate, current_time, lag_features)
        for ac in self._fleet:
            prev = before.get(ac.acid)
            if prev is not None and prev != ac.runway_id:
                switch_counts[(_current_episode_id, ac.acid)] += 1
        return result
    return wrapped


def main() -> None:
    CPSManager._apply_k_cps_constraint = _wrap_apply_k_cps_constraint(CPSManager._apply_k_cps_constraint)
    CPSManager._assign_runways_dynamic = _wrap_assign_runways_dynamic(CPSManager._assign_runways_dynamic)

    cfg = ExperimentConfig(
        model=CPSModelConfig(
            delta_t_plan=DELTA_T_PLAN,
            delta_update=DELTA_UPDATE,
            eta_surrogate_path=ETA_SURROGATE_PATH,
        ),
        session=SessionConfig(
            pretrained_run_id=RUN_ID,
            eval_episodes=EPISODES,
            do_train=False,
        ),
        env=CPSEnvConfig(env_kwargs=CPSEnvKwargsConfig(runways=RUNWAYS)),
    )
    experiment = CPSCoordinationExperiment(cfg)
    recat_matrix = experiment._load_recat_matrix()
    surrogate = experiment._build_surrogate()
    print(f"eta_surrogate: {'loaded' if surrogate else 'none (naive straight-line ETA)'}")

    env = experiment._make_multi_agent_env(
        CAP, n_aircraft_total=TOTAL_ARRIVALS_PER_EPISODE, spawn_window_s=SPAWN_WINDOW_S,
    )
    model = experiment.make_model(env)

    available_runways = list(RUNWAYS or ALL_RUNWAYS)
    cps_manager = CPSManager(
        k_cps=K_CPS,
        recat_matrix=recat_matrix,
        runway_assignment_mode=MODE,
        delta_t_plan=DELTA_T_PLAN,
        delta_update=DELTA_UPDATE,
        available_runways=available_runways,
        trajectory_buffer=TrajectoryBuffer(),
        fairness_weight=FAIRNESS_WEIGHT,
        enable_cross_cycle_runway_seeding=False,  # matches --disable-cross-cycle-runway-seeding
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

    # --- Build per-(episode_id, acid) aggregate table ---
    slack_df = pd.DataFrame(slack_rows, columns=["episode_id", "acid", "t", "margin", "penalty", "is_stalled"])
    slack_agg = slack_df.groupby(["episode_id", "acid"]).agg(
        n_cycles=("margin", "size"),
        mean_margin=("margin", "mean"),
        frac_margin_saturated=("margin", lambda s: float((s < CPSManager.SLACK_REFERENCE_S).mean())),
        mean_penalty=("penalty", "mean"),
    ).reset_index()

    switch_df = pd.DataFrame(
        [(ep, acid, n) for (ep, acid), n in switch_counts.items()],
        columns=["episode_id", "acid", "n_switches"],
    )

    outcome_df = pd.DataFrame([
        {"episode_id": rec.episode_id, "acid": rec.acid, "stall_detected": rec.stall_detected, "success": rec.success}
        for rec in all_records
    ])

    joined = outcome_df.merge(slack_agg, on=["episode_id", "acid"], how="left") \
                        .merge(switch_df, on=["episode_id", "acid"], how="left")
    joined["n_switches"] = joined["n_switches"].fillna(0).astype(int)

    out_path = "experiments/cps_eval/capacity_sweep_50ac_surrogate_fix/phase2_slack_churn_trace_cap50.parquet"
    joined.to_parquet(out_path)
    print(f"\nWrote {len(joined)} rows to {out_path}")

    # --- Headline comparison: stalled vs non-stalled ---
    for label, mask in [("stalled", joined["stall_detected"]), ("non-stalled", ~joined["stall_detected"])]:
        sub = joined[mask]
        print(
            f"{label:>12} n={len(sub):4d}  "
            f"mean_margin={sub['mean_margin'].mean():8.1f}s  "
            f"frac_margin_saturated={sub['frac_margin_saturated'].mean():.3f}  "
            f"mean_penalty={sub['mean_penalty'].mean():8.1f}  "
            f"mean_switches={sub['n_switches'].mean():.2f}"
        )

    overall_frac_saturated = joined["frac_margin_saturated"].mean()
    print(f"\nfleet-wide mean frac_margin_saturated (margin < SLACK_REFERENCE_S={CPSManager.SLACK_REFERENCE_S}s): "
          f"{overall_frac_saturated:.3f}")


if __name__ == "__main__":
    main()
