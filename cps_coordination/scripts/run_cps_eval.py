"""
cps_coordination/scripts/run_cps_eval.py
-------------------------------------------
CLI: builds the multi-agent env + CPSManager + frozen SAC worker, runs the
matched-seed three-pass evaluation (CPS pass + static-TTA pass + solo/
unconstrained pass, roadmap step 7 plus the static-TTA addition) per
episode, and streams per-aircraft + per-pair-separation rows to the
telemetry Parquet collectors (roadmap step 8, see ``telemetry.py``).

Usage
-----
  python cps_coordination/scripts/run_cps_eval.py \\
      --run-id 20260615_095840 --episodes 100 --n-aircraft 5 \\
      --k-cps 3 --mode static --save-path experiments/cps_eval/manual_run

  # All options:
  python cps_coordination/scripts/run_cps_eval.py \\
      --run-id <pretrained_run_id> \\
      --episodes 100 \\
      --n-aircraft 5 \\
      --k-cps 3 \\
      --mode static \\
      --delta-t-plan 120 \\
      --delta-update 1.0 \\
      --runways 18C 24 27 \\
      --eta-surrogate-path cps_coordination/models/eta_surrogate.pkl \\
      --save-path experiments/cps_eval/manual_run \\
      --chunk-size 25 \\
      --seed-base 0
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

from bluesky_gym.envs.pathplanning_goal_env import ALL_RUNWAYS
from bluesky_gym.experiment.config import ExperimentConfig, SessionConfig

from cps_coordination.coordination.cps_manager import CPSManager
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.config import CPSEnvConfig, CPSEnvKwargsConfig, CPSModelConfig
from cps_coordination.experiments.coordination_baseline import (
    CPSCoordinationExperiment,
    _EpisodeRecord,
)
from cps_coordination.testing.telemetry import (
    AircraftTelemetryRow,
    SeparationTelemetryRow,
    build_collectors,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the CPS coordination three-pass (CPS + static + solo) evaluation "
            "and log telemetry to Parquet for offline metric recomputation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id", type=str, required=True,
                   help="Pretrained worker run_id under "
                        "experiments/PathPlanningGoalEnv-v0/SAC/models/.")
    p.add_argument("--episodes", type=int, default=100, help="Number of episodes (M).")
    p.add_argument("--n-aircraft", type=int, default=5, help="Aircraft per episode (N_a).")
    p.add_argument("--k-cps", type=int, default=3)
    p.add_argument("--freeze-remaining-time-budget", action="store_true", default=False,
                   help="Test A (TTA feedback-loop falsification): freeze "
                        "remaining_time_budget at its first-committed value per acid.")
    p.add_argument("--remaining-time-budget-cap-s", type=float, default=None,
                   help="Test B: cap remaining_time_budget fed to the surrogate (seconds).")
    p.add_argument("--disable-cross-cycle-runway-seeding", action="store_true", default=False,
                   help="Test C: never seed a runway's greedy-schedule state from "
                        "prior replanning cycles (isolates the surrogate-feature loop).")
    p.add_argument("--mode", type=str, default="static", choices=["static", "dynamic"],
                   help="runway_assignment_mode")
    p.add_argument("--delta-t-plan", type=int, default=120)
    p.add_argument("--delta-update", type=float, default=1.0)
    p.add_argument("--runways", type=str, nargs="*", default=None,
                   help="Subset of runways to sample from (default: all 12).")
    p.add_argument("--eta-surrogate-path", type=str, default=None,
                   help="Path to a fitted ETASurrogate .pkl. Defaults to "
                        "cps_coordination/models/eta_surrogate.pkl if present.")
    p.add_argument("--save-path", type=str, default="experiments/cps_eval/manual_run",
                   help="Directory for cps_eval_aircraft.parquet / cps_eval_separation.parquet.")
    p.add_argument("--chunk-size", type=int, default=25,
                   help="Episodes buffered per Parquet flush.")
    p.add_argument("--seed-base", type=int, default=0,
                   help="Episode seed = seed_base + episode_index; matched between "
                        "the CPS and solo passes of the same episode, varied across "
                        "episodes.")
    p.add_argument("--no-fresh-start", action="store_true", default=False,
                   help="Append to existing Parquet files instead of overwriting.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Print progress every N episodes.")
    return p.parse_args()


def _log_episode(
    aircraft_collector,
    separation_collector,
    ep_idx: int,
    records: List[_EpisodeRecord],
    k_cps: int,
    mode: str,
    recat_matrix: Dict[str, Dict[str, float]],
) -> None:
    """Append one episode's joined two-pass records to both telemetry streams."""
    for rec in records:
        aircraft_collector.collect_step(
            **AircraftTelemetryRow(
                episode_id=ep_idx,
                acid=rec.acid,
                flight_id=rec.flight_id,
                runway_id=rec.runway_id,
                wake_cat=rec.wake_cat,
                k_cps=k_cps,
                runway_assignment_mode=mode,
                assigned_tta=rec.assigned_tta,
                actual_landing_time=rec.actual_landing_time,
                rta_error_cps=rec.rta_error_cps,
                rta_error_static=rec.rta_error_static,
                rta_error_solo=rec.rta_error_solo,
                tta_updated_mid_trajectory=rec.tta_updated_mid_trajectory,
                stall_detected=rec.stall_detected,
                success=rec.success,
                death_cause=rec.death_cause,
                traj_x=list(rec.traj_x),
                traj_y=list(rec.traj_y),
            ).as_dict()
        )
    # VerboseDataCollector stores this batch unconditionally and just tags
    # every row's `is_success` column with the value passed here — always
    # True, since we always want to keep every episode's rows (per-aircraft
    # `success` above is the real per-row outcome flag).
    aircraft_collector.finalise_episode(success=True)

    by_runway: Dict[str, List[_EpisodeRecord]] = {}
    for rec in records:
        if rec.success:
            by_runway.setdefault(rec.runway_id, []).append(rec)

    for rwy, recs in by_runway.items():
        recs.sort(key=lambda r: r.actual_landing_time)
        for prev, curr in zip(recs, recs[1:]):
            required = recat_matrix.get(prev.wake_cat, {}).get(curr.wake_cat, 90.0)
            separation_collector.collect_step(
                **SeparationTelemetryRow(
                    episode_id=ep_idx,
                    runway_id=rwy,
                    acid_lead=prev.acid,
                    acid_trail=curr.acid,
                    gap_actual_s=curr.actual_landing_time - prev.actual_landing_time,
                    required_sep_s=required,
                ).as_dict()
            )
    separation_collector.finalise_episode(success=True)


def main() -> None:
    args = _parse_args()

    cfg = ExperimentConfig(
        model=CPSModelConfig(
            k_cps=args.k_cps,
            runway_assignment_mode=args.mode,
            delta_t_plan=args.delta_t_plan,
            delta_update=args.delta_update,
            eta_surrogate_path=args.eta_surrogate_path,
            freeze_remaining_time_budget=args.freeze_remaining_time_budget,
            remaining_time_budget_cap_s=args.remaining_time_budget_cap_s,
            enable_cross_cycle_runway_seeding=not args.disable_cross_cycle_runway_seeding,
        ),
        session=SessionConfig(
            pretrained_run_id=args.run_id,
            eval_episodes=args.episodes,
            do_train=False,
        ),
        env=CPSEnvConfig(env_kwargs=CPSEnvKwargsConfig(runways=args.runways)),
    )
    experiment = CPSCoordinationExperiment(cfg)

    recat_matrix = experiment._load_recat_matrix()
    surrogate = experiment._build_surrogate()
    print(f"eta_surrogate: {'loaded' if surrogate else 'none (naive straight-line ETA)'}")

    env = experiment._make_multi_agent_env(args.n_aircraft)
    model = experiment.make_model(env)

    # `CPSManager.__init__` does `available_runways or []` -- passing a bare
    # `None`/empty `args.runways` straight through silently produces an
    # *empty* candidate list, which makes `_assign_runways_dynamic` no-op
    # unconditionally (`not self.available_runways` short-circuits it). Same
    # bug class already found + fixed once in `run_batch_eval.py`'s standalone
    # `_new_manager` -- resolved once here, the same way, so this script's
    # dynamic mode doesn't silently degrade to static behaviour by default.
    available_runways = list(args.runways or ALL_RUNWAYS)

    def _new_manager() -> CPSManager:
        return CPSManager(
            k_cps=args.k_cps,
            recat_matrix=recat_matrix,
            runway_assignment_mode=args.mode,
            delta_t_plan=args.delta_t_plan,
            delta_update=args.delta_update,
            available_runways=available_runways,
            # See coordination_baseline.py::evaluate()'s _new_cps_manager for why
            # this is required (unwired -> zeroed lag features -> degraded ETA).
            trajectory_buffer=TrajectoryBuffer(),
            enable_cross_cycle_runway_seeding=not args.disable_cross_cycle_runway_seeding,
        )

    cps_manager = _new_manager()
    static_manager = _new_manager()
    solo_manager = _new_manager()

    aircraft_collector, separation_collector = build_collectors(
        args.save_path, chunk_size=args.chunk_size, fresh_start=not args.no_fresh_start,
    )

    print(
        f"\nCPS three-pass evaluation -> {args.save_path}"
        f"\n  episodes={args.episodes}, aircraft/episode={args.n_aircraft}"
        f"\n  k_cps={args.k_cps}, mode={args.mode}\n"
    )

    try:
        for ep_idx in range(args.episodes):
            ep_seed = args.seed_base + ep_idx

            cps_records = experiment._run_episode(
                env=env, model=model, cps_manager=cps_manager, surrogate=surrogate,
                deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="cps",
                track_trajectory=True,
            )
            cps_manager.reset()

            static_records = experiment._run_episode(
                env=env, model=model, cps_manager=static_manager, surrogate=surrogate,
                deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="static",
                track_trajectory=False,
            )
            static_manager.reset()

            solo_records = experiment._run_episode(
                env=env, model=model, cps_manager=solo_manager, surrogate=surrogate,
                deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="solo",
                track_trajectory=False,
            )
            solo_manager.reset()

            ep_records = experiment._join_three_pass(cps_records, static_records, solo_records)
            _log_episode(
                aircraft_collector, separation_collector, ep_idx, ep_records,
                k_cps=args.k_cps, mode=args.mode, recat_matrix=recat_matrix,
            )

            if (ep_idx + 1) % args.log_every == 0 or ep_idx == args.episodes - 1:
                print(f"[{ep_idx + 1}/{args.episodes}] episodes logged")
    finally:
        aircraft_collector.close()
        separation_collector.close()
        env.close()

    print(f"\nDone. Telemetry written to {args.save_path}/"
          f"{{cps_eval_aircraft.parquet, cps_eval_separation.parquet}}")


if __name__ == "__main__":
    main()
