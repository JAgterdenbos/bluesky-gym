"""
cps_coordination/testing/run_batch_eval.py
-------------------------------------------
Production batch runner for the M=10,000-episode CPS coordination
scale-up evaluation (Phase III roadmap Step 10). Wraps the same
CPSCoordinationExperiment / CPSManager / MultiAgentPathPlanningGoalEnv
machinery ``run_cps_eval.py`` already exercises for a single
(k_cps, mode, fairness_weight) run, adding:

  - a parameter sweep across ``k_cps``, ``runway_assignment_mode``, and
    ``fairness_weight`` (default: 0/1/3 x static/dynamic x [0.0], 6
    combinations), reusing one shared
    env/model/surrogate/recat-matrix across the whole sweep (only cheap
    objects -- CPSManager pairs and Parquet collectors -- are rebuilt per
    combination) since BlueSky init + frozen-SAC load are the expensive
    parts and this matters at 6 x M=10,000 scale;
  - SIGINT/SIGTERM handling that flushes and closes the in-flight combo's
    Parquet streams cleanly before exiting, rather than corrupting a
    partial chunk;
  - a rolling arrival stream (``total_arrivals_per_episode`` >
    ``max_concurrent_aircraft``) with time-windowed spawning
    (``spawn_window_s``), read from the ``cps_eval:`` section of the
    ``--config`` YAML (see ``cps_coordination/configs/cps_scale_10k.yaml``).

Usage
-----
  python cps_coordination/testing/run_batch_eval.py \\
      --run-id 20260301_120000 \\
      --config cps_coordination/configs/cps_scale_10k.yaml \\
      --save-path-root experiments/cps_eval/scale_10k

  # Capped local sanity check (see smoke_test_step10.py for the real one):
  python cps_coordination/testing/run_batch_eval.py \\
      --run-id 20260301_120000 --episodes 10 \\
      --k-cps-sweep 0 3 --mode-sweep static dynamic \\
      --fairness-weight-sweep 0.0 0.3 \\
      --save-path-root /tmp/cps_batch_smoke
"""

from __future__ import annotations

import argparse
import itertools
import os
import signal
from types import FrameType
from typing import Any, Dict, List, Optional

import yaml

from bluesky_gym.experiment.config import ExperimentConfig, SessionConfig
from bluesky_gym.envs.pathplanning_goal_env import ALL_RUNWAYS

from cps_coordination.coordination.cps_manager import CPSManager
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.config import CPSEnvConfig, CPSEnvKwargsConfig, CPSModelConfig
from cps_coordination.experiments.coordination_baseline import CPSCoordinationExperiment
from cps_coordination.testing.run_cps_eval import _log_episode
from cps_coordination.testing.telemetry import build_collectors

_DEFAULT_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "configs", "cps_scale_10k.yaml")
)

# Set by the SIGINT/SIGTERM handler; checked between episodes and between
# sweep combinations so an interrupted run still flushes/closes cleanly
# instead of corrupting an in-flight Parquet chunk.
_stop_requested = False


def _request_stop(signum: int, frame: Optional[FrameType]) -> None:
    global _stop_requested
    name = signal.Signals(signum).name
    print(f"\n[run_batch_eval] received {name} - finishing the current episode, "
          f"then flushing and closing telemetry before exit...")
    _stop_requested = True


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _parse_args() -> argparse.Namespace:
    defaults = _load_yaml(_DEFAULT_CONFIG)
    model_d = defaults.get("model", {})
    cps_eval_d = defaults.get("cps_eval", {})
    logging_d = defaults.get("logging", {})
    session_d = defaults.get("session", {})

    p = argparse.ArgumentParser(
        description=(
            "Batch CPS coordination evaluation: sweep k_cps x "
            "runway_assignment_mode x fairness_weight, three-pass (CPS + static + solo) "
            "per episode, streaming telemetry to Parquet. Production driver "
            "for the M=10,000 scale-up (Phase III roadmap Step 10)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id", type=str, required=True,
                   help="Pretrained worker run_id under "
                        "experiments/PathPlanningGoalEnv-v0/SAC/models/.")
    p.add_argument("--config", type=str, default=_DEFAULT_CONFIG,
                   help="YAML config providing defaults for the flags below "
                        "(cps_scale_10k.yaml's model:/cps_eval:/logging: sections).")
    p.add_argument("--episodes", type=int, default=session_d.get("eval_episodes", 10000),
                   help="Episodes (M) per (k_cps, mode) combination.")
    p.add_argument("--k-cps-sweep", type=int, nargs="+", default=[0, 1, 3],
                   help="k_cps values to sweep.")
    p.add_argument("--mode-sweep", type=str, nargs="+", default=["static", "dynamic"],
                   choices=["static", "dynamic"], help="runway_assignment_mode values to sweep.")
    p.add_argument("--fairness-weight-sweep", type=float, nargs="+",
                   default=model_d.get("fairness_weight_sweep", [0.0]),
                   help="k-CPS slack-protection weight values to sweep (0.0 = FCFS, ablation).")
    p.add_argument("--disable-cross-cycle-runway-seeding", action="store_true", default=False,
                   help="Test C: never seed a runway's greedy-schedule state from "
                        "prior replanning cycles (isolates the surrogate-feature loop).")
    p.add_argument("--max-concurrent-aircraft", type=int,
                   default=cps_eval_d.get("max_concurrent_aircraft", 5))
    p.add_argument("--total-arrivals-per-episode", type=int,
                   default=cps_eval_d.get("total_arrivals_per_episode", 10))
    p.add_argument("--spawn-window-s", type=float,
                   default=cps_eval_d.get("spawn_window_s", 0.0))
    p.add_argument("--delta-t-plan", type=int, default=model_d.get("delta_t_plan", 120))
    p.add_argument("--delta-update", type=float, default=model_d.get("delta_update", 1.0))
    p.add_argument("--runways", type=str, nargs="*", default=None,
                   help="Subset of runways to sample from (default: all 12).")
    p.add_argument("--eta-surrogate-path", type=str,
                   default=model_d.get("eta_surrogate_path"),
                   help="Path to a fitted ETASurrogate .pkl. Defaults to "
                        "cps_coordination/models/eta_surrogate.pkl if present.")
    p.add_argument("--save-path-root", type=str, default="experiments/cps_eval/scale_10k",
                   help="Root directory; each (k_cps, mode) combo writes to "
                        "{save-path-root}/k{k_cps}_{mode}/.")
    p.add_argument("--chunk-size", type=int, default=logging_d.get("chunk_size", 250),
                   help="Episodes buffered per Parquet flush (per combo, per stream).")
    p.add_argument("--seed-base", type=int, default=0,
                   help="Episode seed = seed_base + episode_index; matched between "
                        "the CPS, static, and solo passes of the same episode.")
    p.add_argument("--no-fresh-start", action="store_true", default=False,
                   help="Append to existing Parquet files instead of overwriting.")
    p.add_argument("--log-every", type=int, default=100,
                   help="Print progress every N episodes.")
    return p.parse_args()


def _new_manager(
    k_cps: int,
    mode: str,
    recat_matrix: Dict[str, Dict[str, float]],
    delta_t_plan: int,
    delta_update: float,
    available_runways: Optional[List[str]],
    fairness_weight: float,
    enable_cross_cycle_runway_seeding: bool,
) -> CPSManager:
    return CPSManager(
        k_cps=k_cps,
        recat_matrix=recat_matrix,
        runway_assignment_mode=mode,
        delta_t_plan=delta_t_plan,
        delta_update=delta_update,
        available_runways=available_runways,
        # See coordination_baseline.py::evaluate()'s _new_cps_manager for why
        # this is required (unwired -> zeroed lag features -> degraded ETA).
        trajectory_buffer=TrajectoryBuffer(),
        fairness_weight=fairness_weight,
        enable_cross_cycle_runway_seeding=enable_cross_cycle_runway_seeding,
    )


def run_sweep(args: argparse.Namespace) -> None:
    """Run the full (k_cps x mode) sweep. Split out from ``main()`` so
    ``smoke_test_step10.py`` can call it directly with capped params
    instead of going through argparse."""
    cfg = ExperimentConfig(
        model=CPSModelConfig(
            delta_t_plan=args.delta_t_plan,
            delta_update=args.delta_update,
            eta_surrogate_path=args.eta_surrogate_path,
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

    # One shared env/model for the entire sweep -- BlueSky init and the
    # frozen-SAC load are the expensive parts; only CPSManager pairs and
    # Parquet collectors are rebuilt per (k_cps, mode) combination below.
    env = experiment._make_multi_agent_env(
        args.max_concurrent_aircraft,
        n_aircraft_total=args.total_arrivals_per_episode,
        spawn_window_s=args.spawn_window_s,
    )
    model = experiment.make_model(env)

    # CPSManager.__init__ does `available_runways or []` -- passing the raw
    # `args.runways` (None by default, since --runways is opt-in) silently
    # produces an *empty* candidate list, which makes _assign_runways_dynamic
    # no-op unconditionally (`not self.available_runways` short-circuits it)
    # for every dynamic-mode combo that doesn't explicitly pass --runways.
    # coordination_baseline.py's own _new_cps_manager already falls back to
    # ALL_RUNWAYS; this script's standalone _new_manager needs the same
    # fallback, resolved once here rather than inside _new_manager so every
    # combo (and both cps_manager/solo_manager) gets the identical resolved
    # list.
    available_runways = list(args.runways or ALL_RUNWAYS)

    enable_cross_cycle_runway_seeding = not args.disable_cross_cycle_runway_seeding
    combos = list(itertools.product(args.k_cps_sweep, args.mode_sweep, args.fairness_weight_sweep))
    print(
        f"\nCPS batch evaluation -> {args.save_path_root}"
        f"\n  combos={combos}"
        f"\n  enable_cross_cycle_runway_seeding={enable_cross_cycle_runway_seeding}"
        f"\n  episodes/combo={args.episodes}, max_concurrent_aircraft={args.max_concurrent_aircraft}, "
        f"total_arrivals_per_episode={args.total_arrivals_per_episode}, spawn_window_s={args.spawn_window_s}\n"
    )

    try:
        for k_cps, mode, fairness_weight in combos:
            if _stop_requested:
                print("[run_batch_eval] stop requested before starting next combo - exiting sweep.")
                break

            save_path = os.path.join(args.save_path_root, f"k{k_cps}_{mode}_fw{fairness_weight:g}")
            cps_manager = _new_manager(k_cps, mode, recat_matrix, args.delta_t_plan,
                                        args.delta_update, available_runways,
                                        fairness_weight, enable_cross_cycle_runway_seeding)
            static_manager = _new_manager(k_cps, mode, recat_matrix, args.delta_t_plan,
                                           args.delta_update, available_runways,
                                           fairness_weight, enable_cross_cycle_runway_seeding)
            solo_manager = _new_manager(k_cps, mode, recat_matrix, args.delta_t_plan,
                                         args.delta_update, available_runways,
                                         fairness_weight, enable_cross_cycle_runway_seeding)
            aircraft_collector, separation_collector = build_collectors(
                save_path, chunk_size=args.chunk_size, fresh_start=not args.no_fresh_start,
            )

            print(f"\n--- combo k_cps={k_cps}, mode={mode}, fairness_weight={fairness_weight:g} -> {save_path} ---")
            try:
                for ep_idx in range(args.episodes):
                    if _stop_requested:
                        print(f"[run_batch_eval] stop requested mid-combo "
                              f"(k_cps={k_cps}, mode={mode}) at episode {ep_idx}/{args.episodes} - "
                              f"flushing and closing this combo's telemetry.")
                        break

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
                        k_cps=k_cps, mode=mode, recat_matrix=recat_matrix,
                        fairness_weight=fairness_weight,
                    )

                    if (ep_idx + 1) % args.log_every == 0 or ep_idx == args.episodes - 1:
                        print(f"  [{ep_idx + 1}/{args.episodes}] episodes logged")
            finally:
                aircraft_collector.close()
                separation_collector.close()
                print(f"--- combo k_cps={k_cps}, mode={mode}, fairness_weight={fairness_weight:g} done -> {save_path} ---")
    finally:
        env.close()

    print(f"\nDone. Telemetry written under {args.save_path_root}/k<k_cps>_<mode>_fw<fairness_weight>/")


def main() -> None:
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    args = _parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
