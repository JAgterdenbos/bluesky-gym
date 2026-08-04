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
import shutil
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


def _build_parser() -> argparse.ArgumentParser:
    """Split out from _parse_args() so callers that need a Namespace with
    every flag's real default (e.g. smoke_test_step10.py, which only wants
    to override a handful of fields) can do
    ``_build_parser().parse_args([...])`` instead of hand-building an
    argparse.Namespace -- the latter has silently broken twice in one
    session (missing --episode-id-offset, then missing --resume) every
    time a new flag was added here, since nothing forces a hand-built
    Namespace to stay in sync with this parser."""
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
    p.add_argument("--episode-id-offset", type=int, default=0,
                   help="Added to episode_index to form the logged episode_id. "
                        "Use this to shard one combo's M episodes across several "
                        "concurrent processes (each covering a disjoint --episodes "
                        "range via --seed-base, with a matching disjoint "
                        "--episode-id-offset) without colliding episode_ids when "
                        "the shards' Parquet output is later concatenated -- "
                        "c_sep and other metrics group by (runway_id, episode_id), "
                        "so colliding ids across shards would silently corrupt them.")
    p.add_argument("--no-fresh-start", action="store_true", default=False,
                   help="Append to existing Parquet files instead of overwriting.")
    p.add_argument("--resume", action="store_true", default=False,
                   help="Per combo, resume from the highest episode_id already durably "
                        "written to that combo's cps_eval_aircraft.parquet (accounting "
                        "for --episode-id-offset) instead of starting at episode 0. A "
                        "combo whose file already covers --episodes is skipped entirely. "
                        "Safe against both clean (SIGINT/SIGTERM) and hard-crash "
                        "interruption: the collector's close() force-flushes any "
                        "buffered episodes first, so the file never has a gap or a "
                        "duplicate -- worst case after a hard crash is re-running up to "
                        "--chunk-size episodes that were computed but not yet flushed, "
                        "never silent data loss/corruption. Overrides --no-fresh-start "
                        "per combo (always appends once a resume point > 0 is found).")
    p.add_argument("--log-every", type=int, default=100,
                   help="Print progress every N episodes.")
    return p


def _parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def _resolve_resume_start(save_path: str, episode_id_offset: int) -> int:
    """Returns the episode_index to resume this combo's loop at, derived
    from the highest episode_id already durably present in this combo's
    cps_eval_aircraft.parquet (0 if the file doesn't exist, is empty, or
    can't be read). Deliberately reads from the data itself rather than a
    separately-tracked checkpoint counter -- a checkpoint file could desync
    from what the collector actually flushed; the Parquet file's own
    contents can't."""
    path = os.path.join(save_path, "cps_eval_aircraft.parquet")
    if not os.path.exists(path):
        return 0
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=["episode_id"])
        if table.num_rows == 0:
            return 0
        max_episode_id = int(pc.max(table.column("episode_id")).as_py())
    except Exception as exc:
        print(f"[run_batch_eval] --resume: couldn't read {path} ({exc}) -- "
              f"treating as no prior progress, starting this combo at episode 0.")
        return 0
    return max(0, max_episode_id - episode_id_offset + 1)


def _merge_resume_delta(save_path: str, delta_path: str) -> None:
    """Merges a resumed combo's newly-written delta Parquet files into the
    pre-existing combo directory.

    Parquet has no true row-group append -- ParquetDataCollector._flush
    always opens a fresh pyarrow.parquet.ParquetWriter on the target path,
    which truncates it, regardless of the collector's own fresh_start flag
    (that flag only controls a pre-emptive unlink(), not the writer's own
    open mode). Writing a resumed combo's new episodes in place would
    therefore silently destroy the already-durable episodes _resolve_resume_
    start found -- confirmed by an end-to-end resume test before this fix
    existed. Instead, resumed episodes are written to a throwaway
    `{save_path}__resume_delta` directory (fresh_start=True there, since
    it's brand new) and combined here via read+concat+atomic-replace once
    the combo's episode loop finishes (however it finished -- this runs in
    a `finally`, so an interrupted resume still merges whatever was
    completed, and a subsequent --resume re-derives the next start point
    from the newly-merged file, same as any other run)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for filename in ("cps_eval_aircraft.parquet", "cps_eval_separation.parquet"):
        old_path = os.path.join(save_path, filename)
        new_path = os.path.join(delta_path, filename)
        if not os.path.exists(new_path):
            continue
        tables = [pq.read_table(old_path)] if os.path.exists(old_path) else []
        tables.append(pq.read_table(new_path))
        merged = pa.concat_tables(tables)
        tmp_path = old_path + ".merging"
        pq.write_table(merged, tmp_path)
        os.replace(tmp_path, old_path)

    shutil.rmtree(delta_path, ignore_errors=True)


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

            start_ep_idx = 0
            resuming = False
            if args.resume:
                start_ep_idx = _resolve_resume_start(save_path, args.episode_id_offset)
                if start_ep_idx >= args.episodes:
                    print(f"\n--- combo k_cps={k_cps}, mode={mode}, fairness_weight={fairness_weight:g} "
                          f"-> {save_path}: already complete ({start_ep_idx}/{args.episodes} episodes "
                          f"durably written), skipping ---")
                    continue
                if start_ep_idx > 0:
                    resuming = True
                    print(f"\n--- combo k_cps={k_cps}, mode={mode}, fairness_weight={fairness_weight:g} "
                          f"-> {save_path}: resuming at episode {start_ep_idx}/{args.episodes} "
                          f"({start_ep_idx} already durably written) ---")

            # Parquet has no true row-group append (see _merge_resume_delta's
            # docstring) -- a resumed combo writes to a throwaway delta dir
            # and gets merged with the pre-existing file once its loop ends,
            # rather than risking an in-place write clobbering already-
            # durable episodes.
            collector_path = f"{save_path}__resume_delta" if resuming else save_path
            collector_fresh_start = True if resuming else not args.no_fresh_start

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
                collector_path, chunk_size=args.chunk_size, fresh_start=collector_fresh_start,
            )

            print(f"\n--- combo k_cps={k_cps}, mode={mode}, fairness_weight={fairness_weight:g} -> {save_path} ---")
            try:
                for ep_idx in range(start_ep_idx, args.episodes):
                    if _stop_requested:
                        print(f"[run_batch_eval] stop requested mid-combo "
                              f"(k_cps={k_cps}, mode={mode}) at episode {ep_idx}/{args.episodes} - "
                              f"flushing and closing this combo's telemetry.")
                        break

                    ep_seed = args.seed_base + ep_idx
                    # Only the logged episode_id is offset -- the seed and the
                    # progress-print index stay tied to this process's local
                    # loop, so seed_base is still the sole seed-sharding knob.
                    episode_id = args.episode_id_offset + ep_idx

                    cps_records = experiment._run_episode(
                        env=env, model=model, cps_manager=cps_manager, surrogate=surrogate,
                        deterministic=True, ep_idx=episode_id, seed=ep_seed, tta_mode="cps",
                        track_trajectory=True,
                    )
                    cps_manager.reset()

                    static_records = experiment._run_episode(
                        env=env, model=model, cps_manager=static_manager, surrogate=surrogate,
                        deterministic=True, ep_idx=episode_id, seed=ep_seed, tta_mode="static",
                        track_trajectory=False,
                    )
                    static_manager.reset()

                    solo_records = experiment._run_episode(
                        env=env, model=model, cps_manager=solo_manager, surrogate=surrogate,
                        deterministic=True, ep_idx=episode_id, seed=ep_seed, tta_mode="solo",
                        track_trajectory=False,
                    )
                    solo_manager.reset()

                    ep_records = experiment._join_three_pass(cps_records, static_records, solo_records)
                    _log_episode(
                        aircraft_collector, separation_collector, episode_id, ep_records,
                        k_cps=k_cps, mode=mode, recat_matrix=recat_matrix,
                        fairness_weight=fairness_weight,
                    )

                    if (ep_idx + 1) % args.log_every == 0 or ep_idx == args.episodes - 1:
                        print(f"  [{ep_idx + 1}/{args.episodes}] episodes logged")
            finally:
                aircraft_collector.close()
                separation_collector.close()
                if resuming:
                    _merge_resume_delta(save_path, collector_path)
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
