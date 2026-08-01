"""
cps_coordination/testing/diagnose_success_rate.py
---------------------------------------------------
Empirical diagnosis for the single-agent (>95%) vs. multi-agent (<2%)
``is_success`` gap reported against ``MultiAgentPathPlanningGoalEnv``.

Neither existing regression gate (``validate_multiagent_env.py``,
``validate_cps_pipeline.py``) prints a success-rate percentage — they check
specific mechanics (index bookkeeping, "does TTA injection change behavior",
RECAT separation, surrogate-vs-naive divergence). This script closes that gap
by running the same frozen worker through four conditions that progressively
add back the pieces the CPS pipeline introduces over the single-agent env,
isolating which one (if any) drives the success-rate collapse:

  1. Single-agent ``PathPlanningGoalEnv-v0`` baseline (no CPS layer at all).
  2. Multi-agent env, ``tta_mode="solo"``, naive straight-line ETA, N=1.
     Closest multi-agent analog to (1) — isolates the env/goal-injection
     mechanics themselves from any ETA-prediction or scheduling error.
  3. Multi-agent env, ``tta_mode="solo"``, real ``ETASurrogate``, N=1.
     Isolates the surrogate's prediction accuracy (bypasses k-CPS scheduling
     entirely — ``"solo"`` injects ``ac.eta`` directly).
  4. Multi-agent env, ``tta_mode="cps"``, real ``ETASurrogate``, k_cps=0,
     N=5 on a single runway (forces real RECAT-EU separation pressure).
     Isolates whether ``CPSManager``'s greedy forward scheduler is pushing
     TTAs beyond what the frozen worker can absorb.

For condition 4's first episode, also prints a per-decision-step trace
(``desired_goal[...,2]`` denormalized to seconds, ``sim_time``, ``spawn_time``,
``death_cause``) so the failure mode is visible directly, not just in
aggregate.

Run: python cps_coordination/testing/diagnose_success_rate.py [--episodes 30]
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import Counter
from typing import List, Optional

import numpy as np
import gymnasium as gym

import bluesky_gym
from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME, MAX_TIME
from bluesky_gym.experiment.config import ExperimentConfig, SessionConfig

from cps_coordination.coordination.cps_manager import CPSManager
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.config import CPSEnvConfig, CPSEnvKwargsConfig, CPSModelConfig
from cps_coordination.experiments.coordination_baseline import (
    CPSCoordinationExperiment,
    _EpisodeRecord,
)

SEED_BASE = 1000


def _find_pretrained_run_id() -> str:
    """Same glob pattern as validate_cps_pipeline.py::_find_pretrained_run_id."""
    candidates = sorted(
        glob.glob("experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip")
    )
    if not candidates:
        raise RuntimeError(
            "No frozen SAC model found under experiments/PathPlanningGoalEnv-v0/SAC/models/"
        )
    return os.path.basename(os.path.dirname(candidates[-1]))


def _make_experiment(
    k_cps: int,
    mode: str,
    runways: Optional[List[str]],
    reduced_wake_separation: bool = False,
    wake_separation_scale: float = 0.5,
) -> CPSCoordinationExperiment:
    run_id = _find_pretrained_run_id()
    cfg = ExperimentConfig(
        model=CPSModelConfig(
            k_cps=k_cps,
            runway_assignment_mode=mode,
            reduced_wake_separation=reduced_wake_separation,
            wake_separation_scale=wake_separation_scale,
        ),
        session=SessionConfig(pretrained_run_id=run_id, eval_episodes=1, do_train=False),
        env=CPSEnvConfig(env_kwargs=CPSEnvKwargsConfig(runways=runways)),
    )
    return CPSCoordinationExperiment(cfg)


def _summarize(label: str, records: List[_EpisodeRecord]) -> None:
    n = len(records)
    print(f"\n=== {label} ===")
    if n == 0:
        print("  no records")
        return
    n_success = sum(1 for r in records if r.success)
    death_causes = Counter(r.death_cause for r in records)
    rta_errs = [abs(r.rta_error_cps) for r in records if not math.isnan(r.rta_error_cps)]

    print(f"  n_aircraft   : {n}")
    print(f"  success_rate : {n_success}/{n} = {n_success / n:.2%}")
    print(f"  death_cause  : {dict(death_causes)}")
    if rta_errs:
        arr = np.array(rta_errs)
        print(
            f"  |rta_error|s : mean={arr.mean():.1f} median={np.median(arr):.1f} "
            f"p90={np.percentile(arr, 90):.1f} n={len(arr)}"
        )
    else:
        print("  |rta_error|s : n/a")


def _load_matching_rta_sampler(pretrained_model_path: str):
    """Load the same GeoRunwaySampler the frozen worker was trained/officially
    evaluated with, from the sibling config.yaml's env.env_kwargs.rta_sampler_path.

    Without this, ``gym.make("PathPlanningGoalEnv-v0")`` defaults to
    ``rta_sampler=None`` (``use_rta=False``) -- a genuinely different, RTA-less
    task the worker was never trained/eval'd against, which is not a fair
    single-agent baseline to diff the multi-agent conditions against.
    """
    import yaml
    from path_planning.rta.sampling import GeoRunwaySampler

    config_path = os.path.join(os.path.dirname(pretrained_model_path), "config.yaml")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    sampler_path = cfg.get("env", {}).get("env_kwargs", {}).get("rta_sampler_path")
    if not sampler_path:
        print(f"WARNING: no rta_sampler_path found in {config_path} -- "
              "condition 1 will run without RTA (not comparable to the official eval).")
        return None
    return GeoRunwaySampler.load(sampler_path)


def run_single_agent_baseline(model, n_episodes: int, pretrained_model_path: str) -> List[dict]:
    """Condition 1: plain PathPlanningGoalEnv-v0, same frozen model, same
    RTA sampler the worker was actually trained/officially evaluated with."""
    bluesky_gym.register_envs()
    rta_sampler = _load_matching_rta_sampler(pretrained_model_path)
    env = gym.make("PathPlanningGoalEnv-v0", rta_sampler=rta_sampler).unwrapped
    records = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=SEED_BASE + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        records.append({
            "success": bool(info.get("is_success", False)),
            "death_cause": info.get("death_cause"),
        })
    env.close()
    return records


def _print_single_agent_summary(records: List[dict]) -> None:
    n = len(records)
    n_success = sum(1 for r in records if r["success"])
    death_causes = Counter(r["death_cause"] for r in records)
    print("\n=== Condition 1: single-agent PathPlanningGoalEnv-v0 baseline ===")
    print(f"  n_episodes   : {n}")
    print(f"  success_rate : {n_success}/{n} = {n_success / n:.2%}")
    print(f"  death_cause  : {dict(death_causes)}")


def _run_episode_verbose(
    experiment: CPSCoordinationExperiment,
    env,
    model,
    cps_manager: CPSManager,
    surrogate,
    seed: int,
    tta_mode: str,
) -> List[_EpisodeRecord]:
    """Diagnostic-only copy of CPSCoordinationExperiment._run_episode's core
    loop with per-decision-step trace printing. Does not modify production
    code; mirrors coordination_baseline.py:654-744 exactly, minus two-pass/
    trajectory bookkeeping this script doesn't need.
    """
    obs, info_list = env.reset(seed=seed)
    sim_time = 0.0
    records: dict = {}
    last_tta: dict = {}
    arrival_order: dict = {}
    step_i = 0

    print(f"\n  --- verbose trace (tta_mode={tta_mode!r}) ---")
    while not env.is_episode_done():
        for info in info_list:
            acid = info["acid"]
            if acid not in arrival_order:
                arrival_order[acid] = len(arrival_order)

        fleet = experiment._build_fleet(obs, info_list, sim_time)
        acid_to_slot = {info["acid"]: info["slot"] for info in info_list}

        changed = cps_manager.update_fleet(
            aircraft=fleet, current_time=sim_time, surrogate=surrogate
        )
        for ac in fleet:
            slot = acid_to_slot[ac.acid]
            if ac.runway_id != env.current_runway[slot]:
                env.set_runway(slot, ac.runway_id)

        if tta_mode == "cps":
            for acid in changed:
                tta = cps_manager.get_tta(acid)
                if tta is not None:
                    last_tta[acid] = tta
                    env.set_tta(acid_to_slot[acid], tta)
        else:
            for ac in fleet:
                last_tta[ac.acid] = ac.eta
                env.set_tta(acid_to_slot[ac.acid], ac.eta)

        obs, info_list = env.get_active_batch()

        for info in info_list:
            slot = info["slot"]
            desired_t_s = float(env.goal_vector[slot][2]) * MAX_TIME
            print(
                f"    step={step_i:>3} slot={slot} acid={info['acid']:<8} "
                f"desired_t_s={desired_t_s:>9.1f} sim_time={info['sim_time']:>8.1f} "
                f"spawn_time={info['spawn_time']:>8.1f} death_cause={info['death_cause']}"
            )

        actions, _ = model.predict(obs, deterministic=True)
        _obs_t, _rew, terminated, truncated, info_terminal = env.step(actions)

        for row, info in enumerate(info_terminal):
            if terminated[row] or truncated[row]:
                acid = info["acid"]
                spawn_time = float(info.get("spawn_time", 0.0))
                landing_time = spawn_time + float(info.get("sim_time", sim_time))
                assigned_tta = last_tta.get(acid, float("nan"))
                rta_error_cps = (
                    abs(landing_time - assigned_tta)
                    if not math.isnan(assigned_tta) else float("nan")
                )
                print(
                    f"    >>> TERMINAL acid={acid} death_cause={info.get('death_cause')} "
                    f"is_success={info.get('is_success')} landing_time={landing_time:.1f} "
                    f"assigned_tta={assigned_tta:.1f} rta_error={rta_error_cps:.1f}"
                )
                records[acid] = _EpisodeRecord(
                    acid=acid,
                    episode_id=0,  # _run_episode_verbose is only ever invoked for ep==0
                    runway_id=str(info.get("current_runway", "")),
                    wake_cat="C",
                    assigned_tta=assigned_tta,
                    actual_landing_time=landing_time,
                    rta_error_cps=rta_error_cps,
                    rta_error_solo=float("nan"),
                    tta_updated_mid_trajectory=False,
                    success=bool(info.get("is_success", False)),
                    arrival_index=arrival_order[acid],
                    death_cause=info.get("death_cause"),
                )
        sim_time += ACTION_TIME
        obs, info_list = env.get_active_batch()
        step_i += 1

    return list(records.values())


def run_multi_agent_condition(
    experiment: CPSCoordinationExperiment,
    model,
    n_episodes: int,
    n_aircraft: int,
    tta_mode: str,
    surrogate,
    k_cps: int,
    trace_first_episode: bool = False,
) -> List[_EpisodeRecord]:
    env = experiment._make_multi_agent_env(n_aircraft)
    recat_matrix = experiment._load_recat_matrix()
    runways = experiment.cfg.env.env_kwargs.runways
    mode = experiment.cfg.model.runway_assignment_mode

    all_records: List[_EpisodeRecord] = []
    for ep in range(n_episodes):
        cps_manager = CPSManager(
            k_cps=k_cps,
            recat_matrix=recat_matrix,
            runway_assignment_mode=mode,
            delta_t_plan=120,
            delta_update=1.0,
            available_runways=runways,
            trajectory_buffer=TrajectoryBuffer(),
        )
        seed = SEED_BASE + ep
        if trace_first_episode and ep == 0:
            records = _run_episode_verbose(
                experiment, env, model, cps_manager, surrogate, seed, tta_mode
            )
        else:
            records = experiment._run_episode(
                env=env, model=model, cps_manager=cps_manager, surrogate=surrogate,
                deterministic=True, ep_idx=ep, seed=seed, tta_mode=tta_mode,
            )
        all_records.extend(records)
    env.close()
    return all_records


def run_wake_separation_sweep(
    model, surrogate, episodes_per_point: int, scales: List[float]
) -> None:
    """Sweep CPSModelConfig.wake_separation_scale across *scales* (1.0 = full,
    unscaled RECAT-EU minima) at the same adversarial N=5/single-runway/k_cps=0
    setup as diagnostic condition 4, and print a summary table.

    Every point is built with reduced_wake_separation=True and its own scale
    (a scale of 1.0 is mathematically identical to reduced_wake_separation=False
    -- see CPSCoordinationExperiment._load_recat_matrix -- so this sweep can
    include the unscaled baseline as just another point on the same axis).
    """
    rows = []
    for scale in scales:
        experiment = _make_experiment(
            k_cps=0, mode="static", runways=["27"],
            reduced_wake_separation=True, wake_separation_scale=scale,
        )
        records = run_multi_agent_condition(
            experiment, model, episodes_per_point, n_aircraft=5,
            tta_mode="cps", surrogate=surrogate, k_cps=0,
        )
        n = len(records)
        n_success = sum(1 for r in records if r.success)
        death_causes = Counter(r.death_cause for r in records)
        rta_errs = [abs(r.rta_error_cps) for r in records if not math.isnan(r.rta_error_cps)]
        arr = np.array(rta_errs) if rta_errs else np.array([np.nan])
        rows.append({
            "scale": scale,
            "n": n,
            "success_rate": n_success / n if n else float("nan"),
            "mean_err": arr.mean(),
            "median_err": np.median(arr),
            "death_cause": dict(death_causes),
        })
        print(f"  scale={scale:<5} success={n_success}/{n} "
              f"mean_err={arr.mean():.1f}s median_err={np.median(arr):.1f}s "
              f"death_cause={dict(death_causes)}")

    print("\n=== Wake-separation-scale sweep summary "
          "(k_cps=0, single runway, N=5, real surrogate) ===")
    print(f"{'scale':>7}{'success_rate':>14}{'mean_err_s':>12}{'median_err_s':>14}")
    for row in rows:
        print(f"{row['scale']:>7}{row['success_rate']:>14.2%}"
              f"{row['mean_err']:>12.1f}{row['median_err']:>14.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=30,
                    help="Episodes for conditions 1-3; conditions 4-5 use episodes//3 "
                         "(5 aircraft/episode, more expensive per episode).")
    p.add_argument("--wake-separation-scale", type=float, default=0.5,
                    help="Condition 5's RECAT-EU separation multiplier "
                         "(CPSModelConfig.wake_separation_scale).")
    p.add_argument("--sweep-scales", type=float, nargs="+", default=None,
                    help="If given, run ONLY a wake_separation_scale sweep at these "
                         "values (e.g. --sweep-scales 1.0 0.75 0.5 0.25 0.1) instead "
                         "of conditions 1-5.")
    p.add_argument("--sweep-episodes", type=int, default=10,
                    help="Episodes per sweep point (N=5 aircraft/episode).")
    args = p.parse_args()

    experiment_default = _make_experiment(k_cps=0, mode="static", runways=None)
    model = experiment_default.make_model(experiment_default._make_multi_agent_env(1))
    print(f"Frozen worker: {experiment_default.cfg.session.pretrained_model_path}")

    if args.sweep_scales is not None:
        surrogate = experiment_default._build_surrogate()
        if surrogate is None:
            print("WARNING: no eta_surrogate.pkl found -- cannot run the sweep "
                  "(it requires a real surrogate).")
            return
        print(f"Sweep: scales={args.sweep_scales}, episodes/point={args.sweep_episodes}\n")
        run_wake_separation_sweep(model, surrogate, args.sweep_episodes, args.sweep_scales)
        return

    print(f"Episodes per condition: {args.episodes} (condition 4: {max(1, args.episodes // 3)})")

    # --- Condition 1: single-agent baseline ---
    single_records = run_single_agent_baseline(
        model, args.episodes, experiment_default.cfg.session.pretrained_model_path
    )
    _print_single_agent_summary(single_records)

    # --- Condition 2: multi-agent, solo TTA, naive ETA, N=1 ---
    records_2 = run_multi_agent_condition(
        experiment_default, model, args.episodes, n_aircraft=1,
        tta_mode="solo", surrogate=None, k_cps=0,
    )
    _summarize("Condition 2: multi-agent, solo TTA, naive straight-line ETA, N=1", records_2)

    surrogate = experiment_default._build_surrogate()
    if surrogate is None:
        print("\nWARNING: no eta_surrogate.pkl found under cps_coordination/models/ "
              "-- skipping conditions 3 & 4 (real-surrogate conditions).")
        return

    # --- Condition 3: multi-agent, solo TTA, real surrogate, N=1 ---
    records_3 = run_multi_agent_condition(
        experiment_default, model, args.episodes, n_aircraft=1,
        tta_mode="solo", surrogate=surrogate, k_cps=0,
    )
    _summarize("Condition 3: multi-agent, solo TTA, real ETASurrogate, N=1", records_3)

    # --- Condition 4: multi-agent, CPS TTA, k_cps=0, real surrogate, N=5, single runway ---
    experiment_single_rwy = _make_experiment(k_cps=0, mode="static", runways=["27"])
    records_4 = run_multi_agent_condition(
        experiment_single_rwy, model, max(1, args.episodes // 3), n_aircraft=5,
        tta_mode="cps", surrogate=surrogate, k_cps=0, trace_first_episode=True,
    )
    _summarize(
        "Condition 4: multi-agent, CPS TTA (k_cps=0, single runway, real separation "
        "pressure), real surrogate, N=5, FULL RECAT-EU separation",
        records_4,
    )

    # --- Condition 5: same as condition 4, but with reduced_wake_separation ---
    experiment_reduced_sep = _make_experiment(
        k_cps=0, mode="static", runways=["27"],
        reduced_wake_separation=True, wake_separation_scale=args.wake_separation_scale,
    )
    records_5 = run_multi_agent_condition(
        experiment_reduced_sep, model, max(1, args.episodes // 3), n_aircraft=5,
        tta_mode="cps", surrogate=surrogate, k_cps=0,
    )
    _summarize(
        f"Condition 5: multi-agent, CPS TTA (k_cps=0, single runway, real separation "
        f"pressure), real surrogate, N=5, REDUCED wake separation (x{args.wake_separation_scale})",
        records_5,
    )


if __name__ == "__main__":
    main()
