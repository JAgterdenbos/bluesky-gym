"""
run_ttg_sweep.py
----------------
Loads or trains a DecomposedSAC for each agent type (spatial, tbalp),
runs the (TTG × heading) dominance grid, and writes:

  ttg_sweep_grid.csv   — one row per (agent_type, ttg_norm, heading)
  ttg_sweep_scalar.csv — one row per agent_type (aggregated metrics)

Usage
-----
    python run_ttg_sweep.py \
        [--spatial-run-id  reward_sweep_gb10] \
        [--tbalp-run-id    tbalp_gb10] \
        [--timesteps       100000] \
        [--n-eval-episodes 10] \
        [--n-context-steps 30] \
        [--pretrained-variant final_model] \
        [--no-training] \
        [--output experiments/ttg_sweep/results/ttg_sweep_grid.csv]
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import bluesky_gym
from path_planning.critic.ttg_experiment import (
    TTGDominanceExperiment,
    TTGSweepEnvKwargsConfig,
    TTGSweepEnvConfig,
    TTGSweepModelConfig,
    AGENT_TYPES,
)
from path_planning.critic.continouos_critic import DecomposedSAC
from bluesky_gym.experiment import ExperimentConfig, SessionConfig

# Fixed goal_bonus for this experiment — reward scale is not the variable here
GOAL_BONUS = 10.0


def build_config(
    run_id: str,
    timesteps: int,
    do_train: bool,
    pretrained_variant: str,
    use_rta: bool,
) -> ExperimentConfig:
    env_kwargs = TTGSweepEnvKwargsConfig(goal_bonus=GOAL_BONUS, use_rta=use_rta)
    env_cfg    = TTGSweepEnvConfig(env_kwargs=env_kwargs)
    model_cfg  = TTGSweepModelConfig()
    session    = SessionConfig(
        total_timesteps=timesteps,
        do_train=do_train,
        do_evaluate=False,
        pretrained_model_variant=pretrained_variant,
        pretrained_run_id=run_id if not do_train else None,
    )
    return ExperimentConfig(
        run_id=run_id,
        env=env_cfg,
        model=model_cfg,
        session=session,
    )


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")


def run_sweep(
    spatial_run_id:    str,
    tbalp_run_id:      str,
    timesteps:         int,
    output:            str,
    n_eval_episodes:   int,
    n_context_steps:   int,
    pretrained_variant: str,
    no_training:       bool,
) -> None:
    bluesky_gym.register_envs()

    output_path  = Path(output)
    scalar_path  = output_path.with_name(output_path.stem.replace("_grid", "_scalar") + ".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # (agent_type, run_id, use_rta)
    conditions = [
        ("spatial", spatial_run_id, False),
        ("tbalp",   tbalp_run_id,   True),
    ]

    all_grid:   List[Dict[str, Any]] = []
    all_scalar: List[Dict[str, Any]] = []

    SEP = "=" * 60

    for agent_type, run_id, use_rta in conditions:
        pattern = f"./experiments/*/*/models/{run_id}/{pretrained_variant}.zip"
        hits    = glob.glob(pattern)
        do_train = not (no_training and bool(hits))

        cfg = build_config(run_id, timesteps, do_train, pretrained_variant, use_rta)

        exp = TTGDominanceExperiment(
            cfg,
            agent_type=agent_type,
            n_eval_episodes=n_eval_episodes,
            n_context_steps=n_context_steps,
        )

        print(f"\n{SEP}")
        print(f"  agent_type={agent_type} | run_id={run_id} | use_rta={use_rta}")
        mode = f"Loading {pretrained_variant} from {hits[0]}" if not do_train else f"Training | timesteps={timesteps}"
        print(f"  {mode}")
        print(SEP)

        if not do_train:
            exp.load_model()
        else:
            exp.train()

        model  = exp._model
        probes = exp.build_probes(model)
        ctx    = exp.build_context(model)
        report = exp.probe_critic(model, probes, context=ctx)
        exp.on_probe_complete(report)

        all_grid.extend(exp.grid_rows)

        # Aggregate scalar row per agent type
        if exp.grid_rows:
            lam_vals = np.array([r["lambda_median"]    for r in exp.grid_rows])
            cos_vals = np.array([r["cos_theta_median"] for r in exp.grid_rows])
            sr       = exp.grid_rows[0]["success_rate"]
            all_scalar.append({
                "agent_type":       agent_type,
                "lambda_mean":      float(lam_vals.mean()),
                "lambda_std":       float(lam_vals.std()),
                "lambda_median":    float(np.median(lam_vals)),
                "lambda_p05":       float(np.percentile(lam_vals, 5)),
                "lambda_p95":       float(np.percentile(lam_vals, 95)),
                "cos_theta_mean":   float(cos_vals.mean()),
                "cos_theta_std":    float(cos_vals.std()),
                "cos_theta_median": float(np.median(cos_vals)),
                "success_rate":     sr,
            })

    _write_csv(all_grid,   output_path)
    _write_csv(all_scalar, scalar_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial-run-id",     type=str, default="reward_sweep_gb10",
                        help="run_id of the trained spatial (no-RTA) model")
    parser.add_argument("--tbalp-run-id",       type=str, default="tbalp_gb10",
                        help="run_id of the trained TBALP (with-RTA) model")
    parser.add_argument("--timesteps",          type=int, default=100_000)
    parser.add_argument("--n-eval-episodes",    type=int, default=10)
    parser.add_argument("--n-context-steps",    type=int, default=30)
    parser.add_argument("--pretrained-variant", type=str, default="final_model",
                        choices=["final_model", "best_model", "checkpoint_model"])
    parser.add_argument("--no-training",        action="store_true",
                        help="Skip training; requires pretrained models on disk.")
    parser.add_argument("--output", type=str,
                        default="experiments/ttg_sweep/results/ttg_sweep_grid.csv")
    args = parser.parse_args()

    run_sweep(
        spatial_run_id=args.spatial_run_id,
        tbalp_run_id=args.tbalp_run_id,
        timesteps=args.timesteps,
        output=args.output,
        n_eval_episodes=args.n_eval_episodes,
        n_context_steps=args.n_context_steps,
        pretrained_variant=args.pretrained_variant,
        no_training=args.no_training,
    )
