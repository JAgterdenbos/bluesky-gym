"""
run_reward_sweep.py
-------------------
Trains DecomposedSAC for each goal_bonus, evaluates a heading sweep,
computes dominance metrics, and writes two CSVs:

  reward_sweep.csv          — one row per goal_bonus (scalar aggregates)
  reward_sweep_heading.csv  — one row per (goal_bonus, heading) with
                              mean/std of λ, cos θ across all context states

Usage
-----
    python run_reward_sweep.py \
        [--timesteps 100000] \
        [--n-eval-episodes 10] \
        [--n-context-steps 30] \
        [--n-headings 72] \
        [--pretrained-variant final_model|best_model|checkpoint_model] \
        [--output experiments/reward_sweep/results/reward_sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from path_planning.critic.reward_experiment import (
    RewardScaleCriticExperiment,
    RewardSweepEnvConfig,
    RewardSweepModelConfig,
    RewardSweepEnvKwargsConfig,
)
from path_planning.experiment.base_critic import CriticProbe, ProbeReport
from path_planning.critic.continouos_critic import (
    DecomposedSAC,
    compute_exact_gradients,
    dominance_metrics,
)
from path_planning.critic.utils import heading_encoder, obs_adapter
from bluesky_gym.experiment import ExperimentConfig, SessionConfig

GOAL_BONUS_VALUES = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0, 10_000.0, 100_000.0, 1_000_000.0]


def build_config(goal_bonus: float, timesteps: int, run_id: str, do_train: bool = True, pretrained_variant: str = "final_model") -> ExperimentConfig:
    env_kwargs = RewardSweepEnvKwargsConfig(goal_bonus=goal_bonus)
    env_cfg    = RewardSweepEnvConfig(env_kwargs=env_kwargs)
    model_cfg  = RewardSweepModelConfig()
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


class _SweepCollector(RewardScaleCriticExperiment):
    """
    Extends RewardScaleCriticExperiment with:
      - Configurable n_eval_episodes and n_context_steps.
      - Full episode rollout to collect success rate, mean episode reward,
        and mean episode length.
      - Heading sweep stored in two flat row-lists (scalar CSV + heading CSV).
    """

    def __init__(
        self,
        cfg,
        goal_bonus: float,
        n_eval_episodes: int = 10,
        n_context_steps: int = 30,
        n_headings: int = 72,
    ):
        super().__init__(cfg)
        self._goal_bonus      = goal_bonus
        self.n_eval_episodes  = n_eval_episodes
        self.n_context_steps  = n_context_steps
        self.n_headings       = n_headings
        self.scalar_rows:  List[Dict[str, Any]] = []
        self.heading_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Q-value probe (no grad, used by base probe_critic)
    # ------------------------------------------------------------------

    def build_probes(self, model) -> List[CriticProbe]:
        def q_values_only(q1, q2, obs, action):
            import torch
            (a0, a1), (s0, s1) = model.critic.forward_decomposed(obs, action)
            return {
                "q_aug":    float(torch.min(a0, a1).mean().item()),
                "q_sparse": float(torch.min(s0, s1).mean().item()),
                "q_total":  float((torch.min(a0, a1) + torch.min(s0, s1)).mean().item()),
            }

        return [
            CriticProbe(
                name="decomp",
                sweep_values=np.linspace(-np.pi, np.pi, self.n_headings),
                encoder=heading_encoder,
                obs_adapter=obs_adapter,
                agg=q_values_only,
            )
        ]

    # ------------------------------------------------------------------
    # Episode rollout — returns context tensors + eval stats
    # ------------------------------------------------------------------

    def _rollout_episodes(self, model) -> tuple[dict, dict]:
        """
        Runs n_eval_episodes deterministic episodes.

        Returns
        -------
        context_np : dict[str, np.ndarray]  — evenly-sampled states per episode,
                     concatenated, shape (total_states, dim_k)
        eval_stats : dict with keys success_rate, mean_ep_reward, mean_ep_length
        """
        import torch

        env = self.make_env()

        successes:   List[bool]  = []
        ep_rewards:  List[float] = []
        ep_lengths:  List[int]   = []
        all_obs:     List[dict]  = []

        for _ in range(self.n_eval_episodes):
            obs, _ = env.reset()
            ep_obs:    List[dict] = []
            ep_reward: float = 0.0
            ep_len:    int   = 0
            done = False

            while not done and ep_len < self.n_context_steps: 
                ep_obs.append({k: torch.as_tensor(np.array(v), dtype=torch.float32)
                               for k, v in obs.items()})
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_reward += float(r)
                ep_len    += 1
                done = term or trunc

            successes.append(info.get("death_cause", "") == "success")
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)

            idx    = np.linspace(0, len(ep_obs) - 1, self.n_context_steps, dtype=int)
            ep_obs = [ep_obs[i] for i in idx]
            all_obs.extend(ep_obs)

        env.close()

        # Stack into numpy arrays
        context_np = {
            k: np.stack([o[k].numpy() for o in all_obs])
            for k in all_obs[0].keys()
        }
        eval_stats = {
            "success_rate":    float(np.mean(successes)),
            "mean_ep_reward":  float(np.mean(ep_rewards)),
            "std_ep_reward":   float(np.std(ep_rewards)),
            "mean_ep_length":  float(np.mean(ep_lengths)),
            "n_eval_episodes": self.n_eval_episodes,
            "n_context_states": len(all_obs),
        }
        return context_np, eval_stats

    # ------------------------------------------------------------------
    # Post-probe: dominance sweep + success rate
    # ------------------------------------------------------------------

    def on_probe_complete(self, report: ProbeReport) -> None:
        model = self._model
        if not isinstance(model, DecomposedSAC):
            return

        sweep_headings = np.linspace(-np.pi, np.pi, self.n_headings)

        context_np, eval_stats = self._rollout_episodes(model)
        n_states = context_np[next(iter(context_np))].shape[0]

        # lam_arr / cos_arr: (n_headings, n_states)
        lam_arr = np.empty((self.n_headings, n_states), dtype=np.float32)
        cos_arr = np.empty((self.n_headings, n_states), dtype=np.float32)

        for hi, heading in enumerate(sweep_headings):
            action_batch = np.tile(heading_encoder(heading), (n_states, 1))
            g_aug, g_sp, _, _ = compute_exact_gradients(model, context_np, action_batch)
            lam, cos = dominance_metrics(g_aug, g_sp)
            lam_arr[hi] = lam
            cos_arr[hi] = cos

        lam_flat = lam_arr.ravel()
        cos_flat = cos_arr.ravel()

        # Q aggregate from probe
        decomp  = report.by_probe("decomp")
        summary = decomp.summary if decomp else {}

        scalar_row: Dict[str, Any] = {
            "goal_bonus":       self._goal_bonus,
            "lambda_mean":      float(np.mean(lam_flat)),
            "lambda_std":       float(np.std(lam_flat)),
            "lambda_median":    float(np.median(lam_flat)),
            "lambda_p05":       float(np.percentile(lam_flat, 5)),
            "lambda_p95":       float(np.percentile(lam_flat, 95)),
            "lambda_max":       float(np.max(lam_flat)),
            "cos_theta_mean":   float(np.mean(cos_flat)),
            "cos_theta_std":    float(np.std(cos_flat)),
            "cos_theta_median": float(np.median(cos_flat)),
            "cos_theta_p05":    float(np.percentile(cos_flat, 5)),
            "cos_theta_p95":    float(np.percentile(cos_flat, 95)),
            "cos_theta_min":    float(np.min(cos_flat)),
            **eval_stats,
            **{f"q_{k}": v for k, v in summary.items()},
        }
        self.scalar_rows.append(scalar_row)

        # Per-heading rows (for fig3)
        for hi, heading in enumerate(sweep_headings):
            self.heading_rows.append({
                "goal_bonus":       self._goal_bonus,
                "heading_rad":      float(heading),
                "heading_deg":      float(np.degrees(heading)),
                "lambda_mean":      float(lam_arr[hi].mean()),
                "lambda_std":       float(lam_arr[hi].std()),
                "lambda_median":    float(np.median(lam_arr[hi])),
                "cos_theta_mean":   float(cos_arr[hi].mean()),
                "cos_theta_std":    float(cos_arr[hi].std()),
                "cos_theta_median": float(np.median(cos_arr[hi])),
                # Carry eval stats so plots can cross-reference
                "success_rate":     eval_stats["success_rate"],
                "mean_ep_reward":   eval_stats["mean_ep_reward"],
            })

        print(
            f"  goal_bonus={self._goal_bonus:.1f} | "
            f"λ={scalar_row['lambda_mean']:.3f}±{scalar_row['lambda_std']:.3f} | "
            f"cos={scalar_row['cos_theta_mean']:.3f} | "
            f"success={eval_stats['success_rate']:.2%} | "
            f"n_states={n_states}"
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
    timesteps: int,
    output: str,
    n_eval_episodes: int,
    n_context_steps: int,
    n_headings: int,
    pretrained_variant: str = "final_model",
    no_training: bool = False,
) -> None:
    import bluesky_gym, glob as _glob
    bluesky_gym.register_envs()

    output_path  = Path(output)
    heading_path = output_path.with_name(output_path.stem + "_heading.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_scalar:  List[Dict[str, Any]] = []
    all_heading: List[Dict[str, Any]] = []

    SEP = "=" * 60

    for goal_bonus in GOAL_BONUS_VALUES:
        run_id  = f"reward_sweep_gb{goal_bonus:.0f}"
        pattern = f"./experiments/*/*/models/{run_id}/{pretrained_variant}.zip"
        hits = _glob.glob(pattern)

        no_training = no_training and bool(hits)  # Only skip training if the pretrained model actually exists on disk.)
        
        cfg = build_config(goal_bonus, timesteps, run_id, do_train=not no_training, pretrained_variant=pretrained_variant)


        if no_training:
            mode_label = f"Loading {pretrained_variant} from {hits[0]}"
        else:
            mode_label = f"Training | timesteps={timesteps}"

        exp = _SweepCollector(
            cfg,
            goal_bonus=goal_bonus,
            n_eval_episodes=n_eval_episodes,
            n_context_steps=n_context_steps,
            n_headings=n_headings,
        )

        print(f"\n{SEP}")
        print(f"  goal_bonus={goal_bonus} | {mode_label}")
        print(f"  eval_episodes={n_eval_episodes} | context_steps={n_context_steps} | headings={n_headings}")
        print(SEP)

        # train() handles both cases: trains from scratch or loads pretrained model.
        # load_model() is called by run() when do_train=False + pretrained_model_path is set,
        # but here we call train() directly so _model is always populated after this line.
        if no_training:
            exp.load_model()
        else:
            exp.train()

        model  = exp._model
        probes = exp.build_probes(model)
        ctx    = exp.build_context(model)
        report = exp.probe_critic(model, probes, context=ctx)
        exp.on_probe_complete(report)

        all_scalar.extend(exp.scalar_rows)
        all_heading.extend(exp.heading_rows)

    _write_csv(all_scalar,  output_path)
    _write_csv(all_heading, heading_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",       type=int,   default=100_000)
    parser.add_argument("--n-eval-episodes", type=int,   default=10,
                        help="Episodes to roll out for success rate + context collection")
    parser.add_argument("--n-context-steps", type=int,   default=30,
                        help="States sampled per episode for gradient computation")
    parser.add_argument("--n-headings",      type=int,   default=72,
                        help="Number of heading probe points (evenly spaced over [-π, π])")
    parser.add_argument(
        "--pretrained-variant", type=str, default="final_model",
        choices=["final_model", "best_model", "checkpoint_model"],
        help=(
            "Which saved checkpoint to load when a matching run_id exists on disk. "
            "If no matching run is found the condition is trained from scratch instead."
        ),
    )
    parser.add_argument("--no-training", action="store_true",
                        help="If set, skips training and only runs evaluation + probes. "
                             "Requires that the specified pretrained_variant exists on disk for each goal_bonus.")
    parser.add_argument("--output", type=str,
                        default="experiments/reward_sweep/results/reward_sweep.csv")
    args = parser.parse_args()

    run_sweep(
        timesteps=args.timesteps,
        output=args.output,
        n_eval_episodes=args.n_eval_episodes,
        n_context_steps=args.n_context_steps,
        n_headings=args.n_headings,
        pretrained_variant=args.pretrained_variant,
        no_training=args.no_training
    )