from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import numpy as np
import gymnasium as gym

from path_planning.experiment.base import (
    PathPlanningEnvKwargsConfig,
    PathPlanningEnvConfig,
    PathPlanningModelConfig,
)
from path_planning.experiment.base_critic import BaseCriticExperiment, CriticProbe
from path_planning.critic.continouos_critic import (
    DecomposedSAC,
    compute_exact_gradients,
    dominance_metrics,
    SAC
)
from path_planning.critic.utils import heading_encoder, obs_adapter

DEFAULT_GOAL_BONUS = 10.0

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

def patch_goal_bonus(env: gym.Env, goal_bonus: float) -> None:
    """
    Monkey-patches _get_terminated on the unwrapped env to replace the
    hardcoded 10.0 success reward with goal_bonus.
    """
    base = env.unwrapped
    original = base._get_terminated.__func__  # unbound method

    bonus_diff = goal_bonus - DEFAULT_GOAL_BONUS

    def _get_terminated_patched(self):
        result = original(self)
        if self.death_cause == "success":
            self.segment_reward += bonus_diff
        return result

    import types
    base._get_terminated = types.MethodType(_get_terminated_patched, base)

class RewardDecompositionWrapper(gym.Wrapper):
    """
    Splits the env's reward signal into dense and sparse components so
    DecomposedSAC can store them separately in its replay buffer.

    Adds to info on every step:
        "reward_dense"  : R_dense  (path-length + population noise ticks)
        "reward_sparse" : R_sparse (goal bonus or failure penalty, non-zero
                                    only at terminal steps)

    Also overrides compute_reward so HER relabelling uses the same
    goal_bonus as the live rollout.
    """

    def __init__(self, env: gym.Env, goal_bonus: float):
        super().__init__(env)
        self.goal_bonus = goal_bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped

        r_dense  = float(base.step_reward)
        r_sparse = float(reward) - r_dense

        info["reward_dense"]  = r_dense
        info["reward_sparse"] = r_sparse

        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        infos: list,
    ) -> np.ndarray:
        """
        HER calls this to relabel rewards when the goal is swapped.

        We reconstruct R_dense from the stored info key and re-derive
        R_sparse from the (possibly relabelled) death_cause.

        Note: for HER spatial-only, any terminal arrival counts as
        success (wrong_runway included), so we award goal_bonus there too.
        """
        r_dense = np.array(
            [i.get("reward_dense", i.get("step_reward", 0.0)) for i in infos],
            dtype=np.float32,
        )

        sparse: list[float] = []
        for i in infos:
            cause = i.get("death_cause", "")
            if cause in ("success", "wrong_runway"):
                sparse.append(self.goal_bonus)
            elif cause in ("restrict", "timeout", "out_of_bounds"):
                sparse.append(-1.0)
            else:
                sparse.append(0.0)

        return r_dense + np.array(sparse, dtype=np.float32)
            
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RewardSweepEnvKwargsConfig(PathPlanningEnvKwargsConfig):
    """Adds goal_bonus on top of the standard env kwargs."""
    goal_bonus: float = 10.0


@dataclass
class RewardSweepEnvConfig(PathPlanningEnvConfig):
    env_kwargs: RewardSweepEnvKwargsConfig = field(
        default_factory=RewardSweepEnvKwargsConfig
    )


@dataclass
class RewardSweepModelConfig(PathPlanningModelConfig):
    """Forces use_her=False — DecomposedSAC uses its own replay buffer."""
    use_her: bool = False
    algorithm: type = DecomposedSAC




# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class RewardScaleCriticExperiment(BaseCriticExperiment):
    """
    Trains a DecomposedSAC on PathPlanningGoalEnv and probes the Q-landscape
    over a heading sweep to study gradient dominance between Q_aug and
    Q_sparse as goal_bonus magnitude varies.

    Single run (one goal_bonus from cfg):
        exp = RewardScaleCriticExperiment(cfg)
        exp.run()

    Sweep across goal_bonus values (retrains each time):
        exp.run_sweep(
            [{"goal_bonus": b} for b in [1, 5, 10, 50, 100, 500, 1000]],
            output_path="results/reward_sweep.csv",
            extra_columns=["goal_bonus"],
        )
    """

    env_config_cls   = RewardSweepEnvConfig
    model_config_cls = RewardSweepModelConfig

    # ------------------------------------------------------------------ #
    # Env                                                                  #
    # ------------------------------------------------------------------ #

    def _extract_patch_kwargs(self, env_kwargs: dict) -> dict:
        """Pop goal_bonus so it goes to apply_env_patches, not gym.make()."""
        return {"goal_bonus": env_kwargs.pop("goal_bonus", DEFAULT_GOAL_BONUS)}

    def apply_env_patches(self, env: gym.Env, **patch_kwargs) -> gym.Env:
        """Wrap the base env with the reward-decomposition layer."""
        goal_bonus = patch_kwargs.get("goal_bonus", DEFAULT_GOAL_BONUS)
        patch_goal_bonus(env, goal_bonus)
        return RewardDecompositionWrapper(env, goal_bonus)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #

    def make_model(self, env: gym.Env):
        mcfg = self.cfg.model

        assert not getattr(mcfg, "use_her", False), (
            "DecomposedSAC does not support HerReplayBuffer. Set use_her=False."
        )

        return DecomposedSAC(
            "MultiInputPolicy",
            env,
            learning_rate=mcfg.learning_rate,
            policy_kwargs=mcfg.policy_kwargs,
            verbose=mcfg.verbose,
        )

    # ------------------------------------------------------------------ #
    # Probes                                                               #
    # ------------------------------------------------------------------ #

    def build_probes(self, model) -> List[CriticProbe]:
        """Q-value probe only. Dominance is computed post-hoc in on_probe_complete."""

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
                sweep_values=np.linspace(-np.pi, np.pi, 100),
                encoder=heading_encoder,
                obs_adapter=obs_adapter,
                agg=q_values_only,
            )
        ]

    # ------------------------------------------------------------------ #
    # Dominance analysis                                                   #
    # ------------------------------------------------------------------ #

    def analyse_dominance(self, obs: np.ndarray, actions: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Computes exact gradient-based dominance metrics for a batch of actions.
        obs: (Batch, Obs_Dim) or (Obs_Dim,)
        actions: (Batch, Act_Dim)
        """
        if not isinstance(self._model, DecomposedSAC):
            batch_size = actions.shape[0]
            return {
                "lambda": np.zeros(batch_size), 
                "cos_theta": np.zeros(batch_size), 
                "q_aug": np.zeros(batch_size), 
                "q_sparse": np.zeros(batch_size)
            }

        # Ensure obs is tiled to match the action batch size if it's a single state
        if obs.ndim == 1:
            obs = np.tile(obs, (actions.shape[0], 1))

        grad_aug, grad_sparse, q_aug, q_sp = compute_exact_gradients(
            self._model, obs, actions
        )
        
        lam, cos_theta = dominance_metrics(grad_aug, grad_sparse)

        return {
            "lambda": lam,
            "cos_theta": cos_theta,
            "q_aug": q_aug,
            "q_sparse": q_sp
        }
    
#TODO: Run the experiment for 100_000 timesteps per goal_bonus value, sweep across a range of goal_bonus values, and plot the resulting dominance metrics to understand how reward scaling affects the critic's learning dynamics. The goal_bonus values should be something like [0, 1, 5, 10, 50, 100, 1000]. And what else should we do, for analysing the results? Maybe also plot the reward curves and success rates to see how performance correlates with dominance metrics?