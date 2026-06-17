"""
ttg_experiment.py
-----------------
Probes the decomposed critic's dominance landscape over a synthetic grid of
(x, y, t, TTG, heading) for two agent types:

  spatial  — trained without RTA; desired_goal[2] = 0 always
  tbalp    — trained with RTA sampler; desired_goal[2] encodes normalised RTA

The environment is initialised once per agent type only to sample a set of
valid desired_goal vectors (one per runway via env.reset()). After that the
env is closed and never touched again. All obs tensors are constructed
synthetically from a uniform grid over (x, y, t).

Environment contract
--------------------
obs dict keys: "observation" (3,), "achieved_goal" (3,), "desired_goal" (3,)
  observation[0]   = normalised x
  observation[1]   = normalised y
  observation[2]   = normalised elapsed time  (t / MAX_TIME)
  desired_goal[0]  = normalised goal x
  desired_goal[1]  = normalised goal y
  desired_goal[2]  = normalised RTA  (0.0 for spatial agent)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

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
)
from path_planning.critic.utils import heading_encoder, obs_adapter
from path_planning.critic.reward_experiment import (
    RewardDecompositionWrapper,
    DEFAULT_GOAL_BONUS,
    patch_goal_bonus,
    RewardSweepModelConfig,
)

# ── sweep constants ───────────────────────────────────────────────────────────

# Normalised state grid (uniform coverage of reachable space)
X_VALUES   = np.linspace(-0.5, 0.5, 7)
Y_VALUES   = np.linspace(-0.5, 0.5, 7)
T_VALUES   = np.linspace(0.0,  0.4, 5)   # normalised elapsed time

TTG_VALUES_NORM = np.linspace(-0.1, 0.3, 9)   # negative = already late
HEADING_VALUES  = np.linspace(-np.pi, np.pi, 36)

N_RUNWAY_SAMPLES = 5   # how many env resets to collect desired_goal vectors

AGENT_TYPES = ("spatial", "tbalp")


# ── desired_goal sampling ─────────────────────────────────────────────────────

def sample_desired_goals(env: gym.Env, n: int) -> np.ndarray:
    """
    Resets the env n times and collects the desired_goal vector from each.
    Returns array of shape (n, 3).
    """
    goals = []
    for _ in range(n):
        obs, _ = env.reset()
        goals.append(obs["desired_goal"].copy())
    env.close()
    return np.stack(goals)   # (n, 3)


# ── synthetic obs construction ────────────────────────────────────────────────

def build_obs_batch(
    x: float,
    y: float,
    t: float,
    desired_goals: np.ndarray,
    ttg_norm: float,
    is_spatial: bool,
) -> dict:
    """
    Constructs a batch obs dict for one (x, y, t, TTG) point, replicated
    across all desired_goal vectors.

    desired_goals : (G, 3)
    returns       : dict with arrays of shape (G, 3)
    """
    G = len(desired_goals)
    obs_vec      = np.tile([x, y, t], (G, 1)).astype(np.float32)
    achieved_vec = obs_vec.copy()

    goal_vec = desired_goals.copy().astype(np.float32)
    if not is_spatial:
        # Set RTA so that TTG = ttg_norm at this elapsed time t
        goal_vec[:, 2] = t + ttg_norm

    return {
        "observation":   obs_vec,
        "achieved_goal": achieved_vec,
        "desired_goal":  goal_vec,
    }


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class TTGSweepEnvKwargsConfig(PathPlanningEnvKwargsConfig):
    goal_bonus: float = DEFAULT_GOAL_BONUS
    use_rta:    bool  = False


@dataclass
class TTGSweepEnvConfig(PathPlanningEnvConfig):
    env_kwargs: TTGSweepEnvKwargsConfig = field(
        default_factory=TTGSweepEnvKwargsConfig
    )


@dataclass
class TTGSweepModelConfig(RewardSweepModelConfig):
    pass


# ── experiment ────────────────────────────────────────────────────────────────

class TTGDominanceExperiment(BaseCriticExperiment):
    """
    Runs the (x, y, t, TTG, heading) dominance sweep for one agent type.

    The env is used only once at the start to collect desired_goal vectors;
    all gradient evaluations use synthetically constructed obs dicts.

    agent_type : "spatial" | "tbalp"
    """

    env_config_cls   = TTGSweepEnvConfig
    model_config_cls = TTGSweepModelConfig

    def __init__(self, cfg, agent_type: str, n_runway_samples: int = N_RUNWAY_SAMPLES):
        super().__init__(cfg)
        assert agent_type in AGENT_TYPES, f"agent_type must be one of {AGENT_TYPES}"
        self.agent_type       = agent_type
        self.n_runway_samples = n_runway_samples
        self.grid_rows: List[Dict] = []

    # ── env ───────────────────────────────────────────────────────────────────

    def _extract_patch_kwargs(self, env_kwargs: dict) -> dict:
        return {
            "goal_bonus": env_kwargs.pop("goal_bonus", DEFAULT_GOAL_BONUS),
            "use_rta":    env_kwargs.pop("use_rta", False),
        }

    def apply_env_patches(self, env: gym.Env, **patch_kwargs) -> gym.Env:
        goal_bonus = patch_kwargs.get("goal_bonus", DEFAULT_GOAL_BONUS)
        patch_goal_bonus(env, goal_bonus)
        return RewardDecompositionWrapper(env, goal_bonus=goal_bonus)

    # ── model ─────────────────────────────────────────────────────────────────

    def make_model(self, env: gym.Env):
        mcfg = self.cfg.model
        return DecomposedSAC(
            "MultiInputPolicy",
            env,
            learning_rate=mcfg.learning_rate,
            policy_kwargs=mcfg.policy_kwargs,
            verbose=mcfg.verbose,
        )

    # ── probes (unused but required by base class) ────────────────────────────

    def build_probes(self, model) -> List[CriticProbe]:
        return []

    # ── dominance grid ────────────────────────────────────────────────────────

    def on_probe_complete(self, report) -> None:
        model = self._model
        if not isinstance(model, DecomposedSAC):
            return

        is_spatial = (self.agent_type == "spatial")
        ttg_sweep  = np.array([0.0]) if is_spatial else TTG_VALUES_NORM

        # Collect desired_goal vectors from a fresh env — then discard the env
        env           = self.make_env()
        desired_goals = sample_desired_goals(env, self.n_runway_samples)
        # env is already closed inside sample_desired_goals

        total_points = (len(X_VALUES) * len(Y_VALUES) * len(T_VALUES)
                        * len(ttg_sweep) * len(HEADING_VALUES))
        print(f"  agent={self.agent_type} | grid points={total_points:,} | "
              f"goals={len(desired_goals)}")

        for x in X_VALUES:
            for y in Y_VALUES:
                for t in T_VALUES:
                    for ttg_norm in ttg_sweep:
                        obs_batch = build_obs_batch(
                            x, y, t, desired_goals, ttg_norm, is_spatial
                        )
                        n_states = len(desired_goals)

                        lam_arr = np.empty((len(HEADING_VALUES), n_states), dtype=np.float32)
                        cos_arr = np.empty((len(HEADING_VALUES), n_states), dtype=np.float32)

                        for hi, heading in enumerate(HEADING_VALUES):
                            action_batch = np.tile(heading_encoder(heading), (n_states, 1))
                            g_aug, g_sp, _, _ = compute_exact_gradients(
                                model, obs_batch, action_batch
                            )
                            lam, cos = dominance_metrics(g_aug, g_sp)
                            lam_arr[hi] = lam
                            cos_arr[hi] = cos

                        for hi, heading in enumerate(HEADING_VALUES):
                            self.grid_rows.append({
                                "agent_type":       self.agent_type,
                                "x_norm":           float(x),
                                "y_norm":           float(y),
                                "t_norm":           float(t),
                                "ttg_norm":         float(ttg_norm),
                                "ttg_seconds":      float(ttg_norm * 21600),
                                "heading_rad":      float(heading),
                                "heading_deg":      float(np.degrees(heading)),
                                "lambda_mean":      float(lam_arr[hi].mean()),
                                "lambda_std":       float(lam_arr[hi].std()),
                                "lambda_median":    float(np.median(lam_arr[hi])),
                                "lambda_p05":       float(np.percentile(lam_arr[hi], 5)),
                                "lambda_p95":       float(np.percentile(lam_arr[hi], 95)),
                                "cos_theta_mean":   float(cos_arr[hi].mean()),
                                "cos_theta_std":    float(cos_arr[hi].std()),
                                "cos_theta_median": float(np.median(cos_arr[hi])),
                            })

        print(f"  done | rows written={len(self.grid_rows):,}")