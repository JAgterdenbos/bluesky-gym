"""
path_planning/experiment.py
----------------------------
PathPlanning-specific experiment definition.

This is the only file a user needs to write to plug their environment
into the bluesky_gym experiment framework.  It contains:

  PathPlanningModelConfig    - adds net_arch, policy_kwargs, and HER fields
  PathPlanningEnvKwargsConfig - gym.make() kwargs (action_mode, use_rta, runways)
  PathPlanningEnvConfig      - wraps the above; sets env_name, group_key, success_key
  PathPlanningExperiment     - make_env / make_model / metric_extractor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, cast

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from bluesky_gym.experiment import (
    BaseExperiment,
    EnvKwargsConfig,
    EnvConfig,
    ModelConfig,
    MetricExtractor,
)


# ---------------------------------------------------------------------------
# Config subclasses
# ---------------------------------------------------------------------------

@dataclass()
class PathPlanningModelConfig(ModelConfig):
    """ModelConfig for PathPlanning.

    Adds network architecture, policy_kwargs, and HER hyper-parameters
    on top of the sparse ModelConfig base.

    All fields become CLI flags automatically:
      --model-algorithm SAC
      --model-learning-rate 3e-4
      --model-net-arch ...         (not directly CLI-able as a dict, set via YAML)
      --model-use-her / --model-no-use-her
      --model-her-n-sampled-goal 4
      --model-her-goal-selection-strategy future
    """

    algorithm: Type = SAC

    # Network architecture — not exposed as a CLI flag (dict type not supported),
    # but can be overridden via a YAML config file.
    net_arch: Dict[str, List[int]] = field(
        default_factory=lambda: dict(pi=[256, 256, 256], qf=[256, 256, 256])
    )

    use_her:                     bool = True
    her_n_sampled_goal:          int  = 4
    her_goal_selection_strategy: str  = "final"   # "future" | "final" | "episode"

    @property
    def policy_kwargs(self) -> Dict[str, Any]:
        return dict(net_arch=self.net_arch)

@dataclass
class PathPlanningEnvKwargsConfig(EnvKwargsConfig):
    """gym.make() kwargs for PathPlanningGoalEnv.

    All fields become CLI flags automatically:
      --env-action-mode hdg
      --env-use-rta / --env-no-use-rta
      --env-runways 27 18R 06
    """

    action_mode: str                 = "hdg"
    use_rta:     bool                = False
    runways:     Optional[List[str]] = field(default_factory=lambda: None)


@dataclass
class PathPlanningEnvConfig(EnvConfig):
    """Full environment config for PathPlanningGoalEnv.

    Sets the env_name, group_key, and success_key used by the framework,
    and nests the PathPlanningEnvKwargsConfig for gym.make() kwargs.

    Additional CLI flags from EnvConfig:
      --env-env-name PathPlanningGoalEnv-v0
      --env-group-key current_runway
      --env-success-key is_success
    """

    env_kwargs:  PathPlanningEnvKwargsConfig = field(
        default_factory=PathPlanningEnvKwargsConfig
    )
    env_name:    str = "PathPlanningGoalEnv-v0"
    group_key:   str = "current_runway"
    success_key: str = "is_success"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class PathPlanningExperiment(BaseExperiment):
    """Experiment for PathPlanningGoalEnv-v0.

    Plugs in:
      - SAC + HerReplayBuffer (or SAC without HER if use_her=False)
      - Monitor wrapper
      - PathPlanning-specific metrics: flight_time, path_length_km, noise_reward
    """

    model_config_cls = PathPlanningModelConfig
    env_config_cls   = PathPlanningEnvConfig

    # ------------------------------------------------------------------ #
    # Abstract methods                                                     #
    # ------------------------------------------------------------------ #

    def make_env(
        self,
        env_kwargs:  dict,
        render_mode: str | None = None,
    ) -> gym.Env:
        """Create a Monitor-wrapped PathPlanningGoalEnv."""
        env_name = self.cfg.env.env_name

        if env_name is None:
            raise ValueError("env_name is not set in config!")

        env = gym.make(
            env_name
            render_mode=render_mode,
            **env_kwargs,
        )
        return Monitor(env)

    def make_model(self, env: gym.Env):
        cfg  = self.cfg
        mcfg = mcfg = cast(PathPlanningModelConfig, cfg.model)  # PathPlanningModelConfig

        algorithm_cls = mcfg.get_algorithm()

        if mcfg.use_her:
            return algorithm_cls(
                "MultiInputPolicy",
                env,
                learning_rate=mcfg.learning_rate,
                policy_kwargs=mcfg.policy_kwargs,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=mcfg.her_n_sampled_goal,
                    goal_selection_strategy=mcfg.her_goal_selection_strategy,
                    copy_info_dict=True,
                ),
                verbose=mcfg.verbose,
            )
        else:
            return algorithm_cls(
                "MultiInputPolicy",
                env,
                learning_rate=mcfg.learning_rate,
                policy_kwargs=mcfg.policy_kwargs,
                verbose=mcfg.verbose,
            )

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def metric_extractor(cls) -> MetricExtractor:
        def _path_length_km(info: dict, _ok: bool) -> float:
            plw      = info.get("path_length_weight", 0.0)
            path_rew = info.get("average_path_rew",   0.0)
            if plw == 0:
                return float("nan")
            return float((path_rew / plw) * 1.852)   # NM → km

        return MetricExtractor(
            extractors={
                "flight_time":    lambda info, ok: (
                    info.get("sim_time", float("nan")) / 60 if ok else float("nan")
                ),
                "path_length_km": _path_length_km,
                "noise_reward":   lambda info, _ok: float(
                    info.get("average_noise_rew", float("nan"))
                ),
            },
            display=["flight_time", "path_length_km", "noise_reward"],
        )