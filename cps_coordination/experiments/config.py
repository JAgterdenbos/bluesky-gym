"""
cps_coordination/experiments/config.py
---------------------------------------
CPS-specific configuration dataclasses extending the bluesky_gym experiment
framework.  All fields become auto-generated CLI flags via
ExperimentConfig._build_parser() — no argparse boilerplate required.

Exported classes
----------------
  CPSModelConfig      — k_cps, delta_t_plan, delta_update, runway_assignment_mode
  CPSEnvKwargsConfig  — v_app, rta_sampler_path, runways, action_mode
  CPSEnvConfig        — env_name, group_key, success_key + nested env_kwargs
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Type

from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME
from bluesky_gym.experiment import (
    EnvConfig,
    EnvKwargsConfig,
    ModelConfig,
)

from cps_coordination.coordination.cps_manager import CPSManager

__all__ = [
    "CPSModelConfig",
    "CPSEnvKwargsConfig",
    "CPSEnvConfig",
]


@dataclass
class CPSModelConfig(ModelConfig):
    """ModelConfig for the CPS coordination layer.

    The CPS coordination algorithm is not a trainable RL policy; it wraps
    a *frozen* pre-trained spatial-temporal path-planning worker agent.
    The ``algorithm`` field is set to :class:`CPSManager` as a sentinel so
    ``ExperimentConfig._build_paths()`` produces a meaningful directory name
    (``experiments/PathPlanningGoalEnv-v0/CPSManager/…``).

    Auto-generated CLI flags
    ------------------------
      --model-k-cps              int   Maximum positional shift from FCFS rank.
      --model-delta-t-plan       int   Replanning interval (simulation seconds).
      --model-delta-update       float Minimum TTA change to push a worker update (s).
      --model-runway-assignment-mode  str  "static" | "dynamic"
    """

    # CPS coordination hyper-parameters
    k_cps: int = 3
    """k-CPS window: max positions an aircraft may shift from FCFS rank."""

    delta_t_plan: int = 60
    """Replanning interval in simulation seconds (ΔT_plan)."""

    delta_update: float = 1.0
    """Minimum TTA shift (seconds) that triggers a worker goal update (δ_update)."""

    runway_assignment_mode: str = "dynamic"
    """Runway assignment strategy: ``"static"`` or ``"dynamic"``."""

    eta_surrogate_path: Optional[str] = None
    """Path to a fitted :class:`ETASurrogate` (joblib pickle). ``None`` falls
    back to the canonical ``cps_coordination/models/eta_surrogate.pkl`` if it
    exists, else CPSManager runs without ETA prediction (initial ETA
    estimates are never refreshed after fleet construction)."""

    reduced_wake_separation: bool = False
    """If True, scale every RECAT-EU separation value (``_load_recat_matrix``)
    by ``wake_separation_scale`` before it reaches both the greedy scheduler
    and the C_sep compliance metric — an idealized reduced-separation ATC
    scenario (e.g. Time-Based Separation / wake-sensor-equipped operations),
    not today's full RECAT-EU minima. The frozen worker's own delay-absorption
    capacity (~20 min of path-stretching, measured empirically from its
    training-time RTA sampler) is far smaller than the multi-aircraft delays
    full RECAT-EU separation can force onto a shared runway — this flag lets
    the CPS coordination methodology be evaluated against a frozen worker
    without first requiring a wider-trained DTG/RTA sampler (tracked as
    future work). ``False`` (default) uses the real, unscaled matrix."""

    wake_separation_scale: float = 0.5
    """Multiplier applied to the RECAT-EU matrix when
    ``reduced_wake_separation=True`` (e.g. 0.5 = half the real minima).
    Exposed as its own field, not hardcoded, so a sweep across separation
    stringency is a single CLI flag away. Ignored when
    ``reduced_wake_separation=False``."""

    def __post_init__(self) -> None:
        # Set CPSManager as the sentinel algorithm for path / display purposes.
        # We do NOT call super().__post_init__() because that would try to
        # resolve ``algorithm`` as a string name via resolve_algorithm().
        self.algorithm = CPSManager  # type: ignore[assignment]

    def validate(self) -> None:
        """Override: CPS coordination requires no SB3 algorithm."""
        if self.k_cps < 0:
            raise ValueError(f"k_cps must be >= 0, got {self.k_cps}.")
        if self.delta_t_plan <= 0:
            raise ValueError(f"delta_t_plan must be > 0, got {self.delta_t_plan}.")
        if self.delta_update < 0:
            raise ValueError(f"delta_update must be >= 0, got {self.delta_update}.")
        if self.wake_separation_scale <= 0:
            raise ValueError(
                f"wake_separation_scale must be > 0, got {self.wake_separation_scale}."
            )
        if self.runway_assignment_mode not in {"static", "dynamic"}:
            raise ValueError(
                f"runway_assignment_mode must be 'static' or 'dynamic', "
                f"got '{self.runway_assignment_mode}'."
            )
        # The worker only reacts to a new desired_goal once per ACTION_TIME
        # (120s) decision step, so a replanning interval shorter than that
        # (or not a clean multiple of it) burns CPS replanning compute the
        # worker can never actually observe the intermediate result of.
        if self.delta_t_plan < ACTION_TIME:
            warnings.warn(
                f"delta_t_plan={self.delta_t_plan}s is shorter than the worker's "
                f"decision cadence ACTION_TIME={ACTION_TIME}s — replans between "
                "decision steps are wasted since the worker only sees the TTA "
                "current at its next step() call.",
                stacklevel=2,
            )
        elif self.delta_t_plan % ACTION_TIME != 0:
            warnings.warn(
                f"delta_t_plan={self.delta_t_plan}s is not a clean multiple of "
                f"ACTION_TIME={ACTION_TIME}s — replanning cadence will drift "
                "relative to the worker's decision cadence.",
                stacklevel=2,
            )

    def get_algorithm(self) -> Type:
        """Return CPSManager as the coordination 'algorithm' class."""
        return CPSManager

    def resolve_algorithm(self, name: str) -> Type:
        if name == "CPSManager":
            return CPSManager
        raise ValueError(
            f"Unknown CPS algorithm: '{name}'. "
            "The only valid algorithm for this experiment is 'CPSManager'."
        )


@dataclass
class CPSEnvKwargsConfig(EnvKwargsConfig):
    """gym.make() kwargs forwarded to the path-planning worker environment.

    Auto-generated CLI flags
    ------------------------
      --env-action-mode       str           Worker action mode ("hdg" | "wpt").
      --env-runways           list[str]     Active runway identifiers.
    """

    action_mode: str = "hdg"
    """Action mode passed to the worker environment."""

    runways: Optional[List[str]] = field(default_factory=lambda: None)
    """Subset of active runways.  ``None`` → use all runways registered
    in the environment."""

    def get_group_kwarg_name(self) -> Optional[str]:
        return "runways"


@dataclass
class CPSEnvConfig(EnvConfig):
    """Full environment configuration for the CPS coordination experiment.

    Auto-generated CLI flags (in addition to nested CPSEnvKwargsConfig)
    ------------------------
      --env-env-name      str   Experiment-path namespace (NOT a gym.make() id — see below).
      --env-group-key     str   Info-dict key for episode grouping.
      --env-success-key   str   Info-dict key for success signal.

    ``env_name`` is deliberately **not** ``"PathPlanningGoalEnv-v0"``: it feeds
    ``ExperimentConfig._build_paths()``, which creates
    ``./experiments/{env_name}/{algo}/{logs,models}/{run_id}/`` on every
    ``ExperimentConfig`` construction (i.e. every CPS eval run). Reusing the
    worker's own env id would scatter empty run-id folders into the same
    directory tree as the actual trained-worker checkpoints
    (``experiments/PathPlanningGoalEnv-v0/SAC/models/...``). The frozen
    worker's real gym id is hardcoded separately in
    ``coordination_baseline.py``'s ``make_env`` (the only place that needs it,
    and only for framework-compat single-episode preview — the real CPS eval
    loop uses ``MultiAgentPathPlanningGoalEnv`` directly, never ``gym.make``).
    """

    env_kwargs: CPSEnvKwargsConfig = field(default_factory=CPSEnvKwargsConfig)
    env_name: str = "CPSCoordination"
    group_key: str = "current_runway"
    success_key: str = "is_success"
