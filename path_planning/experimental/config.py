"""
config.py
---------
Hierarchical configuration for RL experiments.

Structure
---------
  EnvKwargsConfig   - extra kwargs forwarded verbatim to gym.make()
  ModelConfig       - algorithm, architecture, HER settings
  SessionConfig     - env id, timesteps, runway splits, eval settings
  ExperimentConfig  - root object; owns paths, save/load, and CLI parsing

Design notes
------------
- `ExperimentConfig` is the single object passed around everywhere.
- Paths are derived lazily in __post_init__ so they are consistent.
- save() / load() round-trip through JSON so enjoy.py can reconstruct
  the exact config from a run_id without any hardcoded defaults.
- from_args() wires argparse directly onto the dataclass fields so
  main.py stays a thin entry point.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from stable_baselines3 import SAC

# ---------------------------------------------------------------------------
# Extra environment kwargs
# ---------------------------------------------------------------------------

@dataclass
class EnvKwargsConfig:
    """Everything forwarded verbatim to gym.make() beyond env_id.

    Having a dedicated dataclass (rather than a raw dict) means every
    supported knob is documented and type-checked here rather than
    scattered across make_env() calls.
    """
    action_mode: str         = "hdg"
    use_rta:     bool        = False
    runways:     Optional[List[str]] = None   # None → all runways


# ---------------------------------------------------------------------------
# Model / algorithm
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Algorithm choice, network architecture, and HER hyper-parameters."""

    algorithm: Type = SAC

    # Network architecture - shared format understood by SB3's policy_kwargs
    net_arch: Dict[str, List[int]] = field(
        default_factory=lambda: dict(pi=[256, 256, 256], qf=[256, 256, 256])
    )
    learning_rate: float = 3e-4

    # HER-specific
    her_n_sampled_goal: int        = 4
    her_goal_selection_strategy: str = "future"

    # Convenience: verbose level passed to model.learn()
    verbose: int = 1

    @property
    def policy_kwargs(self) -> Dict[str, Any]:
        return dict(net_arch=self.net_arch)


# ---------------------------------------------------------------------------
# Session (what to run)
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Controls which environment to use, how long to train, and what to eval."""

    env_name: str = "PathPlanningGoalEnv-v0"

    total_timesteps: int = 250_000

    # Runway splits — None means "use all available runways"
    train_runways: Optional[List[str]] = field(default_factory=lambda: None)
    eval_runways:  Optional[List[str]] = field(default_factory=lambda: ["27", "18R"])

    eval_episodes: int  = 10
    do_train:      bool = True
    do_evaluate:   bool = True

    # How often EvalCallback runs (in env steps)
    eval_freq: int = 5_000

# ---------------------------------------------------------------------------
# Root experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Root configuration object.  One instance per experiment run.

    Path layout
    -----------
    ./experiments/<env_name>/<algo>/
        logs/<run_id>/
        models/<run_id>/
            final_model.zip
            checkpoint_model.zip
            config.json          ← written by save(); read by load()
    """

    model:   ModelConfig   = field(default_factory=ModelConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    env:     EnvKwargsConfig = field(default_factory=EnvKwargsConfig)

    # Generated once; stable for the lifetime of the run
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # ------------------------------------------------------------------ #
    # Derived paths (set in __post_init__)                                #
    # ------------------------------------------------------------------ #
    # These are NOT dataclass fields (no type annotation) so they are not
    # included in asdict() / JSON serialisation automatically.  We expose
    # them as plain attributes instead.

    def __post_init__(self) -> None:
        self._build_paths()

    def _build_paths(self) -> None:
        base = (
            f"./experiments"
            f"/{self.session.env_name}"
            f"/{self.model.algorithm.__name__}"
        )
        self.log_dir   = f"{base}/logs/{self.run_id}/"
        self.save_path = f"{base}/models/{self.run_id}/"
        self.save_freq = max(5_000, self.session.total_timesteps // 100)
        os.makedirs(self.log_dir,   exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def train_env_kwargs(self) -> Dict[str, Any]:
        """kwargs for the training environment (uses train_runways)."""
        return dict(
            action_mode=self.env.action_mode,
            use_rta=self.env.use_rta,
            runways=self.session.train_runways,
        )

    @property
    def eval_env_kwargs(self) -> Dict[str, Any]:
        """kwargs for the evaluation environment (uses eval_runways)."""
        return dict(
            action_mode=self.env.action_mode,
            use_rta=self.env.use_rta,
            runways=self.session.eval_runways,
        )

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def _to_dict(self) -> dict:
        """Serialise to a plain dict suitable for JSON.

        `algorithm` is a class object — we store its name and reconstruct
        on load.  Everything else is a primitive or a nested dict/list.
        """
        d = asdict(self)
        # Replace the class object with its string name
        d["model"]["algorithm"] = self.model.algorithm.__name__
        return d

    def save(self) -> None:
        """Write config.json next to the model files."""
        path = os.path.join(self.save_path, "config.json")
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    @classmethod
    def load(cls, run_id: str) -> "ExperimentConfig":
        """Reconstruct an ExperimentConfig from a previously saved run.

        Searches for the config.json under all known algorithm/env
        combinations.  Raises FileNotFoundError with a helpful message
        if the run cannot be located.
        """
        import glob
        pattern = f"./experiments/*/*/models/{run_id}/config.json"
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(
                f"No config.json found for run_id='{run_id}'.\n"
                f"Searched: {pattern}"
            )
        with open(matches[0]) as f:
            d = json.load(f)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> "ExperimentConfig":
        """Reconstruct from the dict produced by _to_dict()."""
        from stable_baselines3 import SAC # noqa: F401

        _ALGO_MAP: Dict[str, Type] = {
            "SAC": SAC,
            # Extend here as you add more algorithms
        }
        algo_name = d["model"].pop("algorithm", "SAC")
        algo_cls  = _ALGO_MAP.get(algo_name, SAC)

        model_cfg   = ModelConfig(algorithm=algo_cls, **d["model"])
        session_cfg = SessionConfig(**d["session"])
        env_cfg     = EnvKwargsConfig(**d["env"])

        # Bypass __post_init__ path creation with the original run_id
        obj = cls.__new__(cls)
        object.__setattr__(obj, "model",   model_cfg)
        object.__setattr__(obj, "session", session_cfg)
        object.__setattr__(obj, "env",     env_cfg)
        object.__setattr__(obj, "run_id",  d["run_id"])
        obj.__post_init__()   # rebuild paths with the loaded run_id
        return obj

    # ------------------------------------------------------------------ #
    # CLI / argparse                                                       #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Build an ExperimentConfig from a parsed argparse Namespace.

        Only overrides fields that were explicitly passed on the command
        line; everything else keeps its dataclass default.  This keeps
        main.py free of config logic.
        """
        cfg = cls()

        # Session overrides
        if args.timesteps is not None:
            cfg.session.total_timesteps = args.timesteps
        if args.train_runways is not None:
            cfg.session.train_runways = args.train_runways
        if args.eval_runways is not None:
            cfg.session.eval_runways = args.eval_runways
        if args.eval_episodes is not None:
            cfg.session.eval_episodes = args.eval_episodes
        if args.no_train:
            cfg.session.do_train = False
        if args.no_eval:
            cfg.session.do_evaluate = False

        # Model overrides
        if args.lr is not None:
            cfg.model.learning_rate = args.lr

        # Env overrides
        if args.action_mode is not None:
            cfg.env.action_mode = args.action_mode

        # Paths must be rebuilt after any change that affects them
        cfg._build_paths()
        return cfg