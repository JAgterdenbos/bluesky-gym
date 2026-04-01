"""
experiment.py
-------------
Experiment lifecycle abstraction.

Classes
-------
  BaseExperiment          - abstract base; owns train / evaluate / run
  PathPlanningExperiment  - concrete implementation for PathPlanningGoalEnv

Why a class hierarchy?
----------------------
The original main.py had free functions (train, evaluate) that touched
config internals directly.  Wrapping these in a class means:
  - Each experiment type can override only what it needs to change.
  - The `run()` entry point is identical for every experiment, so main.py
    stays a thin CLI shim.
  - make_env / make_model / get_callbacks can be overridden independently
    (e.g. swap in a vectorised env without touching evaluation logic).
"""

from __future__ import annotations

import abc
from typing import Optional, TYPE_CHECKING

import gymnasium as gym
import bluesky_gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from callbacks import get_callbacks
from config import ExperimentConfig

 
if TYPE_CHECKING:
    from evaluate import MetricExtractor


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseExperiment(abc.ABC):
    """Abstract experiment base class.

    Subclasses must implement:
      - make_env()   - returns a (optionally wrapped) Gymnasium env
      - make_model() - returns an SB3 model ready for .learn()

    Subclasses may override:
      - get_callbacks() - returns a CallbackList for .learn()
      - evaluate()      - post-training evaluation loop
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abc.abstractmethod
    def make_env(self, env_kwargs: dict, render_mode: Optional[str] = None) -> gym.Env:
        """Construct and return a single (wrapped) environment instance."""
        ...

    @abc.abstractmethod
    def make_model(self, env: gym.Env):
        """Construct and return an SB3 model bound to `env`."""
        ...

    # ------------------------------------------------------------------ #
    # Metric extraction (override per env)                               #
    # ------------------------------------------------------------------ #
 
    @classmethod
    def metric_extractor(cls) -> "MetricExtractor | None":
        """Return a MetricExtractor for this experiment type, or None.
 
        Override in subclasses to declare which extra fields from the env's
        info dict should be collected and aggregated by evaluate.py.
        Returning None means only the built-in metrics (is_success,
        total_reward) are collected.
        """
        return None

    # ------------------------------------------------------------------ #
    # Overridable defaults                                               #
    # ------------------------------------------------------------------ #

    def get_callbacks(self, eval_env: gym.Env):
        """Return a CallbackList.  Delegates to callbacks.get_callbacks()
        by default; override to add experiment-specific callbacks."""
        callbacks, _ = get_callbacks(self.cfg, eval_env)
        return callbacks

    def evaluate(self, model) -> dict[str, list[bool]]:
        """Run `cfg.session.eval_episodes` episodes and print per-runway
        success rates.  Returns the raw results dict for programmatic use.

        This implementation works for any env that puts 'current_runway'
        and 'is_success' in its info dict.  Override if your env differs.
        """
        cfg = self.cfg
        print(f"\n📊 Evaluating model from {cfg.save_path}/final_model.zip …")

        eval_env = self.make_env(cfg.eval_env_kwargs, render_mode="human")
        model.set_env(eval_env)

        # Collect results keyed by runway (or whatever the env calls it)
        results: dict[str, list[bool]] = {}

        for ep in range(cfg.session.eval_episodes):
            done = truncated = False
            obs, info = eval_env.reset()
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = eval_env.step(action)

            rwy = info.get("current_runway", "unknown")
            win = info.get("is_success", False)
            results.setdefault(rwy, []).append(win)

        eval_env.close()

        print("\nEvaluation Results:")
        for rwy, outcomes in sorted(results.items()):
            n    = len(outcomes)
            wins = sum(outcomes)
            print(f"  Runway {rwy}: {wins}/{n}  ({wins/n:.1%})")

        return results
    
    def train_log_interval(self, total_timesteps: int) -> int:
        """Determines how often to log training progress.  Defaults to 1% of total timesteps, with a minimum of 1,000."""
        return max(1_000, total_timesteps // 100)

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Full training loop: build envs → model → learn → save."""
        cfg = self.cfg
        log_interval = self.train_log_interval(cfg.session.total_timesteps)

        train_env = self.make_env(cfg.train_env_kwargs)
        eval_env  = self.make_env(cfg.eval_env_kwargs)

        model     = self.make_model(train_env)
        callbacks = self.get_callbacks(eval_env)

        print(f"\n🏋️  Training for {cfg.session.total_timesteps:,} steps …")
        model.learn(
            total_timesteps=cfg.session.total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
        )

        model.save(f"{cfg.save_path}/final_model")
        cfg.save()   # persist config.json alongside the model
        print(f"✅ Training complete.  Model saved → {cfg.save_path}")

        train_env.close()
        eval_env.close()

        self._model = model   # cache for the evaluate() call in run()

    def run(self) -> None:
        """Convenience entry point: train (optional) → evaluate (optional)."""
        bluesky_gym.register_envs()
        cfg = self.cfg
        print(f"▶️  Experiment {cfg.run_id}  |  env={cfg.session.env_name}"
              f"  |  algo={cfg.model.algorithm.__name__}")

        model = None

        if cfg.session.do_train:
            self.train()
            model = getattr(self, "_model", None)

        if cfg.session.do_evaluate:
            if model is None:
                # Load from disk when we skipped training (e.g. --no-train flag)
                model = cfg.model.algorithm.load(f"{cfg.save_path}/final_model")
            self.evaluate(model)


# ---------------------------------------------------------------------------
# Path Planning (concrete)
# ---------------------------------------------------------------------------

class PathPlanningExperiment(BaseExperiment):
    """Experiment for PathPlanningGoalEnv-v0.

    Adds HER replay buffer and MultiInputPolicy to the SB3 model.
    Everything else (logging, eval loop, CLI) is inherited from
    BaseExperiment.
    """

    @classmethod
    def metric_extractor(cls) -> "MetricExtractor":
        """PathPlanning-specific metrics extracted from the env's info dict.
 
        flight_time_min - elapsed sim time in minutes (nan on failure so the
                          aggregate reflects successful episodes only)
        path_length_km  - total path length recovered from the reward signal
        noise_reward    - cumulative population-exposure reward component
        """
        from evaluate import MetricExtractor
 
        def _path_length_km(info: dict, _ok: bool) -> float:
            plw      = info.get("path_length_weight", 0.0)
            path_rew = info.get("average_path_rew",   0.0)
            if plw == 0:
                return float("nan")
            return float((path_rew / plw) * 1.852)   # NM → km
         
        return MetricExtractor(
            extractors={
                # Runway success is redundant with is_success but allows per-runway breakdowns in the eval summary
                "runway_success": lambda info, ok: 1.0 if ok else 0.0,
                # Only record flight time for successful episodes
                "flight_time": lambda info, ok: (
                    info.get("sim_time", float("nan")) / 60 if ok else float("nan")
                ),
                "path_length_km":  _path_length_km,
                "noise_reward":    lambda info, _ok: float(
                    info.get("average_noise_rew", float("nan"))
                ),
            },
            # flight_time uses nanmean implicitly (nan on failures is ignored)
            # Add explicit overrides here if you need e.g. np.nanstd for one metric
            aggregators={

            },
            display=["flight_time", "path_length_km", "noise_reward"],
        )

    def make_env(
        self,
        env_kwargs: dict,
        render_mode: Optional[str] = None,
    ) -> gym.Env:
        """Create a Monitor-wrapped PathPlanningGoalEnv."""
        env = gym.make(
            self.cfg.session.env_name,
            render_mode=render_mode,
            **env_kwargs,
        )
        return Monitor(env)

    def make_model(self, env: gym.Env):
        """SAC + HER with the architecture from ModelConfig."""
        cfg   = self.cfg
        mcfg  = cfg.model
        return mcfg.algorithm(
            "MultiInputPolicy",
            env,
            learning_rate=mcfg.learning_rate,
            policy_kwargs=mcfg.policy_kwargs,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=mcfg.her_n_sampled_goal,
                goal_selection_strategy=mcfg.her_goal_selection_strategy,
            ),
            verbose=mcfg.verbose,
        )