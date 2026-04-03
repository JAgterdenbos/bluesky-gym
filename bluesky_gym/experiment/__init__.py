"""
bluesky_gym/experiment/__init__.py
------------------------------------
Public API for the experiment framework.

Import surface
--------------
  from bluesky_gym.experiment import (
      # Entry point
      run_experiment,

      # Base classes to subclass
      BaseExperiment,
      EnvKwargsConfig,
      EnvConfig,
      ModelConfig,
      SessionConfig,
      ExperimentConfig,

      # Evaluation extension point
      MetricExtractor,

      # Plotting extension point
      plot_training_curves,
      plot_eval_summary,
      plot_eval_episodes,

      # Standalone CLIs (rarely needed directly)
      run_evaluate_cli,
      run_enjoy_cli,
      enjoy,
      compare_runs_main,
      plot_main
  )
"""

from .runner          import run_experiment
from .base_experiment import BaseExperiment
from .config          import (
    EnvKwargsConfig,
    EnvConfig,
    ModelConfig,
    SessionConfig,
    ExperimentConfig,
)
from .evaluate        import MetricExtractor, run_evaluate_cli
from .enjoy           import run_enjoy_cli, enjoy
from .compare_runs    import main as compare_runs_main
from .plot            import (
    main as plot_main, 
    plot_training_curves,
    plot_eval_summary, 
    plot_eval_episodes
)

__all__ = [
    "run_experiment",
    "BaseExperiment",
    "EnvKwargsConfig",
    "EnvConfig",
    "ModelConfig",
    "SessionConfig",
    "ExperimentConfig",
    "MetricExtractor",
    "run_evaluate_cli",
    "run_enjoy_cli",
    "enjoy",
    "compare_runs_main",
    "plot_main",
    "plot_training_curves",
    "plot_eval_summary",
    "plot_eval_episodes",
]