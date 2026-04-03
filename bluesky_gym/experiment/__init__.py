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

      # Algorithm registry
      ALGO_MAP,
      register_algorithm,

      # Evaluation extension point
      MetricExtractor,

      # Standalone CLIs (rarely needed directly)
      run_evaluate_cli,
      run_enjoy_cli,
      enjoy,
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
]