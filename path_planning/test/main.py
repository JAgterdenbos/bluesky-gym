"""
path_planning/main.py
----------------------
Entry point for all PathPlanning experiment actions.

Usage
-----
  # Train with dataclass defaults
  python main.py

  # Load from YAML config, override one field on top
  python main.py --config configs/her_wpt.yaml --session-total-timesteps 500000

  # Session overrides
  python main.py --session-total-timesteps 500000
  python main.py --session-train-groups 27 18R 06
  python main.py --session-eval-groups 27 18R
  python main.py --session-eval-episodes 20
  python main.py --session-no-do-evaluate
  python main.py --session-track-training-evals

  # Model overrides (PathPlanningModelConfig fields)
  python main.py --model-learning-rate 1e-4
  python main.py --model-no-use-her
  python main.py --model-her-n-sampled-goal 8
  python main.py --model-her-goal-selection-strategy future
  python main.py --model-algorithm TD3

  # Env overrides (PathPlanningEnvConfig + PathPlanningEnvKwargsConfig fields)
  python main.py --env-action-mode wpt
  python main.py --env-use-rta
  python main.py --env-runways 27 18R

  # Evaluate / watch a saved model
  python main.py --mode evaluate --run-id 20260331_134059
  python main.py --mode enjoy    --run-id 20260331_134059 --groups 27

  # Full flag list
  python main.py --help
"""

from bluesky_gym.experiment import run_experiment
from experiment import PathPlanningExperiment

if __name__ == "__main__":
    run_experiment(PathPlanningExperiment)