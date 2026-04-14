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
  python main.py evaluate --run-id 20260331_134059
  python main.py enjoy    --run-id 20260331_134059 --groups 27

  # Full flag list
  python main.py --help
"""

from path_planning.experiment import PathPlanningExperiment
from path_planning.registry import PathPlanningRegistry

from path_planning.rta import run_collection_cli, run_fit_and_plot_cli, run_analyse_cli

#TODO: factorise the rta commands out into a common rta command with subcommands

def main():
    custom_commands = {
        "collect-rta": (run_collection_cli, "Collect rta data step-by-step per successful episode."),
        "analyse-rta": (run_analyse_cli, "Analyse rta data."),
        "fit-and-plot-rta": (run_fit_and_plot_cli, "Fit and plot rta data."),
    }

    registry = PathPlanningRegistry()
    registry.run_experiment(PathPlanningExperiment, custom_commands)

if __name__ == "__main__":
    main()