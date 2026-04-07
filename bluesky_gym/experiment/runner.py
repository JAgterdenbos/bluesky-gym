"""
bluesky_gym/experiment/runner.py
----------------------------------
run_experiment() — the single public entry point for user scripts.

This replaces the hand-written main.py / evaluate.py / enjoy.py scripts.
Users call it once and get a fully automatic CLI whose flags are derived
from their config dataclasses.

Usage in a user's project
--------------------------
  # my_project/main.py
  from bluesky_gym.experiment import run_experiment
  from my_project.experiment import PathPlanningExperiment

  if __name__ == "__main__":
      run_experiment(PathPlanningExperiment)

Generated CLI (example with PathPlanningExperiment)
-----------------------------------------------------
  python main.py --help

  # Train with defaults
  python main.py train

  # 500 k steps, custom learning rate, skip evaluation
  python main.py train --session-total-timesteps 500000 --model-learning-rate 1e-4 --session-no-do-evaluate

  # Evaluation only (load a previous run)
  python main.py evaluate --run-id 20260331_134059

  # Env overrides (from your EnvConfig + EnvKwargsConfig subclass fields)
  python main.py train --env-action-mode wpt

  # Load from YAML, override one field on top
  python main.py train --config experiments/my_config.yaml --session-total-timesteps 1000000

Commands
--------
  train           Train (and optionally evaluate) a new model (default).
  evaluate        Run detailed evaluation on a saved model.
  enjoy           Watch/record a saved model.
  generate-config Generate a default config.yaml for this experiment.
"""

from __future__ import annotations

import sys
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_experiment import BaseExperiment


#TODO: Check if first building the 
def run_experiment(experiment_cls: "Type[BaseExperiment]") -> None:
    """Build a CLI from experiment_cls's config dataclasses and run it."""
    from .config import ExperimentConfig
    from .evaluate import run_evaluate_cli
    from .enjoy import run_enjoy_cli
    from .plot import main as run_plot_cli
    from .compare_runs import main as run_compare_cli

    model_cls = experiment_cls.model_config_cls
    env_cls   = experiment_cls.env_config_cls

    # ── Build parser from dataclass fields ─────────────────────────────
    parser = ExperimentConfig._build_parser(
        model_config_cls=model_cls,
        env_config_cls=env_cls,
        description=f"Train / evaluate {experiment_cls.__name__}.",
    )

    # ── Subcommands ──────────────────────────────────────────────────────
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("train", help="Train (and optionally evaluate) a new model. [default]")
    subparsers.add_parser("evaluate", help="Run detailed evaluation on a saved model.")
    subparsers.add_parser("enjoy", help="Watch/record a saved model.")
    subparsers.add_parser("generate-config", help="Generate a default config.yaml for this experiment.")
    subparsers.add_parser("plot", help="Plot training curves or evaluation results.")
    subparsers.add_parser("compare", help="Compare training metrics across multiple runs.")

    # We use parse_known_args so the sub-CLIs can parse their own specific flags
    args, _ = parser.parse_known_args()

    # Default to train if no command is specified
    command = args.command or "train"

    # ── Delegate to the right sub-CLI ───────────────────────────────────
    if command == "generate-config":
        cfg = ExperimentConfig.from_args(args, model_cls, env_cls)
        # Override save path to drop it in the current working directory
        cfg.save_path = "."  #TODO: make this configurable
        cfg.save()
        print("✅ Default config saved to ./config.yaml")
        return

    if command == "evaluate":
        _reparse_and_run(run_evaluate_cli, experiment_cls, command)
        return

    if command == "enjoy":
        _reparse_and_run(run_enjoy_cli, experiment_cls, command)
        return
    
    if command == "plot":
        _reparse_and_run(run_plot_cli, experiment_cls, command)
        return
    
    if command == "compare":
        _reparse_and_run(run_compare_cli, experiment_cls, command)
        return

    # ── command == "train" ──────────────────────────────────────────────
    run_id = getattr(args, "run_id", None)

    if run_id:
        # Load a saved config and respect any explicit CLI overrides
        cfg = ExperimentConfig.load(run_id, model_cls, env_cls)
        cfg = _apply_cli_overrides(cfg, args, model_cls, env_cls)
    else:
        cfg = ExperimentConfig.from_args(args, model_cls, env_cls)

    experiment_cls(cfg).run()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reparse_and_run(sub_cli_fn, experiment_cls, command_name: str) -> None:
    """Strip the command from sys.argv and hand off to a sub-CLI function."""
    argv = [a for a in sys.argv[1:] if a != command_name]
    sys.argv = [sys.argv[0]] + argv
    sub_cli_fn(experiment_cls)


def _apply_cli_overrides(cfg, args, model_cls, env_cls):
    """Apply explicit CLI args on top of a loaded config."""
    from dataclasses import fields
    from .config import SessionConfig, _field_dest, _MISSING

    for section, dc_cls in [
        ("session", SessionConfig),
        ("model",   model_cls),
        ("env",     env_cls),
    ]:
        sub = getattr(cfg, section)
        for f in fields(dc_cls):
            dest = _field_dest(section, f.name)
            val  = getattr(args, dest, _MISSING)
            if val is _MISSING or val is None:
                continue
            
            setattr(sub, f.name, val)

    cfg._build_paths()
    return cfg