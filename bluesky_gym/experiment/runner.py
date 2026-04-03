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
  python main.py

  # 500 k steps, custom learning rate, skip evaluation
  python main.py --session-total-timesteps 500000 --model-learning-rate 1e-4 --session-no-do-evaluate

  # Evaluation only (load a previous run)
  python main.py --run-id 20260331_134059 --session-no-do-train

  # Env overrides (from your EnvConfig + EnvKwargsConfig subclass fields)
  python main.py --env-action-mode wpt
  python main.py --env-use-rta
  python main.py --env-runways 27 18R

  # Load from YAML, override one field on top
  python main.py --config experiments/my_config.yaml --session-total-timesteps 1000000

Mode flags (added on top of dataclass fields)
----------------------------------------------
  --run-id ID         Load a saved config (used for eval/enjoy).
  --config PATH       Load a YAML config file before CLI overrides.
  --mode {train,evaluate,enjoy}
                      Default: train.  Runs the corresponding lifecycle.
"""

from __future__ import annotations

import sys
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_experiment import BaseExperiment


def run_experiment(experiment_cls: "Type[BaseExperiment]") -> None:
    """Build a CLI from experiment_cls's config dataclasses and run it.

    Parameters
    ----------
    experiment_cls : Type[BaseExperiment]
        Your experiment subclass.  Its class variables
        ``model_config_cls`` and ``env_config_cls`` determine which
        dataclass fields are exposed as CLI flags.
    """
    from .config import ExperimentConfig
    from .evaluate import run_evaluate_cli
    from .enjoy import run_enjoy_cli

    model_cls = experiment_cls.model_config_cls
    env_cls   = experiment_cls.env_config_cls

    # ── Build parser from dataclass fields ─────────────────────────────
    parser = ExperimentConfig._build_parser(
        model_config_cls=model_cls,
        env_config_cls=env_cls,
        description=f"Train / evaluate {experiment_cls.__name__}.",
    )

    # ── Mode flag ───────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "enjoy"],
        default="train",
        help=(
            "train   — train (and optionally evaluate) a new model.\n"
            "evaluate — run detailed evaluation on a saved model.\n"
            "enjoy   — watch/record a saved model."
        ),
    )

    args = parser.parse_args()

    # ── Delegate to the right sub-CLI ───────────────────────────────────
    if args.mode == "evaluate":
        # evaluate and enjoy have their own minimal parsers; re-parse
        _reparse_and_run(run_evaluate_cli, experiment_cls)
        return

    if args.mode == "enjoy":
        _reparse_and_run(run_enjoy_cli, experiment_cls)
        return

    # ── mode == "train" (default) ───────────────────────────────────────
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

def _reparse_and_run(sub_cli_fn, experiment_cls) -> None:
    """Strip --mode from sys.argv and hand off to a sub-CLI function."""
    # Remove --mode <value> so the sub-parser doesn't see it
    argv = [a for a in sys.argv[1:] if a not in ("--mode", "train", "evaluate", "enjoy")]
    sys.argv = [sys.argv[0]] + argv
    sub_cli_fn(experiment_cls)


def _apply_cli_overrides(cfg, args, model_cls, env_cls):
    """Apply explicit CLI args on top of a loaded config.

    Called when --run-id is given alongside other flags so users can do:
        python main.py --run-id 20260331 --session-eval-episodes 50
    """
    from dataclasses import fields
    # Removed ALGO_MAP from the imports
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