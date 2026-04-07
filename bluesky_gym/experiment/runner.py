from __future__ import annotations

import os
import sys
import textwrap
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base_experiment import BaseExperiment


def _print_global_help(experiment_cls: "Type[BaseExperiment]") -> None:
    """Prints a clean, top-level CLI menu (bypassing argparse clutter)."""
    script_name = os.path.basename(sys.argv[0])
    help_text = f"""\
    Usage: python {script_name} <command> [options]

    CLI for {experiment_cls.__name__}.

    Commands:
      train            Train (and optionally evaluate) a new model. [default]
      evaluate         Run detailed evaluation on a saved model.
      enjoy            Watch or record a saved model.
      plot             Plot training curves or evaluation results.
      compare          Compare training metrics across multiple runs.
      generate-config  Generate a default config.yaml for this experiment.

    Type 'python {script_name} <command> --help' for details on a specific command.
    """
    print(textwrap.dedent(help_text))


def run_experiment(experiment_cls: "Type[BaseExperiment]") -> None:
    """Single entry point for the CLI."""
    from .config import ExperimentConfig
    from .evaluate import run_evaluate_cli
    from .enjoy import run_enjoy_cli
    from .plot import run_plot_cli
    from .compare_runs import run_compare_cli

    known_commands = {"train", "evaluate", "enjoy", "generate-config", "plot", "compare"}
    
    # ── 1. Find the Command ──────────────────────────────────────────────
    command = None
    for arg in sys.argv[1:]:
        if arg in known_commands:
            command = arg
            break

    # ── 2. Global Help Intercept ─────────────────────────────────────────
    # If no command is given, but --help is requested, show our clean global menu
    if command is None and any(arg in ["-h", "--help"] for arg in sys.argv):
        _print_global_help(experiment_cls)
        sys.exit(0)

    # Default to 'train' if nothing is specified
    if command is None:
        command = "train"

    # ── 3. Sub-script Dispatch ───────────────────────────────────────────
    # Hand off execution so sub-scripts can use their own argparse natively
    if command == "evaluate":
        return _reparse_and_run(run_evaluate_cli, experiment_cls, command)
    if command == "enjoy":
        return _reparse_and_run(run_enjoy_cli, experiment_cls, command)
    if command == "plot":
        return _reparse_and_run(run_plot_cli, experiment_cls, command)
    if command == "compare":
        return _reparse_and_run(run_compare_cli, experiment_cls, command)

    # ── 4. Train & Config Parsers ────────────────────────────────────────
    model_cls = experiment_cls.model_config_cls
    env_cls   = experiment_cls.env_config_cls

    parser = ExperimentConfig._build_parser(
        model_config_cls=model_cls,
        env_config_cls=env_cls,
        description=f"Train / evaluate {experiment_cls.__name__}.\n\n(Run 'python {os.path.basename(sys.argv[0])} --help' to see all commands)",
    )

    # Remove the command from argv so standard argparse doesn't trip on it
    argv = sys.argv[:]
    for i in range(1, len(argv)):
        if argv[i] == command:
            argv.pop(i)
            break
            
    # If the user typed `python main.py train --help`, the parser catches it right here!
    args, _ = parser.parse_known_args(argv[1:])

    # ── 5. Execute Core Commands ─────────────────────────────────────────
    if command == "generate-config":
        cfg = ExperimentConfig.from_args(args, model_cls, env_cls)
        cfg.save_path = "."  
        cfg.save()
        print("✅ Default config saved to ./config.yaml")
        return

    # Train
    run_id = getattr(args, "run_id", None)
    if run_id:
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
    argv = sys.argv[:]
    # Remove only the FIRST occurrence of the command to protect argument values
    for i in range(1, len(argv)):
        if argv[i] == command_name:
            argv.pop(i)
            break
    sys.argv = argv
    sub_cli_fn(experiment_cls)

def _apply_cli_overrides(cfg, args, model_cls, env_cls):
    """Apply explicit CLI args on top of a loaded config."""
    from dataclasses import fields
    from .config import SessionConfig, _field_dest, _MISSING

    for section, dc_cls in [("session", SessionConfig), ("model", model_cls), ("env", env_cls)]:
        sub = getattr(cfg, section)
        for f in fields(dc_cls):
            dest = _field_dest(section, f.name)
            val  = getattr(args, dest, _MISSING)
            if val is not _MISSING and val is not None:
                setattr(sub, f.name, val)

    cfg._build_paths()
    return cfg