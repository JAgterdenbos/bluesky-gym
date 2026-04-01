"""
main.py
-------
CLI entry point.  All experiment logic lives in experiment.py and config.py;
this file only wires argparse → ExperimentConfig → PathPlanningExperiment.

Usage examples
--------------
  # Default run (250 k steps, all runways for training)
  python main.py

  # 500 k steps, custom eval runways, skip evaluation
  python main.py --timesteps 500000 --eval-runways 27 18R --no-eval

  # Training only, custom learning rate
  python main.py --lr 1e-4 --no-eval

  # Evaluation only (load a previous run)
  python main.py --no-train --run-id 20260331_134059
"""

import argparse

from config import ExperimentConfig
from experiment import PathPlanningExperiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train / evaluate a PathPlanning RL agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Session ──────────────────────────────────────────────────────────
    p.add_argument(
        "--timesteps", type=int, default=None,
        metavar="N",
        help="Total environment steps for training.",
    )
    p.add_argument(
        "--train-runways", nargs="+", default=None,
        metavar="RWY",
        help="Runway IDs to use during training (e.g. 27 18R 06). "
             "Omit to use all runways.",
    )
    p.add_argument(
        "--eval-runways", nargs="+", default=None,
        metavar="RWY",
        help="Runway IDs to use during evaluation.",
    )
    p.add_argument(
        "--eval-episodes", type=int, default=None,
        metavar="N",
        help="Number of episodes to run during evaluation.",
    )
    p.add_argument(
        "--no-train", action="store_true",
        help="Skip training (useful for eval-only runs or debugging).",
    )
    p.add_argument(
        "--no-eval", action="store_true",
        help="Skip post-training evaluation.",
    )

    # ── Model ────────────────────────────────────────────────────────────
    p.add_argument(
        "--lr", type=float, default=None,
        metavar="LR",
        help="Learning rate override.",
    )

    # ── Environment ──────────────────────────────────────────────────────
    p.add_argument(
        "--action-mode", choices=["hdg", "wpt"], default=None,
        help="Action mode passed to the environment.",
    )

    # ── Misc ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--run-id", type=str, default=None,
        metavar="ID",
        help="Existing run ID to load config from (used with --no-train).",
    )

    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # If a run-id is given and we are not training, load the saved config
    # so all paths are reconstructed correctly.
    if args.run_id and args.no_train:
        cfg = ExperimentConfig.load(args.run_id)
        # Still respect any explicit CLI overrides on top of the loaded config
        if args.eval_runways:
            cfg.session.eval_runways = args.eval_runways
        if args.eval_episodes:
            cfg.session.eval_episodes = args.eval_episodes
        cfg.session.do_train    = False
        cfg.session.do_evaluate = not args.no_eval
    else:
        cfg = ExperimentConfig.from_args(args)

    experiment = PathPlanningExperiment(cfg)
    experiment.run()


if __name__ == "__main__":
    main()