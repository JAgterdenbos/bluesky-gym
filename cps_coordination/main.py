"""
cps_coordination/main.py
-------------------------
CLI entry point for the CPS coordination package.

Configuration dataclasses live in :mod:`cps_coordination.experiments.config`
and are re-exported here for backwards compatibility.

CLI usage
---------
  # Evaluate with defaults
  python -m cps_coordination train

  # Load from YAML, override one field
  python -m cps_coordination train --config configs/cps_base.yaml --model-k-cps 5

  # Override CPS model fields
  python -m cps_coordination train --model-k-cps 3
  python -m cps_coordination train --model-delta-t-plan 120
  python -m cps_coordination train --model-runway-assignment-mode static

  # Override env fields
  python -m cps_coordination train --env-v-app 135.0
  python -m cps_coordination train --env-runways 27 18R

  # Point at a pre-trained worker policy
  python -m cps_coordination train --session-pretrained-run-id 20260301_120000

  # Evaluate / watch a saved CPS run
  python -m cps_coordination evaluate --run-id 20260301_120000
  python -m cps_coordination enjoy    --run-id 20260301_120000

  # Full flag list
  python -m cps_coordination --help
"""

# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Primary CLI entry point for cps_coordination experiments.

    Delegates to the framework runner via a :class:`CPSCoordinationRegistry`
    so the standard ``train / evaluate / enjoy / registry`` sub-commands are
    all available out of the box.
    """
    from cps_coordination.experiments.coordination_baseline import (
        CPSCoordinationExperiment,
        CPSCoordinationRegistry,
    )

    registry = CPSCoordinationRegistry()
    registry.run_experiment(CPSCoordinationExperiment)


if __name__ == "__main__":
    main()
