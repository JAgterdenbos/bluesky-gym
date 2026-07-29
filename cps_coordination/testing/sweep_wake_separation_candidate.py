"""
cps_coordination/testing/sweep_wake_separation_candidate.py
---------------------------------------------------------------
Phase E (see .claude/eta_surrogate_accuracy_plan.md, Verification step 4):
run diagnose_success_rate.py's existing `wake_separation_scale` sweep against
an arbitrary serialized ETASurrogate candidate, not just the production
`eta_surrogate.pkl` main() always loads via `_build_surrogate()`.

Thin wrapper only -- reuses `run_wake_separation_sweep`/`_make_experiment`
from diagnose_success_rate.py verbatim; the only difference from `--sweep-
scales` on that script is which surrogate .pkl gets passed in.

Usage
-----
  python cps_coordination/testing/sweep_wake_separation_candidate.py \\
      --surrogate cps_coordination/models/eta_surrogate_combined_candidate.pkl \\
      --scales 1.0 0.75 0.5 0.25 0.1 \\
      --episodes 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cps_coordination.coordination.eta_surrogate import ETASurrogate
from cps_coordination.testing.diagnose_success_rate import (
    _make_experiment,
    run_wake_separation_sweep,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surrogate", type=Path, required=True)
    p.add_argument("--scales", type=float, nargs="+",
                   default=[1.0, 0.75, 0.5, 0.25, 0.1])
    p.add_argument("--episodes", type=int, default=10,
                   help="Episodes per sweep point (N=5 aircraft/episode).")
    args = p.parse_args()

    experiment = _make_experiment(k_cps=0, mode="static", runways=None)
    model = experiment.make_model(experiment._make_multi_agent_env(1))
    print(f"Frozen worker: {experiment.cfg.session.pretrained_model_path}")

    print(f"Loading surrogate from: {args.surrogate}")
    surrogate = ETASurrogate.load(args.surrogate)
    print(f"  {surrogate!r}")

    print(f"Sweep: scales={args.scales}, episodes/point={args.episodes}\n")
    run_wake_separation_sweep(model, surrogate, args.episodes, args.scales)


if __name__ == "__main__":
    main()
