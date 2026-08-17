"""
cps_coordination/scripts/verify_metrics_sanity.py
------------------------------------------------------
Diagnostic (not a pass/fail gate) that independently cross-checks
throughput (Gamma/Gamma_r) from raw Parquet telemetry, and prints the
spawn-rate-vs-measured-rate arithmetic per combo -- the manual reasoning in
Finding 4 of
`.claude/plans/cps_metrics_audit_and_density_rescale_plan.md`, turned into
a reusable tool so it doesn't have to be re-derived by hand every time the
production dataset changes (25-ac now, 50-ac once that lands).

Goes in `scripts/` (not `testing/`) per this repo's convention -- report/
figure/diagnostic generators live in scripts/, only pass/fail gates live in
testing/.

Two checks per combo
---------------------
1. **Independent throughput cross-check.** Recomputes Gamma/Gamma_r
   straight from `cps_eval_aircraft.parquet` via a code path that does not
   call (or share any helper with) `cps_metrics_offline.recompute_metrics`
   or `cps_coordination.experiments.metrics._compute_throughput` -- same
   cross-check spirit as `c_sep_from_pairs` vs. `c_sep_from_landings` in
   `cps_metrics_offline.py`. A mismatch means one of the two Gamma
   implementations has a bug, not that either can be trusted by default.
2. **Spawn-rate-vs-measured-rate arithmetic.** Prints the raw
   `total_arrivals_per_episode`/`spawn_window_s` schedule rate, the
   measured Gamma, the mean per-episode landing span, and the ratio
   between spawn rate and Gamma -- then compares that ratio against the
   value predicted purely from (mean episode span / spawn window) *
   success_rate (Finding 4's dwell-time/concurrency-cap explanation for why
   Gamma comes in well under the raw spawn rate). Flags any combo where the
   two ratios diverge beyond `--ratio-tolerance`, since that would mean the
   gap is no longer explained by queuing/dwell time alone and needs fresh
   investigation.

Usage
-----
  python cps_coordination/scripts/verify_metrics_sanity.py \\
      --sweep-root <production_sweep_root> \\
      --total-arrivals-per-episode 50 --spawn-window-s 2400.0

Runtime: seconds (pure Parquet + pandas/numpy, no BlueSky/SB3 import) --
same offline-analysis pattern as `cps_metrics_offline.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cps_coordination.scripts.cps_metrics_offline import (
    load_recat_matrix,
    load_telemetry,
    recompute_metrics,
)
from cps_coordination.scripts.summarize_batch_sweep import _COMBO_RE, discover_combos


def independent_throughput(aircraft_df: pd.DataFrame) -> Dict[str, Any]:
    """Recompute Gamma/Gamma_r from scratch, independent of
    ``recompute_metrics``'s / ``_compute_throughput``'s code path.

    Deliberately reimplements the elapsed-time-denominator logic (sum of
    each episode's own max successful landing time, not a pooled max())
    inline rather than importing it, so this really is a second,
    independent derivation -- not the same function called twice.
    """
    successful = aircraft_df[aircraft_df["success"]]
    if successful.empty:
        return {
            "gamma": float("nan"),
            "gamma_std": float("nan"),
            "gamma_r": {},
            "total_time_s": float("nan"),
            "mean_episode_span_s": float("nan"),
        }

    episode_spans_s = successful.groupby("episode_id")["actual_landing_time"].max()
    total_time_s = float(episode_spans_s.sum())
    window_h = max(total_time_s / 3600.0, 1e-6)

    n_total = len(successful)
    gamma = n_total / window_h
    gamma_r = {
        str(rwy): len(group) / window_h
        for rwy, group in successful.groupby("runway_id")
    }

    # gamma_std: dispersion of each episode's OWN landings/hour ratio --
    # independent of (and a second derivation from) the same per-episode
    # partition metrics.py's own gamma_std uses, per this script's
    # from-scratch cross-check purpose.
    episode_counts = successful.groupby("episode_id").size()
    per_episode_gamma = (episode_counts / (episode_spans_s / 3600.0).clip(lower=1e-6)).to_numpy()
    gamma_std = (
        float(np.std(per_episode_gamma, ddof=1)) if len(per_episode_gamma) >= 2 else float("nan")
    )

    return {
        "gamma": gamma,
        "gamma_std": gamma_std,
        "gamma_r": gamma_r,
        "total_time_s": total_time_s,
        "mean_episode_span_s": float(episode_spans_s.mean()),
    }


def _fmt_gamma_r(gamma_r: Dict[str, float]) -> str:
    return ", ".join(f"{rwy}={v:.2f}" for rwy, v in sorted(gamma_r.items()))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Independent throughput cross-check + spawn-rate-vs-measured-rate "
                     "sanity arithmetic, per combo in a CPS eval sweep root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sweep-root", type=str, required=True,
                   help="Directory containing k<K>_<mode>_fw<FW>/ combo subdirectories.")
    p.add_argument("--total-arrivals-per-episode", type=float, required=True,
                   help="cps_scale_10k.yaml's total_arrivals_per_episode (spawn schedule "
                        "numerator) -- required, not defaulted: a silent wrong default here "
                        "would corrupt the spawn-rate-vs-measured-rate arithmetic this script "
                        "exists to check, exactly the class of mistake it's meant to catch.")
    p.add_argument("--spawn-window-s", type=float, default=2400.0,
                   help="cps_scale_10k.yaml's spawn_window_s (spawn schedule denominator).")
    p.add_argument("--sep-tolerance-s", type=float, default=5.0)
    p.add_argument("--rta-tolerance-s", type=float, default=60.0)
    p.add_argument("--ratio-tolerance", type=float, default=0.15,
                   help="Fractional tolerance between the measured spawn/Gamma ratio and the "
                        "ratio predicted from (mean episode span / spawn window) * success_rate "
                        "before flagging a combo as no longer explained by Finding 4's reasoning.")
    p.add_argument("--recat-config", type=str, default=None,
                   help="Path to cps_base.yaml (defaults to cps_metrics_offline's default).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    recat_matrix = (
        load_recat_matrix(args.recat_config) if args.recat_config else load_recat_matrix()
    )
    spawn_rate_ac_h = args.total_arrivals_per_episode / (args.spawn_window_s / 3600.0)
    spawn_window_h = args.spawn_window_s / 3600.0

    combo_dirs = discover_combos(args.sweep_root)
    if not combo_dirs:
        print(f"No k<K>_<mode>_fw<FW> combo directories found under {args.sweep_root}")
        raise SystemExit(1)

    print("\n--- verify_metrics_sanity: independent throughput cross-check ---")
    print(f"Sweep root       : {args.sweep_root}")
    print(f"Spawn schedule    : {args.total_arrivals_per_episode:.0f} arrivals / "
          f"{spawn_window_h:.4f}h window = {spawn_rate_ac_h:.2f} ac/h\n")

    any_gamma_mismatch = False
    any_ratio_flag = False

    for combo_dir in combo_dirs:
        combo_name = os.path.basename(combo_dir)
        m = _COMBO_RE.match(combo_name)
        assert m is not None

        aircraft_df, separation_df = load_telemetry(combo_dir)
        offline_metrics = recompute_metrics(
            aircraft_df, separation_df, recat_matrix,
            sep_tolerance_s=args.sep_tolerance_s, rta_tolerance_s=args.rta_tolerance_s,
        )
        indep = independent_throughput(aircraft_df)

        gamma_offline = float(offline_metrics["gamma"])
        gamma_offline_std = offline_metrics.get("gamma_std", "nan")
        gamma_indep = indep["gamma"]
        gamma_indep_std = indep["gamma_std"]
        # atol accounts for recompute_metrics's round(gamma, 4) -- this
        # comparison is otherwise between two independently-derived exact
        # values, so anything beyond rounding slack is a real mismatch.
        gamma_agree = np.isclose(gamma_offline, gamma_indep, rtol=0.0, atol=1e-4)
        if not gamma_agree:
            any_gamma_mismatch = True

        success_rate = float(offline_metrics["success_rate"])
        mean_span_h = indep["mean_episode_span_s"] / 3600.0
        measured_ratio = spawn_rate_ac_h / gamma_indep if gamma_indep else float("nan")
        predicted_ratio = (mean_span_h / spawn_window_h) * success_rate if spawn_window_h else float("nan")
        ratio_rel_diff = (
            abs(measured_ratio - predicted_ratio) / predicted_ratio
            if predicted_ratio not in (0.0, float("nan")) and not np.isnan(predicted_ratio)
            else float("nan")
        )
        ratio_flagged = (
            np.isnan(ratio_rel_diff) or ratio_rel_diff > args.ratio_tolerance
        )
        if ratio_flagged:
            any_ratio_flag = True

        print(f"[{combo_name}]")
        print(f"  gamma (offline recompute_metrics)      = {gamma_offline:.4f} ac/h "
              f"(std across episodes: {gamma_offline_std})")
        print(f"  gamma (independent from-scratch)       = {gamma_indep:.4f} ac/h "
              f"(std across episodes: {gamma_indep_std:.4f})  "
              f"{'AGREE' if gamma_agree else 'MISMATCH -- one of the two Gamma implementations has a bug'}")
        print(f"  gamma_r (independent)                  = {_fmt_gamma_r(indep['gamma_r'])}")
        print(f"  success_rate                           = {success_rate:.4f}")
        print(f"  mean episode span                      = {mean_span_h:.3f}h "
              f"({indep['mean_episode_span_s']:.0f}s)")
        print(f"  measured ratio  (spawn_rate / gamma)   = {measured_ratio:.3f}x")
        print(f"  predicted ratio (span/window * success)= {predicted_ratio:.3f}x")
        print(f"  ratio relative difference               = "
              f"{'n/a' if np.isnan(ratio_rel_diff) else f'{ratio_rel_diff:.1%}'}  "
              f"{'FLAGGED -- gap not explained by Finding 4 reasoning' if ratio_flagged else 'ok'}")
        print()

    print("--- Summary ---")
    print(f"Gamma cross-check : {'ALL AGREE' if not any_gamma_mismatch else 'MISMATCH DETECTED -- see above'}")
    print(f"Ratio sanity check: {'ALL EXPLAINED' if not any_ratio_flag else 'FLAGGED COMBO(S) -- see above'}")


if __name__ == "__main__":
    main()
