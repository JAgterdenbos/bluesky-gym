"""
scratchpad/phase5a_thrash_analysis.py
--------------------------------------------
Phase 5a of .claude/plans/stall_rate_investigation.md: offline thrash analysis
on the already-collected phase4_detector_trace_cap50.parquet (no new runs).

Tests hypothesis 1 (argmin thrashing in _apply_k_cps_constraint's per-cycle
greedy reordering, same bug class as the already-fixed _assign_runways_dynamic
thrashing): under stable circumstances (same runway, same competitor set),
tta should be monotonically non-decreasing (RECAT separation only ever adds
delay). A tta DECREASE while the aircraft's own runway_id is unchanged means
either its position in the k-CPS ordering shifted, or some other aircraft on
that runway was reassigned/reordered -- both within scope of what a k-CPS
stability fix would address. Splits decreases into "runway changed this step"
(attributable to already-fixed dynamic reassignment) vs "runway unchanged"
(attributable to k-CPS position-ordering volatility), and correlates the
same-runway decrease rate per flight against eventual failure.

Usage: uv run python scratchpad/phase5a_thrash_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROOT = "experiments/cps_eval/capacity_sweep_50ac_surrogate_fix"


def main() -> None:
    trace = pd.read_parquet(f"{ROOT}/phase4_detector_trace_cap50.parquet")
    outcome = pd.read_parquet(f"{ROOT}/phase4_detector_outcome_cap50.parquet")

    # Same acid-reuse-within-episode disambiguation as phase4_detector_comparison.py.
    outcome = outcome.copy()
    outcome["local_rank"] = outcome.groupby(["episode_id", "acid"]).cumcount()
    outcome_idx = outcome.set_index(["episode_id", "acid", "local_rank"])

    trace = trace.sort_values(["episode_id", "acid", "spawn_time", "t"])
    trace["local_rank"] = (
        trace.groupby(["episode_id", "acid"])["spawn_time"].rank(method="dense").astype(int) - 1
    )

    rows = []
    for (ep, acid, rank), g in trace.groupby(["episode_id", "acid", "local_rank"]):
        g = g.sort_values("t")
        tta = g["tta"].to_numpy(dtype=float)
        rwy = g["runway_id"].to_numpy()
        if len(tta) < 3 or np.isnan(tta).all():
            continue

        valid = ~np.isnan(tta)
        n_valid_steps = int(valid.sum()) - 1  # number of consecutive valid pairs
        if n_valid_steps <= 0:
            continue

        n_decrease_same_rwy = 0
        n_decrease_rwy_changed = 0
        max_decrease = 0.0
        for i in range(1, len(tta)):
            if not (valid[i] and valid[i - 1]):
                continue
            delta = tta[i] - tta[i - 1]
            if delta < -1e-6:  # a real decrease, not float noise
                if rwy[i] == rwy[i - 1]:
                    n_decrease_same_rwy += 1
                else:
                    n_decrease_rwy_changed += 1
                max_decrease = max(max_decrease, -delta)

        key = (ep, acid, rank)
        failed = key in outcome_idx.index and not bool(outcome_idx.loc[key, "success"])
        rows.append({
            "episode_id": ep, "acid": acid, "local_rank": rank,
            "n_steps": n_valid_steps,
            "n_decrease_same_rwy": n_decrease_same_rwy,
            "n_decrease_rwy_changed": n_decrease_rwy_changed,
            "frac_decrease_same_rwy": n_decrease_same_rwy / n_valid_steps,
            "max_decrease_s": max_decrease,
            "failed": failed,
        })

    df = pd.DataFrame(rows)
    print(f"n flights analyzed: {len(df)}")
    print(f"n failed: {df['failed'].sum()}  n succeeded: {(~df['failed']).sum()}")
    print()
    print("=== same-runway tta decreases (candidate k-CPS thrash signal) ===")
    for label, mask in [("failed", df["failed"]), ("succeeded", ~df["failed"])]:
        sub = df[mask]
        print(
            f"{label:>10} n={len(sub):4d}  "
            f"mean_n_decrease_same_rwy={sub['n_decrease_same_rwy'].mean():.2f}  "
            f"frac_flights_with_any_same_rwy_decrease={(sub['n_decrease_same_rwy']>0).mean():.3f}  "
            f"mean_frac_decrease_same_rwy={sub['frac_decrease_same_rwy'].mean():.4f}  "
            f"mean_max_decrease_s={sub['max_decrease_s'].mean():.1f}"
        )
    print()
    print("=== runway-changed tta decreases (already-fixed dynamic-reassignment channel, for comparison) ===")
    for label, mask in [("failed", df["failed"]), ("succeeded", ~df["failed"])]:
        sub = df[mask]
        print(
            f"{label:>10} n={len(sub):4d}  "
            f"mean_n_decrease_rwy_changed={sub['n_decrease_rwy_changed'].mean():.2f}  "
            f"frac_flights_with_any={(sub['n_decrease_rwy_changed']>0).mean():.3f}"
        )

    # Correlation check
    corr = df[["n_decrease_same_rwy", "failed"]].corr().iloc[0, 1]
    print(f"\ncorr(n_decrease_same_rwy, failed) = {corr:.3f}")

    df.to_parquet(f"{ROOT}/phase5a_thrash_analysis.parquet")
    print(f"\nWrote per-flight table to {ROOT}/phase5a_thrash_analysis.parquet")


if __name__ == "__main__":
    main()
