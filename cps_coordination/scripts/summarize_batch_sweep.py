"""
cps_coordination/scripts/summarize_batch_sweep.py
----------------------------------------------------
Tabulate ``cps_metrics_offline.recompute_metrics`` across every combo
directory produced by ``run_batch_eval.py``'s sweep (``k{k_cps}_{mode}/``)
-- the "load and compare across combos" step exit criterion #6 in
``.claude/plans/phase3_cps_coordination_plan.md`` needs, without re-running
anything.

Not a new metrics pipeline -- purely a thin wrapper that discovers combo
directories under a sweep root and calls the existing offline recompute
once per directory.

The combo-directory naming used to include a ``_fw{fairness_weight}``
suffix; ``fairness_weight`` was removed from the codebase 2026-08-12 (see
``.claude/plans/stall_rate_investigation.md``). The regex below still
accepts the old suffix (optional) so this script keeps working against
every already-collected sweep from before the removal, not just new ones.

Run: python cps_coordination/scripts/summarize_batch_sweep.py --sweep-root <path>
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Any, Dict, List

import pandas as pd

from cps_coordination.scripts.cps_metrics_offline import (
    load_recat_matrix,
    load_telemetry,
    recompute_metrics,
)

_COMBO_RE = re.compile(r"^k(?P<k_cps>\d+)_(?P<mode>static|dynamic)(?:_fw(?P<fw>[\d.]+))?$")


def discover_combos(sweep_root: str) -> List[str]:
    return sorted(
        d for d in glob.glob(os.path.join(sweep_root, "k*"))
        if os.path.isdir(d) and _COMBO_RE.match(os.path.basename(d))
        and os.path.exists(os.path.join(d, "cps_eval_aircraft.parquet"))
    )


def summarize(sweep_root: str, sep_tolerance_s: float, rta_tolerance_s: float) -> pd.DataFrame:
    recat_matrix = load_recat_matrix()
    rows: List[Dict[str, Any]] = []

    for combo_dir in discover_combos(sweep_root):
        m = _COMBO_RE.match(os.path.basename(combo_dir))
        assert m is not None
        aircraft_df, separation_df = load_telemetry(combo_dir)
        metrics = recompute_metrics(
            aircraft_df, separation_df, recat_matrix,
            sep_tolerance_s=sep_tolerance_s, rta_tolerance_s=rta_tolerance_s,
        )
        fw = m.group("fw")
        rows.append({
            "k_cps": int(m.group("k_cps")),
            "mode": m.group("mode"),
            "fairness_weight": float(fw) if fw is not None else None,  # None: post-removal combo dir
            "n_episodes": metrics.get("n_episodes"),
            "n_aircraft": metrics.get("n_aircraft"),
            "success_rate": metrics.get("success_rate"),
            "c_sep": metrics.get("c_sep"),
            "c_sep_from_landings_crosscheck": metrics.get("c_sep_from_landings_crosscheck"),
            "r_rec": metrics.get("r_rec"),
            "delta_epsilon_vs_static": metrics.get("delta_epsilon_vs_static"),
            "delta_epsilon_vs_uncoordinated": metrics.get("delta_epsilon_vs_uncoordinated"),
            "rho_ripple": metrics.get("rho_ripple"),
            "stall_unrecovered": metrics.get("stall_unrecovered"),
            "stall_recovery_rate": metrics.get("stall_recovery_rate"),
            "stall_rate": metrics.get("stall_rate"),
        })

    return pd.DataFrame(rows).sort_values(["k_cps", "mode", "fairness_weight"]).reset_index(drop=True)


def check_fairness_weight_nonvacuous(sweep_root: str) -> None:
    """Legacy check, relevant only for sweep roots collected before
    fairness_weight's removal (2026-08-12, see
    .claude/plans/stall_rate_investigation.md) that still have multiple
    `_fw{value}` combo directories for the same (k_cps, mode) pair.
    Confirmed fairness_weight actually reordered aircraft at fixed (k_cps,
    mode) when k_cps > 0 -- the concrete bar exit criterion #6 set. At
    k_cps == 0, the old `_apply_k_cps_constraint` short-circuited to plain
    FCFS regardless of fairness_weight, so byte-identical rows there were
    the *expected*, correct result, not a false negative. A no-op (prints
    nothing) against any post-removal sweep root, since there is only ever
    one directory per (k_cps, mode) there.
    """
    combos = discover_combos(sweep_root)
    by_k_mode: Dict[tuple, List[str]] = {}
    for d in combos:
        m = _COMBO_RE.match(os.path.basename(d))
        assert m is not None
        by_k_mode.setdefault((int(m.group("k_cps")), m.group("mode")), []).append(d)

    print("\n--- fairness_weight non-vacuousness check ---")
    for (k_cps, mode), dirs in sorted(by_k_mode.items()):
        if len(dirs) < 2:
            continue
        # Sort by flight_id, not acid: under a rolling-arrival stream, the
        # same per-slot acid (e.g. "AC000") is reused by multiple distinct
        # flights within one episode, so sorting by acid alone doesn't
        # uniquely/stably order rows and can produce spurious "differences"
        # that are really just row-order mismatches (confirmed: an
        # acid-sorted comparison here disagreed by ~1e-12 on values that are
        # exactly equal once ordered by the actually-unique flight_id).
        dfs = {os.path.basename(d): load_telemetry(d)[0] for d in dirs}
        names = list(dfs.keys())
        base = dfs[names[0]].sort_values(["episode_id", "flight_id"])["assigned_tta"].reset_index(drop=True)
        for other in names[1:]:
            other_series = dfs[other].sort_values(["episode_id", "flight_id"])["assigned_tta"].reset_index(drop=True)
            identical = bool((base - other_series).abs().max() < 1e-6)
            expectation = "expected (k_cps=0 short-circuits to FCFS)" if k_cps == 0 else "UNEXPECTED if identical"
            print(f"  k_cps={k_cps} mode={mode}: {names[0]} vs {other} assigned_tta identical={identical} ({expectation})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sweep-root", type=str, required=True)
    p.add_argument("--sep-tolerance-s", type=float, default=5.0)
    p.add_argument("--rta-tolerance-s", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = summarize(args.sweep_root, args.sep_tolerance_s, args.rta_tolerance_s)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    check_fairness_weight_nonvacuous(args.sweep_root)


if __name__ == "__main__":
    main()
