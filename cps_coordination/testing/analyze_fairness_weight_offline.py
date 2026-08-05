"""
cps_coordination/testing/analyze_fairness_weight_offline.py
--------------------------------------------------------------
Offline analysis of the local `fairness_weight` calibration sweep (Stage 1
coarse + Stage 2 refine, see `.claude/plans/step10_execution_and_data_
collection_plan.md`'s "fairness_weight calibrated via Stage 1/2 sweep"
section). This is deliverable #1 of that plan (the analytical justification
for `CPSModelConfig.fairness_weight`'s deployed default) in re-runnable
script form, rather than a static prose document that can drift from the
data it describes.

Only `k_cps=3` combos are meaningful here: `fairness_weight` is a proven
no-op at `k_cps=0` (`cps_manager.py::_apply_k_cps_constraint` short-circuits
to plain FCFS whenever `k_cps == 0 or fairness_weight <= 0.0`) -- any
`k_cps=0` combo found in a sweep root is loaded (for the collision/sanity
checks) but excluded from the fairness_weight comparison table.

Reuses existing helpers rather than reimplementing Parquet loading or
metric math: `load_all_combos`/`validate_occurrence_ordering` from
`step10_deep_analysis.py`, `recompute_metrics`/`load_recat_matrix` from
`cps_metrics_offline.py`.

Usage
-----
  # Stage 1 only (coarse):
  python cps_coordination/testing/analyze_fairness_weight_offline.py \\
      --sweep-roots cps_coordination/data/fairness_weight_calibration_sweep/stage1

  # Stage 1 + Stage 2 combined (final recommendation):
  python cps_coordination/testing/analyze_fairness_weight_offline.py \\
      --sweep-roots cps_coordination/data/fairness_weight_calibration_sweep/stage1 \\
                    cps_coordination/data/fairness_weight_calibration_sweep/stage2 \\
      --out cps_coordination/data/fairness_weight_calibration_sweep/report.txt

Runtime: seconds (pure Parquet + pandas/numpy, no BlueSky/SB3 import, no
simulation) -- matches the existing offline-analysis tooling's pattern.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
from contextlib import redirect_stdout
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cps_coordination.testing.cps_metrics_offline import load_recat_matrix, recompute_metrics
from cps_coordination.testing.step10_deep_analysis import load_all_combos, validate_occurrence_ordering
from cps_coordination.testing.summarize_batch_sweep import _COMBO_RE

DEFAULT_FW_BASELINE = 0.0
DEFAULT_SUCCESS_REGRESSION_TOLERANCE_PP = 2.0
DEFAULT_JOINT_TOLERANCE_PP = 1.0
DEFAULT_SEP_TOLERANCE_S = 5.0
DEFAULT_RTA_TOLERANCE_S = 60.0


def _binom_se(p: float, n: int) -> float:
    if n <= 0 or p != p:  # p != p -> nan
        return float("nan")
    return math.sqrt(max(p, 0.0) * max(1.0 - p, 0.0) / n)


def load_and_tag_combos(sweep_roots: List[str]) -> pd.DataFrame:
    """Load every (k_cps, mode, fw) combo across the given sweep roots,
    recompute headline metrics, and return one row per combo.

    Raises if the same combo name (e.g. ``k3_dynamic_fw0.5``) appears in
    more than one sweep root -- that would mean two different runs claim
    to be the same calibration point, an ambiguity this script refuses to
    silently resolve (e.g. by picking one arbitrarily or averaging).
    """
    recat = load_recat_matrix()
    seen: Dict[str, str] = {}  # combo name -> source root
    rows = []

    for root in sweep_roots:
        all_data = load_all_combos(root)
        print(f"Loaded {len(all_data)} combos from {root}")
        validate_occurrence_ordering(all_data)

        for name, (df, sep) in all_data.items():
            if name in seen:
                raise ValueError(
                    f"Combo '{name}' found in both {seen[name]!r} and {root!r} -- "
                    "ambiguous calibration point, refusing to silently pick one."
                )
            seen[name] = root

            m = _COMBO_RE.match(name)
            if m is None:
                print(f"  WARNING: '{name}' does not match the k<k_cps>_<mode>_fw<fw> "
                      "naming convention -- skipped.")
                continue

            k_cps = int(m.group("k_cps"))
            mode = m.group("mode")
            fw = float(m.group("fw"))
            metrics = recompute_metrics(
                df, sep, recat,
                sep_tolerance_s=DEFAULT_SEP_TOLERANCE_S,
                rta_tolerance_s=DEFAULT_RTA_TOLERANCE_S,
            )
            n_aircraft = int(len(df))
            n_stall_detected = int(df["stall_detected"].sum()) if "stall_detected" in df else 0

            rows.append({
                "combo": name,
                "source_root": root,
                "k_cps": k_cps,
                "mode": mode,
                "fw": fw,
                "n_episodes": metrics["n_episodes"],
                "n_aircraft": n_aircraft,
                "success_rate": metrics["success_rate"],
                "success_se": _binom_se(metrics["success_rate"], n_aircraft),
                "stall_recovery_rate": (
                    metrics["stall_recovery_rate"] if metrics["stall_recovery_rate"] != "nan" else float("nan")
                ),
                "n_stall_detected": n_stall_detected,
                "stall_recovery_se": _binom_se(
                    metrics["stall_recovery_rate"] if metrics["stall_recovery_rate"] != "nan" else float("nan"),
                    n_stall_detected,
                ),
                "stall_unrecovered": metrics["stall_unrecovered"],
                "c_sep": metrics["c_sep"],
            })

    return pd.DataFrame(rows)


def print_curve_table(table: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("fairness_weight -> headline metrics, per mode (k_cps=3 only)")
    print("=" * 100)
    k3 = table[table["k_cps"] == 3].sort_values(["mode", "fw"])
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(k3[[
            "mode", "fw", "source_root", "n_episodes", "n_aircraft",
            "success_rate", "success_se", "stall_recovery_rate", "stall_recovery_se",
            "stall_unrecovered", "c_sep",
        ]].to_string(index=False))

    k0 = table[table["k_cps"] == 0]
    if len(k0):
        print(f"\n({len(k0)} k_cps=0 combo(s) loaded and sanity-checked but excluded from "
              "the fairness_weight comparison -- fw is a proven no-op at k_cps=0.)")


def _select_per_mode(
    k3: pd.DataFrame, mode: str, fw_baseline: float, success_regression_tolerance_pp: float,
) -> Tuple[Optional[float], pd.DataFrame, str]:
    """Return (winning_fw, eligible_rows, reasoning_str) for one mode."""
    rows = k3[k3["mode"] == mode].copy()
    if rows.empty:
        return None, rows, f"no data for mode={mode!r}"

    baseline_rows = rows[np.isclose(rows["fw"], fw_baseline)]
    if baseline_rows.empty:
        return None, rows, (
            f"no fw={fw_baseline} baseline row for mode={mode!r} -- cannot apply the "
            "success-regression constraint, skipping selection for this mode"
        )
    baseline_success = float(baseline_rows.iloc[0]["success_rate"])
    threshold = baseline_success - success_regression_tolerance_pp / 100.0

    eligible = rows[rows["success_rate"] >= threshold]
    eligible = eligible[eligible["stall_recovery_rate"].notna()]
    if eligible.empty:
        return None, rows, (
            f"mode={mode!r}: no candidate meets the success >= {threshold:.4f} "
            f"(baseline {baseline_success:.4f} - {success_regression_tolerance_pp}pp) "
            "constraint with a defined stall_recovery_rate -- no recommendation for this mode"
        )

    winner_row = eligible.loc[eligible["stall_recovery_rate"].idxmax()]
    reasoning = (
        f"mode={mode!r}: baseline (fw={fw_baseline}) success_rate={baseline_success:.4f}; "
        f"{len(eligible)}/{len(rows)} candidates meet success >= {threshold:.4f}; "
        f"winner fw={winner_row['fw']:g} with stall_recovery_rate={winner_row['stall_recovery_rate']:.4f} "
        f"(success_rate={winner_row['success_rate']:.4f})"
    )
    return float(winner_row["fw"]), rows, reasoning


def recommend(
    table: pd.DataFrame,
    fw_baseline: float = DEFAULT_FW_BASELINE,
    success_regression_tolerance_pp: float = DEFAULT_SUCCESS_REGRESSION_TOLERANCE_PP,
    joint_tolerance_pp: float = DEFAULT_JOINT_TOLERANCE_PP,
) -> Optional[float]:
    print("\n" + "=" * 100)
    print("SELECTION")
    print("=" * 100)
    k3 = table[table["k_cps"] == 3]
    if k3.empty:
        print("No k_cps=3 data loaded -- cannot recommend a fairness_weight.")
        return None

    winners: Dict[str, Optional[float]] = {}
    per_mode_rows: Dict[str, pd.DataFrame] = {}
    for mode in ("static", "dynamic"):
        fw, rows, reasoning = _select_per_mode(k3, mode, fw_baseline, success_regression_tolerance_pp)
        winners[mode] = fw
        per_mode_rows[mode] = rows
        print(f"  {reasoning}")

    if any(v is None for v in winners.values()):
        print("\nAt least one mode has no valid winner -- cannot produce a joint recommendation. "
              "Re-run with more/different fw candidates before finalizing.")
        return None

    static_fw, dynamic_fw = winners["static"], winners["dynamic"]

    if math.isclose(static_fw, dynamic_fw):
        print(f"\nBoth modes agree: fairness_weight = {static_fw:g}. FINAL RECOMMENDATION: {static_fw:g}")
        return static_fw

    print(f"\nModes disagree: static wants fw={static_fw:g}, dynamic wants fw={dynamic_fw:g}. "
          "Looking for a jointly-good compromise (a shared tested fw within "
          f"{joint_tolerance_pp}pp of stall_recovery_rate for BOTH modes' own optima).")

    static_rows = per_mode_rows["static"].set_index("fw")
    dynamic_rows = per_mode_rows["dynamic"].set_index("fw")
    static_best_rec = float(static_rows.loc[static_fw, "stall_recovery_rate"])
    dynamic_best_rec = float(dynamic_rows.loc[dynamic_fw, "stall_recovery_rate"])

    shared_fw = sorted(set(static_rows.index) & set(dynamic_rows.index))
    best_shared: Optional[float] = None
    best_shared_max_gap = float("inf")
    for fw in shared_fw:
        s_rec = static_rows.loc[fw, "stall_recovery_rate"]
        d_rec = dynamic_rows.loc[fw, "stall_recovery_rate"]
        if pd.isna(s_rec) or pd.isna(d_rec):
            continue
        gap = max(static_best_rec - float(s_rec), dynamic_best_rec - float(d_rec))
        if gap < best_shared_max_gap:
            best_shared_max_gap = gap
            best_shared = float(fw)

    if best_shared is not None and best_shared_max_gap <= joint_tolerance_pp / 100.0:
        print(f"Found jointly-good compromise: fw={best_shared:g} "
              f"(worst-case gap to either mode's own optimum: {best_shared_max_gap * 100:.2f}pp). "
              f"FINAL RECOMMENDATION: {best_shared:g}")
        return best_shared

    print("No shared tested fw value is within tolerance of BOTH modes' own optima. "
          "This is a real finding, not a script limitation: a genuinely mode-specific "
          "fairness_weight may be warranted -- consider deploying different values for "
          "static vs. dynamic runway_assignment_mode instead of one global default, and "
          "flag this for the thesis write-up rather than silently averaging it away.")
    if best_shared is not None:
        print(f"Closest available compromise (gap={best_shared_max_gap * 100:.2f}pp > "
              f"{joint_tolerance_pp}pp tolerance): fw={best_shared:g}. "
              "FALLBACK RECOMMENDATION (compromise, not a clean joint winner): "
              f"{best_shared:g}")
        return best_shared

    fallback = round((static_fw + dynamic_fw) / 2, 4)
    print(f"No shared tested fw values at all between the two modes' eligible sets -- "
          f"falling back to the numeric average of the two per-mode winners: {fallback:g}. "
          "This is a weak fallback (not itself a tested point) -- prefer re-running Stage 2 "
          "with overlapping candidate values if this triggers.")
    return fallback


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-roots", type=str, nargs="+", required=True,
                   help="One or more sweep-root directories (e.g. Stage 1 and Stage 2 of the "
                        "fairness_weight calibration sweep) to combine into one analysis.")
    p.add_argument("--fw-baseline", type=float, default=DEFAULT_FW_BASELINE,
                   help="fairness_weight value treated as the no-fairness-reordering reference "
                        f"for the success-regression constraint (default {DEFAULT_FW_BASELINE}).")
    p.add_argument("--success-regression-tolerance-pp", type=float,
                   default=DEFAULT_SUCCESS_REGRESSION_TOLERANCE_PP,
                   help="Max success_rate regression (percentage points) below the fw-baseline "
                        f"allowed for a candidate to be eligible (default {DEFAULT_SUCCESS_REGRESSION_TOLERANCE_PP}).")
    p.add_argument("--joint-tolerance-pp", type=float, default=DEFAULT_JOINT_TOLERANCE_PP,
                   help="Max gap (percentage points) to either mode's own optimum for a shared "
                        f"fw value to count as a joint winner (default {DEFAULT_JOINT_TOLERANCE_PP}).")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to also write the full printed report as text.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    buf = io.StringIO()
    with redirect_stdout(buf):
        table = load_and_tag_combos(args.sweep_roots)
        print_curve_table(table)
        winner = recommend(
            table,
            fw_baseline=args.fw_baseline,
            success_regression_tolerance_pp=args.success_regression_tolerance_pp,
            joint_tolerance_pp=args.joint_tolerance_pp,
        )
        print("\n" + "=" * 100)
        if winner is not None:
            print(f"fairness_weight CALIBRATION RESULT: {winner:g}")
        else:
            print("fairness_weight CALIBRATION RESULT: none (see selection notes above)")
        print("=" * 100)

    report = buf.getvalue()
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(report)
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
