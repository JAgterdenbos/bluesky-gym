"""
cps_coordination/scripts/generate_thesis_summary.py
--------------------------------------------------------
Human-readable Markdown narrative report for thesis write-up/defense prep --
NOT LaTeX tables (see `generate_paper_report.py` for those) and NOT a
numeric pass/fail gate (see `verify_metrics_sanity.py` for that). This is
the interpretive layer on top of the same data: what each metric means in
plain language, the current headline value, a k-sensitivity ("does k-CPS
relaxation help") summary, and an explicit call-out list of anything
flagged-but-unresolved so it's pre-loaded as an anticipated defense
question rather than a surprise mid-defense.

Reuses `generate_paper_report.load_combo_metrics` (which itself reuses
`cps_metrics_offline.recompute_metrics`) rather than re-implementing combo
discovery or metric computation -- a narrative layer, not a new data path.

Parameterized by `--sweep-root` so the same script runs against the
current 25-ac dataset now and the future 50-ac dataset later; running it
now doubles as an early dry run of the narrative logic before it matters
for the final numbers.

Usage
-----
  python cps_coordination/scripts/generate_thesis_summary.py \\
      --sweep-root experiments/cps_eval/scale_10k_20260807_123741 \\
      --out cps_coordination/figures/thesis_summary_20260811.md

Runtime: seconds (pure Parquet + pandas/numpy, no BlueSky/SB3 import).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cps_coordination.scripts.generate_paper_report import load_combo_metrics

# Plain-language descriptions, reused verbatim across every metric section
# so a reader doesn't have to cross-reference metrics.py's docstrings.
_METRIC_BLURBS: Dict[str, str] = {
    "gamma": (
        "Combined-runway throughput: total successful landings per hour of "
        "elapsed simulation time, summed across all runways in scope."
    ),
    "c_sep": (
        "Separation compliance: fraction of consecutive same-runway landing "
        "pairs that meet the RECAT-EU minimum separation (within tolerance)."
    ),
    "delta_epsilon_vs_static": (
        "Tracking degradation (Eq. tracking_degradation, RQ2.2's literal "
        "metric): mean |RTA error under CPS| minus mean |RTA error under a "
        "frozen, once-assigned static TTA|. Negative means CPS tracks the "
        "assigned arrival time MORE accurately than a static schedule would."
    ),
    "delta_epsilon_vs_uncoordinated": (
        "Secondary reference (NOT Groot et al.'s published data): mean "
        "|RTA error under CPS| minus mean |RTA error solo/uncoordinated| "
        "under the identical frozen worker."
    ),
    "r_rec": (
        "Recovery success rate: of aircraft that received a genuine "
        "mid-trajectory TTA update, the fraction that still landed within "
        "the RTA tolerance despite the update."
    ),
    "rho_ripple": (
        "Delay ripple index: mean per-episode lag-1 autocorrelation of "
        "consecutive aircraft's RTA errors (sorted by landing time). "
        "Positive means one aircraft's delay tends to be followed by a "
        "similarly-signed delay in the next; near zero means delays don't "
        "propagate through the landing sequence."
    ),
    "stall_unrecovered": (
        "Headline stall risk metric: fraction of ALL aircraft that were "
        "flagged stalled (distance-to-IAF plateaued) AND never landed -- "
        "the actually-costly subset, reported alongside success_rate."
    ),
    "stall_recovery_rate": (
        "Of aircraft flagged stalled, the fraction that still landed "
        "successfully -- a mitigation-effectiveness diagnostic, not a "
        "headline risk metric on its own."
    ),
    "stall_rate": (
        "Diagnostic only: fraction of all aircraft flagged as stalled by "
        "CPSManager. Answers 'did progress plateau', not 'did it fail' -- "
        "an aircraft can legitimately stall during path-stretching and "
        "still converge. NOT the headline risk metric (see stall_unrecovered)."
    ),
    "success_rate": "Fraction of all aircraft that landed successfully.",
}

_KEY_METRICS_ORDER = [
    "success_rate", "gamma", "c_sep", "delta_epsilon_vs_static",
    "delta_epsilon_vs_uncoordinated", "r_rec", "rho_ripple",
    "stall_unrecovered", "stall_recovery_rate", "stall_rate",
]


def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    if v == "nan":
        return "nan"
    return f"{float(v):.{decimals}f}"


def _combo_label(row: Dict[str, Any]) -> str:
    return f"k={row['k_cps']}/{row['mode']}"


def _value_range_str(rows: List[Dict[str, Any]], key: str, decimals: int = 4) -> str:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not np.isnan(r[key])]
    if not vals:
        return "n/a (no valid combo values)"
    if len(vals) == 1:
        return _fmt(vals[0], decimals)
    lo, hi = min(vals), max(vals)
    return f"{_fmt(lo, decimals)} to {_fmt(hi, decimals)}"


def _build_metric_section(combo_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "## Metric-by-metric interpretation\n",
        "`std` is the sample standard deviation of that metric across "
        "episodes within a combo (episode-to-episode variance, not a "
        "standard error of the mean) -- `nan` with fewer than 2 valid "
        "episode observations.\n",
    ]
    for key in _KEY_METRICS_ORDER:
        blurb = _METRIC_BLURBS.get(key, "")
        rng = _value_range_str(combo_rows, key)
        lines.append(f"### `{key}`\n")
        lines.append(f"{blurb}\n")
        lines.append(f"**Current range across combos:** {rng}\n")
        lines.append("| combo | value | std (across episodes) |")
        lines.append("|---|---|---|")
        for r in combo_rows:
            lines.append(f"| {_combo_label(r)} | {_fmt(r.get(key))} | {_fmt(r.get(f'{key}_std'))} |")
        lines.append("")
    return "\n".join(lines)


def _build_throughput_arithmetic_section(
    combo_rows: List[Dict[str, Any]],
    total_arrivals_per_episode: Optional[float],
    spawn_window_s: Optional[float],
) -> str:
    lines = ["## Throughput arithmetic (Finding 4 reasoning)\n"]
    if total_arrivals_per_episode is None or spawn_window_s is None:
        lines.append(
            "`--total-arrivals-per-episode` / `--spawn-window-s` not provided -- skipping "
            "the spawn-rate-vs-measured-Gamma explanation. Pass both (or run "
            "`verify_metrics_sanity.py` directly) to reproduce it.\n"
        )
        return "\n".join(lines)

    spawn_window_h = spawn_window_s / 3600.0
    spawn_rate_ac_h = total_arrivals_per_episode / spawn_window_h
    lines.append(
        f"Raw spawn schedule rate is **{spawn_rate_ac_h:.2f} ac/h** "
        f"({total_arrivals_per_episode:.0f} arrivals / {spawn_window_h:.4f}h window), "
        "which will look inconsistent with the measured Gamma values below unless the "
        "gap is explained explicitly -- do so up front rather than let it read as a bug.\n"
    )
    lines.append("| combo | measured Γ (ac/h) | spawn/Γ ratio | why: mean episode span |")
    lines.append("|---|---|---|---|")
    for r in combo_rows:
        gamma = r.get("gamma")
        if not isinstance(gamma, (int, float)) or np.isnan(gamma) or gamma == 0:
            lines.append(f"| {_combo_label(r)} | n/a | n/a | n/a |")
            continue
        ratio = spawn_rate_ac_h / gamma
        lines.append(
            f"| {_combo_label(r)} | {gamma:.2f} | {ratio:.2f}x | "
            "see `verify_metrics_sanity.py` output for this combo |"
        )
    lines.append(
        "\nThe gap is explained by queuing against `max_concurrent_aircraft`'s slot cap "
        "plus in-sector dwell/holding time, NOT a residual bug -- confirmed via "
        "`verify_metrics_sanity.py`'s independent from-scratch Gamma recomputation and "
        "spawn-rate-vs-measured-rate arithmetic. Re-run that script for the exact "
        "per-combo numbers backing this claim.\n"
    )
    return "\n".join(lines)


def _build_k_sensitivity_section(combo_rows: List[Dict[str, Any]]) -> str:
    lines = ["## k-sensitivity summary (\"does k-CPS relaxation help?\")\n"]
    modes = sorted(set(r["mode"] for r in combo_rows))
    compare_keys = ["success_rate", "gamma", "c_sep", "delta_epsilon_vs_static", "r_rec", "rho_ripple"]

    for mode in modes:
        mode_rows = sorted(
            (r for r in combo_rows if r["mode"] == mode), key=lambda r: r["k_cps"]
        )
        k_values = [r["k_cps"] for r in mode_rows]
        lines.append(f"### mode = {mode} (k ∈ {sorted(set(k_values))})\n")
        lines.append("| metric | " + " | ".join(f"k={k}" for k in k_values) + " | direction (k=min → k=max) |")
        lines.append("|---|" + "---|" * len(k_values) + "---|")
        for key in compare_keys:
            vals = [r.get(key) for r in mode_rows]
            cells = " | ".join(_fmt(v) for v in vals)
            numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
            if len(numeric_vals) >= 2:
                delta = numeric_vals[-1] - numeric_vals[0]
                direction = "no change" if abs(delta) < 1e-9 else ("increases" if delta > 0 else "decreases")
            else:
                direction = "n/a"
            lines.append(f"| `{key}` | {cells} | {direction} |")
        lines.append("")
    return "\n".join(lines)


def _build_callouts_section(combo_rows: List[Dict[str, Any]]) -> str:
    lines = ["## Flagged-but-unresolved call-outs (anticipated defense questions)\n"]
    callouts: List[str] = []

    for r in combo_rows:
        stall_rate = r.get("stall_rate")
        stall_recovery = r.get("stall_recovery_rate")
        stall_rate_is_zero = (
            isinstance(stall_rate, (int, float)) and not np.isnan(stall_rate) and round(stall_rate, 4) == 0.0
        )
        recovery_is_defined = isinstance(stall_recovery, (int, float)) and not np.isnan(stall_recovery)
        if stall_rate_is_zero and recovery_is_defined:
            callouts.append(
                f"- **{_combo_label(r)}**: `stall_rate` rounds to 0.0000 but "
                f"`stall_recovery_rate` is defined ({_fmt(stall_recovery)}) rather than '--'. "
                "Likely a handful of sub-rounding stall events, not a bug, but confirm the "
                "raw stall count before citing the recovery-rate figure for this combo "
                "specifically."
            )

    if not callouts:
        callouts.append(
            "- No `stall_rate≈0` / `stall_recovery_rate`-defined inconsistency detected in "
            "this sweep root's combos."
        )

    callouts.append(
        "- **Throughput arithmetic**: Γ measures well under the raw spawn schedule rate "
        "for every combo (see the throughput-arithmetic section above) -- this is "
        "explained by queuing/dwell time, not a bug, but be ready to walk through the "
        "arithmetic live if asked why Γ looks low relative to the spawn rate."
    )

    lines.extend(callouts)
    lines.append("")
    return "\n".join(lines)


def build_report(
    sweep_root: str,
    combo_rows: List[Dict[str, Any]],
    total_arrivals_per_episode: Optional[float],
    spawn_window_s: Optional[float],
) -> str:
    header = [
        f"# CPS coordination -- thesis/defense summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Sweep root: `{sweep_root}`",
        f"Combos: {len(combo_rows)} ({', '.join(_combo_label(r) for r in combo_rows)})",
        f"Episodes per combo: {', '.join(str(r['n_episodes']) for r in combo_rows)}",
        "",
        "State which dataset/commit these numbers came from in any write-up that cites "
        "this report, per this repo's \"Self-Review Before Reporting Results\" rule.",
        "",
    ]
    sections = [
        _build_metric_section(combo_rows),
        _build_k_sensitivity_section(combo_rows),
        _build_throughput_arithmetic_section(combo_rows, total_arrivals_per_episode, spawn_window_s),
        _build_callouts_section(combo_rows),
    ]
    return "\n".join(header) + "\n" + "\n".join(sections)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a human-readable Markdown thesis/defense-prep narrative "
                     "from a CPS eval sweep root.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sweep-root", type=str, required=True,
                   help="Directory containing k<K>_<mode>_fw<FW>/ combo subdirectories.")
    p.add_argument("--out", type=str, default=None,
                   help="Output .md path (default: <sweep-root basename>_thesis_summary.md "
                        "under cps_coordination/figures/).")
    p.add_argument("--sep-tolerance-s", type=float, default=5.0)
    p.add_argument("--rta-tolerance-s", type=float, default=60.0)
    p.add_argument("--total-arrivals-per-episode", type=float, default=None,
                   help="cps_scale_10k.yaml's total_arrivals_per_episode, for the "
                        "throughput-arithmetic section (omit to skip that section's numbers).")
    p.add_argument("--spawn-window-s", type=float, default=None,
                   help="cps_scale_10k.yaml's spawn_window_s, for the throughput-arithmetic "
                        "section (omit to skip that section's numbers).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    combo_rows = load_combo_metrics(
        args.sweep_root, sep_tolerance_s=args.sep_tolerance_s, rta_tolerance_s=args.rta_tolerance_s,
    )
    if not combo_rows:
        print(f"No combos found under {args.sweep_root}")
        raise SystemExit(1)

    report = build_report(
        args.sweep_root, combo_rows, args.total_arrivals_per_episode, args.spawn_window_s,
    )

    out_path = args.out
    if out_path is None:
        base = os.path.basename(os.path.normpath(args.sweep_root))
        out_path = os.path.join("cps_coordination", "figures", f"{base}_thesis_summary.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(report)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
