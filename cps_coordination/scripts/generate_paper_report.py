"""
cps_coordination/scripts/generate_paper_report.py
-----------------------------------------------------
Single consolidated script producing every LaTeX-ready table/figure the
Phase III thesis chapter needs. Design history and decisions live in
`.claude/plans/phase3_cps_coordination_plan.md`'s seventh-session section
(the original `reporting_code_plan.md` this script implemented was folded
in there and deleted once superseded). One script, one output folder, per
explicit user request -- every paper-facing artifact can be regenerated
from a single command instead of being scattered across
`cps_coordination/scripts/`.

Wraps, does not reimplement, four existing scripts:
  - cps_metrics_offline.py       (per-combo metric recomputation)
  - summarize_batch_sweep.py     (combo discovery)
  - step10_deep_analysis.py      (stall/tortuosity/collision figures)
  - analyze_fairness_weight_offline.py (fairness_weight calibration analysis)

Produces
--------
  tab_throughput_results.tex   -- tab:throughput_results
  tab_delay_ripple.tex         -- tab:delay_ripple
  fig_runway_load_balance.png  -- fig:runway_load_balance
  tab_fairness_weight_calibration.tex -- appendix calibration table
  tab_ratchet_ablation.tex     -- appendix ratchet-ablation table
  fig1-4_*.png                 -- repointed from step10_deep_analysis.py

Usage
-----
  # Smoke-test against diagnostic-scale data (correct 4-combo shape, M=30):
  python cps_coordination/scripts/generate_paper_report.py \\
      --sweep-root cps_coordination/data/step10_verification_new_density_final \\
      --out-dir cps_coordination/figures/paper_report_smoketest

  # Real run, once the M=2,000/4-combo production sweep exists:
  python cps_coordination/scripts/generate_paper_report.py \\
      --sweep-root <production_sweep_root> \\
      --out-dir cps_coordination/figures/paper_report

Runtime: seconds (pure Parquet + pandas/numpy/matplotlib, no BlueSky/SB3
import, no simulation) -- same offline-analysis pattern as the four scripts
it wraps.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cps_coordination.scripts.analyze_fairness_weight_offline import load_and_tag_combos
from cps_coordination.scripts.cps_metrics_offline import (
    load_recat_matrix, load_telemetry, recompute_metrics,
)
from cps_coordination.scripts.step10_deep_analysis import (
    load_all_combos, validate_occurrence_ordering, workstream3_stalling,
)
from cps_coordination.scripts.summarize_batch_sweep import _COMBO_RE, discover_combos

_DEFAULT_FAIRNESS_SWEEP_ROOTS = [
    "cps_coordination/data/fairness_weight_calibration_sweep/stage1_ratchet_on",
    "cps_coordination/data/fairness_weight_calibration_sweep/stage2",
]
# NOT "stage1" (no suffix) -- that sweep ran under the production ratchet-OFF
# default, where stall_detected fires on ~0% of aircraft, so it carries no
# fairness_weight calibration signal (confirmed by direct comparison while
# building this script: stage1 alone gives success_rate ~0.98-0.999 for
# every fw, "no candidate meets" for both modes). stage1_ratchet_on +
# stage2 reproduces phase3_cps_coordination_plan.md's documented Stage 1/2
# table almost exactly (dynamic fw=0.5 success=0.576, static fw=1.0
# success=0.716) -- verified, not assumed.

_DPI = 180


# ──────────────────────────────────────────────────────────────────────────
# Generic LaTeX table helper
# ──────────────────────────────────────────────────────────────────────────


def _df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Booktabs-style LaTeX table from a DataFrame whose cells are already
    formatted strings, and column names are already valid LaTeX (this
    function does no numeric formatting or escaping itself -- callers
    pre-format so column-specific logic stays local to the table that
    needs it. Deliberately NOT auto-escaping underscores here: several
    column names use raw LaTeX math subscripts (e.g. r"$C_{sep}$") where
    escaping "_" would break the subscript, while plain-text column names
    with a literal underscore (e.g. "stall\\_unrecovered") already carry
    their own "\\_" -- a blanket replace would double-escape those)."""
    n_cols = len(df.columns)
    col_format = "l" + "c" * (n_cols - 1)
    header = " & ".join(str(c) for c in df.columns) + r" \\"
    rows = []
    for _, row in df.iterrows():
        rows.append(" & ".join(str(v) for v in row) + r" \\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _pct(x: Any) -> str:
    return "--" if x != x or x == "nan" else f"{float(x) * 100:.1f}\\%"


def _num(x: Any, decimals: int = 2) -> str:
    return "--" if x != x or x == "nan" else f"{float(x):.{decimals}f}"


# ──────────────────────────────────────────────────────────────────────────
# Combo metric loading (reuses discover_combos + recompute_metrics directly,
# not summarize_batch_sweep.summarize()'s DataFrame -- that wrapper drops
# `gamma`/`gamma_r`, which fig:runway_load_balance needs)
# ──────────────────────────────────────────────────────────────────────────


def load_combo_metrics(
    sweep_root: str, sep_tolerance_s: float = 5.0, rta_tolerance_s: float = 60.0,
) -> List[Dict[str, Any]]:
    recat = load_recat_matrix()
    rows: List[Dict[str, Any]] = []
    for combo_dir in discover_combos(sweep_root):
        m = _COMBO_RE.match(os.path.basename(combo_dir))
        assert m is not None
        aircraft_df, separation_df = load_telemetry(combo_dir)
        metrics = recompute_metrics(
            aircraft_df, separation_df, recat,
            sep_tolerance_s=sep_tolerance_s, rta_tolerance_s=rta_tolerance_s,
        )
        metrics["k_cps"] = int(m.group("k_cps"))
        metrics["mode"] = m.group("mode")
        metrics["fairness_weight"] = float(m.group("fw"))
        metrics["combo"] = os.path.basename(combo_dir)
        rows.append(metrics)
    return sorted(rows, key=lambda r: (r["k_cps"], r["mode"]))


# ──────────────────────────────────────────────────────────────────────────
# tab:throughput_results
# ──────────────────────────────────────────────────────────────────────────


def build_throughput_table(combo_rows: List[Dict[str, Any]]) -> str:
    out_rows = []
    for r in combo_rows:
        out_rows.append({
            "k": r["k_cps"],
            "mode": r["mode"],
            "M (episodes)": r["n_episodes"],
            "N (aircraft)": r["n_aircraft"],
            "success rate": _pct(r["success_rate"]),
            r"$\Gamma$ (ac/h)": _num(r["gamma"], 2),
            r"$C_{sep}$": _pct(r["c_sep"]),
            r"$R_{rec}$": _pct(r["r_rec"]),
        })
    df = pd.DataFrame(out_rows)
    return _df_to_latex(
        df,
        caption="Throughput and separation-compliance results across the "
                "$k_{cps} \\times$ runway\\_assignment\\_mode grid.",
        label="tab:throughput_results",
    )


# ──────────────────────────────────────────────────────────────────────────
# tab:delay_ripple
# ──────────────────────────────────────────────────────────────────────────


def build_delay_ripple_table(combo_rows: List[Dict[str, Any]]) -> str:
    out_rows = []
    for r in combo_rows:
        out_rows.append({
            "k": r["k_cps"],
            "mode": r["mode"],
            r"$\Delta\epsilon_{static}$ (s)": _num(r["delta_epsilon_vs_static"], 1),
            r"$\Delta\epsilon_{uncoord}$ (s)": _num(r["delta_epsilon_vs_uncoordinated"], 1),
            r"$\rho_{ripple}$": _num(r["rho_ripple"], 3),
            "stall\\_unrecovered": _pct(r["stall_unrecovered"]),
            "stall\\_recovery\\_rate": _pct(r["stall_recovery_rate"]),
        })
    df = pd.DataFrame(out_rows)
    return _df_to_latex(
        df,
        caption="Delay-tracking degradation ($\\Delta\\epsilon$), ripple "
                "propagation, and stall recovery across the "
                "$k_{cps} \\times$ runway\\_assignment\\_mode grid. "
                "$\\Delta\\epsilon_{static}$ is the literal "
                "eq:tracking\\_degradation quantity (RQ2.2 headline); "
                "$\\Delta\\epsilon_{uncoord}$ is a secondary, "
                "explicitly-not-Groot-et-al.\\ reference against an "
                "uncoordinated control run.",
        label="tab:delay_ripple",
    )


# ──────────────────────────────────────────────────────────────────────────
# fig:runway_load_balance -- includes Groot et al. Table 11/12 comparison
# ──────────────────────────────────────────────────────────────────────────

# Groot et al. (`hierarchical_RL_arrival_draft.pdf`, "Autonomous air traffic
# control with hierarchical reinforcement learning in real world scenarios",
# D.J. Groot, J. Ellerbroek, J.M. Hoekstra, groot2026model), Table 11 (p.18,
# total landing-interval violations by traffic scenario, Base/SA/MA) and
# Table 12 (pp.18-19, same metric, MA model only, split 18R vs. 27) --
# see .claude/plans/phase3_cps_coordination_plan.md's sixth-session section
# and .claude/plans/paper_update_plan.md item 3. Transcribed 2026-08-06 from
# values the user extracted via Gemini (both tables' cells reported as
# clearly legible, no inferred/guessed values) and cross-checked here for
# internal consistency: Table 11's MA row equals Table 12's 18R+27 sum in
# every scenario (e.g. Medium: 51 == 50+1; Low: 9 == 9+0) -- a good sign the
# transcription is accurate, not just plausible-looking.
#
# "Landing-interval violation" definition (Sec 5.4, p.15, quoted exactly):
# "The landing interval is defined as the time between two consecutive
# landings on the same runway, which should not exceed 50 s according to
# Eurocontrol's Optimised Runway Occupancy Time Spacings for Arrivals [40]."
# Explicitly a DIFFERENT metric from this codebase's Gamma_r (violation
# count vs. throughput) -- label/caption must say so, never overlay as if
# directly comparable.
GROOT_VIOLATION_DEFINITION = (
    "consecutive landings on the same runway $<$50s apart "
    "(Eurocontrol Optimised ROT Spacings for Arrivals)"
)

# Table 11: total landing-interval violations by traffic scenario, all three
# models (Base = non-learning baseline, SA = single-agent RL, MA = the
# multi-agent hierarchical RL coordination system this codebase is compared
# against). Not itself plotted in fig:runway_load_balance (no per-runway
# split) -- rendered as a small reference table for the appendix instead.
GROOT_TABLE11_VIOLATIONS_BY_METHOD: Dict[str, Dict[str, int]] = {
    "Base": {"Synthetic (Low)": 1136, "Synthetic (Medium)": 2313, "Synthetic (High)": 3826,
             "Historical (Jan)": 2009, "Historical (Mar)": 2065, "Historical (Jul)": 2093},
    "SA":   {"Synthetic (Low)": 1488, "Synthetic (Medium)": 2889, "Synthetic (High)": 4127,
             "Historical (Jan)": 2509, "Historical (Mar)": 2696, "Historical (Jul)": 2626},
    "MA":   {"Synthetic (Low)": 9, "Synthetic (Medium)": 51, "Synthetic (High)": 131,
             "Historical (Jan)": 58, "Historical (Mar)": 72, "Historical (Jul)": 64},
}

# Table 12: same metric, MA model only, split by runway -- the direct
# comparison point for fig:runway_load_balance (shows the runway-overload
# signature: 18R absorbs nearly all violations, 27 is nearly clean).
GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY: Optional[Dict[str, Dict[str, float]]] = {
    "Synthetic (Low)": {"18R": 9, "27": 0},
    "Synthetic (Medium)": {"18R": 50, "27": 1},
    "Synthetic (High)": {"18R": 131, "27": 0},
    "Historical (Jan)": {"18R": 58, "27": 0},
    "Historical (Mar)": {"18R": 72, "27": 0},
    "Historical (Jul)": {"18R": 64, "27": 0},
}


def build_runway_load_balance_figure(combo_rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    combos = [r["combo"] for r in combo_rows]
    runways = sorted({rwy for r in combo_rows for rwy in r["gamma_r"].keys()})

    have_groot = GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY is not None
    fig, axes = plt.subplots(1, 2 if have_groot else 1, figsize=(12 if have_groot else 7, 5))
    ax_gamma = axes[0] if have_groot else axes

    width = 0.8 / max(len(combos), 1)
    x = np.arange(len(runways))
    for i, r in enumerate(combo_rows):
        vals = [r["gamma_r"].get(rwy, 0.0) if r["gamma_r"].get(rwy, 0.0) != "nan" else 0.0 for rwy in runways]
        ax_gamma.bar(x + i * width - 0.4 + width / 2, vals, width, label=r["combo"])
    ax_gamma.set_xticks(x)
    ax_gamma.set_xticklabels(runways)
    ax_gamma.set_xlabel("Runway")
    ax_gamma.set_ylabel(r"$\Gamma_r$ (aircraft/hour)")
    ax_gamma.set_title("CPS coordination throughput by runway")
    ax_gamma.legend(fontsize=7)

    if have_groot:
        ax_groot = axes[1]
        scenarios = list(GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY.keys())
        groot_runways = sorted({rwy for v in GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY.values() for rwy in v})
        gw = 0.8 / max(len(scenarios), 1)
        gx = np.arange(len(groot_runways))
        for i, scenario in enumerate(scenarios):
            vals = [GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY[scenario].get(rwy, 0.0) for rwy in groot_runways]
            ax_groot.bar(gx + i * gw - 0.4 + gw / 2, vals, gw, label=scenario)
        ax_groot.set_xticks(gx)
        ax_groot.set_xticklabels(groot_runways)
        ax_groot.set_xlabel("Runway")
        ax_groot.set_ylabel("Landing-interval violations (Groot et al., MA)")
        ax_groot.set_title(
            "Groot et al. Table 12, MA model\n"
            "(landing-interval violations, $<$50s gap --\n"
            "reference only, NOT the same metric as $\\Gamma_r$)",
            fontsize=10,
        )
        ax_groot.legend(fontsize=7)
    else:
        print("WARNING: GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY not populated -- "
              "fig:runway_load_balance renders CPS throughput only, no Groot "
              "et al. comparison panel. Ask the user to paste Table 11/12 "
              "values from hierarchical_RL_arrival_draft.pdf.")

    fig.suptitle("fig:runway_load_balance")
    fig.tight_layout()
    out_path = out_dir / "fig_runway_load_balance.png"
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)
    return out_path


def build_groot_table11_reference_table() -> str:
    """Appendix reference table for Groot et al. Table 11 (Base/SA/MA, no
    runway split -- not itself plotted in fig:runway_load_balance, which
    needs Table 12's per-runway granularity, but transcribed per the plan's
    explicit "transcribe both into the reference dict" instruction)."""
    scenarios = list(next(iter(GROOT_TABLE11_VIOLATIONS_BY_METHOD.values())).keys())
    out_rows = []
    for method, row in GROOT_TABLE11_VIOLATIONS_BY_METHOD.items():
        out_rows.append({"method": method, **{s: str(row[s]) for s in scenarios}})
    df = pd.DataFrame(out_rows)
    return _df_to_latex(
        df,
        caption="Groot et al., Table 11: total landing-interval violations "
                f"({GROOT_VIOLATION_DEFINITION}) by traffic scenario, "
                "Base/SA/MA models. Reference only -- landing-interval "
                "violations are a different metric from this codebase's "
                "$\\Gamma_r$ throughput, not directly comparable.",
        label="tab:groot_table11_reference",
    )


# ──────────────────────────────────────────────────────────────────────────
# Appendix: fairness_weight Stage 1/2 calibration table (real data, not
# blocked -- wraps analyze_fairness_weight_offline.load_and_tag_combos)
# ──────────────────────────────────────────────────────────────────────────


def build_fairness_weight_calibration_table(sweep_roots: List[str]) -> str:
    table = load_and_tag_combos(sweep_roots)
    k3 = table[table["k_cps"] == 3].sort_values(["mode", "fw"])
    out_rows = []
    for _, row in k3.iterrows():
        out_rows.append({
            "mode": row["mode"],
            "fw": f"{row['fw']:g}",
            "M (episodes)": int(row["n_episodes"]),
            "success rate": _pct(row["success_rate"]),
            "stall\\_recovery\\_rate": _pct(row["stall_recovery_rate"]),
            "stall\\_unrecovered": _pct(row["stall_unrecovered"]),
            r"$C_{sep}$": _pct(row["c_sep"]),
        })
    df = pd.DataFrame(out_rows)
    return _df_to_latex(
        df,
        caption="Stage 1/2 $fairness\\_weight$ calibration sweep "
                "($k_{cps}=3$ only -- a proven no-op at $k_{cps}=0$), run "
                "under the ratchet-ON diagnostic regime to obtain signal "
                "(the production ratchet-OFF default drives "
                "stall\\_detected to $\\sim$0\\%, leaving nothing to "
                "calibrate against). Deployed production defaults: "
                "static=1.0, dynamic=0.5.",
        label="tab:fairness_weight_calibration",
    )


# ──────────────────────────────────────────────────────────────────────────
# Appendix: ratchet ablation table -- transcribed from
# phase3_cps_coordination_plan.md's "Vector 1" and "Ratchet sign-off, scale
# verification" sections. No per-arm sweep directory survives on disk under
# an obvious name (checked: no cps_coordination/data/ subdirectory matching
# vector1/falsif/freeze/clamp/ratchet -- confirmed by search, not assumed),
# so these are transcribed once from the plan's own documented tables/prose,
# not recomputed from Parquet.
# ──────────────────────────────────────────────────────────────────────────

# Single-wave falsification test (seed_base=1000, N=5, single-runway ["27"],
# k_cps=0, M=15) -- phase3_cps_coordination_plan.md, "Vector 1" section.
_RATCHET_VECTOR1_SINGLE_WAVE = [
    ("baseline (ratchet on)", 0.280, 0.573, 0.360),
    ("feature freeze alone", 0.253, 0.693, 0.440),
    ("feature clamp (1800s)", 0.347, 0.640, 0.373),
    ("freeze + ratchet off", 0.480, 1.000, 0.520),
    ("ratchet off alone", 1.000, 0.000, 0.000),
]

# Definitive go/no-go, rolling-arrival scenario (max_concurrent_aircraft=5,
# total_arrivals_per_episode=10, spawn_window_s=1800, single runway,
# seed_base=1000, M=20, n=200/arm) -- same section, "Definitive go/no-go".
_RATCHET_VECTOR1_ROLLING_ARRIVAL = [
    ("baseline (ratchet=True)", 0.395, 0.9153, 0.47),
    ("ratchet=False", 0.910, 0.9383, 0.09),
]

# Real-scale (M=30, full 8-combo grid, spawn_window_s=1800) confirmation --
# "Ratchet sign-off, scale verification & Step 10 artifact generation"
# section. Only aggregate ranges + one concrete example are given in the
# plan narrative (no full per-combo table survives), transcribed as-is --
# not fabricating individual per-combo numbers the source doesn't state.
_RATCHET_M30_SCALE_CONFIRMATION = [
    ("stall\\_rate / stall\\_unrecovered, ratchet on -> off", "17\\%--84\\%", "exactly 0\\% (every combo)"),
    ("success\\_rate improvement, ratchet on -> off", "--", "+12.1pp to +40.7pp (every combo)"),
    ("example: k0\\_dynamic success\\_rate", "58.3\\%", "99.0\\%"),
    ("caveat", "--", "a couple of dynamic-mode combos still fall short of 100\\% "
                      "(99.0\\%, 99.7\\%), from unrelated causes "
                      "(wrong\\_runway/out\\_of\\_bounds/restrict)"),
]


def build_ratchet_ablation_tables() -> str:
    df1 = pd.DataFrame(
        [{"arm": a, "success rate": _pct(s), "stall\\_rate": _pct(st), "stall\\_unrecovered": _pct(u)}
         for a, s, st, u in _RATCHET_VECTOR1_SINGLE_WAVE]
    )
    tex1 = _df_to_latex(
        df1,
        caption="Vector 1 falsification tests, single-wave scenario "
                "($k_{cps}=0$, single runway, $M=15$). Source: "
                "phase3\\_cps\\_coordination\\_plan.md, \\S Vector 1.",
        label="tab:ratchet_vector1_single_wave",
    )

    df2 = pd.DataFrame(
        [{"arm": a, "success rate": _pct(s), r"$C_{sep}$": _num(c, 4), "stall\\_unrecovered": _pct(u)}
         for a, s, c, u in _RATCHET_VECTOR1_ROLLING_ARRIVAL]
    )
    tex2 = _df_to_latex(
        df2,
        caption="Definitive ratchet go/no-go, rolling-arrival scenario "
                "(the scenario built to exercise the ratchet's protective "
                "purpose; $M=20$, $n=200$/arm). Source: "
                "phase3\\_cps\\_coordination\\_plan.md, \\S Vector 1.",
        label="tab:ratchet_rolling_arrival_gonogo",
    )

    df3 = pd.DataFrame(
        [{"metric": m, "ratchet on": before, "ratchet off": after}
         for m, before, after in _RATCHET_M30_SCALE_CONFIRMATION]
    )
    tex3 = _df_to_latex(
        df3,
        caption="Real-scale confirmation ($M=30$, full 8-combo grid, "
                "spawn\\_window\\_s=1800). Aggregate ranges only -- no "
                "full per-combo table survives in the source narrative. "
                "Source: phase3\\_cps\\_coordination\\_plan.md, "
                "\\S Ratchet sign-off, scale verification \\& Step 10 "
                "artifact generation.",
        label="tab:ratchet_m30_confirmation",
    )

    return tex1 + "\n" + tex2 + "\n" + tex3


# ──────────────────────────────────────────────────────────────────────────
# Repoint step10_deep_analysis.py's fig1-4 at the same output folder
# ──────────────────────────────────────────────────────────────────────────


def repoint_deep_analysis_figures(sweep_root: str, out_dir: Path) -> bool:
    """Attempts fig1-4 generation via step10_deep_analysis.py's own
    workstream3_stalling, confirmed compatible with the 4-combo production
    naming convention by a dry run against step10_verification_new_density_
    final/ while building this script (see phase3_cps_coordination_plan.md's
    seventh-session section) -- fig4 needed a small fix (see
    step10_deep_analysis.py's _pick_k3_combo) since it originally hardcoded
    a "k3_{mode}_fw0" combo name that doesn't exist under the per-mode-
    calibrated production fairness_weight layout.

    Wrapped in try/except (not asserted unconditionally) -- this function's
    whole point per the plan is "verify, don't assume it works unmodified,"
    so a genuine incompatibility on some future sweep-root shape should
    degrade to a clear warning, not crash the rest of the report.
    """
    try:
        all_data = load_all_combos(sweep_root)
        validate_occurrence_ordering(all_data)
        workstream3_stalling(all_data, out_dir)
        return True
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"WARNING: step10_deep_analysis.py fig1-4 repoint failed against "
              f"'{sweep_root}': {exc!r}. Skipping fig1-4 -- investigate before "
              f"treating this sweep-root as report-ready.")
        return False


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-root", type=str, required=True,
                   help="Completed k_cps x mode 4-combo sweep root for "
                        "tab:throughput_results/tab:delay_ripple/fig:runway_load_balance "
                        "and the repointed step10_deep_analysis.py figures. "
                        "Use cps_coordination/data/step10_verification_new_density_final "
                        "for a smoke test; point at the real M=2,000 production "
                        "sweep once it exists.")
    p.add_argument("--out-dir", type=str, default="cps_coordination/figures/paper_report",
                   help="Output folder for every regenerated .tex/.png artifact.")
    p.add_argument("--fairness-sweep-roots", type=str, nargs="+",
                   default=_DEFAULT_FAIRNESS_SWEEP_ROOTS,
                   help="Sweep roots for the fairness_weight calibration appendix table.")
    p.add_argument("--sep-tolerance-s", type=float, default=5.0)
    p.add_argument("--rta-tolerance-s", type=float, default=60.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== generate_paper_report.py -> {out_dir} ===")

    combo_rows = load_combo_metrics(args.sweep_root, args.sep_tolerance_s, args.rta_tolerance_s)
    print(f"Loaded {len(combo_rows)} combos from {args.sweep_root}: "
          f"{[r['combo'] for r in combo_rows]}")
    if len(combo_rows) != 4:
        print(f"WARNING: expected 4 combos (k_cps in {{0,3}} x mode in "
              f"{{static,dynamic}}), found {len(combo_rows)}. Proceeding anyway "
              f"but this sweep-root may not be the intended shape.")

    (out_dir / "tab_throughput_results.tex").write_text(build_throughput_table(combo_rows))
    print(f"Wrote -> {out_dir / 'tab_throughput_results.tex'}")

    (out_dir / "tab_delay_ripple.tex").write_text(build_delay_ripple_table(combo_rows))
    print(f"Wrote -> {out_dir / 'tab_delay_ripple.tex'}")

    fig_path = build_runway_load_balance_figure(combo_rows, out_dir)
    print(f"Wrote -> {fig_path}")

    if GROOT_TABLE11_VIOLATIONS_BY_METHOD is not None:
        (out_dir / "tab_groot_table11_reference.tex").write_text(build_groot_table11_reference_table())
        print(f"Wrote -> {out_dir / 'tab_groot_table11_reference.tex'}")

    print(f"\nLoading fairness_weight calibration data from {args.fairness_sweep_roots} ...")
    (out_dir / "tab_fairness_weight_calibration.tex").write_text(
        build_fairness_weight_calibration_table(args.fairness_sweep_roots)
    )
    print(f"Wrote -> {out_dir / 'tab_fairness_weight_calibration.tex'}")

    (out_dir / "tab_ratchet_ablation.tex").write_text(build_ratchet_ablation_tables())
    print(f"Wrote -> {out_dir / 'tab_ratchet_ablation.tex'}")

    print(f"\nRepointing step10_deep_analysis.py fig1-4 at {out_dir} ...")
    ok = repoint_deep_analysis_figures(args.sweep_root, out_dir)
    print("fig1-4 repoint: " + ("OK" if ok else "FAILED (see warning above)"))

    print("\n" + "=" * 100)
    print(f"DONE. Report artifacts written to {out_dir}/")
    if GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY is None:
        print("REMINDER: Groot et al. Table 11/12 numbers not yet populated in this "
              "script (GROOT_TABLE12_MA_VIOLATIONS_BY_RUNWAY) -- fig:runway_load_balance "
              "has no comparison panel until they're pasted in.")
    print("=" * 100)


if __name__ == "__main__":
    main()
