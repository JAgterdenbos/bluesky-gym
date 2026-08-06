"""
cps_coordination/scripts/step10_deep_analysis.py
--------------------------------------------------
Deep, reproducible re-analysis of the M=100, 8-combo "step 10 sanity sweep"
(k_cps in {0,3} x mode in {static,dynamic} x fairness_weight in {0.0,0.3}),
per ``.claude/plans/step10_sweep_deep_analysis_plan.md``.

Purpose (distinct from the pre-launch gate check in
``phase3_cps_coordination_plan.md``): not "is it safe to launch" but "do we
understand the system" -- broad data sanity, how CPSManager/ETASurrogate/the
frozen SAC worker interact, and the stalling mechanism in depth.

This script also documents and quantifies a same-step-slot-recycling state
carryover bug discovered during this analysis (see the "SLOT-RECYCLING BUG"
section below and the accompanying report,
``.claude/plans/step10_deep_analysis_findings.md``, for the full narrative
and code-level root cause). It is READ-ONLY on production code
(``cps_manager.py``, ``coordination_baseline.py``, ``metrics.py``) --
nothing here is a fix, only measurement.

Reuses ``cps_metrics_offline.recompute_metrics``/``summarize_batch_sweep``'s
combo-discovery rather than reimplementing metric recomputation, and
``spatial_visitation_analysis.py``'s ``compute_tortuosity``/``build_heatmap``
for trajectory-shape analysis (both already the pattern
``cps_metrics_offline.py`` follows).

Usage
-----
  python cps_coordination/scripts/step10_deep_analysis.py \\
      --sweep-root cps_coordination/data/step10_sanity_sweep_regenerated

  # Diff headline metrics against another already-analyzed sweep (e.g. a
  # ratchet on/off or spawn_window_s before/after comparison):
  python cps_coordination/scripts/step10_deep_analysis.py \\
      --sweep-root cps_coordination/data/<new_sweep> \\
      --compare-root cps_coordination/data/<old_sweep>

Runtime: seconds (pure Parquet + pandas/numpy/matplotlib, no BlueSky/SB3
import, no simulation) -- this is entirely offline analysis of already-
collected telemetry. See ``regenerate_step10_sanity_sweep.sh`` in this
directory for how the underlying data was originally collected (~1.5-2h,
NOT run by this script).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cps_coordination.scripts.cps_metrics_offline import (
    explode_trajectories, load_recat_matrix, load_telemetry, recompute_metrics,
    recompute_separation_compliance,
)
from cps_coordination.scripts.summarize_batch_sweep import _COMBO_RE, discover_combos
from path_planning.rta.testing.spatial_visitation_analysis import (
    build_heatmap, compute_information_metrics, compute_tortuosity,
)

_DPI = 180


def _apply_style() -> None:
    """Mirrors surrogate_analyse.py::_apply_style() -- same rcParams block,
    so figures from this script read as one system with the rest of the
    package's plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F7F7F7",
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": "#333333",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 8,
        "figure.dpi": _DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def _combo_key(name: str) -> str:
    m = _COMBO_RE.match(name)
    return f"k{m.group('k_cps')}_{m.group('mode')}"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading + occurrence annotation
# ──────────────────────────────────────────────────────────────────────────────


def load_all_combos(sweep_root: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load every combo's (aircraft_df, separation_df), annotated with a
    same-slot ``occurrence`` column (1 = first physical aircraft to occupy
    a given (episode_id, acid) slot that episode, 2 = second, ...).

    ``occurrence`` is derived from Parquet row order (validated below to
    agree 100% with an independent ``actual_landing_time``-rank derivation
    on every combo -- row order is chronological, since the collector
    appends records in the order ``_run_episode`` produces terminations).
    """
    combos = discover_combos(sweep_root)
    out = {}
    for combo_dir in combos:
        name = os.path.basename(combo_dir)
        aircraft_df, separation_df = load_telemetry(combo_dir)
        aircraft_df = aircraft_df.reset_index(drop=True)
        aircraft_df["row_order"] = aircraft_df.index
        aircraft_df["occurrence"] = (
            aircraft_df.groupby(["episode_id", "acid"])["row_order"]
            .rank(method="first").astype(int)
        )
        out[name] = (aircraft_df, separation_df)
    return out


def validate_occurrence_ordering(all_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    print("\n" + "=" * 100)
    print("VALIDATION: row-order-derived `occurrence` vs actual_landing_time-derived rank")
    print("=" * 100)
    for name, (df, _) in all_data.items():
        by_land = df.groupby(["episode_id", "acid"])["actual_landing_time"].rank(method="first").astype(int)
        agree = float((df["occurrence"] == by_land).mean())
        print(f"  {name:<18} agreement={agree:.4f}  (1.0 required for `occurrence` to be trustworthy)")
        assert agree == 1.0, f"{name}: row order does not match landing-time order -- occurrence proxy invalid"


# ──────────────────────────────────────────────────────────────────────────────
# Workstream 1: broad sanity sweep
# ──────────────────────────────────────────────────────────────────────────────


def workstream1_sanity(all_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], recat) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("WORKSTREAM 1: broad sanity sweep across all 8 combos")
    print("=" * 100)

    rows = []
    for name, (df, sep) in all_data.items():
        m = _COMBO_RE.match(name)
        rec = {
            "combo": name, "k_cps": int(m.group("k_cps")), "mode": m.group("mode"),
            "fw": float(m.group("fw")),
            "n_rows": len(df),
            "n_nan_rta_cps": int(df["rta_error_cps"].isna().sum()),
            "n_inf_any": int(np.isinf(
                df[[c for c in ("assigned_tta", "actual_landing_time", "rta_error_cps",
                                 "rta_error_static", "rta_error_solo") if c in df.columns]]
                .astype(float)
            ).sum().sum()),
            "wake_cat_uniform_C": bool((df["wake_cat"] == "C").all()),
            "n_runways_used": int(df["runway_id"].nunique()),
            "death_causes": {k: int(v) for k, v in df["death_cause"].fillna("success").value_counts().items()},
            # death_cause is never null (every row is a terminated aircraft) and is
            # NOT a proxy for `success`: death_cause=="success" means "crossed the
            # correct runway's sink" (a spatial/geometric outcome), while `success`
            # additionally requires landing within the RTA tolerance window
            # (multi_agent_pathplanning_env.py::_get_info: is_success = on_time AND
            # correct_runway). So success==True must imply death_cause=="success",
            # but the converse does NOT hold (a geometrically-correct-but-late
            # landing is death_cause="success" with success=False) -- only check
            # the direction that's actually required.
            "death_cause_success_consistent": bool(
                (~df["success"] | (df["death_cause"] == "success")).all()
            ),
            "neg_sep_gaps": int((sep["gap_actual_s"] < 0).sum()) if not sep.empty else 0,
            "flight_id_collision_rows": int(df.duplicated(subset=["episode_id", "flight_id"], keep=False).sum()),
            "flight_id_collision_pct": round(
                100 * df.duplicated(subset=["episode_id", "flight_id"], keep=False).mean(), 1
            ),
        }
        c_sep_manual = recompute_separation_compliance(sep, 5.0)
        metrics = recompute_metrics(df, sep, recat)
        rec["success_rate_reported"] = metrics["success_rate"]
        rec["success_rate_manual"] = round(float(df["success"].mean()), 4)
        rec["c_sep_reported"] = metrics["c_sep"]
        rec["c_sep_manual"] = round(c_sep_manual, 4) if not np.isnan(c_sep_manual) else "nan"
        rows.append(rec)

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out[["combo", "n_rows", "n_nan_rta_cps", "n_inf_any", "wake_cat_uniform_C",
                    "n_runways_used", "death_cause_success_consistent", "neg_sep_gaps",
                    "flight_id_collision_pct", "success_rate_reported", "success_rate_manual",
                    "c_sep_reported", "c_sep_manual"]].to_string(index=False))
    print("\ndeath_cause breakdowns:")
    for _, r in out.iterrows():
        print(f"  {r['combo']:<18} {r['death_causes']}")

    contaminated = out[out["flight_id_collision_pct"] > 1.0]
    if not contaminated.empty:
        print(
            "\n" + "!" * 100 + "\n"
            "WARNING: flight_id collisions above 1% detected in "
            f"{len(contaminated)}/{len(out)} combo(s):\n"
            f"  {', '.join(contaminated['combo'])}\n"
            "This is the signature of the pre-fix same-step slot-recycling state-carryover\n"
            "bug (see slot_recycling_bug_report below) -- this sweep-root is almost certainly\n"
            "STALE/PRE-FIX DATA, not the validated post-fix telemetry. Headline metrics below\n"
            "(success_rate, c_sep, stall_rate, etc.) should NOT be reported or compared against\n"
            "post-fix numbers until you've confirmed which code generated this sweep-root.\n"
            + "!" * 100
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Slot-recycling bug quantification
# ──────────────────────────────────────────────────────────────────────────────


def slot_recycling_bug_report(all_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("SLOT-RECYCLING STATE-CARRYOVER BUG: quantification")
    print("=" * 100)
    print(
        "Root cause (read-only finding, not fixed here): _run_episode's arrival-detection\n"
        "(`current_acids - prev_active_acids`, coordination_baseline.py ~L667-675) and\n"
        "CPSManager.update_fleet's eviction detection (`old_acids - new_acids`,\n"
        "cps_manager.py ~L323-326) both use a *lagged* one-step-behind set difference to\n"
        "detect 'this acid string just departed'. MultiAgentPathPlanningGoalEnv._finalize_step\n"
        "refills a freed slot with the next scheduled arrival *within the same env.step() call*\n"
        "that terminated the previous occupant, reusing the identical acid string\n"
        "(f'AC{slot:03d}'). When that happens, the acid never appears absent from the observed\n"
        "acid-string set between two consecutive loop iterations, so neither detector fires --\n"
        "every per-acid accumulator keyed on that detection (arrival_order/assigned_once/\n"
        "mid_traj_updated/last_tta/frozen_remaining_time_budget in _run_episode; \n"
        "_best_distance_km/_best_distance_time/_stalled_acids/_frozen_eta/_frozen_runway/\n"
        "_frozen_tta in CPSManager) silently persists from the OLD physical aircraft into the\n"
        "NEW one occupying the same slot.\n"
    )

    rows = []
    for name, (df, _) in all_data.items():
        occ_stall = df.groupby("occurrence")["stall_detected"].mean()
        occ_counts = df["occurrence"].value_counts().sort_index()

        # Runway-hijack signature: for occurrence>=2 rows, does runway_id match
        # the immediately preceding occupant of the same slot far above the
        # ~1/n_runways baseline expected under independent assignment?
        g = df.sort_values(["episode_id", "acid", "occurrence"]).copy()
        g["prev_runway"] = g.groupby(["episode_id", "acid"])["runway_id"].shift(1)
        later = g[g["occurrence"] >= 2]
        runway_match_rate = float((later["runway_id"] == later["prev_runway"]).mean()) if len(later) else float("nan")
        baseline_match_rate = float((df["runway_id"].value_counts(normalize=True) ** 2).sum())

        # rta_error_solo join-corruption signature: among flight_id collision
        # groups, fraction where ALL rows share an identical rta_error_solo
        # despite distinct actual_landing_time (== cross-contaminated join).
        dup_mask = df.duplicated(subset=["episode_id", "flight_id"], keep=False)
        groups = df[dup_mask].groupby(["episode_id", "flight_id"])
        n_groups = groups.ngroups
        n_identical_solo_distinct_landing = sum(
            1 for _, gr in groups
            if gr["rta_error_solo"].nunique(dropna=False) == 1 and gr["actual_landing_time"].nunique() > 1
        )

        rows.append({
            "combo": name,
            "stall_occ1": round(float(occ_stall.get(1, float("nan"))), 4),
            "stall_occ2": round(float(occ_stall.get(2, float("nan"))), 4),
            "stall_occ3plus": round(float(occ_stall[occ_stall.index >= 3].mean()), 4) if (occ_stall.index >= 3).any() else float("nan"),
            "stall_pooled_reported": round(float(df["stall_detected"].mean()), 4),
            "n_occ1": int(occ_counts.get(1, 0)),
            "n_occ2plus": int(occ_counts[occ_counts.index >= 2].sum()),
            "runway_hijack_match_rate": round(runway_match_rate, 4),
            "runway_hijack_baseline": round(baseline_match_rate, 4),
            "solo_join_corrupted_groups": n_identical_solo_distinct_landing,
            "solo_join_total_collision_groups": n_groups,
        })

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.to_string(index=False))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Clean-subset (occurrence==1) headline metrics vs pooled
# ──────────────────────────────────────────────────────────────────────────────


def clean_vs_pooled_headline(all_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], recat) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("CLEAN-SUBSET (occurrence==1 only) vs POOLED headline metrics")
    print("=" * 100)
    print(
        "occurrence==1 rows are the first physical occupant of each (episode, acid) slot --\n"
        "never subject to the slot-recycling carryover above (CPSManager.reset() runs fresh at\n"
        "the start of every episode). This is the most trustworthy subset for characterizing\n"
        "the SYSTEM's genuine behaviour, at the cost of ~half the sample (n_occ1=500/1000).\n"
    )
    rows = []
    for name, (df, sep) in all_data.items():
        m = _COMBO_RE.match(name)
        clean = df[df["occurrence"] == 1]
        # Separation pairs can't be cleanly attributed to "occurrence==1 only"
        # without re-deriving pairs (a compliant pair needs BOTH landings to be
        # clean) -- keep c_sep on the full separation_df (unaffected by this
        # bug family; see note in the report) and only clean success/stall here.
        rows.append({
            "combo": name,
            "success_pooled": round(float(df["success"].mean()), 4),
            "success_clean": round(float(clean["success"].mean()), 4),
            "stall_pooled": round(float(df["stall_detected"].mean()), 4),
            "stall_clean": round(float(clean["stall_detected"].mean()), 4),
            "stall_unrecovered_pooled": round(float((df["stall_detected"] & ~df["success"]).mean()), 4),
            "stall_unrecovered_clean": round(float((clean["stall_detected"] & ~clean["success"]).mean()), 4),
        })
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(out.to_string(index=False))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Workstream 2: CPSManager / ETASurrogate / frozen-worker interaction
# ──────────────────────────────────────────────────────────────────────────────


def workstream2_model_interaction(all_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    print("\n" + "=" * 100)
    print("WORKSTREAM 2: CPSManager / ETASurrogate / frozen-worker interaction (clean subset)")
    print("=" * 100)
    for name, (df, _) in all_data.items():
        clean = df[df["occurrence"] == 1]
        upd = clean[clean["tta_updated_mid_trajectory"]]
        r_rec_clean = float((upd["rta_error_cps"].abs() <= 60.0).mean()) if len(upd) else float("nan")
        print(
            f"  {name:<18} n_clean={len(clean):4d}  "
            f"mid_traj_updated_rate={clean['tta_updated_mid_trajectory'].mean():.3f}  "
            f"R_rec(clean)={r_rec_clean:.3f}  "
            f"|rta_error_cps| mean={clean['rta_error_cps'].abs().mean():8.1f}s  "
            f"median={clean['rta_error_cps'].abs().median():8.1f}s  "
            f"p90={clean['rta_error_cps'].abs().quantile(0.9):8.1f}s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Workstream 3: stalling deep-dive
# ──────────────────────────────────────────────────────────────────────────────


def workstream3_stalling(
    all_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], out_dir: Path,
) -> None:
    print("\n" + "=" * 100)
    print("WORKSTREAM 3: stalling deep-dive (clean subset, occurrence==1)")
    print("=" * 100)

    # --- fairness_weight effect on stall_recovery_rate at k_cps=3, clean subset ---
    print("\n-- fairness_weight effect on stall_recovery_rate (k_cps=3, occurrence==1 only) --")
    for mode in ("dynamic", "static"):
        for fw in (0.0, 0.3):
            name = f"k3_{mode}_fw{fw:g}"
            if name not in all_data:
                continue
            df, _ = all_data[name]
            clean = df[df["occurrence"] == 1]
            stalled = clean[clean["stall_detected"]]
            rec_rate = float(stalled["success"].mean()) if len(stalled) else float("nan")
            print(f"  {name:<18} n_stalled_clean={len(stalled):4d}  stall_recovery_rate={rec_rate:.4f}")

    # --- death_cause breakdown for stalled vs non-stalled (clean subset) ---
    print("\n-- death_cause | stalled vs non-stalled (clean subset) --")
    for name, (df, _) in all_data.items():
        clean = df[df["occurrence"] == 1]
        for flag, label in ((True, "stalled"), (False, "non-stalled")):
            sub = clean[clean["stall_detected"] == flag]
            if sub.empty:
                continue
            dc = sub["death_cause"].fillna("success").value_counts(normalize=True).round(3).to_dict()
            print(f"  {name:<18} {label:<12} n={len(sub):4d}  {dc}")

    # --- trajectory tortuosity: stalled vs non-stalled, both modes (traj_x/traj_y
    # logged for the CPS pass only per telemetry.py, but that pass runs regardless
    # of runway_assignment_mode, so static-mode combos have trajectories too) ---
    print("\n-- trajectory tortuosity: stalled vs non-stalled (clean subset, all combos) --")
    tort_records = []
    for name, (df, _) in all_data.items():
        clean = df[df["occurrence"] == 1].copy()
        clean = clean[clean["traj_x"].apply(len) >= 3]  # need >=3 points for a meaningful shape
        if clean.empty:
            continue
        for flag in (True, False):
            sub = clean[clean["stall_detected"] == flag]
            if sub.empty:
                continue
            pc = explode_trajectories(sub.assign(episode_id=sub["flight_id"]))  # per-flight grouping
            if pc.empty:
                continue
            per_flight_tort = []
            for _, g in pc.groupby("episode"):
                if len(g) < 3:
                    continue
                per_flight_tort.append(compute_tortuosity(g.assign(episode="x")))
            if not per_flight_tort:
                continue
            arr = np.array(per_flight_tort)
            tort_records.append({
                "combo": name, "stalled": flag, "n": len(arr),
                "tortuosity_mean": float(np.mean(arr)), "tortuosity_median": float(np.median(arr)),
                "tortuosity_p90": float(np.quantile(arr, 0.9)),
            })
            print(f"  {name:<18} stalled={flag!s:<5} n={len(arr):4d}  "
                  f"tortuosity mean={np.mean(arr):7.2f}  median={np.median(arr):7.2f}  p90={np.quantile(arr,0.9):7.2f}")

    if tort_records and any(r["stalled"] for r in tort_records):
        tort_df = pd.DataFrame(tort_records)
        tort_df["mode"] = tort_df["combo"].apply(lambda n: _COMBO_RE.match(n).group("mode"))
        print("\n-- tortuosity ratio (stalled/non-stalled median), by mode --")
        piv = tort_df.pivot_table(index=["combo", "mode"], columns="stalled",
                                   values="tortuosity_median").rename(columns={True: "stalled", False: "non_stalled"})
        if "stalled" in piv.columns and "non_stalled" in piv.columns:
            piv["ratio"] = piv["stalled"] / piv["non_stalled"]
            for mode, sub in piv.groupby("mode"):
                print(f"  mode={mode:<8} ratio range {sub['ratio'].min():.2f}-{sub['ratio'].max():.2f}, "
                      f"mean {sub['ratio'].mean():.2f}  (n_combos={len(sub)})")
    elif tort_records:
        print("\n-- tortuosity ratio (stalled/non-stalled median), by mode --")
        print("  no stalled aircraft in this sweep -- ratio undefined, nothing to compare")

    # --- Figure: stall-rate-by-occurrence bar chart (the headline bug-cascade figure) ---
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.1
    combos_sorted = sorted(all_data.keys(), key=lambda n: (_COMBO_RE.match(n).group("mode"),
                                                             _COMBO_RE.match(n).group("k_cps"),
                                                             _COMBO_RE.match(n).group("fw")))
    x = np.arange(4)  # occurrence 1,2,3,4+
    for i, name in enumerate(combos_sorted):
        df, _ = all_data[name]
        occ = df["occurrence"].clip(upper=4)
        rates = occ.groupby(occ).apply(lambda idx: df.loc[idx.index, "stall_detected"].mean())
        rates = rates.reindex([1, 2, 3, 4])
        ax.bar(x + i * width - 3.5 * width, rates.values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["1st\n(clean)", "2nd", "3rd", "4th+"])
    ax.set_xlabel("Occurrence of this acid-slot within the episode")
    ax.set_ylabel("stall_detected rate")
    ax.set_title("Stall-flag cascade across slot-recycling occurrences\n(all 8 combos)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.savefig(out_dir / "fig1_stall_cascade_by_occurrence.png", dpi=_DPI)
    plt.close(fig)
    print(f"\nSaved -> {out_dir / 'fig1_stall_cascade_by_occurrence.png'}")

    # --- Figure: runway-hijack match rate vs baseline ---
    fig, ax = plt.subplots(figsize=(7, 5))
    names, match_rates = [], []
    for name in combos_sorted:
        df, _ = all_data[name]
        g = df.sort_values(["episode_id", "acid", "occurrence"]).copy()
        g["prev_runway"] = g.groupby(["episode_id", "acid"])["runway_id"].shift(1)
        later = g[g["occurrence"] >= 2]
        match_rates.append(float((later["runway_id"] == later["prev_runway"]).mean()))
        names.append(name)
    baseline = float(next(iter(all_data.values()))[0]["runway_id"].value_counts(normalize=True).pow(2).sum())
    ax.bar(names, match_rates, color="#C44E52")
    ax.axhline(baseline, color="#333333", linestyle="--", linewidth=1.2,
               label=f"independent-assignment baseline ({baseline:.2f})")
    ax.set_ylabel("P(runway_id == previous slot occupant's runway_id)")
    ax.set_title("Runway-assignment \"hijack\" signature\n(occurrence >= 2 rows, all combos)")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "fig2_runway_hijack_signature.png", dpi=_DPI)
    plt.close(fig)
    print(f"Saved -> {out_dir / 'fig2_runway_hijack_signature.png'}")

    # --- Figure: clean-subset stall_recovery_rate by fairness_weight (k_cps=3) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, mode in zip(axes, ("dynamic", "static")):
        fws, rates = [], []
        for fw in (0.0, 0.3):
            name = f"k3_{mode}_fw{fw:g}"
            if name not in all_data:
                continue
            df, _ = all_data[name]
            clean = df[df["occurrence"] == 1]
            stalled = clean[clean["stall_detected"]]
            rates.append(float(stalled["success"].mean()) if len(stalled) else float("nan"))
            fws.append(str(fw))
        ax.bar(fws, rates, color=["#4C72B0", "#DD8452"])
        ax.set_title(f"k_cps=3, mode={mode}")
        ax.set_xlabel("fairness_weight")
    axes[0].set_ylabel("stall_recovery_rate (clean subset)")
    fig.suptitle("fairness_weight effect on stall recovery (occurrence==1 only)")
    fig.savefig(out_dir / "fig3_fairness_weight_stall_recovery_clean.png", dpi=_DPI)
    plt.close(fig)
    print(f"Saved -> {out_dir / 'fig3_fairness_weight_stall_recovery_clean.png'}")

    # --- Figure: spatial heatmap, stalled vs non-stalled, dynamic AND static (k3, clean) ---
    # Combo naming isn't fixed across sweep layouts: the old 8-combo sanity
    # sweep used a shared fw in {0.0, 0.3} per mode, while the production
    # 4-combo layout deploys a per-mode calibrated fw (no "fw0" combo exists
    # for either mode there) -- so this picks whatever single k3 combo is
    # present per mode instead of assuming "fw0", preferring fw=0.0 when
    # multiple fw values for the same mode are present (old-layout baseline).
    def _pick_k3_combo(mode: str) -> str | None:
        candidates = sorted(
            (n for n in all_data if (m := _COMBO_RE.match(n)) and m.group("k_cps") == "3" and m.group("mode") == mode),
            key=lambda n: float(_COMBO_RE.match(n).group("fw")),
        )
        if not candidates:
            return None
        zero_fw = [n for n in candidates if float(_COMBO_RE.match(n).group("fw")) == 0.0]
        return zero_fw[0] if zero_fw else candidates[0]

    mode_to_combo = {m: _pick_k3_combo(m) for m in ("dynamic", "static")}
    modes_present = [m for m, name in mode_to_combo.items() if name is not None]
    if modes_present:
        fig, axes = plt.subplots(len(modes_present), 2, figsize=(11, 5 * len(modes_present)), squeeze=False)
        for row, mode in enumerate(modes_present):
            name = mode_to_combo[mode]
            df, _ = all_data[name]
            clean = df[(df["occurrence"] == 1) & (df["traj_x"].apply(len) >= 3)]
            for ax, flag, title in zip(axes[row], (True, False), ("Stalled", "Non-stalled")):
                sub = clean[clean["stall_detected"] == flag]
                if sub.empty:
                    ax.set_title(f"{mode}: {title} (n=0)")
                    continue
                pc = explode_trajectories(sub.assign(episode_id=sub["flight_id"]))
                if pc.empty:
                    ax.set_title(f"{mode}: {title} (n=0 points)")
                    continue
                _, H_log, vmin, vmax, extent, _ = build_heatmap(pc, bins=150)
                ax.imshow(H_log, origin="lower", extent=extent, cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")
                ax.set_title(f"{mode}: {title} (n_flights={sub.shape[0]})")
        combo_desc = ", ".join(f"{m}={mode_to_combo[m]}" for m in modes_present)
        fig.suptitle(f"Spatial visitation: k_cps=3, occurrence==1 only ({combo_desc})")
        fig.savefig(out_dir / "fig4_spatial_stalled_vs_nonstalled_k3.png", dpi=_DPI)
        plt.close(fig)
        print(f"Saved -> {out_dir / 'fig4_spatial_stalled_vs_nonstalled_k3.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-sweep comparison (e.g. ratchet on/off, spawn_window_s 0 vs 1800, any
# two already-analyzed sweep-roots) -- formalizes the "old -> new" tables that
# otherwise get built by hand, combo by combo, every time two sweeps need
# comparing (see phase3_cps_coordination_plan.md's pre/post-fix tables).
# ──────────────────────────────────────────────────────────────────────────────


def compare_sweeps(
    new_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    old_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    recat,
    new_label: str, old_label: str,
) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print(f"COMPARISON: {old_label!r} (old) -> {new_label!r} (new)")
    print("=" * 100)

    common = sorted(set(new_data) & set(old_data))
    missing_new = sorted(set(old_data) - set(new_data))
    missing_old = sorted(set(new_data) - set(old_data))
    if missing_new:
        print(f"  combos only in old, skipped: {missing_new}")
    if missing_old:
        print(f"  combos only in new, skipped: {missing_old}")

    def _row(df: pd.DataFrame, sep: pd.DataFrame) -> dict:
        metrics = recompute_metrics(df, sep, recat)
        g = df.sort_values(["episode_id", "acid", "occurrence"]).copy()
        g["prev_runway"] = g.groupby(["episode_id", "acid"])["runway_id"].shift(1)
        later = g[g["occurrence"] >= 2]
        hijack = float((later["runway_id"] == later["prev_runway"]).mean()) if len(later) else float("nan")
        baseline = float((df["runway_id"].value_counts(normalize=True) ** 2).sum())
        return {
            "success_rate": metrics["success_rate"],
            "c_sep": metrics["c_sep"],
            "stall_rate": round(float(df["stall_detected"].mean()), 4),
            "stall_unrecovered": round(float((df["stall_detected"] & ~df["success"]).mean()), 4),
            "flight_id_collision_pct": round(100 * df.duplicated(subset=["episode_id", "flight_id"], keep=False).mean(), 1),
            "runway_hijack_match_rate": round(hijack, 4) if hijack == hijack else float("nan"),
            "runway_hijack_baseline": round(baseline, 4),
        }

    print(f"  (cells below read as '{old_label} -> {new_label}')")

    rows = []
    for name in common:
        old_row = _row(*old_data[name])
        new_row = _row(*new_data[name])
        rec = {"combo": name}
        for key in old_row:
            rec[key] = f"{old_row[key]:.4g} -> {new_row[key]:.4g}"
        rows.append(rec)

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_colwidth", 40):
        print(out.to_string(index=False))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-root", type=str, default="cps_coordination/data/step10_sanity_sweep_regenerated",
                   help="Sweep to analyze. Defaults to the validated post-fix regenerated sweep -- "
                        "NOT cps_coordination/data/step10_sanity_sweep, which is stale pre-fix data "
                        "kept around only for historical before/after comparisons.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Figure output dir. Defaults to cps_coordination/figures/stall_analysis__<sweep-root "
                        "basename> so analyzing different sweeps never silently overwrites each other's figures.")
    p.add_argument("--compare-root", type=str, default=None,
                   help="Optional second sweep-root (e.g. a before/after or ratchet on/off run) to diff "
                        "headline metrics against, combo by combo. --sweep-root is treated as 'new', "
                        "--compare-root as 'old'.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"cps_coordination/figures/stall_analysis__{Path(args.sweep_root).name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    recat = load_recat_matrix()
    all_data = load_all_combos(args.sweep_root)
    print(f"Loaded {len(all_data)} combos from {args.sweep_root}")

    validate_occurrence_ordering(all_data)
    workstream1_sanity(all_data, recat)
    slot_recycling_bug_report(all_data)
    clean_vs_pooled_headline(all_data, recat)
    workstream2_model_interaction(all_data)
    workstream3_stalling(all_data, out_dir)

    if args.compare_root:
        old_data = load_all_combos(args.compare_root)
        print(f"Loaded {len(old_data)} combos from {args.compare_root} (comparison baseline)")
        validate_occurrence_ordering(old_data)
        compare_sweeps(all_data, old_data, recat, Path(args.sweep_root).name, Path(args.compare_root).name)

    print("\n" + "=" * 100)
    print(f"DONE. Figures written to {out_dir}/")
    print("=" * 100)


if __name__ == "__main__":
    main()
