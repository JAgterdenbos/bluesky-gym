"""
cps_coordination/scripts/analyze_hysteresis_cap_resweep.py
------------------------------------------------------------
Analysis for the concurrency-cap resweep + reassignment-guard-timing
sensitivity sweep
(.claude/plans/concurrency_cap_and_reassignment_guard_resweep.md), applying
that plan's exact decision rules to the grid produced by
``run_hysteresis_cap_resweep.sh``.

Per-cell metrics (reusing the existing offline pipeline, not a new one):
  - gamma (current, post-f20e155 per-episode combined-runway definition),
    delta_epsilon_vs_static, mean_flight_time_s (tau), stall_rate,
    stall_unrecovered -- all via cps_metrics_offline.recompute_metrics
    (same function summarize_batch_sweep.py wraps for the k_cps x mode
    grid). mean_flight_time_s was ADDED to the telemetry schema
    (flight_time_s column, _EpisodeRecord/AIRCRAFT_COLUMNS/
    AircraftTelemetryRow) specifically for this resweep, 2026-08-20 --
    neither the on-disk cps_eval_aircraft.parquet nor the in-process
    _EpisodeRecord carried a spawn/start-time field before that, only
    actual_landing_time (global episode clock, not per-aircraft flight
    duration). Any telemetry collected before that change has no
    flight_time_s column and mean_flight_time_s reads as NaN for it.
  - death_cause == "wrong_runway" count/fraction -- direct from
    cps_eval_aircraft.parquet, not in recompute_metrics's own output.
  - switch_rate / true_oscillator_count -- from cps_eval_reassignment.parquet
    (only present when --log-reassignment-events was on), reusing the exact
    "true back-and-forth oscillator" definition from
    scratchpad/analyze_reassignment.py (switch_count > n_distinct_runways-1
    for an acid visiting >1 runway in one episode).

Decision rules (verbatim from the plan doc):
  - Cap: per hysteresis slice, walk the grid in ascending cap order; the
    "cumulative Gamma range" denominator is the FULL slice's
    Gamma(max_cap) - Gamma(min_cap) (fixed once per slice), NOT a running
    partial sum -- this specific point is what the plan's v2 revision
    corrected out of v1's staged-bracket design (a bracket subset doesn't
    have the full range to normalize against). Stop at the first cap where
    the next step's Gamma gain is <10% of that full range; report the cap
    *before* that step. Reverse-engineered and verified byte-for-byte
    against the original capacity sweep's worked numbers in
    phase3_cps_coordination_findings.md (35->42 = 5.8%, 27->35 = 16.3%,
    etc.) before being trusted here.
  - Cap robustness: compare the winning cap across all hysteresis slices.
  - Hysteresis: at the resolved cap, pick the value maximizing Gamma,
    subject to (a) delta_epsilon_vs_static not regressing vs. 240s and
    (b) switch/thrash rate not reproducing Vector 8's pre-fix
    order-of-magnitude blowup (~6 switches/aircraft over M=30). Ties
    within noise keep 240s (status quo).

Run: python cps_coordination/scripts/analyze_hysteresis_cap_resweep.py \
    --sweep-root experiments/cps_eval/hysteresis_cap_resweep_<timestamp>
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cps_coordination.scripts.cps_metrics_offline import (
    load_recat_matrix,
    load_telemetry,
    recompute_metrics,
)

_CELL_RE = re.compile(r"^cap(?P<cap>\d+)_hyst(?P<hyst>[\d.]+)$")

# Vector 8's pre-fix thrashing pathology, for the hysteresis guardrail
# below -- ~8,950 switches / (30 episodes x 50 concurrent aircraft) ~= 6.0
# switches/aircraft, see cps_manager.py's REASSIGNMENT_HYSTERESIS_S
# docstring. A guardrail multiple of this, not an exact reproduction
# requirement -- any candidate landing within an order of magnitude of this
# is the thing to flag, not just an exact match.
_VECTOR8_SWITCHES_PER_AIRCRAFT = 6.0
_THRASH_GUARDRAIL_FRACTION = 0.5  # flag if switches/aircraft exceeds half of Vector 8's blowup


def discover_cells(sweep_root: str, k_cps: int, mode: str) -> List[Tuple[int, float, str]]:
    cells = []
    for d in sorted(glob.glob(os.path.join(sweep_root, "cap*_hyst*"))):
        base = os.path.basename(d)
        m = _CELL_RE.match(base)
        if not m:
            continue
        combo_dir = os.path.join(d, f"k{k_cps}_{mode}")
        if not os.path.exists(os.path.join(combo_dir, "cps_eval_aircraft.parquet")):
            continue
        cells.append((int(m.group("cap")), float(m.group("hyst")), combo_dir))
    return cells


def _true_oscillator_count(reassignment_df: pd.DataFrame) -> int:
    """Verbatim port of scratchpad/analyze_reassignment.py's oscillation
    check: an acid that visits >1 distinct runway in one episode AND whose
    switch-event count exceeds (n_distinct_runways - 1) must have gone
    back-and-forth at least once, not just moved monotonically toward a
    better runway."""
    if reassignment_df.empty:
        return 0
    per_ac = reassignment_df.groupby(["episode_id", "acid"])["current_runway"].nunique()
    multi = per_ac[per_ac > 1]
    if not len(multi):
        return 0
    switch_counts = (
        reassignment_df[reassignment_df["switched"]].groupby(["episode_id", "acid"]).size()
    )
    osc = switch_counts[switch_counts.index.isin(multi.index)]
    if not len(osc):
        return 0
    return int((osc > (multi.reindex(osc.index) - 1)).sum())


def _num(x: Any) -> float:
    """cps_metrics_offline.recompute_metrics returns the string literal
    "nan" (not float NaN) for several fields' NaN case (delta_epsilon_vs_static,
    stall_rate, stall_unrecovered, mean_flight_time_s, the *_std companions,
    etc.) -- a pretty-printing convention from that module, not a numeric
    one. Left un-normalized, pd.notna()/comparisons downstream would treat
    the string "nan" as a valid, truthy, non-null value (it's a non-empty
    string) instead of missing data. Normalize once here rather than
    re-deriving this everywhere it's consumed."""
    if isinstance(x, str):
        return float("nan") if x == "nan" else float(x)
    return float(x) if x is not None else float("nan")


def analyze_cell(cap: int, hyst: float, combo_dir: str, recat_matrix: Dict) -> Dict[str, Any]:
    aircraft_df, separation_df = load_telemetry(combo_dir)
    metrics = recompute_metrics(aircraft_df, separation_df, recat_matrix)

    n_aircraft = len(aircraft_df)
    n_episodes = int(aircraft_df["episode_id"].nunique()) if n_aircraft else 0
    wrong_runway_n = int((aircraft_df["death_cause"] == "wrong_runway").sum()) if n_aircraft else 0

    row: Dict[str, Any] = {
        "cap": cap,
        "hyst_s": hyst,
        "n_episodes": n_episodes,
        "n_aircraft": n_aircraft,
        "gamma": _num(metrics.get("gamma", float("nan"))),
        "gamma_std": _num(metrics.get("gamma_std", float("nan"))),
        "success_rate": _num(metrics.get("success_rate", float("nan"))),
        "delta_epsilon_vs_static": _num(metrics.get("delta_epsilon_vs_static", float("nan"))),
        "mean_flight_time_s": _num(metrics.get("mean_flight_time_s", float("nan"))),
        "stall_rate": _num(metrics.get("stall_rate", float("nan"))),
        "stall_unrecovered": _num(metrics.get("stall_unrecovered", float("nan"))),
        "wrong_runway_n": wrong_runway_n,
        "wrong_runway_frac": (wrong_runway_n / n_aircraft) if n_aircraft else float("nan"),
    }

    reassignment_path = os.path.join(combo_dir, "cps_eval_reassignment.parquet")
    if os.path.exists(reassignment_path):
        reassignment_df = pd.read_parquet(reassignment_path)
        row["switch_rate"] = float(reassignment_df["switched"].mean()) if len(reassignment_df) else float("nan")
        row["n_switches"] = int(reassignment_df["switched"].sum())
        row["switches_per_aircraft"] = row["n_switches"] / n_aircraft if n_aircraft else float("nan")
        row["true_oscillator_count"] = _true_oscillator_count(reassignment_df)
    else:
        row["switch_rate"] = float("nan")
        row["n_switches"] = None
        row["switches_per_aircraft"] = float("nan")
        row["true_oscillator_count"] = None

    return row


def build_summary(sweep_root: str, k_cps: int, mode: str) -> pd.DataFrame:
    recat_matrix = load_recat_matrix()
    cells = discover_cells(sweep_root, k_cps, mode)
    if not cells:
        raise SystemExit(
            f"No complete cap*_hyst* cells with k{k_cps}_{mode}/cps_eval_aircraft.parquet "
            f"found under {sweep_root!r}."
        )
    rows = [analyze_cell(cap, hyst, combo_dir, recat_matrix) for cap, hyst, combo_dir in cells]
    df = pd.DataFrame(rows).sort_values(["hyst_s", "cap"]).reset_index(drop=True)
    return df


def apply_cap_stopping_rule(slice_df: pd.DataFrame) -> Dict[str, Any]:
    """10%-of-full-cumulative-Gamma-range stopping rule, one hysteresis
    slice at a time. See module docstring for the worked-example derivation
    that pinned down "full range" (not a running partial sum) as the
    correct denominator."""
    s = slice_df.sort_values("cap").reset_index(drop=True)
    caps = s["cap"].to_numpy()
    gammas = s["gamma"].to_numpy(dtype=float)
    full_range = gammas[-1] - gammas[0]

    steps = []
    winner = int(caps[0])
    for i in range(1, len(caps)):
        gain = gammas[i] - gammas[i - 1]
        pct = (gain / full_range * 100.0) if full_range > 0 else float("nan")
        steps.append({"from_cap": int(caps[i - 1]), "to_cap": int(caps[i]),
                       "gamma_gain": round(gain, 4), "pct_of_full_range": round(pct, 2)})
        if pct >= 10.0 or not np.isfinite(pct):
            winner = int(caps[i])
        else:
            break  # first sub-10% step -- stop, winner stays at caps[i-1]

    return {"winner_cap": winner, "full_range": round(full_range, 4), "steps": steps}


def resolve_hysteresis(df: pd.DataFrame, resolved_cap: int, status_quo_hyst: float = 240.0) -> Dict[str, Any]:
    at_cap = df[df["cap"] == resolved_cap].sort_values("hyst_s").reset_index(drop=True)
    if at_cap.empty:
        return {"error": f"no rows at resolved cap={resolved_cap}"}

    baseline = at_cap[at_cap["hyst_s"] == status_quo_hyst]
    baseline_delta_eps = float(baseline["delta_epsilon_vs_static"].iloc[0]) if len(baseline) else float("nan")
    baseline_tau = float(baseline["mean_flight_time_s"].iloc[0]) if len(baseline) else float("nan")

    candidates = []
    for _, row in at_cap.iterrows():
        thrash_flag = (
            row["switches_per_aircraft"] > _VECTOR8_SWITCHES_PER_AIRCRAFT * _THRASH_GUARDRAIL_FRACTION
            if pd.notna(row["switches_per_aircraft"]) else False
        )
        # delta_epsilon_vs_static/tau guardrail pair, per the plan doc: EITHER
        # regressing vs. the 240s baseline trips this. delta_epsilon_vs_static
        # is "the one actually designed to catch" guard-timing tracking
        # degradation (plan's own text); tau is a secondary corroborating
        # signal (a uniform thrash-induced slowdown could show up in tau
        # without moving delta_epsilon_vs_static, since the latter is an
        # error *magnitude* comparison, not a duration one).
        eps_regress = (
            pd.notna(row["delta_epsilon_vs_static"]) and pd.notna(baseline_delta_eps)
            and row["delta_epsilon_vs_static"] > baseline_delta_eps
        )
        tau_regress = (
            pd.notna(row["mean_flight_time_s"]) and pd.notna(baseline_tau)
            and row["mean_flight_time_s"] > baseline_tau
        )
        candidates.append({
            "hyst_s": row["hyst_s"], "gamma": row["gamma"],
            "delta_epsilon_vs_static": row["delta_epsilon_vs_static"],
            "mean_flight_time_s": row["mean_flight_time_s"],
            "switches_per_aircraft": row["switches_per_aircraft"],
            "thrash_guardrail_tripped": thrash_flag,
            "regresses_tracking_vs_240s": eps_regress or tau_regress,
            "regresses_delta_eps": eps_regress,
            "regresses_tau": tau_regress,
        })

    eligible = [c for c in candidates if not c["thrash_guardrail_tripped"] and not c["regresses_tracking_vs_240s"]]
    pool = eligible if eligible else candidates  # fall back to all, but flag it
    best = max(pool, key=lambda c: c["gamma"] if pd.notna(c["gamma"]) else -np.inf)

    return {
        "resolved_cap": resolved_cap,
        "candidates": candidates,
        "eligible_after_guardrails": [c["hyst_s"] for c in eligible],
        "fell_back_to_all_candidates": not eligible,
        "picked_hyst_s": best["hyst_s"],
        "picked_gamma": best["gamma"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=str, required=True)
    p.add_argument("--k-cps", type=int, default=3)
    p.add_argument("--mode", type=str, default="dynamic")
    p.add_argument("--status-quo-hyst", type=float, default=240.0)
    p.add_argument("--out-csv", type=str, default=None,
                   help="Defaults to <sweep-root>/analysis_summary.csv")
    args = p.parse_args()

    df = build_summary(args.sweep_root, args.k_cps, args.mode)
    out_csv = args.out_csv or os.path.join(args.sweep_root, "analysis_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote per-cell summary ({len(df)} rows) -> {out_csv}\n")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))

    hyst_values = sorted(df["hyst_s"].unique())
    n_expected_caps = df["cap"].nunique()
    print(f"\n=== Cap decision (10%-of-full-Gamma-range stopping rule, per hysteresis slice) ===")
    cap_results = {}
    for hyst in hyst_values:
        slice_df = df[df["hyst_s"] == hyst]
        if len(slice_df) < n_expected_caps:
            print(f"hyst={hyst}s: INCOMPLETE ({len(slice_df)}/{n_expected_caps} caps present) -- skipping")
            continue
        result = apply_cap_stopping_rule(slice_df)
        cap_results[hyst] = result
        print(f"\nhyst={hyst}s: full_range={result['full_range']} ac/h -> winner cap={result['winner_cap']}")
        for step in result["steps"]:
            flag = "STOP" if step["pct_of_full_range"] < 10.0 else "continue"
            print(f"    {step['from_cap']:>3} -> {step['to_cap']:>3}: "
                  f"gain={step['gamma_gain']:+.3f} ac/h ({step['pct_of_full_range']:.1f}% of range) [{flag}]")

    if cap_results:
        winners = {hyst: r["winner_cap"] for hyst, r in cap_results.items()}
        unique_winners = set(winners.values())
        print(f"\nCap winners per slice: {winners}")
        if len(unique_winners) == 1:
            resolved_cap = unique_winners.pop()
            print(f"CONSISTENT across all {len(cap_results)} complete slices -> resolved cap = {resolved_cap}")
        else:
            print(f"*** CAP DECISION SHIFTS ACROSS HYSTERESIS SLICES: {winners} -- "
                  f"per the plan's decision rule, this needs an explicit joint call, "
                  f"not silently picking one slice. STOP and report this to the user. ***")
            resolved_cap = None

        if resolved_cap is not None:
            print(f"\n=== Hysteresis decision at resolved cap={resolved_cap} "
                  f"(maximize Gamma, subject to guardrails vs. {args.status_quo_hyst}s) ===")
            hres = resolve_hysteresis(df, resolved_cap, args.status_quo_hyst)
            for c in hres["candidates"]:
                flags = []
                if c["thrash_guardrail_tripped"]:
                    flags.append("THRASH GUARDRAIL TRIPPED")
                if c["regresses_delta_eps"]:
                    flags.append("REGRESSES delta_epsilon_vs_static vs 240s")
                if c["regresses_tau"]:
                    flags.append("REGRESSES tau (mean_flight_time_s) vs 240s")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                print(f"  hyst={c['hyst_s']:>5.0f}s: gamma={c['gamma']:.3f} ac/h  "
                      f"delta_eps_vs_static={c['delta_epsilon_vs_static']}  "
                      f"tau={c['mean_flight_time_s']}  "
                      f"switches/ac={c['switches_per_aircraft']}{flag_str}")
            if hres["fell_back_to_all_candidates"]:
                print("  *** ALL candidates tripped a guardrail -- fell back to raw max-Gamma "
                      "pick across all candidates. Report this explicitly, do not present it "
                      "as a clean guardrail-respecting pick. ***")
            print(f"  -> picked hysteresis = {hres['picked_hyst_s']}s "
                  f"(gamma={hres['picked_gamma']})")

            if resolved_cap != 35 or hres["picked_hyst_s"] != args.status_quo_hyst:
                print(f"\n*** RESOLVED VALUES DIFFER FROM CURRENT PRODUCTION "
                      f"(cap=35, hysteresis={args.status_quo_hyst}s): "
                      f"resolved cap={resolved_cap}, hysteresis={hres['picked_hyst_s']}s. "
                      f"Per the plan's critical trap, STOP and tell the user -- do not silently "
                      f"propagate or assume an M=2,000 rerun is warranted. ***")


if __name__ == "__main__":
    main()
