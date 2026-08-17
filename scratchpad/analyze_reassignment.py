import pandas as pd
import numpy as np

from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME


def _parse_per_runway(series: pd.Series, runway: str) -> pd.Series:
    """Parse one runway's value out of a '18R:val,27:val'-joined telemetry
    column (eta_per_runway/sigma_per_runway/queue_delay_per_runway all share
    this convention -- see telemetry.py)."""
    pattern = rf"{runway}:([-\d.]+)"
    return series.str.extract(pattern, expand=False).astype(float)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion -- used to judge
    whether an observed 18R-split movement at diagnostic scale (M=50, a few
    hundred both-eligible rows/combo) is distinguishable from noise, not just
    a directional nudge. Production scale (Vector 9) had p<1e-5 on ~2,000
    episodes; diagnostic scale won't reach that power, so an honest interval
    matters more than a point estimate here."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (center - half, center + half)


def analyze(path: str, label: str) -> dict:
    """Print + return the standard reassignment-telemetry diagnostics for one
    (combo) parquet file. Returns a small summary dict so callers can build a
    cross-combo table (see the load_balance_weight_s sweep section below)."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label}: not ready yet ({path}) ===\n")
        return {}

    print(f"=== {label} (n_rows={len(df)}) ===")
    switch_rate = df["switched"].mean()
    print(f"  overall switch rate: {switch_rate:.4f}")

    both_eligible = df[df["eligible_runways"] == "18R,27"]
    both_eligible_frac = len(both_eligible) / len(df) if len(df) else float("nan")
    print(f"  rows with BOTH runways eligible: {len(both_eligible)} ({both_eligible_frac:.2%})")
    split_18r = float("nan")
    split_18r_lo, split_18r_hi = float("nan"), float("nan")
    if len(both_eligible):
        split = both_eligible["chosen_runway"].value_counts(normalize=True).round(3).to_dict()
        split_18r = split.get("18R", 0.0)
        n_18r = int((both_eligible["chosen_runway"] == "18R").sum())
        split_18r_lo, split_18r_hi = wilson_ci(n_18r, len(both_eligible))
        print(f"  chosen_runway split when both eligible: {split} "
              f"(18R 95% CI: [{split_18r_lo:.3f}, {split_18r_hi:.3f}], n={len(both_eligible)})")

    switches = df[df["switched"]]
    print(f"  n_switches={len(switches)}")
    if len(switches):
        print(f"  switch direction (chosen_runway among switches): {switches['chosen_runway'].value_counts(normalize=True).round(3).to_dict()}")
        print(f"  eta_gap_s (raw-physical) among switches: mean={switches['eta_gap_s'].mean():.1f}s, median={switches['eta_gap_s'].median():.1f}s, min={switches['eta_gap_s'].min():.1f}s, max={switches['eta_gap_s'].max():.1f}s")
        if "cost_gap_s" in switches.columns:
            # cost_gap_s is the load-adjusted gap that actually drove the
            # decision (adjusted_eta[current] - masked_eta[chosen]); eta_gap_s
            # is the same comparison on raw eta_matrix. Their difference is
            # exactly the load-balancing term's own contribution to *this*
            # switch -- how much of the decision the new term, not raw ETA,
            # is responsible for.
            load_contribution = switches["cost_gap_s"] - switches["eta_gap_s"]
            print(f"  cost_gap_s (load-adjusted decision basis) among switches: mean={switches['cost_gap_s'].mean():.1f}s, median={switches['cost_gap_s'].median():.1f}s")
            print(f"  load-term's own contribution to the switch decision (cost_gap_s - eta_gap_s): "
                  f"mean={load_contribution.mean():.1f}s, median={load_contribution.median():.1f}s")
            # Switches raw ETA alone would NOT have justified (eta_gap_s <= 0,
            # i.e. the load term is doing more than tie-breaking -- it's
            # overriding a raw-ETA preference for staying/the other runway).
            overridden = switches[switches["eta_gap_s"] <= 0]
            print(f"  switches where raw ETA alone did not favor the chosen runway (eta_gap_s<=0): "
                  f"{len(overridden)}/{len(switches)} ({len(overridden)/len(switches):.2%})")

    # Per-(episode,acid) oscillation check: how many distinct runways does each acid visit across the episode?
    per_ac = df.groupby(["episode_id", "acid"])["current_runway"].nunique()
    multi_frac = (per_ac > 1).mean() if len(per_ac) else float("nan")
    print(f"  aircraft visiting >1 distinct runway across their episode: {(per_ac > 1).sum()}/{len(per_ac)} ({multi_frac:.2%})")
    multi = per_ac[per_ac > 1]
    n_true_osc = 0
    if len(multi):
        # for these, count actual switch events (not just distinct runways -- could bounce back)
        switch_counts = df[df["switched"]].groupby(["episode_id", "acid"]).size()
        osc = switch_counts[switch_counts.index.isin(multi.index)]
        print(f"  switch-event count distribution for multi-runway acids: {osc.value_counts().sort_index().to_dict()}")
        # true oscillation = visits >1 runway but switch count > distinct_runways-1 (implies a back-and-forth)
        n_true_osc = int((osc > (multi.reindex(osc.index) - 1)).sum())
        print(f"  of those, true back-and-forth oscillators (switch_count > n_distinct-1): {n_true_osc}")

    # Forced-off-eligibility check: current runway not in the eligible set this cycle
    # -> if this also triggers a switch, the aircraft is being pushed off a runway
    # it'd otherwise stay on, purely because its FCFS-relative rank drifted, not
    # because the destination is genuinely better. (Vector 9's separate, still-
    # out-of-scope finding -- tracked here only as a side observation.)
    df["current_eligible"] = df.apply(lambda r: r["current_runway"] in r["eligible_runways"].split(","), axis=1)
    forced = df[~df["current_eligible"]]
    print(f"  rows where current runway is INELIGIBLE this cycle (forced off): {len(forced)} ({len(forced)/len(df):.2%})")
    if len(forced):
        forced_switched = forced[forced["switched"]]
        print(f"    of those, actually switched: {len(forced_switched)}/{len(forced)} ({len(forced_switched)/len(forced):.2%})")
        print(f"    forced-switch destination: {forced_switched['chosen_runway'].value_counts(normalize=True).round(3).to_dict()}")
        print(f"    forced-switch eta_gap_s: mean={forced_switched['eta_gap_s'].mean():.1f}s, negative-gap fraction: {(forced_switched['eta_gap_s']<0).mean():.2%}")

    print()
    return {
        "n_rows": len(df),
        "switch_rate": switch_rate,
        "both_eligible_frac": both_eligible_frac,
        "split_18r_when_both_eligible": split_18r,
        "split_18r_ci_lo": split_18r_lo,
        "split_18r_ci_hi": split_18r_hi,
        "true_oscillator_count": n_true_osc,
    }


def analyze_stall_by_runway(path: str, label: str) -> dict:
    """The reassignment stream (``analyze`` above) measures the *mechanism*
    (which runway a decision cycle picks). This measures the actual
    *outcome* Vector 9's production-scale finding was about: whether
    stall_detected concentrates on one runway disproportionately to its
    share of landed traffic, from cps_eval_aircraft.parquet (populated for
    every combo regardless of --log-reassignment-events). Closes the loop
    from the intermediate choice-split metric back to the real thing that
    matters."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (aircraft): not ready yet ({path}) ===\n")
        return {}

    df = df[df["runway_assignment_mode"] == "dynamic"]
    if df.empty:
        return {}

    traffic_share = df["runway_id"].value_counts(normalize=True).round(3).to_dict()
    stall_by_rwy = df.groupby("runway_id")["stall_detected"].mean().round(4).to_dict()
    stall_share = (
        df[df["stall_detected"]]["runway_id"].value_counts(normalize=True).round(3).to_dict()
        if df["stall_detected"].any() else {}
    )
    print(f"=== {label} (aircraft outcomes, n={len(df)}) ===")
    print(f"  landed-traffic share by runway: {traffic_share}")
    print(f"  stall_detected rate by runway: {stall_by_rwy}")
    print(f"  share of ALL stalls by runway: {stall_share}")
    print()
    return {
        "traffic_share_18r": traffic_share.get("18R", float("nan")),
        "stall_rate_18r": stall_by_rwy.get("18R", float("nan")),
        "stall_rate_27": stall_by_rwy.get("27", float("nan")),
        "stall_share_18r": stall_share.get("18R", float("nan")),
        "overall_stall_rate": float(df["stall_detected"].mean()),
    }


def analyze_occupancy_balance(path: str, label: str) -> dict:
    """Direct mechanism check, independent of the choice-split proxy: at
    every actual decision cycle (episode_id, current_time), how many
    aircraft are *live-assigned* to each runway right now (``current_runway``,
    i.e. the state the load term itself reads via ``member``)? If the fix is
    correcting the real mechanism -- not just nudging an aggregate stat --
    this per-cycle imbalance should shrink as weight increases, not just the
    both-eligible choice split."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (occupancy balance): not ready yet ({path}) ===\n")
        return {}

    counts = df.groupby(["episode_id", "current_time", "current_runway"]).size().unstack(fill_value=0)
    for rwy in ("18R", "27"):
        if rwy not in counts.columns:
            counts[rwy] = 0
    imbalance = (counts["18R"] - counts["27"]).abs()
    total = counts["18R"] + counts["27"]
    share_18r = (counts["18R"] / total.replace(0, np.nan)).dropna()
    print(f"=== {label} (live occupancy balance, n_cycles={len(counts)}) ===")
    print(f"  mean |n_18R - n_27| per decision cycle: {imbalance.mean():.3f}")
    print(f"  mean live 18R occupancy share per decision cycle: {share_18r.mean():.3f}")
    print()
    return {
        "mean_occupancy_imbalance": float(imbalance.mean()),
        "mean_live_18r_share": float(share_18r.mean()),
    }


def analyze_outcomes(path: str, label: str) -> dict:
    """Side-effect guard: does load-balancing move the actual thing the RL
    worker/CPS pipeline is optimized for (success rate, RTA tracking error),
    not just runway choice? A cost term that improves balance while quietly
    degrading tracking accuracy or landing success would be a bad trade even
    if the 18R split looks better."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (outcomes): not ready yet ({path}) ===\n")
        return {}

    success_rate = float(df["success"].mean())
    abs_rta_err = df["rta_error_cps"].abs()
    print(f"=== {label} (success/RTA outcomes, n={len(df)}) ===")
    print(f"  success rate: {success_rate:.4f}")
    print(f"  |rta_error_cps|: mean={abs_rta_err.mean():.1f}s, median={abs_rta_err.median():.1f}s, p90={abs_rta_err.quantile(0.9):.1f}s")
    print()
    return {
        "success_rate": success_rate,
        "abs_rta_error_mean_s": float(abs_rta_err.mean()),
        "abs_rta_error_p90_s": float(abs_rta_err.quantile(0.9)),
    }


def analyze_separation(path: str, label: str) -> dict:
    """Safety guard: does the extra churn a nonzero weight can introduce
    (more switches -> different greedy-scheduling order) increase RECAT-EU
    separation violations? Tuning runway *choice* shouldn't be allowed to
    quietly erode the thing the greedy scheduler exists to guarantee."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (separation): not ready yet ({path}) ===\n")
        return {}
    if df.empty:
        return {}

    violation = df["gap_actual_s"] < df["required_sep_s"]
    violation_rate = float(violation.mean())
    print(f"=== {label} (separation compliance, n_pairs={len(df)}) ===")
    print(f"  separation violation rate: {violation_rate:.4%} ({int(violation.sum())}/{len(df)})")
    if violation.any():
        deficits = (df.loc[violation, "required_sep_s"] - df.loc[violation, "gap_actual_s"])
        print(f"  violation deficit: mean={deficits.mean():.1f}s, max={deficits.max():.1f}s")
    print()
    return {"separation_violation_rate": violation_rate, "n_pairs": len(df)}


def analyze_flight_duration_by_runway(path: str, label: str) -> dict:
    """Direct test of the differential-dwell-time hypothesis raised by
    Regime A's wrong-direction result: if 18R's flight duration is
    systematically shorter than 27's, it clears aircraft faster and so
    carries lower *standing* occupancy even while receiving the majority of
    new assignments -- exactly the mismatch that would make an
    instantaneous-occupancy-count penalty self-reinforcing (rewarding the
    runway that's already winning) rather than corrective. ``len(traj_x)``
    is a per-decision-step count specific to each aircraft's own flight, so
    (unlike ``actual_landing_time``) it's a valid duration proxy in both the
    fixed-pool (Regime A) and staggered-spawn (Regime B) harnesses."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (flight duration): not ready yet ({path}) ===\n")
        return {}

    df = df[df["runway_assignment_mode"] == "dynamic"].copy()
    if df.empty:
        return {}
    df["duration_s"] = df["traj_x"].apply(len) * ACTION_TIME

    by_rwy = df.groupby("runway_id")["duration_s"].agg(["mean", "median", "count"])
    print(f"=== {label} (flight duration by runway, n={len(df)}) ===")
    print(f"  {by_rwy.to_dict('index')}")
    dur_18r = by_rwy["mean"].get("18R", float("nan"))
    dur_27 = by_rwy["mean"].get("27", float("nan"))
    if not (np.isnan(dur_18r) or np.isnan(dur_27)):
        print(f"  18R mean duration is {dur_18r - dur_27:+.1f}s vs 27 "
              f"({'shorter/faster turnover' if dur_18r < dur_27 else 'longer/slower turnover'})")
    print()
    return {
        "mean_duration_18r_s": float(dur_18r) if not np.isnan(dur_18r) else float("nan"),
        "mean_duration_27_s": float(dur_27) if not np.isnan(dur_27) else float("nan"),
        "n_landed_18r": int(by_rwy["count"].get("18R", 0)),
        "n_landed_27": int(by_rwy["count"].get("27", 0)),
    }


def analyze_raw_eta_advantage(path: str, label: str) -> dict:
    """Uses the eta_per_runway telemetry column (raw eta_matrix, computed
    before the load term and therefore weight-independent by construction)
    to directly measure whether 18R carries a persistent raw-ETA advantage
    over 27 among both-eligible decisions -- i.e. how much of the *original*
    bias (before any load-balancing) is explained by raw predicted ETA alone,
    independent of queue state. Quantifies the root-cause split Vector 9
    left as an open causal question, rather than just re-confirming the
    aggregate choice-split is biased."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (raw ETA advantage): not ready yet ({path}) ===\n")
        return {}

    both_eligible = df[df["eligible_runways"] == "18R,27"]
    if both_eligible.empty or "eta_per_runway" not in both_eligible.columns:
        return {}

    eta_18r = _parse_per_runway(both_eligible["eta_per_runway"], "18R")
    eta_27 = _parse_per_runway(both_eligible["eta_per_runway"], "27")
    advantage = eta_27 - eta_18r  # positive => 18R has the lower (better) raw ETA
    print(f"=== {label} (raw ETA advantage of 18R over 27, both-eligible rows, n={len(both_eligible)}) ===")
    print(f"  mean={advantage.mean():.1f}s, median={advantage.median():.1f}s, "
          f"frac favoring 18R (advantage>0)={float((advantage > 0).mean()):.3f}")
    print()
    return {
        "raw_eta_advantage_18r_mean_s": float(advantage.mean()),
        "raw_eta_advantage_18r_frac_positive": float((advantage > 0).mean()),
    }


def analyze_queue_delay_per_runway(path: str, label: str) -> dict:
    """queue_delay_weight_s sweep (cps_runway_queue_delay_fix.md): uses the
    queue_delay_per_runway telemetry column (raw seconds estimate, logged
    before the weight multiply -- see telemetry.py) to measure the new
    term's own magnitude and direction among both-eligible decisions,
    independent of which weight was swept. If the design is doing what it's
    meant to (penalizing 18R more than 27, consistent with 18R's raw-ETA
    advantage and higher observed traffic share), the 18R-minus-27 delay gap
    should be positive and the fraction of rows where 18R carries the larger
    penalty should be well above 0.5."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (queue-delay penalty): not ready yet ({path}) ===\n")
        return {}

    both_eligible = df[df["eligible_runways"] == "18R,27"]
    if both_eligible.empty or "queue_delay_per_runway" not in both_eligible.columns:
        return {}

    delay_18r = _parse_per_runway(both_eligible["queue_delay_per_runway"], "18R")
    delay_27 = _parse_per_runway(both_eligible["queue_delay_per_runway"], "27")
    gap = delay_18r - delay_27  # positive => 18R penalized more than 27
    print(f"=== {label} (queue_delay_penalty, 18R vs 27, both-eligible rows, n={len(both_eligible)}) ===")
    print(f"  18R mean penalty={delay_18r.mean():.1f}s, 27 mean penalty={delay_27.mean():.1f}s")
    print(f"  18R-minus-27 gap: mean={gap.mean():.1f}s, median={gap.median():.1f}s, "
          f"frac rows where 18R penalized more (gap>0)={float((gap > 0).mean()):.3f}")
    print()
    return {
        "queue_delay_18r_mean_s": float(delay_18r.mean()),
        "queue_delay_27_mean_s": float(delay_27.mean()),
        "queue_delay_18r_minus_27_mean_s": float(gap.mean()),
        "queue_delay_frac_18r_penalized_more": float((gap > 0).mean()),
    }


def analyze_split_trend_over_episode(path: str, label: str, n_bins: int = 4) -> dict:
    """Bins both-eligible decisions by within-episode elapsed time
    (current_time) into quartiles and reports the 18R-choice-split per bin --
    tests whether the bias *compounds* over the course of an episode
    (consistent with a self-reinforcing/runaway dynamic) vs. staying flat
    (consistent with a static, occupancy-independent raw-ETA bias)."""
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"=== {label} (split trend): not ready yet ({path}) ===\n")
        return {}

    both_eligible = df[df["eligible_runways"] == "18R,27"].copy()
    if len(both_eligible) < n_bins * 5:
        return {}

    both_eligible["time_bin"] = pd.qcut(both_eligible["current_time"], n_bins, duplicates="drop")
    per_bin = both_eligible.groupby("time_bin", observed=True)["chosen_runway"].apply(
        lambda s: float((s == "18R").mean())
    )
    print(f"=== {label} (18R-split by within-episode time bin) ===")
    for time_bin, split in per_bin.items():
        print(f"  {time_bin}: 18R-split={split:.3f}")
    print()
    bins = list(per_bin.values)
    return {
        "split_trend_first_bin": bins[0] if bins else float("nan"),
        "split_trend_last_bin": bins[-1] if bins else float("nan"),
        "split_trend_delta": (bins[-1] - bins[0]) if len(bins) >= 2 else float("nan"),
    }


if __name__ == "__main__":
    print("############ Pre-fix baseline (M=30, scratchpad/reassignment_diagnostic/) ############\n")
    print("NOTE: this baseline was collected via run_cps_eval.py (fixed n_aircraft=35,")
    print("Regime A-shaped), NOT run_batch_eval.py's rolling-arrival-stream Regime B used")
    print("below -- printed for context/reference only, not used in a numeric cross-check")
    print("against the Regime B sweep (the two harnesses aren't directly comparable).\n")
    baseline_rows = []
    for k, combo in ((0, "k0_dynamic"), (1, "k1_dynamic"), (3, "k3_dynamic")):
        summary = analyze(f"scratchpad/reassignment_diagnostic/{combo}/cps_eval_reassignment.parquet", combo)
        if summary:
            baseline_rows.append({"k_cps": k, "weight": "baseline_M30", **summary})

    print("\n############ queue_delay_weight_s sweep (M=50, scratchpad/queue_delay_sweep/) ############\n")
    print("Regime B only (run_batch_eval.py, rolling-arrival stream) -- Regime A dropped for")
    print("this sweep as not comparable to the M=2,000 production config (see")
    print(".claude/plans/cps_runway_queue_delay_fix.md).\n")
    weights = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    k_values = [0, 1, 3]

    def _run_all_checks(base: str, label: str) -> dict:
        row = {}
        row.update(analyze(f"{base}/cps_eval_reassignment.parquet", label))
        row.update(analyze_occupancy_balance(f"{base}/cps_eval_reassignment.parquet", label))
        row.update(analyze_raw_eta_advantage(f"{base}/cps_eval_reassignment.parquet", label))
        row.update(analyze_queue_delay_per_runway(f"{base}/cps_eval_reassignment.parquet", label))
        row.update(analyze_split_trend_over_episode(f"{base}/cps_eval_reassignment.parquet", label))
        row.update(analyze_stall_by_runway(f"{base}/cps_eval_aircraft.parquet", label))
        row.update(analyze_flight_duration_by_runway(f"{base}/cps_eval_aircraft.parquet", label))
        row.update(analyze_outcomes(f"{base}/cps_eval_aircraft.parquet", label))
        row.update(analyze_separation(f"{base}/cps_eval_separation.parquet", label))
        return row

    print("--- Regime B (run_batch_eval.py, rolling-arrival stream) ---\n")
    regime_b_rows = []
    for w in weights:
        for k in k_values:
            base = f"scratchpad/queue_delay_sweep/w{w}/k{k}_dynamic"
            row = _run_all_checks(base, f"regimeB k={k} weight={w}")
            if row:
                regime_b_rows.append({"k_cps": k, "weight": w, **row})

    DOSE_RESPONSE_COLS = [
        "k_cps", "weight", "split_18r_when_both_eligible", "split_18r_ci_lo", "split_18r_ci_hi",
        "split_trend_first_bin", "split_trend_last_bin", "split_trend_delta",
        "mean_live_18r_share", "raw_eta_advantage_18r_mean_s", "raw_eta_advantage_18r_frac_positive",
        "queue_delay_18r_minus_27_mean_s", "queue_delay_frac_18r_penalized_more",
        "mean_duration_18r_s", "mean_duration_27_s",
        "switch_rate", "true_oscillator_count",
        "stall_rate_18r", "stall_rate_27", "success_rate", "abs_rta_error_mean_s",
        "separation_violation_rate",
    ]
    for label, rows in (("Regime B", regime_b_rows),):
        if not rows:
            continue
        print(f"\n=== {label} dose-response table (18R-split target: closer to 0.5, away from 0.65-0.73) ===")
        df = pd.DataFrame(rows).sort_values(["k_cps", "weight"])
        cols = [c for c in DOSE_RESPONSE_COLS if c in df.columns]
        with pd.option_context("display.width", 220, "display.max_columns", None):
            print(df[cols].to_string(index=False))
