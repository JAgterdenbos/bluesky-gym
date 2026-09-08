"""
cps_coordination/scripts/phase3_significance_tests.py
--------------------------------------------------------
Significance testing for the Phase III k_cps x runway_assignment_mode grid
(tab:throughput_results, tab:delay_ripple). Those tables report mean +/- std
(episode-to-episode variance) only -- this script adds paired/independent
hypothesis tests, Holm-Bonferroni-corrected p-values, and effect sizes for
Gamma, C_sep, and Delta_epsilon_static, plus a one-sample test of rho_ripple
against zero, so the paper can cite significance rather than just descriptive
spread.

Derives its own per-episode series from the raw Parquet telemetry (no
per-episode metrics file exists on disk -- cps_metrics_offline.py computes
per-episode values internally for its `_std` companion figures but never
persists them). The per-episode formulas here mirror
`cps_metrics_offline.recompute_metrics`'s internal per-episode grouping
exactly (same success filter, same episode-span/ratio definitions), and
`recompute_metrics` itself is imported and run to provide the published
aggregate each derived series is checked against -- see
`self_review_gate`.

Usage
-----
  python cps_coordination/scripts/phase3_significance_tests.py \\
      --sweep-root experiments/cps_eval/scale_10k_20260820_153701 \\
      --out-dir cps_coordination/figures/paper_report_20260821_cap50
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cps_coordination.scripts.cps_metrics_offline import (
    load_recat_matrix,
    load_telemetry,
    recompute_metrics,
)
from cps_coordination.scripts.generate_paper_report import _df_to_latex
from cps_coordination.scripts.summarize_batch_sweep import _COMBO_RE, discover_combos

SEP_TOLERANCE_S = 5.0
ALPHA = 0.05
# |skewness| / |excess kurtosis| above these, or a bounded [0,1]/[-1,1]
# metric with mass piled near a boundary, is treated as "not normal enough"
# for a t-test regardless of Shapiro's p-value -- Shapiro's power at n~2000
# rejects near-trivial deviations, so shape is the primary signal and
# Shapiro is corroborating, not decisive (see module docstring / plan).
SKEW_THRESHOLD = 0.5
KURTOSIS_THRESHOLD = 1.0


# ──────────────────────────────────────────────────────────────────────────
# Per-episode series construction (mirrors cps_metrics_offline.recompute_metrics
# exactly, but returns the per-episode arrays instead of collapsing to mean/std)
# ──────────────────────────────────────────────────────────────────────────


def per_episode_gamma(aircraft_df: pd.DataFrame) -> pd.Series:
    successful = aircraft_df[aircraft_df["success"]]
    if successful.empty:
        return pd.Series(dtype=float)
    episode_counts = successful.groupby("episode_id").size()
    landing_g = successful.groupby("episode_id")["actual_landing_time"]
    episode_span_h = ((landing_g.max() - landing_g.min()) / 3600.0).clip(lower=1e-6)
    valid_episodes = episode_counts[episode_counts >= 2].index
    return (episode_counts[valid_episodes] / episode_span_h[valid_episodes]).astype(float)


def per_episode_c_sep(separation_df: pd.DataFrame, tolerance_s: float = SEP_TOLERANCE_S) -> pd.Series:
    if separation_df.empty:
        return pd.Series(dtype=float)
    compliant = (separation_df["gap_actual_s"] >= (separation_df["required_sep_s"] - tolerance_s)).astype(float)
    return compliant.groupby(separation_df["episode_id"]).mean()


def per_episode_delta_epsilon(aircraft_df: pd.DataFrame, reference_col: str) -> pd.Series:
    cps_valid = ~aircraft_df["rta_error_cps"].isna()
    ref_valid = ~aircraft_df[reference_col].isna()
    mask = (cps_valid & ref_valid).to_numpy()
    values = (aircraft_df["rta_error_cps"].abs() - aircraft_df[reference_col].abs()).to_numpy()
    episode_ids = aircraft_df["episode_id"].to_numpy()[mask]
    values = values[mask]
    return pd.Series(values, index=episode_ids).groupby(level=0).mean()


def per_episode_rho_ripple(aircraft_df: pd.DataFrame) -> pd.Series:
    successful = aircraft_df[aircraft_df["success"]]
    if successful.empty:
        return pd.Series(dtype=float)
    group_ids = successful["episode_id"].to_numpy()
    order_keys = successful["actual_landing_time"].to_numpy()
    values = successful["rta_error_cps"].to_numpy(dtype=float)

    order = np.lexsort((order_keys, group_ids))
    g = group_ids[order]
    v = values[order]
    same_group = g[:-1] == g[1:]
    if not np.any(same_group):
        return pd.Series(dtype=float)
    x = v[:-1][same_group]
    y = v[1:][same_group]
    g_pairs = g[:-1][same_group]

    result: Dict[int, float] = {}
    for eid in np.unique(g_pairs):
        m = g_pairs == eid
        xi, yi = x[m], y[m]
        if len(xi) < 1 or np.all(xi == xi[0]) or np.all(yi == yi[0]):
            continue
        rho = np.corrcoef(xi, yi)[0, 1]
        if not np.isnan(rho):
            result[int(eid)] = float(rho)
    return pd.Series(result)


@dataclass
class ConditionData:
    k_cps: int
    mode: str
    combo: str
    aircraft_df: pd.DataFrame
    separation_df: pd.DataFrame
    gamma: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    c_sep: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    delta_eps_static: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rho_ripple: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    reported: Dict[str, Any] = field(default_factory=dict)


def load_conditions(sweep_root: str) -> Dict[Tuple[int, str], ConditionData]:
    recat_matrix = load_recat_matrix()
    conditions: Dict[Tuple[int, str], ConditionData] = {}
    for combo_dir in discover_combos(sweep_root):
        m = _COMBO_RE.match(os.path.basename(combo_dir))
        assert m is not None
        k_cps, mode = int(m.group("k_cps")), m.group("mode")
        aircraft_df, separation_df = load_telemetry(combo_dir)
        cd = ConditionData(
            k_cps=k_cps, mode=mode, combo=os.path.basename(combo_dir),
            aircraft_df=aircraft_df, separation_df=separation_df,
        )
        cd.gamma = per_episode_gamma(aircraft_df)
        cd.c_sep = per_episode_c_sep(separation_df)
        cd.delta_eps_static = per_episode_delta_epsilon(aircraft_df, "rta_error_static")
        cd.rho_ripple = per_episode_rho_ripple(aircraft_df)
        cd.reported = recompute_metrics(aircraft_df, separation_df, recat_matrix)
        conditions[(k_cps, mode)] = cd
    return conditions


def self_review_gate(conditions: Dict[Tuple[int, str], ConditionData]) -> List[str]:
    """Compare our derived per-episode series' mean against the paper's own
    published aggregate for the same combo. Must match (within float
    tolerance) or every downstream test is untrustworthy."""
    problems = []
    checks = [
        ("gamma", "gamma"),
        ("c_sep", "c_sep"),
        ("delta_eps_static", "delta_epsilon_vs_static"),
    ]
    for key, cd in conditions.items():
        for series_attr, reported_key in checks:
            series = getattr(cd, series_attr)
            reported = cd.reported.get(reported_key)
            if reported in (None, "nan") or series.empty:
                continue
            derived_mean = float(series.mean())
            if abs(derived_mean - float(reported)) > 1e-3:
                problems.append(
                    f"{cd.combo}: derived {series_attr} mean={derived_mean:.6f} "
                    f"!= reported {reported_key}={reported}"
                )
        reported_rho = cd.reported.get("rho_ripple")
        if reported_rho not in (None, "nan") and not cd.rho_ripple.empty:
            derived_mean = float(cd.rho_ripple.mean())
            if abs(derived_mean - float(reported_rho)) > 1e-3:
                problems.append(
                    f"{cd.combo}: derived rho_ripple mean={derived_mean:.6f} "
                    f"!= reported rho_ripple={reported_rho}"
                )
    return problems


# ──────────────────────────────────────────────────────────────────────────
# Cross-condition matching verification (paired vs. independent)
# ──────────────────────────────────────────────────────────────────────────


def verify_matching(conditions: Dict[Tuple[int, str], ConditionData]) -> Tuple[bool, str]:
    """The `solo` pass never sees k_cps or runway_assignment_mode, so if the
    same seed_base + episode_index scheme underlies every combo (as
    run_batch_eval.py's code sets it up), per-(episode_id, acid) rta_error_solo
    values should be bit-identical (up to float round-trip through Parquet)
    across all 6 conditions. If confirmed, episode_id is a valid pairing key
    for every comparison in this script, not just within one (k, mode)."""
    combos = list(conditions.values())
    reference = combos[0]
    ref_solo = reference.aircraft_df.set_index(["episode_id", "acid"])["rta_error_solo"]
    ref_solo = ref_solo[~ref_solo.index.duplicated()]

    mismatches = []
    for cd in combos[1:]:
        other_solo = cd.aircraft_df.set_index(["episode_id", "acid"])["rta_error_solo"]
        other_solo = other_solo[~other_solo.index.duplicated()]
        common = ref_solo.index.intersection(other_solo.index)
        if len(common) == 0:
            mismatches.append(f"{cd.combo}: no overlapping (episode_id, acid) keys with {reference.combo}")
            continue
        a = ref_solo.loc[common].to_numpy(dtype=float)
        b = other_solo.loc[common].to_numpy(dtype=float)
        both_valid = ~(np.isnan(a) | np.isnan(b))
        if both_valid.sum() == 0:
            continue
        if not np.allclose(a[both_valid], b[both_valid], atol=1e-6, rtol=1e-6):
            n_mismatch = int((~np.isclose(a[both_valid], b[both_valid], atol=1e-6, rtol=1e-6)).sum())
            mismatches.append(
                f"{cd.combo} vs {reference.combo}: {n_mismatch}/{both_valid.sum()} "
                f"solo-pass rta_error_solo values differ"
            )

    if mismatches:
        return False, "Solo-pass values differ across conditions:\n  " + "\n  ".join(mismatches)
    return True, (
        f"Solo-pass rta_error_solo matches exactly (atol=1e-6) across all "
        f"{len(combos)} conditions for every shared (episode_id, acid) key "
        f"-> episode_id is a valid pairing key across k and mode."
    )


# ──────────────────────────────────────────────────────────────────────────
# Distribution shape + test selection
# ──────────────────────────────────────────────────────────────────────────


def _shape_diagnostics(x: np.ndarray) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if len(x) < 8:
        return {"skew": float("nan"), "kurtosis": float("nan"), "shapiro_p": float("nan")}
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x))  # excess kurtosis (Fisher, normal=0)
    # Shapiro's exact test caps out around n=5000; subsample for speed/validity
    # at n=2000 this isn't hit, kept as a guard for future larger M.
    sample = x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
    shapiro_p = float(stats.shapiro(sample).pvalue)
    return {"skew": skew, "kurtosis": kurt, "shapiro_p": shapiro_p}


def _looks_normal(diag: Dict[str, float]) -> bool:
    if np.isnan(diag["skew"]) or np.isnan(diag["kurtosis"]):
        return False
    return abs(diag["skew"]) < SKEW_THRESHOLD and abs(diag["kurtosis"]) < KURTOSIS_THRESHOLD


# ──────────────────────────────────────────────────────────────────────────
# Effect sizes
# ──────────────────────────────────────────────────────────────────────────


def cohens_d_paired(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else float("nan")


def cohens_d_independent(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else float("nan")


def cohens_d_one_sample(x: np.ndarray) -> float:
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd) if sd > 0 else float("nan")


def rank_biserial_wilcoxon(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation: (n+ - n-) sums of signed
    ranks, normalized by total rank sum."""
    diff = diff[diff != 0]
    if len(diff) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else float("nan")


def rank_biserial_mannwhitney(a: np.ndarray, b: np.ndarray, u_stat: float) -> float:
    na, nb = len(a), len(b)
    return float(1 - (2 * u_stat) / (na * nb))


# ──────────────────────────────────────────────────────────────────────────
# Hypothesis tests
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TestResult:
    metric: str
    comparison: str
    test: str
    n: int
    statistic: float
    p_raw: float
    p_holm: Optional[float] = None
    effect_size: float = float("nan")
    effect_size_name: str = ""
    significant: Optional[bool] = None
    note: str = ""


def paired_test(metric: str, comparison: str, a: pd.Series, b: pd.Series) -> TestResult:
    common = a.index.intersection(b.index)
    x, y = a.loc[common].to_numpy(dtype=float), b.loc[common].to_numpy(dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    diff = x - y
    diag = _shape_diagnostics(diff)
    if _looks_normal(diag):
        stat, p = stats.ttest_rel(x, y)
        return TestResult(
            metric=metric, comparison=comparison, test="paired t-test",
            n=len(diff), statistic=float(stat), p_raw=float(p),
            effect_size=cohens_d_paired(diff), effect_size_name="Cohen's d (paired)",
            note=f"skew={diag['skew']:.2f}, kurt={diag['kurtosis']:.2f}, shapiro_p={diag['shapiro_p']:.3g}",
        )
    stat, p = stats.wilcoxon(x, y)
    return TestResult(
        metric=metric, comparison=comparison, test="Wilcoxon signed-rank",
        n=len(diff), statistic=float(stat), p_raw=float(p),
        effect_size=rank_biserial_wilcoxon(diff), effect_size_name="matched-pairs rank-biserial r",
        note=f"skew={diag['skew']:.2f}, kurt={diag['kurtosis']:.2f}, shapiro_p={diag['shapiro_p']:.3g}",
    )


def independent_test(metric: str, comparison: str, a: pd.Series, b: pd.Series) -> TestResult:
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    diag_a, diag_b = _shape_diagnostics(x), _shape_diagnostics(y)
    if _looks_normal(diag_a) and _looks_normal(diag_b):
        stat, p = stats.ttest_ind(x, y, equal_var=False)
        return TestResult(
            metric=metric, comparison=comparison, test="Welch's t-test",
            n=len(x) + len(y), statistic=float(stat), p_raw=float(p),
            effect_size=cohens_d_independent(x, y), effect_size_name="Cohen's d (independent)",
            note=f"skew_a={diag_a['skew']:.2f}, skew_b={diag_b['skew']:.2f}",
        )
    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return TestResult(
        metric=metric, comparison=comparison, test="Mann-Whitney U",
        n=len(x) + len(y), statistic=float(stat), p_raw=float(p),
        effect_size=rank_biserial_mannwhitney(x, y, float(stat)), effect_size_name="rank-biserial r",
        note=f"skew_a={diag_a['skew']:.2f}, skew_b={diag_b['skew']:.2f}",
    )


def one_sample_test(metric: str, comparison: str, x: pd.Series) -> TestResult:
    values = x.to_numpy(dtype=float)
    values = values[~np.isnan(values)]
    diag = _shape_diagnostics(values)
    if _looks_normal(diag):
        stat, p = stats.ttest_1samp(values, 0.0)
        return TestResult(
            metric=metric, comparison=comparison, test="one-sample t-test (vs 0)",
            n=len(values), statistic=float(stat), p_raw=float(p),
            effect_size=cohens_d_one_sample(values), effect_size_name="Cohen's d (one-sample)",
            note=f"skew={diag['skew']:.2f}, kurt={diag['kurtosis']:.2f}, shapiro_p={diag['shapiro_p']:.3g}",
        )
    stat, p = stats.wilcoxon(values)
    return TestResult(
        metric=metric, comparison=comparison, test="Wilcoxon signed-rank (vs 0)",
        n=len(values), statistic=float(stat), p_raw=float(p),
        effect_size=rank_biserial_wilcoxon(values), effect_size_name="matched-pairs rank-biserial r",
        note=f"skew={diag['skew']:.2f}, kurt={diag['kurtosis']:.2f}, shapiro_p={diag['shapiro_p']:.3g}",
    )


def holm_bonferroni(results: List[TestResult]) -> None:
    """In-place Holm-Bonferroni step-down correction across `results`
    (one family -- caller groups by metric before calling)."""
    order = sorted(range(len(results)), key=lambda i: results[i].p_raw)
    m = len(results)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * results[idx].p_raw
        running_max = max(running_max, adj)
        p_holm = min(running_max, 1.0)
        results[idx].p_holm = p_holm
        results[idx].significant = p_holm < ALPHA


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────

PAIRWISE_COMPARISONS = [
    ("k0_vs_k1_dynamic", (0, "dynamic"), (1, "dynamic")),
    ("k1_vs_k3_dynamic", (1, "dynamic"), (3, "dynamic")),
    ("k0_vs_k3_dynamic", (0, "dynamic"), (3, "dynamic")),
    ("static_vs_dynamic_k0", (0, "static"), (0, "dynamic")),
    ("static_vs_dynamic_k1", (1, "static"), (1, "dynamic")),
    ("static_vs_dynamic_k3", (3, "static"), (3, "dynamic")),
]

METRIC_SERIES_ATTR = {
    "Gamma": "gamma",
    "C_sep": "c_sep",
    "Delta_epsilon_static": "delta_eps_static",
}


def run_pairwise_family(
    metric: str, conditions: Dict[Tuple[int, str], ConditionData], paired: bool,
) -> List[TestResult]:
    attr = METRIC_SERIES_ATTR[metric]
    results = []
    for comparison, key_a, key_b in PAIRWISE_COMPARISONS:
        a = getattr(conditions[key_a], attr)
        b = getattr(conditions[key_b], attr)
        fn = paired_test if paired else independent_test
        results.append(fn(metric, comparison, a, b))
    holm_bonferroni(results)
    return results


def run_rho_ripple_family(conditions: Dict[Tuple[int, str], ConditionData]) -> List[TestResult]:
    results = []
    for (k_cps, mode), cd in sorted(conditions.items()):
        results.append(one_sample_test("rho_ripple", f"k{k_cps}_{mode}_vs_0", cd.rho_ripple))
    holm_bonferroni(results)
    return results


def build_latex_table(all_results: List[TestResult]) -> str:
    rows = []
    for r in all_results:
        rows.append({
            "Metric": r.metric.replace("_", "\\_"),
            "Comparison": r.comparison.replace("_", "\\_"),
            "Test": r.test,
            "$n$": r.n,
            "Statistic": f"{r.statistic:.3f}",
            "$p$ (Holm)": (
                "--" if r.p_holm is None
                else "$<10^{-300}$" if r.p_holm == 0.0
                else f"{r.p_holm:.3g}"
            ),
            "Effect size": f"{r.effect_size:.3f}" if not np.isnan(r.effect_size) else "--",
            "Sig.\\ ($\\alpha$=0.05)": "Yes" if r.significant else "No",
        })
    df = pd.DataFrame(rows)
    return _df_to_latex(
        df,
        caption="Significance tests for the $k_{cps} \\times$ "
                "runway\\_assignment\\_mode grid comparisons reported "
                "descriptively in \\autoref{tab:throughput_results} and "
                "\\autoref{tab:delay_ripple}. $p$-values are Holm-Bonferroni "
                "corrected within each metric's family of 6 pairwise tests "
                "(or 6 one-sample tests for $\\rho_{ripple}$).",
        label="tab:significance_tests",
    )


def write_notes(
    out_dir: str, matching_ok: bool, matching_msg: str, review_problems: List[str],
    all_results: List[TestResult],
) -> None:
    lines = ["# Phase III significance testing -- notes\n"]
    lines.append("## Self-review gate (derived vs. published means)\n")
    if review_problems:
        lines.append("**MISMATCHES FOUND -- results below are NOT trustworthy until resolved:**\n")
        lines.extend(f"- {p}" for p in review_problems)
    else:
        lines.append("All derived per-episode series' means match the published "
                      "`tab:throughput_results`/`tab:delay_ripple` mean values within 1e-3.\n")

    lines.append("\n## Cross-condition matching verification\n")
    lines.append(matching_msg + "\n")
    lines.append(
        f"Decision: comparisons run as **{'paired (Wilcoxon signed-rank / paired t-test)' if matching_ok else 'independent (Mann-Whitney U / Welch t-test)'}**.\n"
    )

    lines.append("\n## Consistency check against results.tex / discussion.tex / conclusion.tex\n")
    by_key = {(r.metric, r.comparison): r for r in all_results}

    c_sep_01 = by_key.get(("C_sep", "k0_vs_k1_dynamic"))
    c_sep_13 = by_key.get(("C_sep", "k1_vs_k3_dynamic"))
    if c_sep_01 and c_sep_13:
        lines.append(
            f"- discussion.tex claims C_sep's static/dynamic gap 'opens up between k=0 and "
            f"k=1, then holds roughly flat from k=1 to k=3'. Test result: k0-vs-k1 "
            f"p_holm={c_sep_01.p_holm:.3g} (sig={c_sep_01.significant}, "
            f"effect={c_sep_01.effect_size:.3f}); k1-vs-k3 p_holm={c_sep_13.p_holm:.3g} "
            f"(sig={c_sep_13.significant}, effect={c_sep_13.effect_size:.3f}). "
        )
        if c_sep_13.significant and not c_sep_01.significant:
            lines.append("  **CONTRADICTS the claim** -- k1-vs-k3 is significant while k0-vs-k1 is not.\n")
        elif c_sep_13.significant and c_sep_01.significant:
            lines.append(
                "  Both are statistically significant at n=2000 (high power can make even a "
                "small residual k1-vs-k3 shift detectable) -- compare the effect sizes above, "
                "not just significance, before deciding whether 'holds roughly flat' still holds "
                "as a *practical*-significance claim.\n"
            )
        else:
            lines.append("  Consistent with the claim (k0-vs-k1 significant, k1-vs-k3 not).\n")

    gamma_13 = by_key.get(("Gamma", "k1_vs_k3_dynamic"))
    gamma_sd_1 = by_key.get(("Gamma", "static_vs_dynamic_k1"))
    gamma_sd_3 = by_key.get(("Gamma", "static_vs_dynamic_k3"))
    if gamma_13 and gamma_sd_1 and gamma_sd_3:
        lines.append(
            f"\n- conclusion.tex claims throughput 'converg[es] toward the static value as k "
            f"increases'. Test result: k1-vs-k3 (dynamic) p_holm={gamma_13.p_holm:.3g} "
            f"(sig={gamma_13.significant}); static-vs-dynamic gap at k=1 effect="
            f"{gamma_sd_1.effect_size:.3f} vs. at k=3 effect={gamma_sd_3.effect_size:.3f}. "
        )
        if abs(gamma_sd_3.effect_size) < abs(gamma_sd_1.effect_size):
            lines.append("  Effect size shrinks from k=1 to k=3, consistent with the convergence claim.\n")
        else:
            lines.append("  Effect size does NOT shrink from k=1 to k=3 -- **weakens the convergence claim**.\n")

    with open(os.path.join(out_dir, "significance_tests_notes.md"), "w") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-root", default="experiments/cps_eval/scale_10k_20260820_153701")
    p.add_argument("--out-dir", default="cps_coordination/figures/paper_report_20260821_cap50")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading combos from {args.sweep_root} ...")
    conditions = load_conditions(args.sweep_root)
    expected = {(k, m) for k in (0, 1, 3) for m in ("static", "dynamic")}
    missing = expected - set(conditions.keys())
    if missing:
        print(f"WARNING: missing combos: {sorted(missing)}")
    for key, cd in sorted(conditions.items()):
        print(f"  {cd.combo}: n_episodes(gamma)={len(cd.gamma)}, n_episodes(c_sep)={len(cd.c_sep)}, "
              f"n_episodes(delta_eps)={len(cd.delta_eps_static)}, n_episodes(rho_ripple)={len(cd.rho_ripple)}")

    review_problems = self_review_gate(conditions)
    if review_problems:
        print("SELF-REVIEW GATE FAILED:")
        for prob in review_problems:
            print(f"  - {prob}")
    else:
        print("Self-review gate passed: derived per-episode means match published aggregates.")

    matching_ok, matching_msg = verify_matching(conditions)
    print(matching_msg)

    all_results: List[TestResult] = []
    for metric in METRIC_SERIES_ATTR:
        all_results.extend(run_pairwise_family(metric, conditions, paired=matching_ok))
    all_results.extend(run_rho_ripple_family(conditions))

    rows = []
    for r in all_results:
        rows.append({
            "metric": r.metric, "comparison": r.comparison, "test": r.test, "n": r.n,
            "statistic": r.statistic, "p_raw": r.p_raw, "p_holm": r.p_holm,
            "effect_size": r.effect_size, "effect_size_name": r.effect_size_name,
            "significant_alpha_0.05": r.significant, "note": r.note,
        })
    full_df = pd.DataFrame(rows)
    full_csv_path = os.path.join(args.out_dir, "significance_tests_full.csv")
    full_df.to_csv(full_csv_path, index=False)
    print(f"Wrote {full_csv_path}")

    latex = build_latex_table(all_results)
    tex_path = os.path.join(args.out_dir, "tab_significance_tests.tex")
    with open(tex_path, "w") as fh:
        fh.write(latex)
    print(f"Wrote {tex_path}")

    write_notes(args.out_dir, matching_ok, matching_msg, review_problems, all_results)
    print(f"Wrote {os.path.join(args.out_dir, 'significance_tests_notes.md')}")

    print("\n--- Summary ---")
    print(full_df.to_string(index=False))


if __name__ == "__main__":
    main()
