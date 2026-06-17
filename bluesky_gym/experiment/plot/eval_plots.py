"""
Evaluation plotting suite.

Generates visual analytics for post-training model evaluations. This includes 
single-run performance dashboards, cross-run success rate summaries, and 
detailed episode-by-episode timelines. It dynamically accommodates custom 
metrics extracted via the `MetricExtractor` class.
"""

from __future__ import annotations

from .style import _apply_style, _color, _plt, _get_ticker, _smooth, _save_or_show

import math
import numpy as np
from typing import Optional

def _extra_numeric_keys(rows: list[dict], yaml_data: Optional[dict] = None, metrics: Optional[list[str]] = None) -> list[str]:
    """
    Return extra numeric column names beyond the four fixed fields.
    
    If yaml_data is provided, it cross-references the 'overall' keys to ensure
    we only plot metrics that were explicitly aggregated.
    """
    fixed = {"episode", "group", "is_success", "total_reward"}
    if not rows:
        return []

    candidates = [
        k for k in rows[0] 
        if k not in fixed
        and isinstance(rows[0][k], (int, float))
        and any(not (isinstance(r[k], float) and math.isnan(r[k])) for r in rows)
    ]

    if yaml_data and "overall" in yaml_data:
        # The YAML 'overall' section contains keys like 'mean_total_reward', 
        # 'std_total_reward', and our extras.
        yaml_keys = yaml_data["overall"].keys()
        
        # We only keep candidates that exist in the YAML 
        # (The MetricExtractor output keys match the CSV column names)
        candidates = [k for k in candidates if k in yaml_keys]

    if metrics is not None:
        candidates = [k for k in candidates if k in metrics]

    return candidates

def plot_eval_dashboard(
    label:   str,
    rows:    list[dict],
    yaml_data: dict | None = None,
    out_dir: Optional[str] = None,
    title:   Optional[str] = None,
    metrics: Optional[list[str]] = None,
) -> None:
    """
    Single-run evaluation dashboard with dynamic grid sizing.

    Always shows:
      1. Reward distribution by group (violin + jitter)
      2. Success rate by group (horizontal bar)
      3. Episode timeline (scatter + rolling mean)
      
    Then dynamically appends:
      - A bar chart for *each* extra metric extracted.
      - Or a reward histogram if no extra metrics exist.
    """
    _apply_style()
    plt = _plt()
    ticker = _get_ticker()

    groups       = sorted({r["group"] for r in rows})
    n_groups     = len(groups)
    group_colors = {g: _color(i) for i, g in enumerate(groups)}
    extras       = _extra_numeric_keys(rows, yaml_data, metrics)
    by_group     = {g: [r for r in rows if r["group"] == g] for g in groups}

    # ── Dynamic Grid Calculation ─────────────────────────────────────────────
    # Base 3 plots + either 1 plot per extra metric OR 1 fallback histogram
    n_plots = 3 + (len(extras) if extras else 1)
    
    # Force columns to be either 2 or 3 for optimal viewing
    cols = 3 if n_plots >= 5 or n_plots == 3 else 2
    n_rows_grid = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(n_rows_grid, cols, figsize=(cols * 6.5, n_rows_grid * 4.5))
    
    # Flatten axes for easy sequential iteration
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    ax_idx = 0

    # ── 1. Reward violin + jitter ────────────────────────────────────────────
    ax_viol = axes[ax_idx]; ax_idx += 1
    reward_data = [np.array([r["total_reward"] for r in by_group[g]]) for g in groups]
    vp = ax_viol.violinplot(reward_data, positions=range(n_groups),
                             showmedians=True, showextrema=False)
    for pc, g in zip(vp["bodies"], groups):
        pc.set_facecolor(group_colors[g])
        pc.set_alpha(0.55)
    vp["cmedians"].set_color("#333333")
    vp["cmedians"].set_linewidth(2)
    for j, g in enumerate(groups):
        jx = np.random.default_rng(j).uniform(-0.12, 0.12, len(by_group[g]))
        ys = [r["total_reward"] for r in by_group[g]]
        cs = [("#2ecc71" if r["is_success"] else "#e74c3c") for r in by_group[g]]
        ax_viol.scatter(j + jx, ys, c=cs, s=18, alpha=0.7, zorder=3, linewidths=0)
    ax_viol.set_xticks(range(n_groups))
    ax_viol.set_xticklabels(groups)
    ax_viol.set_xlabel("Group")
    ax_viol.set_ylabel("Total Reward")
    ax_viol.set_title("Reward Distribution by Group")
    
    from matplotlib.lines import Line2D
    ax_viol.legend(
        handles=[Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7, label="success"),
                 Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=7, label="failure")],
        fontsize=8,
    )

    # ── 2. Success rate horizontal bars ──────────────────────────────────────
    ax_sr = axes[ax_idx]; ax_idx += 1
    sr_vals = [sum(r["is_success"] for r in by_group[g]) / len(by_group[g]) for g in groups]
    n_eps   = [len(by_group[g]) for g in groups]
    bars = ax_sr.barh(range(n_groups), sr_vals, color=[group_colors[g] for g in groups],
                      alpha=0.75, zorder=3)
    for j, (bar, sr, n) in enumerate(zip(bars, sr_vals, n_eps)):
        ax_sr.text(min(sr + 0.05, 1), bar.get_y() + bar.get_height() / 2,
                   f"{sr:.1%}", va="center", fontsize=8, color="#333333")
        ax_sr.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                   f"(n={n})", va="center", fontsize=8, color="#333333")
    overall_sr = sum(r["is_success"] for r in rows) / len(rows)
    ax_sr.axvline(overall_sr, color="#444444", linewidth=1.5, linestyle="--", label=f"overall {overall_sr:.1%}")
    ax_sr.set_yticks(range(n_groups))
    ax_sr.set_yticklabels(groups)
    ax_sr.set_xlim(0, 1.15)
    ax_sr.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_sr.set_xlabel("Success Rate")
    ax_sr.set_ylabel("Group")
    ax_sr.set_title("Success Rate by Group")
    ax_sr.legend(fontsize=8)

    # ── 3. Episode timeline ──────────────────────────────────────────────────
    ax_time = axes[ax_idx]; ax_idx += 1
    eps     = [r["episode"]      for r in rows]
    rewards = [r["total_reward"] for r in rows]
    colors  = [("#2ecc71" if r["is_success"] else "#e74c3c") for r in rows]
    ax_time.scatter(eps, rewards, c=colors, s=20, alpha=0.75, zorder=3, linewidths=0)
    if len(rows) >= 5:
        rm = _smooth(np.array(rewards, dtype=float), max(3, len(rows) // 10))
        ax_time.plot(eps, rm, color="#444444", linewidth=1.5, label="rolling mean", zorder=4)
        ax_time.legend(
            handles=[Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ecc71", markersize=7, label="success"),
                     Line2D([0],[0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=7, label="failure"),
                     Line2D([0],[0], color="#444444", linewidth=1.5, label="rolling mean")],
            fontsize=8
        )
    ax_time.set_xlabel("Episode")
    ax_time.set_ylabel("Total Reward")
    ax_time.set_title("Episode Timeline")

    # ── 4. Dynamic Extra Metrics (or Fallback Histogram) ─────────────────────
    if extras:
        # Generate a bar chart for every extra metric we tracked
        for key in extras:
            ax_ext = axes[ax_idx]
            ax_idx += 1
            
            vals = [np.nanmean([r[key] for r in by_group[g]]) for g in groups]
            stds = [np.nanstd( [r[key] for r in by_group[g]]) for g in groups]
            
            bars2 = ax_ext.bar(range(n_groups), vals, yerr=stds,
                               color=[group_colors[g] for g in groups],
                               alpha=0.75, capsize=4, zorder=3,
                               error_kw=dict(elinewidth=1, ecolor="#666666", alpha=1))
            
            max_val = max([abs(x) for x in vals]) if vals else 0
            offset = max(max_val * 0.02, 0.01)
            for bar, v in zip(bars2, vals):
                sign = 1 if v >= 0 else -1
                y_pos = v + (sign * offset)
                va_align = "bottom" if v >= 0 else "top"
                
                ax_ext.text(bar.get_x() + bar.get_width() / 2, 
                            y_pos,
                            f"{v:.2f}", 
                            ha="center", 
                            va=va_align,
                            fontsize=7, 
                            rotation=90,
                            color="#333333",
                            # zorder=4 ensures the text and its box draw OVER the error bars
                            zorder=4, 
                            # Solid white box masks the error bar running behind the text
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
                
            ax_ext.set_xticks(range(n_groups))
            ax_ext.set_xticklabels(groups)
            ax_ext.set_xlabel("Group")
            ax_ext.set_ylabel(key.replace("_", " ").title())
            ax_ext.set_title(f"{key.replace('_', ' ').title()} by Group  (mean ± std)")
    else:
        # Fallback: overall reward histogram coloured by group
        ax_hist = axes[ax_idx]; ax_idx += 1
        for g in groups:
            vals = [r["total_reward"] for r in by_group[g]]
            ax_hist.hist(vals, bins=12, alpha=0.55, color=group_colors[g], label=g, zorder=3)
        ax_hist.set_xlabel("Total Reward")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Reward Histogram by Group")
        ax_hist.legend(fontsize=8)

    # ── Clean up empty subplots ──────────────────────────────────────────────
    for i in range(ax_idx, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title or f"Evaluation Dashboard — {label}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
    _save_or_show(fig, out_dir, f"eval_dashboard_{label}.png", plt)

def plot_eval_summary(
    labels:    list[str],
    yaml_data: list[dict],
    metric:    str           = "success_rate",
    out_dir:   Optional[str] = None,
    title:     Optional[str] = None,
) -> None:
    """
    Cross-run grouped bar chart for one metric, with mean±std error bars
    drawn from the YAML overall section.  Overall value annotated as a
    dashed line per run.
    """
    _apply_style()
    plt = _plt()
    ticker = _get_ticker()

    all_groups = sorted({g for d in yaml_data for g in d.get("per_group", {}).keys()})
    if not all_groups:
        return

    n_runs, n_groups = len(labels), len(all_groups)
    bar_w = 0.75 / n_runs
    x = np.arange(n_groups)

    # Infer std key: e.g. success_rate → no std in YAML; mean_total_reward → std_total_reward
    std_key_map = {"mean_total_reward": "std_total_reward"}
    std_key = std_key_map.get(metric)

    is_pct = metric in ("success_rate",)
    fmt_v  = (lambda v: f"{v:.1%}") if is_pct else (lambda v: f"{v:.2f}")

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.4 + 2), 5))

    for i, (label, d) in enumerate(zip(labels, yaml_data)):
        per_group = d.get("per_group", {})
        overall   = d.get("overall",   {})

        vals = np.array([per_group.get(g, {}).get(metric, np.nan) for g in all_groups])
        errs = None
        if std_key:
            errs = np.array([per_group.get(g, {}).get(std_key, np.nan) for g in all_groups])

        offset = (i - (n_runs - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, vals,
            width=bar_w * 0.9,
            color=_color(i), alpha=0.80,
            yerr=errs if errs is not None else None,
            capsize=3,
            error_kw=dict(elinewidth=1, ecolor="#555555"),
            label=label, zorder=3,
        )
        # Value labels
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.01 if is_pct else 0.05),
                        fmt_v(v),
                        rotation=90 if is_pct else 0,
                        ha="center", va="bottom", fontsize=7, color="#333333")

        # Overall dashed line
        ov = overall.get(metric, np.nan)
        if not np.isnan(ov):
            ax.axhline(ov, color=_color(i), linewidth=1.2, linestyle="--",
                       alpha=0.6, label=f"{label} overall ({fmt_v(ov)})")

    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=30, ha="right")
    ax.set_xlabel("Group")
    ax.set_ylabel(metric.replace("_", " ").title())
    if is_pct:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1.15)
    ax.set_title(title or f"Eval — {metric.replace('_', ' ').title()} by Group")
    ax.legend(fontsize=8, ncol=min(n_runs * 2, 4))
    fig.tight_layout()
    _save_or_show(fig, out_dir, "eval_summary.png", plt)


def plot_eval_episodes(
    labels:   list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    title:    Optional[str] = None,
) -> None:
    """
    Per-run 2x2 comparison grid:
      [0,0]  Reward boxplots by group (one subplot per run, shared y)
      [0,1]  Success rate grouped bars across runs
      [1,0]  Episode timeline scatter for all runs overlaid
      [1,1]  Reward distributions (overlapping histograms per run)
    """
    _apply_style()
    plt  = _plt()
    ticker = _get_ticker()

    all_groups = sorted({r["group"] for rows in all_rows for r in rows})
    n_runs = len(labels)

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)
    ax_box  = fig.add_subplot(gs[0, 0])
    ax_sr   = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    # ── [0,0]  Reward boxplots — all runs side-by-side per group ────────────
    x = np.arange(len(all_groups))
    bw = 0.7 / n_runs
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        by_group = {g: [r["total_reward"] for r in rows if r["group"] == g]
                    for g in all_groups}
        offset = (i - (n_runs - 1) / 2) * bw
        bp = ax_box.boxplot(
            [by_group.get(g, [0]) for g in all_groups],
            positions=x + offset,
            widths=bw * 0.85,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=_color(i), linewidth=1.2),
            capprops=dict(color=_color(i), linewidth=1.2),
            flierprops=dict(marker=".", color=_color(i), alpha=0.5, markersize=4),
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(_color(i))
            patch.set_alpha(0.65)
        # Invisible bar for legend
        ax_box.bar(0, 0, color=_color(i), alpha=0.65, label=label)

    ax_box.set_xticks(x)
    ax_box.set_xticklabels(all_groups, rotation=20, ha="right")
    ax_box.set_ylabel("Total Reward")
    ax_box.set_title("Reward Distribution by Group")
    ax_box.legend(fontsize=8)

    min_v = 1

    # ── [0,1]  Success rate grouped bars ─────────────────────────────────────
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        by_group = {g: [r for r in rows if r["group"] == g] for g in all_groups}
        sr_vals  = [sum(r["is_success"] for r in by_group.get(g, [])) /
                    max(1, len(by_group.get(g, []))) for g in all_groups]
        offset   = (i - (n_runs - 1) / 2) * bw
        bars     = ax_sr.bar(x + offset, sr_vals, width=bw * 0.85,
                             color=_color(i), alpha=0.80, label=label, zorder=3)
        for bar, v in zip(bars, sr_vals):
            min_v = min(min_v, v)
            if v >= 0.999:
                continue  # Skip labels for perfect scores to avoid clutter
            ax_sr.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + 0.003,
                       f"{v:.1%}", ha="center", fontsize=4, color="#333333", rotation=90)

    ax_sr.set_xticks(x)
    ax_sr.set_xticklabels(all_groups, rotation=20, ha="right")
    ax_sr.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_sr.set_ylim(min_v*0.95 - 0.02, 1.08)
    ax_sr.set_xlabel("Group")
    ax_sr.set_ylabel("Success Rate")
    ax_sr.set_title("Success Rate by Group")
    ax_sr.legend(fontsize=8, loc="lower left")

    # ── [1,0]  Episode timeline overlay ──────────────────────────────────────
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        eps     = [r["episode"]      for r in rows]
        rewards = np.array([r["total_reward"] for r in rows], dtype=float)
        rm      = _smooth(rewards, max(3, len(rows) // 10))
        ax_time.plot(eps, rm, color=_color(i), linewidth=2.0, label=label, zorder=3)
        ax_time.fill_between(eps, rewards, rm,
                             color=_color(i), alpha=0.08, linewidth=0)

    ax_time.set_xlabel("Episode")
    ax_time.set_ylabel("Total Reward")
    ax_time.set_title("Episode Timeline (rolling mean)")
    ax_time.legend(fontsize=8)

    # ── [1,1]  Reward histograms ──────────────────────────────────────────────
    all_rewards = [r["total_reward"] for rows in all_rows for r in rows]
    bins = np.linspace(min(all_rewards), max(all_rewards), 20)
    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        rewards = [r["total_reward"] for r in rows]
        ax_hist.hist(rewards, bins=bins, color=_color(i), alpha=0.55,
                     label=label, zorder=3)
        # Median line
        med = float(np.median(rewards))
        ax_hist.axvline(med, color=_color(i), linewidth=1.5, linestyle="--")

    ax_hist.set_xlabel("Total Reward")
    ax_hist.set_ylabel("Episodes")
    ax_hist.set_title("Reward Distribution (dashed = median)")
    ax_hist.legend(fontsize=8)

    fig.suptitle(title or "Evaluation Comparison", fontsize=13, fontweight="bold")
    _save_or_show(fig, out_dir, "eval_episodes.png", plt)

def _compute_pareto_front(
    run_ids:   list[str],
    summaries: list[dict],
    rank_spec: dict,
) -> tuple[list[str], dict[str, int]]:
    """
    Identify non-dominated runs given rank_spec objectives.

    Returns (front_ids, dominated_count) where dominated_count[rid] is the
    number of other runs that strictly dominate run rid.
    """
    metrics = [
        m for m in rank_spec
        if rank_spec[m] in (max, min) and any(
            isinstance(s.get(m), (int, float)) and not np.isnan(s.get(m, float("nan")))
            for s in summaries
        )
    ]

    def _val(s: dict, m: str) -> float:
        v = s.get(m, float("nan"))
        if not isinstance(v, (int, float)):
            return float("nan")
        return -float(v) if rank_spec[m] is min else float(v)

    dominated_count: dict[str, int] = {rid: 0 for rid in run_ids}
    n = len(run_ids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            at_least_as_good = True
            strictly_better  = False
            for m in metrics:
                vi = _val(summaries[i], m)
                vj = _val(summaries[j], m)
                if np.isnan(vi) or np.isnan(vj):
                    continue
                if vj < vi:
                    at_least_as_good = False
                    break
                if vj > vi:
                    strictly_better = True
            if at_least_as_good and strictly_better:
                dominated_count[run_ids[i]] += 1

    front_ids = [rid for rid in run_ids if dominated_count[rid] == 0]
    return front_ids, dominated_count


def plot_compare_evaluations(
    run_ids:             list[str],
    all_raw_rows:        list[list[dict]],
    overall_summaries:   list[dict],
    per_group_summaries: list[dict],
    rank_spec:           Optional[dict] = None,
    out_dir:             Optional[str]  = None,
    title:               Optional[str]  = None,
) -> None:
    """
    Comprehensive multi-panel comparison figure for post-training evaluations.

    Designed to consume the return value of compare_runs.compare_evaluations()
    directly:

        run_ids, all_raw_rows, overall_summaries, per_group_summaries, rank_spec
            = compare_evaluations(...)
        plot_compare_evaluations(run_ids, all_raw_rows, overall_summaries,
                                 per_group_summaries, rank_spec)

    Panels (2 × 3 grid)
    -------------------
    [0,0]  Overall success rate    — horizontal bars; Pareto-front runs starred.
    [0,1]  Overall mean reward     — bars with ±std; Pareto-front runs starred.
    [0,2]  Pareto domination rank  — horizontal bars showing dominated_count per
                                     run (0 = on the front); front members starred.
    [1,0]  Reward distributions    — overlapping violins from raw episode data.
    [1,1]  Win-count scoreboard    — bar showing per-metric wins per run.
    [1,2]  Pareto scatter          — 2-D scatter of the top-2 ranked objectives
                                     with the Pareto front highlighted.  Falls
                                     back to a dominance-count bar when <2
                                     plottable objectives exist.

    Parameters
    ----------
    rank_spec : dict mapping metric name → callable (max/min), as returned by
                compare_evaluations().  Used to determine winners and Pareto
                membership.  If None, no winner annotations are drawn.
    """
    _apply_style()
    plt    = _plt()
    ticker = _get_ticker()

    n_runs    = len(run_ids)
    run_color = {rid: _color(i) for i, rid in enumerate(run_ids)}
    rank_spec = rank_spec or {}

    # ── Pareto computation ────────────────────────────────────────────────────
    front_ids: list[str] = []
    dominated_count: dict[str, int] = {rid: 0 for rid in run_ids}
    if rank_spec and n_runs > 1:
        front_ids, dominated_count = _compute_pareto_front(
            run_ids, overall_summaries, rank_spec
        )

    def _winners(summaries: list[dict], metric: str) -> set[str]:
        fn = rank_spec.get(metric)
        if fn is None or fn not in (max, min):
            return set()
        pairs = [
            (rid, s.get(metric))
            for rid, s in zip(run_ids, summaries)
            if isinstance(s.get(metric), (int, float))
            and not (isinstance(s.get(metric), float) and np.isnan(s.get(metric)))
        ]
        if not pairs:
            return set()
        best = fn(v for _, v in pairs)
        return {rid for rid, v in pairs if v == best}

    win_counts: dict[str, int] = {rid: 0 for rid in run_ids}
    for metric in rank_spec:
        for rid in _winners(overall_summaries, metric):
            win_counts[rid] += 1

    y_pos = np.arange(n_runs)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # ── [0,0]  Overall success rate ──────────────────────────────────────────
    ax      = axes[0, 0]
    sr_vals = [s.get("success_rate", np.nan) for s in overall_summaries]
    bars    = ax.barh(y_pos, sr_vals, color=[run_color[r] for r in run_ids],
                      alpha=0.80, zorder=3)
    for bar, rid, v in zip(bars, run_ids, sr_vals):
        if np.isnan(v):
            continue
        star = "★ " if rid in front_ids else ""
        ax.text(min(v + 0.03, 1.10), bar.get_y() + bar.get_height() / 2,
                f"{star}{v:.1%}", va="center", fontsize=8, color="#333333")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(run_ids)
    ax.set_xlim(0, 1.20)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Success Rate")
    ax.set_title("Overall Success Rate  (★ = Pareto front)")

    # ── [0,1]  Overall mean reward ───────────────────────────────────────────
    ax       = axes[0, 1]
    rew_vals = np.array([s.get("mean_total_reward", np.nan) for s in overall_summaries])
    rew_stds = np.array([s.get("std_total_reward",  np.nan) for s in overall_summaries])
    bars = ax.bar(
        y_pos, rew_vals,
        yerr=np.where(np.isnan(rew_stds), 0, rew_stds),
        color=[run_color[r] for r in run_ids],
        alpha=0.80, capsize=4, zorder=3,
        error_kw=dict(elinewidth=1, ecolor="#555555"),
    )
    for bar, rid, v in zip(bars, run_ids, rew_vals):
        if np.isnan(v):
            continue
        star = "★" if rid in front_ids else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.02 + 0.01,
                f"{star}{v:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(y_pos)
    ax.set_xticklabels(run_ids, rotation=15, ha="right")
    ax.set_ylabel("Mean Total Reward")
    ax.set_title("Overall Mean Reward (±std)  (★ = Pareto front)")

    # ── [0,2]  Pareto domination rank ────────────────────────────────────────
    ax          = axes[0, 2]
    sorted_dom  = sorted(run_ids, key=lambda r: dominated_count[r])
    dom_vals    = [dominated_count[r] for r in sorted_dom]
    y_dom       = np.arange(len(sorted_dom))
    bar_colors  = [
        run_color[r] if r not in front_ids else run_color[r]
        for r in sorted_dom
    ]
    bars = ax.barh(y_dom, dom_vals, color=bar_colors, alpha=0.80, zorder=3)
    for bar, rid, v in zip(bars, sorted_dom, dom_vals):
        star = "★ " if rid in front_ids else "  "
        label_str = f"{star}{rid}  (dominated by {v})"
        ax.text(
            max(v + 0.05, 0.1),
            bar.get_y() + bar.get_height() / 2,
            str(v), va="center", fontsize=9, fontweight="bold", color="#333333",
        )
    ax.set_yticks(y_dom)
    ax.set_yticklabels([("★ " if r in front_ids else "  ") + r for r in sorted_dom])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Number of runs that dominate this run")
    ax.set_title("Pareto Domination Rank  (0 = non-dominated)")

    # ── [1,0]  Reward distributions ──────────────────────────────────────────
    ax               = axes[1, 0]
    all_rewards_flat = [r["total_reward"] for rows in all_raw_rows for r in rows]
    if all_rewards_flat and n_runs > 1:
        vp = ax.violinplot(
            [[r["total_reward"] for r in rows] for rows in all_raw_rows],
            positions=np.arange(n_runs),
            showmedians=True, showextrema=False,
        )
        for pc, rid in zip(vp["bodies"], run_ids):
            pc.set_facecolor(run_color[rid])
            pc.set_alpha(0.55)
        vp["cmedians"].set_color("#333333")
        vp["cmedians"].set_linewidth(2)
        ax.set_xticks(np.arange(n_runs))
        ax.set_xticklabels(run_ids, rotation=15, ha="right")
    elif all_rewards_flat:
        ax.hist([r["total_reward"] for r in all_raw_rows[0]],
                bins=15, color=run_color[run_ids[0]], alpha=0.75, zorder=3)
        ax.set_xticks([])
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution (violin)")

    # ── [1,1]  Win-count scoreboard ──────────────────────────────────────────
    ax          = axes[1, 1]
    sorted_runs = sorted(win_counts, key=lambda r: win_counts[r], reverse=True)
    wc_vals     = [win_counts[r] for r in sorted_runs]
    y_sc        = np.arange(len(sorted_runs))
    bars        = ax.barh(y_sc, wc_vals,
                          color=[run_color[r] for r in sorted_runs],
                          alpha=0.80, zorder=3)
    for bar, v in zip(bars, wc_vals):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_yticks(y_sc)
    ax.set_yticklabels(sorted_runs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_title("Per-Metric Win Count")

    if rank_spec:
        def dir_labels(fn):
            if fn is max:
                return '↑'
            elif fn is min:
                return '↓'
            else:
                return '-'
    
        direction_strs = [
            f"{m}({dir_labels(fn)})"
            for m, fn in rank_spec.items()
        ]
        ax.set_xlabel(
            f"Metrics won (of {len(rank_spec)}): " + ", ".join(direction_strs),
            fontsize=7,
        )
    else:
        ax.set_xlabel("Metrics won")

    # ── [1,2]  Pareto scatter (top-2 objectives) or fallback ─────────────────
    ax = axes[1, 2]
    # Collect objectives that have valid values for every run
    plottable = [
        m for m in rank_spec
        if sum(
            1 for s in overall_summaries
            if isinstance(s.get(m), (int, float))
            and not np.isnan(s.get(m, float("nan")))
        ) == n_runs
    ]

    if len(plottable) >= 2:
        mx, my = plottable[0], plottable[1]
        dir_x = rank_spec[mx]
        dir_y = rank_spec[my]

        xs = np.array([s.get(mx, np.nan) for s in overall_summaries])
        ys = np.array([s.get(my, np.nan) for s in overall_summaries])

        # Plot all runs
        for i, (rid, x, y) in enumerate(zip(run_ids, xs, ys)):
            is_front = rid in front_ids
            marker   = "*" if is_front else "o"
            size     = 220 if is_front else 80
            zorder   = 5 if is_front else 3
            ax.scatter(x, y, color=run_color[rid], marker=marker,
                       s=size, zorder=zorder, linewidths=0.5,
                       edgecolors="#333333" if is_front else "none")
            ax.annotate(
                ("★ " if is_front else "") + rid,
                (x, y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="#333333",
            )

        # Draw step-line connecting Pareto-front points (sorted by x)
        if len(front_ids) >= 2:
            front_pts = sorted(
                [(s.get(mx, np.nan), s.get(my, np.nan))
                 for rid, s in zip(run_ids, overall_summaries)
                 if rid in front_ids],
                key=lambda p: p[0],
                reverse=(dir_x is max),
            )
            fx, fy = zip(*front_pts)
            ax.step(fx, fy, where="post", color="#888888",
                    linewidth=1.2, linestyle="--", zorder=2, label="Pareto front")
            ax.legend(fontsize=8)

        hint_x = "↑" if dir_x is max else "↓"
        hint_y = "↑" if dir_y is max else "↓"
        ax.set_xlabel(f"{mx} {hint_x}", fontsize=9)
        ax.set_ylabel(f"{my} {hint_y}", fontsize=9)
        ax.set_title(f"Pareto Scatter: {mx} vs {my}")

    else:
        # Fallback: dominance-count bar (same data as [0,2] but vertical)
        sorted_dom2 = sorted(run_ids, key=lambda r: dominated_count[r])
        dom_vals2   = [dominated_count[r] for r in sorted_dom2]
        y_d2        = np.arange(len(sorted_dom2))
        ax.bar(y_d2, dom_vals2,
               color=[run_color[r] for r in sorted_dom2],
               alpha=0.80, zorder=3)
        ax.set_xticks(y_d2)
        ax.set_xticklabels(
            [("★ " if r in front_ids else "") + r for r in sorted_dom2],
            rotation=15, ha="right",
        )
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_ylabel("Dominated by N other runs")
        ax.set_title("Pareto Domination Count  (0 = non-dominated)")

    fig.suptitle(title or "Evaluation Comparison — Model/Run Ranking",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.97))
    _save_or_show(fig, out_dir, "eval_compare.png", plt)