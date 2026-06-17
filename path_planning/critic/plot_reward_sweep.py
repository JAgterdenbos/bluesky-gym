"""
plot_reward_sweep.py
--------------------------
Data-agnostic: all annotations and subtitle claims are computed from the
loaded DataFrames at runtime. No hard-coded data values in strings.

λ = ‖∇Q_sparse‖ / ‖∇Q_aug‖ is a ratio of L2 norms — strictly ≥ 0.
Any axis or ribbon that would show negative λ is a bug; all λ axes are
clipped to [0, ...].

Figures
-------
  fig1_dominance_vs_bonus.png
  fig2_q_decomposition.png
  fig3_heading_sweep.png
  fig4_dominance_space.png

Usage
-----
    python plot_reward_sweep.py \
        [--csv   experiments/reward_sweep/results/reward_sweep.csv] \
        [--heading-csv experiments/reward_sweep/results/reward_sweep_heading.csv] \
        [--outdir experiments/reward_sweep/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# ── global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f8f9fa",
    "axes.grid":          True,
    "grid.color":         "#e5e5e5",
    "grid.linewidth":     0,
    "axes.axisbelow":     True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": True,
    "axes.edgecolor":     "#cccccc",
    "font.size":          11,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    9,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
})

# 7 qualitatively distinct colours — order matches sorted goal_bonus values
GB_COLOURS = [
    "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
    "#9467bd", "#8c564b", "#17becf",
]

C_AUG    = "#2166ac"
C_SPARSE = "#d6604d"
C_TOTAL  = "#4dac26"
C_SR     = "#2ca02c"


def _should_use_log_scale(data: np.ndarray, threshold: int = 3) -> bool:
    """Check if the dynamic range justifies a log scale (ignoring zero)."""
    pos_data = data[data > 0]
    if len(pos_data) < 2:
        return False
    return np.log10(pos_data.max() / pos_data.min()) >= threshold


def _apply_y_scale(ax: plt.Axes, data: np.ndarray, threshold: int = 3,
                   linthresh: float | None = None) -> None:
    """
    Auto-select y-axis scale based on data characteristics:
      - Bounded [-1, 1]: linear (e.g. cos θ, success rate)
      - Strictly positive, large dynamic range: log
      - Mixed-sign or zero-crossing, large dynamic range: symlog
      - Otherwise: linear

    linthresh: override the symlog linear-region boundary.
      Default (None) uses the median |non-zero| value, which suits line plots.
      Pass the smallest |non-zero| value for bar charts so the linear band is
      tight around zero and small bars are resolved in log space.
    """
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return

    lo, hi = finite.min(), finite.max()

    # Bounded [-1, 1]: always linear (cos θ, correlation, success rate)
    if lo >= -1.0 and hi <= 1.0:
        return

    has_neg  = lo < 0
    has_zero = (finite == 0).any()
    pos      = finite[finite > 0]
    large_range = len(pos) >= 2 and np.log10(pos.max() / pos.min()) >= threshold

    if not large_range:
        return

    if has_neg or has_zero:
        nz = np.abs(finite[finite != 0])
        if linthresh is None:
            linthresh = float(np.median(nz)) if len(nz) else 1.0
        ax.set_yscale("symlog", linthresh=linthresh)
    else:
        ax.set_yscale("log")


def _best_inset_loc(ax: plt.Axes,
                    data_xs: np.ndarray, data_ys: np.ndarray,
                    w: float = 0.42, h: float = 0.34,
                    margin: float = 0.03) -> tuple[float, float]:
    """
    Pick the axes-fraction corner [x0, y0] for an inset of size (w, h) that
    minimises overlap with the plotted data.

    Tests four corners; counts data points (in axes coords) that land inside
    each candidate rectangle; returns the corner with fewest.
    Ties broken by: top-right, top-left, bottom-right, bottom-left.
    """
    candidates = [
        (1 - w - margin, 1 - h - margin),  # top-right
        (margin,         1 - h - margin),  # top-left
        (1 - w - margin, margin),          # bottom-right
        (margin,         margin),          # bottom-left
    ]

    inv_axes    = ax.transAxes.inverted()
    dat_to_disp = ax.transData

    # filter out non-finite before transforming
    mask = np.isfinite(data_xs) & np.isfinite(data_ys)
    if mask.sum() == 0:
        return candidates[0]

    pts_axes = inv_axes.transform(
        dat_to_disp.transform(np.column_stack([data_xs[mask], data_ys[mask]]))
    )

    best_loc, best_count = candidates[0], np.inf
    for x0, y0 in candidates:
        inside = (
            (pts_axes[:, 0] >= x0) & (pts_axes[:, 0] <= x0 + w) &
            (pts_axes[:, 1] >= y0) & (pts_axes[:, 1] <= y0 + h)
        )
        count = inside.sum()
        if count < best_count:
            best_count = count
            best_loc   = (x0, y0)

    return best_loc


def _apply_x_scale(ax: plt.Axes, xs: np.ndarray, data_vals: np.ndarray, labels: list[str]) -> None:
    """Applies either a linear or symlog scale depending on the data spread."""
    if _should_use_log_scale(data_vals):
        # We plot against the actual data values on a symlog scale
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(data_vals.min(), data_vals.max())
        ax.set_xticks(data_vals)
        ax.set_xticklabels(labels)
    else:
        # We plot against categorical indices and label them
        ax.set_xlim(xs[0], xs[-1])
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)



def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def _cat_x(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    gb = df["goal_bonus"].values
    labels = [str(int(g)) if g == int(g) else str(g) for g in gb]
    
    if _should_use_log_scale(gb):
        # Plotting against actual values handles spacing naturally
        return gb, gb, labels
    else:
        # Plotting categorically keeps even spacing
        return np.arange(len(df)), gb, labels


def _sr_twin(ax, xs, sr: np.ndarray) -> None:
    ax2 = ax.twinx()
    ax2.plot(xs, sr, "D--", color=C_SR, lw=1.4, ms=7, label="success rate")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Success rate", color=C_SR, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=C_SR)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(C_SR)
    return ax2


def _describe_lam(df: pd.DataFrame) -> str:
    """Auto-generate a factual subtitle for the λ panel from the data."""
    med = df["lambda_median"]
    frac_above_1 = (med > 1).mean()
    direction = "above" if frac_above_1 > 0.5 else "below"
    return (
        f"Median λ range: {med.min():.2f} – {med.max():.2f}  |  "
        f"{frac_above_1:.0%} of conditions have median λ > 1 (sparse leads)"
    )


def _describe_cos(df: pd.DataFrame) -> str:
    """Auto-generate a factual subtitle for the cos θ panel from the data."""
    med = df["cos_theta_median"]
    frac_pos = (med > 0).mean()
    return (
        f"Median cos θ range: {med.min():.2f} – {med.max():.2f}  |  "
        f"{frac_pos:.0%} of conditions have positive median alignment"
    )


def _describe_qaug(df: pd.DataFrame) -> str:
    aug = df["q_q_aug_mean"]
    rng = aug.max() - aug.min()
    return (
        f"Q_aug range across conditions: {aug.min():.2f} – {aug.max():.2f}  "
        f"(Δ = {rng:.2f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_no_overlap(ax, xs, ys, labels, color, fontsize=8, box_ec=None):
    """
    Annotate a line with value labels, alternating above/below to reduce overlap.
    Uses a minimum pixel-distance check to skip labels that would still collide.
    """
    box_ec = box_ec or color
    offsets = [10, -18]  # points: alternate above / below
    prev_display = None
    min_sep = 30  # pixels — skip label if too close to previous

    for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
        disp = ax.transData.transform((x, y))
        if prev_display is not None:
            sep = np.hypot(*(disp - prev_display))
            if sep < min_sep:
                continue
        dy = offsets[i % 2]
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", fontsize=fontsize,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=box_ec, alpha=0.85, lw=0.5))
        prev_display = disp


def plot_dominance_vs_bonus(df: pd.DataFrame, outdir: Path) -> None:
    xs, gb_vals, xlabels = _cat_x(df)
    sr = df["success_rate"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Critic Dominance Metrics vs Reward Scale",
                 fontweight="bold", fontsize=13, y=1.01)

    # ── Left: λ ──────────────────────────────────────────────────────────────
    ax = axes[0]

    p05 = np.clip(df["lambda_p05"].values, 0, None)
    p95 = df["lambda_p95"].values
    med = df["lambda_median"].values
    men = df["lambda_mean"].values

    # cap ribbon so it doesn't dwarf the median line
    y_ceil = min(p95.max(), med.max() * 6)

    ax.fill_between(xs, p05, np.clip(p95, 0, y_ceil),
                    alpha=0.15, color=C_AUG, label="p05 – p95 band")
    ax.plot(xs, med, "o-", color=C_AUG, lw=2.2, ms=8, zorder=4, label="λ median")
    ax.plot(xs, men, "o:", color=C_AUG, lw=1.1, ms=3, zorder=4, label="λ mean")

    # annotate median only — mean is too close, clutters without adding info
    _annotate_no_overlap(ax, xs, med, [f"{v:.2f}" for v in med], C_AUG, fontsize=8)

    ax.axhline(1.0, color="#555", lw=1.0, ls="--", label="λ = 1  (balance)")
    _apply_y_scale(ax, np.concatenate([p05, med, men, p95]))
    ax.set_ylim(bottom=0)
    _apply_x_scale(ax, xs, gb_vals, xlabels)
    ax.set_xlabel("goal_bonus")
    ax.set_ylabel("λ  =  ‖∇Q_sparse‖ / ‖∇Q_aug‖")
    ax.set_title(_describe_lam(df), fontsize=9)

    ax2 = _sr_twin(ax, xs, sr)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # ── Right: cos θ ──────────────────────────────────────────────────────────
    ax = axes[1]

    med_c = df["cos_theta_median"].values
    men_c = df["cos_theta_mean"].values
    p05_c = df["cos_theta_p05"].values
    p95_c = df["cos_theta_p95"].values

    # lighter band — cos θ spans [-1,1] so the ribbon is visually very large
    ax.fill_between(xs, p05_c, p95_c, alpha=0.10, color=C_SPARSE,
                    label="p05 – p95 band")
    ax.plot(xs, med_c, "s-", color=C_SPARSE, lw=2.2, ms=8, zorder=4,
            label="cos θ median")
    ax.plot(xs, men_c, "s:", color=C_SPARSE, lw=1.1, ms=3, zorder=4,
            label="cos θ mean")

    # annotate median only, alternating side
    _annotate_no_overlap(ax, xs, med_c, [f"{v:.2f}" for v in med_c],
                         C_SPARSE, fontsize=8)

    ax.axhline(0.0, color="#555", lw=1.0, ls="--", label="cos θ = 0  (orthogonal)")
    _apply_y_scale(ax, np.concatenate([p05_c, med_c, men_c, p95_c]))
    ax.set_ylim(-1.05, 1.05)
    _apply_x_scale(ax, xs, gb_vals, xlabels)
    ax.set_xlabel("goal_bonus")
    ax.set_ylabel("cos θ")
    ax.set_title(_describe_cos(df), fontsize=9)

    # inset: data-aware corner selection
    INS_W, INS_H = 0.44, 0.36
    fig.canvas.draw()  # force layout so transData is valid
    ix0, iy0 = _best_inset_loc(ax, xs, med_c, w=INS_W, h=INS_H)
    pad = 0.05
    ins_ylo = max(-1.0, med_c.min() - pad)
    ins_yhi = min( 1.0, med_c.max() + pad)
    axins = ax.inset_axes([ix0, iy0, INS_W, INS_H])
    axins.plot(xs, med_c, "s-", color=C_SPARSE, lw=1.8, ms=5)
    axins.set_ylim(ins_ylo, ins_yhi)
    axins.set_xticks(xs)
    axins.set_xticklabels(xlabels, fontsize=6, rotation=45, ha="right")
    axins.tick_params(axis="y", labelsize=6)
    axins.set_title("Median  (zoomed)", fontsize=7)
    axins.set_facecolor("#f0f0f0")
    for sp in axins.spines.values():
        sp.set_linewidth(0.5)

    ax2 = _sr_twin(ax, xs, sr)
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1, l1, fontsize=8, loc="lower right")

    fig.tight_layout()
    _save(fig, outdir / "fig1_dominance_vs_bonus.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2
# ─────────────────────────────────────────────────────────────────────────────

def plot_q_decomposition(df: pd.DataFrame, outdir: Path) -> None:
    xs, gb_vals, xlabels = _cat_x(df)
    n = len(df)

    is_log = _should_use_log_scale(gb_vals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Q-Function Decomposition vs Reward Scale",
                 fontweight="bold", fontsize=13, y=1.01)

    # ── Left: line plot ───────────────────────────────────────────────────────
    ax = axes[0]

    def _band(col_m, col_s, color, label, marker):
        m = df[col_m].values
        s = df[col_s].values
        ax.fill_between(xs, m - s, m + s, alpha=0.12, color=color)
        ax.plot(xs, m, f"{marker}-", color=color, lw=2.2, ms=8, label=label, zorder=3)

    _band("q_q_aug_mean",    "q_q_aug_std",    C_AUG,    "Q_aug",    "o")
    _band("q_q_sparse_mean", "q_q_sparse_std", C_SPARSE, "Q_sparse", "s")
    _band("q_q_total_mean",  "q_q_total_std",  C_TOTAL,  "Q_total",  "^")

    ax.axhline(0, color="#aaa", lw=0.6, ls=":")

    _apply_x_scale(ax, xs, gb_vals, xlabels)
    ax.set_xlabel("goal_bonus")
    ax.set_ylabel("Mean Q  (over heading sweep)")
    ax.set_title(_describe_qaug(df), fontsize=9)
    ax.legend(fontsize=8)
    _apply_y_scale(ax, np.concatenate([
        df["q_q_aug_mean"].values, df["q_q_sparse_mean"].values, df["q_q_total_mean"].values
    ]))

    # ── Right: grouped bar chart ──────────────────────────────────────────────
    ax = axes[1]

    q_aug_vals   = df["q_q_aug_mean"].values
    q_aug_stds   = df["q_q_aug_std"].values
    q_sp_vals    = df["q_q_sparse_mean"].values
    q_sp_stds    = df["q_q_sparse_std"].values

    if is_log:
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xticks(gb_vals)
        ax.set_xticklabels(xlabels)

        for i, val in enumerate(gb_vals):
            w = (val * 0.35) if val > 0 else 0.35

            ax.bar(val - w / 2, q_aug_vals[i], width=w, color=C_AUG, alpha=0.85,
                   yerr=q_aug_stds[i], capsize=3, error_kw={"lw": 1.1},
                   label="Q_aug" if i == 0 else "")
            ax.bar(val + w / 2, q_sp_vals[i], width=w, color=C_SPARSE, alpha=0.85,
                   yerr=q_sp_stds[i], capsize=3, error_kw={"lw": 1.1},
                   label="Q_sparse" if i == 0 else "")

            # ratio annotation: only where both are non-zero and have the same sign
            # (opposite-sign ratio is misleading); skip near-zero aug to avoid ÷0
            if np.abs(q_aug_vals[i]) > 1e-6 and (q_aug_vals[i] * q_sp_vals[i] > 0):
                r = q_sp_vals[i] / q_aug_vals[i]
                tip_y = q_sp_vals[i] + np.sign(q_sp_vals[i]) * q_sp_stds[i]
                ax.annotate(f"×{r:.1f}", (val + w / 2, tip_y),
                            textcoords="offset points",
                            xytext=(0, 5 * np.sign(tip_y or 1)),
                            fontsize=7, color=C_SPARSE, ha="center")
    else:
        width = 0.35
        ax.bar(xs - width / 2, q_aug_vals, width, color=C_AUG, alpha=0.85, label="Q_aug",
               yerr=q_aug_stds, capsize=4, error_kw={"lw": 1.2})
        ax.bar(xs + width / 2, q_sp_vals,  width, color=C_SPARSE, alpha=0.85, label="Q_sparse",
               yerr=q_sp_stds,  capsize=4, error_kw={"lw": 1.2})
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels)

        for x, aug, sp, sp_std in zip(xs, q_aug_vals, q_sp_vals, q_sp_stds):
            if np.abs(aug) > 1e-6 and (aug * sp > 0):
                r = sp / aug
                tip_y = sp + np.sign(sp) * sp_std
                ax.annotate(f"×{r:.1f}", (x + width / 2, tip_y),
                            textcoords="offset points",
                            xytext=(0, 5 * np.sign(tip_y or 1)),
                            fontsize=7, color=C_SPARSE, ha="center")

    ax.axhline(0, color="#aaa", lw=0.6, ls=":")
    ax.set_xlabel("goal_bonus")
    ax.set_ylabel("Mean Q ± std")

    # linthresh = smallest non-zero |Q| so the linear band is tight around 0
    # and even the smallest bars are resolved in log space
    all_q = np.concatenate([q_aug_vals, q_sp_vals])
    nz_q  = np.abs(all_q[all_q != 0])
    bar_linthresh = float(nz_q.min()) if len(nz_q) else 1.0
    _apply_y_scale(ax, all_q, linthresh=bar_linthresh)

    sp_min, sp_max = q_sp_vals.min(), q_sp_vals.max()
    ax.set_title(
        f"Q_sparse range: {sp_min:.2f} – {sp_max:.2f}  |  "
        f"Q_aug range: {q_aug_vals.min():.2f} – {q_aug_vals.max():.2f}",
        fontsize=9,
    )
    ax.legend(fontsize=8)

    # inset: episode reward — data-aware corner selection
    if "mean_ep_reward" in df.columns:
        INS_W, INS_H = 0.40, 0.30
        fig.canvas.draw()
        ep_ys = df["mean_ep_reward"].values
        ix0, iy0 = _best_inset_loc(ax, xs, ep_ys, w=INS_W, h=INS_H)
        axins = ax.inset_axes([ix0, iy0, INS_W, INS_H])
        std_col = df.get("std_ep_reward", pd.Series(np.zeros(n)))
        axins.errorbar(xs, ep_ys, yerr=std_col,
                       fmt="^-", color="#333", lw=1.5, ms=5, capsize=3)
        axins.set_xticks(xs)
        axins.set_xticklabels(xlabels, fontsize=6, rotation=45, ha="right")
        axins.tick_params(axis="y", labelsize=6)
        axins.set_title("Episode reward", fontsize=7)
        axins.set_facecolor("#f0f0f0")
        axins.grid(color="white", linewidth=0.8)
        for sp in axins.spines.values():
            sp.set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, outdir / "fig2_q_decomposition.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3
# ─────────────────────────────────────────────────────────────────────────────

def plot_heading_sweep(hdf: pd.DataFrame, outdir: Path) -> None:
    if hdf is None or hdf.empty:
        print("No heading CSV — skipping Fig 3.")
        return

    bonus_vals = sorted(hdf["goal_bonus"].unique())
    # Generate colours from viridis if we have many values
    if len(bonus_vals) <= len(GB_COLOURS):
        colours = GB_COLOURS[:len(bonus_vals)]
    else:
        colours = cm.viridis(np.linspace(0.1, 0.9, len(bonus_vals)))
        
    WINDOW = 5

    # Compute y-axis limits from data
    all_lam = hdf["lambda_median"].values
    all_cos = hdf["cos_theta_median"].values
    lam_lo  = max(0.0, np.percentile(all_lam, 2) * 0.85)
    lam_hi  = np.percentile(all_lam, 98) * 1.15
    cos_lo  = np.percentile(all_cos, 2)  - 0.05
    cos_hi  = np.percentile(all_cos, 98) + 0.05

    # Compute headings flatness metric for subtitle
    lam_ranges = []
    for gb in bonus_vals:
        sub = hdf[hdf["goal_bonus"] == gb]["lambda_median"]
        lam_ranges.append(sub.max() - sub.min())
    mean_range = np.mean(lam_ranges)
    mean_lam   = all_lam.mean()
    flatness_pct = mean_range / mean_lam * 100  # range as % of mean

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    fig.suptitle("Dominance Landscape over Heading Sweep",
                 fontweight="bold", fontsize=13)

    for color, gb in zip(colours, bonus_vals):
        sub = hdf[hdf["goal_bonus"] == gb].sort_values("heading_deg")
        hdg = sub["heading_deg"].values
        lbl = f"gb = {int(gb) if gb == int(gb) else gb}"

        lam_s = pd.Series(sub["lambda_median"].values).rolling(
            WINDOW, center=True, min_periods=1).mean().values
        cos_s = pd.Series(sub["cos_theta_median"].values).rolling(
            WINDOW, center=True, min_periods=1).mean().values

        axes[0].plot(hdg, lam_s, color=color, lw=1.8, label=lbl)
        axes[1].plot(hdg, cos_s, color=color, lw=1.8, label=lbl)

    axes[0].set_ylim(lam_lo, lam_hi)
    axes[0].axhline(1.0, color="#555", lw=0.8, ls="--", label="λ = 1")
    axes[0].set_ylabel("λ  (median, smoothed)")
    axes[0].set_title(
        f"Mean per-condition λ range across headings: {mean_range:.2f}  "
        f"({flatness_pct:.1f}% of mean λ = {mean_lam:.2f})",
        fontsize=9,
    )
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _apply_y_scale(axes[0], all_lam)

    axes[1].set_ylim(cos_lo, cos_hi)
    axes[1].axhline(0.0, color="#555", lw=0.8, ls="--", label="cos θ = 0")
    axes[1].set_ylabel("cos θ  (median, smoothed)")
    axes[1].set_xlabel("Heading  (deg)")
    _apply_y_scale(axes[1], all_cos)

    # cos θ subtitle: fraction of (condition, heading) points above 0
    frac_pos_cos = (hdf["cos_theta_median"] > 0).mean()
    axes[1].set_title(
        f"{frac_pos_cos:.0%} of (condition × heading) points have cos θ_median > 0",
        fontsize=9,
    )
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    _save(fig, outdir / "fig3_heading_sweep.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4
# ─────────────────────────────────────────────────────────────────────────────

def plot_dominance_space(df: pd.DataFrame, outdir: Path) -> None:
    if "success_rate" not in df.columns:
        print("No success_rate — skipping Fig 4.")
        return

    x  = df["lambda_median"].values
    y  = df["cos_theta_median"].values
    sr = df["success_rate"].values
    gb = df["goal_bonus"].values

    # λ ≥ 0 always; guard against any floating-point edge case
    x = np.clip(x, 0, None)

    # size proportional to success_rate
    sr_range = sr.max() - sr.min()
    if sr_range < 1e-6:
        sz = np.full(len(sr), 300.0)
    else:
        sz = 150 + 550 * (sr - sr.min()) / sr_range

    gb_log = np.log10(gb + 1)
    norm   = mcolors.Normalize(vmin=gb_log.min(), vmax=gb_log.max())

    # Axis limits: data range + 15% padding
    def _lims(arr, pad=0.15):
        lo, hi = arr.min(), arr.max()
        margin = (hi - lo) * pad if (hi - lo) > 1e-9 else 0.1
        return lo - margin, hi + margin

    xlim = _lims(x)
    ylim = _lims(y)

    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(x, y, c=gb_log, s=sz, cmap="viridis", norm=norm,
                    edgecolors="k", linewidths=0.9, zorder=4, alpha=0.92)

    # annotations: offset direction chosen to avoid overlap
    n = len(df)
    offsets = [(-35, 30), (35, 30), (35, -30), (-35, -30), (0, 45), (0, -45), (-45, 0)]
    
    for i, (xi, yi, g, s) in enumerate(zip(x, y, gb, sr)):
        dx, dy = offsets[i % len(offsets)]
        label = f"gb={int(g) if g == int(g) else g}\nsr={s:.0%}"
        ax.annotate(
            label, (xi, yi),
            textcoords="offset points", xytext=(dx, dy),
            fontsize=8.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#ccc", lw=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#999", lw=0.8),
        )

    ax.axhline(0.0, color="#888", lw=0.8, ls="--", alpha=0.6, label="cos θ = 0")
    ax.axvline(1.0, color="#888", lw=0.8, ls="--", alpha=0.6, label="λ = 1")

    ax.set_xlim(max(0, xlim[0]), xlim[1])  # λ cannot be negative
    ax.set_ylim(ylim)
    ax.set_xlabel("λ_median  =  ‖∇Q_sparse‖ / ‖∇Q_aug‖", fontsize=11)
    ax.set_ylabel("cos θ_median  (gradient alignment)", fontsize=11)

    # computed subtitle
    spearman_lam = df["lambda_median"].corr(df["success_rate"], method="spearman")
    spearman_cos = df["cos_theta_median"].corr(df["success_rate"], method="spearman")
    ax.set_title(
        "Dominance Space — One Point per Reward-Scale Condition\n"
        f"Spearman ρ: λ vs success = {spearman_lam:+.2f}  |  "
        f"cos θ vs success = {spearman_cos:+.2f}\n"
        "Point size ∝ success rate  |  Colour = log₁₀(goal_bonus + 1)",
        fontsize=10, linespacing=1.7,
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log₁₀(goal_bonus + 1)", fontsize=9)
    gb_ticks = np.log10(gb + 1)
    cbar.set_ticks(gb_ticks)
    cbar.set_ticklabels([str(int(g) if g == int(g) else g) for g in gb], fontsize=8)

    # size legend
    sr_examples = np.linspace(sr.min(), sr.max(), 3)
    for sr_val in sr_examples:
        sz_ex = 150 + 550 * (sr_val - sr.min()) / (sr_range + 1e-9)
        ax.scatter([], [], s=sz_ex, c="grey", alpha=0.7,
                   edgecolors="k", lw=0.8, label=f"sr = {sr_val:.0%}")
    ax.legend(title="Success rate (size)", fontsize=8, title_fontsize=8,
              loc="lower right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, outdir / "fig4_dominance_space.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default="experiments/reward_sweep/results/reward_sweep.csv")
    parser.add_argument("--heading-csv", default="experiments/reward_sweep/results/reward_sweep_heading.csv")
    parser.add_argument("--outdir",      default="experiments/reward_sweep/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Scalar CSV: {len(df)} rows")

    hdf = None
    heading_path = Path(args.heading_csv)
    if heading_path.exists():
        hdf = pd.read_csv(heading_path)
        print(f"Heading CSV: {len(hdf)} rows")
    else:
        print(f"Heading CSV not found at {heading_path} — skipping Fig 3.")

    plot_dominance_vs_bonus(df, outdir)
    plot_q_decomposition(df, outdir)
    plot_heading_sweep(hdf, outdir)
    plot_dominance_space(df, outdir)


if __name__ == "__main__":
    main()