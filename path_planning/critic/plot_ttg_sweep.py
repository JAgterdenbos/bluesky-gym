"""
plot_ttg_sweep.py
-----------------
Four figures answering RQ 2.3: how does shifting from spatial to TBALP
alter the critic's internal prioritisation?

Figures
-------
  fig1_ttg_dominance_comparison.png  — λ and cos θ vs heading, spatial vs TBALP
                                       (aggregated over TTG for TBALP, single
                                        condition for spatial)
  fig2_ttg_heatmap.png               — λ and cos θ as (TTG × heading) heatmaps
                                       for the TBALP agent only
  fig3_ttg_profile.png               — λ and cos θ vs TTG (averaged over heading)
                                       for TBALP, with spatial baseline band
  fig4_dominance_space_agents.png    — scatter in (λ_median, cos θ_median) space,
                                       one point per (agent_type × TTG) condition

Usage
-----
    python plot_ttg_sweep.py \
        [--grid-csv  experiments/ttg_sweep/results/ttg_sweep_grid.csv] \
        [--scalar-csv experiments/ttg_sweep/results/ttg_sweep_scalar.csv] \
        [--outdir    experiments/ttg_sweep/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.ndimage import uniform_filter1d

# ── style (matches reward sweep plots) ───────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "white",
    "axes.facecolor":     "#f8f9fa",
    "axes.grid":          True,
    "grid.color":         "#e5e5e5",
    "grid.linewidth":     0.6,
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

C_SPATIAL = "#2166ac"
C_TBALP   = "#d6604d"
C_AUG     = "#2166ac"
C_SPARSE  = "#d6604d"
WINDOW    = 5


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def _smooth(arr: np.ndarray, w: int = WINDOW) -> np.ndarray:
    return uniform_filter1d(arr, size=w, mode="nearest")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — heading sweep comparison (spatial vs TBALP, TTG-aggregated)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heading_comparison(gdf: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    fig.suptitle(
        "Dominance Landscape over Heading — Spatial vs TBALP Agent",
        fontweight="bold", fontsize=13,
    )

    for agent, color, ls in [("spatial", C_SPATIAL, "-"), ("tbalp", C_TBALP, "--")]:
        sub = gdf[gdf["agent_type"] == agent]
        if sub.empty:
            continue

        # Aggregate over TTG for TBALP; spatial has only one TTG value anyway
        agg = (sub.groupby("heading_deg")
                  .agg(lam_med=("lambda_median", "median"),
                       cos_med=("cos_theta_median", "median"))
                  .reset_index()
                  .sort_values("heading_deg"))

        hdg   = agg["heading_deg"].values
        lam_s = _smooth(agg["lam_med"].values)
        cos_s = _smooth(agg["cos_med"].values)
        label = agent.upper()

        axes[0].plot(hdg, lam_s, color=color, lw=2.2, ls=ls, label=label)
        axes[1].plot(hdg, cos_s, color=color, lw=2.2, ls=ls, label=label)

        # Shade std band (across TTG values for TBALP)
        lam_std = sub.groupby("heading_deg")["lambda_median"].std().reindex(agg["heading_deg"]).fillna(0).values
        cos_std = sub.groupby("heading_deg")["cos_theta_median"].std().reindex(agg["heading_deg"]).fillna(0).values
        axes[0].fill_between(hdg, lam_s - lam_std, lam_s + lam_std, alpha=0.12, color=color)
        axes[1].fill_between(hdg, cos_s - cos_std, cos_s + cos_std, alpha=0.12, color=color)

    all_lam = gdf["lambda_median"].values
    axes[0].set_ylim(max(0, np.percentile(all_lam, 2) * 0.85),
                     np.percentile(all_lam, 98) * 1.15)
    axes[0].axhline(1.0, color="#555", lw=0.8, ls=":", label="λ = 1")
    axes[0].set_ylabel("λ  (median, smoothed)")
    axes[0].legend(fontsize=9)

    all_cos = gdf["cos_theta_median"].values
    axes[1].set_ylim(np.percentile(all_cos, 2) - 0.05,
                     np.percentile(all_cos, 98) + 0.05)
    axes[1].axhline(0.0, color="#555", lw=0.8, ls=":", label="cos θ = 0")
    axes[1].set_ylabel("cos θ  (median, smoothed)")
    axes[1].set_xlabel("Heading  (deg)")
    axes[1].legend(fontsize=9)

    frac_tbalp_above = (
        gdf[gdf["agent_type"] == "tbalp"]["lambda_median"] >
        gdf[gdf["agent_type"] == "spatial"]["lambda_median"].median()
    ).mean()
    axes[0].set_title(
        f"TBALP λ exceeds spatial median in {frac_tbalp_above:.0%} of (heading × TTG) points",
        fontsize=9,
    )

    fig.tight_layout()
    _save(fig, outdir / "fig1_ttg_dominance_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — (TTG × heading) heatmaps for TBALP
# ─────────────────────────────────────────────────────────────────────────────

def plot_ttg_heatmap(gdf: pd.DataFrame, outdir: Path) -> None:
    tbalp = gdf[gdf["agent_type"] == "tbalp"]
    if tbalp.empty:
        print("No TBALP data — skipping Fig 2.")
        return

    ttg_vals = np.sort(tbalp["ttg_norm"].unique())
    hdg_vals = np.sort(tbalp["heading_deg"].unique())

    def _pivot(col: str) -> np.ndarray:
        return (tbalp.pivot_table(index="ttg_norm", columns="heading_deg",
                                  values=col, aggfunc="median")
                     .reindex(index=ttg_vals, columns=hdg_vals)
                     .values)

    lam_grid = _pivot("lambda_median")
    cos_grid = _pivot("cos_theta_median")

    ttg_s = ttg_vals * 21600  # normalised → seconds (MAX_TIME = 6 h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "TBALP Agent: Dominance Heatmap over (TTG × Heading)",
        fontweight="bold", fontsize=13, y=1.01,
    )

    im0 = axes[0].imshow(
        lam_grid, aspect="auto", origin="lower",
        extent=[hdg_vals[0], hdg_vals[-1], ttg_s[0], ttg_s[-1]],
        cmap="RdBu_r", vmin=0,
    )
    axes[0].set_xlabel("Heading  (deg)")
    axes[0].set_ylabel("TTG  (s)")
    axes[0].set_title("λ  =  ‖∇Q_sparse‖ / ‖∇Q_aug‖", fontsize=10)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="λ median")

    vmax_cos = max(abs(cos_grid.min()), abs(cos_grid.max()))
    im1 = axes[1].imshow(
        cos_grid, aspect="auto", origin="lower",
        extent=[hdg_vals[0], hdg_vals[-1], ttg_s[0], ttg_s[-1]],
        cmap="RdBu", vmin=-vmax_cos, vmax=vmax_cos,
    )
    axes[1].set_xlabel("Heading  (deg)")
    axes[1].set_ylabel("TTG  (s)")
    axes[1].set_title("cos θ  (gradient alignment)", fontsize=10)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="cos θ median")

    fig.tight_layout()
    _save(fig, outdir / "fig2_ttg_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — λ and cos θ vs TTG (heading-averaged), with spatial baseline
# ─────────────────────────────────────────────────────────────────────────────

def plot_ttg_profile(gdf: pd.DataFrame, outdir: Path) -> None:
    tbalp   = gdf[gdf["agent_type"] == "tbalp"]
    spatial = gdf[gdf["agent_type"] == "spatial"]

    if tbalp.empty:
        print("No TBALP data — skipping Fig 3.")
        return

    ttg_agg = (tbalp.groupby("ttg_norm")
                     .agg(
                         lam_med=("lambda_median",    "median"),
                         lam_std=("lambda_median",    "std"),
                         cos_med=("cos_theta_median", "median"),
                         cos_std=("cos_theta_median", "std"),
                     )
                     .reset_index()
                     .sort_values("ttg_norm"))

    ttg_s   = ttg_agg["ttg_norm"].values * 21600
    lam_med = ttg_agg["lam_med"].values
    lam_std = ttg_agg["lam_std"].values
    cos_med = ttg_agg["cos_med"].values
    cos_std = ttg_agg["cos_std"].values

    # Spatial baseline (single scalar)
    sp_lam = spatial["lambda_median"].median()    if not spatial.empty else np.nan
    sp_cos = spatial["cos_theta_median"].median() if not spatial.empty else np.nan
    sp_lam_std = spatial["lambda_median"].std()    if not spatial.empty else 0.0
    sp_cos_std = spatial["cos_theta_median"].std() if not spatial.empty else 0.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        "Dominance vs Time-to-Go — TBALP vs Spatial Baseline",
        fontweight="bold", fontsize=13,
    )

    # λ panel
    axes[0].fill_between(ttg_s, lam_med - lam_std, lam_med + lam_std,
                          alpha=0.18, color=C_TBALP)
    axes[0].plot(ttg_s, lam_med, "o-", color=C_TBALP, lw=2.2, ms=7, label="TBALP")
    if not np.isnan(sp_lam):
        axes[0].axhline(sp_lam, color=C_SPATIAL, lw=1.8, ls="--", label="Spatial median")
        axes[0].axhspan(sp_lam - sp_lam_std, sp_lam + sp_lam_std,
                        alpha=0.10, color=C_SPATIAL)
    axes[0].axhline(1.0, color="#555", lw=0.8, ls=":", label="λ = 1")
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("λ  median  (over heading)")
    axes[0].legend(fontsize=9)

    # cos θ panel
    axes[1].fill_between(ttg_s, cos_med - cos_std, cos_med + cos_std,
                          alpha=0.18, color=C_TBALP)
    axes[1].plot(ttg_s, cos_med, "s-", color=C_TBALP, lw=2.2, ms=7, label="TBALP")
    if not np.isnan(sp_cos):
        axes[1].axhline(sp_cos, color=C_SPATIAL, lw=1.8, ls="--", label="Spatial median")
        axes[1].axhspan(sp_cos - sp_cos_std, sp_cos + sp_cos_std,
                        alpha=0.10, color=C_SPATIAL)
    axes[1].axhline(0.0, color="#555", lw=0.8, ls=":", label="cos θ = 0")
    axes[1].set_ylabel("cos θ  median  (over heading)")
    axes[1].set_xlabel("TTG  (s)  —  negative = late")
    axes[1].legend(fontsize=9)

    # Computed subtitle on TTG panel
    if len(ttg_s) > 1:
        lam_range = lam_med.max() - lam_med.min()
        axes[0].set_title(
            f"λ range over TTG: {lam_med.min():.2f} – {lam_med.max():.2f}  "
            f"(Δ = {lam_range:.2f})",
            fontsize=9,
        )

    fig.tight_layout()
    _save(fig, outdir / "fig3_ttg_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — dominance space scatter, one point per (agent × TTG)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dominance_space(gdf: pd.DataFrame, outdir: Path) -> None:
    # One point per (agent_type, ttg_norm): median over headings
    pts = (gdf.groupby(["agent_type", "ttg_norm"])
               .agg(lam=("lambda_median",    "median"),
                    cos=("cos_theta_median", "median"),
                    sr= ("success_rate",     "first"))
               .reset_index())

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        "Dominance Space — Spatial vs TBALP  (one point per agent × TTG condition)",
        fontweight="bold", fontsize=12,
    )

    for agent, color, marker in [("spatial", C_SPATIAL, "o"), ("tbalp", C_TBALP, "s")]:
        sub = pts[pts["agent_type"] == agent]
        if sub.empty:
            continue

        x   = np.clip(sub["lam"].values, 0, None)
        y   = sub["cos"].values
        ttg = sub["ttg_norm"].values * 21600

        # size proportional to TTG magnitude (absolute urgency)
        sz = 200 + 400 * (np.abs(ttg) / (np.abs(ttg).max() + 1e-9))

        sc = ax.scatter(x, y, c=ttg, s=sz, cmap="coolwarm",
                        marker=marker, edgecolors="k", linewidths=0.8,
                        zorder=4, alpha=0.88, label=agent.upper(),
                        vmin=-max(abs(ttg)), vmax=max(abs(ttg)))

        for xi, yi, ti in zip(x, y, ttg):
            ax.annotate(
                f"{ti:+.0f}s", (xi, yi),
                textcoords="offset points", xytext=(0, 10),
                fontsize=7.5, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.6, alpha=0.85),
            )

    ax.axhline(0.0, color="#888", lw=0.8, ls="--", alpha=0.6, label="cos θ = 0")
    ax.axvline(1.0, color="#888", lw=0.8, ls="--", alpha=0.6, label="λ = 1")
    ax.set_xlim(left=0)
    ax.set_xlabel("λ_median  =  ‖∇Q_sparse‖ / ‖∇Q_aug‖", fontsize=11)
    ax.set_ylabel("cos θ_median  (gradient alignment)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")

    # Spearman TTG vs λ for TBALP
    tbalp_pts = pts[pts["agent_type"] == "tbalp"]
    if len(tbalp_pts) > 2:
        rho_lam = tbalp_pts["ttg_norm"].corr(tbalp_pts["lam"], method="spearman")
        rho_cos = tbalp_pts["ttg_norm"].corr(tbalp_pts["cos"], method="spearman")
        ax.set_title(
            f"TBALP Spearman ρ: TTG vs λ = {rho_lam:+.2f}  |  TTG vs cos θ = {rho_cos:+.2f}\n"
            "Point size ∝ |TTG|  |  Colour = TTG (s)  |  ○ = Spatial  □ = TBALP",
            fontsize=9, linespacing=1.6,
        )

    fig.tight_layout()
    _save(fig, outdir / "fig4_dominance_space_agents.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-csv",   default="experiments/ttg_sweep/results/ttg_sweep_grid.csv")
    parser.add_argument("--scalar-csv", default="experiments/ttg_sweep/results/ttg_sweep_scalar.csv")
    parser.add_argument("--outdir",     default="experiments/ttg_sweep/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gdf = pd.read_csv(args.grid_csv)
    print(f"Grid CSV: {len(gdf)} rows, agents: {gdf['agent_type'].unique().tolist()}")

    plot_heading_comparison(gdf, outdir)
    plot_ttg_heatmap(gdf, outdir)
    plot_ttg_profile(gdf, outdir)
    plot_dominance_space(gdf, outdir)


if __name__ == "__main__":
    main()
