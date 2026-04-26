"""
analyse.py  —  DTG Data Analysis
=================================
Analyses collected data to inform DTG (Distance-To-Go) sampler design.

Data contract
-------------
- x, y          : normalised aircraft position  [-1, 1]
- t             : normalised elapsed time        [0, 1]
- total_dist_km : total path distance for the episode (km)
- path_len      : distance already flown (km)
- runway        : runway identifier string

Derived:
- dist_to_go = total_dist_km - path_len  (km, the sampler target)

Both analysis modes are always run and compared side-by-side:

  independent
      Target : dist_to_go
      Features: x, y (or r, theta), runway

  dependent
      Target : dist_to_go
      Features: x, y (or r, theta), t, runway

Sections
--------
1. Data health          — episode counts, step-0 presence, missing values
2. DTG derivation       — validate dist_to_go = total_dist_km - path_len
3. DTG distribution     — per-runway histograms for BOTH modes (side-by-side rows)
4. Spatial coverage     — spawn (x, y) scatter coloured by dist_to_go
5. Spatial conditioning — variance reduction table for both modes
6. Mode comparison      — RMSE comparison + feature importance bar charts
7. Correlation analysis — Pearson & Spearman correlations with dist_to_go
8. Episode lengths      — distribution of episode lengths per runway
9. Temporal patterns    — how dist_to_go evolves over normalised time t
10. Summary             — stats for both modes + actionable verdict

Usage
-----
    python analyse.py rta_data.csv
    python analyse.py rta_data.csv --save-plots ./plots
    python analyse.py rta_data.csv --runway 18C 36L
    python analyse.py rta_data.csv --no-plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import lstsq
from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Mode config
# ─────────────────────────────────────────────────────────────────────────────

class ModeConfig:
    """Bundles all mode-specific labels and column references in one place."""

    PALETTE = {
        "independent": "#4C9BE8",   # blue
        "dependent":   "#E8834C",   # orange
    }

    def __init__(self, mode: str):
        if mode not in ("independent", "dependent"):
            raise ValueError(f"Unknown mode '{mode}'.")
        self.mode = mode

    @property
    def target_col(self) -> str:
        return "dist_to_go"

    @property
    def target_label(self) -> str:
        return "dist_to_go = total_dist_km − path_len  (km)"

    @property
    def target_str(self) -> str:
        base = r"$P(dist\_to\_go \mid x, y"
        if self.mode == "dependent":
            base += r", t"
        return base + r", rwy)$"

    @property
    def features(self) -> list[str]:
        return ["x", "y"] if self.mode == "independent" else ["x", "y", "t"]

    @property
    def sampler_str(self) -> str:
        return f"P(dtg | {', '.join(self.features)}, runway)"

    @property
    def color(self) -> str:
        return self.PALETTE[self.mode]

    def __str__(self) -> str:
        return self.mode


MODES = [ModeConfig("independent"), ModeConfig("dependent")]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    required = {"episode", "step", "x", "y", "t", "runway", "total_dist_km", "path_len"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dist_to_go"] = df["total_dist_km"] - df["path_len"]
    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["theta"] = np.arctan2(df["y"], df["x"])
    return df


def _step0(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["step"] == 0].copy()


def _save_and_show_fig(fig, save_dir: Path | None, name: str):
    if save_dir is None:
        plt.show()
        return
    
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / name, bbox_inches="tight", dpi=150)
    print(f"  saved → {save_dir / name}")


def _section_header(n: int, title: str):
    print(f"\n{'=' * 60}")
    print(f"SECTION {n} — {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Data health
# ─────────────────────────────────────────────────────────────────────────────

def check_data_health(df: pd.DataFrame) -> None:
    _section_header(1, "DATA HEALTH")
    runways = sorted(df["runway"].unique())
    print(f"Total rows       : {len(df):,}")
    print(f"Runways          : {runways}")
    print(f"Episodes         : {df['episode'].nunique()}")
    print(f"Missing values   : {df.isnull().sum().sum()}")

    has_s0 = df[df["step"] == 0]["episode"].nunique()
    total_eps = df["episode"].nunique()
    flag = "  ⚠️  Some episodes missing step-0 — re-collect" if has_s0 < total_eps else "  ✅"
    print(f"Episodes with step-0: {has_s0}/{total_eps}{flag}")

    print("\nEpisodes per runway:")
    for rwy, n in df.drop_duplicates("episode").groupby("runway").size().sort_index().items():
        print(f"  {rwy:>4s}: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — DTG derivation & validation
# ─────────────────────────────────────────────────────────────────────────────

def check_dtg(df: pd.DataFrame) -> None:
    _section_header(2, "DTG VALIDATION")
    print("dist_to_go = total_dist_km − path_len  (km, should be ≥ 0)\n")
    s0 = _step0(df)

    for col, label in [
        ("total_dist_km", "total_dist_km at step 0 (km)"),
        ("path_len",      "path_len      at step 0 (km)"),
        ("dist_to_go",    "dist_to_go    at step 0 (km)"),
    ]:
        print(f"{label}:")
        print(s0[col].describe().round(4).to_string())
        print()

    neg = (df["dist_to_go"] < -1e-3).sum()
    print(f"Negative dist_to_go : {neg}", "  ⚠️" if neg > 0 else "  ✅")

    # Sanity check: at episode end, dist_to_go should approach 0
    last_steps = df.sort_values("step").groupby("episode").last()
    near_zero = (last_steps["dist_to_go"].abs() < 1.0).mean() * 100
    print(f"Episodes where final dist_to_go < 1 km: {near_zero:.1f}%",
          "  ✅" if near_zero > 90 else "  ⚠️  check path_len / total_dist_km columns")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Target distribution at step 0  (BOTH modes, two rows)
# ─────────────────────────────────────────────────────────────────────────────

# Section 3 — all three items (shared x-axis, KDE overlay, annotation box) are implemented below.

def plot_dtg_distribution(
    s0: pd.DataFrame,
    runways: list[str],
    save_dir: Path | None,
    plot: bool = True,
) -> None:
    """
    Histogram + KDE of dist_to_go at spawn (step 0) for each runway.

    A single panel per runway avoids redundancy — both independent and
    dependent modes share the same target column, so one distribution
    plot is sufficient. The KDE overlay makes the shape easier to read
    than a histogram alone and helps spot multimodality (e.g. two
    dominant spawn distances) that would justify a mixture sampler.

    Per-runway descriptive statistics are also printed for the thesis
    appendix.
    """
    _section_header(3, "DTG DISTRIBUTION AT STEP 0  (per runway)")

    if not plot:
        print("Plotting disabled. Skipping...")
        return

    from scipy.stats import gaussian_kde
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    print("dist_to_go summary at step-0 (both modes share this target):\n")
    stats_df = s0.groupby("runway")["dist_to_go"].describe()[
        ["count", "mean", "std", "min", "50%", "max"]
    ]
    print(stats_df.round(2).to_string())

    ncols = min(len(runways), 3)
    nrows = (len(runways) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharey=False, squeeze=False)

    # Shared x-range for comparability across runways
    global_min = s0["dist_to_go"].min()
    global_max = s0["dist_to_go"].max()
    x_grid = np.linspace(global_min, global_max, 300)

    # Storage for global legend handles
    legend_handles = []

    for idx, rwy in enumerate(runways):
        ax = axes[idx // ncols][idx % ncols]
        data = s0[s0["runway"] == rwy]["dist_to_go"].dropna()

        if data.empty:
            ax.set_visible(False)
            continue

        # 1. Histogram
        n, bins, patches = ax.hist(data, bins=25, density=True,
                                   color="#4C9BE8", edgecolor="white", 
                                   linewidth=0.5, alpha=0.55)

        # 2. KDE overlay
        if len(data) > 3:
            kde = gaussian_kde(data, bw_method="scott")
            line, = ax.plot(x_grid, kde(x_grid), color="#1A5FA8", linewidth=2.0)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.12, color="#1A5FA8")

        # 3. Reference lines
        v_med = ax.axvline(data.median(), color="black", linestyle="--", linewidth=1.4)
        v_mean = ax.axvline(data.mean(), color="crimson", linestyle=":", linewidth=1.4)

        # Build Legend Handles (once only)
        if not legend_handles:
            legend_handles = [
                mpatches.Patch(color="#4C9BE8", alpha=0.55, label="Histogram (density)"),
                mlines.Line2D([], [], color="#1A5FA8", lw=2, label="KDE"),
                mlines.Line2D([], [], color="black", ls="--", lw=1.4, label="Median"),
                mlines.Line2D([], [], color="crimson", ls=":", lw=1.4, label="Mean")
            ]

        # 4. Enhanced Annotation box (Replacing individual legends)
        stats_text = (
            f"n: {len(data):,}\n"
            f"μ: {data.mean():.1f} km\n"
            f"med: {data.median():.1f} km\n"
            f"std: {data.std():.1f} km\n"
            f"range: [{data.min():.0f}, {data.max():.0f}]"
        )
        
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

        ax.set_title(f"Runway {rwy}", fontsize=10, fontweight="bold")
        ax.set_xlabel("dtg (km)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Hide unused subplots
    for idx in range(len(runways), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Global Figure Elements
    fig.suptitle(
        "dtg Distribution at Spawn (Step 0)\n"
        r"Target for $P(dtg \mid \mathbf{x})$ and $P(dtg \mid \mathbf{x}, t)$",
        fontsize=12, fontweight="bold", y=0.98
    )

    # Place global legend at the bottom
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, 
               fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.02))

    # Adjust layout to make room for suptitle and global legend
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    _save_and_show_fig(fig, save_dir, "dtg_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Spatial coverage
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_coverage(
    s0: pd.DataFrame,
    runways: list[str],
    save_dir: Path | None = None,
    plot: bool = True,
) -> None:
    """
    Scatter of spawn positions (x, y) coloured by dist_to_go, one panel
    per runway.

    Both independent and dependent modes share the same positional features
    and the same target, so a single scatter per runway is sufficient. The
    colour gradient reveals whether certain spawn regions consistently
    produce short or long dist_to_go values — strong spatial structure
    supports the independent-mode sampler; uniform colour suggests that
    position alone is a poor predictor and 't' may be needed.

    A 2-D hex-bin density inset is added per panel so sparse vs. dense
    spawn zones are immediately visible alongside the colour gradient.

    All subplots share x- and y-axes for direct positional comparability
    across runways. A single centroid annotation table is printed below
    the figure rather than cluttering each panel.
    """
    _section_header(4, "SPAWN POSITION COVERAGE  (coloured by dist_to_go)")

    if not plot:
        print("Plotting disabled. Skipping...")
        return

    n_runways = len(runways)
    ncols = min(n_runways, 3)
    nrows = (n_runways + ncols - 1) // ncols

    vmin = s0["dist_to_go"].min()
    vmax = s0["dist_to_go"].max()

    # sharex/sharey ensures all panels use identical axis ranges
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True,
                             squeeze=False)

    sc_ref = None

    for idx, rwy in enumerate(runways):
        ax = axes[idx // ncols][idx % ncols]
        sub = s0[s0["runway"] == rwy]

        if not sub.empty:
            # Hex-bin density background (no colour bar needed — just texture)
            ax.hexbin(sub["x"], sub["y"], gridsize=18,
                      cmap="Greys", mincnt=1, alpha=0.25, linewidths=0.2,
                      zorder=1)

            sc = ax.scatter(
                sub["x"], sub["y"],
                c=sub["dist_to_go"], cmap="plasma",
                s=18, alpha=0.75, edgecolors="none",
                vmin=vmin, vmax=vmax, zorder=2,
            )
            sc_ref = sc

        ax.set_aspect("equal")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title(f"Runway {rwy}", fontsize=10, fontweight="bold")
        # Only label outer axes to avoid repetition on shared grid
        if idx % ncols == 0:
            ax.set_ylabel("$y$ (normalised)", fontsize=9)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("$x$ (normalised)", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle=":")

    # Hide unused panels
    for idx in range(n_runways, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Single shared colour-bar
    if sc_ref is not None:
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(sc_ref, cax=cax, label="dtg at spawn (km)")

    fig.suptitle(
        "Spatial Spawn Coverage — dtg per (x, y, runway)\n"
        r"Grey hexbins = density  ·  colour = dtg (km)",
        fontsize=12, fontweight="bold",
    )
    _save_and_show_fig(fig, save_dir, "spatial_coverage.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Spatial conditioning value  (BOTH modes)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_incremental_conditioning(
    df_full: pd.DataFrame,
    runways: list[str],
    n_spatial_bins: int = 8,
    n_time_bins: int = 4,
    save_dir: Optional[Path] = None,
    plot: bool = True,
) -> None:
    """
    Measure variance reduction from conditioning on (x, y) alone (independent
    mode) vs. (x, y, t) together (dependent mode).

    IMPORTANT: this function must receive the *full* dataset (all steps), not
    just step-0. At step-0 every episode has t = 0, so all rows fall into the
    same t-bin and the temporal boost is always zero by construction. Using all
    steps gives t genuine variation across [0, 1] so the boost is meaningful.

    The spatial variance reduction is computed on the step-0 slice (spawn
    positions) to match the independent-mode sampler's use case, while the
    temporal boost uses the full trajectory data where t varies.
    """
    _section_header(5, "Incremental Conditioning (BOTH modes)")

    rows = []
    for rwy in runways:
        # ── Spatial reduction (independent mode) ────────────────
        sub_all = df_full[df_full["runway"] == rwy].copy()

        baseline_std = sub_all["dist_to_go"].std()
        if baseline_std <= 0 or np.isnan(baseline_std):
            continue

        sub_all["xb"] = pd.cut(sub_all["x"], bins=n_spatial_bins, labels=False)
        sub_all["yb"] = pd.cut(sub_all["y"], bins=n_spatial_bins, labels=False)
        s_std = sub_all.groupby(["xb", "yb"], observed=True)["dist_to_go"].std().median()
        s_red = (1 - s_std / baseline_std) * 100

        # ── Temporal boost dependent mode) ─────────────────
        baseline_std_all = sub_all["dist_to_go"].std()
        if baseline_std_all <= 0 or np.isnan(baseline_std_all):
            t_boost, total_red = 0.0, s_red
        else:
            sub_all["xb"] = pd.cut(sub_all["x"], bins=n_spatial_bins, labels=False)
            sub_all["yb"] = pd.cut(sub_all["y"], bins=n_spatial_bins, labels=False)
            sub_all["tb"] = pd.cut(sub_all["t"], bins=n_time_bins,    labels=False)

            # Spatial-only baseline on full data (for a fair apples-to-apples
            # comparison before adding t)
            s_std_all = sub_all.groupby(["xb", "yb"], observed=True)["dist_to_go"].std().median()
            s_red_all  = (1 - s_std_all / baseline_std_all) * 100

            st_std    = sub_all.groupby(["xb", "yb", "tb"], observed=True)["dist_to_go"].std().median()
            total_red = (1 - st_std / baseline_std_all) * 100
            t_boost   = total_red - s_red_all

        rows.append({
            "runway":         rwy,
            "spatial_red":    s_red,
            "temporal_boost": t_boost,
            "total_red":      total_red,
        })

    res_df = pd.DataFrame(rows).sort_values("total_red", ascending=False)
    if res_df.empty:
        return
    
    if not plot:
        print("Plotting disabled. Skipping...")
        print(res_df.set_index("runway").round(2).to_string())
        return
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    c_spatial  = "#4A90E2"
    c_temporal = "#FF8C42"

    ax.bar(res_df["runway"], res_df["spatial_red"], color=c_spatial,
           label=r"Spatial reduction  $P(dtg \mid x, y, rwy)$", width=0.7)
    ax.bar(res_df["runway"], res_df["temporal_boost"], bottom=res_df["spatial_red"],
           color=c_temporal, label=r"Temporal boost  $P(dtg \mid x, y, t, rwy)$", width=0.7)

    for i, total in enumerate(res_df["total_red"]):
        ax.text(i, total + 1, f"{total:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=10)

    ax.set_ylabel("Variance Reduction (%)", fontsize=12, fontweight="bold")
    ax.set_title("Incremental Information Gain: Spatial vs. Temporal Conditioning",
                 fontsize=15, pad=20, fontweight="bold")
    ax.set_ylim(min(0, res_df[["spatial_red", "temporal_boost"]].min().min()) - 5, 100)
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.legend(frameon=True, loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=30, ha="right")

    avg_s = res_df["spatial_red"].mean()
    avg_t = res_df["temporal_boost"].mean()
    conclusion = "Non-stationary (use dependent mode)" if avg_t > 10 else "Mostly static (independent mode sufficient)"

    summary_text = (f"AVG Spatial Red: {avg_s:.1f}%\n"
                    f"AVG Temp Boost: {avg_t:.1f}%\n"
                    f"Verdict: {conclusion}")
    plt.text(0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, "spatial_vs_temporal.png")


    print(res_df.set_index("runway").round(2).to_string())
    print(f"\nConclusion: {conclusion}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Mode comparison + Feature importance
# ─────────────────────────────────────────────────────────────────────────────

# Feature importance interpretation notes are rendered inline in _plot_mode_comparison.

def analyse_mode_comparison(df: pd.DataFrame, save_dir: Optional[Path] = None, plot: bool = True) -> None:
    """
    Compare independent vs dependent feature sets in Cartesian and Polar
    coordinates, predicting dist_to_go.

    Feature Importance (Mean Decrease in Impurity):
      'What fraction of total variance reduction is attributed to each feature?'
      Values sum to 1.0 within a model. High 't' importance signals that the
      dependent mode is necessary (relevant once speed varies between episodes).
    """
    _section_header(6, "MODE COMPARISON + FEATURE IMPORTANCE")

    # Contextual note on RMSE magnitude
    dtg_std = df["dist_to_go"].std()
    dtg_range = df["dist_to_go"].max() - df["dist_to_go"].min()
    print(
        f"\n  Dataset dist_to_go: std = {dtg_std:.1f} km, range = {dtg_range:.1f} km\n"
        f"  RMSE values should be interpreted relative to this spread — an RMSE of,\n"
        f"  say, 30 km is expected when dist_to_go spans {dtg_range:.0f} km.  A model that\n"
        f"  always predicts the mean would achieve RMSE = std = {dtg_std:.1f} km.\n"
        f"  To compare models independently of scale, see normalised RMSE (RMSE / std)\n"
        f"  printed below.\n"
    )

    df = df.copy()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    results = {}

    def get_key(mode, coord_sys):
        return f"{mode}_{'polar' if coord_sys == 'polar' else 'cartesian'}"

    for cfg in MODES:
        for coord_sys in ("cartesian", "polar"):
            print(f"\nEvaluating mode: {cfg.mode.upper()}  |  Coordinate system: {coord_sys.title()}")

            mode_key = get_key(cfg.mode, coord_sys)
            base_feats = ["r", "theta"] if coord_sys == "polar" else ["x", "y"]
            feats = base_feats + (["t"] if "t" in cfg.features else [])

            X_tr = train_df[feats].values
            y_tr = train_df["dist_to_go"].values
            X_te = test_df[feats].values
            y_te = test_df["dist_to_go"].values

            # Linear baseline
            coef = lstsq(np.column_stack([X_tr, np.ones(len(X_tr))]), y_tr, rcond=None)[0]
            y_lr = np.column_stack([X_te, np.ones(len(X_te))]) @ coef
            rmse_lr = float(np.sqrt(mean_squared_error(y_te, y_lr)))

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            rmse_rf = float(np.sqrt(mean_squared_error(y_te, rf.predict(X_te))))

            results[mode_key] = {
                "cfg": cfg, "coord_sys": coord_sys,
                "rmse_lr": rmse_lr, "rmse_rf": rmse_rf,
                "importances": rf.feature_importances_, "features": feats,
            }

    print(f"\n{'Mode':<26} {'Features':<14} {'Linear RMSE':>13} {'RF RMSE':>10} {'nRMSE (RF)':>12}")
    print("-" * 78)
    for mode, r in results.items():
        feats_str = "+".join(r["features"])
        nrmse = r["rmse_rf"] / dtg_std
        print(f"  {mode:<24} {feats_str:<14} {r['rmse_lr']:>13.2f} {r['rmse_rf']:>10.2f} {nrmse:>12.3f}")

    def get_rf_improvement(coord_sys):
        ind = results[get_key("independent", coord_sys)]
        dep = results[get_key("dependent",   coord_sys)]
        improvement = (ind["rmse_rf"] - dep["rmse_rf"]) / ind["rmse_rf"] * 100
        print(f"\nAdding 't' changes RF RMSE by: {improvement:+.2f}% ({coord_sys})")
        return improvement

    rf_impr_cart = get_rf_improvement("cartesian")
    rf_impr_pol  = get_rf_improvement("polar")

    if rf_impr_cart > rf_impr_pol:
        print("\nRF RMSE improvement is greater in Cartesian coordinates.")
    elif rf_impr_cart < rf_impr_pol:
        print("\nRF RMSE improvement is greater in Polar coordinates.")
    else:
        print("\nRF RMSE improvement is the same in Cartesian and Polar coordinates.")

    rf_impr = max(rf_impr_cart, rf_impr_pol)

    print("─" * 60)
    print("💡 VERDICT")
    print("─" * 60)
    if rf_impr > 10:
        print(
            "  Use DEPENDENT mode — including 't' reduces RF RMSE by\n"
            f"  {rf_impr:.1f}%. Sampler: P(dist_to_go | x, y, t, runway)."
        )
    elif rf_impr > 3:
        print(
            "  Marginal benefit from 't'. Use DEPENDENT mode if the\n"
            "  sampler may be called mid-episode; otherwise INDEPENDENT\n"
            "  is simpler and nearly as accurate."
        )
    else:
        print(
            "  Use INDEPENDENT mode — 't' adds negligible predictive\n"
            f"  value ({rf_impr:.1f}% RMSE change). Speed is likely constant.\n"
            "  Sampler: P(dist_to_go | x, y, runway).\n"
        )

    if plot:
        _plot_mode_comparison(results, save_dir)

def _plot_mode_comparison(results: dict, save_dir: Optional[Path] = None) -> None:
    fig = plt.figure(figsize=(20, 11))
    # Slightly wider right margin to accommodate the new labels
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.2, 1.1, 1.1], wspace=0.4, hspace=0.4)
    ax_rmse = fig.add_subplot(gs[:, 0])

    plot_order = [
        "independent_cartesian", "dependent_cartesian",
        "independent_polar",     "dependent_polar",
    ]
    fi_coords = [(0, 1), (0, 2), (1, 1), (1, 2)]

    # --- SECTION 1: RMSE PERFORMANCE COMPARISON ---
    n_modes = len(plot_order)
    x_clusters = np.arange(2)
    width = 0.18
    offsets = np.linspace(-(n_modes - 1) * width / 2, (n_modes - 1) * width / 2, n_modes)

    for i, mode_key in enumerate(plot_order):
        r = results[mode_key]
        vals = [r["rmse_lr"], r["rmse_rf"]]
        bars = ax_rmse.bar(
            x_clusters + offsets[i], vals, width,
            label=mode_key.replace("_", " ").title(),
            edgecolor="black", linewidth=0.8,
            color=r["cfg"].color,
            alpha=0.6 if "independent" in mode_key else 0.9,
        )
        for bar in bars:
            ax_rmse.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="black",
                          linewidth=0.5, boxstyle="round"),
            )

    ax_rmse.set_xticks(x_clusters)
    ax_rmse.set_xticklabels(["LINEAR BASELINE", "RANDOM FOREST"], fontsize=11, fontweight="bold")
    ax_rmse.set_ylabel("RMSE — dtg (km)", fontsize=10)
    ax_rmse.set_title("Performance by Estimator Type", fontsize=12, fontweight="bold")
    ax_rmse.legend(fontsize=8, frameon=True, loc='upper right')

    # --- SECTION 2: FEATURE IMPORTANCE & DECISION LOGIC ---
    for idx, mode_key in enumerate(plot_order):
        ax = fig.add_subplot(gs[fi_coords[idx]])
        r = results[mode_key]
        feats, imps = r["features"], r["importances"]
        
        # Calculate context for 't' (is it helpful?)
        is_dependent = "dependent" in mode_key
        base_mode = mode_key.replace("dependent", "independent")
        rmse_improvement = 0.0
        
        if is_dependent and base_mode in results:
            indep_rmse = results[base_mode]["rmse_rf"]
            dep_rmse = r["rmse_rf"]
            # Percentage improvement (Higher is better)
            rmse_improvement = (indep_rmse - dep_rmse) / indep_rmse * 100

        bar_colors = ["#4C9BE8" if f in ("x", "y", "r", "theta") else "#E86B4C"
                      for f in feats]
        bars = ax.barh(np.arange(len(feats)), imps, color=bar_colors,
                       edgecolor="white", height=0.6, zorder=2)

        uniform_val = 1.0 / len(feats)
        ax.axvline(uniform_val, color="red", linestyle="--", linewidth=1.2, alpha=0.6,
                   label=f"Uniform ({uniform_val:.2f})", zorder=3)

        # Basic FI values on bars
        for bar, imp in zip(bars, imps):
            ax.text(imp + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{imp:.3f}", va="center", fontsize=8, fontweight="bold", zorder=4)

        # Per-Row Interpretation Logic
        avg_imp = 1.0 / len(feats)
        for i, (f, imp) in enumerate(zip(feats, imps)):
            strength_ratio = imp / avg_imp
            
            # 1. Determine Signal Strength
            if imp == max(imps): status = "Primary"
            elif strength_ratio > 1.5: status = "Strong"
            elif strength_ratio > 0.8: status = "Moderate"
            else: status = "Weak"

            # 2. Decision logic for feature 't'
            if f == "t":
                # Criteria: Needs to be important AND provide > 5% RMSE boost
                needs_t = strength_ratio > 1.2 and rmse_improvement > 5.0
                verdict = "KEEP" if needs_t else "DROP"
                v_color = "#27AE60" if needs_t else "#C0392B" # Emerald Green / Pomegranate Red
                
                label_text = f"{verdict}: {rmse_improvement:.1f}% Boost"
                ax.text(1.05, i, label_text, va='center', fontsize=7.5, 
                        color='white', fontweight='bold', transform=ax.get_yaxis_transform(),
                        bbox=dict(facecolor=v_color, edgecolor='none', boxstyle='round,pad=0.3'))
            else:
                # Regular label for non-temporal features
                label_text = f"{strength_ratio:.1f}x ({status})"
                ax.text(1.05, i, label_text, va='center', fontsize=7, 
                        color='#555555', style='italic', transform=ax.get_yaxis_transform())

        ax.set_yticks(np.arange(len(feats)))
        ax.set_yticklabels(feats, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1.8) # Room for labels
        ax.set_title(mode_key.replace("_", " ").upper(),
                     fontsize=10, fontweight="bold", color=r["cfg"].color)
        ax.legend(fontsize=7, framealpha=0.8, loc='lower right')
        ax.axhspan(np.argmax(imps) - 0.35, np.argmax(imps) + 0.35,
                   color="gold", alpha=0.1, zorder=1)

    # Note section for clarify
    note_text = (
        r"$\bf{Independent}$: $P(dist\_to\_go \mid x, y, rwy)$" + "\n" +
        r"$\bf{Dependent}$: $P(dist\_to\_go \mid x, y, t, rwy)$" + "\n" +
        r"$\bf{Verdict}$: Based on FI vs. Independent RMSE improvement."
    )
    ax_rmse.text(0.05, 0.05, note_text, transform=ax_rmse.transAxes, fontsize=9,
                 verticalalignment="bottom",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="gray", alpha=0.9))

    fig.suptitle("DTG Analysis: Feature Importance & Temporal Necessity Verdict",
                 fontsize=16, fontweight="bold", y=0.98)
    
    _save_and_show_fig(fig, save_dir, "mode_comparison_with_verdict.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

# Section 7 uses full df for all features (avoids ConstantInputWarning from step-0 only),
# suppresses the warning explicitly, improves visuals, and adds interpretation notes.


def analyse_correlations(df: pd.DataFrame, save_dir: Optional[Path] = None, plot: bool = True) -> None:
    """
    Pearson and Spearman correlations between each feature and dist_to_go,
    **plus** a full feature-feature cross-correlation matrix.

    Data used:
      - Spatial features (x, y, r, theta): full trajectory.
      - Temporal feature (t): full trajectory, because t is constant (= 0)
        at step-0 — Using all steps gives t genuine variation
        across [0, 1] and produces meaningful correlation estimates.

    Feature → target correlations:
      - Pearson detects linear relationships and is sensitive to outliers.
      - Spearman detects any monotonic relationship and is robust to
        non-normality — more appropriate for spatial features like r,
        where the relationship with dist_to_go is often monotonic but
        not strictly linear.
      Reporting both guards against false negatives: a near-zero Pearson
      with a high Spearman signals a non-linear but still useful predictor.
      Per-runway breakdowns surface runway-specific structure that global
      correlations may mask.

    Feature–feature cross-correlations:
      Identifies collinearity among predictors (e.g. r ≈ f(x, y)),
      which is important for sampler design — highly collinear features
      add redundancy without reducing RMSE and may increase sampler
      complexity unnecessarily. High |ρ| between two features (> 0.8)
      suggests one can be dropped.

    How to interpret the results for sampler design:
      - High |feature→target| correlation  → strong predictor; include it.
      - Near-zero correlation for 't'       → independent mode is sufficient.
      - High |feature→feature| correlation  → redundancy; one can be dropped
        to simplify the sampler without loss of accuracy.
    """
    import warnings
    from scipy.stats import ConstantInputWarning

    _section_header(7, "CORRELATION ANALYSIS")

    spatial_feats = ["x", "y", "r", "theta"]
    all_features  = ["x", "y", "r", "theta", "t"]
    target        = "dist_to_go"

    def _safe_corr(a, b):
        """Return (pearson_r, p, spearman_r, p) suppressing ConstantInputWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            pr, pp = pearsonr(a, b)
            sr, sp = spearmanr(a, b)
        return pr, pp, sr, sp

    print("Pearson (linear) and Spearman (monotonic) correlations with dist_to_go")
    print("  Spatial features (x, y, r, theta): computed over all steps\n")
    print("  Temporal feature (t)             : computed over all steps\n")

    print(f"\n  {'Feature':<10} {'Data':<10} {'Pearson r':>10} {'p':>8}  {'Spearman ρ':>11} {'p':>8}  {'sig':>4}")
    print("  " + "-" * 72)

    for feat in all_features:
        data = df.dropna(subset=[feat, target])
        src_label = "all steps"
        pr, pp, sr, sp = _safe_corr(data[feat], data[target])
        sig = "***" if min(pp, sp) < 0.001 else "**" if min(pp, sp) < 0.01 else "*" if min(pp, sp) < 0.05 else "ns"
        print(f"  {feat:<10} {src_label:<10} {pr:>+10.4f} {pp:>8.4f}  {sr:>+11.4f} {sp:>8.4f}  {sig:>4}")

    print("\nPer-runway Spearman correlations (key features):")
    key_feats = ["x", "y", "r", "t"]
    header = f"  {'Runway':<8}" + "".join(f"{f:>10}" for f in key_feats)
    print(header)
    print("  " + "-" * (8 + 10 * len(key_feats)))
    for rwy in sorted(df["runway"].unique()):
        row_str = f"  {rwy:<8}"
        for feat in key_feats:
            sub = df[df["runway"] == rwy].dropna(subset=[feat, target])
            if len(sub) < 3 or sub[feat].nunique() < 2:
                row_str += f"{'NaN':>10}"
            else:
                rho, _ = spearmanr(sub[feat], sub[target])
                row_str += f"{rho:>+10.3f}"
        print(row_str)

    # ── Build correlation matrices ────────────────────────────────────────────
    runways = sorted(df["runway"].unique())

    # Feature → target: per runway
    corr_matrix = np.zeros((len(all_features), len(runways)))
    for j, rwy in enumerate(runways):
        for i, feat in enumerate(all_features):
            sub = df[df["runway"] == rwy].dropna(subset=[feat, target])
            if len(sub) < 3 or sub[feat].nunique() < 2:
                corr_matrix[i, j] = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    rho, _ = spearmanr(sub[feat], sub[target])
                corr_matrix[i, j] = rho

    # Feature–feature: use full df so t has genuine variation
    n_f = len(all_features)
    cross = np.zeros((n_f, n_f))
    data_ff = df.dropna(subset=all_features)
    for i, fi in enumerate(all_features):
        for j, fj in enumerate(all_features):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                rho, _ = spearmanr(data_ff[fi], data_ff[fj])
            cross[i, j] = rho

    # ── Plot: cleaner 2-panel figure ─────────────────────────────────────────
    if plot:
        fig_w = max(14, len(runways) * 1.4 + 8)
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.5),
                                gridspec_kw={"width_ratios": [max(1, len(runways) * 0.6), 1]})

        _plot_corr_heatmap(
            axes[0], corr_matrix, row_labels=all_features, col_labels=runways,
            title="Feature → dtg\nSpearman ρ per runway",
            cbar_label="Spearman ρ",
        )

        print("\nFeature–feature Spearman cross-correlations (full dataset):")
        _plot_corr_heatmap(
            axes[1], cross, row_labels=all_features, col_labels=all_features,
            title="Feature–Feature Cross-Correlation\nSpearman ρ (full dataset)",
            cbar_label="Spearman ρ",
        )

        fig.suptitle("Correlation Analysis — Spearman ρ Heatmaps",
                    fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        _save_and_show_fig(fig, save_dir, "correlation_heatmap.png")

    # Print collinearity warnings
    print(f"\n  {'Feature pair':<20} {'Spearman ρ':>12}  {'flag'}")
    print("  " + "-" * 52)
    for i in range(n_f):
        for j in range(i + 1, n_f):
            rho = cross[i, j]
            flag = "⚠️  collinear — consider dropping one" if abs(rho) > 0.8 else ""
            print(f"  {all_features[i]:<8} ↔ {all_features[j]:<8}  {rho:>+.4f}  {flag}")

    print(
        "\n  Interpretation guide:\n"
        "  • High |feature→target| ρ      → strong predictor; keep in sampler.\n"
        "  • Near-zero ρ for 't'           → independent mode may be sufficient.\n"
        "  • |feature↔feature| ρ > 0.8    → collinearity; one feature is redundant.\n"
        "    e.g. r = √(x²+y²) is always collinear with x and y by construction."
    )

def _plot_corr_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cbar_label: str,
) -> None:
    """Render a tidy annotated heatmap on *ax*."""
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = matrix[i, j]
            if np.isnan(v):
                label, txt_col = "NaN", "#888888"
            else:
                label  = f"{v:+.2f}"
                txt_col = "white" if abs(v) > 0.5 else "#1a1a1a"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=8, color=txt_col, fontweight="bold")

    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 — Episode length distribution
# ─────────────────────────────────────────────────────────────────────────────

# CV = std / mean.  A CV of 0.5 means the standard deviation is 50 % of the mean
# episode length — a commonly used rule-of-thumb threshold above which length variance
# is large enough to meaningfully shift the t ∈ [0,1] distribution and potentially
# confound the dependent-mode sampler if it is trained without stratification by length.

def analyse_episode_lengths(df: pd.DataFrame, save_dir: Optional[Path] = None, plot: bool = True) -> None:
    """
    Distribution of episode lengths (number of steps) per runway.

    Why this matters for the thesis:
      Highly variable episode lengths affect sampler generalisation in two ways:
        1. A sampler trained predominantly on short episodes may produce poor
           dist_to_go estimates when queried mid-way through a long episode,
           because the joint distribution P(dtg | x, y, t, rwy) shifts with
           episode length.
        2. If runways differ substantially in mean length, the normalised time
           t ∈ [0, 1] maps to different absolute distances per runway — this
           cross-runway mismatch is a key motivation for runway-conditioning.

    The coefficient of variation (CV = std/mean) is the primary flag:
      CV > 0.5  → high length variance — consider stratified data collection
                  or episode-length conditioning in the sampler.
      CV ≤ 0.5  → acceptably homogeneous — normalised time t is reliable.

    Outlier episodes (length > Q3 + 1.5·IQR) are flagged by runway as they
    may reflect data collection artefacts (e.g. stuck agents, infinite loops).
    """
    _section_header(8, "EPISODE LENGTH DISTRIBUTION")

    ep_lengths = df.groupby("episode")["step"].max().rename("max_step")
    ep_meta    = df.groupby("episode")["runway"].first()
    ep_df      = pd.concat([ep_lengths, ep_meta], axis=1)

    # ── Printed stats ──────────────────────────────────────────────────────────
    print("Global episode length stats (steps):")
    desc = ep_df["max_step"].describe(percentiles=[0.25, 0.5, 0.75])
    print(desc.round(1).to_string())
    print()

    print("Per-runway episode length stats (count / mean / std / median / CV):")
    rwy_stats = ep_df.groupby("runway")["max_step"].agg(
        count="count", mean="mean", std="std", median="median"
    )
    rwy_stats["CV"] = rwy_stats["std"] / rwy_stats["mean"]
    print(rwy_stats.round(2).to_string())
    print()

    runways = sorted(df["runway"].unique())

    # ── IQR outlier detection per runway ──────────────────────────────────────
    print("Outlier detection (> Q3 + 1.5·IQR per runway):")
    for rwy in runways:
        data = ep_df[ep_df["runway"] == rwy]["max_step"]
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr   = q3 - q1
        thresh = q3 + 1.5 * iqr
        n_out  = (data > thresh).sum()
        cv     = data.std() / data.mean() if data.mean() > 0 else 0
        cv_flag = "  ⚠️  high variance (CV > 0.5) — consider stratified sampling" if cv > 0.5 else "  ✅"
        out_flag = f"  ⚠️  {n_out} outlier episode(s) (>{thresh:.0f} steps)" if n_out > 0 else "  ✅  no outliers"
        print(f"  {rwy:>4s}: mean={data.mean():.0f}  std={data.std():.0f}  CV={cv:.2f}{cv_flag}")
        print(f"         {out_flag}")

    if not plot:
        return

    # ── Figure: violin + strip chart ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 3.5 * len(runways)), 5),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Violin per runway
    ax = axes[0]
    palette = plt.get_cmap("tab10")
    violin_data = [ep_df[ep_df["runway"] == rwy]["max_step"].values for rwy in runways]
    parts = ax.violinplot(violin_data, positions=range(len(runways)),
                          showmedians=True, showextrema=True, widths=0.65)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(palette(i % 10))
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    # Strip (individual episodes) overlaid
    rng = np.random.default_rng(42)
    for i, rwy in enumerate(runways):
        data = ep_df[ep_df["runway"] == rwy]["max_step"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(i + jitter, data, s=10, alpha=0.4, color=palette(i % 10),
                   edgecolors="none", zorder=3)

    ax.set_xticks(range(len(runways)))
    ax.set_xticklabels([f"{r}" for r in runways], fontsize=9)
    ax.set_xlabel("Runway", fontsize=10)
    ax.set_ylabel("Episode length (steps)", fontsize=10)
    ax.set_title("Episode Length Distribution per Runway", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Legend explaining violin components
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    legend_handles_violin = [
        mpatches.Patch(facecolor="steelblue", alpha=0.55, label="Density (violin body)"),
        mlines.Line2D([], [], color="black", lw=2, label="Median"),
        mlines.Line2D([], [], color="gray", lw=1, label="Min / Max whiskers"),
        plt.scatter([], [], s=10, color="gray", alpha=0.5, label="Individual episodes"),
    ]
    ax.legend(handles=legend_handles_violin, fontsize=8, frameon=True,
              loc="upper right", title="Legend", title_fontsize=8)

    # CV bar chart
    ax2 = axes[1]
    cvs = [
        ep_df[ep_df["runway"] == rwy]["max_step"].std() /
        ep_df[ep_df["runway"] == rwy]["max_step"].mean()
        for rwy in runways
    ]
    bar_colors = ["#E84C4C" if cv > 0.5 else "#4C9BE8" for cv in cvs]
    ax2.barh(range(len(runways)), cvs, color=bar_colors, edgecolor="white", height=0.6)
    ax2.axvline(0.5, color="crimson", linestyle="--", linewidth=1.3,
                label="CV = 0.5 threshold")
    for i, cv in enumerate(cvs):
        ax2.text(cv + 0.01, i, f"{cv:.2f}", va="center", fontsize=9, fontweight="bold")
    ax2.set_yticks(range(len(runways)))
    ax2.set_yticklabels([f"{r}" for r in runways], fontsize=9)
    ax2.set_xlabel("Coefficient of Variation (CV)", fontsize=10)
    ax2.set_ylabel("Runway", fontsize=10)
    ax2.set_title("Episode Length CV\n(red bar → high variance)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, frameon=True)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.set_xlim(0, max(cvs) * 1.3 + 0.2)

    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, "episode_lengths.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 9 — Temporal patterns in dist_to_go
# ─────────────────────────────────────────────────────────────────────────────

def analyse_temporal_patterns(df: pd.DataFrame, save_dir: Optional[Path] = None, plot: bool = True) -> None:
    """
    How dist_to_go evolves over normalised time t ∈ [0, 1].

    In the idealised case, dist_to_go decreases linearly with t (constant
    speed). Deviations — curves, bumps, or per-runway divergence — signal
    variable speed or non-trivial routing, which is the main justification
    for including t as a feature in the dependent mode sampler. If the
    relationship is linear and consistent across runways, t adds little
    beyond what position (x, y) already encodes.

    How to interpret the temporal patterns:
      - Strong linear decrease (R² ≈ 1) across all runways → speed is
        approximately constant; t encodes the same information as position
        and the independent mode is likely sufficient.
      - Deviations from linearity (low R², large residuals in View 4) →
        variable speed or non-trivial routing; t is informative beyond
        position and the dependent mode should be preferred.
      - Diverging per-runway curves in View 3 → runway-specific speed
        profiles; runway conditioning is important for both modes.
      - Shrinking spread over time in View 5 (heteroscedasticity) →
        the sampler's uncertainty is higher early in the episode; the
        dependent mode can exploit t to reduce this.

    Five views are produced:
      1. Mean ± std band + spaghetti sample (per runway) — reveals both
         average shape and episode-to-episode variability.
      2. R² and RMSE of linear fit (per runway) — quantifies how much of
         the variance is explained by a linear t model.
      3. Overlay of per-runway mean curves — cross-runway speed heterogeneity.
      4. Non-linearity residuals (global) — bar chart of mean dtg − linear fit;
         large bars justify a non-parametric sampler.
      5. Conditional distribution: violin plots of dist_to_go in coarse t bins
         — shows whether the spread changes over time (heteroscedasticity).

    The x-axis upper limit is set to the maximum observed t in the data,
    not a hard-coded 1.0, so truncated episodes don't cause empty plot regions.
    """
    import matplotlib.lines as mlines
    import matplotlib.patheffects as mpeffects
    from scipy.stats import linregress
    
    _section_header(9, "TEMPORAL PATTERNS IN dist_to_go")

    if not plot:
        print("Plotting disabled — skipping temporal pattern analysis.")
        return

    runways  = sorted(df["runway"].unique())
    t_max    = df["t"].max()          # dynamic upper limit
    n_bins   = 20
    t_edges  = np.linspace(0, t_max, n_bins + 1)
    t_mids   = (t_edges[:-1] + t_edges[1:]) / 2

    print(f"  Normalised time range observed: [0, {t_max:.4f}]\n"
          f"  (x-axes clipped to t_max = {t_max:.4f} — not hard-coded to 1.0)\n")

    # ── View 1 + 2: per-runway mean ± std bands + spaghetti + R² ─────────────
    ncols = min(len(runways), 3)
    nrows = (len(runways) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.5 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    rng = np.random.default_rng(0)

    for idx, rwy in enumerate(runways):
        ax  = axes[idx // ncols][idx % ncols]
        sub = df[df["runway"] == rwy].copy()
        sub["t_bin"] = pd.cut(sub["t"], bins=t_edges, labels=False, include_lowest=True)
        grouped = sub.groupby("t_bin", observed=True)["dist_to_go"]
        means = grouped.mean().reindex(range(n_bins))
        stds  = grouped.std().reindex(range(n_bins))

        # Spaghetti: random sample of up to 40 individual episode trajectories
        ep_ids = sub["episode"].unique()
        sample_eps = rng.choice(ep_ids, size=min(40, len(ep_ids)), replace=False)
        for ep in sample_eps:
            ep_data = sub[sub["episode"] == ep].sort_values("t")
            ax.plot(ep_data["t"], ep_data["dist_to_go"],
                    color="#4C9BE8", alpha=0.08, linewidth=0.7, zorder=1)

        # Mean ± std band
        ax.fill_between(t_mids, means - stds, means + stds,
                        alpha=0.30, color="#4C9BE8", zorder=2)
        ax.plot(t_mids, means, color="#1A5FA8", lw=2.2, zorder=3)

        # Linear fit + R² annotation
        valid = ~means.isna()
        r_sq, rmse_lin = np.nan, np.nan
        if valid.sum() > 1:
            coef = np.polyfit(t_mids[valid], means[valid], 1)
            y_fit = np.polyval(coef, t_mids)
            ax.plot(t_mids, y_fit,
                    color="crimson", lw=1.4, linestyle="--", alpha=0.85, zorder=4)

            # R² on the binned means
            ss_res = np.nansum((means[valid] - y_fit[valid]) ** 2)
            ss_tot = np.nansum((means[valid] - np.nanmean(means[valid])) ** 2)
            r_sq   = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse_lin = np.sqrt(np.nanmean((means[valid] - y_fit[valid]) ** 2))

        ann = f"$R^2$={r_sq:.3f}\nRMSE={rmse_lin:.2f} km" if not np.isnan(r_sq) else ""
        ax.text(0.03, 0.05, ann, transform=ax.transAxes, fontsize=7.5,
                va="bottom", bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="white", alpha=0.8, edgecolor="#aaa"))

        ax.set_xlim(0, t_max)
        ax.set_title(f"Runway {rwy}", fontweight="bold", fontsize=10)
        if idx % ncols == 0:
            ax.set_ylabel("dtg (km)", fontsize=9)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Normalised time $t$", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    # Global legend (replaces per-subplot legends to reduce clutter)
    global_legend_handles = [
        mlines.Line2D([], [], color="#4C9BE8", alpha=0.3, lw=6, label="±1 std band"),
        mlines.Line2D([], [], color="#4C9BE8", alpha=0.3, lw=1, label="Individual episodes"),
        mlines.Line2D([], [], color="#1A5FA8", lw=2.2, label="Mean dtg"),
        mlines.Line2D([], [], color="crimson", lw=1.4, linestyle="--", label="Linear fit"),
    ]
    fig.legend(handles=global_legend_handles, loc="lower center", ncol=4,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.0))

    for idx in range(len(runways), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("dtg vs Normalised Time $t$ — per Runway",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    _save_and_show_fig(fig, save_dir, "temporal_dtg_per_runway.png")

    # ── View 3: overlay of per-runway mean curves (improved) ──────────────────
    fig, axes3 = plt.subplots(1, 2, figsize=(14, 5),
                              gridspec_kw={"width_ratios": [1.6, 1]})
    cmap = plt.get_cmap("tab10")

    runway_means = {}
    for i, rwy in enumerate(runways):
        sub = df[df["runway"] == rwy].copy()
        sub["t_bin"] = pd.cut(sub["t"], bins=t_edges, labels=False, include_lowest=True)
        means = sub.groupby("t_bin", observed=True)["dist_to_go"].mean().reindex(range(n_bins))
        runway_means[rwy] = means
        axes3[0].plot(t_mids, means, lw=2, color=cmap(i % 10), label=f"Runway {rwy}")

    axes3[0].set_xlim(0, t_max)
    axes3[0].set_xlabel("Normalised time $t$", fontsize=11)
    axes3[0].set_ylabel("Mean dtg (km)", fontsize=11)
    axes3[0].set_title("Mean dtg Trajectory — Runway Comparison\n"
                       "(diverging curves → runway-specific speed profiles → favour dependent mode)",
                       fontsize=10, fontweight="bold")
    axes3[0].legend(fontsize=9, frameon=True)
    axes3[0].grid(True, linestyle="--", alpha=0.5)

    # ── Right panel: per-runway speed-proxy (slope + R² of linear fit) ──────────
    #
    # What is the "slope"?
    #   We fit  mean_dtg(t) ≈ slope × t + intercept  per runway.
    #   Because dtg decreases as the aircraft flies, slope is NEGATIVE.
    #   A steeper (more-negative) slope means dtg drops faster per unit of
    #   normalised time — i.e. the aircraft covers its remaining distance at a
    #   higher effective rate within that episode.
    #
    # What does R² tell us?
    #   R² measures how well a straight line explains the mean dtg trajectory.
    #   R² ≈ 1 → speed is roughly constant (linear decrease).
    #   R² << 1 → speed varies (non-linear trajectory) — 't' is informative
    #             beyond position and the DEPENDENT sampler is preferred.
    #
    # How to read the chart for sampler design:
    #   • Bars similar in length  → runways have homogeneous speed profiles;
    #     runway-conditioning on slope adds little.
    #   • Bars very different     → runway-specific speed profiles; important
    #     to condition the sampler on runway.
    #   • Low R² (faint bar)     → non-linear trajectory; include 't' as a
    #     feature (dependent mode).

    slope_data = []   # (runway, slope, intercept, r_sq, colour)

    for i, rwy in enumerate(runways):
        means = runway_means[rwy]
        valid = ~means.isna()
        
        if valid.sum() > 1:
            # Use scipy.stats.linregress for a cleaner, more robust linear fit
            res = linregress(t_mids[valid], means[valid])
            r_sq = res.rvalue ** 2 if not np.isnan(res.rvalue) else np.nan
            slope_data.append((rwy, float(res.slope), float(res.intercept), r_sq, cmap(i % 10)))
        else:
            slope_data.append((rwy, np.nan, np.nan, np.nan, "#cccccc"))

    valid_slopes = [s for _, s, *_ in slope_data if not np.isnan(s)]
    mean_slope   = float(np.mean(valid_slopes)) if valid_slopes else 0.0
    slope_std    = float(np.std(valid_slopes))  if len(valid_slopes) > 1 else 0.0

    min_val = min(valid_slopes) if valid_slopes else -10

    # Calculate dynamic padding for labels based on data range
    if valid_slopes:
        slope_range = abs(max(valid_slopes) - min_val)
        label_pad = slope_range * 0.05 if slope_range > 0 else 0.5
    else:
        label_pad = 0.5

    ax_sp = axes3[1]
    bar_h = 0.55
    y_pos = np.arange(len(slope_data))

    # Modernize plot by pushing grid behind bars and removing top/right spines
    ax_sp.set_axisbelow(True)
    ax_sp.grid(axis="x", linestyle="--", alpha=0.4, color="#cccccc")
    ax_sp.spines['top'].set_visible(False)
    ax_sp.spines['right'].set_visible(False)

    r2_column_x = abs(min_val) * 0.02 if valid_slopes else 5

    for i, (rwy, slope, intercept, r_sq, colour) in enumerate(slope_data):
        if np.isnan(slope):
            ax_sp.barh(i, 0, height=bar_h, color="#cccccc", edgecolor="white")
            ax_sp.text(0.02, i, "n/a", va="center", fontsize=8, color="#888")
            continue

        alpha = float(np.clip(r_sq, 0.3, 1.0)) if not np.isnan(r_sq) else 0.4
        is_outlier = abs(slope - mean_slope) > slope_std
        edge_col   = "#C0392B" if is_outlier else "white"
        edge_lw    = 2.0       if is_outlier else 0.8

        # Draw the bar
        ax_sp.barh(i, slope, height=bar_h,
                color=colour, alpha=alpha,
                edgecolor=edge_col, linewidth=edge_lw)

        # --- Speed Value: Centered inside the bar ---
        ax_sp.text(slope / 2, i,
                f"{slope:.1f} km/t",
                va="center", ha="center", 
                fontsize=8, fontweight="bold",
                color="white",
                path_effects=[mpeffects.withStroke(linewidth=1.5, foreground="black", alpha=0.4)])

        # --- R² Value: Placed in a neat column to the right of the zero line ---
        # We use a gray color so it doesn't compete with the primary 'speed' data
        ax_sp.text(r2_column_x, i,
                f"$R^2$ = {r_sq:.2f}",
                va="center", ha="left", 
                fontsize=8, fontweight="medium",
                color="#555555")

    # Fleet-mean reference line
    ax_sp.axvline(mean_slope, color="#555555", linestyle="--", lw=1.5,
                zorder=5, label=f"Fleet mean ({mean_slope:.1f} km/t)")
    # Zero line
    ax_sp.axvline(0, color="black", lw=1.2, zorder=4)

    ax_sp.set_yticks(y_pos)
    ax_sp.set_yticklabels([d[0] for d in slope_data], fontsize=9, fontweight="bold")
    ax_sp.invert_yaxis()

    # Dynamic x-limit: leave enough room for the labels based on our dynamic padding
    if valid_slopes:
        ax_sp.set_xlim(min_val * 1.15, abs(min_val) * 0.25)

    ax_sp.set_xlabel("Linear slope (km per unit normalised time t)", fontsize=9, labelpad=8)
    ax_sp.set_title(
        "Speed Proxy per Runway",
        fontsize=11, fontweight="bold", pad=12,
    )

    # Subtitle annotation box
    explanation = (
        "Bar length  = how fast dtg drops per unit t\n"
        "Bar alpha   = R² (opaque → linear; faint → non-linear)\n"
        "Red edge    = deviates > 1σ from fleet mean\n"
        "Dashed line = fleet mean slope"
    )
    # Placed slightly safer to avoid cutting off at bottom edge of figure
    ax_sp.text(0.0, -0.25, explanation,
           transform=ax_sp.transAxes, fontsize=7.5, va="top",
           color="#444444", style="italic",
           bbox=dict(facecolor="#f9f9f9", edgecolor="#e0e0e0",
                     boxstyle="round,pad=0.4", alpha=0.95))

    ax_sp.legend(fontsize=8, frameon=True, loc="lower right", edgecolor="#cccccc")

    fig.suptitle(
        "Mean dtg Trajectory & Speed Proxy — Runway Comparison",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # Using constrained_layout (or a slightly looser tight_layout) helps prevent text clipping
    plt.tight_layout(rect=(0, 0.12, 1, 0.95))
    _save_and_show_fig(fig, save_dir, "temporal_dtg_overlay.png")

    # ── View 4: non-linearity residuals (global) — improved ───────────────────
    df2 = df.copy()
    df2["t_bin"] = pd.cut(df2["t"], bins=t_edges, labels=False, include_lowest=True)
    global_means = df2.groupby("t_bin", observed=True)["dist_to_go"].mean().reindex(range(n_bins))
    valid   = ~global_means.isna()
    coef    = np.polyfit(t_mids[valid], global_means[valid], 1)
    y_fit   = np.polyval(coef, t_mids)
    residuals = global_means - y_fit

    fig, axes4 = plt.subplots(1, 2, figsize=(13, 4.5),
                               gridspec_kw={"width_ratios": [1.6, 1]})

    # Left: residual bar chart
    bar_w = (t_mids[1] - t_mids[0]) * 0.85
    axes4[0].bar(t_mids, residuals, width=bar_w,
                 color=np.where(residuals >= 0, "#4C9BE8", "#E8834C"),
                 edgecolor="white", linewidth=0.4, label="Residual")
    axes4[0].plot(t_mids, global_means, color="#1A1A1A", lw=1.5,
                  linestyle="-", label="Mean dtg", zorder=3)
    axes4[0].plot(t_mids, y_fit, color="crimson", lw=1.5,
                  linestyle="--", label="Linear fit", zorder=4)
    axes4[0].axhline(0, color="black", lw=1)
    axes4[0].set_xlim(0, t_max)
    axes4[0].set_xlabel("Normalised time $t$", fontsize=11)
    axes4[0].set_ylabel("km", fontsize=11)
    axes4[0].set_title("Non-linearity Residuals: Mean dtg - Linear Fit\n"
                       "(coloured bars = deviation; black = actual mean; red dashes = linear fit)",
                       fontsize=10, fontweight="bold")
    axes4[0].legend(fontsize=8, frameon=True)
    axes4[0].grid(axis="y", linestyle="--", alpha=0.5)

    max_resid = residuals.abs().max()
    verdict = (
        "Non-linear — 't' likely informative beyond position."
        if max_resid > 1.0 else
        "Near-linear — 't' may add little over position alone."
    )
    print(f"\n  Max non-linearity residual: {max_resid:.2f} km")
    print(f"  {verdict}")
    axes4[0].text(0.02, 0.97, verdict, transform=axes4[0].transAxes, fontsize=9,
                  va="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # Right panel: cumulative absolute residual — shows where non-linearity accumulates
    cum_resid = residuals.abs().cumsum()
    axes4[1].fill_between(t_mids, cum_resid, alpha=0.35, color="#E8834C")
    axes4[1].plot(t_mids, cum_resid, color="#C04000", lw=2)
    axes4[1].set_xlim(0, t_max)
    axes4[1].set_xlabel("Normalised time $t$", fontsize=11)
    axes4[1].set_ylabel("Cumulative |residual| (km)", fontsize=10)
    axes4[1].set_title("Cumulative Absolute Residual\n(steep rise = non-linearity concentrated there)",
                       fontsize=10, fontweight="bold")
    axes4[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, "temporal_nonlinearity.png")

    # ── View 5: conditional distribution (violin per coarse t bin) — improved ─
    n_coarse     = 5
    coarse_edges = np.linspace(0, t_max, n_coarse + 1)
    coarse_labels = [f"[{coarse_edges[k]:.2f},{coarse_edges[k+1]:.2f})"
                     for k in range(n_coarse)]
    df3 = df.copy()
    df3["t_coarse"] = pd.cut(df3["t"], bins=coarse_edges,
                             labels=coarse_labels, include_lowest=True)

    ncols_v5 = min(len(runways), 3)
    nrows_v5 = (len(runways) + ncols_v5 - 1) // ncols_v5
    fig, axes5 = plt.subplots(nrows_v5, ncols_v5,
                               figsize=(4.5 * ncols_v5, 4.5 * nrows_v5),
                               sharey=True, squeeze=False)

    for idx, rwy in enumerate(runways):
        ax = axes5[idx // ncols_v5][idx % ncols_v5]
        sub = df3[df3["runway"] == rwy]
        violin_data = [
            sub[sub["t_coarse"] == lbl]["dist_to_go"].dropna().values
            for lbl in coarse_labels
        ]
        positions_used = [i for i, d in enumerate(violin_data) if len(d) > 1]
        violin_data_clean = [violin_data[i] for i in positions_used]

        if violin_data_clean:
            parts = ax.violinplot(violin_data_clean, positions=positions_used,
                                  showmedians=True, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor("#4C9BE8")
                pc.set_alpha(0.5)
            parts["cmedians"].set_color("crimson")
            parts["cmedians"].set_linewidth(1.8)

            # Overlay IQR boxes for better quantile readability
            for pos, data in zip(positions_used, violin_data_clean):
                q1, q3 = np.percentile(data, [25, 75])
                ax.vlines(pos, q1, q3, color="#1A5FA8", linewidth=4, alpha=0.5, zorder=3)

        ax.set_xticks(range(len(coarse_labels)))
        ax.set_xticklabels(coarse_labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"Runway {rwy}", fontweight="bold", fontsize=10)
        if idx % ncols_v5 == 0:
            ax.set_ylabel("dtg (km)", fontsize=9)
        ax.set_xlabel("t bin", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    for idx in range(len(runways), nrows_v5 * ncols_v5):
        axes5[idx // ncols_v5][idx % ncols_v5].set_visible(False)

    fig.suptitle(
        "dtg Conditional Distribution in Coarse t Bins\n"
        "(violin = density · crimson line = median · blue bar = IQR)\n",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, "temporal_conditional_dist.png")


# ─────────────────────────────────────────────────────────────────────────────
# Section 10 — Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, s0: pd.DataFrame) -> None:
    _section_header(10, "SUMMARY")

    n_eps  = df["episode"].nunique()
    n_rwys = df["runway"].nunique()
    has_s0 = s0["episode"].nunique() == n_eps

    print(f"  Dataset         : {n_eps} episodes across {n_rwys} runways")
    print(f"  Step-0 complete : {'✅' if has_s0 else '⚠️  missing step-0 rows'}")
    print()

    target = s0["dist_to_go"]
    print(f"  Target    : dist_to_go = total_dist_km − path_len  (km)")
    print(f"  Range     : [{target.min():.1f}, {target.max():.1f}] km   std={target.std():.1f} km")
    print()

    for cfg in MODES:
        print(f"  ── {cfg.mode.upper()} ──")
        print(f"    Sampler : {cfg.sampler_str}")
    print()

    print("  See Section 6 VERDICT for the recommended mode.")
    print("  See Section 7 for feature correlation strengths.")
    print("  See Section 9 for temporal non-linearity — if the residual")
    print("  plot shows large deviations, 't' is worth including.")
    print()
    print("  NOTE: current data has constant speed — independent and dependent")
    print("  modes will show similar RMSE. Re-run once variable-speed data is")
    print("  available to get a meaningful verdict on whether 't' helps.")


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyse(
    df,
    runway_filter=None,
    no_plots=False,
    xy_bins=10,
    t_bins=10,
    save_dir=None,
):
    if runway_filter:
        unknown = set(runway_filter) - set(df["runway"].unique())
        if unknown:
            print(f"⚠️  Unknown runways requested: {unknown}")
        df = df[df["runway"].isin(runway_filter)]

    check_data_health(df)
    df = _enrich(df)
    s0 = _step0(df)

    if s0.empty:
        print("\n⚠️  No step-0 rows found. Re-collect data with the updated collect.py.")
        return None

    runways = sorted(df["runway"].unique())

    check_dtg(df)

    plotting = not no_plots

    plot_dtg_distribution(s0, runways, save_dir, plotting)
    plot_spatial_coverage(s0, runways, save_dir, plotting)

    analyse_incremental_conditioning(
        df, runways, n_spatial_bins=xy_bins, n_time_bins=t_bins, save_dir=save_dir, plot=plotting
    )

    analyse_mode_comparison(df, save_dir, plotting)

    analyse_correlations(df, save_dir=save_dir, plot=plotting)
    analyse_episode_lengths(df, save_dir=save_dir, plot=plotting)
    analyse_temporal_patterns(df, save_dir=save_dir, plot=plotting)

    print_summary(df, s0)
    return df, s0

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _get_args():
    p = argparse.ArgumentParser(description="Analyse collected DTG data (both modes always run).")
    p.add_argument("data",         type=str, help="Path to .csv or .parquet file.")
    p.add_argument("--save-plots", type=str, default=None, help="Directory to save plots.")
    p.add_argument("--runway",     type=str, nargs="+", default=None,
                   help="Subset of runways to analyse.")
    p.add_argument("--xy-bins",    type=int, default=8,
                   help="Grid bins for spatial conditioning (default: 8).")
    p.add_argument("--t-bins",     type=int, default=4,
                   help="Grid bins for temporal conditioning (default: 4).")
    p.add_argument("--no-plots",   action="store_true", help="Skip all plots (stats only).")
    return p.parse_args()


def run_analyse_cli(experiment_cls):
    args = _get_args()
    save_dir = Path(args.save_plots) if args.save_plots else None

    print(f"Loading : {args.data}")
    print("Running both modes: independent + dependent")

    df_raw = _load(args.data)
    analyse(
        df=df_raw,
        runway_filter=args.runway,
        no_plots=args.no_plots,
        xy_bins=args.xy_bins,
        t_bins=args.t_bins,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    run_analyse_cli(None)