"""
analyse.py  —  RTA Data Analysis
=================================
Analyses collected RTA data to inform sampler design.

Data contract (from collect.py)
--------------------------------
- x, y  : normalised aircraft position  [-1, 1]
- t     : normalised elapsed time        [0, 1]
- rta   : normalised t at episode end    [0, 1]  (backfilled from final obs)

Both analysis modes are always run and compared side-by-side:

  independent
      Target : rta_remaining = rta - t
      Sampler: P(rta_remaining | x, y, runway)
      → Drop t from sampler; used only at reset where t ≈ constant

  dependent
      Target : rta
      Sampler: P(rta | x, y, t, runway)
      → Keep t in sampler; useful if sampler is called mid-episode

Sections
--------
1. Data health          — episode counts, step-0 presence, missing values
2. RTA remaining        — derive and validate rta_remaining = rta - t
3. RTA distribution     — per-runway histograms for BOTH modes (side-by-side rows)
4. Spatial coverage     — spawn (x, y) scatter for BOTH modes (side-by-side rows)
5. Spatial conditioning — variance reduction table for both modes
6. Mode comparison      — RMSE comparison + feature importance bar charts with explanation
7. Summary              — stats for both modes + actionable verdict

Usage
-----
    python analyse.py rta_data.csv
    python analyse.py rta_data.csv --save-plots ./plots
    python analyse.py rta_data.csv --runway 18C 36L
    python analyse.py rta_data.csv --no-plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import lstsq

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ─────────────────────────────────────────────────────────────────────────────
# Mode config  (used internally — no longer a CLI argument)
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
        return "rta_remaining" if self.mode == "independent" else "rta"

    @property
    def target_label(self) -> str:
        return "rta_remaining = rta - t" if self.mode == "independent" else "rta  (= t_final)"
    
    @property
    def target_str(self) -> str:
        return r"$P(rta\_remaining \mid x, y, rwy)$" if self.mode == "independent" else r"$P(rta \mid x, y, t, rwy)$"

    @property
    def features(self) -> list[str]:
        return ["x", "y"] if self.mode == "independent" else ["x", "y", "t"]

    @property
    def sampler_str(self) -> str:
        return f"P({self.target_col} | {', '.join(self.features)}, runway)"

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
    required = {"episode", "step", "x", "y", "t", "runway", "rta"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rta_remaining"] = df["rta"] - df["t"]
    df["r"] = np.sqrt(df["x"]**2 + df["y"]**2)
    df["theta"] = np.arctan2(df["y"], df["x"])
    return df


def _step0(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["step"] == 0].copy()


def _savefig(fig, save_dir: Path | None, name: str):
    if save_dir is not None:
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

    has_s0    = df[df["step"] == 0]["episode"].nunique()
    total_eps = df["episode"].nunique()
    flag = "  ⚠️  Some episodes missing step-0 — re-collect" if has_s0 < total_eps else "  ✅"
    print(f"Episodes with step-0: {has_s0}/{total_eps}{flag}")

    print("\nEpisodes per runway:")
    for rwy, n in df.drop_duplicates("episode").groupby("runway").size().sort_index().items():
        print(f"  {rwy:>4s}: {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — RTA remaining derivation & validation
# ─────────────────────────────────────────────────────────────────────────────

def check_rta_remaining(df: pd.DataFrame) -> None:
    _section_header(2, "RTA REMAINING VALIDATION")
    print("rta_remaining = rta − t  (normalised, range should be [0, 1])\n")
    s0 = _step0(df)

    for col, label in [("rta", "rta (t_final) at step 0"),
                        ("t",   "t (t_start)   at step 0"),
                        ("rta_remaining", "rta_remaining at step 0")]:
        print(f"{label}:")
        print(s0[col].describe().round(4).to_string())
        print()

    neg  = (df["rta_remaining"] < -1e-6).sum()
    over = (df["rta_remaining"] > 1.0 + 1e-6).sum()
    print(f"Negative rta_remaining : {neg}",  "  ⚠️" if neg  > 0 else "  ✅")
    print(f"rta_remaining > 1.0    : {over}",  "  ⚠️" if over > 0 else "  ✅")

    rta_per_ep   = df.groupby("episode")["rta"].nunique()
    all_constant = (rta_per_ep == 1).all()
    print(f"rta constant per episode: {'✅' if all_constant else '⚠️  rta varies within episodes'}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Target distribution at step 0  (BOTH modes, two rows)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rta_distribution(
    s0: pd.DataFrame,
    runways: list[str],
    save_dir: Path | None,
) -> None:
    _section_header(3, "TARGET DISTRIBUTION AT STEP 0  [BOTH MODES]")

    for cfg in MODES:
        print(f"\n{cfg.mode.upper()} — target: {cfg.target_label}")
        stats = s0.groupby("runway")[cfg.target_col].describe()[
            ["count", "mean", "std", "min", "50%", "max"]
        ]
        print(stats.round(4).to_string())

    # Pack 2 runways per column -> 4 rows total
    ncols = max(1, (len(runways) + 1) // 2)
    nrows = 4
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(6.0, 3.5 * ncols), 10),
        sharex="col", 
        sharey="row",
        squeeze=False
    )

    for rwy_idx, rwy in enumerate(runways):
        col = rwy_idx // 2
        half = rwy_idx % 2  # 0 for top half (rows 0-1), 1 for bottom half (rows 2-3)

        for mode_idx, cfg in enumerate(MODES):
            row = (half * 2) + mode_idx
            ax = axes[row, col]
            
            data = s0[s0["runway"] == rwy][cfg.target_col]
            
            if not data.empty:
                ax.hist(data, bins=20, color=cfg.color, edgecolor="white",
                        linewidth=0.5, alpha=0.85)
                ax.axvline(data.median(), color="black", linestyle="--", linewidth=1.2,
                           label=f"med={data.median():.3f}")
                ax.axvline(data.mean(),   color="crimson", linestyle=":",  linewidth=1.2,
                           label=f"μ={data.mean():.3f}")
                ax.legend(fontsize=6)

            # Titles go above the first row of each runway block
            if mode_idx == 0:
                ax.set_title(f"Runway {rwy}", fontsize=9, fontweight="bold")

            # Y-labels only for the first column
            if col == 0:
                ax.set_ylabel(f"{cfg.target_str}\n\ncount", fontsize=8)

            # X-labels only for the logical bottom of the current column
            is_last_in_col = (row == 3) or (row == 1 and rwy_idx == len(runways) - 1)
            if is_last_in_col:
                ax.set_xlabel(f"{cfg.target_col} [-]", fontsize=8)

    # Clean up empty axes if the number of runways is odd
    if len(runways) % 2 != 0:
        axes[2, ncols - 1].set_visible(False)
        axes[3, ncols - 1].set_visible(False)
        # Turn tick labels back on for the new bottom axis of that column
        axes[1, ncols - 1].tick_params(labelbottom=True)

    # Update suptitle to reflect new layout
    fig.suptitle("Target Distribution at Step 0", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_dir, "rta_distribution_both.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Spatial coverage  (BOTH modes, two rows)
# ─────────────────────────────────────────────────────────────────────────────
def plot_spatial_coverage(
    s0: pd.DataFrame,
    runways: list[str],
    save_dir: Path | None = None,
) -> None:
    _section_header(4, "SPAWN POSITION COVERAGE")

    targets = [
        {"col": "rta_remaining", "label": "RTA Remaining"},
        {"col": "rta", "label": "RTA"}
    ]
    
    n_runways = len(runways)
    ncols = (n_runways + 1) // 2
    nrows = 4 

    fig = plt.figure(figsize=(max(8, 4 * ncols), 12))
    gs = gridspec.GridSpec(nrows, ncols + 1, width_ratios=[1]*ncols + [0.05], wspace=0.1, hspace=0.15)

    all_vals = pd.concat([s0["rta"], s0["rta_remaining"]])
    vmin, vmax = all_vals.min(), all_vals.max()

    master_ax = None  # This will be our anchor for sharing
    sc = None

    for rwy_idx, rwy in enumerate(runways):
        col_idx = rwy_idx // 2
        half = rwy_idx % 2 
        sub = s0[s0["runway"] == rwy]
        
        for t_idx, target in enumerate(targets):
            row_idx = (half * 2) + t_idx
            
            # Use sharex and sharey with the master_ax if it exists
            ax = fig.add_subplot(gs[row_idx, col_idx], sharex=master_ax, sharey=master_ax)
            
            if master_ax is None:
                master_ax = ax # The very first subplot becomes the master

            if not sub.empty:
                sc = ax.scatter(
                    sub["x"], sub["y"],
                    c=sub[target["col"]], cmap="plasma",
                    s=15, alpha=0.6, edgecolors="none",
                    vmin=vmin, vmax=vmax
                )
            
            ax.set_aspect("equal")
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            
            # --- CLEANUP INTERIOR TICKS ---
            # Only show Y-axis labels on the leftmost column
            if col_idx > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(f"{target['label']}\n\n$y$ [-]", fontsize=9)

            # Only show X-axis labels on the logical "bottom" plots
            is_bottom = (row_idx == 3) or (rwy_idx == n_runways - 1 and t_idx == 1)
            if not is_bottom:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("$x$ [-]", fontsize=9)

            if t_idx == 0:
                ax.set_title(f"Runway {rwy}", fontsize=10, fontweight="bold")

    if sc is not None:
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(sc, cax=cax, label="RTA / RTA Remaining [-]")

    fig.suptitle("Spatial Distribution: Shared Coordinate System", fontsize=14, fontweight="bold", y=0.96)

    # Clean up empty slots
    if n_runways % 2 != 0:
        for r in range(2, 4):
            ax_dummy = fig.add_subplot(gs[r, ncols-1])
            ax_dummy.axis('off')

    if save_dir:
        _savefig(fig, save_dir, "spatial_coverage_shared.png")
    
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Spatial conditioning value  (BOTH modes)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_incremental_conditioning(
    s0: pd.DataFrame,
    runways: list[str],
    n_spatial_bins: int = 8,
    n_time_bins: int = 4,
    save_dir: Path | None = None,
) -> None:
    _section_header(5, "Incremental Conditioning (BOTH modes)")

    t_ms   = "rta_remaining"
    t_temp = "rta"
    
    rows = []
    for rwy in runways:
        sub = s0[s0["runway"] == rwy].copy()
        if len(sub) < 10:
            continue
        
        # Separate baselines for each target
        m_std_ms   = sub[t_ms].std()    # baseline for rta_remaining
        m_std_temp = sub[t_temp].std()  # baseline for rta
        
        if m_std_ms <= 0 or np.isnan(m_std_ms):
            continue
        if m_std_temp <= 0 or np.isnan(m_std_temp):
            continue

        sub["xb"] = pd.cut(sub["x"], bins=n_spatial_bins, labels=False)
        sub["yb"] = pd.cut(sub["y"], bins=n_spatial_bins, labels=False)
        sub["tb"] = pd.cut(sub["t"],  bins=n_time_bins,   labels=False)

        # 1. Spatial reduction — P(rta_remaining | x, y, runway)
        s_std = sub.groupby(["xb", "yb"], observed=True)[t_ms].std().median()
        s_red = (1 - s_std / m_std_ms) * 100

        # 2. Temporal reduction — P(rta | x, y, t, runway), using its own baseline
        st_std    = sub.groupby(["xb", "yb", "tb"], observed=True)[t_temp].std().median()
        total_red = (1 - st_std / m_std_temp) * 100

        # t_boost is the delta the temporal bin adds, within the rta target's own scale
        t_boost = total_red - s_red

        rows.append({
            "runway":        rwy,
            "spatial_red":   s_red,
            "temporal_boost": t_boost,
            "total_red":     total_red,
        })

    res_df = pd.DataFrame(rows).sort_values("total_red", ascending=False)
    if res_df.empty: return

    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-whitegrid') # Clean, modern look
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Professional colors
    c_spatial = "#4A90E2" # Steel Blue
    c_temporal = "#FF8C42" # Coral Orange

    bars_s = ax.bar(res_df["runway"], res_df["spatial_red"], color=c_spatial, 
                    label=r"Spatial Reduction ($P(rta\_rem. \mid x, y, rwy)$)", width=0.7)
    bars_t = ax.bar(res_df["runway"], res_df["temporal_boost"], bottom=res_df["spatial_red"], 
                    color=c_temporal, label=r"Temporal Boost ($P(rta \mid x, y, t, rwy)$)", width=0.7)

    # Add labels on top of bars
    for i, total in enumerate(res_df["total_red"]):
        ax.text(i, total + 1, f"{total:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Styling
    ax.set_ylabel("Variance Reduction (%)", fontsize=12, fontweight='bold')
    ax.set_title("Incremental Information Gain: Spatial vs. Temporal Conditioning", 
                 fontsize=15, pad=20, fontweight='bold')
    ax.set_ylim(min(0, res_df["temporal_boost"].min() + res_df["spatial_red"].min()) - 5, 100)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    
    # Legend & Grid
    ax.legend(frameon=True, loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(rotation=30, ha='right')

    # Summary Text Box
    avg_s = res_df["spatial_red"].mean()
    avg_t = res_df["temporal_boost"].mean()
    conclusion = "Non-stationary" if avg_t > 10 else "Mostly Static"
    
    summary_text = (f"AVG Spatial Red: {avg_s:.1f}%\n"
                    f"AVG Temp Boost: {avg_t:.1f}%\n"
                    f"Verdict: {conclusion}")
    
    plt.text(0.02, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    plt.tight_layout()
    _savefig(fig, save_dir, "spatial_vs_temporal.png")
    plt.show()

    # Final Print
    print(res_df.set_index("runway").round(2).to_string())
    print(f"\nConclusion: {conclusion}")
# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Mode comparison + Feature Importance charts
# ─────────────────────────────────────────────────────────────────────────────

def analyse_mode_comparison(df: pd.DataFrame, save_dir: Path | None) -> None:
    """
    Evaluates model performance across different feature sets (Independent vs Dependent)
    and coordinate systems (Cartesian vs Polar).
    
    Feature Importance (Mean Decrease in Impurity) measures:
    'What fraction of the total variance reduction in the target is explained 
    by splits on each feature across all trees?'
      • Values sum to 1.0 within a model.
      • High 't' importance suggests the dependent mode is necessary.
      • Values are relative; do not compare raw magnitudes across different models.
    """
    _section_header(6, "MODE COMPARISON + FEATURE IMPORTANCE")

    # 1. Feature Engineering: Add Polar Coordinates
    # Assuming origin (0,0) is the relevant reference point (e.g., runway threshold)
    df = df.copy()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    results = {}
    
    # Define the comparison groups
    # 'MODES' should be accessible or passed in; here we adapt the loop to handle 
    # both the Mode (Indep/Dep) and the Coordinate System (XY vs R-Theta)
    def get_key(mode, coord_sys):
        suffix = "_polar" if coord_sys == 'polar' else "_cartesian"
        return f"{mode}{suffix}"

    for cfg in MODES:
        for coord_sys in ['cartesian', 'polar']:
            mode_key = get_key(cfg.mode, coord_sys)
            
            # Swap x/y for r/theta if in polar mode
            base_feats = ['r', 'theta'] if coord_sys == 'polar' else ['x', 'y']
            # If the mode is 'dependent', include 't'
            feats = base_feats + (['t'] if 't' in cfg.features else [])
            
            target_col = cfg.target_col
            X_tr, y_tr = train_df[feats].values, train_df[target_col].values
            X_te, y_te = test_df[feats].values,  test_df[target_col].values

            # Linear baseline
            coef = lstsq(np.column_stack([X_tr, np.ones(len(X_tr))]), y_tr)[0]
            y_lr = np.column_stack([X_te, np.ones(len(X_te))]) @ coef
            rmse_lr = np.sqrt(mean_squared_error(y_te, y_lr))

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            rmse_rf = np.sqrt(mean_squared_error(y_te, rf.predict(X_te)))

            results[mode_key] = {
                "cfg": cfg, "coord_sys": coord_sys, "rmse_lr": rmse_lr, "rmse_rf": rmse_rf,
                "importances": rf.feature_importances_, "features": feats
            }

    # ── Table & Verdict Logic (Simplified for brevity) ───────────────────────
    print(f"\n{'Mode':<14} {'Features':<12} {'Linear RMSE':>13} {'RF RMSE':>10}")
    print("-" * 52)
    for mode, r in results.items():
        feats_str = "+".join(r["features"])
        print(f"  {mode:<12} {feats_str:<12} {r['rmse_lr']:>13.4f} {r['rmse_rf']:>10.4f}")


    def get_rf_improvement(coord_sys):
        ind, dep = results[get_key("independent", coord_sys)], results[get_key("dependent", coord_sys)]
        rf_improvement = (ind["rmse_rf"] - dep["rmse_rf"]) / ind["rmse_rf"] * 100
        print(f"\nAdding 't' changes RF RMSE by: {rf_improvement:+.2f}% for {coord_sys} mode")
        return rf_improvement, ind, dep
    
    rf_improvement_cart, ind, dep = get_rf_improvement("cartesian")
    rf_improvement_pol, ind, dep = get_rf_improvement("polar")

    print("─" * 60)
    print("💡 VERDICT")
    print("─" * 60)
    if rf_improvement_cart > 10:
        print(
            "  Use DEPENDENT mode — including 't' reduces RF RMSE by\n"
            f"  {rf_improvement_cart:.1f}%, which is significant.\n"
            "  The sampler should be P(rta | x, y, t, runway)."
        )
    elif rf_improvement_cart > 3:
        print(
            "  Marginal benefit from 't'. Consider DEPENDENT mode if\n"
            "  the sampler may be called mid-episode; otherwise INDEPENDENT\n"
            "  is simpler and nearly as accurate."
        )
    else:
        print(
            "  Use INDEPENDENT mode — 't' adds negligible predictive\n"
            f"  value ({rf_improvement_cart:.1f}% RMSE change).\n"
            "  The sampler should be P(rta_remaining | x, y, runway)."
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_mode_comparison(results, save_dir)

def _plot_mode_comparison(
    results: dict, 
    save_dir: Path | None
) -> None:
    """
    Diagnostic figure:
    - Left: Clustered RMSE (Linear vs RF) showing all modes.
    - Right (2x2): Feature importance with uniform baseline.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.3, 1, 1], wspace=0.35, hspace=0.4)
    
    ax_rmse = fig.add_subplot(gs[:, 0])
    
    plot_order = ["independent_cartesian", "dependent_cartesian", "independent_polar", "dependent_polar"]
    fi_coords = [(0, 1), (0, 2), (1, 1), (1, 2)]

    # ── 1. Clustered RMSE Bar Chart ───────────────────────────────────────────
    # Clusters: 0 = Linear, 1 = Random Forest
    n_modes = len(plot_order)
    x_clusters = np.arange(2) 
    width = 0.18  # Width of individual bars within clusters
    
    # Offsets to center the 4 bars around the cluster tick
    offsets = np.linspace(- (n_modes-1)*width/2, (n_modes-1)*width/2, n_modes)

    # Instead of having a lable inside just make them be apart of the legend with their own colour
    for i, mode_key in enumerate(plot_order):
        r = results[mode_key]
        color = r["cfg"].color
        
        # Linear bar in cluster 0, RF bar in cluster 1
        vals = [r["rmse_lr"], r["rmse_rf"]]
        
        # We plot each mode across both clusters
        bars = ax_rmse.bar(x_clusters + offsets[i], vals, width, 
                           label=mode_key.replace('_', ' ').title(),
                           edgecolor="black", linewidth=0.8,
                           alpha=0.6 if i % 2 == 0 else 0.9) # Slightly differentiate indep/dep

        # Add text labels inside or on top
        for j, bar in enumerate(bars):
            # RMSE Value on top
            ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.001,
                         f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5, boxstyle="round")
                        )

    ax_rmse.set_xticks(x_clusters)
    ax_rmse.set_xticklabels(["LINEAR BASELINE", "RANDOM FOREST"], fontsize=11, fontweight="bold")
    ax_rmse.set_ylabel("RMSE (Test Set)", fontsize=10)
    ax_rmse.set_title("Performance by Estimator Type", fontsize=12, fontweight="bold")
    ax_rmse.legend(fontsize=8, frameon=True)

    # ── 2. Feature Importance 2x2 Grid ────────────────────────────────────────
    for idx, mode_key in enumerate(plot_order):
        ax = fig.add_subplot(gs[fi_coords[idx]])
        r = results[mode_key]
        feats, imps = r["features"], r["importances"]
        
        bar_colors = ["#4C9BE8" if f in ("x", "y", "r", "theta") else "#E86B4C" for f in feats]
        bars = ax.barh(np.arange(len(feats)), imps, color=bar_colors, 
                             edgecolor="white", height=0.6, zorder=2)

        # Uniform distribution line (1 / number of features)
        uniform_val = 1.0 / len(feats)
        ax.axvline(uniform_val, color="red", linestyle="--", linewidth=1.2, alpha=0.6, 
                   label=f"Uniform ({uniform_val:.2f})", zorder=3)

        # Feature Importance Labels with semi-transparent background
        for bar, imp in zip(bars, imps):
            ax.text(
                imp + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f"{imp:.3f}", 
                va="center", 
                fontsize=8,
                fontweight='bold',
                zorder=4,  # Ensure it is above the uniform line
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5)
            )

        ax.set_yticks(np.arange(len(feats)))
        ax.set_yticklabels(feats, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1.1)
        ax.set_title(f"{mode_key.replace('_', ' ').upper()}", 
                     fontsize=10, fontweight="bold", color=r["cfg"].color)
        
        # Legend for the uniform line
        ax.legend(fontsize=7, framealpha=0.8)
        
        # Highlight top feature
        ax.axhspan(np.argmax(imps) - 0.35, np.argmax(imps) + 0.35, 
                   color="gold", alpha=0.1, zorder=1)
        
    # ── 3. Definition Note Box ────────────────────────────────────────────────
    # We use ax_rmse as the anchor, placing it in the bottom-left corner
    note_text = (
        r"$\bf{Independent}$: $P(rta\_remaining \mid x, y, rwy)$" + "\n" +
        r"$\bf{Dependent}$: $P(rta \mid x, y, t, rwy)$"
    )

    ax_rmse.text(
        0.05, 0.05, note_text, 
        transform=ax_rmse.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(
            boxstyle='round,pad=0.5', 
            facecolor='white', 
            edgecolor='gray', 
            alpha=0.9
        )
    )

    fig.suptitle("Sampler Mode Analysis: Estimator Clusters & Feature Saliency", fontsize=16, fontweight="bold", y=0.98)
    _savefig(fig, save_dir, "mode_comparison_clustered.png")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, s0: pd.DataFrame) -> None:
    _section_header(7, "SUMMARY")

    n_eps  = df["episode"].nunique()
    n_rwys = df["runway"].nunique()
    has_s0 = s0["episode"].nunique() == n_eps

    print(f"  Dataset         : {n_eps} episodes across {n_rwys} runways")
    print(f"  Step-0 complete : {'✅' if has_s0 else '⚠️  missing step-0 rows'}")
    print()

    for cfg in MODES:
        target = s0[cfg.target_col]
        print(f"  ── {cfg.mode.upper()} ──")
        print(f"    Target    : {cfg.target_label}")
        print(f"    Range     : [{target.min():.4f}, {target.max():.4f}]  std={target.std():.4f}")
        print(f"    Sampler   : {cfg.sampler_str}")
        print()

    print("  See Section 6 VERDICT above for the recommended mode.")
    print()
    print("  Surrogate model (separate concern):")
    print("    Estimates rta_remaining mid-episode given (x, y, t, runway)")
    print("    Sampler is only called at reset — surrogate handles inference")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _get_args():
    p = argparse.ArgumentParser(description="Analyse collected RTA data (both modes always run).")
    p.add_argument("data",         type=str, help="Path to collected .csv or .parquet file.")
    p.add_argument("--save-plots", type=str, default=None, help="Directory to save plots.")
    p.add_argument("--runway",     type=str, nargs="+", default=None,
                   help="Subset of runways to analyse.")
    p.add_argument("--xy-bins",    type=int, default=8,
                   help="Grid bins for spatial conditioning (default: 8).")
    p.add_argument("--t-bins",     type=int, default=4,
                   help="Grid bins for temporal conditioning (default: 4).")
    p.add_argument("--no-plots",   action="store_true", help="Skip all plots (stats only).")
    return p.parse_args()


def analyse(df, runway_filter=None, no_plots=False, xy_bins=10, t_bins=10, save_dir=None):
    """
    Core logic for processing data and running the analysis suite.
    """
    # Filter runways if requested
    if runway_filter:
        unknown = set(runway_filter) - set(df["runway"].unique())
        if unknown:
            print(f"⚠️  Unknown runways requested: {unknown}")
        df = df[df["runway"].isin(runway_filter)]

    # ── Section 1: Health & Enrichment ──
    check_data_health(df)
    df = _enrich(df)
    s0 = _step0(df)

    if s0.empty:
        print("\n⚠️  No step-0 rows found. Re-collect data with the updated collect.py.")
        return None # Return None or raise an exception instead of sys.exit

    runways = sorted(df["runway"].unique())

    # ── Section 2: RTA Checks ──
    check_rta_remaining(df)

    # ── Section 3 & 4: Spatial & RTA Plots ──
    if not no_plots:
        plot_rta_distribution(s0, runways, save_dir)
        plot_spatial_coverage(s0, runways, save_dir)
    else:
        print("\nSECTIONS 3 & 4 — skipped (--no-plots)")

    # ── Section 5: Incremental Conditioning ──
    analyse_incremental_conditioning(
        s0, runways, n_spatial_bins=xy_bins, n_time_bins=t_bins, save_dir=save_dir
    )

    # ── Section 6: Mode Comparison ──
    # Section 6 still runs to print stats even if plots are disabled
    plot_dir = save_dir if not no_plots else None
    analyse_mode_comparison(df, plot_dir)

    # ── Section 7: Summary ──
    print_summary(df, s0)
    
    return df, s0


def run_analyse_cli(experiment_cls):
    """
    CLI Wrapper: Handles argument parsing and IO.
    """
    args = _get_args()
    save_dir = Path(args.save_plots) if args.save_plots else None

    print(f"Loading : {args.data}")
    print(f"Running both modes: independent + dependent")
    
    df_raw = _load(args.data)
    
    # Call the logic-only function
    results = analyse(
        df=df_raw,
        runway_filter=args.runway,
        no_plots=args.no_plots,
        xy_bins=args.xy_bins,
        t_bins=args.t_bins,
        save_dir=save_dir
    )

if __name__ == "__main__":
    run_analyse_cli(None)