"""
cps_coordination/scripts/surrogate_analyse.py
----------------------------------------------
Exploratory analysis to justify feature and coordinate-system choices for the
ETASurrogate.  Outputs publication-ready figures to cps_coordination/figures/.

Usage
-----
  python cps_coordination/scripts/surrogate_analyse.py <path-to-data>
  python cps_coordination/scripts/surrogate_analyse.py \\
      path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import norm as sp_norm, pearsonr, probplot
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME, RTA_TOLERANCE
from cps_coordination.scripts.surrogate_data import (
    MAX_TIME,
    add_lag_features,
    build_feature_matrix,
    compute_iaf_reference_from_env,
    engineer_geometric_features,
    engineer_target_time_feature,
    load_and_prepare,
)

_FIGURES_DIR = Path(__file__).parent.parent / "figures"
_DEFAULT_DATA = (
    "path_planning/rta/data/temporal/No_HER_main/500_training_rta_data.parquet"
)
_DPI = 180

_RUNWAY_COLOURS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#17BECF",
]

RTA_TOLERANCE_SEC = RTA_TOLERANCE * MAX_TIME  # ±60s


def _tolerance_ratio_str(mae_seconds: float) -> str:
    return f"{mae_seconds:.1f}s ({mae_seconds / RTA_TOLERANCE_SEC:.1f}x RTA_TOLERANCE)"

# ── Global style ──────────────────────────────────────────────────────────────

def _apply_style() -> None:
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


# ── 1. Data loading & preparation ─────────────────────────────────────────────

def load_and_clean_data(data_path: str | Path) -> pd.DataFrame:
    """Load and pre-process rollout data via the shared ``surrogate_data``
    pipeline, then attach this file's own EDA-only convenience columns.

    Returns the full success-filtered dataset including both modelling rows
    (steps_to_go > 0) and terminal states (steps_to_go == 0).

    ``r``/``theta``/``heading_sin``/``heading_cos`` are plotting/ablation-only
    columns (this file's own Cartesian-vs-polar exploration, per Finding 1) —
    not part of ``surrogate_data.py``'s canonical feature matrix, so they stay
    local rather than being imported.

    Callers that need only modelling rows should filter afterwards:
        model_df = df[df["steps_to_go"] > 0].dropna(...)
    """
    df = load_and_prepare(Path(data_path))
    df = engineer_target_time_feature(df)

    df["r"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    df["theta"] = np.arctan2(df["y"], df["x"])
    df["heading_sin"] = np.sin(df["heading"])
    df["heading_cos"] = np.cos(df["heading"])

    return df


def _runway_palette(runways: list[str]) -> dict[str, str]:
    return {rwy: _RUNWAY_COLOURS[i % len(_RUNWAY_COLOURS)]
            for i, rwy in enumerate(sorted(runways))}


# ── 2. Sanity checks ──────────────────────────────────────────────────────────

def print_sanity_checks(df: pd.DataFrame) -> None:
    step_counts  = df.groupby("episode")["step"].agg(["count", "nunique"])
    dup_episodes = (step_counts["count"] != step_counts["nunique"]).sum()
    extra_rows   = int(step_counts["count"].sum() - step_counts["nunique"].sum())
    print(
        f"  Duplicate steps:   {dup_episodes} episodes contain duplicate step values "
        f"({extra_rows} extra rows)."
    )
    max_step     = df.groupby("episode")["step"].transform("max")
    non_terminal = int(((df["steps_to_go"] == 0) & (df["step"] != max_step)).sum())
    status = "clean" if non_terminal == 0 else "WARNING — non-terminal zeros present"
    print(
        f"  Terminal-only zero: steps_to_go==0 at non-terminal steps → "
        f"{non_terminal} ({status})."
    )


def print_coordinate_ablation(df: pd.DataFrame) -> None:
    target = df["time_to_go"].to_numpy()
    pairs  = [
        ("Cartesian", [("x", df["x"]), ("y", df["y"])]),
        ("Polar",     [("r", df["r"]), ("θ", df["theta"])]),
    ]
    print("\nPearson correlation with time_to_go — Cartesian vs Polar")
    print(f"{'System':<12} {'Feature':<8} {'r':>8}  {'p-value':>12}")
    print("-" * 46)
    for system, features in pairs:
        for name, series in features:
            r_val, p_val = pearsonr(series.to_numpy(), target)
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            print(f"{system:<12} {name:<8} {r_val:>8.4f}  {p_str:>12}")
        print()


# ── 3. Exploratory plots ──────────────────────────────────────────────────────

def plot_distribution_per_runway(df: pd.DataFrame, out_dir: Path) -> None:
    runway_order = sorted(df["runway"].unique())
    palette      = _runway_palette(runway_order)
    data_arrays  = [df.loc[df["runway"] == rwy, "time_to_go"].to_numpy()
                    for rwy in runway_order]

    fig, ax = plt.subplots(figsize=(15, 5))
    positions = list(range(len(runway_order)))
    vp = ax.violinplot(
        data_arrays, positions=positions,
        widths=0.72, showmedians=False, showextrema=False, showmeans=False,
    )
    for body, rwy in zip(vp["bodies"], runway_order):
        body.set_facecolor(palette[rwy])
        body.set_edgecolor("white")
        body.set_linewidth(1.2)
        body.set_alpha(0.82)
    for i, (arr, _) in enumerate(zip(data_arrays, runway_order)):
        q25, med, q75 = np.percentile(arr, [25, 50, 75])
        ax.plot([i, i], [q25, q75], color="white", lw=3, solid_capstyle="round", zorder=4)
        ax.scatter(i, med, color="white", s=28, zorder=5, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(runway_order, rotation=30, ha="right")
    ax.set_xlabel("Runway", labelpad=8)
    ax.set_ylabel("Time to go", labelpad=8)
    ax.set_title("time_to_go distribution per runway  (successful episodes only)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_steps_to_go_per_runway.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig1_steps_to_go_per_runway.png")


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    candidate = ["r", "theta", "t", "heading", "path_len", "time_to_go"]
    available = [c for c in candidate if c in df.columns]
    labels_map = {"theta": "θ"}
    labels = [labels_map.get(c, c) for c in available]
    corr   = df[available].corr(method="pearson").to_numpy()

    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    n    = len(labels)

    fig, ax = plt.subplots(figsize=(7, 6.2))
    im = ax.imshow(corr, cmap=cmap, norm=norm, aspect="equal")
    for i in range(n):
        for j in range(n):
            val  = corr[i, j]
            rgba = cmap(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if brightness < 0.55 else "#222222",
                    fontweight="bold" if abs(val) > 0.5 else "normal")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor("white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", labelpad=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Pearson correlation matrix — features vs time_to_go")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_correlation_heatmap.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig2_correlation_heatmap.png")


def _scatter_by_runway(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    runway_col: pd.Series,
    palette: dict[str, str],
) -> None:
    for rwy, colour in palette.items():
        mask = runway_col == rwy
        ax.scatter(x_vals[mask], y_vals[mask], c=colour, label=rwy,
                   s=4, alpha=0.35, linewidths=0, rasterized=True)


def plot_r_vs_steps(df: pd.DataFrame, out_dir: Path) -> None:
    palette = _runway_palette(df["runway"].unique().tolist())
    sample  = df.sample(min(60_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _scatter_by_runway(ax, sample["r"].to_numpy(), sample["time_to_go"].to_numpy(),
                       sample["runway"], palette)
    ax.set_xlabel("r  =  √(x² + y²)", labelpad=8)
    ax.set_ylabel("Time to go", labelpad=8)
    ax.set_title("Radial distance vs time_to_go — coloured by runway")
    leg = ax.legend(title="Runway", title_fontsize=8, markerscale=3.5, ncol=3,
                    loc="upper left", borderpad=0.8, handletextpad=0.4, columnspacing=0.8)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3a_r_vs_steps_to_go.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig3a_r_vs_steps_to_go.png")


def plot_theta_vs_steps(df: pd.DataFrame, out_dir: Path) -> None:
    palette = _runway_palette(df["runway"].unique().tolist())
    sample  = df.sample(min(60_000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _scatter_by_runway(ax, np.degrees(sample["theta"].to_numpy()),
                       sample["time_to_go"].to_numpy(), sample["runway"], palette)
    ax.set_xlabel("θ  =  arctan2(y, x)   [degrees]", labelpad=8)
    ax.set_ylabel("Time to go", labelpad=8)
    ax.set_title("Bearing angle vs time_to_go — coloured by runway")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(45))
    leg = ax.legend(title="Runway", title_fontsize=8, markerscale=3.5, ncol=3,
                    loc="upper left", borderpad=0.8, handletextpad=0.4, columnspacing=0.8)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3b_theta_vs_steps_to_go.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig3b_theta_vs_steps_to_go.png")


def plot_polar_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    sample = df.sample(min(80_000, len(df)), random_state=42)
    ttg    = sample["time_to_go"].to_numpy()
    norm   = mcolors.Normalize(vmin=ttg.min(), vmax=ttg.max())

    fig = plt.figure(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("white")
    ax  = fig.add_subplot(projection="polar")
    ax.set_facecolor("#F0F0F0")
    sc = ax.scatter(sample["theta"].to_numpy(), sample["r"].to_numpy(),
                    c=ttg, cmap="plasma", norm=norm,
                    s=4, alpha=0.4, linewidths=0, rasterized=True)
    ax.set_rlabel_position(112.5)
    ax.tick_params(colors="#555555", labelsize=8)
    ax.grid(color="white", linewidth=0.9, linestyle="-", alpha=0.9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CCCCCC")
    cbar = fig.colorbar(sc, ax=ax, pad=0.11, shrink=0.75, aspect=22)
    cbar.set_label("Time to go", labelpad=10, fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Spatial structure  —  (r, θ) coloured by time_to_go",
                 pad=18, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_polar_scatter.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig4_polar_scatter.png")


def plot_iqr_outliers(df: pd.DataFrame, out_dir: Path) -> None:
    runway_order = sorted(df["runway"].unique())
    palette      = _runway_palette(runway_order)
    data_arrays  = [df.loc[df["runway"] == rwy, "time_to_go"].to_numpy()
                    for rwy in runway_order]

    fig, ax = plt.subplots(figsize=(15, 5))
    bp = ax.boxplot(
        data_arrays, positions=list(range(len(runway_order))),
        patch_artist=True, widths=0.6, showfliers=True,
        flierprops=dict(marker=".", markersize=2.5, alpha=0.25,
                        linestyle="none", markeredgewidth=0),
        medianprops=dict(color="white", linewidth=2.2),
        whiskerprops=dict(color="#666666", linewidth=1),
        capprops=dict(color="#666666", linewidth=1.2),
        boxprops=dict(linewidth=0),
    )
    for patch, rwy in zip(bp["boxes"], runway_order):
        patch.set_facecolor(palette[rwy])
        patch.set_alpha(0.82)
    for flier, rwy in zip(bp["fliers"], runway_order):
        flier.set_markerfacecolor(palette[rwy])
        flier.set_markeredgecolor("none")
    ax.set_xticks(range(len(runway_order)))
    ax.set_xticklabels(runway_order, rotation=30, ha="right")
    ax.set_xlabel("Runway", labelpad=8)
    ax.set_ylabel("Time to go", labelpad=8)
    ax.set_title("IQR box plot — time_to_go per runway  (outliers as dots)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_iqr_outliers.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig6_iqr_outliers.png")


def plot_heading_roses(df: pd.DataFrame, out_dir: Path) -> None:
    runways    = sorted(df["runway"].unique())
    palette    = _runway_palette(runways)
    n_cols     = 4
    n_rows     = (len(runways) + n_cols - 1) // n_cols
    n_bins     = 36
    theta_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bar_width  = 2 * np.pi / n_bins

    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.2))
    fig.patch.set_facecolor("white")
    for idx, rwy in enumerate(runways):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="polar")
        ax.set_facecolor("#F0F0F0")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        heading_rad = df.loc[df["runway"] == rwy, "heading"].to_numpy()
        counts, _   = np.histogram(heading_rad % (2 * np.pi), bins=theta_bins)
        ax.bar(theta_bins[:-1], counts, width=bar_width, bottom=0,
               color=palette[rwy], alpha=0.82, linewidth=0)
        ax.set_title(f"RWY {rwy}", fontsize=9, fontweight="bold", pad=10)
        ax.set_yticks([])
        ax.set_thetagrids(range(0, 360, 45),
                          labels=["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                          fontsize=6.5)
        ax.grid(color="white", linewidth=0.8, alpha=0.85)
    for idx in range(len(runways), n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, idx + 1).set_visible(False)
    fig.suptitle("Aircraft heading distribution per runway  (all successful steps)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "fig14_heading_roses.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig14_heading_roses.png")


# ── 4. Model training & evaluation ────────────────────────────────────────────

_ET_PARAMS: dict = dict(
    n_estimators=100, max_depth=15, min_samples_leaf=10,
    max_features="sqrt", n_jobs=-1, random_state=42,
)


_ENGINEERED_COLS = [
    "r_sq",
    "along_track_dist", "cross_track_error", "heading_error",
    "delta_atd", "cumabs_cte", "heading_volatility",
    "remaining_time_budget",
]


def _build_features(
    df: pd.DataFrame,
    runway_encoder: LabelEncoder,
    use_polar: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    spatial    = (
        df[["r", "theta"]].rename(columns={"theta": "θ"})
        if use_polar else df[["x", "y"]]
    )
    extra      = df[[c for c in _ENGINEERED_COLS if c in df.columns]]
    rwy_codes  = pd.Series(
        runway_encoder.transform(df["runway"]).astype(np.float64),
        index=df.index, name="runway",
    )
    feature_df = pd.concat(
        [spatial, df[["t", "heading_sin", "heading_cos"]], extra, rwy_codes], axis=1
    )
    return (
        feature_df.to_numpy(dtype=np.float64),
        df["time_to_go"].to_numpy(dtype=np.float64),
        list(feature_df.columns),
    )


def _et_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "R²":   r2_score(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def reduce_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop features below importance threshold using a fast preliminary ET fit.

    Returns the reduced training matrix, a boolean mask of shape (n_features,)
    for applying the same column selection to a validation set via
    ``X_val[:, mask]``, and the surviving feature names.
    """
    scout = ExtraTreesRegressor(**_ET_PARAMS).fit(X_train, y_train)
    mask = scout.feature_importances_ >= threshold

    kept    = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
    dropped = [feature_names[i] for i in range(len(feature_names)) if not mask[i]]
    print(
        f"    Feature reduction: {mask.sum()}/{len(feature_names)} kept"
        + (f"  |  dropped: {dropped}" if dropped else "  |  none dropped")
    )
    return X_train[:, mask], mask, kept


def _plot_feature_importance(
    model: ExtraTreesRegressor,
    feature_names: list[str],
    out_dir: Path,
) -> None:
    importances  = model.feature_importances_
    idx          = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp   = importances[idx]
    colours = ["#DD8452" if n == "runway" else "#4C72B0" for n in sorted_names]

    fig, ax = plt.subplots(figsize=(max(8, len(feature_names) * 0.55), 5.5))
    ax.bar(range(len(sorted_names)), sorted_imp, color=colours, width=0.7,
           linewidth=0, zorder=3)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean impurity decrease", labelpad=8)
    ax.set_title("Extra Trees feature importances  —  polar feature set")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", label="Continuous features"),
        Patch(facecolor="#DD8452", label="Runway (label-encoded)"),
    ], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_et_feature_importance.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig5_et_feature_importance.png")


def cross_validate_and_evaluate_et(
    df: pd.DataFrame,
    all_runways: list[str],
    out_dir: Path,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """5-fold group cross-validation with strict fold-local feature engineering.

    The IAF reference is exact env geometry (``compute_iaf_reference_from_env``)
    — a pure function of the runway list, not the data — so it carries no
    leakage risk and is computed once up front. Lag features and the feature
    reduction mask remain fold-local: no information from validation episodes
    leaks in at any stage.

    Returns
    -------
    oof_df         : DataFrame of all validation rows (post-engineer + post-lag),
                     row-aligned with the three arrays below.
    y_oof          : concatenated true targets across all folds
    y_pred_polar   : out-of-fold predictions (polar coordinate model)
    y_pred_cart    : out-of-fold predictions (Cartesian coordinate model)
    """
    gkf = GroupKFold(n_splits=n_splits)
    iaf_ref = compute_iaf_reference_from_env(all_runways)
    runway_encoder = LabelEncoder().fit(all_runways)
    val_eng_list: list[pd.DataFrame] = []
    oof_y_true:   list[np.ndarray]   = []
    oof_y_pred_p: list[np.ndarray]   = []
    oof_y_pred_c: list[np.ndarray]   = []
    fold_metrics: list[dict]         = []
    last_model_p: ExtraTreesRegressor | None = None
    last_names_r: list[str] = []

    print(f"\n5-Fold Group Cross-Validation — Extra Trees")
    print(f"  Total rows: {len(df):,}  |  episodes: {df['episode'].nunique():,}")

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(df, groups=df["episode"])
    ):
        print(f"\n  Fold {fold_idx + 1}/{n_splits} ...")

        # a) Raw fold splits
        raw_train = df.iloc[train_idx]
        raw_val   = df.iloc[val_idx]

        # c) Filter to modelling rows
        train_m = (
            raw_train[raw_train["steps_to_go"] > 0]
            .dropna(subset=["time_to_go", "r", "theta"])
        )
        val_m = (
            raw_val[raw_val["steps_to_go"] > 0]
            .dropna(subset=["time_to_go", "r", "theta"])
        )

        # d) Geometric features (exact env-derived IAF reference)
        train_eng = engineer_geometric_features(train_m, iaf_ref)
        val_eng   = engineer_geometric_features(val_m,   iaf_ref)

        # e) Lag features (episode-grouped, no cross-fold bleed)
        train_eng = add_lag_features(train_eng)
        val_eng   = add_lag_features(val_eng)

        # f) Feature matrices for polar and Cartesian models
        X_tr_p, y_tr, feat_names = _build_features(
            train_eng, runway_encoder, use_polar=True,
        )
        X_va_p, y_va, _ = _build_features(
            val_eng, runway_encoder, use_polar=True,
        )
        X_tr_c, _, _ = _build_features(
            train_eng, runway_encoder, use_polar=False,
        )
        X_va_c, _, _ = _build_features(
            val_eng, runway_encoder, use_polar=False,
        )

        # g) Feature reduction (mask derived from training fold only)
        print(f"    Polar  ", end="")
        X_tr_p_r, mask_p, names_r = reduce_features(X_tr_p, y_tr, feat_names)
        X_va_p_r = X_va_p[:, mask_p]
        print(f"    Cartesian ", end="")
        X_tr_c_r, mask_c, _ = reduce_features(X_tr_c, y_tr, feat_names)
        X_va_c_r = X_va_c[:, mask_c]

        # h) Train
        model_p = ExtraTreesRegressor(**_ET_PARAMS).fit(X_tr_p_r, y_tr)
        model_c = ExtraTreesRegressor(**_ET_PARAMS).fit(X_tr_c_r, y_tr)

        # i) Predict on validation fold
        p_pred = model_p.predict(X_va_p_r)
        c_pred = model_c.predict(X_va_c_r)

        # j) Accumulate
        val_eng_list.append(val_eng)
        oof_y_true.append(y_va)
        oof_y_pred_p.append(p_pred)
        oof_y_pred_c.append(c_pred)
        m = _et_metrics(y_va, p_pred)
        fold_metrics.append(m | {"fold": fold_idx + 1})
        print(
            f"    Fold {fold_idx + 1} — R²={m['R²']:.4f}  "
            f"MAE={m['MAE']:.2f} s  RMSE={m['RMSE']:.2f} s"
        )
        last_model_p = model_p
        last_names_r = names_r

    # Concatenate OOF results (reset index so positional alignment is clean)
    oof_df   = pd.concat(val_eng_list).reset_index(drop=True)
    y_oof    = np.concatenate(oof_y_true)
    y_pred_p = np.concatenate(oof_y_pred_p)
    y_pred_c = np.concatenate(oof_y_pred_c)

    r2s    = [m["R²"]   for m in fold_metrics]
    maes   = [m["MAE"]  for m in fold_metrics]
    rmses  = [m["RMSE"] for m in fold_metrics]

    print(f"\n  5-Fold OOF Summary — polar model")
    print(f"  {'Metric':<8} {'Mean':>10} {'Std':>10}")
    print("  " + "-" * 30)
    print(f"  {'R²':<8} {np.mean(r2s):>10.4f} {np.std(r2s):>10.4f}")
    print(f"  {'MAE':<8} {np.mean(maes):>10.4f} {np.std(maes):>10.4f}")
    print(f"  {'RMSE':<8} {np.mean(rmses):>10.4f} {np.std(rmses):>10.4f}")
    print(f"  MAE = {_tolerance_ratio_str(np.mean(maes))}")

    # Feature importance from last fold's polar model (diagnostic)
    if last_model_p is not None:
        _plot_feature_importance(last_model_p, last_names_r, out_dir)

    return oof_df, y_oof, y_pred_p, y_pred_c


# ── 5. Prediction diagnostics ─────────────────────────────────────────────────

def plot_prediction_scatter(
    y_test: np.ndarray,
    y_pred_polar: np.ndarray,
    y_pred_cart: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, y_pred, label in zip(
        axes, [y_pred_polar, y_pred_cart], ["Polar (r, θ)", "Cartesian (x, y)"]
    ):
        ax.set_facecolor("white")
        lo = min(y_test.min(), y_pred.min())
        hi = max(y_test.max(), y_pred.max())
        hb = ax.hexbin(y_test, y_pred, gridsize=60, cmap="Blues",
                       mincnt=1, bins="log", linewidths=0.2)
        ax.plot([lo, hi], [lo, hi], color="#C44E52", lw=1.5, ls="--", zorder=5)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        ax.text(0.05, 0.93, f"R² = {r2:.4f}\nMAE = {mae:.4f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.92))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual time to go", labelpad=8)
        ax.set_ylabel("Predicted time to go", labelpad=8)
        ax.set_title(f"Predicted vs Actual — {label}")
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.88)
        cb.set_label("log₁₀(count)", labelpad=8, fontsize=8)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_prediction_scatter.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig7_prediction_scatter.png")


def plot_error_map(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    x        = test_df["x"].to_numpy()
    y        = test_df["y"].to_numpy()
    residual = y_pred - y_test
    abs_res  = np.abs(residual)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax   = axes[0]
    vmax = float(np.percentile(abs_res, 95))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    sc   = ax.scatter(x, y, c=residual, cmap="RdBu_r", norm=norm,
                      s=3, alpha=0.35, linewidths=0, rasterized=True)
    cb   = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.88)
    cb.set_label("Residual  (pred − actual)", labelpad=8, fontsize=8)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("x", labelpad=8); ax.set_ylabel("y", labelpad=8)
    ax.set_title("Signed prediction error — spatial  (polar model)")
    ax.set_aspect("equal", adjustable="box")

    ax   = axes[1]
    hb   = ax.hexbin(x, y, C=abs_res, gridsize=50, cmap="YlOrRd",
                     reduce_C_function=np.mean, mincnt=10, linewidths=0.2)
    cb   = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.88)
    cb.set_label("Mean |residual|", labelpad=8, fontsize=8)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    ax.set_xlabel("x", labelpad=8); ax.set_ylabel("y", labelpad=8)
    ax.set_title("Mean absolute error per spatial bin  (polar model)")
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(out_dir / "fig8_error_map.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig8_error_map.png")


def print_runway_metrics(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_polar: np.ndarray,
    y_pred_cart: np.ndarray,
) -> None:
    print("\nPer-runway metrics on held-out episodes")
    print(f"{'Runway':<8} {'R²(pol)':>9} {'MAE(pol)':>10} {'RMSE(pol)':>11}"
          f" {'R²(cart)':>10} {'MAE(cart)':>10} {'RMSE(cart)':>11}")
    print("-" * 74)
    for rwy in sorted(test_df["runway"].unique()):
        mask = (test_df["runway"] == rwy).to_numpy()
        yt   = y_test[mask]
        mp   = _et_metrics(yt, y_pred_polar[mask])
        mc   = _et_metrics(yt, y_pred_cart[mask])
        print(
            f"{rwy:<8} {mp['R²']:>9.4f} {mp['MAE']:>10.4f} {mp['RMSE']:>11.4f}"
            f" {mc['R²']:>10.4f} {mc['MAE']:>10.4f} {mc['RMSE']:>11.4f}"
        )


def plot_runway_metrics(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_polar: np.ndarray,
    y_pred_cart: np.ndarray,
    out_dir: Path,
) -> None:
    runways         = sorted(test_df["runway"].unique())
    mae_polar, mae_cart = [], []
    for rwy in runways:
        mask = (test_df["runway"] == rwy).to_numpy()
        yt   = y_test[mask]
        mae_polar.append(mean_absolute_error(yt, y_pred_polar[mask]))
        mae_cart.append(mean_absolute_error(yt,  y_pred_cart[mask]))

    x, w = np.arange(len(runways)), 0.35
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w / 2, mae_polar, width=w, label="Polar (r, θ)",
           color="#4C72B0", linewidth=0, zorder=3)
    ax.bar(x + w / 2, mae_cart,  width=w, label="Cartesian (x, y)",
           color="#DD8452", linewidth=0, zorder=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(runways, rotation=30, ha="right")
    ax.set_xlabel("Runway", labelpad=8)
    ax.set_ylabel("MAE", labelpad=8)
    ax.set_title("Per-runway MAE — polar vs Cartesian feature set")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(out_dir / "fig9_runway_mae.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig9_runway_mae.png")


def plot_residual_analysis(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    residual  = y_pred - y_test
    mu, sigma = float(residual.mean()), float(residual.std())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(residual, bins=100, density=True,
            color="#4C72B0", alpha=0.75, linewidth=0, zorder=3)
    x_fit = np.linspace(residual.min(), residual.max(), 400)
    ax.plot(x_fit, sp_norm.pdf(x_fit, mu, sigma),
            color="#C44E52", lw=2, zorder=4, label=f"N(μ={mu:.4f}, σ={sigma:.4f})")
    ax.axvline(0, color="#333333", lw=1, ls="--", zorder=5)
    ax.set_xlabel("Residual  (pred − actual)", labelpad=8)
    ax.set_ylabel("Density", labelpad=8)
    ax.set_title("Residual distribution  (polar model)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.set_facecolor("white")
    (osm, osr), (slope, intercept, _) = probplot(residual, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color="#4C72B0",
               linewidths=0, rasterized=True, zorder=3)
    ax.plot([osm[0], osm[-1]],
            [slope * osm[0] + intercept, slope * osm[-1] + intercept],
            color="#C44E52", lw=1.5, zorder=4)
    ax.set_xlabel("Theoretical quantiles", labelpad=8)
    ax.set_ylabel("Sample quantiles", labelpad=8)
    ax.set_title("Q-Q plot of residuals vs Normal  (polar model)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig10_residual_distribution.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig10_residual_distribution.png")


def plot_error_by_horizon(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    residual = y_pred - y_test
    abs_res  = np.abs(residual)
    n_bins   = 30
    bins     = np.linspace(y_test.min(), y_test.max(), n_bins + 1)
    centres  = (bins[:-1] + bins[1:]) / 2
    idx      = np.clip(np.digitize(y_test, bins) - 1, 0, n_bins - 1)

    mae_bins  = np.array([abs_res[idx == b].mean() if (idx == b).any() else np.nan
                          for b in range(n_bins)])
    bias_bins = np.array([residual[idx == b].mean() if (idx == b).any() else np.nan
                          for b in range(n_bins)])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.fill_between(centres, 0, mae_bins, color="#4C72B0", alpha=0.2, zorder=2)
    ax.plot(centres, mae_bins, color="#4C72B0", lw=2, zorder=3)
    ax.set_xlabel("Actual time to go", labelpad=8)
    ax.set_ylabel("MAE", labelpad=8)
    ax.set_title("MAE vs prediction horizon  (polar model)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    ax = axes[1]
    ax.fill_between(centres, 0, bias_bins, where=bias_bins >= 0,
                    color="#55A868", alpha=0.25, zorder=2)
    ax.fill_between(centres, 0, bias_bins, where=bias_bins < 0,
                    color="#C44E52", alpha=0.25, zorder=2)
    ax.plot(centres, bias_bins, color="#333333", lw=2, zorder=3)
    ax.axhline(0, color="#333333", lw=1, ls="--", zorder=4)
    ax.set_xlabel("Actual time to go", labelpad=8)
    ax.set_ylabel("Mean bias  (pred − actual)", labelpad=8)
    ax.set_title("Prediction bias vs horizon  (polar model)")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    fig.tight_layout()
    fig.savefig(out_dir / "fig11_error_by_horizon.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig11_error_by_horizon.png")


def plot_feature_vs_error(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
) -> None:
    residual = y_pred - y_test
    n_bins   = 25
    features = [
        ("r",       test_df["r"].to_numpy(),                     "r  =  √(x² + y²)"),
        ("θ",       np.degrees(test_df["theta"].to_numpy()),      "θ  =  arctan2(y, x)  [°]"),
        ("t",       test_df["t"].to_numpy(),                      "t  (simulation time)"),
        ("heading", test_df["heading"].to_numpy(),                "Heading  [rad]"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (name, vals, xlabel) in zip(axes, features):
        edges   = np.unique(np.percentile(vals, np.linspace(0, 100, n_bins + 1)))
        centres = (edges[:-1] + edges[1:]) / 2
        bidx    = np.clip(np.digitize(vals, edges) - 1, 0, len(centres) - 1)
        means   = np.array([residual[bidx == b].mean() if (bidx == b).any() else np.nan
                            for b in range(len(centres))])
        stds    = np.array([residual[bidx == b].std()  if (bidx == b).any() else np.nan
                            for b in range(len(centres))])
        ax.fill_between(centres, means - stds, means + stds,
                        color="#4C72B0", alpha=0.18, zorder=2)
        ax.plot(centres, means, color="#4C72B0", lw=2, zorder=3)
        ax.axhline(0, color="#C44E52", lw=1, ls="--", zorder=4)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel("Mean residual" if ax is axes[0] else "", labelpad=8)
        ax.set_title(f"Bias vs {name}")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    fig.suptitle("Prediction bias as function of each feature  (polar model)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig12_feature_bias.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig12_feature_bias.png")


# ── 6. Advanced experiments ───────────────────────────────────────────────────

def plot_learning_curve(
    df: pd.DataFrame,
    all_runways: list[str],
    out_dir: Path,
) -> None:
    """Learning curve using fold-0 of a 5-fold group split as a fixed held-out set."""
    print("\nFitting learning curve (this may take a moment) ...")

    # Derive a single representative fold-0 train/val split
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(df, groups=df["episode"]))
    raw_train = df.iloc[train_idx]
    raw_val   = df.iloc[val_idx]
    iaf_ref        = compute_iaf_reference_from_env(all_runways)
    runway_encoder = LabelEncoder().fit(all_runways)
    _dnf = dict(subset=["time_to_go", "r", "theta"])
    train_df = engineer_geometric_features(
        raw_train[raw_train["steps_to_go"] > 0].dropna(**_dnf), iaf_ref
    )
    test_df = engineer_geometric_features(
        raw_val[raw_val["steps_to_go"] > 0].dropna(**_dnf), iaf_ref
    )

    X_te, y_test, _ = _build_features(test_df, runway_encoder, use_polar=True)

    rng       = np.random.default_rng(42)
    episodes  = rng.permutation(sorted(train_df["episode"].unique()))
    fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    params    = {**_ET_PARAMS, "n_estimators": 50}

    n_ep_list, r2_list, mae_list = [], [], []
    for frac in fractions:
        n          = max(1, int(len(episodes) * frac))
        sub        = train_df[train_df["episode"].isin(set(episodes[:n]))]
        X_tr, y_tr, _ = _build_features(sub, runway_encoder, use_polar=True)
        y_hat      = ExtraTreesRegressor(**params).fit(X_tr, y_tr).predict(X_te)
        n_ep_list.append(n)
        r2_list.append(r2_score(y_test, y_hat))
        mae_list.append(mean_absolute_error(y_test, y_hat))
        print(f"  {frac*100:>4.0f}%  n_ep={n:>5}  R²={r2_list[-1]:.4f}  MAE={_tolerance_ratio_str(mae_list[-1])}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(n_ep_list, r2_list, color="#4C72B0", lw=2, marker="o", markersize=5, zorder=3)
    axes[0].set_xlabel("Training episodes", labelpad=8)
    axes[0].set_ylabel("R²  (held-out test set)", labelpad=8)
    axes[0].set_title("Learning curve — R²")
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(6))
    axes[1].plot(n_ep_list, mae_list, color="#DD8452", lw=2, marker="o", markersize=5, zorder=3)
    axes[1].set_xlabel("Training episodes", labelpad=8)
    axes[1].set_ylabel("MAE  [s]  (held-out test set)", labelpad=8)
    axes[1].set_title("Learning curve — MAE")
    axes[1].yaxis.set_major_locator(mticker.MaxNLocator(6))
    fig.tight_layout()
    fig.savefig(out_dir / "fig13_learning_curve.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig13_learning_curve.png")


def plot_transformation_analysis(
    df: pd.DataFrame,
    all_runways: list[str],
    out_dir: Path,
) -> None:
    """Target transform comparison using fold-0 of a 5-fold group split."""
    transforms = [
        ("identity", lambda y: y,  lambda yp: yp),
        ("log1p",    np.log1p,     np.expm1),
        ("sqrt",     np.sqrt,      lambda yp: np.clip(yp, 0.0, None) ** 2),
    ]
    colours = ["#4C72B0", "#55A868", "#DD8452"]

    # Derive a single representative fold-0 train/val split
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(df, groups=df["episode"]))
    raw_train = df.iloc[train_idx]
    raw_val   = df.iloc[val_idx]
    iaf_ref        = compute_iaf_reference_from_env(all_runways)
    runway_encoder = LabelEncoder().fit(all_runways)
    _dnf = dict(subset=["time_to_go", "r", "theta"])
    train_df = engineer_geometric_features(
        raw_train[raw_train["steps_to_go"] > 0].dropna(**_dnf), iaf_ref
    )
    test_df = engineer_geometric_features(
        raw_val[raw_val["steps_to_go"] > 0].dropna(**_dnf), iaf_ref
    )

    print("\nFitting transformation analysis models ...")
    X_tr, y_train_raw, _ = _build_features(train_df, runway_encoder, use_polar=True)
    X_te, y_test_raw,  _ = _build_features(test_df,  runway_encoder, use_polar=True)

    n_bins  = 30
    h_bins  = np.linspace(y_test_raw.min(), y_test_raw.max(), n_bins + 1)
    centres = (h_bins[:-1] + h_bins[1:]) / 2
    idx_h   = np.clip(np.digitize(y_test_raw, h_bins) - 1, 0, n_bins - 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    print(f"\n{'Transform':<12} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
    print("-" * 46)

    for row, (name, fwd, inv) in enumerate(transforms):
        colour   = colours[row]
        y_tr_t   = fwd(y_train_raw)
        print(f"  Training {name} model ...", end=" ", flush=True)
        y_pred_t = ExtraTreesRegressor(**_ET_PARAMS).fit(X_tr, y_tr_t).predict(X_te)
        print("done")
        y_pred   = inv(y_pred_t)
        residual = y_pred - y_test_raw
        r2   = r2_score(y_test_raw, y_pred)
        mae  = mean_absolute_error(y_test_raw, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test_raw, y_pred)))
        print(f"{name:<12} {r2:>10.4f} {mae:>10.4f} {rmse:>10.4f}")
        print(f"  MAE = {_tolerance_ratio_str(mae)}")

        mu, sigma = float(residual.mean()), float(residual.std())
        ax        = axes[row, 0]
        ax.hist(residual, bins=100, density=True,
                color=colour, alpha=0.72, linewidth=0, zorder=3)
        x_fit = np.linspace(residual.min(), residual.max(), 400)
        ax.plot(x_fit, sp_norm.pdf(x_fit, mu, sigma),
                color="#C44E52", lw=2, zorder=4, label=f"N(μ={mu:.4f}, σ={sigma:.4f})")
        ax.axvline(0, color="#333333", lw=1, ls="--", zorder=5)
        ax.set_xlabel("Residual  (pred − actual)", labelpad=6)
        ax.set_ylabel("Density", labelpad=6)
        ax.set_title(f"{name}  —  residuals\nμ={mu:.4f}, σ={sigma:.4f}", fontsize=10)
        ax.legend(fontsize=7)

        ax   = axes[row, 1]
        ax.set_facecolor("white")
        lo   = float(min(y_test_raw.min(), y_pred.min()))
        hi   = float(max(y_test_raw.max(), y_pred.max()))
        hb   = ax.hexbin(y_test_raw, y_pred, gridsize=60, cmap="Blues",
                         mincnt=1, bins="log", linewidths=0.2)
        ax.plot([lo, hi], [lo, hi], color="#C44E52", lw=1.5, ls="--", zorder=5)
        ax.text(0.05, 0.93, f"R² = {r2:.4f}\nMAE = {mae:.4f}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.92))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual time to go", labelpad=6)
        ax.set_ylabel("Predicted time to go", labelpad=6)
        ax.set_title(f"{name}  —  predicted vs actual", fontsize=10)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("log₁₀(count)", labelpad=6, fontsize=7)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=6)

        bias_bins = np.array([
            residual[idx_h == b].mean() if (idx_h == b).any() else np.nan
            for b in range(n_bins)
        ])
        ax = axes[row, 2]
        ax.fill_between(centres, 0, bias_bins, where=bias_bins >= 0,
                        color="#55A868", alpha=0.25, zorder=2)
        ax.fill_between(centres, 0, bias_bins, where=bias_bins < 0,
                        color="#C44E52", alpha=0.25, zorder=2)
        ax.plot(centres, bias_bins, color="#333333", lw=2, zorder=3)
        ax.axhline(0, color="#333333", lw=1, ls="--", zorder=4)
        ax.set_xlabel("Actual time to go", labelpad=6)
        ax.set_ylabel("Mean bias  (pred − actual)", labelpad=6)
        ax.set_title(f"{name}  —  bias vs horizon", fontsize=10)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    fig.suptitle(
        "Target transformation comparison  —  identity vs log1p vs sqrt\n"
        "(all metrics back-transformed to original step units)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig15_transformation_analysis.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig15_transformation_analysis.png")


# ── 6a. Target-formulation ablation (Finding 1 / action-plan step 3) ──────────

def run_target_formulation_ablation(
    df: pd.DataFrame,
    all_runways: list[str],
    n_splits: int = 5,
) -> dict:
    """Controlled ablation: does the continuous ``time_to_go`` target close
    the label-noise gap ``steps_to_go x ACTION_TIME`` carries (Finding 1)?

    Same exact production feature set (surrogate_data.py's canonical
    13-column recipe -- the 14th ``remaining_time_budget`` column is
    Finding 2's feature, tested separately in Phase C2), same ET
    hyperparameters, same GroupKFold splits for both arms -- only the
    target formulation changes. Both arms are scored in seconds against the
    SAME true continuous ``time_to_go`` ground truth (never
    ``steps_to_go*ACTION_TIME`` on both sides), so this isolates the effect
    cleanly, unlike the existing uncontrolled numbers already on disk
    (fig7/fig15: different encoding, no feature-importance reduction, no
    ``remaining_time_budget``).
    """
    gkf = GroupKFold(n_splits=n_splits)
    iaf_ref        = compute_iaf_reference_from_env(all_runways)
    runway_encoder = LabelEncoder().fit(all_runways)

    arms = [("steps_to_go x ACTION_TIME", "steps"), ("continuous time_to_go", "seconds")]
    results: dict = {}

    print(f"\nTarget-formulation ablation — {n_splits}-fold GroupKFold, "
          f"exact production feature set (13 cols)")
    print(f"  {'Arm':<28} {'R2':>8} {'MAE':>10} {'RMSE':>10}")
    print("  " + "-" * 60)

    for label, target in arms:
        fold_y_sec:    list[np.ndarray] = []
        fold_pred_sec: list[np.ndarray] = []

        for train_idx, val_idx in gkf.split(df, groups=df["episode"]):
            raw_train = df.iloc[train_idx]
            raw_val   = df.iloc[val_idx]

            train_m = (raw_train[raw_train["steps_to_go"] > 0]
                       .dropna(subset=["time_to_go", "r", "theta"]))
            val_m   = (raw_val[raw_val["steps_to_go"] > 0]
                       .dropna(subset=["time_to_go", "r", "theta"]))

            train_m = engineer_geometric_features(train_m, iaf_ref)
            val_m   = engineer_geometric_features(val_m,   iaf_ref)
            train_m = add_lag_features(train_m)
            val_m   = add_lag_features(val_m)
            train_m = engineer_target_time_feature(train_m)
            val_m   = engineer_target_time_feature(val_m)

            X_tr_full, y_tr, names_full = build_feature_matrix(
                train_m, runway_encoder, target=target,
            )
            X_va_full, y_va, _ = build_feature_matrix(
                val_m, runway_encoder, target=target,
            )
            # Drop the 14th (remaining_time_budget) column -- Finding 2's
            # feature is Phase C2's separate experiment, out of scope here.
            X_tr, names = X_tr_full[:, :-1], names_full[:-1]
            X_va        = X_va_full[:, :-1]

            X_tr_r, mask, _ = reduce_features(X_tr, y_tr, names)
            X_va_r           = X_va[:, mask]

            model  = ExtraTreesRegressor(**_ET_PARAMS).fit(X_tr_r, y_tr)
            y_pred = model.predict(X_va_r)

            if target == "steps":
                y_true_sec = val_m["time_to_go"].to_numpy(dtype=float)
                y_pred_sec = y_pred * ACTION_TIME
            else:
                y_true_sec = y_va
                y_pred_sec = y_pred

            fold_y_sec.append(y_true_sec)
            fold_pred_sec.append(y_pred_sec)

        y_oof    = np.concatenate(fold_y_sec)
        pred_oof = np.concatenate(fold_pred_sec)
        metrics  = _et_metrics(y_oof, pred_oof)
        results[target] = {
            "label": label, "y_test": y_oof, "y_pred": pred_oof, "metrics": metrics,
        }
        print(f"  {label:<28} {metrics['R²']:>8.4f} {metrics['MAE']:>10.1f} {metrics['RMSE']:>10.1f}")

    steps_mae   = results["steps"]["metrics"]["MAE"]
    seconds_mae = results["seconds"]["metrics"]["MAE"]
    improvement = 100.0 * (steps_mae - seconds_mae) / steps_mae
    print(f"\n  steps_to_go x ACTION_TIME : MAE = {_tolerance_ratio_str(steps_mae)}")
    print(f"  continuous time_to_go     : MAE = {_tolerance_ratio_str(seconds_mae)}")
    print(f"  Continuous target {'improves' if improvement > 0 else 'worsens'} "
          f"MAE by {improvement:+.1f}%")

    return results


def plot_target_formulation_ablation(results: dict, out_dir: Path) -> None:
    """Fig 21 — predicted-vs-actual and residual comparison for both arms."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colours = {"steps": "#4C72B0", "seconds": "#C44E52"}

    for row, target in enumerate(["steps", "seconds"]):
        r      = results[target]
        y_test, y_pred = r["y_test"], r["y_pred"]
        colour = colours[target]

        ax   = axes[row, 0]
        lo   = float(min(y_test.min(), y_pred.min()))
        hi   = float(max(y_test.max(), y_pred.max()))
        hb   = ax.hexbin(y_test, y_pred, gridsize=60, cmap="Blues",
                         mincnt=1, bins="log", linewidths=0.2)
        ax.plot([lo, hi], [lo, hi], color="#333333", lw=1.5, ls="--", zorder=5)
        ax.text(0.05, 0.93, f"R² = {r['metrics']['R²']:.4f}\nMAE = {r['metrics']['MAE']:.1f}s",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.92))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual time_to_go  [s]", labelpad=8)
        ax.set_ylabel("Predicted  [s]", labelpad=8)
        ax.set_title(f"{r['label']}  —  predicted vs actual")
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("log₁₀(count)", labelpad=6, fontsize=7)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)

        ax       = axes[row, 1]
        residual = y_pred - y_test
        ax.hist(residual, bins=100, density=True, color=colour, alpha=0.75,
                linewidth=0, zorder=3)
        ax.axvline(0, color="#333333", lw=1, ls="--", zorder=5)
        ax.set_xlabel("Residual  (pred − actual)  [s]", labelpad=8)
        ax.set_ylabel("Density", labelpad=8)
        ax.set_title(f"{r['label']}  —  residuals  "
                     f"(μ={residual.mean():.1f}, σ={residual.std():.1f})")

    fig.suptitle(
        "Target-formulation ablation  —  steps_to_go×ACTION_TIME vs continuous time_to_go\n"
        "(same feature set, same ET hyperparameters, same GroupKFold splits)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig21_target_ablation.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig21_target_ablation.png")


# ── 6b. Champion model — full feature set + data-driven transform ─────────────

# Module-level transform library avoids closure issues with loop-defined lambdas
_TRANSFORM_LIBRARY: list[tuple[str, object, object]] = [
    ("identity", lambda y: y,    lambda yp: yp),
    ("log1p",    np.log1p,       np.expm1),
    ("sqrt",     np.sqrt,        lambda yp: np.clip(yp, 0.0, None) ** 2),
]

# Feature groups for colour-coded importance bars
_FEAT_COLOUR_MAP: dict[str, str] = {
    "r":                  "#4C72B0",
    "theta":              "#4C72B0",
    "t":                  "#55A868",
    "r_sq":               "#55A868",
    "heading_sin":        "#DD8452",
    "heading_cos":        "#DD8452",
    "along_track_dist":   "#C44E52",
    "cross_track_error":  "#C44E52",
    "heading_error":      "#C44E52",
    # Config D — historical / lag features
    "delta_atd":          "#8172B3",
    "cumabs_cte":         "#8172B3",
    "heading_volatility": "#8172B3",
    # Finding 2 — goal-conditioned active temporal target
    "remaining_time_budget": "#CCB974",
}


def select_best_transformation(
    df: pd.DataFrame,
    all_runways: list[str],
    sig_threshold_pct: float = 1.5,
    n_splits: int = 5,
) -> dict:
    """Test identity/log1p/sqrt on the full engineered feature set; select by OOF RMSE.

    Runs the complete pipeline — geo features, lag features, and reduce_features
    — inside each fold so the transform comparison is leak-free.  Only adopts a
    non-identity transform when the RMSE improvement exceeds *sig_threshold_pct*
    percent, avoiding back-transform artefacts for marginal gains.

    Returns a dict with the winning transform, OOF predictions, and a final model
    fitted on the full dataset (for feature importance extraction).
    """
    gkf = GroupKFold(n_splits=n_splits)
    iaf_ref        = compute_iaf_reference_from_env(all_runways)
    runway_encoder = LabelEncoder().fit(all_runways)

    print(f"\n  {'Transform':<12} {'R²':>10} {'MAE':>10} {'RMSE':>10}")
    print("  " + "-" * 46)

    results = []

    for name, fwd, inv in _TRANSFORM_LIBRARY:
        fold_y_te:   list[np.ndarray] = []
        fold_y_pred: list[np.ndarray] = []

        for train_idx, val_idx in gkf.split(df, groups=df["episode"]):
            raw_train = df.iloc[train_idx]
            raw_val   = df.iloc[val_idx]

            train_m = (raw_train[raw_train["steps_to_go"] > 0]
                       .dropna(subset=["time_to_go", "r", "theta"]))
            val_m   = (raw_val[raw_val["steps_to_go"] > 0]
                       .dropna(subset=["time_to_go", "r", "theta"]))

            train_m = engineer_geometric_features(train_m, iaf_ref)
            val_m   = engineer_geometric_features(val_m,   iaf_ref)
            train_m = add_lag_features(train_m)
            val_m   = add_lag_features(val_m)

            X_tr, y_tr_raw, feat_names = _build_features(
                train_m, runway_encoder, use_polar=True,
            )
            X_te, y_te_raw, _ = _build_features(
                val_m, runway_encoder, use_polar=True,
            )

            X_tr_r, mask, _ = reduce_features(X_tr, y_tr_raw, feat_names)
            X_te_r          = X_te[:, mask]

            model_fold = ExtraTreesRegressor(**_ET_PARAMS).fit(X_tr_r, fwd(y_tr_raw))
            fold_y_te.append(y_te_raw)
            fold_y_pred.append(inv(model_fold.predict(X_te_r)))

        y_te_oof   = np.concatenate(fold_y_te)
        y_pred_oof = np.concatenate(fold_y_pred)
        metrics    = _et_metrics(y_te_oof, y_pred_oof)
        bias       = float(np.mean(y_pred_oof - y_te_oof))
        results.append({
            "name":    name,
            "fwd":     fwd,
            "inv":     inv,
            "y_test":  y_te_oof,
            "y_pred":  y_pred_oof,
            "metrics": metrics,
            "bias":    bias,
        })
        print(f"  {name:<12} {metrics['R²']:>10.4f} {metrics['MAE']:>10.4f} "
              f"{metrics['RMSE']:>10.4f}  bias={bias:>+9.1f}")
        print(f"    MAE = {_tolerance_ratio_str(metrics['MAE'])}")

    identity_mae = results[0]["metrics"]["MAE"]
    best_overall  = min(results, key=lambda r: r["metrics"]["MAE"])
    improvement   = 100.0 * (identity_mae - best_overall["metrics"]["MAE"]) / identity_mae

    print(f"\n  Threshold: {sig_threshold_pct:.1f}% MAE reduction")
    print(f"  Best:      {best_overall['name']}  "
          f"(improvement = {improvement:+.2f}% over identity)  "
          f"MAE = {_tolerance_ratio_str(best_overall['metrics']['MAE'])}")

    if improvement >= sig_threshold_pct:
        print(f"  Decision:  significant — champion uses [{best_overall['name']}]")
        winner = best_overall
    else:
        print(f"  Decision:  below threshold — champion uses [identity]")
        winner = results[0]

    winner["improvement_pct"] = improvement
    winner["significant"]     = improvement >= sig_threshold_pct

    # Final model on full data under winning transform; reduce_features applied globally
    full_m = df[df["steps_to_go"] > 0].dropna(subset=["time_to_go", "r", "theta"])
    full_m = engineer_geometric_features(full_m, iaf_ref)
    full_m = add_lag_features(full_m)
    X_full, y_full, feat_names_full = _build_features(
        full_m, runway_encoder, use_polar=True,
    )
    X_full_r, _, names_r = reduce_features(X_full, y_full, feat_names_full)
    winner["model"]         = ExtraTreesRegressor(**_ET_PARAMS).fit(
        X_full_r, winner["fwd"](y_full)
    )
    winner["feature_names"] = names_r
    # All three transforms' OOF metrics (incl. the new per-transform `bias`),
    # not just the winner's -- needed to report the full comparison table.
    winner["all_results"]   = results

    return winner


def plot_champion_feature_importance(
    model: ExtraTreesRegressor,
    feature_names: list[str],
    transform_name: str,
    out_dir: Path,
) -> None:
    """Fig 17 — feature importances colour-coded by feature group."""
    importances  = model.feature_importances_
    idx          = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp   = importances[idx]

    def _colour(name: str) -> str:
        return _FEAT_COLOUR_MAP.get(name, "#8C8C8C")

    colours = [_colour(n) for n in sorted_names]

    fig, ax = plt.subplots(figsize=(max(9, len(feature_names) * 0.62), 5.5))
    ax.bar(range(len(sorted_names)), sorted_imp, color=colours, width=0.72,
           linewidth=0, zorder=3)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean impurity decrease", labelpad=8)
    ax.set_title(
        f"Champion feature importances  (auto-reduced)  ·  transform={transform_name}"
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.legend(handles=[
        Patch(facecolor="#4C72B0", label="Polar spatial  (r, θ)"),
        Patch(facecolor="#55A868", label="Auxiliary  (t, r²)"),
        Patch(facecolor="#DD8452", label="Heading-encoded  (sin h, cos h)"),
        Patch(facecolor="#C44E52", label="IAF-relative geometry  (ATD, CTE, ΔH)"),
        Patch(facecolor="#8172B3", label="Historical / lag  (ΔATD, Σ|CTE|, h-vol)"),
        Patch(facecolor="#CCB974", label="Goal-conditioned target  (remaining_time_budget)"),
        Patch(facecolor="#8C8C8C", label="Runway (label-encoded)"),
    ], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig17_champion_feature_importance.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig17_champion_feature_importance.png")


def plot_champion_spatial_comparison(
    test_eng: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_champion: np.ndarray,
    y_pred_baseline: np.ndarray,
    transform_name: str,
    out_dir: Path,
) -> None:
    """Fig 18 — signed-error scatter and mean-|error| hexbin for baseline vs champion."""
    x = test_eng["x"].to_numpy()
    y = test_eng["y"].to_numpy()

    pairs = [
        ("Baseline polar",                  y_pred_baseline),
        (f"Champion  [{transform_name}]",   y_pred_champion),
    ]
    vmax = float(np.percentile(
        np.abs(np.concatenate([y_pred_baseline - y_test, y_pred_champion - y_test])), 95
    ))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for row, (label, y_pred) in enumerate(pairs):
        res     = y_pred - y_test
        abs_res = np.abs(res)

        ax   = axes[row, 0]
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        sc   = ax.scatter(x, y, c=res, cmap="RdBu_r", norm=norm,
                          s=3, alpha=0.35, linewidths=0, rasterized=True)
        cb   = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Residual  (pred − actual)", labelpad=6, fontsize=8)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
        ax.set_xlabel("x", labelpad=6); ax.set_ylabel("y", labelpad=6)
        ax.set_title(f"{label}  —  signed error")
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.03, 0.97, f"MAE = {abs_res.mean():.2f}", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

        ax = axes[row, 1]
        hb = ax.hexbin(x, y, C=abs_res, gridsize=50, cmap="YlOrRd",
                       reduce_C_function=np.mean, mincnt=10, linewidths=0.2)
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label("Mean |residual|", labelpad=6, fontsize=8)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
        ax.set_xlabel("x", labelpad=6); ax.set_ylabel("y", labelpad=6)
        ax.set_title(f"{label}  —  mean |error| per bin")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Spatial error: baseline polar vs champion model",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig18_champion_spatial_comparison.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig18_champion_spatial_comparison.png")


def plot_champion_runway_comparison(
    test_eng: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_champion: np.ndarray,
    y_pred_baseline: np.ndarray,
    all_runways: list[str],
    transform_name: str,
    out_dir: Path,
) -> None:
    """Fig 19 — per-runway MAE and RMSE: baseline polar vs champion."""
    runways                   = sorted(test_eng["runway"].unique())
    mae_base, mae_champ       = [], []
    rmse_base, rmse_champ     = [], []

    for rwy in runways:
        mask = (test_eng["runway"] == rwy).to_numpy()
        yt   = y_test[mask]
        mae_base.append(mean_absolute_error(yt,  y_pred_baseline[mask]))
        mae_champ.append(mean_absolute_error(yt, y_pred_champion[mask]))
        rmse_base.append(float(np.sqrt(mean_squared_error(yt, y_pred_baseline[mask]))))
        rmse_champ.append(float(np.sqrt(mean_squared_error(yt, y_pred_champion[mask]))))

    x, w = np.arange(len(runways)), 0.35
    fig, axes = plt.subplots(1, 2, figsize=(max(13, len(runways) * 1.1), 5))

    for ax, (base_vals, champ_vals, metric_label) in zip(
        axes,
        [(mae_base,  mae_champ,  "MAE"),
         (rmse_base, rmse_champ, "RMSE")],
    ):
        ax.bar(x - w / 2, base_vals,  width=w, label="Baseline polar",
               color="#4C72B0", linewidth=0, zorder=3, alpha=0.88)
        ax.bar(x + w / 2, champ_vals, width=w,
               label=f"Champion  [{transform_name}]",
               color="#C44E52", linewidth=0, zorder=3, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(runways, rotation=30, ha="right")
        ax.set_xlabel("Runway", labelpad=8)
        ax.set_ylabel(metric_label, labelpad=8)
        ax.set_title(f"Per-runway {metric_label.split()[0]}  —  baseline vs champion")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(6))

    fig.tight_layout()
    fig.savefig(out_dir / "fig19_champion_runway_comparison.png", dpi=_DPI)
    plt.close(fig)
    print("  Saved fig19_champion_runway_comparison.png")


def plot_champion_geo_bias(
    test_eng: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    transform_name: str,
    out_dir: Path,
) -> None:
    """Fig 20 — residual bias vs the three new IAF-relative geometric features."""
    residual = y_pred - y_test
    n_bins   = 30

    features = [
        ("along_track_dist",
         test_eng["along_track_dist"].to_numpy(),
         "Along-track distance  [sim units]"),
        ("cross_track_error",
         test_eng["cross_track_error"].to_numpy(),
         "Cross-track error  [sim units]"),
        ("heading_error",
         np.degrees(test_eng["heading_error"].to_numpy()),
         "Heading error  [°]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, vals, xlabel) in zip(axes, features):
        edges   = np.unique(np.percentile(vals, np.linspace(0, 100, n_bins + 1)))
        centres = (edges[:-1] + edges[1:]) / 2
        bidx    = np.clip(np.digitize(vals, edges) - 1, 0, len(centres) - 1)
        means   = np.array([residual[bidx == b].mean() if (bidx == b).any() else np.nan
                            for b in range(len(centres))])
        stds    = np.array([residual[bidx == b].std()  if (bidx == b).any() else np.nan
                            for b in range(len(centres))])
        ax.fill_between(centres, means - stds, means + stds,
                        color="#C44E52", alpha=0.18, zorder=2)
        ax.plot(centres, means, color="#C44E52", lw=2, zorder=3)
        ax.axhline(0, color="#333333", lw=1, ls="--", zorder=4)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel("Mean residual" if ax is axes[0] else "", labelpad=8)
        ax.set_title(f"Bias vs {name}")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

    fig.suptitle(
        f"Champion residual bias vs IAF-relative features  [{transform_name}]",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig20_champion_geo_bias.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig20_champion_geo_bias.png")


# ── 7. CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETA surrogate EDA: feature and coordinate-system analysis."
    )
    parser.add_argument(
        "data", nargs="?", default=_DEFAULT_DATA,
        help=f"Path to rollout parquet or CSV (default: {_DEFAULT_DATA})",
    )
    args = parser.parse_args()

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # ── Load — returns full df including terminal states ─────────────────────
    print(f"Loading data from: {args.data}")
    df = load_and_clean_data(args.data)
    model_df = df[df["steps_to_go"] > 0].dropna(subset=["time_to_go", "r", "theta"])
    all_runways = sorted(df["runway"].unique())
    print(
        f"  {len(model_df):,} modelling rows  ·  "
        f"{model_df['episode'].nunique():,} successful episodes  ·  "
        f"{len(all_runways)} runways  ·  "
        f"{(df['steps_to_go'] == 0).sum():,} IAF terminal states"
    )

    # ── Sanity ───────────────────────────────────────────────────────────────
    print("\nSanity checks ...")
    print_sanity_checks(model_df)

    # ── Exploratory figures (modelling rows only) ─────────────────────────────
    print("\nGenerating exploratory figures ...")
    plot_distribution_per_runway(model_df, _FIGURES_DIR)
    plot_correlation_heatmap(model_df, _FIGURES_DIR)
    plot_r_vs_steps(model_df, _FIGURES_DIR)
    plot_theta_vs_steps(model_df, _FIGURES_DIR)
    plot_polar_scatter(model_df, _FIGURES_DIR)
    plot_iqr_outliers(model_df, _FIGURES_DIR)
    plot_heading_roses(model_df, _FIGURES_DIR)
    print_coordinate_ablation(model_df)

    # ── 5-fold group cross-validated baseline model ───────────────────────────
    print("\nRunning 5-fold group cross-validation ...")
    oof_df, y_oof, y_pred_p, y_pred_c = cross_validate_and_evaluate_et(
        df, all_runways, _FIGURES_DIR,
    )

    # ── Prediction diagnostics (OOF arrays) ──────────────────────────────────
    print("\nGenerating diagnostic figures ...")
    plot_prediction_scatter(y_oof, y_pred_p, y_pred_c, _FIGURES_DIR)
    plot_error_map(oof_df, y_oof, y_pred_p, _FIGURES_DIR)
    print_runway_metrics(oof_df, y_oof, y_pred_p, y_pred_c)
    plot_runway_metrics(oof_df, y_oof, y_pred_p, y_pred_c, _FIGURES_DIR)
    plot_residual_analysis(y_oof, y_pred_p, _FIGURES_DIR)
    plot_error_by_horizon(y_oof, y_pred_p, _FIGURES_DIR)
    plot_feature_vs_error(oof_df, y_oof, y_pred_p, _FIGURES_DIR)

    # ── Advanced experiments ──────────────────────────────────────────────────
    print("\nGenerating advanced experiment figures ...")
    plot_learning_curve(df, all_runways, _FIGURES_DIR)
    plot_transformation_analysis(df, all_runways, _FIGURES_DIR)

    # ── Target-formulation ablation (Finding 1 / action-plan step 3) ─────────
    ablation_results = run_target_formulation_ablation(df, all_runways)
    plot_target_formulation_ablation(ablation_results, _FIGURES_DIR)

    # ── Champion pipeline — full feature set × best transform (data-driven) ──
    print("\nChampion selection (full auto-reduced feature set) ...")
    print("  Testing target transformations (5-fold OOF) ...")
    champion = select_best_transformation(df, all_runways)
    tf = champion["name"]
    print(f"\n  Champion summary:")
    print(f"    Features  : full engineered set (auto-reduced by importance threshold)")
    print(f"    Transform : {tf}  "
          f"({'significant' if champion['significant'] else 'no gain, identity used'}; "
          f"improvement = {champion['improvement_pct']:+.2f}%)")
    print(f"    R²={champion['metrics']['R²']:.4f}  "
          f"MAE={champion['metrics']['MAE']:.4f}  "
          f"RMSE={champion['metrics']['RMSE']:.4f}")
    print(f"    MAE = {_tolerance_ratio_str(champion['metrics']['MAE'])}")

    print("\nGenerating champion diagnostic figures ...")
    plot_champion_feature_importance(
        champion["model"], champion["feature_names"],
        tf, _FIGURES_DIR,
    )
    plot_champion_spatial_comparison(
        oof_df, champion["y_test"], champion["y_pred"], y_pred_p,
        tf, _FIGURES_DIR,
    )
    plot_champion_runway_comparison(
        oof_df, champion["y_test"], champion["y_pred"], y_pred_p,
        all_runways, tf, _FIGURES_DIR,
    )
    plot_champion_geo_bias(
        oof_df, champion["y_test"], champion["y_pred"],
        tf, _FIGURES_DIR,
    )

    print(f"\nAll figures saved to {_FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()
