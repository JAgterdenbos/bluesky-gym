from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .samplers import SamplerRegistry

from typing import Optional, List, Dict, Any

# ── Global Plotting Configuration ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.axisbelow":     True,
})

# ── Tie tolerance: models within this fraction of the best are co-winners ─────
TIE_TOLERANCE = 0.01   # 1 % relative gap

# ── Metrics ───────────────────────────────────────────────────────────────────

def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Preferred over standard MAPE because it is stable when y_true approaches
    zero and is bounded to [0 %, 200 %].
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / (denominator + epsilon)) * 100)

def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    h: int = 1,
) -> tuple[float, float]:
    """
    Diebold-Mariano test: is model A significantly better than model B?
    Uses squared error loss. Returns (dm_stat, p_value).
    A negative DM stat means A has lower errors (better).
    """
    from scipy.stats import t as t_dist

    e_a = (y_true - y_pred_a) ** 2
    e_b = (y_true - y_pred_b) ** 2
    d   = e_a - e_b
    n   = len(d)
    d_bar = d.mean()

    gamma_0 = np.var(d, ddof=1)
    nw_var  = gamma_0
    for lag in range(1, h + 1):
        gamma_lag = np.cov(d[lag:], d[:-lag])[0, 1]
        nw_var   += 2 * gamma_lag
    nw_var = max(nw_var, 1e-12)

    dm_stat = d_bar / np.sqrt(nw_var / n)
    p_value = 2 * t_dist.sf(abs(dm_stat), df=n - 1)
    return float(dm_stat), float(p_value)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the standard regression metric dict for an array pair."""
    return {
        "R2":        r2_score(y_true, y_pred),
        "MAE":       mean_absolute_error(y_true, y_pred),
        "RMSE":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE (%)": smape(y_true, y_pred),
    }

def _agg_cv_metrics(fold_metrics: List[dict]) -> dict:
    """
    Aggregate a list of per-fold metric dicts into mean ± std summary.

    Returns a flat dict with keys like 'R2_mean', 'R2_std', etc.,
    plus a combined 'R2_mean±std' string for display.
    """
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"]  = float(vals.std())
        out[f"{k}"]      = f"{vals.mean():.4f} ± {vals.std():.4f}"
    return out

# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_prep_data(
    filepath: str,
    use_polar: bool = False,
    include_t: bool = False,
    grab_min_dist: bool = False,
    use_log_target: bool = False,
    epsilon: float = 1e-6,
) -> tuple[list[str], np.ndarray, np.ndarray, pd.Series]:
    """
    Load data, derive dist_to_go, and return per-sample arrays.

    Returns
    -------
    runways       : list of unique runway identifiers
    X_all         : (N, 2 or 3) feature matrix — [coord1, coord2] or [..., t]
    y_all         : (N,) dist_to_go array (km) (log-transformed if requested)
    runway_labels : (N,) Series of runway strings aligned with X_all / y_all
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"📦 Loading data from {filepath} (polar={use_polar})...")
    df = pd.read_csv(filepath) if path.suffix == ".csv" else pd.read_parquet(filepath)

    required = {"x", "y", "t", "runway", "total_dist_km", "path_len"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["dist_to_go"] = df["total_dist_km"] - df["path_len"]

    if grab_min_dist:
        initial_len = len(df)
        df["x_rounded"] = df["x"].round(3)
        df["y_rounded"] = df["y"].round(3)
        df = (
            df.sort_values("dist_to_go", ascending=True) 
            .drop_duplicates(subset=["x_rounded", "y_rounded", "runway"], keep="first")
            .drop(columns=["x_rounded", "y_rounded"])
        )
        print(f"🎯 Filtered to min dist_to_go. Rows reduced: {initial_len} -> {len(df)}")

    if use_polar:
        df["r"]     = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        df["theta"] = np.arctan2(df["y"], df["x"])
        spatial_cols = ["r", "theta"]
    else:
        spatial_cols = ["x", "y"]

    feature_cols = spatial_cols + (["t"] if include_t else [])

    X_all = df[feature_cols].values
    y_all = df["dist_to_go"].values

    if use_log_target:
        """
        Log-Transformation Rationale:
        By transforming the target variable using $y' = \ln(y + \epsilon)$, we convert 
        the multiplicative spatial error (heteroscedasticity, where error variance 
        grows linearly with distance) into an additive error. This stabilizes the 
        variance across the spatial domain, satisfying the homoscedasticity 
        assumption required by the Gauss-Markov theorem for optimal unbiased estimation.
        """
        y_all = np.log(y_all + epsilon)
        print("🔧 Applied log-transformation to the target variable.")

    runway_labels = df["runway"].reset_index(drop=True)
    runways = df["runway"].unique().tolist()

    print(f"✅ Loaded {len(df)} samples across {len(runways)} runways.")
    return runways, X_all, y_all, runway_labels


def make_dist_bins(
    y_all: np.ndarray,
    n_bins: int,
    min_test_dist: float = 0.1,
) -> tuple[list[float], list[str]]:
    """
    Build *n_bins* equal-frequency (quantile) distance bins from the data.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
 
    y_nontrivial = y_all[y_all > min_test_dist]
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(y_nontrivial, quantiles)
 
    edges[0]  = min_test_dist
    edges[-1] = np.inf
 
    edges = sorted(set(np.round(edges, 4)))
    if len(edges) < n_bins + 1:
        actual_bins = len(edges) - 1
        print(f"Warning: Requested {n_bins} bins but only {actual_bins} unique quantile "
              f"edges found — using {actual_bins} bins.")
 
    bins = list(edges)
 
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        labels.append(f">{lo:.1f} km" if np.isinf(hi) else f"{lo:.1f}-{hi:.1f} km")
 
    return bins, labels


# ── Saving helpers ─────────────────────────────────────────────────────────────

def _save_and_show_fig(fig, save_dir: Optional[Path], name: str):
    if save_dir is None:
        plt.show()
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / name
    fig.savefig(save_path, bbox_inches="tight")
    print(f"  saved → {save_path}")

def _is_tied_best(value: float, best: float, lower_is_better: bool) -> bool:
    if best == 0:
        return np.isclose(value, best)
    
    if lower_is_better:
        return value <= best * (1 + TIE_TOLERANCE)
    else:
        return value >= best * (1 - TIE_TOLERANCE)


# ── Cross-validated evaluation ────────────────────────────────────────────────

def evaluate_models_cv(
    runways: list[str],
    X_all: np.ndarray,
    y_all: np.ndarray,
    runway_labels: pd.Series,
    n_splits: int = 5,
    min_test_dist: float = 0.1,
    dist_bins: Optional[list[float]] = None,
    bin_labels: Optional[list[str]] = None,
    use_log_target: bool = False,
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, np.ndarray, np.ndarray, dict]:
    """
    K-fold cross-validation across all registered samplers.
    """
    dist_bins, bin_labels = dist_bins or [], bin_labels or []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    available_samplers = [
        n for n in SamplerRegistry.list_available()
        if n not in {"KDEDTGSampler", "RadiusNeighborsDTGSampler"}
    ]

    fold_details:  dict[str, list[dict]] = {n: [] for n in available_samplers}
    rwy_r2_folds:  dict[str, dict[str, list[float]]] = {
        n: {r: [] for r in runways} for n in available_samplers
    }
    bin_mae_folds: dict[str, dict[str, list[float]]] = {
        n: {b: [] for b in bin_labels} for n in available_samplers
    }

    last_fold_y_true: np.ndarray | None = None
    last_fold_preds:  dict[str, np.ndarray] = {}

    indices = np.arange(len(y_all))
    runway_arr = runway_labels.values

    print(f"\n🚀 Cross-validating {len(available_samplers)} samplers "
          f"({n_splits} folds, test filter >{min_test_dist} km)...\n")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"── Fold {fold_idx + 1}/{n_splits} ────────────────────────────")

        X_train_list, y_train_list = [], []
        X_test_list,  y_test_list  = [], []
        for rwy in runways:
            rwy_mask_train = runway_arr[train_idx] == rwy
            rwy_mask_test  = runway_arr[test_idx]  == rwy

            X_train_list.append(X_all[train_idx][rwy_mask_train])
            y_train_list.append(y_all[train_idx][rwy_mask_train])

            X_te_rwy = X_all[test_idx][rwy_mask_test]
            y_te_rwy = y_all[test_idx][rwy_mask_test]

            # Filter trivial near-runway points (ensure filtering is applied to physical distances)
            if use_log_target:
                physical_y_te_rwy = np.exp(y_te_rwy) - epsilon
                keep = physical_y_te_rwy > min_test_dist
            else:
                keep = y_te_rwy > min_test_dist

            X_test_list.append(X_te_rwy[keep])
            y_test_list.append(y_te_rwy[keep])

        y_true_fold = np.concatenate(y_test_list)

        # Ensure bins evaluate on physical km distances
        if use_log_target:
            y_true_fold_metric = np.exp(y_true_fold) - epsilon
        else:
            y_true_fold_metric = y_true_fold

        bin_indices_fold = pd.cut(
            y_true_fold_metric,
            bins=dist_bins, labels=bin_labels, right=True,
        )

        if fold_idx == n_splits - 1:
            last_fold_y_true = y_true_fold_metric

        for name in available_samplers:
            try:
                print(f"\r  Evaluating sampler: {name}...".ljust(50), end="", flush=True)
                sampler = SamplerRegistry.make(name)
                sampler.fit(X_train_list, y_train_list, runways)

                y_pred_rwy_parts = []
                for i, rwy in enumerate(runways):
                    if len(X_test_list[i]) == 0:
                        continue
                    y_pred_rwy = sampler.sample(rwy, X_test_list[i])
                    y_pred_rwy_parts.append(y_pred_rwy)

                    # Inverse transform per-runway metrics to evaluate in physical units
                    if use_log_target:
                        y_test_rwy_metric = np.exp(y_test_list[i]) - epsilon
                        y_pred_rwy_metric = np.exp(y_pred_rwy) - epsilon
                    else:
                        y_test_rwy_metric = y_test_list[i]
                        y_pred_rwy_metric = y_pred_rwy

                    rwy_r2_folds[name][rwy].append(
                        r2_score(y_test_rwy_metric, y_pred_rwy_metric)
                    )

                y_pred_fold = np.concatenate(y_pred_rwy_parts)

                assert len(y_pred_fold) == len(y_true_fold), \
                    f"{name}: pred/true length mismatch ({len(y_pred_fold)} vs {len(y_true_fold)})"

                # Exponentiate predictions if target was logged
                if use_log_target:
                    y_pred_fold_metric = np.exp(y_pred_fold) - epsilon
                else:
                    y_pred_fold_metric = y_pred_fold

                fold_details[name].append(_compute_metrics(y_true_fold_metric, y_pred_fold_metric))

                # Per-distance-bin MAE
                for bl in bin_labels:
                    mask = bin_indices_fold == bl
                    if mask.sum() > 0:
                        bin_mae_folds[name][bl].append(
                            mean_absolute_error(y_true_fold_metric[mask], y_pred_fold_metric[mask])
                        )
                    else:
                        bin_mae_folds[name][bl].append(np.nan)

                if fold_idx == n_splits - 1:
                    last_fold_preds[name] = y_pred_fold_metric

            except Exception as exc:
                print(f"  ⚠️  {name} failed on fold {fold_idx + 1}: {exc}")
                fold_details[name].append({
                    "R2": np.nan, "MAE": np.nan,
                    "RMSE": np.nan, "sMAPE (%)": np.nan,
                })

        print(f"\r   Done — {len(available_samplers)} models evaluated.\n".ljust(100), flush=True)

    # ── Aggregate results ──────────────────────────────────────────────────
    global_rows  = []
    per_rwy_rows = {}
    bin_rows     = {}

    best_r2_mean = -np.inf
    best_name    = ""

    for name in available_samplers:
        folds = fold_details[name]
        if not folds or all(np.isnan(f["R2"]) for f in folds):
            continue

        agg = _agg_cv_metrics(folds)

        rwy_r2_means = {}
        for rwy in runways:
            vals = [v for v in rwy_r2_folds[name][rwy] if not np.isnan(v)]
            rwy_r2_means[rwy] = float(np.mean(vals)) if vals else np.nan
        per_rwy_rows[name] = rwy_r2_means

        bin_summary = {}
        for bl in bin_labels:
            vals = [v for v in bin_mae_folds[name][bl] if not np.isnan(v)]
            if vals:
                m, s = float(np.mean(vals)), float(np.std(vals))
                bin_summary[bl] = f"{m:.3f} ± {s:.3f}"
            else:
                bin_summary[bl] = "N/A"
        bin_rows[name] = bin_summary

        r2_mean = agg["R2_mean"]
        global_rows.append({
            "Model":       name,
            "R2":          agg["R2"],
            "MAE (km)":    agg["MAE"],
            "RMSE (km)":   agg["RMSE"],
            "sMAPE (%)":   agg["sMAPE (%)"],
            "R2_mean":     agg["R2_mean"],
            "R2_std":      agg["R2_std"],
            "MAE_mean":    agg["MAE_mean"],
            "MAE_std":     agg["MAE_std"],
            "RMSE_mean":   agg["RMSE_mean"],
            "RMSE_std":    agg["RMSE_std"],
            "sMAPE_mean":  agg["sMAPE (%)_mean"],
            "sMAPE_std":   agg["sMAPE (%)_std"],
        })

        if r2_mean > best_r2_mean:
            best_r2_mean = r2_mean
            best_name    = name

    results_df = (
        pd.DataFrame(global_rows)
        .sort_values("R2_mean", ascending=False)
        .reset_index(drop=True)
    )
    per_rwy_df   = pd.DataFrame(per_rwy_rows).T.loc[results_df["Model"]]
    bin_metrics_df = (
        pd.DataFrame(bin_rows).T
        .loc[results_df["Model"]]
        .rename_axis("Model")
        .reset_index()
    )

    best_y_true = last_fold_y_true if last_fold_y_true is not None else np.array([])
    best_y_pred = last_fold_preds.get(best_name, np.array([]))

    return (
        results_df, per_rwy_df, bin_metrics_df,
        last_fold_preds, best_y_true, best_y_pred, fold_details,
    )


# ── Colour helpers ─────────────────────────────────────────────────────────────

_PALETTE = {
    "best":  "#3266ad",
    "tied":  "#5d9fd8",
    "other": "#9db5c8",
}

def _bar_colors(values: pd.Series, lower_is_better: bool) -> list[str]:
    best = values.min() if lower_is_better else values.max()
    colors = []
    for v in values:
        if np.isclose(v, best):
            colors.append(_PALETTE["best"])
        elif _is_tied_best(v, best, lower_is_better):
            colors.append(_PALETTE["tied"])
        else:
            colors.append(_PALETTE["other"])
    return colors


# ── Plotting logic (Unchanged structurally) ───────────────────────────────────

def plot_benchmark_metrics(results_df: pd.DataFrame, save_dir: Optional[Path] = None):
    metrics = [
        ("R2_mean",    "R2_std",    "R² score",                    False, "%.4f"),
        ("MAE_mean",   "MAE_std",   "Mean Absolute Error (km)",    True,  "%.3f"),
        ("RMSE_mean",  "RMSE_std",  "Root Mean Squared Error (km)", True,  "%.3f"),
        ("sMAPE_mean", "sMAPE_std", "Symmetric MAPE (%)",           True,  "%.2f"),
    ]

    n_models   = len(results_df)
    bar_height = 0.45
    fig_h      = max(2.5, n_models * 0.42 + 1.2)
    fig, axes  = plt.subplots(2, 2, figsize=(16, fig_h * 2))
    axes = axes.flatten()

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=_PALETTE["best"],  label="Best"),
        plt.Rectangle((0, 0), 1, 1, color=_PALETTE["tied"],
                      label=f"Tied best (±{TIE_TOLERANCE:.0%})"),
        plt.Rectangle((0, 0), 1, 1, color=_PALETTE["other"], label="Other"),
    ]

    for ax, (col, col_std, title, asc, fmt) in zip(axes, metrics):
        df_s   = results_df.sort_values(by=col, ascending=asc).reset_index(drop=True)
        colors = _bar_colors(df_s[col], lower_is_better=asc)

        bars = ax.barh(
            df_s["Model"], df_s[col],
            xerr=df_s[col_std],
            color=colors, edgecolor="white", linewidth=0.5, height=bar_height,
            error_kw=dict(ecolor="#444", capsize=3, elinewidth=1),
        )
        for bar, (_, row) in zip(bars, df_s.iterrows()):
            label = f"{fmt % row[col]}  (±{fmt % row[col_std]})"
            ax.text(
                row[col] + row[col_std],
                bar.get_y() + bar.get_height() / 2,
                f"  {label}", va="center", ha="left", fontsize=7.5, color="#444",
            )

        ax.set_title(title, pad=8)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="#ccc")
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.20)

    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("Benchmark — DTG sampler comparison (cross-validated mean ± std)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, "benchmark_metrics.png")

def plot_runway_heatmap(per_rwy_df: pd.DataFrame, save_dir: Optional[Path] = None):
    if per_rwy_df.empty:
        return

    data    = per_rwy_df.values.astype(float)
    models  = list(per_rwy_df.index)
    runways = list(per_rwy_df.columns)

    fig, ax = plt.subplots(
        figsize=(max(6, len(runways) * 1.4), max(4, len(models) * 0.6 + 1.5))
    )

    vmin = max(0, np.nanmin(data) - 0.05)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dtg", ["#d94f3d", "#f5c342", "#3aad6e"]
    )
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=1.0)

    for ri in range(len(models)):
        for ci in range(len(runways)):
            v = data[ri, ci]
            txt_col = "white" if v < (vmin + 1.0) / 2 else "#1a1a1a"
            ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color=txt_col, fontweight="bold")

    ax.set_xticks(range(len(runways)));  ax.set_xticklabels(runways)
    ax.set_yticks(range(len(models)));   ax.set_yticklabels(models)
    ax.set_title("Per-runway R²  (CV mean across folds)", pad=10)
    ax.grid(False)

    plt.colorbar(im, ax=ax, label="R²", fraction=0.03, pad=0.02)
    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, "runway_heatmap.png")

def plot_bin_metrics(
    bin_metrics_df: pd.DataFrame,
    results_df: pd.DataFrame,
    bin_labels: Optional[list[str]] = None,
    save_dir: Optional[Path] = None,
):
    bin_labels = bin_labels or []

    means_dict: dict[str, list[float]] = {}
    stds_dict:  dict[str, list[float]] = {}

    for _, row in bin_metrics_df.iterrows():
        model = row["Model"]
        m_vals, s_vals = [], []
        for bl in bin_labels:
            cell = row.get(bl, "N/A")
            if isinstance(cell, str) and "±" in cell:
                parts = cell.split("±")
                m_vals.append(float(parts[0].strip()))
                s_vals.append(float(parts[1].strip()))
            else:
                m_vals.append(np.nan)
                s_vals.append(np.nan)
        means_dict[model] = m_vals
        stds_dict[model]  = s_vals

    models     = list(means_dict.keys())
    n_models   = len(models)
    n_bins     = len(bin_labels)
    bar_w      = 0.8 / n_models
    x          = np.arange(n_bins)

    palette = plt.cm.Blues(np.linspace(0.85, 0.35, n_models))

    fig, ax = plt.subplots(figsize=(max(10, n_bins * 2), 5))

    for i, model in enumerate(models):
        offset = (i - n_models / 2 + 0.5) * bar_w
        ax.bar(
            x + offset,
            means_dict[model],
            width=bar_w * 0.9,
            yerr=stds_dict[model],
            label=model,
            color=palette[i],
            edgecolor="white",
            linewidth=0.4,
            error_kw=dict(ecolor="#555", capsize=3, elinewidth=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Distance-to-go bin")
    ax.set_ylabel("MAE (km)  [mean ± std over folds]")
    ax.set_title(
        "Per-distance-bin MAE — cross-validated\n"
        "(lower is better; error bars = std across CV folds)",
        pad=8,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#ccc")
    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, "bin_mae.png")

def plot_parity_and_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: Optional[Path] = None,
):
    residuals = y_pred - y_true
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(y_true, y_pred, alpha=0.1, edgecolors="none",
                color="#3266ad", s=3, rasterized=True)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction (y = x)")
    ax1.set_title(f"Parity plot — {model_name}\n(last CV fold)")
    ax1.set_xlabel("Actual dist_to_go (km)")
    ax1.set_ylabel("Predicted dist_to_go (km)")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.hist(residuals, bins=80, color="#5d9fd8", edgecolor="white",
             linewidth=0.4, density=True, alpha=0.85, label="Residuals")
    ax2.axvline(residuals.mean(), color="#e67e22", linestyle="-", lw=1.2,
                label=f"Mean = {residuals.mean():.2f} km")

    try:
        from scipy.stats import gaussian_kde
        kde    = gaussian_kde(residuals)
        kde_x  = np.linspace(residuals.min(), residuals.max(), 300)
        bw_val = kde.factor
        ax2.plot(kde_x, kde(kde_x), color="#1a4a80", lw=1.8, alpha=0.8,
                 label=f"KDE (BW={bw_val:.3f})")
    except ImportError:
        pass

    bias_note = f"Bias = {residuals.mean():.2f} km"
    std_note  = f"Std = {residuals.std():.2f} km"
    ax2.set_title(f"Residual distribution — {model_name}\n{bias_note}   {std_note}")
    ax2.set_xlabel("Residual (predicted − actual)  (km)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    _save_and_show_fig(fig, save_dir, f"parity_residuals_{model_name}.png")

# ── Statistical tests ─────────────────────────────────────────────────────────

def print_statistical_tests(results_df: pd.DataFrame, all_predictions: dict, y_true: np.ndarray):
    from scipy.stats import wilcoxon

    best_name = results_df.iloc[0]["Model"]
    pred_best = all_predictions.get(best_name)
    if pred_best is None:
        print("⚠️  Best model predictions not available for statistical tests.")
        return

    print(f"\n📊  Statistical tests  vs  {best_name}  (last fold)")
    print(f"{'Model':<35} {'DM stat':>9} {'DM p':>8} {'W p':>8}  {'sig':>5}")
    print("-" * 72)

    for _, row in results_df.iterrows():
        name = row["Model"]
        if name == best_name or name not in all_predictions:
            continue

        dm_stat, dm_p = diebold_mariano_test(y_true, all_predictions[name], pred_best)

        try:
            errors_other = (y_true - all_predictions[name]) ** 2
            errors_best  = (y_true - pred_best) ** 2
            _, w_p = wilcoxon(errors_other, errors_best, zero_method="wilcox")
        except ValueError:
            w_p = 1.0

        both_sig = dm_p < 0.05 and w_p < 0.05
        stars = "***" if both_sig and max(dm_p, w_p) < 0.001 \
           else "**"  if both_sig and max(dm_p, w_p) < 0.01  \
           else "*"   if both_sig                             \
           else "ns"

        print(f"  {name:<33} {dm_stat:>+9.3f} {dm_p:>8.4f} {w_p:>8.4f}  {stars:>5}")

    print("\n  * p<0.05  ** p<0.01  *** p<0.001  (both tests must agree)")

def benchmark(
    data_path: str,
    n_splits: int = 5,
    use_polar: bool = False,
    save_dir: Optional[str] = None,
    include_t: bool = False,
    grab_min_dist: bool = False,
    min_test_dist: float = 0.1,
    n_bins: int = 5,
    use_log_target: bool = False,
    epsilon: float = 1e-6,
):
    out_path = Path(save_dir) if save_dir is not None else None

    if out_path is not None:
        mode = "polar" if use_polar else "cartesian"
        with_t = "with_t" if include_t else "no_t"
        min_dist = "min_dist" if grab_min_dist else "all_points"
        log_dir = "log_target" if use_log_target else "linear_target"
        out_path = out_path / f"{mode}_{with_t}_{min_dist}_{log_dir}"
        out_path.mkdir(parents=True, exist_ok=True)

    runways, X_all, y_all, runway_labels = load_and_prep_data(
        data_path,
        use_polar=use_polar,
        include_t=include_t,
        grab_min_dist=grab_min_dist,
        use_log_target=use_log_target,
        epsilon=epsilon,
    )

    # Ensure distance bins are constructed based on physical km, not the log scale
    y_all_phys = np.exp(y_all) - epsilon if use_log_target else y_all
    dist_bins, bin_labels = make_dist_bins(y_all_phys, n_bins=n_bins, min_test_dist=min_test_dist)
    
    print(f"📐 Distance bins ({n_bins} equal-frequency): {', '.join(bin_labels)}")

    (
        results_df, per_rwy_df, bin_metrics_df,
        last_fold_preds, best_y_true, best_y_pred, fold_details,
    ) = evaluate_models_cv(
        runways, X_all, y_all, runway_labels,
        n_splits=n_splits,
        min_test_dist=min_test_dist,
        dist_bins=dist_bins,
        bin_labels=bin_labels,
        use_log_target=use_log_target,
        epsilon=epsilon,
    )

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"🏆  BENCHMARK RESULTS  —  dist_to_go target  "
          f"(polar={use_polar}, log_target={use_log_target}, {n_splits}-fold CV)")
    print(f"{'=' * 75}")

    if not results_df.empty:
        display_cols = ["Model", "R2", "MAE (km)", "RMSE (km)", "sMAPE (%)"]
        print(results_df[display_cols].to_string(index=False))

    print(f"\n{'─' * 75}")
    print("📏  Per-distance-bin MAE  (mean ± std across folds):")
    print(f"{'─' * 75}")
    if not bin_metrics_df.empty:
        print(bin_metrics_df.to_string(index=False))

    if not results_df.empty:
        print_statistical_tests(results_df, last_fold_preds, best_y_true)

        best_r2 = results_df["R2_mean"].max()
        tied = results_df[results_df["R2_mean"] >= best_r2 * (1 - TIE_TOLERANCE)]
        if len(tied) > 1:
            print(f"\n⚠️  {len(tied)} models tied best for R² "
                  f"(within {TIE_TOLERANCE:.0%}): "
                  f"{', '.join(tied['Model'].tolist())}")
    print("=" * 75)

    # ── Plots ──────────────────────────────────────────────────────────────
    if not results_df.empty:
        plot_benchmark_metrics(results_df, save_dir=out_path)
        plot_runway_heatmap(per_rwy_df, save_dir=out_path)
        plot_bin_metrics(bin_metrics_df, results_df,
                         bin_labels=bin_labels, save_dir=out_path)

    best_name = results_df.iloc[0]["Model"] if not results_df.empty else None
    if best_name and best_y_true.size > 0 and best_y_pred.size > 0:
        plot_parity_and_residuals(best_y_true, best_y_pred, best_name, save_dir=out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def run_benchmark_cli(experiment_cls=None):
    import argparse
    global TIE_TOLERANCE

    parser = argparse.ArgumentParser(description="Benchmark DTG Samplers (Cross-Validated)")
    parser.add_argument("data",           type=str,   help="Path to data (.csv or .parquet)")
    parser.add_argument("--n_splits",     type=int,   default=5,
                        help="Number of CV folds (default 5)")
    parser.add_argument("--polar",        action="store_true",
                        help="Use polar coordinates for spatial features")
    parser.add_argument("--out",          type=str,   default=None,
                        help="Directory to save artefacts")
    parser.add_argument("--tie_tol",      type=float, default=TIE_TOLERANCE,
                        help="Relative tolerance for declaring a tie (default 0.01 = 1%%)")
    parser.add_argument("--include_t",    action="store_true",
                        help="Include time feature t in the benchmark")
    parser.add_argument("--min_dist",     action="store_true",
                        help="Only keep the minimum dist_to_go per (x, y, runway)")
    parser.add_argument("--min_test_dist", type=float, default=0.1,
                        help="Exclude test points below this dist_to_go in km (default 0.1)")
    parser.add_argument("--n_bins",        type=int,   default=5,
                        help="Number of equal-frequency distance bins for per-bin MAE (default 5)")
    parser.add_argument("--log_target",    action="store_true",
                        help="Apply a log transformation to the target variable to stabilize variance.")
    args = parser.parse_args()

    TIE_TOLERANCE = args.tie_tol
    benchmark(
        args.data,
        n_splits=args.n_splits,
        use_polar=args.polar,
        save_dir=args.out,
        include_t=args.include_t,
        grab_min_dist=args.min_dist,
        min_test_dist=args.min_test_dist,
        n_bins=args.n_bins,
        use_log_target=args.log_target,
    )

if __name__ == "__main__":
    run_benchmark_cli()