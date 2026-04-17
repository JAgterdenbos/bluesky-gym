import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .samplers import SamplerRegistry

from typing import Optional

# ── Global Plotting Configuration ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,          # Ensures crisp rendering
    'axes.spines.top': False,   # Clean, modern look
    'axes.spines.right': False,
    'axes.axisbelow': True,     # Grid lines go behind plot elements
})

# ── Tie tolerance: models within this fraction of the best are co-winners ─────
TIE_TOLERANCE = 0.01   # 1% relative gap


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE).

    Preferred over standard MAPE because it is stable when y_true approaches
    zero (common in RTA remaining tasks) and is bounded to [0 %, 200 %].
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / (denominator + epsilon)) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return the standard regression metric dict for an array pair."""
    return {
        "R2":        r2_score(y_true, y_pred),
        "MAE":       mean_absolute_error(y_true, y_pred),
        "RMSE":      float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE (%)": smape(y_true, y_pred),
    }


def load_and_prep_data(
    filepath: str,
    test_size: float = 0.2,
    random_state: int = 42,
    use_polar: bool = False,
):
    """Load data, calculate rta_remaining, and optionally convert to polar coords."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"📦 Loading data from {filepath} (Polar={use_polar})...")
    df = pd.read_csv(filepath) if path.suffix == ".csv" else pd.read_parquet(filepath)

    df["rta_remaining"] = df["rta"] - df["t"]

    if use_polar:
        df["r"]   = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
        df["phi"] = np.arctan2(df["y"], df["x"])
        feature_cols = ["r", "phi"]
    else:
        feature_cols = ["x", "y"]

    runways = df["runway"].unique().tolist()
    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []

    for rwy in runways:
        df_rwy = df[df["runway"] == rwy]
        X = df_rwy[feature_cols].values
        y = df_rwy["rta_remaining"].values
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train_list.append(X_tr);  y_train_list.append(y_tr)
        X_test_list.append(X_te);   y_test_list.append(y_te)

    print(f"✅ Loaded {len(df)} samples across {len(runways)} runways.")
    return runways, X_train_list, y_train_list, X_test_list, y_test_list


def _save_fig(fig, save_dir: Path | None, name: str):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / name
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved → {save_path}")


def _is_tied_best(value: float, best: float, lower_is_better: bool) -> bool:
    """True when *value* is within TIE_TOLERANCE of *best*."""
    if best == 0:
        return np.isclose(value, best)
    relative_gap = abs(value - best) / abs(best)
    if lower_is_better:
        return value <= best * (1 + TIE_TOLERANCE)
    else:
        return value >= best * (1 - TIE_TOLERANCE)


def evaluate_models(runways, X_train, y_train, X_test, y_test):
    """
    Fit all registered samplers and evaluate per-runway + global metrics.
    """
    global_rows   = []
    per_rwy_rows  = {}          
    best_r2       = -float("inf")
    best_sampler, best_sampler_name = None, ""
    best_y_true, best_y_pred        = [], []

    available_samplers = SamplerRegistry.list_available()
    print(f"\n🚀 Evaluating {len(available_samplers)} samplers...\n")

    for name in available_samplers:

        try:
            print(f"  Fitting {name:<30}", end="", flush=True)
            sampler = SamplerRegistry.make(name)
            sampler.fit(X_train, y_train, runways)

            y_true_all, y_pred_all = [], []
            rwy_r2 = {}

            for i, rwy in enumerate(runways):
                y_pred = sampler.sample(rwy, X_test[i])
                y_true_all.extend(y_test[i])
                y_pred_all.extend(y_pred)
                rwy_r2[rwy] = r2_score(y_test[i], y_pred)

            y_true_all = np.array(y_true_all)
            y_pred_all = np.array(y_pred_all)

            metrics        = _compute_metrics(y_true_all, y_pred_all)
            per_rwy_rows[name] = rwy_r2

            rwy_r2_vals = np.array(list(rwy_r2.values()))
            global_rows.append({
                "Model":           name,
                **metrics,
                "R2 std (runways)": float(np.std(rwy_r2_vals)),
            })

            print(f"[Done] R2: {metrics['R2']:.4f}  (runway std: {np.std(rwy_r2_vals):.4f})")

            if metrics["R2"] > best_r2:
                best_r2, best_sampler, best_sampler_name = metrics["R2"], sampler, name
                best_y_true, best_y_pred = y_true_all, y_pred_all

        except Exception as e:
            print(f"[FAILED] → {e}")

    results_df  = pd.DataFrame(global_rows).sort_values(by="R2", ascending=False)
    per_rwy_df  = pd.DataFrame(per_rwy_rows).T          
    per_rwy_df  = per_rwy_df.loc[results_df["Model"]]   

    return results_df, per_rwy_df, best_sampler, best_sampler_name, best_y_true, best_y_pred


# ── Colour helpers ─────────────────────────────────────────────────────────────

_PALETTE = {
    "best":  "#3266ad",   # single best
    "tied":  "#5d9fd8",   # tied within TIE_TOLERANCE
    "other": "#9db5c8",   # the rest
}

def _bar_colors(values: pd.Series, lower_is_better: bool) -> list[str]:
    """Return a colour for each bar based on tie status."""
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


# ── Plot: per-metric horizontal bar charts ─────────────────────────────────────

def plot_benchmark_metrics(results_df: pd.DataFrame, save_dir: Path | None = None):
    """
    Four horizontal bar charts (one per metric), each sorted by that metric,
    with gold / blue / grey colouring for best / tied / other.
    """
    metrics = [
        ("R2",        "R² score",                          False, "%.4f"),
        ("MAE",       "Mean Absolute Error",               True,  "%.4f"),
        ("RMSE",      "Root Mean Squared Error",           True,  "%.4f"),
        ("sMAPE (%)", "Symmetric MAPE %",                  True,  "%.1f"),
    ]

    n_models = len(results_df)
    bar_height = 0.45
    fig_h_per_metric = max(2.5, n_models * 0.42 + 1.2)
    fig, axes = plt.subplots(2, 2, figsize=(16, fig_h_per_metric * 2))
    axes = axes.flatten()

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=_PALETTE["best"],  label="Best"),
        plt.Rectangle((0, 0), 1, 1, color=_PALETTE["tied"],  label=f"Tied best (±{TIE_TOLERANCE:.0%})"),
        plt.Rectangle((0, 0), 1, 1, color=_PALETTE["other"], label="Other"),
    ]

    for ax, (col, title, asc, fmt) in zip(axes, metrics):
        df_s  = results_df.sort_values(by=col, ascending=asc).reset_index(drop=True)
        colors = _bar_colors(df_s[col], lower_is_better=asc)

        bars = ax.barh(
            df_s["Model"], df_s[col],
            color=colors, edgecolor="white", linewidth=0.5, height=bar_height,
        )

        # Matplotlib 3.4+ native bar labels (cleaner & prevents clipping)
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=8.5, color="#444")

        ax.set_title(title, pad=8)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5, color="#ccc")

        # Pad right so value labels don't clip
        x_min, x_max = ax.get_xlim()
        ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.12)

    fig.legend(
        handles=legend_patches, loc="lower center",
        ncol=3, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("Benchmark — model comparison", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_dir, "benchmark_metrics.png")
    plt.show()


# ── Plot: per-runway R² heatmap ───────────────────────────────────────────────

def plot_runway_heatmap(per_rwy_df: pd.DataFrame, save_dir: Path | None = None):
    """
    Heatmap of per-runway R² scores. Pure Matplotlib implementation.
    """
    if per_rwy_df.empty:
        return

    data   = per_rwy_df.values.astype(float)
    models = list(per_rwy_df.index)
    runways = list(per_rwy_df.columns)

    fig, ax = plt.subplots(figsize=(max(6, len(runways) * 1.4), max(4, len(models) * 0.6 + 1.5)))

    vmin = max(0, data.min() - 0.05)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rta", ["#d94f3d", "#f5c342", "#3aad6e"]
    )
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=1.0)

    # Per-runway best — outline the winning cell(s)
    col_best = data.max(axis=0)
    for ri in range(len(models)):
        for ci in range(len(runways)):
            v = data[ri, ci]
            txt_col = "white" if v < (vmin + 1.0) / 2 else "#1a1a1a"
            ax.text(ci, ri, f"{v:.4f}", ha="center", va="center",
                    fontsize=9, color=txt_col, fontweight="bold")

    ax.set_xticks(range(len(runways)));  ax.set_xticklabels(runways)
    ax.set_yticks(range(len(models)));   ax.set_yticklabels(models)
    ax.set_title("Per-runway R² (outlined = best or tied per runway)", pad=10)
    
    # Hide grid for heatmap
    ax.grid(False)
    
    plt.colorbar(im, ax=ax, label="R²", fraction=0.03, pad=0.02)
    plt.tight_layout()
    _save_fig(fig, save_dir, "runway_heatmap.png")
    plt.show()


# ── Plot: parity + residuals ──────────────────────────────────────────────────

def plot_parity_and_residuals(
    y_true: np.ndarray, y_pred: np.ndarray,
    model_name: str, save_dir: Path | None = None,
):
    residuals = y_pred - y_true
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # — Parity ——————————————————————————————————————————————
    ax1.scatter(y_true, y_pred, alpha=0.1, edgecolors="none",
                color="#3266ad", s=3, rasterized=True)
    
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction (y = x)")
    ax1.set_title(f"Parity plot — {model_name}")
    ax1.set_xlabel("Actual RTA remaining")
    ax1.set_ylabel("Predicted RTA remaining")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)

    # — Residuals ————————————————————————————————————————————
    ax2.hist(residuals, bins=80, color="#5d9fd8", edgecolor="white",
             linewidth=0.4, density=True, alpha=0.85, label="Residuals")
    ax2.axvline(residuals.mean(), color="#e67e22", linestyle="-", lw=1.2,
                label=f"Mean = {residuals.mean():.4f}")

    # KDE overlay with parameter extraction
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residuals)
        kde_x = np.linspace(residuals.min(), residuals.max(), 300)
        
        # Extracting parameters:
        # factor: The bandwidth factor (Scott/Silverman rule result)
        # n: Number of data points used
        bw_val = kde.factor
        n_points = kde.n
        
        ax2.plot(kde_x, kde(kde_x), color="#1a4a80", lw=1.8, alpha=0.8, 
                 label=f"KDE (BW={bw_val:.3f}, N={n_points})")
    except ImportError:
        pass

    bias_note = f"Bias (mean residual) = {residuals.mean():.4f}"
    std_note  = f"Std = {residuals.std():.4f}"
    ax2.set_title(f"Residual distribution — {model_name}\n{bias_note}   {std_note}")
    ax2.set_xlabel("Residual (predicted - actual)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    _save_fig(fig, save_dir, f"parity_residuals_{model_name}.png")
    plt.show()


# ── Core benchmark orchestration ──────────────────────────────────────────────

def benchmark(
    data_path: str,
    test_size: float = 0.2,
    use_polar: bool = False,
    save_dir: Optional[str] = None,
):
    out_path = Path(save_dir) if save_dir is not None else None

    # 1. Prepare data
    runways, X_train, y_train, X_test, y_test = load_and_prep_data(
        data_path, test_size, use_polar=use_polar
    )

    # 2. Run evaluation
    results_df, per_rwy_df, best_sampler, best_name, best_y_true, best_y_pred = (
        evaluate_models(runways, X_train, y_train, X_test, y_test)
    )

    # 3. Print summary with ties flagged
    print(f"\n{'=' * 65}")
    print(f"🏆  BENCHMARK RESULTS  (Polar={use_polar})")
    print(f"{'=' * 65}")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_r2 = results_df["R2"].max()
    tied = results_df[results_df["R2"] >= best_r2 * (1 - TIE_TOLERANCE)]
    if len(tied) > 1:
        print(f"\n⚠️  {len(tied)} models are tied best for R² "
              f"(within {TIE_TOLERANCE:.0%}): {', '.join(tied['Model'].tolist())}")
    print("=" * 65)

    # 4. Visualisations
    if not results_df.empty:
        plot_benchmark_metrics(results_df, save_dir=out_path)
        plot_runway_heatmap(per_rwy_df, save_dir=out_path)

    if best_sampler is not None:
        plot_parity_and_residuals(best_y_true, best_y_pred, best_name, save_dir=out_path)

# ── CLI ────────────────────────────────────────────────────────────────────────

def run_benchmark_cli(experiment_cls):
    import argparse
    global TIE_TOLERANCE

    parser = argparse.ArgumentParser(description="Benchmark RTA Samplers")
    parser.add_argument("data",         type=str,   help="Path to data (.csv or .parquet)")
    parser.add_argument("--test_size",  type=float, default=0.2,  help="Fraction of data for testing")
    parser.add_argument("--polar",      action="store_true",       help="Use polar coordinates")
    parser.add_argument("--out",        type=str,   default=None,  help="Directory to save artefacts")
    parser.add_argument("--tie_tol",    type=float, default=TIE_TOLERANCE,
                        help="Relative tolerance for declaring a tie (default 0.01 = 1%)")
    args = parser.parse_args()

    TIE_TOLERANCE = args.tie_tol

    benchmark(args.data, args.test_size, args.polar, args.out)


if __name__ == "__main__":
    run_benchmark_cli(None)