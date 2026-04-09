"""
bluesky_gym/experiment/plot.py
---------------------------------
Plotting utilities for training curves and evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy matplotlib import so the module can be imported without a display
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return plt


def _mpl():
    import matplotlib as mpl
    return mpl


# ---------------------------------------------------------------------------
# Colour palette & style
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _apply_style() -> None:
    """Apply a clean, publication-ready style to all subsequent plots."""
    plt = _plt()
    plt.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       "#F8F8F8",
        "axes.edgecolor":       "#CCCCCC",
        "axes.linewidth":       0.8,
        "axes.grid":            True,
        "grid.color":           "white",
        "grid.linewidth":       1.0,
        "grid.linestyle":       "-",
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.color":          "#555555",
        "ytick.color":          "#555555",
        "axes.labelcolor":      "#333333",
        "axes.titleweight":     "bold",
        "axes.titlesize":       11,
        "axes.labelsize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
        "legend.fontsize":      8,
        "figure.dpi":           120,
        "savefig.dpi":          150,
        "savefig.bbox":         "tight",
        "font.family":          "sans-serif",
    })


# ---------------------------------------------------------------------------
# Data loaders & path finders
# ---------------------------------------------------------------------------

def _load_training_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "timestep":     int(row["timestep"]),
                "mean_reward":  float(row["mean_reward"]),
                "std_reward":   _safe_float(row.get("std_reward")),
                "success_rate": _safe_float(row.get("success_rate")),
            })
    return rows


def _safe_float(v: Optional[str]) -> float:
    if v is None or str(v).strip() in ("", "nan", "None"):
        return float("nan")
    return float(v)


def _find_training_csv(run_id: str, base: str = "./experiments") -> str:
    pattern = os.path.join(base, f"*/*/logs/{run_id}/training_evals.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No training_evals.csv for run_id='{run_id}'. "
            f"Was session.track_training_evals=True during training?"
        )
    return matches[0]


def _find_all_training_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    pattern = os.path.join(base, "*/*/logs/*/training_evals.csv")
    results = []
    for path in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(path))
        results.append((run_id, path))
    return results


def _load_merged_csv(path: str) -> tuple[list[str], list[list[dict]]]:
    """Parse a merged comparison CSV back into (run_ids, all_rows)."""
    with open(path, newline="") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        raw_rows   = list(reader)

    run_ids: list[str] = []
    seen: set[str] = set()
    for col in fieldnames:
        if "__mean_reward" in col:
            rid = col.replace("__mean_reward", "")
            if rid not in seen:
                run_ids.append(rid)
                seen.add(rid)

    if not run_ids:
        raise ValueError(f"No '<run_id>__mean_reward' columns found in {path}.")

    all_rows: list[list[dict]] = []
    for rid in run_ids:
        rows: list[dict] = []
        for raw in raw_rows:
            mean_r = raw.get(f"{rid}__mean_reward", "").strip()
            if mean_r == "":
                continue
            rows.append({
                "timestep":     int(raw["timestep"]),
                "mean_reward":  float(mean_r),
                "std_reward":   _safe_float(raw.get(f"{rid}__std_reward")),
                "success_rate": _safe_float(raw.get(f"{rid}__success_rate")),
            })
        all_rows.append(rows)

    return run_ids, all_rows


def _find_eval_files(run_id: str, extension: str, base: str = "./experiments") -> list[str]:
    pattern = os.path.join(base, f"*/*/models/{run_id}/eval_{run_id}_*.{extension}")
    matches = glob.glob(pattern)
    if not matches:
        print(f"⚠️  No eval {extension} files found for run_id='{run_id}'")
    return sorted(matches)


def _load_eval_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _load_eval_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "episode":      int(row["episode"]),
                "group":        row["group"],
                "is_success":   row["is_success"].lower() in ("true", "1", "yes"),
                "total_reward": float(row["total_reward"]),
            })
    return rows


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def _smooth(vals: list[float] | np.ndarray, w: int) -> np.ndarray:
    arr = np.array(vals, dtype=float)
    if w <= 1:
        return arr
    kernel = np.ones(w) / w
    # Use 'valid' convolution and pad back to original length
    padded = np.pad(arr, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _shade_std(
    ax,
    steps:   np.ndarray,
    rewards: np.ndarray,
    stds:    np.ndarray,
    color:   str,
    smooth_w: int = 1,
) -> None:
    """Draw a ±1 std shaded band around the reward curve."""
    valid = ~np.isnan(stds)
    if not valid.any():
        return
    s = steps[valid]
    r = _smooth(rewards[valid], smooth_w)
    d = _smooth(stds[valid],    smooth_w)
    ax.fill_between(s, r - d, r + d, color=color, alpha=0.15, linewidth=0)


# ---------------------------------------------------------------------------
# Training curve plots
# ---------------------------------------------------------------------------

def plot_training_curves(
    labels:  list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    smooth:   int           = 1,
    title:    Optional[str] = None,
) -> None:
    """
    Plot mean reward (+ ±1 std band) and success rate curves for each run,
    with best-checkpoint markers and a convergence annotation.
    """
    _apply_style()
    plt = _plt()
    mpl = _mpl()

    has_success = any(
        not math.isnan(r["success_rate"])
        for rows in all_rows for r in rows
    )
    n_panels = 2 if has_success else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7 * n_panels, 4.5), squeeze=False
    )
    ax_rew = axes[0, 0]
    ax_sr  = axes[0, 1] if has_success else None

    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        if not rows:
            continue
        color = _color(i)
        steps   = np.array([r["timestep"]    for r in rows])
        rewards = np.array([r["mean_reward"] for r in rows], dtype=float)
        stds    = np.array([r["std_reward"]  for r in rows], dtype=float)

        rew_s = _smooth(rewards, smooth)

        # ── Reward panel ────────────────────────────────────────────────────
        ax_rew.plot(steps, rew_s, color=color, label=label, linewidth=2.0, zorder=3)
        _shade_std(ax_rew, steps, rewards, stds, color, smooth_w=smooth)

        # Best-checkpoint star marker
        best_idx = int(np.argmax(rew_s))
        ax_rew.scatter(
            steps[best_idx], rew_s[best_idx],
            color=color, s=80, zorder=5, marker="*",
            edgecolors="white", linewidths=0.5,
        )

        # ── Success rate panel ───────────────────────────────────────────────
        if ax_sr is not None:
            sr = np.array([r["success_rate"] for r in rows], dtype=float)
            valid = ~np.isnan(sr)
            if valid.any():
                sr_s = _smooth(sr[valid], smooth)
                ax_sr.plot(
                    steps[valid], sr_s,
                    color=color, label=label, linewidth=2.0, zorder=3,
                )
                ax_sr.fill_between(
                    steps[valid], 0, sr_s,
                    color=color, alpha=0.08, linewidth=0,
                )

    # Axes decoration — reward
    ax_rew.set_title(title or "Mean Reward")
    ax_rew.set_xlabel("Environment Steps")
    ax_rew.set_ylabel("Mean Reward")
    ax_rew.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
    )
    ax_rew.legend(title="Run", title_fontsize=8)

    # Axes decoration — success rate
    if ax_sr is not None:
        ax_sr.set_title("Success Rate")
        ax_sr.set_xlabel("Environment Steps")
        ax_sr.set_ylabel("Success Rate")
        ax_sr.set_ylim(0, 1.05)
        ax_sr.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax_sr.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
        )
        ax_sr.legend(title="Run", title_fontsize=8)

    _add_star_note(fig)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "training_curves.png", plt)


def plot_comparison_grid(
    labels:  list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    smooth:   int           = 1,
    title:    Optional[str] = None,
) -> None:
    """
    2 x 2 grid giving a richer view of the comparison:
      [0,0] Mean reward curves + std bands
      [0,1] Success rate curves
      [1,0] Reward std over time  (measures policy consistency)
      [1,1] Rolling gap: reward_A - reward_B  (2-run mode only; else skipped)
    """
    _apply_style()
    plt = _plt()
    mpl = _mpl()

    has_success = any(
        not math.isnan(r["success_rate"])
        for rows in all_rows for r in rows
    )
    two_runs = len(labels) == 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_rew, ax_sr   = axes[0, 0], axes[0, 1]
    ax_std, ax_diff = axes[1, 0], axes[1, 1]

    smoothed_rewards: list[tuple[np.ndarray, np.ndarray]] = []

    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        if not rows:
            smoothed_rewards.append((np.array([]), np.array([])))
            continue
        color = _color(i)
        steps   = np.array([r["timestep"]    for r in rows])
        rewards = np.array([r["mean_reward"] for r in rows], dtype=float)
        stds    = np.array([r["std_reward"]  for r in rows], dtype=float)
        sr      = np.array([r["success_rate"] for r in rows], dtype=float)

        rew_s = _smooth(rewards, smooth)
        smoothed_rewards.append((steps, rew_s))

        # [0,0] Reward + std band
        ax_rew.plot(steps, rew_s, color=color, label=label, linewidth=2.0, zorder=3)
        _shade_std(ax_rew, steps, rewards, stds, color, smooth_w=smooth)
        best_idx = int(np.argmax(rew_s))
        ax_rew.scatter(
            steps[best_idx], rew_s[best_idx],
            color=color, s=80, zorder=5, marker="*",
            edgecolors="white", linewidths=0.5,
        )

        # [0,1] Success rate
        if has_success:
            valid = ~np.isnan(sr)
            if valid.any():
                sr_s = _smooth(sr[valid], smooth)
                ax_sr.plot(steps[valid], sr_s, color=color, label=label, linewidth=2.0)
                ax_sr.fill_between(steps[valid], 0, sr_s, color=color, alpha=0.08, linewidth=0)

        # [1,0] Reward std over time
        valid_std = ~np.isnan(stds)
        if valid_std.any():
            std_s = _smooth(stds[valid_std], smooth)
            ax_std.plot(steps[valid_std], std_s, color=color, label=label, linewidth=1.8, linestyle="--")

    # [1,1] Rolling reward gap (run 0 - run 1)
    if two_runs:
        s0, r0 = smoothed_rewards[0]
        s1, r1 = smoothed_rewards[1]
        if len(s0) and len(s1):
            # Interpolate both to a common timestep grid
            common = np.intersect1d(s0, s1)
            if len(common) > 1:
                idx0 = np.isin(s0, common)
                idx1 = np.isin(s1, common)
                gap  = r0[idx0] - r1[idx1]
                gap_s = _smooth(gap, smooth)
                ax_diff.axhline(0, color="#AAAAAA", linewidth=1.0, linestyle=":")
                ax_diff.fill_between(
                    common, gap_s, 0,
                    where=gap_s >= 0, color=_color(0), alpha=0.25, linewidth=0,
                    label=f"{labels[0]} ahead",
                )
                ax_diff.fill_between(
                    common, gap_s, 0,
                    where=gap_s <= 0,  color=_color(1), alpha=0.25, linewidth=0,
                    label=f"{labels[1]} ahead",
                )
                ax_diff.plot(common, gap_s, color="#444444", linewidth=1.5)
                ax_diff.set_title(f"Reward Gap  ({labels[0]} - {labels[1]})")
                ax_diff.set_xlabel("Environment Steps")
                ax_diff.set_ylabel("Δ Mean Reward")
                ax_diff.xaxis.set_major_formatter(
                    mpl.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
                )
                ax_diff.legend(title="Run",fontsize=8)
            else:
                ax_diff.text(0.5, 0.5, "No overlapping timesteps",
                             ha="center", va="center", transform=ax_diff.transAxes,
                             color="#888888")
        else:
            ax_diff.set_visible(False)
    else:
        # More than 2 runs: reuse panel for a reward range band
        all_steps = sorted({r["timestep"] for rows in all_rows for r in rows})
        if all_steps:
            step_arr = np.array(all_steps)
            all_rew_at_step = []
            for rows in all_rows:
                idx_map = {r["timestep"]: r["mean_reward"] for r in rows}
                all_rew_at_step.append([idx_map.get(ts, np.nan) for ts in all_steps])
            mat = np.array(all_rew_at_step, dtype=float)
            lo  = np.nanmin(mat, axis=0)
            hi  = np.nanmax(mat, axis=0)
            mid = np.nanmean(mat, axis=0)
            ax_diff.fill_between(step_arr, lo, hi, color="#888888", alpha=0.2, label="min–max range")
            ax_diff.plot(step_arr, mid, color="#444444", linewidth=1.8, label="mean across runs")
            ax_diff.set_title("Reward Range Across Runs")
            ax_diff.set_xlabel("Environment Steps")
            ax_diff.set_ylabel("Mean Reward")
            ax_diff.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))
            )
            ax_diff.legend(title="Run",fontsize=8)

    # Axes decoration
    fmt_x = mpl.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else str(int(x)))

    ax_rew.set_title("Mean Reward (± 1 std)")
    ax_rew.set_xlabel("Environment Steps")
    ax_rew.set_ylabel("Mean Reward")
    ax_rew.xaxis.set_major_formatter(fmt_x)
    ax_rew.legend(title="Run", title_fontsize=8)

    if has_success:
        ax_sr.set_title("Success Rate")
        ax_sr.set_xlabel("Environment Steps")
        ax_sr.set_ylabel("Success Rate")
        ax_sr.set_ylim(0, 1.05)
        ax_sr.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax_sr.xaxis.set_major_formatter(fmt_x)
        ax_sr.legend(title="Run", title_fontsize=8)
    else:
        ax_sr.set_visible(False)

    ax_std.set_title("Reward Std Dev Over Time  (policy consistency)")
    ax_std.set_xlabel("Environment Steps")
    ax_std.set_ylabel("Std Dev of Reward")
    ax_std.xaxis.set_major_formatter(fmt_x)
    ax_std.legend(title="Run", title_fontsize=8)

    _add_star_note(fig)
    fig.suptitle(title or "Training Comparison", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, out_dir, "training_comparison_grid.png", plt)


def _add_star_note(fig) -> None:
    fig.text(
        0.01, -0.01, "★ = best checkpoint",
        fontsize=7, color="#888888", ha="left",
    )

# ---------------------------------------------------------------------------
# Eval plots (unchanged API, improved styling)
# ---------------------------------------------------------------------------

def plot_eval_summary(
    labels:     list[str],
    yaml_data:  list[dict],
    metric:     str           = "success_rate",
    out_dir:    Optional[str] = None,
    title:      Optional[str] = None,
) -> None:
    _apply_style()
    plt = _plt()
    all_groups = sorted({g for d in yaml_data for g in d.get("per_group", {}).keys()})
    if not all_groups:
        return

    n_runs, n_groups = len(labels), len(all_groups)
    bar_w, x = 0.8 / n_runs, np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.2 + 2), 5))

    for i, (label, d) in enumerate(zip(labels, yaml_data)):
        per_group = d.get("per_group", {})
        vals = [per_group.get(g, {}).get(metric, float("nan")) for g in all_groups]
        bars = ax.bar(
            x + (i - (n_runs - 1) / 2) * bar_w, vals,
            width=bar_w * 0.9, color=_color(i), label=label, zorder=3,
        )
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.1%}" if metric == "success_rate" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Eval Summary — {metric.replace('_', ' ').title()}")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, out_dir, "eval_summary.png", plt)


def plot_eval_episodes(
    labels:    list[str],
    all_rows:  list[list[dict]],
    out_dir:   Optional[str] = None,
    title:     Optional[str] = None,
) -> None:
    _apply_style()
    plt = _plt()
    n_runs = len(labels)
    fig, axes = plt.subplots(1, n_runs, figsize=(max(5, 5 * n_runs), 5), squeeze=False, sharey=True)

    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        ax = axes[0, i]
        by_group: dict[str, list[float]] = {}
        for r in rows:
            by_group.setdefault(r["group"], []).append(r["total_reward"])
        groups = sorted(by_group.keys())
        bp = ax.boxplot(
            [by_group[g] for g in groups],
            labels=groups,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )
        for patch, g_idx in zip(bp["boxes"], range(len(groups))):
            patch.set_facecolor(_color(i))
            patch.set_alpha(0.7)
        ax.set_title(label)
        ax.set_xlabel("Group")
        if i == 0:
            ax.set_ylabel("Total Reward")

    fig.suptitle(title or "Eval Episodes by Group", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, out_dir, "eval_episodes.png", plt)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig, out_dir: Optional[str], filename: str, plt) -> None:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

def plot(
    command:      str,
    runs:         list[str] | None = None,
    discover_all: bool = False,
    from_csv:     str | None = None,
    files:        list[str] | None = None,
    run_id:       str | None = None,
    metric:       str = "success_rate",
    labels:       list[str] | None = None,
    out:          str | None = None,
    title:        str | None = None,
    smooth:       int = 1,
    grid:         bool = False,
) -> None:
    """
    Programmatic entry point for plotting.

    Parameters
    ----------
    from_csv : str, optional
        Path to a merged comparison CSV (output of compare_runs).  When
        provided for the 'training' command, run IDs and data are loaded
        directly from it — no individual training_evals.csv files needed.
    grid : bool
        When True and command='training', render the richer 2×2 comparison
        grid instead of the simple side-by-side panels.
    """
    if command == "training":
        # ── Resolve data source ──────────────────────────────────────────────
        if from_csv:
            run_ids, all_rows = _load_merged_csv(from_csv)
            print(f"📂 Loaded {len(run_ids)} run(s) from {from_csv}")
        elif discover_all:
            discovered = _find_all_training_csvs()
            run_ids  = [r for r, _ in discovered]
            all_rows = [_load_training_csv(p) for _, p in discovered]
        else:
            if not runs:
                print("❌ Error: Provide --runs, --all, or --from-csv for the training command.")
                return
            run_ids  = runs
            all_rows = [_load_training_csv(_find_training_csv(r)) for r in run_ids]

        # Map labels
        plot_labels = labels if labels and len(labels) == len(run_ids) else run_ids
        if labels and len(labels) != len(run_ids):
            print(f"⚠️ Warning: Provided {len(labels)} labels for {len(run_ids)} runs. Defaulting to Run IDs.")

        if grid or len(run_ids) > 1:
            plot_comparison_grid(plot_labels, all_rows, out, smooth, title)
        else:
            plot_training_curves(plot_labels, all_rows, out, smooth, title)

    elif command in ["eval-summary", "eval-episodes"]:
        ext = "yaml" if command == "eval-summary" else "csv"
        eval_files = files
        if not eval_files and run_id:
            eval_files = _find_eval_files(run_id, ext)

        if not eval_files:
            print(f"❌ Error: Provide either 'files' or a 'run_id' with existing {ext} files.")
            return

        plot_labels = labels or [os.path.basename(f) for f in eval_files]

        if command == "eval-summary":
            yaml_data = [_load_eval_yaml(f) for f in eval_files]
            plot_eval_summary(plot_labels, yaml_data, metric, out, title)
        else:
            csv_data = [_load_eval_csv(f) for f in eval_files]
            plot_eval_episodes(plot_labels, csv_data, out, title)
    else:
        print(f"❌ Unknown command: {command}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot training curves and evaluation results.")
    sub = p.add_subparsers(dest="command", required=True)

    # ── training subcommand ──────────────────────────────────────────────────
    tr = sub.add_parser("training", help="Plot training reward / success-rate curves.")
    src = tr.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs",     nargs="+", metavar="RUN_ID",
                     help="One or more run IDs.")
    src.add_argument("--all",      action="store_true",
                     help="Auto-discover all runs.")
    src.add_argument("--from-csv", metavar="PATH",
                     help="Load directly from a merged comparison CSV.")
    tr.add_argument("--labels", nargs="+", default=None,
                    help="Custom labels for the legend. Must match the number of runs.")
    tr.add_argument("--smooth", type=int, default=1,
                    help="Rolling-average window size (default: 1 = no smoothing).")
    tr.add_argument("--grid",   action="store_true",
                    help="Render the richer 2×2 comparison grid.")
    tr.add_argument("--out",    type=str, default=None,
                    help="Output directory for saved plots.")
    tr.add_argument("--title",  type=str, default=None)

    # ── eval-summary subcommand ──────────────────────────────────────────────
    es = sub.add_parser("eval-summary", help="Plot evaluation summary data.")
    es.add_argument("--files",  nargs="+", metavar="YAML_PATH")
    es.add_argument("--run-id", type=str)
    es.add_argument("--metric", type=str, default="success_rate")
    es.add_argument("--labels", nargs="+", default=None)
    es.add_argument("--out",    type=str, default=None)
    es.add_argument("--title",  type=str, default=None)

    # ── eval-episodes subcommand ─────────────────────────────────────────────
    ep = sub.add_parser("eval-episodes", help="Plot evaluation episode data.")
    ep.add_argument("--files",  nargs="+", metavar="CSV_PATH")
    ep.add_argument("--run-id", type=str)
    ep.add_argument("--labels", nargs="+", default=None)
    ep.add_argument("--out",    type=str, default=None)
    ep.add_argument("--title",  type=str, default=None)

    return p


def run_plot_cli(experiment_cls=None) -> None:
    """Standalone CLI entry point."""
    args = _build_parser().parse_args()

    if args.command == "training":
        plot(
            command="training",
            runs=getattr(args, "runs", None),
            discover_all=getattr(args, "all", False),
            from_csv=getattr(args, "from_csv", None),
            smooth=args.smooth,
            grid=getattr(args, "grid", False),
            labels=getattr(args, "labels", None),
            out=args.out,
            title=args.title,
        )
    elif args.command in ["eval-summary", "eval-episodes"]:
        plot(
            command=args.command,
            files=args.files,
            run_id=args.run_id,
            metric=getattr(args, "metric", "success_rate"),
            labels=args.labels,
            out=args.out,
            title=args.title,
        )