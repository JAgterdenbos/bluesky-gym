"""
bluesky_gym/experiment/plot.py
---------------------------------
Plotting utilities for training curves and evaluation results.

Produces publication-ready figures using matplotlib.  No additional
dependencies beyond matplotlib and numpy are required.

Figures
-------
  plot_training_curves(run_ids, ...)
      Overlays reward and success-rate curves from one or more
      training_evals.csv files (written by TrainingEvalLogger).
      Shaded ± std bands are included for reward when available.

  plot_eval_summary(yaml_paths, ...)
      Bar chart of per-group success rates from one or more
      eval_<run_id>_<ts>.yaml files (written by run_evaluate_cli).

  plot_eval_csv(csv_paths, ...)
      Scatter / box plots of per-episode rewards and success,
      grouped by the 'group' column in eval_<run_id>_<ts>.csv files.

Usage
-----
  # Compare training curves for two runs
  python plot.py training --runs 20260401_120000 20260401_130000

  # Compare training curves for all discovered runs
  python plot.py training --all

  # Bar chart from eval YAML files
  python plot.py eval-summary --files path/to/eval_run1.yaml path/to/eval_run2.yaml

  # Per-episode scatter from eval CSV files
  python plot.py eval-episodes --files path/to/eval_run1.csv

  # Save to a specific directory instead of showing interactively
  python plot.py training --runs 20260401_120000 --out ./plots/
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

# ---------------------------------------------------------------------------
# Colour palette — 8 visually distinct colours that cycle for multi-run plots
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Data loaders (thin wrappers around compare_runs loaders)
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
    pattern = os.path.join(base, f"*/*/models/{run_id}/training_evals.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No training_evals.csv for run_id='{run_id}'.  "
            f"Was session.track_training_evals=True during training?"
        )
    return matches[0]


def _find_all_training_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    pattern = os.path.join(base, "*/*/*/training_evals.csv")
    results = []
    for path in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(path))
        results.append((run_id, path))
    return results


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
# Figure 1 — Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    run_ids:  list[str],
    all_rows: list[list[dict]],
    out_dir:  Optional[str] = None,
    smooth:   int           = 1,
    title:    Optional[str] = None,
) -> None:
    """Plot overlaid training curves (reward + success rate) for N runs.

    Parameters
    ----------
    run_ids  : list of run identifier strings (used as legend labels).
    all_rows : list of row-lists as returned by _load_training_csv().
    out_dir  : if given, save PNG there instead of showing interactively.
    smooth   : rolling-average window in eval steps (1 = no smoothing).
    title    : optional figure super-title.
    """
    plt = _plt()

    has_success = any(
        not math.isnan(r["success_rate"])
        for rows in all_rows
        for r in rows
    )
    n_panels = 2 if has_success else 1
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(7 * n_panels, 4.5),
        squeeze=False,
    )
    ax_rew = axes[0, 0]
    ax_sr  = axes[0, 1] if has_success else None

    def _smooth(vals: list[float], w: int) -> np.ndarray:
        if w <= 1:
            return np.array(vals, dtype=float)
        kernel = np.ones(w) / w
        return np.convolve(vals, kernel, mode="same")

    for i, (run_id, rows) in enumerate(zip(run_ids, all_rows)):
        if not rows:
            continue
        color  = _color(i)
        label  = run_id
        steps  = np.array([r["timestep"]    for r in rows])
        rew    = np.array([r["mean_reward"]  for r in rows], dtype=float)
        std    = np.array([r["std_reward"]   for r in rows], dtype=float)
        sr     = np.array([r["success_rate"] for r in rows], dtype=float)

        rew_s = _smooth(rew, smooth)
        std_s = _smooth(std, smooth)

        ax_rew.plot(steps, rew_s, color=color, label=label, linewidth=1.8)
        valid_std = ~np.isnan(std_s)
        if valid_std.any():
            ax_rew.fill_between(
                steps[valid_std],
                (rew_s - std_s)[valid_std],
                (rew_s + std_s)[valid_std],
                color=color, alpha=0.15,
            )

        if ax_sr is not None:
            valid_sr = ~np.isnan(sr)
            if valid_sr.any():
                sr_s = _smooth(sr[valid_sr], smooth)
                ax_sr.plot(steps[valid_sr], sr_s, color=color, label=label, linewidth=1.8)

    # Formatting
    ax_rew.set_xlabel("Timestep")
    ax_rew.set_ylabel("Mean Episode Reward")
    ax_rew.set_title("Training Reward")
    ax_rew.legend(fontsize=8, loc="lower right")
    ax_rew.grid(True, alpha=0.3)
    ax_rew.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
    )

    if ax_sr is not None:
        ax_sr.set_xlabel("Timestep")
        ax_sr.set_ylabel("Success Rate")
        ax_sr.set_title("Training Success Rate")
        ax_sr.set_ylim(-0.05, 1.05)
        ax_sr.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_sr.legend(fontsize=8, loc="lower right")
        ax_sr.grid(True, alpha=0.3)
        ax_sr.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}k")
        )

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save_or_show(fig, out_dir, "training_curves.png", plt)


# ---------------------------------------------------------------------------
# Figure 2 — Eval summary bar chart (from YAML)
# ---------------------------------------------------------------------------

def plot_eval_summary(
    labels:     list[str],
    yaml_data:  list[dict],
    metric:     str           = "success_rate",
    out_dir:    Optional[str] = None,
    title:      Optional[str] = None,
) -> None:
    """Bar chart comparing per-group eval metrics across one or more runs.

    Parameters
    ----------
    labels    : one label per YAML file / run (legend / bar group label).
    yaml_data : list of dicts as returned by _load_eval_yaml().
    metric    : which key to plot from per_group dicts (default: success_rate).
    out_dir   : if given, save PNG there.
    title     : optional figure title.
    """
    plt = _plt()

    # Collect all groups present across all runs
    all_groups: list[str] = []
    for d in yaml_data:
        for g in d.get("per_group", {}).keys():
            if g not in all_groups:
                all_groups.append(g)
    all_groups = sorted(all_groups)

    n_runs   = len(labels)
    n_groups = len(all_groups)
    if n_groups == 0:
        print("No per-group data found in YAML files.")
        return

    bar_w    = 0.8 / n_runs
    x        = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.2 + 2), 5))

    for i, (label, d) in enumerate(zip(labels, yaml_data)):
        per_group = d.get("per_group", {})
        vals = [
            per_group.get(g, {}).get(metric, float("nan"))
            for g in all_groups
        ]
        offset = (i - (n_runs - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, vals,
            width=bar_w * 0.9,
            color=_color(i),
            label=label,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Value labels on top of each bar
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.1%}" if metric == "success_rate" else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_groups, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    if metric == "success_rate":
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title(title or f"Evaluation: {metric.replace('_', ' ').title()} per Group")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Overall success rate as a horizontal dashed line (first run only)
    overall = yaml_data[0].get("overall", {}).get(metric)
    if overall is not None and not math.isnan(float(overall)):
        ax.axhline(
            float(overall), color=_color(0),
            linestyle="--", linewidth=1.2, alpha=0.6,
            label=f"{labels[0]} overall",
        )

    fig.tight_layout()
    _save_or_show(fig, out_dir, "eval_summary.png", plt)


# ---------------------------------------------------------------------------
# Figure 3 — Per-episode scatter / box plot (from eval CSV)
# ---------------------------------------------------------------------------

def plot_eval_episodes(
    labels:    list[str],
    all_rows:  list[list[dict]],
    out_dir:   Optional[str] = None,
    title:     Optional[str] = None,
) -> None:
    """Box plots of per-episode reward grouped by episode group, one panel per run.

    Parameters
    ----------
    labels   : one label per CSV / run.
    all_rows : list of row-lists as returned by _load_eval_csv().
    out_dir  : if given, save PNG there.
    title    : optional figure super-title.
    """
    plt = _plt()

    n_runs  = len(labels)
    fig, axes = plt.subplots(
        1, n_runs,
        figsize=(max(5, 5 * n_runs), 5),
        squeeze=False,
        sharey=True,
    )

    for i, (label, rows) in enumerate(zip(labels, all_rows)):
        ax = axes[0, i]

        # Group rows
        by_group: dict[str, list[float]] = {}
        for r in rows:
            by_group.setdefault(r["group"], []).append(r["total_reward"])

        groups  = sorted(by_group.keys())
        data    = [by_group[g] for g in groups]

        # Box plot
        bp = ax.boxplot(
            data,
            labels=groups,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="grey"),
            capprops=dict(color="grey"),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(_color(i))
            patch.set_alpha(0.75)

        # Overlay success-rate annotation
        by_group_success: dict[str, list[bool]] = {}
        for r in rows:
            by_group_success.setdefault(r["group"], []).append(r["is_success"])

        for j, g in enumerate(groups, start=1):
            outcomes = by_group_success.get(g, [])
            if outcomes:
                sr = sum(outcomes) / len(outcomes)
                ax.text(
                    j, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                    f"{sr:.0%}✓",
                    ha="center", va="bottom", fontsize=8,
                    color=_color(i),
                )

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Group")
        if i == 0:
            ax.set_ylabel("Episode Reward")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    fig.tight_layout()
    _save_or_show(fig, out_dir, "eval_episodes.png", plt)


# ---------------------------------------------------------------------------
# Shared helper
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
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot training curves and evaluation results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── training curves ──────────────────────────────────────────────────
    tr = sub.add_parser("training", help="Plot training reward / success curves.")
    src = tr.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", nargs="+", metavar="RUN_ID",
                     help="One or more run IDs to compare.")
    src.add_argument("--all", action="store_true",
                     help="Auto-discover all runs with training_evals.csv.")
    tr.add_argument("--smooth", type=int, default=1, metavar="W",
                    help="Rolling-average window (in eval checkpoints).")
    tr.add_argument("--out",   type=str, default=None, metavar="DIR",
                    help="Output directory for PNG (omit to show interactively).")
    tr.add_argument("--title", type=str, default=None)

    # ── eval summary (YAML) ──────────────────────────────────────────────
    es = sub.add_parser("eval-summary", help="Bar chart from eval YAML files.")
    es.add_argument("--files", nargs="+", required=True, metavar="YAML_PATH")
    es.add_argument("--metric", type=str, default="success_rate",
                    help="Which metric to plot (e.g. success_rate, mean_total_reward).")
    es.add_argument("--labels", nargs="+", default=None,
                    help="Optional labels; defaults to filenames.")
    es.add_argument("--out",   type=str, default=None, metavar="DIR")
    es.add_argument("--title", type=str, default=None)

    # ── eval episodes (CSV) ──────────────────────────────────────────────
    ep = sub.add_parser("eval-episodes", help="Box plots from per-episode eval CSVs.")
    ep.add_argument("--files", nargs="+", required=True, metavar="CSV_PATH")
    ep.add_argument("--labels", nargs="+", default=None,
                    help="Optional labels; defaults to filenames.")
    ep.add_argument("--out",   type=str, default=None, metavar="DIR")
    ep.add_argument("--title", type=str, default=None)

    return p


def main() -> None:
    p    = _build_parser()
    args = p.parse_args()

    if args.command == "training":
        if args.all:
            discovered = _find_all_training_csvs()
            if not discovered:
                print("No training_evals.csv files found under ./experiments/")
                return
            run_ids   = [r for r, _ in discovered]
            csv_paths = [c for _, c in discovered]
        else:
            run_ids   = args.runs
            csv_paths = [_find_training_csv(r) for r in run_ids]

        all_rows = [_load_training_csv(p) for p in csv_paths]
        print(f"Loaded {len(run_ids)} run(s).")
        plot_training_curves(
            run_ids, all_rows,
            out_dir=args.out,
            smooth=args.smooth,
            title=args.title,
        )

    elif args.command == "eval-summary":
        labels    = args.labels or [os.path.basename(f) for f in args.files]
        yaml_data = [_load_eval_yaml(f) for f in args.files]
        plot_eval_summary(
            labels, yaml_data,
            metric=args.metric,
            out_dir=args.out,
            title=args.title,
        )

    elif args.command == "eval-episodes":
        labels   = args.labels or [os.path.basename(f) for f in args.files]
        all_rows = [_load_eval_csv(f) for f in args.files]
        plot_eval_episodes(
            labels, all_rows,
            out_dir=args.out,
            title=args.title,
        )


if __name__ == "__main__":
    main()