"""
bluesky_gym/experiment/compare_runs.py
----------------------------------------
Load and compare training evaluation curves OR post-training evaluation
YAML summaries from multiple runs.

Training Mode (default)
-----------------------
Each run must have been trained with session.track_training_evals = True,
which writes training_evals.csv to the model save path.

Metrics Explained
-----------------
  * Final Metrics (final_mean_reward, final_success_rate): Performance at the very last evaluation checkpoint.
  * Peak Metrics (best_mean_reward, peak_success_rate): The absolute highest evaluation score achieved during the entire training run. best_at_timestep indicates when this peak occurred.
  * Convergence (steps_to_80pct_peak, steps_to_90pct_peak): The first timestep where the mean reward crossed 80% or 90% of its peak reward. Measures how quickly the model learns.
  * Tail Stability (tail_reward_cv, tail_mean_success_rate): Evaluates the final portion of training (controlled by --tail, default 20%). The Coefficient of Variation (CV) measures reward variance (lower is more stable), while the tail mean success rate shows average late-stage reliability.
  * Sample Efficiency (auc_mean_reward): The Area Under the Curve (AUC) for the reward over time, normalised by total timesteps. Higher AUC indicates the model learned faster and spent more of its training time at higher rewards.
  * Late-Stage Regression (peak_to_final_drop): The difference between the peak reward and the final reward. Evaluates if the model degraded or "forgot" how to succeed at the end of training.

Evaluation Mode (--eval)
-------------------------
Loads the eval_<run_id>_<timestamp>.yaml files produced by evaluate.py and
compares overall + per-group metrics across runs.

Ranking
-------
Pass a RankSpec dict to compare_evaluations() to control how each metric is
ranked. Each entry maps a metric name to a callable that takes a list of
(run_id, value) pairs and returns the best value (e.g. max, min, or a custom
function). The CLI --rank flag accepts comma-separated "metric:direction"
pairs where direction is "max" or "min".

  RankSpec = dict[str, Callable[[list[tuple[str, float]]], float]]

  Example:
    rank_spec = {
        "success_rate":      max,
        "mean_total_reward": max,
        "my_penalty_metric": min,
    }

Output
------
  Console  - per-run summary table (final checkpoint metrics)
  Console  - convergence & stability analysis table
  Console  - per-timestep comparison table (optional, --full)
  Console  - evaluation comparison + ranking table (--eval mode)
  CSV      - merged comparison table → comparison_<timestamp>.csv
  CSV      - evaluation ranking table → eval_comparison_<timestamp>.csv

Usage (training)
----------------
  python compare_runs.py --runs 20260401_120000 20260401_130000
  python compare_runs.py --all
  python compare_runs.py --from-csv ./experiments/comparison_20260401_120000.csv
  python compare_runs.py --all --full
  python compare_runs.py --all --out ./results/comparison.csv
  python compare_runs.py --all --tail 0.15

Usage (evaluation)
------------------
  # Compare latest eval YAML for each run (default metrics ranked by max)
  python compare_runs.py --eval --runs 20260401_120000 20260401_130000

  # Compare all discovered eval YAMLs
  python compare_runs.py --eval --all

  # Override rank direction per metric
  python compare_runs.py --eval --runs run_a run_b --rank success_rate:max,my_penalty:min

  # Show per-group breakdown
  python compare_runs.py --eval --runs run_a run_b --groups
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from datetime import datetime
from typing import Callable, Optional

import numpy as np

# RankSpec maps metric_name -> a callable that accepts a list[tuple[run_id, value]]
# and returns the *best* value among them (e.g. built-in max / min, or any custom fn).
# The winner is the run whose value equals the return value of the callable.
#   Example: {"success_rate": max, "my_penalty": min}
RankSpec = dict[str, Callable[[list[tuple[str, float]]], float]]


# ---------------------------------------------------------------------------
# Loading — individual training_evals.csv files
# ---------------------------------------------------------------------------

def find_all_training_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    """Return [(run_id, csv_path), ...] for every training_evals.csv found."""
    pattern = os.path.join(base, "*/*/logs/*/training_evals.csv")
    results = []
    for path in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(path))
        results.append((run_id, path))
    return results


def find_training_csv(run_id: str, base: str = "./experiments") -> str:
    """Locate training_evals.csv for a specific run_id."""
    pattern = os.path.join(base, f"*/*/logs/{run_id}/training_evals.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No training_evals.csv found for run_id='{run_id}'.\n"
            f"Make sure the run used track_training_evals=true and has finished."
        )
    return matches[0]


def load_training_csv(path: str) -> list[dict]:
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


# ---------------------------------------------------------------------------
# Loading — previously saved merged comparison CSV
# ---------------------------------------------------------------------------

def load_merged_csv(path: str) -> tuple[list[str], list[list[dict]]]:
    """
    Parse a merged comparison CSV (as written by save_merged_csv) back into
    the canonical (run_ids, all_rows) format used everywhere else.

    The merged CSV has columns:
        timestep, <run_id>__mean_reward, <run_id>__std_reward, <run_id>__success_rate, ...

    Returns
    -------
    run_ids  : list of run_id strings in column order
    all_rows : list of per-run row-lists, each entry matching load_training_csv output
    """
    with open(path, newline="") as f:
        reader   = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {path}")
        fieldnames = list(reader.fieldnames)
        raw_rows   = list(reader)

    # Discover run_ids from column names  (strip the __<metric> suffix)
    run_ids: list[str] = []
    seen: set[str] = set()
    for col in fieldnames:
        if "__mean_reward" in col:
            rid = col.replace("__mean_reward", "")
            if rid not in seen:
                run_ids.append(rid)
                seen.add(rid)

    if not run_ids:
        raise ValueError(
            f"No '<run_id>__mean_reward' columns found in {path}.\n"
            f"Make sure it was produced by save_merged_csv()."
        )

    # Rebuild per-run row lists, skipping timesteps where a run has no data
    all_rows: list[list[dict]] = []
    for rid in run_ids:
        rows: list[dict] = []
        for raw in raw_rows:
            mean_r = raw.get(f"{rid}__mean_reward", "").strip()
            if mean_r == "":
                continue   # this run didn't have a checkpoint at this timestep
            rows.append({
                "timestep":     int(raw["timestep"]),
                "mean_reward":  float(mean_r),
                "std_reward":   _safe_float(raw.get(f"{rid}__std_reward")),
                "success_rate": _safe_float(raw.get(f"{rid}__success_rate")),
            })
        all_rows.append(rows)

    return run_ids, all_rows


def _safe_float(v: Optional[str]) -> float:
    if v is None or str(v).strip() in ("", "nan", "None"):
        return float("nan")
    return float(v)


# ---------------------------------------------------------------------------
# Summary stats per run
# ---------------------------------------------------------------------------

def run_summary(run_id: str, rows: list[dict], tail_frac: float = 0.20) -> dict:
    """
    Compute per-run summary statistics.

    Parameters
    ----------
    tail_frac : float
        Fraction of the eval sequence used to define the "converged tail"
        for stability / plateau metrics.  Default = last 20 %.
    """
    if not rows:
        return {"run_id": run_id, "n_evals": 0}

    rewards = np.array([r["mean_reward"]  for r in rows], dtype=float)
    success = np.array([r["success_rate"] for r in rows], dtype=float)
    steps   = np.array([r["timestep"]     for r in rows], dtype=float)
    final   = rows[-1]

    best_idx  = int(np.argmax(rewards))
    peak_rew  = float(rewards[best_idx])
    peak_sr   = float(np.nanmax(success))

    # ── Convergence: first timestep where reward crosses 80 / 90 % of peak ─
    threshold_90 = 0.90 * peak_rew
    threshold_80 = 0.80 * peak_rew
    cross_90 = _first_crossing(steps, rewards, threshold_90)
    cross_80 = _first_crossing(steps, rewards, threshold_80)

    # ── Tail stability (coefficient of variation over last tail_frac evals) ─
    tail_n = max(1, int(len(rows) * tail_frac))
    tail_rewards = rewards[-tail_n:]
    tail_success = success[-tail_n:]
    tail_cv = (float(np.std(tail_rewards)) / abs(float(np.mean(tail_rewards)))
               if np.mean(tail_rewards) != 0 else float("nan"))
    tail_mean_sr = float(np.nanmean(tail_success))

    # ── Sample efficiency: AUC (trapezoid) normalised by total timesteps ───
    _trapz  = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    auc_rew = float(_trapz(rewards, steps) / (steps[-1] - steps[0])) if len(steps) > 1 else float("nan")

    # ── Late-stage regression: final vs best reward ─────────────────────────
    regressed = peak_rew - float(final["mean_reward"])

    return {
        "run_id":                run_id,
        "n_evals":               len(rows),
        "final_timestep":        int(final["timestep"]),
        # Final checkpoint
        "final_mean_reward":     float(final["mean_reward"]),
        "final_std_reward":      float(final["std_reward"]),
        "final_success_rate":    float(final["success_rate"]),
        # Best checkpoint
        "best_mean_reward":      peak_rew,
        "best_at_timestep":      int(rows[best_idx]["timestep"]),
        "peak_success_rate":     peak_sr,
        # Convergence speed
        "steps_to_80pct_peak":   cross_80,
        "steps_to_90pct_peak":   cross_90,
        # Stability in converged tail
        "tail_reward_cv":        tail_cv,        # lower is more stable
        "tail_mean_success_rate": tail_mean_sr,
        # Sample efficiency
        "auc_mean_reward":       auc_rew,
        # Regression from peak to final
        "peak_to_final_drop":    regressed,
    }


def _first_crossing(steps: np.ndarray, values: np.ndarray, threshold: float) -> int:
    """Return the first timestep at which values >= threshold, or -1 if never."""
    idx = np.where(values >= threshold)[0]
    if len(idx) == 0:
        return -1
    return int(steps[idx[0]])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt(v, decimals: int = 3) -> str:
    if isinstance(v, float) and (np.isnan(v) or v == float("nan")):
        return "n/a"
    if v == -1:
        return "never"
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)


def _fmt_pct(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    if v == -1:
        return "never"
    return f"{v:.1%}"


def _fmt_steps(v) -> str:
    if v == -1:
        return "never"
    return f"{v:,}"


def _winner_mask(summaries: list[dict], key: str, higher_is_better: bool = True) -> set[str]:
    """Return the run_id(s) with the best value for a given metric key."""
    vals = [(s["run_id"], s.get(key, float("nan"))) for s in summaries]
    vals = [(rid, v) for rid, v in vals if not (isinstance(v, float) and np.isnan(v)) and v != -1]
    if not vals:
        return set()
    best = max(vals, key=lambda x: x[1]) if higher_is_better else min(vals, key=lambda x: x[1])
    return {best[0]}


def print_summary_table(summaries: list[dict]) -> None:
    """Print a two-section summary: core metrics + convergence/stability."""
    _print_core_table(summaries)
    _print_convergence_table(summaries)


def _print_core_table(summaries: list[dict]) -> None:
    cols = [
        ("run_id",             16, lambda v: str(v),        None),
        ("n_evals",             7, lambda v: str(v),        None),
        ("final_timestep",     14, lambda v: f"{v:,}",      None),
        ("final_mean_reward",  17, lambda v: _fmt(v),       True),
        ("final_success_rate", 18, lambda v: _fmt_pct(v),   True),
        ("best_mean_reward",   16, lambda v: _fmt(v),       True),
        ("best_at_timestep",   15, lambda v: _fmt_steps(v), None),
        ("peak_success_rate",  17, lambda v: _fmt_pct(v),   True),
        ("peak_to_final_drop", 18, lambda v: _fmt(v),       False),
    ]
    _print_table("CORE TRAINING METRICS", summaries, cols)


def _print_convergence_table(summaries: list[dict]) -> None:
    cols = [
        ("run_id",                16, lambda v: str(v),      None),
        ("steps_to_80pct_peak",   18, lambda v: _fmt_steps(v), False),
        ("steps_to_90pct_peak",   18, lambda v: _fmt_steps(v), False),
        ("tail_reward_cv",        15, lambda v: _fmt(v, 4),  False),
        ("tail_mean_success_rate",22, lambda v: _fmt_pct(v), True),
        ("auc_mean_reward",       15, lambda v: _fmt(v, 2),  True),
    ]
    _print_table("CONVERGENCE & STABILITY  (↓ = lower is better)", summaries, cols)


def _print_table(title: str, summaries: list[dict], cols: list) -> None:
    """Generic table printer.  cols = [(key, min_width, fmt_fn, higher_is_better|None)]"""
    winners: dict[str, set[str]] = {}
    for key, _, _, hib in cols:
        if hib is not None:
            winners[key] = _winner_mask(summaries, key, higher_is_better=hib)

    col_w  = {key: max(w, len(key)) for key, w, _, _ in cols}
    sep    = "─" * (sum(col_w[k] + 2 for k, *_ in cols) + 1)
    header = "  ".join(f"{key:>{col_w[key]}}" for key, *_ in cols)

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)

    for s in summaries:
        parts = []
        for key, _, fmt, hib in cols:
            raw   = s.get(key, float("nan"))
            cell  = fmt(raw)
            width = col_w[key]
            if hib is not None and s["run_id"] in winners.get(key, set()):
                cell = f"*{cell}"
            parts.append(f"{cell:>{width}}")
        print("  ".join(parts))

    print(sep)
    if any(hib is not None for _, _, _, hib in cols):
        print("  * = best value for that metric across all runs")
    print()


def print_full_table(run_ids: list[str], all_rows: list[list[dict]]) -> None:
    all_timesteps = sorted({r["timestep"] for rows in all_rows for r in rows})
    indexed = [{r["timestep"]: r for r in rows} for rows in all_rows]

    col_w = 10
    header_parts = ["timestep".rjust(12)]
    for rid in run_ids:
        short = rid[-8:]
        header_parts += [
            f"{'rew_' + short:>{col_w}}",
            f"{'sr_'  + short:>{col_w}}",
        ]

    sep = "─" * (len("  ".join(header_parts)) + 2)
    print(sep)
    print("  FULL TRAINING CURVES (rew = mean_reward, sr = success_rate)")
    print(sep)
    print("  ".join(header_parts))
    print(sep)

    for ts in all_timesteps:
        parts = [f"{ts:>12,}"]
        for idx_map in indexed:
            row = idx_map.get(ts)
            if row:
                parts += [
                    f"{_fmt(row['mean_reward']):>{col_w}}",
                    f"{_fmt_pct(row['success_rate']):>{col_w}}",
                ]
            else:
                parts += [f"{'—':>{col_w}}", f"{'—':>{col_w}}"]
        print("  ".join(parts))
    print(sep)
    print()


def print_head_to_head(summaries: list[dict]) -> None:
    """Print a plain-English verdict per headline metric (2-run comparisons only)."""
    if len(summaries) != 2:
        return

    a, b  = summaries
    metrics = [
        ("peak_success_rate",     True,  "Peak success rate"),
        ("best_mean_reward",      True,  "Best mean reward"),
        ("auc_mean_reward",       True,  "Sample efficiency (AUC)"),
        ("steps_to_90pct_peak",   False, "Speed to 90 % of peak reward"),
        ("tail_reward_cv",        False, "Tail stability (CV)"),
        ("final_mean_reward",     True,  "Final mean reward"),
    ]
    sep = "─" * 70
    print(sep)
    print("  HEAD-TO-HEAD COMPARISON")
    print(sep)
    for key, hib, label in metrics:
        va, vb = a.get(key, float("nan")), b.get(key, float("nan"))
        na = isinstance(va, float) and np.isnan(va)
        nb = isinstance(vb, float) and np.isnan(vb)
        if na and nb:
            verdict = "n/a"
        elif na:
            verdict = f"  {b['run_id']} wins  (no data for {a['run_id']})"
        elif nb:
            verdict = f"  {a['run_id']} wins  (no data for {b['run_id']})"
        elif va == vb:
            verdict = "  tie"
        elif va == -1 and vb == -1:
            verdict = "  both never reached threshold"
        elif va == -1:
            verdict = f"  {b['run_id']} wins  ({a['run_id']} never reached threshold)"
        elif vb == -1:
            verdict = f"  {a['run_id']} wins  ({b['run_id']} never reached threshold)"
        else:
            winner  = a if (hib and va > vb) or (not hib and va < vb) else b
            verdict = f"  {winner['run_id']} wins"
        print(f"  {label:<38} {verdict}")
    print(sep)
    print()


def save_merged_csv(
    run_ids:  list[str],
    all_rows: list[list[dict]],
    path:     str,
) -> None:
    all_timesteps = sorted({r["timestep"] for rows in all_rows for r in rows})
    indexed = [{r["timestep"]: r for r in rows} for rows in all_rows]

    fieldnames = ["timestep"]
    for rid in run_ids:
        fieldnames += [
            f"{rid}__mean_reward",
            f"{rid}__std_reward",
            f"{rid}__success_rate",
        ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ts in all_timesteps:
            row: dict = {"timestep": ts}
            for rid, idx_map in zip(run_ids, indexed):
                r = idx_map.get(ts)
                row[f"{rid}__mean_reward"]  = r["mean_reward"]  if r else ""
                row[f"{rid}__std_reward"]   = r["std_reward"]   if r else ""
                row[f"{rid}__success_rate"] = r["success_rate"] if r else ""
            writer.writerow(row)

    print(f"📄 Merged CSV → {path}")


def save_summary_csv(summaries: list[dict], path: str) -> None:
    """Export per-run summary stats alongside the merged CSV."""
    if not summaries:
        return
    keys = list(summaries[0].keys())
    summary_path = path.replace(".csv", "_summary.csv")
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"📄 Summary  CSV → {summary_path}")


# ---------------------------------------------------------------------------
# Shared core: print tables + optionally save outputs
# ---------------------------------------------------------------------------

def _run_comparison(
    run_ids:  list[str],
    all_rows: list[list[dict]],
    full:     bool,
    out:      Optional[str],
    tail:     float,
    save:     bool = True,
) -> None:
    summaries = [run_summary(rid, rows, tail_frac=tail) for rid, rows in zip(run_ids, all_rows)]
    print_summary_table(summaries)

    if len(summaries) == 2:
        print_head_to_head(summaries)

    if full:
        print_full_table(run_ids, all_rows)

    if save:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out or f"./experiments/results/comparison_{ts}.csv"
        save_merged_csv(run_ids, all_rows, out_path)
        save_summary_csv(summaries, out_path)


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------

def compare(
    runs:         list[str] | str | None = None,
    discover_all: bool  = False,
    from_csv:     str | None = None,
    full:         bool  = False,
    out:          str | None = None,
    tail:         float = 0.20,
) -> None:
    """
    Compare training runs.  Three mutually exclusive data sources:

      runs         - list of run_id strings  (discovers individual CSVs)
      discover_all - scan ./experiments/ for all training_evals.csv files
      from_csv     - path to a previously saved merged comparison CSV
    """
    if from_csv:
        print(f"📂 Loading merged CSV: {from_csv}")
        run_ids, all_rows = load_merged_csv(from_csv)
        print(f"  Found {len(run_ids)} run(s): {', '.join(run_ids)}")
        for rid, rows in zip(run_ids, all_rows):
            print(f"  Loaded {len(rows):>4} eval checkpoints from run {rid}")
        # Don't re-save when reading back an existing CSV
        _run_comparison(run_ids, all_rows, full=full, out=out, tail=tail, save=False)
        return

    if discover_all:
        discovered = find_all_training_csvs()
        if not discovered:
            print("No training_evals.csv files found under ./experiments/")
            return
        run_ids   = [r for r, _ in discovered]
        csv_paths = [p for _, p in discovered]
        print(f"Found {len(run_ids)} run(s): {', '.join(run_ids)}")
    else:
        if not runs:
            print("❌ Error: Must provide runs, discover_all=True, or from_csv=<path>.")
            return
        run_ids   = runs if isinstance(runs, list) else runs.split(",")
        csv_paths = [find_training_csv(rid) for rid in run_ids]

    all_rows = []
    for rid, path in zip(run_ids, csv_paths):
        rows = load_training_csv(path)
        all_rows.append(rows)
        print(f"  Loaded {len(rows):>4} eval checkpoints from run {rid}")

    _run_comparison(run_ids, all_rows, full=full, out=out, tail=tail, save=True)




# ---------------------------------------------------------------------------
# Evaluation CSV — loading, aggregation & discovery
# ---------------------------------------------------------------------------

def find_eval_csvs(run_id: str, base: str = "./experiments") -> list[str]:
    """Return all eval CSV paths for a run, sorted newest-first."""
    pattern = os.path.join(base, f"*/*/models/{run_id}/eval_{run_id}_*.csv")
    print(glob.glob(pattern))
    matches = [p for p in sorted(glob.glob(pattern), reverse=True)
               if not p.endswith("_summary.csv")]
    if not matches:
        raise FileNotFoundError(
            f"No eval_{run_id}_<eval_id>.csv found for run_id='{run_id}'.\n"
            f"Run evaluate.py first to generate evaluation CSVs."
        )
    return matches


def find_eval_csv_by_id(run_id: str, eval_id: str, base: str = "./experiments") -> str:
    """
    Locate a specific eval CSV by its date-stamp suffix.

    The naming convention is eval_<run_id>_<eval_id>.csv where eval_id is the
    timestamp portion, e.g. '20260522_143210'. Raises FileNotFoundError if
    the file cannot be found.
    """
    all_csvs = find_eval_csvs(run_id, base)
    target = f"eval_{run_id}_{eval_id}.csv"
    for path in all_csvs:
        if os.path.basename(path) == target:
            return path
    available = [os.path.basename(p) for p in all_csvs]
    raise FileNotFoundError(
        f"No eval CSV with id '{eval_id}' found for run '{run_id}'.\n"
        f"Available: {available}"
    )


def list_eval_csvs(run_ids: list[str], base: str = "./experiments") -> None:
    """Print all available eval CSVs for each run_id, newest first."""
    for rid in run_ids:
        try:
            paths = find_eval_csvs(rid, base)
        except FileNotFoundError:
            print(f"  {rid}: no eval CSVs found")
            continue
        print(f"\n  {rid}:")
        for path in paths:
            fname   = os.path.basename(path)
            prefix  = f"eval_{rid}_"
            eval_id = fname[len(prefix):].removesuffix(".csv") if fname.startswith(prefix) else fname
            print(f"    {eval_id}  ->  {path}")


def find_all_eval_csvs(base: str = "./experiments") -> list[tuple[str, str]]:
    """Return [(run_id, latest_csv_path), ...] for every run with an eval CSV."""
    pattern = os.path.join(base, "*/*/logs/*/eval_*.csv")
    by_run: dict[str, list[str]] = {}
    for path in glob.glob(pattern):
        if path.endswith("_summary.csv"):
            continue
        run_id = os.path.basename(os.path.dirname(path))
        by_run.setdefault(run_id, []).append(path)
    return [(rid, sorted(paths, reverse=True)[0]) for rid, paths in sorted(by_run.items())]


def load_eval_csv(path: str) -> list[dict]:
    """Load an episode-level eval CSV produced by evaluate.py -> save_csv()."""
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict = {
                "episode":      int(row["episode"]),
                "group":        row["group"],
                "is_success":   row["is_success"].strip().lower() in ("true", "1"),
                "total_reward": float(row["total_reward"]),
            }
            # Carry through any extra metric columns as floats where possible
            for k, v in row.items():
                if k in parsed:
                    continue
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _metric_stats(vals: list[float]) -> dict:
    """Compute a rich set of statistics for a list of numeric values."""
    a = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    if len(a) == 0:
        nan = float("nan")
        return dict(mean=nan, std=nan, median=nan, p5=nan, p95=nan, min=nan, max=nan, n=0)
    return dict(
        mean   = float(np.mean(a)),
        std    = float(np.std(a)),
        median = float(np.median(a)),
        p5     = float(np.percentile(a, 5)),
        p95    = float(np.percentile(a, 95)),
        min    = float(np.min(a)),
        max    = float(np.max(a)),
        n      = len(a),
    )


def aggregate_eval_csv(
    rows: list[dict],
) -> tuple[dict, dict[str, dict]]:
    """
    Aggregate episode-level rows into overall + per-group summary dicts.

    For every numeric metric (total_reward + extras) a full set of statistics
    is computed: mean, std, median, p5, p95, min, max.  These are stored flat
    in the summary dict as <metric>__<stat> keys (e.g. total_reward__median),
    with the plain <metric> key holding the mean for backwards compatibility
    with the ranking and console-table code.

    Returns (overall_summary, {group: group_summary}).
    """
    fixed = {"episode", "group", "is_success", "total_reward"}
    extra_keys = [k for k in rows[0] if k not in fixed
                  and isinstance(rows[0].get(k), (int, float))] if rows else []
    all_metric_keys = ["total_reward"] + extra_keys

    def _summarise(label: str, subset: list[dict]) -> dict:
        n = len(subset)
        if n == 0:
            return {"group": label, "n_episodes": 0,
                    "success_rate": float("nan"),
                    "mean_total_reward": float("nan"),
                    "std_total_reward":  float("nan")}

        summary: dict = {
            "group":        label,
            "n_episodes":   n,
            "success_rate": sum(r["is_success"] for r in subset) / n,
        }

        for k in all_metric_keys:
            vals = [r[k] for r in subset if isinstance(r.get(k), (int, float))]
            stats = _metric_stats(vals)
            # Flat storage: total_reward__mean, total_reward__std, ...
            for stat, v in stats.items():
                summary[f"{k}__{stat}"] = v
            # Convenience aliases that existing ranking / table code uses
            if k == "total_reward":
                summary["mean_total_reward"] = stats["mean"]
                summary["std_total_reward"]  = stats["std"]
            else:
                summary[k] = stats["mean"]   # backwards-compat mean alias

        return summary

    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)

    overall   = _summarise("overall", rows)
    per_group = {g: _summarise(g, recs) for g, recs in sorted(by_group.items())}
    return overall, per_group


def extra_metric_keys(rows: list[dict]) -> list[str]:
    """Return the names of numeric extra columns (beyond the four fixed fields)."""
    fixed = {"episode", "group", "is_success", "total_reward"}
    if not rows:
        return []
    return [k for k in rows[0] if k not in fixed and isinstance(rows[0].get(k), (int, float))]


# ---------------------------------------------------------------------------
# Default RankSpec: max for all numeric metrics in the overall summary
# ---------------------------------------------------------------------------

_DEFAULT_RANK_DIRECTIONS: dict[str, Callable] = {
    "success_rate":      max,
    "mean_total_reward": max,
    "std_total_reward":  min,   # lower spread = more consistent
}


def _build_default_rank_spec(metric_keys: list[str]) -> RankSpec:
    """
    Build a RankSpec from _DEFAULT_RANK_DIRECTIONS for known metrics,
    defaulting to max for any unrecognised numeric metric.
    """
    return {k: _DEFAULT_RANK_DIRECTIONS.get(k, max) for k in metric_keys}


def parse_rank_arg(rank_list: list[str]) -> RankSpec:
    """
    Parse a CLI --rank string into a RankSpec.

    Format: "metric1:max metric2:min metric3:max"
    """
    spec: RankSpec = {}
    print(f"Parsing rank spec: '{rank_list}'")
    for token in rank_list:
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"--rank entries must be 'metric:max' or 'metric:min', got: '{token}'")
        metric, direction = token.rsplit(":", 1)
        metric, direction = metric.strip(), direction.strip().lower()
        if direction == "max":
            spec[metric] = max
        elif direction == "min":
            spec[metric] = min
        elif direction in ("none", "ignore"):
            spec[metric] = None  # Explicitly map to None to signal "do not rank"
        else:
            raise ValueError(f"Direction must be 'max', 'min' or 'none', got: '{direction}'")
    return spec

def _build_rank_spec_from_defaults_and_overrides(metric_keys: list[str], overrides: RankSpec) -> RankSpec:
    """
    Build a RankSpec using defaults for known metrics, max for unknowns,
    and applying any CLI overrides from --rank.

    Overrides take precedence over defaults.  For example, if the default is
    to max "success_rate" but the user specifies "success_rate:min", the
    override will apply and lower success_rate will be ranked better.
    """
    spec = _build_default_rank_spec(metric_keys)
    spec.update(overrides)
    cleaned_spec = {k: fn for k, fn in spec.items() if fn is not None}
    return cleaned_spec


# ---------------------------------------------------------------------------
# Evaluation comparison core
# ---------------------------------------------------------------------------

def _numeric_metrics(summary: dict) -> list[str]:
    """Return top-level rankable metric keys (excludes __stat sub-keys and metadata)."""
    _skip = {"group", "n_episodes"}
    return [
        k for k, v in summary.items()
        if k not in _skip and "__" not in k
        and isinstance(v, (int, float)) and v is not None
    ]


def _rank_runs(
    run_ids:    list[str],
    summaries:  list[dict],
    rank_spec:  RankSpec,
) -> dict[str, str]:
    """
    For each metric in rank_spec, determine the winning run_id.

    Returns {metric: winning_run_id}.  Ties produce "tie".
    """
    winners: dict[str, str] = {}
    for metric, rank_fn in rank_spec.items():
        pairs = [
            (rid, s.get(metric))
            for rid, s in zip(run_ids, summaries)
            if isinstance(s.get(metric), (int, float)) and s.get(metric) is not None
        ]
        if not pairs:
            winners[metric] = "n/a"
            continue
        best_val = rank_fn(v for _, v in pairs)
        best_runs = [rid for rid, v in pairs if v == best_val]
        winners[metric] = "tie" if len(best_runs) > 1 else best_runs[0]
    return winners


def pareto_front(
    run_ids:   list[str],
    summaries: list[dict],
    rank_spec: RankSpec,
) -> tuple[list[str], dict[str, int]]:
    """
    Identify the Pareto-optimal runs given the objectives in rank_spec.

    A run A dominates run B if A is at least as good as B on every objective
    and strictly better on at least one.  "Better" is defined by rank_spec:
    max -> higher is better, min -> lower is better.

    Returns
    -------
    front_ids       : run_ids that are non-dominated
    dominated_count : {run_id: number of objectives on which this run is
                       strictly dominated by at least one other run}
    """
    metrics = [m for m in rank_spec if any(
        isinstance(s.get(m), (int, float)) and not np.isnan(s.get(m, float("nan")))
        for s in summaries
    )]

    def _val(s: dict, m: str) -> float:
        v = s.get(m, float("nan"))
        if not isinstance(v, (int, float)):
            return float("nan")
        # Flip sign for min objectives so "higher is always better" in comparison
        return -float(v) if rank_spec[m] is min else float(v)

    n = len(run_ids)
    dominated_count: dict[str, int] = {rid: 0 for rid in run_ids}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Does run j dominate run i?
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


def print_pareto_table(
    run_ids:   list[str],
    summaries: list[dict],
    rank_spec: RankSpec,
    title:     str = "PARETO FRONT",
) -> None:
    """
    Print a Pareto-dominance summary alongside per-objective values.

    Runs on the Pareto front are marked with *.
    The 'dominated_by' column shows how many other runs dominate this run
    (0 = on the front).
    """
    if len(run_ids) < 2:
        return

    front_ids, dominated_count = pareto_front(run_ids, summaries, rank_spec)
    metrics = [m for m in rank_spec if any(
        isinstance(s.get(m), (int, float)) and not np.isnan(s.get(m, float("nan")))
        for s in summaries
    )]

    id_w  = max(18, max(len(r) for r in run_ids) + 2)
    met_w = max(20, max((len(m) for m in metrics), default=10) + 4) if metrics else 20
    dom_w = 13

    col_widths = [id_w, dom_w] + [met_w] * len(metrics)
    sep = "─" * (sum(col_widths) + 2 * len(col_widths) + 1)

    direction_hint = {m: ("↑" if fn is max else "↓") for m, fn in rank_spec.items()}
    header_parts   = [
        f"{'run_id':>{id_w}}",
        f"{'dominated_by':>{dom_w}}",
    ] + [f"{m + ' ' + direction_hint.get(m, ''):>{met_w}}" for m in metrics]

    print(f"\n{sep}")
    print(f"  {title}  (* = non-dominated)")
    print(sep)
    print("  ".join(header_parts))
    print(sep)

    for rid, s in sorted(zip(run_ids, summaries), key=lambda x: dominated_count[x[0]]):
        dom  = dominated_count[rid]
        star = "*" if rid in front_ids else " "
        parts = [
            f"{star + rid:>{id_w}}",
            f"{dom:>{dom_w}}",
        ]
        for m in metrics:
            v = s.get(m)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cell = "n/a"
            elif m == "success_rate":
                cell = f"{v:.1%}"
            else:
                cell = f"{v:.3f}"
            parts.append(f"{cell:>{met_w}}")
        print("  ".join(parts))

    print(sep)
    print(f"  Pareto front ({len(front_ids)} run(s)): {', '.join(front_ids) or 'none'}")
    print()


def print_eval_summary_table(
    run_ids:   list[str],
    summaries: list[dict],
    rank_spec: RankSpec,
    title:     str = "EVALUATION COMPARISON",
) -> None:
    """
    Print a ranked comparison table for evaluation metrics.

    For each metric, two rows are printed:
      mean ± std   (ranked; winner marked with *)
      med [p5–p95] (contextual; not ranked)
    """
    winners = _rank_runs(run_ids, summaries, rank_spec)

    # Collect top-level metric keys — exclude internal __stat keys, group, n_episodes
    _skip = {"group", "n_episodes"}
    seen: set[str] = set()
    metric_keys: list[str] = []
    for s in summaries:
        for k in s:
            if k in _skip or "__" in k or k in seen:
                continue
            if isinstance(s[k], (int, float)) and s[k] is not None:
                metric_keys.append(k)
                seen.add(k)

    direction_hint = {
        m: ("↑" if fn is max else "↓" if fn is min else "?")
        for m, fn in rank_spec.items()
    }

    id_w  = max(18, max(len(r) for r in run_ids) + 2)
    met_w = max(22, max((len(k) for k in metric_keys), default=10) + 6)
    win_w = max(10, id_w)
    sep   = "─" * (met_w + 2 + (id_w + 2) * len(run_ids) + win_w + 2)
    header = f"{'metric':>{met_w}}  " + "  ".join(f"{r:>{id_w}}" for r in run_ids) + f"  {'winner':>{win_w}}"

    def _fmt_cell(k: str, s: dict, stat: str = "mean") -> str:
        """Format a single cell value for metric k, resolving __stat sub-keys."""
        # success_rate is a plain scalar with no sub-stats
        if k == "success_rate":
            if stat == "mean":
                v = s.get("success_rate")
            else:
                # No distributional sub-stats for a proportion — skip
                return "n/a"
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "n/a"
            return f"{v:.1%}"

        # mean/std_total_reward both index into the total_reward__ sub-stats
        if k in ("mean_total_reward", "std_total_reward"):
            key = f"total_reward__{stat}"
        else:
            key = f"{k}__{stat}"

        v = s.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        return f"{v:.3f}"

    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print(sep)

    for k in metric_keys:
        hint   = direction_hint.get(k, " ")
        winner = winners.get(k, "—")
        label  = f"{k} {hint}" if k in rank_spec else k

        # Row 1: mean ± std
        cells_mean = []
        for rid, s in zip(run_ids, summaries):
            mean_s = _fmt_cell(k, s, "mean")
            std_s  = _fmt_cell(k, s, "std")
            cell   = f"{mean_s} ±{std_s}" if std_s != "n/a" else mean_s
            if rid == winner:
                cell = f"*{cell}"
            cells_mean.append(f"{cell:>{id_w}}")

        # Row 2: median [p5–p95] — suppressed for scalar aggregates
        _no_dist = {"success_rate", "std_total_reward"}
        if k not in _no_dist:
            cells_med = []
            for s in summaries:
                med = _fmt_cell(k, s, "median")
                p5  = _fmt_cell(k, s, "p5")
                p95 = _fmt_cell(k, s, "p95")
                if med == "n/a":
                    cells_med.append(f"{'':>{id_w}}")
                else:
                    cell = f"med {med} [{p5}–{p95}]"
                    cells_med.append(f"{cell:>{id_w}}")
            sub_label = f"{label:>{met_w - 2}}"
            print(f"  {sub_label}  " + "  ".join(cells_mean) + f"  {winner:>{win_w}}")
            print(f"  {'':>{met_w - 2}}  " + "  ".join(cells_med))
        else:
            sub_label = f"{label:>{met_w - 2}}"
            print(f"  {sub_label}  " + "  ".join(cells_mean) + f"  {winner:>{win_w}}")

    print(sep)
    if rank_spec:
        print("  * = best mean for that metric  |  ↑ higher is better  |  ↓ lower is better")
        print("  second row: median and [p5–p95] interval")
    print()


def print_eval_group_tables(
    run_ids:        list[str],
    all_per_group:  list[dict[str, dict]],  # per_group dict per run
    rank_spec:      RankSpec,
) -> None:
    """Print one ranked table + Pareto front per group across all runs."""
    all_groups: set[str] = set()
    for pg in all_per_group:
        all_groups.update(pg.keys())

    for group in sorted(all_groups):
        group_summaries = [pg.get(group, {}) for pg in all_per_group]
        print_eval_summary_table(
            run_ids, group_summaries, rank_spec,
            title=f"EVALUATION COMPARISON  —  group: {group}",
        )
        print_pareto_table(
            run_ids, group_summaries, rank_spec,
            title=f"PARETO FRONT  —  group: {group}",
        )


def save_eval_csv(
    run_ids:             list[str],
    all_rows:            list[list[dict]],
    overall_summaries:   list[dict],
    per_group_summaries: list[dict[str, dict]],
    rank_spec:           RankSpec,
    path:                str,
) -> None:
    """
    Save everything needed to reconstruct a compare_evaluations result.

    Writes two files:

    <path>  — episode-level CSV with a leading 'run_id' column.  Contains every
              raw episode row from every run; used to reconstruct all_rows and to
              regenerate distributional plots (violins, boxplots, histograms).

    <path stem>_summaries.csv  — one row per run.  Columns:
        run_id
        rank_fn__<metric>           — 'max' | 'min' | 'n/a' for every ranked metric
        pareto__on_front            — 'True' | 'False' (overall Pareto membership)
        pareto__dominated_count     — int: number of runs that dominate this run overall
        pareto_group_<g>__on_front        — per-group Pareto membership
        pareto_group_<g>__dominated_count — per-group dominated count
        overall__<key>              — every key in the overall summary dict (including
                                      all metric__stat sub-keys)
        <group>__<key>              — same set of keys, prefixed by group name, for
                                      every group in per_group_summaries

    Together these two files allow load_eval_comparison_csv() to reproduce the
    exact (run_ids, all_rows, overall_summaries, per_group_summaries, rank_spec)
    tuple returned by compare_evaluations().
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    summary_path = path.replace(".csv", "_summaries.csv")

    # ── Pre-compute Pareto memberships ────────────────────────────────────────
    overall_front, overall_dom = (
        pareto_front(run_ids, overall_summaries, rank_spec)
        if len(run_ids) > 1 and rank_spec
        else (run_ids[:], {r: 0 for r in run_ids})
    )

    all_groups: list[str] = []
    seen_g: set[str] = set()
    for pg in per_group_summaries:
        for g in pg:
            if g not in seen_g:
                all_groups.append(g)
                seen_g.add(g)

    group_fronts: dict[str, list[str]]   = {}
    group_doms:   dict[str, dict[str, int]] = {}
    for g in all_groups:
        g_summaries = [pg.get(g, {}) for pg in per_group_summaries]
        if len(run_ids) > 1 and rank_spec:
            gf, gd = pareto_front(run_ids, g_summaries, rank_spec)
        else:
            gf, gd = run_ids[:], {r: 0 for r in run_ids}
        group_fronts[g] = gf
        group_doms[g]   = gd

    # ── Build ordered column set ──────────────────────────────────────────────
    summary_cols: dict[str, None] = {"run_id": None}

    for metric in rank_spec:
        summary_cols[f"rank_fn__{metric}"] = None

    summary_cols["pareto__on_front"]        = None
    summary_cols["pareto__dominated_count"] = None
    for g in all_groups:
        summary_cols[f"pareto_group_{g}__on_front"]        = None
        summary_cols[f"pareto_group_{g}__dominated_count"] = None

    if overall_summaries:
        for k in overall_summaries[0]:
            summary_cols[f"overall__{k}"] = None

    for g in all_groups:
        ref = next((pg[g] for pg in per_group_summaries if g in pg), {})
        for k in ref:
            summary_cols[f"{g}__{k}"] = None

    summary_fieldnames = list(summary_cols)

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rid, overall, per_group in zip(run_ids, overall_summaries, per_group_summaries):
            row: dict = {"run_id": rid}
            for metric, fn in rank_spec.items():
                row[f"rank_fn__{metric}"] = fn.__name__ if callable(fn) else "n/a"
            row["pareto__on_front"]        = rid in overall_front
            row["pareto__dominated_count"] = overall_dom.get(rid, 0)
            for g in all_groups:
                row[f"pareto_group_{g}__on_front"]        = rid in group_fronts.get(g, [])
                row[f"pareto_group_{g}__dominated_count"] = group_doms.get(g, {}).get(rid, 0)
            for k, v in overall.items():
                row[f"overall__{k}"] = v
            for g, gdict in per_group.items():
                for k, v in gdict.items():
                    row[f"{g}__{k}"] = v
            writer.writerow(row)
    print(f"📄 Eval summaries CSV → {summary_path}")


def load_eval_comparison_csv(
    path: str,
) -> tuple[list[str], list[list[dict]], list[dict], list[dict[str, dict]], RankSpec]:
    """
    Reconstruct the full compare_evaluations() return value from the two CSV
    files produced by save_eval_csv().

    Parameters
    ----------
    path : path to the episode-level CSV (the _summaries.csv is inferred by
           replacing '.csv' with '_summaries.csv').

    Returns
    -------
    (run_ids, all_rows, overall_summaries, per_group_summaries, rank_spec)
    — identical in structure to what compare_evaluations() returns.
    """
    summary_path = path.replace(".csv", "_summaries.csv")
    for p in (path, summary_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected eval CSV not found: {p}")

    # ── 1. Episode rows ───────────────────────────────────────────────────────
    run_order: list[str] = []
    seen_runs: set[str]  = set()
    raw_by_run: dict[str, list[dict]] = {}

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rid = row["run_id"]
            if rid not in seen_runs:
                run_order.append(rid)
                seen_runs.add(rid)
                raw_by_run[rid] = []
            parsed: dict = {}
            for k, v in row.items():
                if k == "run_id":
                    continue
                if k == "is_success":
                    parsed[k] = str(v).strip().lower() in ("true", "1")
                elif k == "group":
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v) if v.strip() not in ("", "nan", "None") else float("nan")
                    except (ValueError, AttributeError):
                        parsed[k] = v
            raw_by_run[rid].append(parsed)

    all_rows = [raw_by_run[rid] for rid in run_order]

    # ── 2. Summaries + rank_spec ──────────────────────────────────────────────
    overall_summaries:   list[dict]             = []
    per_group_summaries: list[dict[str, dict]]  = []
    rank_spec: RankSpec = {}

    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        summary_rows = list(reader)

    # Collect group names from column prefixes (columns that are not overall__, rank_fn__, pareto__ or run_id)
    group_names: list[str] = []
    seen_gn: set[str] = set()
    for col in fieldnames:
        if (col in ("run_id",)
                or col.startswith("overall__")
                or col.startswith("rank_fn__")
                or col.startswith("pareto__")
                or col.startswith("pareto_group_")):
            continue
        g = col.split("__")[0]
        if g not in seen_gn:
            group_names.append(g)
            seen_gn.add(g)

    # Build rank_spec from the first row (directions are the same for all runs)
    if summary_rows:
        first = summary_rows[0]
        for col in fieldnames:
            if col.startswith("rank_fn__"):
                metric = col[len("rank_fn__"):]
                direction = first.get(col, "n/a").strip()
                if direction == "max":
                    rank_spec[metric] = max
                elif direction == "min":
                    rank_spec[metric] = min
                # else: omit metrics with "n/a" direction

    def _parse_val(v: str):
        s = str(v).strip()
        if s in ("", "nan", "None"):
            return float("nan")
        try:
            return float(s)
        except ValueError:
            return s

    for srow in summary_rows:
        overall: dict = {}
        per_group: dict[str, dict] = {}

        for col, raw in srow.items():
            if (col == "run_id"
                    or col.startswith("rank_fn__")
                    or col.startswith("pareto__")
                    or col.startswith("pareto_group_")):
                continue
            if col.startswith("overall__"):
                k = col[len("overall__"):]
                overall[k] = _parse_val(raw)
            else:
                # <group>__<key>
                g, _, k = col.partition("__")
                if k:
                    per_group.setdefault(g, {})[k] = _parse_val(raw)

        overall_summaries.append(overall)
        per_group_summaries.append(per_group)

    # Verify row order matches episode CSV (they should, but make it explicit)
    csv_run_ids = [r["run_id"] for r in summary_rows]
    if csv_run_ids != run_order:
        # Re-align to episode CSV order
        idx = {rid: i for i, rid in enumerate(csv_run_ids)}
        overall_summaries   = [overall_summaries[idx[r]]   for r in run_order]
        per_group_summaries = [per_group_summaries[idx[r]] for r in run_order]

    return run_order, all_rows, overall_summaries, per_group_summaries, rank_spec


# ---------------------------------------------------------------------------
# Programmatic API — evaluation comparison
# ---------------------------------------------------------------------------

def compare_evaluations(
    runs:         list[str] | None       = None,
    discover_all: bool                   = False,
    eval_ids:     list[str | None] | None = None,
    rank_spec:    RankSpec | None        = None,
    show_groups:  bool                   = False,
    show_pareto:  bool                   = True,
    out:          str | None             = None,
    base:         str                    = "./experiments",
) -> tuple[list[str], list[list[dict]], list[dict], list[dict[str, dict]], RankSpec]:
    """
    Compare post-training evaluation results across multiple runs.

    Returns (run_ids, all_raw_rows, overall_summaries, per_group_summaries, rank_spec)
    so callers can immediately feed results into plotting functions.

    Loads episode-level eval CSVs (eval_<run_id>_<eval_id>.csv, produced by
    evaluate.py) and aggregates metrics on the fly from raw episode data.

    Parameters
    ----------
    runs         : list of run_id strings.
    discover_all : scan base/ for all runs that have eval CSVs (uses latest
                   eval per run; eval_ids is ignored in this mode).
    eval_ids     : parallel list to runs; each entry is either a timestamp
                   string ('20260522_143210') to select a specific eval, or
                   None to use the latest eval for that run.
                   If omitted entirely, the latest eval is used for all runs.
    rank_spec    : maps metric_name -> rank_fn (max / min / any callable that
                   accepts an iterable of floats and returns the best scalar).
                   Defaults to _DEFAULT_RANK_DIRECTIONS (max for reward/success,
                   min for std).  Metrics not in rank_spec are shown unranked.
    show_groups  : also print per-group breakdown tables.
    out          : override the output CSV path.
    base         : root directory to search for experiment runs.
    """
    if discover_all:
        discovered = find_all_eval_csvs(base)
        if not discovered:
            print("No eval_*.csv files found under ./experiments/")
            return
        run_ids    = [r for r, _ in discovered]
        csv_paths  = [p for _, p in discovered]
    else:
        if not runs:
            print("❌ Error: provide runs or discover_all=True.")
            return
        run_ids = runs if isinstance(runs, list) else runs.split(",")
        # Resolve each run to a specific CSV path
        resolved_ids: list[str | None] = (
            eval_ids if eval_ids is not None else [None] * len(run_ids)
        )
        if len(resolved_ids) != len(run_ids):
            raise ValueError(
                f"eval_ids length ({len(resolved_ids)}) must match runs length ({len(run_ids)})."
            )
        csv_paths = [
            find_eval_csv_by_id(rid, eid, base) if eid else find_eval_csvs(rid, base)[0]
            for rid, eid in zip(run_ids, resolved_ids)
        ]

    print(f"\n📂 Comparing {len(run_ids)} evaluation run(s): {', '.join(run_ids)}")

    all_raw_rows:        list[list[dict]]        = []
    overall_summaries:   list[dict]              = []
    per_group_summaries: list[dict[str, dict]]   = []

    for rid, path in zip(run_ids, csv_paths):
        rows = load_eval_csv(path)
        overall, per_group = aggregate_eval_csv(rows)
        all_raw_rows.append(rows)
        overall_summaries.append(overall)
        per_group_summaries.append(per_group)
        print(f"  Loaded {len(rows):>4} episodes from {path}")

    # Build effective rank_spec from defaults + any caller overrides
    all_metric_keys: list[str] = []
    seen_mk: set[str] = set()
    for s in overall_summaries:
        for k in _numeric_metrics(s):
            if k not in seen_mk:
                all_metric_keys.append(k)
                seen_mk.add(k)

    effective_rank_spec = _build_rank_spec_from_defaults_and_overrides(all_metric_keys, rank_spec or {})

    print_eval_summary_table(run_ids, overall_summaries, effective_rank_spec)
    if show_pareto:
        print_pareto_table(run_ids, overall_summaries, effective_rank_spec)

    if show_groups:
        print_eval_group_tables(run_ids, per_group_summaries, effective_rank_spec)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if out is not None:
        out_path = f"./{out}/eval_comparison_{ts}.csv"
        save_eval_csv(run_ids, all_raw_rows, overall_summaries, per_group_summaries, effective_rank_spec, out_path)

    return run_ids, all_raw_rows, overall_summaries, per_group_summaries, effective_rank_spec




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_compare_cli(experiment_cls=None) -> None:
    """Standalone CLI entry point — supports both training and evaluation comparison."""
    p = argparse.ArgumentParser(
        description=(
            "Compare training curves (default) or evaluation results (--eval) "
            "across multiple runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--eval", action="store_true",
        help="Compare post-training evaluation YAMLs instead of training curves.",
    )

    # ── Data source (shared) ────────────────────────────────────────────────
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-ids", nargs="+", metavar="RUN_ID",
        help="One or more run IDs to compare.",
    )
    source.add_argument(
        "--all", action="store_true",
        help="Auto-discover all runs.",
    )
    source.add_argument(
        "--from-csv", metavar="PATH",
        help="(Training mode only) Re-load a previously saved merged comparison CSV.",
    )

    # ── Training-mode options ────────────────────────────────────────────────
    p.add_argument("--full", action="store_true",
                   help="(Training mode) Also print the full per-timestep table.")
    p.add_argument("--tail", type=float, default=0.20, metavar="FRAC",
                   help="(Training mode) Fraction of eval history used as the converged tail.")

    # ── Evaluation-mode options ──────────────────────────────────────────────
    p.add_argument(
        "--rank", type=str, default=None, metavar="METRIC:DIR METRIC:DIR ...", nargs="+",
        help=(
            "(Eval mode) Space-separated metric:direction pairs to control ranking. "
            "direction is 'max' or 'min'. "
            "Example: --rank success_rate:max,my_penalty:min. "
            "Defaults to max for reward/success, min for std."
        ),
    )
    p.add_argument(
        "--eval-id", nargs="+", default=None, metavar="EVAL_ID",
        help=(
            "(Eval mode) Timestamp ID(s) of specific eval CSV(s) to load, one per run. "
            "Format: YYYYMMDD_HHMMSS. Use 'latest' or omit to use the newest eval. "
            "Example: --eval-id 20260522_143210 latest"
        ),
    )
    p.add_argument(
        "--list-evals", action="store_true",
        help="(Eval mode) List all available eval CSVs for the given runs, then exit.",
    )
    p.add_argument(
        "--no-pareto", action="store_true",
        help="(Eval mode) Suppress the Pareto front table.",
    )
    p.add_argument(
        "--groups", action="store_true",
        help="(Eval mode) Also print per-group breakdown tables.",
    )
    # ── Shared output option ─────────────────────────────────────────────────
    p.add_argument("--out", type=str, default=None, metavar="PATH",
                   help="Directory for output files (CSV + plots).")

    args = p.parse_args()

    if args.eval:
        # --list-evals: just print available CSVs and exit
        if args.list_evals:
            if not args.run_ids:
                p.error("--list-evals requires --run-ids.")
            print()
            list_eval_csvs(args.run_ids)
            return

        # Resolve --eval-id: convert 'latest' sentinel to None
        eval_ids: list[str | None] | None = None
        if args.eval_id:
            eval_ids = [None if e.lower() == "latest" else e for e in args.eval_id]

        rank_spec = parse_rank_arg(args.rank) if args.rank else None
        result = compare_evaluations(
            runs=args.run_ids,
            discover_all=args.all,
            eval_ids=eval_ids,
            rank_spec=rank_spec,
            show_groups=args.groups,
            show_pareto=not args.no_pareto,
            out=args.out,
        )

    else:
        compare(
            runs=args.run_ids,
            discover_all=args.all,
            from_csv=getattr(args, "from_csv", None),
            full=args.full,
            out=args.out,
            tail=args.tail,
        )