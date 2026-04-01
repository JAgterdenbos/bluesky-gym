"""
evaluate.py
-----------
Generic evaluation script for any trained SB3 + GoalEnv model.

Extension point
---------------
Custom per-env metrics are injected via a MetricExtractor — a small object
that knows how to pull named float values out of a final episode info dict,
and how to aggregate them across episodes.

The evaluator itself never knows about env-specific keys like "simt" or
"average_path_rew".  That knowledge lives in the extractor, which is
defined alongside the experiment class in experiment.py and passed in
at call time.

Built-in (env-agnostic) metrics
--------------------------------
  is_success      - from info["is_success"]
  total_reward    - from info["total_reward"]

These are always collected.  Everything else is opt-in via the extractor.

Output
------
  Console  - formatted summary table (overall + per group)
  CSV      - one row per episode     → <save_path>/eval_<run_id>_<ts>.csv
  JSON     - aggregated metrics      → <save_path>/eval_<run_id>_<ts>.json

Usage
-----
  python evaluate.py --run-id 20260331_134059
  python evaluate.py --run-id 20260331_134059 --episodes 50 --runways 27 18R
  python evaluate.py --run-id 20260331_134059 --no-render
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Callable, Optional, TypedDict, cast

import bluesky_gym
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from config import ExperimentConfig
from experiment import PathPlanningExperiment


# ---------------------------------------------------------------------------
# MetricExtractor — the extension point
# ---------------------------------------------------------------------------

class MetricExtractor:
    """Declares which extra metrics to pull from an episode's final info dict,
    and how to aggregate them across episodes.

    Parameters
    ----------
    extractors : dict[str, Callable[[dict, bool], float]]
        Maps a metric name to a function ``(info, is_success) -> float``.
        Receive both the info dict and the success flag so extractors can
        return float("nan") conditionally — e.g. flight_time is only
        meaningful on successful episodes.

    aggregators : dict[str, Callable[[list[float]], float]] | None
        Maps a metric name to an aggregation function.
        Any metric not listed here defaults to np.nanmean.

    display : list[str] | None
        Ordered subset of metric names to show in the console table.
        Defaults to all metrics in insertion order.

    Example (PathPlanning)
    ----------------------
    >>> extractor = MetricExtractor(
    ...     extractors={
    ...         "flight_time_min": lambda info, ok: info.get("simt", float("nan")) / 60,
    ...         "path_length_km":  lambda info, ok: _path_km(info),
    ...         "noise_reward":    lambda info, ok: float(info.get("average_noise_rew", float("nan"))),
    ...     },
    ...     display=["flight_time_min", "path_length_km", "noise_reward"],
    ... )
    """

    def __init__(
        self,
        extractors:  dict[str, Callable[[dict, bool], float]],
        aggregators: dict[str, Callable[[list[float]], float]] | None = None,
        display:     list[str] | None = None,
    ) -> None:
        self.extractors  = extractors
        self.aggregators = aggregators or {}
        self.display     = display or list(extractors.keys())

    def extract(self, info: dict, is_success: bool) -> dict[str, float]:
        """Return ``{name: float}`` for one episode."""
        return {name: fn(info, is_success) for name, fn in self.extractors.items()}

    def aggregate(self, rows: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate a list of per-episode extra-metric dicts into one summary."""
        result: dict[str, float] = {}
        for name in self.extractors:
            values = [r[name] for r in rows]
            agg_fn = self.aggregators.get(name, np.nanmean)
            try:
                result[name] = float(agg_fn(values))
            except Exception:
                result[name] = float("nan")
        return result


# ---------------------------------------------------------------------------
# Episode record (typed dict — accommodates arbitrary extra fields)
# ---------------------------------------------------------------------------
#
# Guaranteed keys in every record:
#   episode       int
#   group         str   (runway id, or whatever groups episodes in this env)
#   is_success    bool
#   total_reward  float
#   + whatever keys the MetricExtractor adds

class EpisodeRecord(TypedDict):
    episode: int
    group: str
    is_success: bool
    total_reward: float
    # Note: Type checkers won't know about extra keys, but won't block access
    # if you use .get() or ignore specific strict-typing flags.


def _make_record(
    episode:      int,
    group:        str,
    is_success:   bool,
    total_reward: float,
    extras:       dict[str, float],
) -> EpisodeRecord:
    rec = {
        "episode":      episode,
        "group":        group,
        "is_success":   is_success,
        "total_reward": total_reward,
        **extras,
    }
    return cast(EpisodeRecord, rec)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    cfg:        ExperimentConfig,
    extractor:  MetricExtractor | None,
    n_episodes: int,
    runways:    Optional[list[str]],
    render:     bool,
) -> list[EpisodeRecord]:
    """Run ``n_episodes`` episodes; return one EpisodeRecord per episode."""

    eval_kwargs = cfg.eval_env_kwargs
    if runways is not None:
        eval_kwargs["runways"] = runways

    render_mode = "human" if render else None
    env   = Monitor(gym.make(cfg.session.env_name, render_mode=render_mode, **eval_kwargs))
    model = cfg.model.algorithm.load(f"{cfg.save_path}/final_model", env=env)

    records: list[EpisodeRecord] = []

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = truncated = False
        info: dict = {}

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            if hasattr(action, "shape") and action.shape == ():
                action = action[()]
            obs, _, done, truncated, info = env.step(action)

        is_success   = bool(info.get("is_success", False))
        total_reward = float(info.get("total_reward", 0.0))
        group        = str(info.get("current_runway", "unknown"))
        extras       = extractor.extract(info, is_success) if extractor else {}

        rec = _make_record(ep, group, is_success, total_reward, extras)
        records.append(rec)

        # Per-episode console line
        status    = "✅" if is_success else "❌"
        extra_str = "  ".join(
            f"{k}={v:.2f}"
            for k, v in extras.items()
            if extractor and k in extractor.display and not np.isnan(v)
        )
        print(
            f"  Ep {ep:>3}/{n_episodes}  {status}  "
            f"rwy={group:<4}  reward={total_reward:+.1f}"
            + (f"  {extra_str}" if extra_str else "")
        )

    env.close()
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_group(
    label:     str,
    recs:      list[EpisodeRecord],
    extractor: MetricExtractor | None,
) -> dict:
    n = len(recs)
    if n == 0:
        base = {
            "group": label, "n_episodes": 0,
            "success_rate": 0.0,
            "mean_total_reward": float("nan"),
            "std_total_reward":  float("nan"),
        }
        if extractor:
            base.update({k: float("nan") for k in extractor.extractors})
        return base

    rewards = [r["total_reward"] for r in recs]
    base = {
        "group":             label,
        "n_episodes":        n,
        "success_rate":      sum(r["is_success"] for r in recs) / n,
        "mean_total_reward": float(np.mean(rewards)),
        "std_total_reward":  float(np.std(rewards)),
    }

    if extractor:
        extra_rows = [{k: r[k] for k in extractor.extractors} for r in recs]
        base.update(extractor.aggregate(extra_rows))

    return base


def aggregate_metrics(
    records:   list[EpisodeRecord],
    extractor: MetricExtractor | None,
) -> tuple[dict, dict[str, dict]]:
    """Return ``(overall_summary, per_group_summary_dict)``."""
    by_group: dict[str, list[EpisodeRecord]] = {}
    for rec in records:
        by_group.setdefault(rec["group"], []).append(rec)

    overall   = _aggregate_group("overall", records, extractor)
    per_group = {
        g: _aggregate_group(g, recs, extractor)
        for g, recs in sorted(by_group.items())
    }
    return overall, per_group


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v:.1%}"


def _fmt_f(v, decimals: int = 2) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "n/a"
    return f"{v:.{decimals}f}"


def print_summary(
    overall:   dict,
    per_group: dict[str, dict],
    extractor: MetricExtractor | None,
) -> None:
    """Print a formatted summary table to stdout."""
    fixed_cols = ["group", "n_episodes", "success_rate",
                  "mean_total_reward", "std_total_reward"]
    extra_cols = extractor.display if extractor else []
    all_cols   = fixed_cols + extra_cols

    col_w  = 14
    sep    = "─" * ((col_w + 2) * len(all_cols) + 1)
    header = "  ".join(f"{c:>{col_w}}" for c in all_cols)

    def _row(m: dict) -> str:
        cells = []
        for c in all_cols:
            v = m.get(c, float("nan"))
            if c == "group":
                cells.append(f"{str(v):>{col_w}}")
            elif c == "n_episodes":
                cells.append(f"{int(v):>{col_w}}")
            elif c == "success_rate":
                cells.append(f"{_fmt_pct(v):>{col_w}}")
            else:
                cells.append(f"{_fmt_f(v):>{col_w}}")
        return "  ".join(cells)

    print(f"\n{sep}")
    print("  EVALUATION SUMMARY")
    print(sep)
    print(header)
    print(sep)
    for m in per_group.values():
        print(_row(m))
    print(sep)
    print(_row(overall))
    print(sep)
    print()


def save_csv(records: list[EpisodeRecord], path: str) -> None:
    if not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"📄 Episode CSV  → {path}")


def save_json(overall: dict, per_group: dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _clean(d: dict) -> dict:
        return {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in d.items()}

    payload = {
        "overall":   _clean(overall),
        "per_group": {g: _clean(m) for g, m in per_group.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"📊 Metrics JSON → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained model with detailed metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id",    type=str, required=True,
                   help="Run ID to evaluate (e.g. 20260331_134059).")
    p.add_argument("--episodes",  type=int, default=None,
                   help="Episodes to run. Defaults to cfg.session.eval_episodes.")
    p.add_argument("--runways",   nargs="+", default=None, metavar="RWY",
                   help="Runway IDs to evaluate on.")
    p.add_argument("--no-render", action="store_true",
                   help="Disable renderer for faster headless evaluation.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    bluesky_gym.register_envs()
    cfg = ExperimentConfig.load(args.run_id)

    n_episodes = args.episodes if args.episodes is not None else cfg.session.eval_episodes
    runways    = args.runways

    print(f"\n🔍 Evaluating run  {cfg.run_id}")
    print(f"   env     = {cfg.session.env_name}")
    print(f"   algo    = {cfg.model.algorithm.__name__}")
    print(f"   model   = {cfg.save_path}/final_model.zip")
    print(f"   n_eps   = {n_episodes}")
    print(f"   runways = {runways or cfg.session.eval_runways}\n")

    # The extractor is a classmethod on the experiment — env-specific
    # knowledge stays in experiment.py, not here.
    extractor = PathPlanningExperiment.metric_extractor()

    records = run_evaluation(
        cfg        = cfg,
        extractor  = extractor,
        n_episodes = n_episodes,
        runways    = runways,
        render     = not args.no_render,
    )

    overall, per_group = aggregate_metrics(records, extractor)
    print_summary(overall, per_group, extractor)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"eval_{cfg.run_id}_{ts}"
    save_csv(records,             os.path.join(cfg.save_path, f"{stem}.csv"))
    save_json(overall, per_group, os.path.join(cfg.save_path, f"{stem}.json"))


if __name__ == "__main__":
    main()