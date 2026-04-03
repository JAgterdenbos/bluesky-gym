"""
bluesky_gym/experiment/evaluate.py
------------------------------------
Generic evaluation script for any trained SB3 + GoalEnv model.

Extension point
---------------
Custom per-env metrics are injected via a MetricExtractor — a small
object that knows how to pull named float values out of a final episode
info dict, and how to aggregate them across episodes.  The evaluator
itself never imports anything env-specific; all that knowledge lives in
the MetricExtractor returned by your experiment's metric_extractor()
classmethod.

Built-in (env-agnostic) metrics
---------------------------------
  is_success    - from info[cfg.env.success_key]
  total_reward  - from info["total_reward"]
  group         - from info[cfg.env.group_key]  (or "all" if None)
 

Output
------
  Console  - formatted summary table (overall + per group)
  CSV      - one row per episode     → <save_path>/eval_<run_id>_<ts>.csv
  YAML     - aggregated metrics      → <save_path>/eval_<run_id>_<ts>.yaml

Usage
-----
  python evaluate.py --run-id 20260331_134059
  python evaluate.py --run-id 20260331_134059 --episodes 50
  python evaluate.py --run-id 20260331_134059 --no-render
"""

from __future__ import annotations

import argparse
import csv
import os
import yaml
from datetime import datetime
from typing import Callable, Optional, Type, TypedDict, cast
 
import bluesky_gym
import numpy as np
 
from .config import ExperimentConfig


# ---------------------------------------------------------------------------
# MetricExtractor — the env-specific extension point
# ---------------------------------------------------------------------------

class MetricExtractor:
    """Declares which extra metrics to pull from an episode's final info dict.

    Parameters
    ----------
    extractors : dict[str, Callable[[dict, bool], float]]
        Maps metric name → function(info, is_success) → float.
        Return float("nan") to mark a metric as invalid for that episode
        (e.g. flight_time on a failed episode).  nanmean is used by default
        so nans are ignored in aggregation.

    aggregators : dict[str, Callable[[list[float]], float]] | None
        Per-metric aggregation overrides.  Defaults to np.nanmean.

    display : list[str] | None
        Ordered subset of metric names shown in the console table.
        Defaults to all metrics in insertion order.
    """

    #TODO: Make extractor and aggregators return any type, not just float.  This would allow things like success-weighted flight time, which is a useful metric but doesn't fit the current float-based design.
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
        return {name: fn(info, is_success) for name, fn in self.extractors.items()}

    def aggregate(self, rows: list[dict[str, float]]) -> dict[str, float]:
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
# Episode record
# ---------------------------------------------------------------------------

class EpisodeRecord(TypedDict):
    episode:      int
    group:        str
    is_success:   bool
    total_reward: float

#TODO: make extras more flexible so it can support non-float metrics if we want to go that route in the future
    # Extra env-specific metrics go here; keys depend on the experiment's MetricExtractor
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
    cfg:            ExperimentConfig,
    experiment_cls,
    n_episodes:     int,
    groups:         Optional[list[str]],
    render:         bool,
) -> list[EpisodeRecord]:
    """Run n_episodes episodes; return one EpisodeRecord per episode."""
 
    extractor   = experiment_cls.metric_extractor()
    success_key = cfg.env.success_key
    group_key   = cfg.env.group_key
 
    eval_kwargs = cfg.eval_env_kwargs
    if groups is not None:
        from .config import _inject_groups
        _inject_groups(eval_kwargs, cfg.env.env_kwargs, groups)
 
    render_mode = "human" if render else None
    experiment  = experiment_cls(cfg)
    env         = experiment.make_env(eval_kwargs, render_mode=render_mode)
    model       = cfg.model.get_algorithm().load(f"{cfg.save_path}/final_model", env=env)
 
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
 
        is_success   = bool(info.get(success_key, False))
        total_reward = float(info.get("total_reward", 0.0))
        group        = str(info.get(group_key, "unknown")) if group_key else "all"
        extras       = extractor.extract(info, is_success) if extractor else {}
 
        rec = _make_record(ep, group, is_success, total_reward, extras)
        records.append(rec)
 
        status    = "✅" if is_success else "❌"
        extra_str = "  ".join(
            f"{k}={v:.2f}"
            for k, v in extras.items()
            if extractor and k in extractor.display
            and not (isinstance(v, float) and np.isnan(v))
        )
        print(
            f"  Ep {ep:>3}/{n_episodes}  {status}  "
            f"group={group:<6}  reward={total_reward:+.1f}"
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
        base = {"group": label, "n_episodes": 0,
                "success_rate": 0.0,
                "mean_total_reward": float("nan"),
                "std_total_reward":  float("nan")}
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
    """Return (overall_summary, per_group_summary_dict)."""
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
    return "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"

def _fmt_f(v, decimals: int = 2) -> str:
    return "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:.{decimals}f}"


def print_summary(
    overall:   dict,
    per_group: dict[str, dict],
    extractor: MetricExtractor | None,
) -> None:
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
            if   c == "group":       cells.append(f"{str(v):>{col_w}}")
            elif c == "n_episodes":  cells.append(f"{int(v):>{col_w}}")
            elif c == "success_rate":cells.append(f"{_fmt_pct(v):>{col_w}}")
            else:                    cells.append(f"{_fmt_f(v):>{col_w}}")
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


def save_yaml_summary(overall: dict, per_group: dict[str, dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _clean(d: dict) -> dict:
        return {k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in d.items()}

    payload = {
        "overall":   _clean(overall),
        "per_group": {g: _clean(m) for g, m in per_group.items()},
    }
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
    print(f"📊 Metrics YAML → {path}")


# ---------------------------------------------------------------------------
# CLI entry point (called by run_experiment)
# ---------------------------------------------------------------------------

def run_evaluate_cli(experiment_cls) -> None:
    """Standalone evaluation CLI for a given experiment class."""
    p = argparse.ArgumentParser(
        description="Evaluate a trained model with detailed metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id",    type=str, required=True)
    p.add_argument("--episodes",  type=int, default=None)
    p.add_argument("--groups",    nargs="+", default=None, metavar="GROUP",
                   help="Group IDs (runways, levels, …) to evaluate on.")
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
 
    bluesky_gym.register_envs()
 
    cfg = ExperimentConfig.load(
        args.run_id,
        model_config_cls=experiment_cls.model_config_cls,
        env_config_cls=experiment_cls.env_config_cls,
    )
 
    n_episodes = args.episodes or cfg.session.eval_episodes
    groups     = args.groups
 
    algo_name = cfg.model.algorithm.__name__ if cfg.model.algorithm else "Unspecified"

    print(f"\n🔍 Evaluating run  {cfg.run_id}")
    print(f"   env     = {cfg.env.env_name}")
    print(f"   algo    = {algo_name}")
    print(f"   model   = {cfg.save_path}/final_model.zip")
    print(f"   n_eps   = {n_episodes}")
    print(f"   groups  = {groups or cfg.session.eval_groups}\n")
 
    records = run_evaluation(
        cfg=cfg,
        experiment_cls=experiment_cls,
        n_episodes=n_episodes,
        groups=groups,
        render=not args.no_render,
    )
 
    extractor          = experiment_cls.metric_extractor()
    overall, per_group = aggregate_metrics(records, extractor)
    print_summary(overall, per_group, extractor)
 
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"eval_{cfg.run_id}_{ts}"
    save_csv(records,                     os.path.join(cfg.save_path, f"{stem}.csv"))
    save_yaml_summary(overall, per_group, os.path.join(cfg.save_path, f"{stem}.yaml"))