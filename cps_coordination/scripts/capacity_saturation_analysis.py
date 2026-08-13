"""
cps_coordination/scripts/capacity_saturation_analysis.py
-------------------------------------------------------------
Independent, non-Gamma-makespan-based throughput/concurrency cross-check for
the max_concurrent_aircraft capacity sweep -- see
``.claude/plans/max_concurrent_aircraft_capacity_sweep.md``, "Remediation plan
for two audit attention points", Attention point A, steps 4-5.

Gamma (``cps_metrics_offline.py::recompute_metrics``) divides total successful
landings by the summed *per-episode makespan* -- the same quantity the plan
doc's "Rounds 1-4" section already flags as inflated by pre-spawn/tail
queuing under a tight cap, not true per-aircraft dwell time. This script
computes two things Gamma's makespan denominator can't provide, per cap
directory:

  - ``W``: true per-aircraft dwell time, ``len(traj_x) * ACTION_TIME``
    (mean/std across every aircraft row, success or failure -- verified
    against the Round 5 parquet files that ``actual_landing_time`` and
    ``traj_x`` are populated for every row regardless of outcome).
  - ``L_mean``: time-averaged concurrent-aircraft count per episode, from
    each aircraft's occupancy interval ``[spawn_time, terminal_time]``
    (``spawn_time := terminal_time - W_i``, ``terminal_time :=
    actual_landing_time``). Mean/std taken across episodes.
  - ``L_peak``: each episode's *maximum* simultaneous occupancy, mean/max
    across episodes -- the "is the cap ever actually binding" diagnostic
    (plan doc step 4): if ``L_peak`` sits well below the cap, natural demand
    never reaches it at this schedule.
  - ``lambda_achieved = L_mean / W`` (ac/h) -- an independent throughput
    estimate via Little's Law, L=lambda*W. Note this is constructed from the
    same occupancy-interval data as L_mean and W, so algebraically it mostly
    recovers (all aircraft)/(episode span) over whatever window the
    occupancy intervals span -- it will only diverge materially from Gamma
    where Gamma's N (successful landings only) or T (successful-landings
    makespan only) differ from this script's N (every aircraft) or T
    (full occupancy span). Reported for direct comparison, not assumed to
    diverge.
  - ``lambda_peak_window`` (ac/h) -- landings-per-hour counted *only* within
    the nominal ``spawn_window_s`` (excludes any post-window holding tail
    entirely). This is the metric that most directly tests the plan doc's
    actual worry: whether the system is closer to the ~90 ac/h RECAT-EU
    ceiling during the genuinely busy period than Gamma's tail-inflated
    denominator suggests.

Usage
-----
  python cps_coordination/scripts/capacity_saturation_analysis.py \\
      --sweep-root experiments/cps_eval/capacity_sweep_50ac_fw_removed \\
      --sweep-root experiments/cps_eval/capacity_sweep_50ac_infill \\
      --combo k3_dynamic --spawn-window-s 2400
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from bluesky_gym.envs.pathplanning_goal_env import ACTION_TIME
from cps_coordination.scripts.cps_metrics_offline import load_telemetry

_CAP_DIR_RE = re.compile(r"cap_(\d+)$")


def _find_cap_dirs(sweep_root: str, combo: str) -> Dict[int, str]:
    """Map cap value -> combo directory, for every ``cap_<N>/<combo>/`` under
    ``sweep_root`` that has a parquet file (matches both the Round 5 layout
    and any new infill sweep root laid out the same way)."""
    caps: Dict[int, str] = {}
    if not os.path.isdir(sweep_root):
        return caps
    for name in sorted(os.listdir(sweep_root)):
        m = _CAP_DIR_RE.match(name)
        if not m:
            continue
        combo_dir = os.path.join(sweep_root, name, combo)
        if os.path.exists(os.path.join(combo_dir, "cps_eval_aircraft.parquet")):
            caps[int(m.group(1))] = combo_dir
    return caps


def _episode_concurrency(
    spawn_times: np.ndarray, terminal_times: np.ndarray
) -> Tuple[float, float]:
    """Time-averaged and peak simultaneous occupancy for one episode, from
    per-aircraft ``[spawn_time, terminal_time]`` intervals.

    Episode window is taken as ``[0, max(terminal_times)]`` -- the spawn
    schedule starts at episode reset (t=0), matching Gamma's own convention
    of measuring elapsed time from episode start.
    """
    if len(spawn_times) == 0:
        return 0.0, 0.0
    events = np.concatenate([spawn_times, terminal_times])
    deltas = np.concatenate([np.ones_like(spawn_times), -np.ones_like(terminal_times)])
    # Slot recycling is atomic: a terminating aircraft's freed slot is
    # reused for the next arrival in the same control step, so a tied
    # (spawn_time == terminal_time) pair must never register as N+1 -- sort
    # departures (-1) before arrivals (+1) at equal timestamps.
    order = np.lexsort((deltas, events))
    events, deltas = events[order], deltas[order]
    n = np.cumsum(deltas)
    episode_span = float(terminal_times.max())
    if episode_span <= 0:
        return 0.0, float(n.max())
    # Area under the step function n(t): sum of n(t) held between consecutive
    # event times, from 0 to the first event and between each following pair.
    boundaries = np.concatenate([[0.0], events])
    n_before = np.concatenate([[0.0], n])
    widths = np.diff(boundaries)
    area = float(np.sum(widths * n_before[:-1]))
    return area / episode_span, float(n.max())


def analyze_cap_dir(save_path: str, spawn_window_s: float) -> Dict[str, float]:
    aircraft_df, _ = load_telemetry(save_path)
    if aircraft_df.empty:
        return {"error": float("nan")}

    w_per_aircraft = aircraft_df["traj_x"].apply(len).to_numpy(dtype=float) * ACTION_TIME
    w_mean, w_std = float(np.mean(w_per_aircraft)), float(np.std(w_per_aircraft, ddof=1))

    terminal = aircraft_df["actual_landing_time"].to_numpy(dtype=float)
    spawn = terminal - w_per_aircraft
    episode_ids = aircraft_df["episode_id"].to_numpy()

    # `acid` literally is the slot index (env spawns as f"AC{slot:03d}"), so
    # within one episode there are exactly `cap` distinct acids and same-acid
    # generations can never overlap -- but len(traj_x)*ACTION_TIME has ~1
    # control-step (120s) of quantization noise per aircraft, which was
    # producing spurious same-slot overlaps (a reconstructed peak concurrency
    # above the physical cap). Clip each acid's estimated spawn time to be no
    # earlier than that same acid's own previous generation's landing time --
    # justified directly by the slot-reuse architecture, not a fudge.
    df_tmp = pd.DataFrame({
        "episode_id": episode_ids, "acid": aircraft_df["acid"].to_numpy(),
        "spawn": spawn, "terminal": terminal,
    }).sort_values(["episode_id", "acid", "terminal"])
    prev_terminal = df_tmp.groupby(["episode_id", "acid"])["terminal"].shift(1)
    df_tmp["spawn"] = np.maximum(df_tmp["spawn"], prev_terminal.fillna(-np.inf))
    df_tmp = df_tmp.sort_index()
    spawn = df_tmp["spawn"].to_numpy()

    l_means: List[float] = []
    l_peaks: List[float] = []
    peak_window_landings: List[int] = []
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        l_mean, l_peak = _episode_concurrency(spawn[mask], terminal[mask])
        l_means.append(l_mean)
        l_peaks.append(l_peak)
        ep_success = aircraft_df["success"].to_numpy()[mask]
        ep_terminal = terminal[mask]
        peak_window_landings.append(int(np.sum(ep_success & (ep_terminal <= spawn_window_s))))

    l_mean_arr, l_peak_arr = np.array(l_means), np.array(l_peaks)
    l_mean_val = float(np.mean(l_mean_arr))
    lambda_achieved = (l_mean_val / w_mean) * 3600.0 if w_mean > 0 else float("nan")
    lambda_peak_window = float(np.mean(peak_window_landings)) / (spawn_window_s / 3600.0)

    return {
        "n_episodes": int(len(l_means)),
        "w_mean_s": round(w_mean, 1),
        "w_std_s": round(w_std, 1),
        "L_mean": round(l_mean_val, 3),
        "L_mean_std": round(float(np.std(l_mean_arr, ddof=1)), 3) if len(l_mean_arr) >= 2 else float("nan"),
        # Caveat: peak concurrency is reconstructed from spawn/terminal
        # estimates with ~1 control-step (120s) of quantization noise per
        # aircraft (spawn_time isn't itself in telemetry, only
        # actual_landing_time and traj_x length) -- treat as approximate,
        # not exact. L_mean/utilization below don't have this problem since
        # per-aircraft noise averages out over the full episode window.
        "L_peak_mean_approx": round(float(np.mean(l_peak_arr)), 2),
        "L_peak_max_approx": round(float(np.max(l_peak_arr)), 2),
        "lambda_achieved_ach": round(lambda_achieved, 3),
        "lambda_peak_window_ach": round(lambda_peak_window, 3),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Little's-Law (L, W, lambda_achieved) and peak-window throughput "
            "cross-check against Gamma's makespan-based figure, per cap "
            "directory in one or more capacity-sweep roots."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sweep-root", type=str, action="append", required=True,
                   help="Root dir containing cap_<N>/<combo>/ subdirs. Repeatable "
                        "to combine the Round 5 root and any new infill root.")
    p.add_argument("--combo", type=str, default="k3_dynamic",
                   help="Combo subdirectory name under each cap_<N>/.")
    p.add_argument("--spawn-window-s", type=float, default=2400.0,
                   help="Nominal arrival window, for lambda_peak_window.")
    p.add_argument("--diverge-threshold", type=float, default=0.10,
                   help="Flag caps where |lambda_achieved - Gamma| / Gamma exceeds this fraction.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    caps: Dict[int, str] = {}
    for root in args.sweep_root:
        found = _find_cap_dirs(root, args.combo)
        overlap = set(found) & set(caps)
        if overlap:
            raise SystemExit(f"cap value(s) {overlap} present in more than one --sweep-root")
        caps.update(found)

    if not caps:
        raise SystemExit("No cap_<N>/<combo>/ directories with telemetry found.")

    rows = []
    for cap in sorted(caps):
        save_path = caps[cap]
        result = analyze_cap_dir(save_path, args.spawn_window_s)
        gamma_yaml = os.path.join(save_path, "cps_metrics_offline.yaml")
        gamma = float("nan")
        if os.path.exists(gamma_yaml):
            with open(gamma_yaml) as fh:
                gamma = float(yaml.safe_load(fh).get("gamma", float("nan")))
        result["cap"] = cap
        result["gamma"] = gamma
        # Robust "is the cap ever binding" signal: mean fraction of the
        # cap's own slot-time actually occupied, direct from L_mean (no
        # peak-reconstruction noise) -- well below 1.0 means slots sit idle
        # on average, i.e. natural demand doesn't need this many slots.
        result["utilization"] = round(result["L_mean"] / cap, 3) if cap > 0 else float("nan")
        if gamma and not np.isnan(gamma):
            result["lambda_vs_gamma_pct"] = round(
                100.0 * (result["lambda_achieved_ach"] - gamma) / gamma, 1
            )
        else:
            result["lambda_vs_gamma_pct"] = float("nan")
        rows.append(result)

    df = pd.DataFrame(rows).set_index("cap")
    cols = ["n_episodes", "gamma", "L_mean", "L_mean_std", "utilization",
            "L_peak_mean_approx", "L_peak_max_approx",
            "w_mean_s", "w_std_s", "lambda_achieved_ach", "lambda_vs_gamma_pct",
            "lambda_peak_window_ach"]
    print(df[cols].to_string())
    print("\n(L_peak_*_approx: reconstructed from actual_landing_time and "
          "len(traj_x)*ACTION_TIME with ~1 control-step of quantization noise "
          "per aircraft -- treat as approximate. utilization = L_mean/cap is "
          "the robust cap-binding signal.)")

    diverging = df[df["lambda_vs_gamma_pct"].abs() > args.diverge_threshold * 100.0]
    if not diverging.empty:
        print(f"\nCaps where |lambda_achieved - Gamma| / Gamma > "
              f"{args.diverge_threshold * 100:.0f}%: {list(diverging.index)}")
    else:
        print(f"\nNo cap diverges from Gamma by more than {args.diverge_threshold * 100:.0f}%.")


if __name__ == "__main__":
    main()
