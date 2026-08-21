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
  - ``lambda_achieved = L_mean_full_span / W`` (ac/h) -- an independent
    throughput estimate via Little's Law, L=lambda*W, computed over the
    **full episode span** ``[0, max(terminal_times)]`` (system-empty at
    t=0), NOT the windowed ``[first successful landing, last successful
    landing]`` span Gamma now uses (post-``f20e155``). This is deliberate,
    not an oversight: Little's Law's L=lambda*W identity only holds exactly
    when the observation window starts (and ends) at a system-empty state.
    Verified empirically (2026-08-20) that the Gamma-matched window does
    NOT satisfy this -- at cap=50, 47 of 50 aircraft are already airborne
    at the first landing (window start), producing a systematic 15-36%
    low bias in a windowed lambda_achieved that grows with cap. Reported
    here as an independent full-span throughput estimate, not a
    window-matched confirmation of the current Gamma -- the two are
    expected to diverge by construction (Gamma's window deliberately
    excludes the empty-to-first-landing transit; this metric's does not),
    see ``lambda_vs_gamma_pct`` below for exactly how much.
  - ``L_mean``/``utilization`` (unaffected by the above): computed over the
    Gamma-matched window ``[first, last successful landing]`` -- a direct
    occupancy measure, no Little's-Law self-consistency assumption
    involved, so the window mismatch above doesn't affect it.
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
_CAP_HYST_DIR_RE = re.compile(r"cap(\d+)_hyst(\d+)$")


def _find_cap_dirs(sweep_root: str, combo: str, hyst_s: float | None = None) -> Dict[int, str]:
    """Map cap value -> combo directory, for every cap directory under
    ``sweep_root`` that has a parquet file.

    Supports two layouts: the Round 5 ``cap_<N>/<combo>/`` layout (matched by
    ``_CAP_DIR_RE``), and the hysteresis-resweep ``cap<N>_hyst<S>/<combo>/``
    layout (matched by ``_CAP_HYST_DIR_RE``, filtered to ``hyst_s`` when given
    -- the resweep root has 4 hysteresis dirs per cap, so without a filter,
    caps would collide across slices)."""
    caps: Dict[int, str] = {}
    if not os.path.isdir(sweep_root):
        return caps
    for name in sorted(os.listdir(sweep_root)):
        m = _CAP_DIR_RE.match(name)
        if m:
            cap = int(m.group(1))
        else:
            m = _CAP_HYST_DIR_RE.match(name)
            if not m:
                continue
            if hyst_s is not None and float(m.group(2)) != hyst_s:
                continue
            cap = int(m.group(1))
        combo_dir = os.path.join(sweep_root, name, combo)
        if os.path.exists(os.path.join(combo_dir, "cps_eval_aircraft.parquet")):
            caps[cap] = combo_dir
    return caps


def _episode_concurrency(
    spawn_times: np.ndarray, terminal_times: np.ndarray, success_mask: np.ndarray
) -> Tuple[float, float]:
    """Time-averaged and peak simultaneous occupancy for one episode, from
    per-aircraft ``[spawn_time, terminal_time]`` intervals.

    Episode window is ``[min, max]`` of *successful* landing times only --
    matching the current Gamma definition (``cps_metrics_offline.py``, post
    ``f20e155``: per-episode span is first-to-last *successful* landing, not
    ``[0, last landing]``). This used to be ``[0, max(terminal_times)]`` to
    match Gamma's *old* convention; that convention no longer exists, so
    ``lambda_achieved`` silently stopped being comparable to Gamma until this
    fix. Episodes with fewer than 2 successful landings have no defined span
    (mirrors ``metrics.py``'s own exclusion) and return NaN.
    """
    if len(spawn_times) == 0 or int(np.sum(success_mask)) < 2:
        return float("nan"), float("nan")
    t0 = float(terminal_times[success_mask].min())
    t1 = float(terminal_times[success_mask].max())
    if t1 <= t0:
        return float("nan"), float("nan")
    events = np.concatenate([spawn_times, terminal_times])
    deltas = np.concatenate([np.ones_like(spawn_times), -np.ones_like(terminal_times)])
    # Slot recycling is atomic: a terminating aircraft's freed slot is
    # reused for the next arrival in the same control step, so a tied
    # (spawn_time == terminal_time) pair must never register as N+1 -- sort
    # departures (-1) before arrivals (+1) at equal timestamps.
    order = np.lexsort((deltas, events))
    events, deltas = events[order], deltas[order]
    n = np.cumsum(deltas)
    # Area under the step function n(t), clipped to the [t0, t1] window --
    # each event-to-event interval contributes only the portion (if any)
    # that overlaps the window.
    boundaries = np.concatenate([[0.0], events])
    n_before = np.concatenate([[0.0], n])
    starts, ends = boundaries[:-1], boundaries[1:]
    widths = np.clip(np.minimum(ends, t1) - np.maximum(starts, t0), 0.0, None)
    area = float(np.sum(widths * n_before[:-1]))
    in_window = (starts < t1) & (ends > t0)
    peak = float(np.max(n_before[:-1][in_window])) if np.any(in_window) else 0.0
    return area / (t1 - t0), peak


def _episode_concurrency_full_span(
    spawn_times: np.ndarray, terminal_times: np.ndarray
) -> Tuple[float, float]:
    """Time-averaged and peak simultaneous occupancy over ``[0, max(terminal_times)]``
    -- the pre-2026-08-20 window, kept deliberately (not a stale leftover) for
    ``lambda_achieved``: Little's Law's L=lambda*W identity requires an
    observation window that starts at a system-empty state, which ``t=0``
    (episode reset, nothing spawned yet) satisfies and the Gamma-matched
    ``[first, last successful landing]`` window (see ``_episode_concurrency``)
    does not -- see that function's docstring and the module docstring for
    the empirical justification.
    """
    if len(spawn_times) == 0:
        return 0.0, 0.0
    events = np.concatenate([spawn_times, terminal_times])
    deltas = np.concatenate([np.ones_like(spawn_times), -np.ones_like(terminal_times)])
    order = np.lexsort((deltas, events))
    events, deltas = events[order], deltas[order]
    n = np.cumsum(deltas)
    episode_span = float(terminal_times.max())
    if episode_span <= 0:
        return 0.0, float(n.max())
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
    l_means_full: List[float] = []
    peak_window_landings: List[int] = []
    n_excluded = 0
    for ep in np.unique(episode_ids):
        mask = episode_ids == ep
        ep_success = aircraft_df["success"].to_numpy()[mask]
        l_mean, l_peak = _episode_concurrency(spawn[mask], terminal[mask], ep_success)
        if np.isnan(l_mean):
            n_excluded += 1
        else:
            l_means.append(l_mean)
            l_peaks.append(l_peak)
        l_mean_full, _ = _episode_concurrency_full_span(spawn[mask], terminal[mask])
        l_means_full.append(l_mean_full)
        ep_terminal = terminal[mask]
        peak_window_landings.append(int(np.sum(ep_success & (ep_terminal <= spawn_window_s))))

    # Episodes with <2 successful landings have no defined Gamma-window span
    # (mirrors metrics.py's own exclusion, see _episode_concurrency) --
    # excluded from L_mean/utilization, not zero-filled. l_means_full (used
    # only for lambda_achieved) has no such exclusion -- its window starts at
    # t=0 regardless of landing count.
    l_mean_arr, l_peak_arr = np.array(l_means), np.array(l_peaks)
    l_mean_val = float(np.mean(l_mean_arr)) if len(l_mean_arr) else float("nan")
    l_mean_full_val = float(np.mean(l_means_full))
    lambda_achieved = (l_mean_full_val / w_mean) * 3600.0 if w_mean > 0 else float("nan")
    lambda_peak_window = float(np.mean(peak_window_landings)) / (spawn_window_s / 3600.0)

    return {
        "n_episodes": int(len(np.unique(episode_ids))),
        "n_excluded_lt2_landings": n_excluded,
        "w_mean_s": round(w_mean, 1),
        "w_std_s": round(w_std, 1),
        "L_mean": round(l_mean_val, 3),
        "L_mean_std": round(float(np.std(l_mean_arr, ddof=1)), 3) if len(l_mean_arr) >= 2 else float("nan"),
        # Caveat: peak concurrency is reconstructed from spawn/terminal
        # estimates with ~1 control-step (120s) of quantization noise per
        # aircraft (spawn_time isn't itself in telemetry, only
        # actual_landing_time and traj_x length) -- treat as approximate,
        # not exact. L_mean/utilization below don't have this problem since
        # per-aircraft noise averages out over the full window.
        "L_peak_mean_approx": round(float(np.mean(l_peak_arr)), 2) if len(l_peak_arr) else float("nan"),
        "L_peak_max_approx": round(float(np.max(l_peak_arr)), 2) if len(l_peak_arr) else float("nan"),
        # L_mean_full_span/lambda_achieved use the [0, last landing] window
        # (system-empty at t=0), deliberately NOT the Gamma-matched window
        # L_mean/utilization use -- see _episode_concurrency_full_span.
        "L_mean_full_span": round(l_mean_full_val, 3),
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
                   help="Root dir containing cap_<N>/<combo>/ (Round 5 layout) or "
                        "cap<N>_hyst<S>/<combo>/ (hysteresis-resweep layout) subdirs. "
                        "Repeatable to combine multiple roots.")
    p.add_argument("--combo", type=str, default="k3_dynamic",
                   help="Combo subdirectory name under each cap dir.")
    p.add_argument("--hyst-s", type=float, default=None,
                   help="For the cap<N>_hyst<S>/ layout only: filter to this "
                        "hysteresis value (each cap has one dir per candidate). "
                        "Required to avoid cap collisions when a sweep root has "
                        "more than one hysteresis value present.")
    p.add_argument("--spawn-window-s", type=float, default=2400.0,
                   help="Nominal arrival window, for lambda_peak_window.")
    p.add_argument("--diverge-threshold", type=float, default=0.10,
                   help="Flag caps where |lambda_achieved - Gamma| / Gamma exceeds this fraction.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    caps: Dict[int, str] = {}
    for root in args.sweep_root:
        found = _find_cap_dirs(root, args.combo, hyst_s=args.hyst_s)
        overlap = set(found) & set(caps)
        if overlap:
            raise SystemExit(f"cap value(s) {overlap} present in more than one --sweep-root")
        caps.update(found)

    if not caps:
        raise SystemExit(
            "No cap_<N>/<combo>/ or cap<N>_hyst<S>/<combo>/ directories with telemetry "
            "found. If using the hysteresis-resweep layout, pass --hyst-s to select one "
            "slice (e.g. --hyst-s 240)."
        )

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
            "L_peak_mean_approx", "L_peak_max_approx", "L_mean_full_span",
            "w_mean_s", "w_std_s", "lambda_achieved_ach", "lambda_vs_gamma_pct",
            "lambda_peak_window_ach"]
    print(df[cols].to_string())
    print("\n(L_peak_*_approx: reconstructed from actual_landing_time and "
          "len(traj_x)*ACTION_TIME with ~1 control-step of quantization noise "
          "per aircraft -- treat as approximate. utilization = L_mean/cap is "
          "the robust cap-binding signal, computed over the Gamma-matched "
          "[first,last successful landing] window. lambda_achieved uses "
          "L_mean_full_span instead (the [0,last landing] window) -- by "
          "construction it is NOT expected to closely match the current, "
          "windowed Gamma; see module docstring for why a Gamma-matched "
          "window breaks Little's Law's L=lambda*W identity here.)")

    if df["gamma"].notna().any():
        diverging = df[df["lambda_vs_gamma_pct"].abs() > args.diverge_threshold * 100.0]
        if not diverging.empty:
            print(f"\nCaps where |lambda_achieved - Gamma| / Gamma > "
                  f"{args.diverge_threshold * 100:.0f}%: {list(diverging.index)} "
                  f"-- expected/by construction, see note above, not a red flag.")
        else:
            print(f"\nNo cap diverges from Gamma by more than {args.diverge_threshold * 100:.0f}%.")
    else:
        print("\n(No cps_metrics_offline.yaml found alongside the telemetry -- "
              "gamma/lambda_vs_gamma_pct left NaN. Compare lambda_achieved "
              "against Gamma from analysis_summary.csv or similar externally.)")


if __name__ == "__main__":
    main()
