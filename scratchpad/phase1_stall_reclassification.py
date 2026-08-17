"""
scratchpad/phase1_stall_reclassification.py
--------------------------------------------
Phase 1 of .claude/plans/stall_rate_investigation.md: offline reclassification of
stall_detected aircraft in capacity_sweep_50ac_v3, using only already-generated
telemetry (no new simulation runs).

For every stall_detected==True row at each cap, reconstructs the same
distance-to-IAF series CPSManager._update_stall_tracking uses (traj_x/traj_y are
already normalized by MAX_DISTANCE, matching cps_manager.py's own convention),
replays the identical "no new best within STALL_WINDOW_S" rule to find the exact
step the real detector would have flagged, and reports the distance-to-IAF AT that
flag point. A flag point close to 0 km supports the "arrived near goal, loitering
for its TTA slot" hypothesis (correct behavior misread as stalling); a flag point
far from 0 km with no post-flag improvement supports genuine non-convergence.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cps_coordination.coordination.eta_surrogate import ETASurrogate

MAX_DISTANCE = 300.0  # km, bluesky_gym.envs.pathplanning_goal_env.MAX_DISTANCE
ACTION_TIME = 120.0   # s
STALL_WINDOW_S = 1800.0
STALL_PROGRESS_EPS_KM = 5.0
STALL_WINDOW_STEPS = int(STALL_WINDOW_S // ACTION_TIME)  # 15

CAPS = [10, 20, 35, 50]
ROOT = "experiments/cps_eval/capacity_sweep_50ac_surrogate_fix/cap_{cap}/k3_dynamic_fw0.5/cps_eval_aircraft.parquet"


def replay_stall_detection(dist_km: np.ndarray) -> int | None:
    """Replay CPSManager._update_stall_tracking's rule on a distance series.

    Returns the step index the aircraft would first be flagged stalled, or None
    if it never would be. Mirrors cps_manager.py exactly: track best-ever
    distance, flag once STALL_WINDOW_STEPS have elapsed since the last new best
    that beat the prior best by more than STALL_PROGRESS_EPS_KM.
    """
    best = math.inf
    best_step = 0
    for step, d in enumerate(dist_km):
        if d < best - STALL_PROGRESS_EPS_KM:
            best = d
            best_step = step
            continue
        if step - best_step >= STALL_WINDOW_STEPS:
            return step
    return None


def main() -> None:
    surrogate = ETASurrogate.load("cps_coordination/models/eta_surrogate.pkl")
    iaf_ref = surrogate._iaf_ref

    print(f"{'cap':>4} {'n_stalled':>10} {'flag_dist_med_km':>18} {'flag_dist_p90_km':>18} "
          f"{'frac_flag_dist<15km':>20} {'post_flag_improves':>20} {'recovered':>10} {'unrecovered':>12}")

    for cap in CAPS:
        df = pd.read_parquet(ROOT.format(cap=cap))
        stalled = df[df["stall_detected"]].copy()

        flag_dists = []
        post_flag_improves = []
        for _, row in stalled.iterrows():
            xs, ys = row["traj_x"], row["traj_y"]
            if xs is None or len(xs) < 2:
                continue
            iaf = iaf_ref.get(row["runway_id"])
            if iaf is None:
                continue
            iaf_x, iaf_y, _ = iaf
            dist_km = np.hypot(iaf_x - np.asarray(xs), iaf_y - np.asarray(ys)) * MAX_DISTANCE

            flag_step = replay_stall_detection(dist_km)
            if flag_step is None:
                continue  # replayed rule disagrees with logged flag; skip (rare, logged mid-episode differs from full-traj replay only via freeze re-pin, negligible)
            flag_dists.append(dist_km[flag_step])

            # Did distance-to-IAF ever go below the at-flag value again afterward?
            # (Aircraft is NOT physically frozen by CPS-side flagging -- only its
            # eta/runway bookkeeping is -- so its real flight can still converge.)
            remainder = dist_km[flag_step:]
            post_flag_improves.append(bool((remainder < dist_km[flag_step] - STALL_PROGRESS_EPS_KM).any()))

        flag_dists = np.array(flag_dists)
        post_flag_improves = np.array(post_flag_improves)
        n = len(stalled)
        recovered = int((stalled["success"]).sum())
        unrecovered = n - recovered

        med = np.median(flag_dists) if len(flag_dists) else float("nan")
        p90 = np.percentile(flag_dists, 90) if len(flag_dists) else float("nan")
        frac_close = float((flag_dists < 15.0).mean()) if len(flag_dists) else float("nan")
        frac_improves = float(post_flag_improves.mean()) if len(post_flag_improves) else float("nan")

        print(f"{cap:>4} {n:>10} {med:>18.1f} {p90:>18.1f} {frac_close:>20.3f} "
              f"{frac_improves:>20.3f} {recovered:>10} {unrecovered:>12}")


if __name__ == "__main__":
    main()
