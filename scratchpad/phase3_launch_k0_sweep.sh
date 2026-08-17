#!/bin/bash
# scratchpad/phase3_launch_k0_sweep.sh
# --------------------------------------
# Phase 3 of .claude/plans/stall_rate_investigation.md: post-fix k_cps=0
# (near-FCFS) isolation sweep, same 4 caps / M=30 / seed_base as the
# capacity_sweep_50ac_surrogate_fix k3_dynamic_fw0.5 sweep, so it diffs
# cleanly against it. Isolates how much of the residual (post dynamic-
# reassignment-fix, post surrogate-fix) stall-rate rise is intrinsic to
# k-CPS/greedy scheduling vs. a floor set by raw RECAT-EU-separation physics.
#
# Resolved config (echoed per CLAUDE.md's "Before Launching Long Runs" rule --
# this mirrors the already-approved capacity_sweep_50ac_surrogate_fix launch,
# only k_cps differs):
#   worker RUN_ID=20260615_095840, eta_surrogate=cps_coordination/models/eta_surrogate.pkl
#   (corrected model), runways=[18R,27], delta_t_plan=120, k_cps=0, mode=dynamic,
#   fairness_weight=0.5, total_arrivals_per_episode=50, spawn_window_s=2400,
#   caps={10,20,35,50}, episodes=30, seed_base=1000,
#   --disable-cross-cycle-runway-seeding
#
# Usage: bash scratchpad/phase3_launch_k0_sweep.sh
set -x
cd /Users/jackagterdenbos/dev/GitHub/bluesky-gym

mkdir -p experiments/cps_eval/capacity_sweep_50ac_surrogate_fix_k0
for N in 10 20 35 50; do
  echo "=== launching k0 cap=$N ==="
  uv run python cps_coordination/scripts/run_batch_eval.py \
    --run-id 20260615_095840 \
    --config cps_coordination/configs/cps_scale_10k.yaml \
    --episodes 30 \
    --k-cps-sweep 0 \
    --mode-sweep dynamic \
    --fairness-weight-sweep 0.5 \
    --disable-cross-cycle-runway-seeding \
    --max-concurrent-aircraft "$N" \
    --total-arrivals-per-episode 50 \
    --spawn-window-s 2400 \
    --eta-surrogate-path cps_coordination/models/eta_surrogate.pkl \
    --save-path-root "experiments/cps_eval/capacity_sweep_50ac_surrogate_fix_k0/cap_${N}" \
    --seed-base 1000 > "/tmp/surrogate_fix_sweep_k0_cap${N}.log" 2>&1
  echo "=== k0 cap=$N exit code: $? ==="
done
echo "ALL DONE" > /tmp/surrogate_fix_sweep_k0_DONE.marker
