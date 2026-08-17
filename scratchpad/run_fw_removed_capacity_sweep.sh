#!/bin/bash
set -e
cd /Users/jackagterdenbos/dev/GitHub/bluesky-gym
for N in 10 20 35 50; do
  uv run python cps_coordination/scripts/run_batch_eval.py \
    --run-id 20260615_095840 \
    --config cps_coordination/configs/cps_scale_10k.yaml \
    --episodes 30 \
    --k-cps-sweep 3 \
    --mode-sweep dynamic \
    --disable-cross-cycle-runway-seeding \
    --max-concurrent-aircraft "$N" \
    --total-arrivals-per-episode 50 \
    --spawn-window-s 2400 \
    --eta-surrogate-path cps_coordination/models/eta_surrogate.pkl \
    --save-path-root "experiments/cps_eval/capacity_sweep_50ac_fw_removed/cap_${N}" \
    --seed-base 1000 \
    > "/tmp/fw_removed_capacity_sweep_cap${N}.log" 2>&1
done
touch /tmp/fw_removed_capacity_sweep_DONE.marker
