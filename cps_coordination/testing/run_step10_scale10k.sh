#!/usr/bin/env bash
# cps_coordination/testing/run_step10_scale10k.sh
# --------------------------------------------------------------
# Production launch script for the M=10,000-episode CPS coordination
# scale-up evaluation (Phase III roadmap Step 10). Wraps
# cps_coordination/testing/run_batch_eval.py with the settings validated
# throughout the pre-launch audit (see .claude/plans/phase3_cps_coordination_plan.md
# and .claude/plans/m10000_launch_readiness_plan.md) -- do not run this
# locally end-to-end; it is sized for a cluster/runner.
#
# WALL-CLOCK COST (measured, not guessed): a capped M=30, spawn_window_s=1800,
# ratchet-off verification sweep of all 8 combos completed in ~21 minutes on
# this machine (~158s/combo at M=30). Extrapolated linearly to M=10,000:
#   ~14.7h PER COMBO
#   ~4.9 DAYS for all 8 combos run sequentially (this script's default mode --
#   run_batch_eval.py processes combos one after another in a single process)
# If your cluster can run more than one process concurrently, launch one
# job per combo instead (see COMBO mode below) -- 8-way parallel cuts this
# to ~14.7h wall-clock. Real runtime will vary with cluster hardware; this
# is a same-machine extrapolation, not a promise.
#
# VALIDATED GRID -- this script pins k_cps/mode/fairness_weight explicitly
# rather than relying on run_batch_eval.py's own bare CLI defaults
# (k_cps=[0,1,3] x mode x fairness_weight=[0.0], 6 combos): every sanity
# sweep, deep-analysis pass, and ratchet on/off comparison this whole
# investigation is built on used k_cps in {0,3} x mode in {static,dynamic}
# x fairness_weight in {0.0,0.3} (8 combos) instead. Launching with the
# script's bare defaults would silently evaluate an untested grid.
#
# RATCHET DEFAULT -- --disable-cross-cycle-runway-seeding is passed below.
# This is the production default per the pre-registered Vector-1 GO decision
# (success_rate +51.5pp, c_sep improves, confirmed at real launch scale in a
# capped M=30/spawn_window_s=1800 verification: stall_rate drops to exactly
# 0% in all 8 combos, runway-hijack signature drops to baseline). It is a
# CLI flag, not a cps_scale_10k.yaml field -- CPSModelConfig.
# enable_cross_cycle_runway_seeding's dataclass default (True) is
# deliberately left alone; this script is what actually applies the decision.
#
# RESUME CAVEAT (known limitation, not fixed here): run_batch_eval.py has no
# episode-level checkpoint/resume -- SIGINT/SIGTERM flushes and closes the
# in-flight combo's Parquet cleanly, but restarting always begins that
# combo's episode loop at ep_idx=0. --no-fresh-start appends rather than
# overwrites, so restarting an interrupted combo without it will silently
# duplicate episode_ids already logged. If a multi-day cluster job might be
# preempted, prefer one job per combo (below) so a failure only costs one
# combo's progress, not the whole sweep, and treat any combo that had to be
# restarted as needing a fresh save-path (don't reuse a partially-written one).
#
# Usage
# -----
#   # Full 8-combo sweep, sequential (~4.9 days, only if your cluster truly
#   # runs this as one long single job):
#   RUN_ID=20260615_095840 ./cps_coordination/testing/run_step10_scale10k.sh
#
#   # One combo per job (recommended for real cluster use -- launch 8 of
#   # these, one per k_cps/mode/fairness_weight combination, however your
#   # scheduler fans out jobs):
#   RUN_ID=20260615_095840 COMBO="3:dynamic:0.3" ./cps_coordination/testing/run_step10_scale10k.sh

set -euo pipefail

# --- REQUIRED ---
RUN_ID="${RUN_ID:?Set RUN_ID to the frozen worker run_id under experiments/PathPlanningGoalEnv-v0/SAC/models/ (e.g. 20260615_095840, the latest run with a final_model.zip as of this script's writing -- verify with: python3 -c \"import glob,os; c=sorted(glob.glob('experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip')); print(os.path.basename(os.path.dirname(c[-1])))\").}"

# --- optional overrides ---
CONFIG="${CONFIG:-cps_coordination/configs/cps_scale_10k.yaml}"
ETA_SURROGATE_PATH="${ETA_SURROGATE_PATH:-cps_coordination/models/eta_surrogate.pkl}"
EPISODES="${EPISODES:-10000}"
SEED_BASE="${SEED_BASE:-0}"
SAVE_ROOT="${SAVE_ROOT:-experiments/cps_eval/scale_10k_$(date +%Y%m%d_%H%M%S)}"

# --- combo selection ---
# COMBO unset (default): full validated 8-combo grid, one sequential process.
# COMBO="k_cps:mode:fw" (e.g. "3:dynamic:0.3"): restrict to exactly that one
# combo -- run this script once per combo, in parallel, across cluster jobs.
if [ -n "${COMBO:-}" ]; then
    IFS=':' read -r K_CPS MODE FW <<< "$COMBO"
    K_CPS_SWEEP=("$K_CPS")
    MODE_SWEEP=("$MODE")
    FW_SWEEP=("$FW")
    echo "Single-combo mode: k_cps=$K_CPS mode=$MODE fairness_weight=$FW"
else
    K_CPS_SWEEP=(0 3)
    MODE_SWEEP=(static dynamic)
    FW_SWEEP=(0.0 0.3)
    echo "Full 8-combo sequential mode -- expect ~4.9 days wall-clock (see script header)."
    echo "Ctrl-C now if you meant to parallelize via COMBO=\"k_cps:mode:fw\" instead."
    sleep 5
fi

uv run python cps_coordination/testing/run_batch_eval.py \
    --run-id "$RUN_ID" \
    --config "$CONFIG" \
    --episodes "$EPISODES" \
    --k-cps-sweep "${K_CPS_SWEEP[@]}" \
    --mode-sweep "${MODE_SWEEP[@]}" \
    --fairness-weight-sweep "${FW_SWEEP[@]}" \
    --disable-cross-cycle-runway-seeding \
    --eta-surrogate-path "$ETA_SURROGATE_PATH" \
    --save-path-root "$SAVE_ROOT" \
    --seed-base "$SEED_BASE"

echo "Done -> $SAVE_ROOT/k<k_cps>_<mode>_fw<fairness_weight>/"
echo "Then: uv run python cps_coordination/testing/step10_deep_analysis.py --sweep-root $SAVE_ROOT"
