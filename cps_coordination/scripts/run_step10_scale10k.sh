#!/usr/bin/env bash
# cps_coordination/scripts/run_step10_scale10k.sh
# --------------------------------------------------------------
# Production launch script for the M=2,000-episode CPS coordination
# scale-up evaluation (Phase III roadmap Step 10 -- rescaled from the
# original M=10,000/low-density config toward higher per-episode traffic
# density and fewer episodes). Wraps cps_coordination/scripts/run_batch_eval.py
# with the settings validated throughout the pre-launch audit (see
# .claude/plans/phase3_cps_coordination_plan.md,
# .claude/plans/m10000_launch_readiness_plan.md, and
# .claude/plans/task-optimize-fairness-fizzy-moth.md for the density/
# fairness_weight rescale) -- do not run this locally end-to-end at full M;
# it is sized for a cluster/runner. Capped local M values (M=10, M=30) are
# fine for sanity/timing checks.
#
# WALL-CLOCK COST -- the ~65.9s/combo figure below is MEASURED, but at the
# OLD density (max_concurrent_aircraft=5, total_arrivals_per_episode=10,
# spawn_window_s=1800, updated 2026-08-04 after a same-day performance-
# optimization session -- n_jobs=1 on the ETA surrogate, Path-object
# caching, and a batched numpy _check_terminal, see
# .claude/plans/we-re-about-to-launch-delegated-hare.md and this doc's own
# "Open concerns" section): the same capped M=30, spawn_window_s=1800,
# ratchet-off verification sweep that originally measured ~158s/combo
# completes in ~57-70s/combo (~65.9s average) with all three fixes,
# extrapolating to ~6.1h/combo, ~2.0 days for 8 combos at the OLD M=10,000/
# 8-combo grid.
#
# At the NEW density (max_concurrent_aircraft=10, total_arrivals_per_episode=25,
# spawn_window_s=2400, cps_scale_10k.yaml) and M=2,000/4-combo grid,
# MEASURED via this exact script at a capped M=30 (mode-specific
# fairness_weight included): k3_static 117s, k3_dynamic 105s at M=30
# (~3.5-3.9s/episode/combo) -> ~2.1h/combo, ~8.2h sequential for all 4
# combos -- well under the original ~4.6h/combo/~18.3h estimate (see
# task-optimize-fairness-fizzy-moth.md and step10_execution_and_data_
# collection_plan.md's "Wall-clock cost" section for the full derivation
# and the verification data at cps_coordination/data/
# step10_verification_new_density_final/). If your cluster can run more
# than one process concurrently,
# launch one job per combo instead (see COMBO mode below). If it has MORE
# than 4 workers, SHARDS (below) can split a single combo's episodes across
# additional processes too. Real runtime will vary with cluster hardware;
# this is a same-machine extrapolation, not a promise.
#
# VALIDATED GRID -- this script pins k_cps/mode/fairness_weight explicitly
# rather than relying on run_batch_eval.py's own bare CLI defaults
# (k_cps=[0,1,3] x mode x fairness_weight=[0.0], 6 combos). k_cps in
# {0,3} x mode in {static,dynamic} (4 combos) was validated throughout the
# k-CPS-fix/ratchet investigation (originally alongside fairness_weight in
# {0.0,0.3}, 8 combos -- see phase3_cps_coordination_plan.md).
# fairness_weight is now FIXED (not swept) via a local Stage 1/2 calibration
# sweep -- see cps_coordination/scripts/analyze_fairness_weight_offline.py
# and task-optimize-fairness-fizzy-moth.md. Launching with the script's bare
# defaults would silently evaluate an untested grid.
#
# fairness_weight IS MODE-SPECIFIC, not one global value: the calibration
# sweep (k_cps=3, ratchet ON to get a real contention signal -- the
# production ratchet-OFF default drives stalling to ~0%, leaving nothing
# for fairness_weight to protect against, so it had to be calibrated in a
# separate diagnostic regime) found a genuine, well-resolved 2x split
# between modes: dynamic mode peaks at fw=0.5 (success_rate 0.576 vs. 0.450
# at fw=0), static mode peaks at fw=1.0 (success_rate 0.716 vs. 0.729 at
# fw=0, stall_recovery_rate 0.564 vs. 0.504 -- success dips slightly,
# stall-recovery improves substantially). No shared value serves both
# modes well (fw=0.75, the naive compromise, is actually a local dip for
# dynamic mode). STATIC_FW/DYNAMIC_FW below apply this per mode.
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
# RESUME (added 2026-08-04) -- set RESUME=1 to pass --resume through to
# run_batch_eval.py: each combo resumes from the highest episode_id already
# durably written to its cps_eval_aircraft.parquet (skipping combos that are
# already complete) instead of restarting at episode 0. Safe against both
# clean (SIGINT/SIGTERM) and hard-crash interruption -- see
# run_batch_eval.py::_resolve_resume_start/_merge_resume_delta's docstrings
# for why (short version: it derives the resume point from the Parquet
# file's own contents, never a separately-tracked counter that could desync).
#
# SHARDING (added 2026-08-04) -- set SHARDS=N and SHARD_INDEX=i (0-based) to
# split ONE combo's EPISODES across N concurrent processes instead of running
# them all in this one process. Requires COMBO=... (one job = one
# (combo, shard) pair -- sharding the full sequential 4-combo grid in a
# single process defeats the point). Each shard writes to its own
# SAVE_ROOT/shard_{i}of{N}/ directory (Parquet has no true row-group append,
# so concurrent shards must never target the same file); once all N shards
# for a combo finish, merge them with:
#   uv run python cps_coordination/scripts/merge_shards.py \
#       --save-root "$SAVE_ROOT" --combo "k3_dynamic_fw0.3" --shards 4
# which verifies there are no colliding episode_ids across shards before
# writing the merged, final-looking SAVE_ROOT/k3_dynamic_fw0.3/ directory.
#
# Usage (COMBO's fw component below must match STATIC_FW/DYNAMIC_FW for
# that combo's mode -- see those variables below)
# -----
#   # Full 4-combo sweep, sequential (~8.2h measured, only if your cluster truly
#   # runs this as one long single job -- runs TWO run_batch_eval.py
#   # invocations, one per mode with its own calibrated fairness_weight):
#   RUN_ID=20260615_095840 ./cps_coordination/scripts/run_step10_scale10k.sh
#
#   # One combo per job (recommended for real cluster use -- launch 4 of
#   # these, one per k_cps/mode combination, with the fw matching that
#   # mode's calibrated value, however your scheduler fans out jobs):
#   RUN_ID=20260615_095840 COMBO="3:dynamic:0.5" ./cps_coordination/scripts/run_step10_scale10k.sh
#   RUN_ID=20260615_095840 COMBO="3:static:1.0" ./cps_coordination/scripts/run_step10_scale10k.sh
#
#   # One combo, sharded across 4 more processes (needs 4 separate
#   # invocations, SHARD_INDEX=0..3, then merge_shards.py afterward):
#   RUN_ID=20260615_095840 COMBO="3:dynamic:0.5" SHARDS=4 SHARD_INDEX=0 ./cps_coordination/scripts/run_step10_scale10k.sh
#
#   # Resume a combo (or full sequential sweep) that was interrupted:
#   RUN_ID=20260615_095840 COMBO="3:dynamic:0.5" RESUME=1 SAVE_ROOT=experiments/cps_eval/scale_10k_20260801_000000 ./cps_coordination/scripts/run_step10_scale10k.sh

set -euo pipefail

# --- REQUIRED ---
RUN_ID="${RUN_ID:?Set RUN_ID to the frozen worker run_id under experiments/PathPlanningGoalEnv-v0/SAC/models/ (e.g. 20260615_095840, the latest run with a final_model.zip as of this writing -- verify with: python3 -c \"import glob,os; c=sorted(glob.glob('experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip')); print(os.path.basename(os.path.dirname(c[-1])))\").}"

# --- optional overrides ---
CONFIG="${CONFIG:-cps_coordination/configs/cps_scale_10k.yaml}"
ETA_SURROGATE_PATH="${ETA_SURROGATE_PATH:-cps_coordination/models/eta_surrogate.pkl}"
EPISODES="${EPISODES:-2000}"
SEED_BASE="${SEED_BASE:-0}"
SAVE_ROOT="${SAVE_ROOT:-experiments/cps_eval/scale_10k_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-}"
SHARDS="${SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

# --- calibrated, mode-specific fairness_weight (see header + task-optimize-
# fairness-fizzy-moth.md for the Stage 1/2 sweep this came from) ---
STATIC_FW="${STATIC_FW:-1.0}"
DYNAMIC_FW="${DYNAMIC_FW:-0.5}"

# --- sharding: split this combo's EPISODES across SHARDS processes ---
RUN_EPISODES="$EPISODES"
RUN_SEED_BASE="$SEED_BASE"
RUN_EPISODE_ID_OFFSET=0
if [ "$SHARDS" -gt 1 ]; then
    SHARD_SIZE=$(( (EPISODES + SHARDS - 1) / SHARDS ))
    SHARD_START=$(( SHARD_INDEX * SHARD_SIZE ))
    if [ "$SHARD_START" -ge "$EPISODES" ]; then
        echo "SHARD_INDEX=$SHARD_INDEX has no episodes to run (start=$SHARD_START >= EPISODES=$EPISODES)." >&2
        exit 1
    fi
    REMAINING=$(( EPISODES - SHARD_START ))
    if [ "$SHARD_SIZE" -lt "$REMAINING" ]; then
        SHARD_EPISODES="$SHARD_SIZE"
    else
        SHARD_EPISODES="$REMAINING"
    fi
    # This shard's own process runs a fresh, local ep_idx loop over
    # [0, SHARD_EPISODES) -- --seed-base and --episode-id-offset are what
    # re-map that local loop back onto the correct GLOBAL episode range
    # [SHARD_START, SHARD_START+SHARD_EPISODES), so seeds/episode_ids match
    # exactly what an unsharded run would have used for those same global
    # episode indices.
    RUN_EPISODES="$SHARD_EPISODES"
    RUN_SEED_BASE=$(( SEED_BASE + SHARD_START ))
    RUN_EPISODE_ID_OFFSET="$SHARD_START"
    SAVE_ROOT="${SAVE_ROOT}/shard_${SHARD_INDEX}of${SHARDS}"
    echo "Sharding: shard $SHARD_INDEX/$SHARDS -> global episodes [$SHARD_START, $((SHARD_START + SHARD_EPISODES))) of $EPISODES total"
fi

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
    RESUME_ARGS=(--resume)
fi

# One (k_cps-pair, mode) invocation of run_batch_eval.py -- fairness_weight
# is mode-specific, so each mode gets its own call rather than one call
# sweeping both modes with a single fw value.
run_one_mode() {
    local mode="$1" fw="$2"
    uv run python cps_coordination/scripts/run_batch_eval.py \
        --run-id "$RUN_ID" \
        --config "$CONFIG" \
        --episodes "$RUN_EPISODES" \
        --k-cps-sweep 0 3 \
        --mode-sweep "$mode" \
        --fairness-weight-sweep "$fw" \
        --disable-cross-cycle-runway-seeding \
        --eta-surrogate-path "$ETA_SURROGATE_PATH" \
        --save-path-root "$SAVE_ROOT" \
        --seed-base "$RUN_SEED_BASE" \
        --episode-id-offset "$RUN_EPISODE_ID_OFFSET" \
        "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"
}

# --- combo selection ---
# COMBO unset (default): full validated 4-combo grid, two sequential
# run_batch_eval.py invocations (one per mode, each with its calibrated fw).
# COMBO="k_cps:mode:fw" (e.g. "3:dynamic:0.5"): restrict to exactly that one
# combo -- run this script once per combo, in parallel, across cluster jobs.
# fw here should match STATIC_FW/DYNAMIC_FW for that combo's mode -- the
# script doesn't enforce this (COMBO is meant to let you override), but a
# mismatch silently deploys an uncalibrated value for that combo.
if [ -n "${COMBO:-}" ]; then
    IFS=':' read -r K_CPS MODE FW <<< "$COMBO"
    echo "Single-combo mode: k_cps=$K_CPS mode=$MODE fairness_weight=$FW"
    uv run python cps_coordination/scripts/run_batch_eval.py \
        --run-id "$RUN_ID" \
        --config "$CONFIG" \
        --episodes "$RUN_EPISODES" \
        --k-cps-sweep "$K_CPS" \
        --mode-sweep "$MODE" \
        --fairness-weight-sweep "$FW" \
        --disable-cross-cycle-runway-seeding \
        --eta-surrogate-path "$ETA_SURROGATE_PATH" \
        --save-path-root "$SAVE_ROOT" \
        --seed-base "$RUN_SEED_BASE" \
        --episode-id-offset "$RUN_EPISODE_ID_OFFSET" \
        "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"
    echo "Done -> $SAVE_ROOT/k${K_CPS}_${MODE}_fw${FW}/"
    if [ "$SHARDS" -gt 1 ]; then
        echo "This was shard $SHARD_INDEX/$SHARDS -- once all $SHARDS shards finish, merge with:"
        echo "  uv run python cps_coordination/scripts/merge_shards.py --save-root <root_without_shard_suffix> --combo k${K_CPS}_${MODE}_fw${FW} --shards $SHARDS"
    fi
else
    if [ "$SHARDS" -gt 1 ]; then
        echo "SHARDS>1 requires COMBO=... (one job = one (combo, shard) pair -- sharding the full sequential grid in one process defeats the point)." >&2
        exit 1
    fi
    echo "Full 4-combo sequential mode -- static fw=$STATIC_FW, dynamic fw=$DYNAMIC_FW."
    echo "Expect ~8.2h (measured) wall-clock total (see script header)."
    echo "Ctrl-C now if you meant to parallelize via COMBO=\"k_cps:mode:fw\" instead."
    sleep 5
    run_one_mode static "$STATIC_FW"
    run_one_mode dynamic "$DYNAMIC_FW"
    echo "Done -> $SAVE_ROOT/k{0,3}_static_fw${STATIC_FW}/ and $SAVE_ROOT/k{0,3}_dynamic_fw${DYNAMIC_FW}/"
fi

echo "Then: uv run python cps_coordination/scripts/step10_deep_analysis.py --sweep-root $SAVE_ROOT"
