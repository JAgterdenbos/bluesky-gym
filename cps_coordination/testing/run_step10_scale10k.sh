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
# WALL-CLOCK COST (measured, not guessed, updated 2026-08-04 after a
# same-day performance-optimization session -- n_jobs=1 on the ETA
# surrogate, Path-object caching, and a batched numpy _check_terminal, see
# .claude/plans/we-re-about-to-launch-delegated-hare.md and this doc's own
# "Open concerns" section): the same capped M=30, spawn_window_s=1800,
# ratchet-off verification sweep that originally measured ~158s/combo now
# completes in ~57-70s/combo (~65.9s average) with all three fixes.
# Extrapolated linearly to M=10,000:
#   ~6.1h PER COMBO
#   ~2.0 DAYS for all 8 combos run sequentially (this script's default mode --
#   run_batch_eval.py processes combos one after another in a single process)
# If your cluster can run more than one process concurrently, launch one
# job per combo instead (see COMBO mode below) -- 8-way parallel cuts this
# to ~6.1h wall-clock. If it has MORE than 8 workers, SHARDS (below) can
# split a single combo's episodes across additional processes too. Real
# runtime will vary with cluster hardware; this is a same-machine
# extrapolation, not a promise.
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
# (combo, shard) pair -- sharding the full sequential 8-combo grid in a
# single process defeats the point). Each shard writes to its own
# SAVE_ROOT/shard_{i}of{N}/ directory (Parquet has no true row-group append,
# so concurrent shards must never target the same file); once all N shards
# for a combo finish, merge them with:
#   uv run python cps_coordination/testing/merge_shards.py \
#       --save-root "$SAVE_ROOT" --combo "k3_dynamic_fw0.3" --shards 4
# which verifies there are no colliding episode_ids across shards before
# writing the merged, final-looking SAVE_ROOT/k3_dynamic_fw0.3/ directory.
#
# Usage
# -----
#   # Full 8-combo sweep, sequential (~2.0 days, only if your cluster truly
#   # runs this as one long single job):
#   RUN_ID=20260615_095840 ./cps_coordination/testing/run_step10_scale10k.sh
#
#   # One combo per job (recommended for real cluster use -- launch 8 of
#   # these, one per k_cps/mode/fairness_weight combination, however your
#   # scheduler fans out jobs):
#   RUN_ID=20260615_095840 COMBO="3:dynamic:0.3" ./cps_coordination/testing/run_step10_scale10k.sh
#
#   # One combo, sharded across 4 more processes (needs 4 separate
#   # invocations, SHARD_INDEX=0..3, then merge_shards.py afterward):
#   RUN_ID=20260615_095840 COMBO="3:dynamic:0.3" SHARDS=4 SHARD_INDEX=0 ./cps_coordination/testing/run_step10_scale10k.sh
#
#   # Resume a combo (or full sequential sweep) that was interrupted:
#   RUN_ID=20260615_095840 COMBO="3:dynamic:0.3" RESUME=1 SAVE_ROOT=experiments/cps_eval/scale_10k_20260801_000000 ./cps_coordination/testing/run_step10_scale10k.sh

set -euo pipefail

# --- REQUIRED ---
RUN_ID="${RUN_ID:?Set RUN_ID to the frozen worker run_id under experiments/PathPlanningGoalEnv-v0/SAC/models/ (e.g. 20260615_095840, the latest run with a final_model.zip as of this writing -- verify with: python3 -c \"import glob,os; c=sorted(glob.glob('experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip')); print(os.path.basename(os.path.dirname(c[-1])))\").}"

# --- optional overrides ---
CONFIG="${CONFIG:-cps_coordination/configs/cps_scale_10k.yaml}"
ETA_SURROGATE_PATH="${ETA_SURROGATE_PATH:-cps_coordination/models/eta_surrogate.pkl}"
EPISODES="${EPISODES:-10000}"
SEED_BASE="${SEED_BASE:-0}"
SAVE_ROOT="${SAVE_ROOT:-experiments/cps_eval/scale_10k_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-}"
SHARDS="${SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

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
    if [ "$SHARDS" -gt 1 ]; then
        echo "SHARDS>1 requires COMBO=... (one job = one (combo, shard) pair -- sharding the full sequential grid in one process defeats the point)." >&2
        exit 1
    fi
    K_CPS_SWEEP=(0 3)
    MODE_SWEEP=(static dynamic)
    FW_SWEEP=(0.0 0.3)
    echo "Full 8-combo sequential mode -- expect ~2.0 days wall-clock (see script header)."
    echo "Ctrl-C now if you meant to parallelize via COMBO=\"k_cps:mode:fw\" instead."
    sleep 5
fi

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

uv run python cps_coordination/testing/run_batch_eval.py \
    --run-id "$RUN_ID" \
    --config "$CONFIG" \
    --episodes "$RUN_EPISODES" \
    --k-cps-sweep "${K_CPS_SWEEP[@]}" \
    --mode-sweep "${MODE_SWEEP[@]}" \
    --fairness-weight-sweep "${FW_SWEEP[@]}" \
    --disable-cross-cycle-runway-seeding \
    --eta-surrogate-path "$ETA_SURROGATE_PATH" \
    --save-path-root "$SAVE_ROOT" \
    --seed-base "$RUN_SEED_BASE" \
    --episode-id-offset "$RUN_EPISODE_ID_OFFSET" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}"

echo "Done -> $SAVE_ROOT/k<k_cps>_<mode>_fw<fairness_weight>/"
if [ "$SHARDS" -gt 1 ]; then
    echo "This was shard $SHARD_INDEX/$SHARDS -- once all $SHARDS shards finish, merge with:"
    echo "  uv run python cps_coordination/testing/merge_shards.py --save-root <root_without_shard_suffix> --combo k${K_CPS_SWEEP[0]}_${MODE_SWEEP[0]}_fw${FW_SWEEP[0]} --shards $SHARDS"
fi
echo "Then: uv run python cps_coordination/testing/step10_deep_analysis.py --sweep-root $SAVE_ROOT"
