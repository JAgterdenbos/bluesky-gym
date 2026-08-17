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
# DENSITY RESCALE (2026-08-13) -- max_concurrent_aircraft 10->35,
# total_arrivals_per_episode 25->50 (cps_scale_10k.yaml), resolved by an
# 8-candidate-cap capacity sweep (see phase3_cps_coordination_plan.md's
# "Density Rescale to 50 ac/episode" section). MEASURED fresh at this exact
# config via this exact script (capped M=10/M=30, single combo, no other
# load): ~7.1s fixed startup overhead + ~3.99s/episode marginal cost ->
# ~2.22h/combo at the real M=2,000 -> ~13.3h sequential for all 6 combos.
# This superseded an earlier ~2.1h/combo/~8.2h/4-combo figure measured at
# the OLD 25-ac/cap=10 density -- not a linear extrapolation, remeasured
# from scratch because both the arrivals increase and the new cap change
# per-episode cost nonlinearly. If your cluster can run more than one
# process concurrently, launch one job per combo instead (see COMBO mode
# below). If it has MORE than 6 workers, SHARDS (below) can split a single
# combo's episodes across additional processes too. Real runtime will vary
# with cluster hardware; this is a same-machine extrapolation, not a promise.
#
# VALIDATED GRID -- this script pins k_cps/mode explicitly rather than
# relying on run_batch_eval.py's own bare CLI defaults (k_cps=[0,1,3] x
# mode, 6 combos -- which happen to now be identical to what's pinned
# below). k_cps in {0,1,3} x mode in {static,dynamic} (6 combos) is the
# validated production grid as of 2026-08-13 -- k=1 was added alongside the
# density rescale above (previously k_cps in {0,3} only, 4 combos, with no
# documented rationale for excluding k=1; see phase3_cps_coordination_plan.md's
# "Density Rescale to 50 ac/episode" section, "k=1 added to this run only").
#
# FAIRNESS_WEIGHT REMOVED (2026-08-12) -- a k-CPS slack-protection cost
# term used to be calibrated per-mode here (static fw=1.0, dynamic fw=0.5,
# via a local Stage 1/2 sweep). .claude/plans/stall_rate_investigation.md
# found fairness_weight=0.0 (exact FCFS ordering) matched or beat every
# tested nonzero value, in both modes, at every congestion level -- the
# earlier calibration was run in an artificial "ratchet ON" regime just to
# get any stall signal at all (real, ratchet-OFF production traffic barely
# stalls at the launch density, giving the calibration nothing genuine to
# tune against). The mechanism itself was removed from CPSManager, so both
# modes now run one shared, unconditional FCFS ordering -- the STATIC_FW/
# DYNAMIC_FW split (and the two-separate-run_batch_eval.py-calls structure
# that only existed to apply it) is gone; one call now sweeps both modes.
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
#       --save-root "$SAVE_ROOT" --combo "k3_dynamic" --shards 4
# which verifies there are no colliding episode_ids across shards before
# writing the merged, final-looking SAVE_ROOT/k3_dynamic/ directory.
#
# Usage
# -----
#   # Full 4-combo sweep, one run_batch_eval.py invocation sweeping both
#   # k_cps and mode together (~8.2h measured; only if your cluster truly
#   # runs this as one long single job):
#   RUN_ID=20260615_095840 ./cps_coordination/scripts/run_step10_scale10k.sh
#
#   # One combo per job (recommended for real cluster use -- launch 4 of
#   # these, one per k_cps/mode combination, however your scheduler fans
#   # out jobs):
#   RUN_ID=20260615_095840 COMBO="3:dynamic" ./cps_coordination/scripts/run_step10_scale10k.sh
#   RUN_ID=20260615_095840 COMBO="3:static" ./cps_coordination/scripts/run_step10_scale10k.sh
#
#   # One combo, sharded across 4 more processes (needs 4 separate
#   # invocations, SHARD_INDEX=0..3, then merge_shards.py afterward):
#   RUN_ID=20260615_095840 COMBO="3:dynamic" SHARDS=4 SHARD_INDEX=0 ./cps_coordination/scripts/run_step10_scale10k.sh
#
#   # Resume a combo (or full sweep) that was interrupted:
#   RUN_ID=20260615_095840 COMBO="3:dynamic" RESUME=1 SAVE_ROOT=experiments/cps_eval/scale_10k_20260801_000000 ./cps_coordination/scripts/run_step10_scale10k.sh

set -euo pipefail

# --- clean, resumable stop on kill -INT / kill -TERM ----------------------
# A plain foreground `uv run python ...` call defeats a script's own signal
# trap: bash defers trap handling until that command returns on its own, AND
# an interactive-shell-style bash actively IGNORES SIGINT while it has a
# foreground child running (so a stray Ctrl-C/kill -INT wouldn't stop this
# script at all, only whatever it happened to be running at the time).
# Backgrounding each run_batch_eval.py invocation (`&` + explicit
# `wait "$CHILD_PID"`, see the full-grid and COMBO branches below) makes
# the wait interruptible, so a single kill -INT/-TERM sent to *this script's
# own PID* (what launch_step10_dedicated_terminal.sh reports and $! captures)
# reliably: forwards the signal to the actual worker -- which has its own
# SIGINT/SIGTERM handler (finishes the in-flight episode, flushes and closes
# the current combo's telemetry) -- waits for it to exit, then exits this
# script itself, critically WITHOUT starting the next mode/combo. Safe to
# resume afterward with RESUME=1.
CHILD_PID=""
_cleanup() {
    trap - INT TERM  # don't re-enter if a second signal arrives mid-cleanup
    echo "" >&2
    echo "[run_step10_scale10k] stop requested -- forwarding to worker (pid ${CHILD_PID:-none yet}) and exiting once it flushes (not starting any further combos/modes)." >&2
    if [ -n "$CHILD_PID" ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 130
}
trap _cleanup INT TERM

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

# Diagnostic-only (Vector 9, phase3_cps_coordination_plan.md): opt-in, off by
# default. Adds a much-higher-row-count cps_eval_reassignment.parquet per
# combo -- only meant for scoped diagnostic-scale runs (small EPISODES/COMBO),
# not routine production launches.
LOG_REASSIGNMENT_EVENTS="${LOG_REASSIGNMENT_EVENTS:-}"
REASSIGNMENT_ARGS=()
if [ -n "$LOG_REASSIGNMENT_EVENTS" ]; then
    REASSIGNMENT_ARGS=(--log-reassignment-events)
fi

# --- combo selection ---
# COMBO unset (default): full validated 6-combo grid, one run_batch_eval.py
# invocation sweeping both k_cps and mode together.
# COMBO="k_cps:mode" (e.g. "3:dynamic"): restrict to exactly that one
# combo -- run this script once per combo, in parallel, across cluster jobs.
if [ -n "${COMBO:-}" ]; then
    IFS=':' read -r K_CPS MODE <<< "$COMBO"
    echo "Single-combo mode: k_cps=$K_CPS mode=$MODE"
    uv run python cps_coordination/scripts/run_batch_eval.py \
        --run-id "$RUN_ID" \
        --config "$CONFIG" \
        --episodes "$RUN_EPISODES" \
        --k-cps-sweep "$K_CPS" \
        --mode-sweep "$MODE" \
        --disable-cross-cycle-runway-seeding \
        --eta-surrogate-path "$ETA_SURROGATE_PATH" \
        --save-path-root "$SAVE_ROOT" \
        --seed-base "$RUN_SEED_BASE" \
        --episode-id-offset "$RUN_EPISODE_ID_OFFSET" \
        "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
        "${REASSIGNMENT_ARGS[@]+"${REASSIGNMENT_ARGS[@]}"}" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    CHILD_PID=""
    echo "Done -> $SAVE_ROOT/k${K_CPS}_${MODE}/"
    if [ "$SHARDS" -gt 1 ]; then
        echo "This was shard $SHARD_INDEX/$SHARDS -- once all $SHARDS shards finish, merge with:"
        echo "  uv run python cps_coordination/scripts/merge_shards.py --save-root <root_without_shard_suffix> --combo k${K_CPS}_${MODE} --shards $SHARDS"
    fi
else
    if [ "$SHARDS" -gt 1 ]; then
        echo "SHARDS>1 requires COMBO=... (one job = one (combo, shard) pair -- sharding the full sequential grid in one process defeats the point)." >&2
        exit 1
    fi
    echo "Full 6-combo mode -- one run_batch_eval.py invocation sweeping k_cps x mode."
    echo "Expect ~13.3h (measured at the 50-ac/cap=35 density, ~2.22h/combo x 6) wall-clock total (see script header)."
    echo "Ctrl-C now if you meant to parallelize via COMBO=\"k_cps:mode\" instead."
    sleep 5
    uv run python cps_coordination/scripts/run_batch_eval.py \
        --run-id "$RUN_ID" \
        --config "$CONFIG" \
        --episodes "$RUN_EPISODES" \
        --k-cps-sweep 0 1 3 \
        --mode-sweep static dynamic \
        --disable-cross-cycle-runway-seeding \
        --eta-surrogate-path "$ETA_SURROGATE_PATH" \
        --save-path-root "$SAVE_ROOT" \
        --seed-base "$RUN_SEED_BASE" \
        --episode-id-offset "$RUN_EPISODE_ID_OFFSET" \
        "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
        "${REASSIGNMENT_ARGS[@]+"${REASSIGNMENT_ARGS[@]}"}" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    CHILD_PID=""
    echo "Done -> $SAVE_ROOT/k{0,1,3}_{static,dynamic}/"
fi

echo "Then: uv run python cps_coordination/scripts/step10_deep_analysis.py --sweep-root $SAVE_ROOT"
