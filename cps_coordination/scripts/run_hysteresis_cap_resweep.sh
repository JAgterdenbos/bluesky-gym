#!/usr/bin/env bash
# cps_coordination/scripts/run_hysteresis_cap_resweep.sh
# --------------------------------------------------------------
# Launcher for the concurrency-cap resweep + reassignment-guard-timing
# sensitivity sweep
# (.claude/plans/concurrency_cap_and_reassignment_guard_resweep.md).
#
# No existing script covers this grid: run_fw_removed_capacity_sweep.sh
# (scratchpad/) only sweeps max_concurrent_aircraft at the fixed default
# REASSIGNMENT_HYSTERESIS_S; run_step10_scale10k.sh sweeps k_cps x mode at a
# fixed cap/hysteresis. This script sweeps max_concurrent_aircraft x
# reassignment_hysteresis_s at fixed k_cps/mode, one run_batch_eval.py
# invocation per (cap, hysteresis) cell, each writing to its own
# save-path-root subdirectory so run_batch_eval.py's own per-combo --resume
# logic (keyed on that subdirectory's k{k_cps}_{mode}/cps_eval_aircraft.parquet)
# applies independently per cell.
#
# Diagnostic scale only (M=30 by default) -- never point this at M=2,000.
# --log-reassignment-events defaults ON here (unlike run_step10_scale10k.sh,
# which defaults it off for production safety) because this sweep's own
# decision rules need the switch/thrash-rate guardrail metric, and this
# script is never meant to run at production scale where the higher row
# count would matter.
#
# CAPS / HYST_VALUES are overridable so this same script also serves:
#   - the baseline 8x4 grid (defaults below),
#   - boundary reseeds (narrow CAPS/HYST_VALUES to the 2-3 caps bracketing
#     the crossover in one hysteresis slice, override SEED_BASE for a fresh
#     seed draw),
#   - the single k=1 confirmation cell (CAPS/HYST_VALUES = one value each,
#     K_CPS=1).
#
# RESUME -- set RESUME=1 to pass --resume through to run_batch_eval.py for
# every cell. A cell whose cps_eval_aircraft.parquet already covers EPISODES
# is skipped by run_batch_eval.py itself (near-instant); a partially-written
# cell resumes from its highest durably-written episode_id. Re-running this
# script with RESUME=1 after an interruption is therefore safe to do exactly
# as-is (same CAPS/HYST_VALUES/SAVE_ROOT) -- it does not need its own
# separate "which cells are done" bookkeeping, since it delegates that to
# run_batch_eval.py per cell, which already provides this (see
# _resolve_resume_start/_merge_resume_delta's docstrings in run_batch_eval.py).
# Per this plan's correction to v1 (do not assume --resume behavior is
# inherited from a different script's usage), this must be empirically
# re-verified against a real interrupt-and-resume before the real grid is
# launched -- not assumed correct from this comment alone.
#
# Usage
# -----
#   # Full 8x4 baseline grid (32 cells, M=30 each):
#   RUN_ID=20260615_095840 ./cps_coordination/scripts/run_hysteresis_cap_resweep.sh
#
#   # Resume an interrupted grid:
#   RUN_ID=20260615_095840 RESUME=1 SAVE_ROOT=experiments/cps_eval/hysteresis_cap_resweep_20260820_120000 \
#       ./cps_coordination/scripts/run_hysteresis_cap_resweep.sh
#
#   # Boundary reseed (e.g. caps 35/42 bracket the 240s-slice crossover, fresh seeds):
#   RUN_ID=20260615_095840 CAPS="35 42" HYST_VALUES="240" SEED_BASE=5000 \
#       SAVE_ROOT=experiments/cps_eval/hysteresis_cap_resweep_20260820_120000/reseed_240 \
#       ./cps_coordination/scripts/run_hysteresis_cap_resweep.sh
#
#   # k=1 confirmation cell at the resolved (cap, hysteresis) pair:
#   RUN_ID=20260615_095840 K_CPS=1 CAPS="35" HYST_VALUES="240" \
#       SAVE_ROOT=experiments/cps_eval/hysteresis_cap_resweep_20260820_120000/k1_confirm \
#       ./cps_coordination/scripts/run_hysteresis_cap_resweep.sh
#
#   # Capped smoke test (small M, one cell):
#   RUN_ID=20260615_095840 EPISODES=10 CAPS="35" HYST_VALUES="360" \
#       SAVE_ROOT=/tmp/smoke_hysteresis_cap_resweep \
#       ./cps_coordination/scripts/run_hysteresis_cap_resweep.sh

set -euo pipefail

# --- clean, resumable stop on kill -INT / kill -TERM ----------------------
# Same pattern as run_step10_scale10k.sh: background each run_batch_eval.py
# invocation so a signal sent to this script's own PID is forwarded to the
# in-flight cell's worker (which flushes/closes cleanly) and this script
# exits WITHOUT starting the next cell. Safe to resume afterward with
# RESUME=1.
CHILD_PID=""
_cleanup() {
    trap - INT TERM
    echo "" >&2
    echo "[run_hysteresis_cap_resweep] stop requested -- forwarding to worker (pid ${CHILD_PID:-none yet}) and exiting once it flushes (not starting any further cells)." >&2
    if [ -n "$CHILD_PID" ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 130
}
trap _cleanup INT TERM

# --- REQUIRED ---
RUN_ID="${RUN_ID:?Set RUN_ID to the frozen worker run_id under experiments/PathPlanningGoalEnv-v0/SAC/models/ (e.g. 20260615_095840, the latest run with a final_model.zip as of this writing -- verify with: python3 -c \"import glob,os; c=sorted(glob.glob('experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip')); print(os.path.basename(os.path.dirname(c[-1])))\").}"

# --- grid (defaults: the plan's baseline 8x4 grid) ---
CAPS="${CAPS:-10 20 27 35 42 50 65 80}"
HYST_VALUES="${HYST_VALUES:-60 120 240 360}"
K_CPS="${K_CPS:-3}"
MODE="${MODE:-dynamic}"

# --- optional overrides ---
CONFIG="${CONFIG:-cps_coordination/configs/cps_scale_10k.yaml}"
ETA_SURROGATE_PATH="${ETA_SURROGATE_PATH:-cps_coordination/models/eta_surrogate.pkl}"
EPISODES="${EPISODES:-30}"
MAX_ARRIVALS="${MAX_ARRIVALS:-50}"
SPAWN_WINDOW_S="${SPAWN_WINDOW_S:-2400}"
SEED_BASE="${SEED_BASE:-1000}"
SAVE_ROOT="${SAVE_ROOT:-experiments/cps_eval/hysteresis_cap_resweep_$(date +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-}"
LOG_REASSIGNMENT_EVENTS="${LOG_REASSIGNMENT_EVENTS:-1}"

RESUME_ARGS=()
if [ -n "$RESUME" ]; then
    RESUME_ARGS=(--resume)
fi

REASSIGNMENT_ARGS=()
if [ -n "$LOG_REASSIGNMENT_EVENTS" ]; then
    REASSIGNMENT_ARGS=(--log-reassignment-events)
fi

N_CAPS=$(echo "$CAPS" | wc -w | tr -d ' ')
N_HYST=$(echo "$HYST_VALUES" | wc -w | tr -d ' ')
N_CELLS=$(( N_CAPS * N_HYST ))

echo "CPS hysteresis/cap resweep -> $SAVE_ROOT"
echo "  k_cps=$K_CPS mode=$MODE"
echo "  CAPS=[$CAPS]"
echo "  HYST_VALUES=[$HYST_VALUES] s"
echo "  cells=$N_CELLS, episodes/cell=$EPISODES"
echo "  max_concurrent_aircraft candidates above, total_arrivals_per_episode=$MAX_ARRIVALS, spawn_window_s=$SPAWN_WINDOW_S"
echo "  run_id=$RUN_ID  config=$CONFIG  eta_surrogate_path=$ETA_SURROGATE_PATH"
echo "  seed_base=$SEED_BASE  resume=${RESUME:-0}  log_reassignment_events=${LOG_REASSIGNMENT_EVENTS:-0}"
echo ""

CELL_I=0
for CAP in $CAPS; do
    for HYST in $HYST_VALUES; do
        CELL_I=$((CELL_I + 1))
        CELL_PATH="$SAVE_ROOT/cap${CAP}_hyst${HYST}"
        echo "--- cell $CELL_I/$N_CELLS: cap=$CAP hysteresis=${HYST}s -> $CELL_PATH ---"
        uv run python cps_coordination/scripts/run_batch_eval.py \
            --run-id "$RUN_ID" \
            --config "$CONFIG" \
            --episodes "$EPISODES" \
            --k-cps-sweep "$K_CPS" \
            --mode-sweep "$MODE" \
            --disable-cross-cycle-runway-seeding \
            --max-concurrent-aircraft "$CAP" \
            --total-arrivals-per-episode "$MAX_ARRIVALS" \
            --spawn-window-s "$SPAWN_WINDOW_S" \
            --reassignment-hysteresis-s "$HYST" \
            --eta-surrogate-path "$ETA_SURROGATE_PATH" \
            --save-path-root "$CELL_PATH" \
            --seed-base "$SEED_BASE" \
            "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
            "${REASSIGNMENT_ARGS[@]+"${REASSIGNMENT_ARGS[@]}"}" &
        CHILD_PID=$!
        wait "$CHILD_PID"
        CHILD_PID=""
    done
done

echo ""
echo "Done -> $SAVE_ROOT/cap<N>_hyst<S>/k${K_CPS}_${MODE}/"
