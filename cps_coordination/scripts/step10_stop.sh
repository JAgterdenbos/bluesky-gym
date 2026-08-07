#!/usr/bin/env bash
# cps_coordination/scripts/step10_stop.sh
# --------------------------------------------------------------
# Reliable clean stop for a Step 10 sweep launched via
# launch_step10_dedicated_terminal.sh / run_step10_scale10k.sh.
#
# Signaling the actual run_batch_eval.py worker PID directly (or, worse,
# whatever PID a launcher happened to capture in $!) is not enough on its
# own: run_step10_scale10k.sh's "full sequential" mode runs static and
# dynamic mode as two SEPARATE run_batch_eval.py invocations, one after the
# other -- stopping only the current worker just makes the wrapper move on
# and start the next mode's worker fresh. run_step10_scale10k.sh now
# installs its own `trap ... INT TERM` that forwards the signal to whichever
# worker is currently running and then exits without starting anything
# further -- but that trap only fires on the WRAPPER's own PID, which is
# not `$!` from a `nohup caffeinate -i ... &` launch either (caffeinate
# forks an internal helper and the backgrounded PID ends up being the
# wrapper script itself only after some exec-chain reshuffling that isn't
# safe to assume from the launch side). This script sidesteps all of that
# by finding the ACTUAL live worker process for this SAVE_ROOT and walking
# up its process ancestry to the run_step10_scale10k.sh process, then
# signaling that -- reliable regardless of how it was launched.
#
# Usage:
#   ./cps_coordination/scripts/step10_stop.sh                  # auto-detect the newest SAVE_ROOT
#   ./cps_coordination/scripts/step10_stop.sh <save_root>

set -euo pipefail
cd "$(dirname "$0")/../.."

SAVE_ROOT="${1:-}"
if [ -z "$SAVE_ROOT" ]; then
    SAVE_ROOT="$(ls -dt experiments/cps_eval/scale_10k_* 2>/dev/null | head -n 1 || true)"
    if [ -z "$SAVE_ROOT" ]; then
        echo "No experiments/cps_eval/scale_10k_* directory found -- pass a SAVE_ROOT explicitly." >&2
        exit 1
    fi
fi

WORKER_PID="$(pgrep -f "run_batch_eval.py.*$SAVE_ROOT" 2>/dev/null | head -n 1 || true)"
if [ -z "$WORKER_PID" ]; then
    echo "No running run_batch_eval.py process found for $SAVE_ROOT -- nothing to stop."
    exit 0
fi

# Walk up the process ancestry (worker -> uv -> run_step10_scale10k.sh) to
# find the wrapper that actually has the stop trap installed.
TARGET_PID=""
PID="$WORKER_PID"
DEPTH=0
while [ "$DEPTH" -lt 6 ] && [ -n "$PID" ] && [ "$PID" != "1" ]; do
    ARGS="$(ps -o args= -p "$PID" 2>/dev/null || true)"
    case "$ARGS" in
        *run_step10_scale10k.sh*)
            TARGET_PID="$PID"
            break
            ;;
    esac
    PID="$(ps -o ppid= -p "$PID" 2>/dev/null | tr -d ' ')"
    DEPTH=$((DEPTH + 1))
done

if [ -z "$TARGET_PID" ]; then
    echo "WARNING: could not find a run_step10_scale10k.sh process in $WORKER_PID's ancestry" >&2
    echo "-- signaling the worker directly instead. This stops the current combo cleanly, but" >&2
    echo "if this is a full multi-mode sequential sweep, the wrapper may then start the next" >&2
    echo "mode's worker rather than exiting. Re-run this script again if that happens." >&2
    TARGET_PID="$WORKER_PID"
fi

echo "Sending SIGTERM to pid $TARGET_PID (save_root=$SAVE_ROOT)..."
kill -TERM "$TARGET_PID"

echo "Waiting for a graceful exit (finishing the in-flight episode, flushing and closing telemetry)..."
for i in $(seq 1 120); do
    STILL="$(pgrep -f "run_batch_eval.py.*$SAVE_ROOT" 2>/dev/null || true)"
    if [ -z "$STILL" ]; then
        echo "Stopped cleanly."
        LOG="$(ls -t cps_coordination/data/step10_launch_*.log 2>/dev/null | head -n 1 || true)"
        if [ -n "$LOG" ]; then
            echo "--- last 10 log lines ---"
            tail -n 10 "$LOG"
        fi
        echo ""
        echo "Resume later: RESUME=1 SAVE_ROOT=$SAVE_ROOT ./cps_coordination/scripts/launch_step10_dedicated_terminal.sh"
        exit 0
    fi
    sleep 1
done

echo "Still running after 120s -- an episode may be slow to finish. Check current state with:" >&2
echo "  ./cps_coordination/scripts/step10_progress.sh $SAVE_ROOT" >&2
exit 1
