#!/usr/bin/env bash
# cps_coordination/scripts/launch_step10_dedicated_terminal.sh
# --------------------------------------------------------------
# One-shot launcher for the Step 10 (M=2,000) production sweep from a
# single macOS Terminal.app window/tab, safe to close once launched.
# Wraps run_step10_scale10k.sh with:
#   - an explicit runway-scope assertion (fails loudly instead of silently
#     falling back to all 12 runways -- see the runway-scope check below)
#   - a printed, human-confirmed summary of the resolved config
#   - a capped M=10 smoke test before every launch (skippable)
#   - nohup + caffeinate -i, backgrounded and disowned, so the run survives
#     the terminal window closing and macOS idle sleep
#
# Run this from the repo root (the parent of cps_coordination/), or from
# anywhere -- it cd's to the repo root itself.
#
# Usage:
#   ./cps_coordination/scripts/launch_step10_dedicated_terminal.sh
#   RUN_ID=20260615_095840 ./cps_coordination/scripts/launch_step10_dedicated_terminal.sh
#   COMBO="3:dynamic:0.5" ./cps_coordination/scripts/launch_step10_dedicated_terminal.sh
#   RESUME=1 SAVE_ROOT=experiments/cps_eval/scale_10k_20260801_000000 ./cps_coordination/scripts/launch_step10_dedicated_terminal.sh
#
# Env vars (all optional):
#   RUN_ID           Frozen worker run_id. Auto-resolved (latest run with a
#                     final_model.zip under
#                     experiments/PathPlanningGoalEnv-v0/SAC/models/) if unset.
#   SAVE_ROOT         Output root. Defaults to a fresh timestamped dir under
#                     experiments/cps_eval/.
#   COMBO             "k_cps:mode:fw" to launch a single combo instead of the
#                     full 4-combo sequential sweep (see
#                     run_step10_scale10k.sh's header for details).
#   RESUME            Set to 1 to resume an interrupted SAVE_ROOT.
#   STATIC_FW         Calibrated static-mode fairness_weight (default 1.0).
#   DYNAMIC_FW        Calibrated dynamic-mode fairness_weight (default 0.5).
#   EXPECTED_RUNWAYS  Runway scope this script asserts cps_scale_10k.yaml
#                     matches before launching (default "18R 27" -- Groot et
#                     al.'s dual-runway baseline scope). Update this only if
#                     you've deliberately changed the production runway scope.
#   SKIP_SMOKE        Set to 1 to skip the pre-launch M=10 smoke test. Not
#                     recommended -- only for a SAVE_ROOT you've already
#                     smoke-tested this session (e.g. a RESUME).

set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root, regardless of where this is invoked from

CONFIG="cps_coordination/configs/cps_scale_10k.yaml"
EXPECTED_RUNWAYS="${EXPECTED_RUNWAYS:-18R 27}"

if [ -z "${RUN_ID:-}" ]; then
    RUN_ID="$(python3 -c "import glob,os; c=sorted(glob.glob('experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip')); print(os.path.basename(os.path.dirname(c[-1])) if c else '')")"
    if [ -z "$RUN_ID" ]; then
        echo "No experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip found -- set RUN_ID explicitly." >&2
        exit 1
    fi
    echo "Auto-resolved RUN_ID=$RUN_ID (latest run with a final_model.zip)"
fi

# --- runway-scope assertion -----------------------------------------------
# run_step10_scale10k.sh never passes --runways itself: it relies entirely
# on run_batch_eval.py's argparse default, which is read from
# cps_scale_10k.yaml's env.env_kwargs.runways at parse time. That's correct
# today (["18R", "27"], Groot et al.'s dual-runway scope) but implicit --
# nothing else stops a future edit to that YAML field from silently
# reintroducing the "runways: null -> falls back to all 12" behavior this
# was fixed away from. Assert it explicitly here so a drift fails loudly
# before a multi-hour launch instead of silently evaluating the wrong scope.
ACTUAL_RUNWAYS="$(uv run python3 -c "
import yaml
d = yaml.safe_load(open('$CONFIG')) or {}
rw = d.get('env', {}).get('env_kwargs', {}).get('runways')
print(' '.join(rw) if rw else '')
")"
if [ "$ACTUAL_RUNWAYS" != "$EXPECTED_RUNWAYS" ]; then
    echo "ABORT: $CONFIG's env.env_kwargs.runways resolved to [$ACTUAL_RUNWAYS], expected [$EXPECTED_RUNWAYS]." >&2
    echo "This would change the production runway scope for this launch. If that's" >&2
    echo "intentional, re-run with EXPECTED_RUNWAYS=\"$ACTUAL_RUNWAYS\" to confirm." >&2
    exit 1
fi

SAVE_ROOT="${SAVE_ROOT:-experiments/cps_eval/scale_10k_$(date +%Y%m%d_%H%M%S)}"
LOG="cps_coordination/data/step10_launch_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

EPISODES="${EPISODES:-2000}"
STATIC_FW="${STATIC_FW:-1.0}"
DYNAMIC_FW="${DYNAMIC_FW:-0.5}"

echo "=================================================================="
echo "Step 10 scale-up launch -- resolved configuration"
echo "  config             : $CONFIG"
echo "  runway scope       : $ACTUAL_RUNWAYS"
echo "  episodes/combo (M) : $EPISODES"
echo "  worker checkpoint  : experiments/PathPlanningGoalEnv-v0/SAC/models/$RUN_ID/final_model.zip"
echo "  fairness_weight    : static=$STATIC_FW dynamic=$DYNAMIC_FW"
echo "  save_root          : $SAVE_ROOT"
echo "  log                : $LOG"
echo "  combo              : ${COMBO:-<full 4-combo sequential sweep, ~8.2h>}"
echo "  resume             : ${RESUME:-0}"
echo "=================================================================="

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "Running capped M=10 smoke test before launch (set SKIP_SMOKE=1 to skip)..."
    uv run python cps_coordination/testing/smoke_test_step10.py
    echo "Smoke test passed."
else
    echo "SKIP_SMOKE=1 -- skipping pre-launch smoke test."
fi

printf "Proceed with the launch above? [y/N] "
read -r CONFIRM
case "$CONFIRM" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
esac

export RUN_ID SAVE_ROOT CONFIG STATIC_FW DYNAMIC_FW EPISODES
if [ -n "${COMBO:-}" ]; then export COMBO; fi
if [ -n "${RESUME:-}" ]; then export RESUME; fi

# Python block-buffers stdout when it's not a TTY (i.e. redirected to a
# file, as below) -- without this, run_batch_eval.py's progress prints sit
# in an internal buffer and don't reach $LOG until it happens to fill up,
# so step10_progress.sh's log-based live signal would lag arbitrarily far
# behind reality. This only affects stdout buffering, not any evaluation
# logic.
export PYTHONUNBUFFERED=1

echo "Launching in background via caffeinate + nohup..."
nohup caffeinate -i ./cps_coordination/scripts/run_step10_scale10k.sh > "$LOG" 2>&1 &
disown
PID=$!

echo ""
echo "Launched. PID=$PID (informational only -- use step10_stop.sh below to stop, not kill $PID directly)"
echo "Log:       $LOG"
echo "Save root: $SAVE_ROOT"
echo ""
echo "Progress:                 ./cps_coordination/scripts/step10_progress.sh $SAVE_ROOT"
echo "Watch + notify on stop:    ./cps_coordination/scripts/step10_progress.sh $SAVE_ROOT --watch --notify"
echo "Stop cleanly (resumable):  ./cps_coordination/scripts/step10_stop.sh $SAVE_ROOT"
echo "Resume later:              RESUME=1 SAVE_ROOT=$SAVE_ROOT ./cps_coordination/scripts/launch_step10_dedicated_terminal.sh"
echo "After it finishes:         uv run python cps_coordination/scripts/step10_deep_analysis.py --sweep-root $SAVE_ROOT"
