#!/usr/bin/env bash
# cps_coordination/testing/regenerate_step10_sanity_sweep.sh
# --------------------------------------------------------------
# Documents (does NOT execute automatically) how to regenerate the M=100,
# 8-combo "step 10 sanity sweep" analyzed by
# cps_coordination/testing/step10_deep_analysis.py and
# .claude/plans/step10_deep_analysis_findings.md.
#
# WARNING: this takes roughly 1.5-2 HOURS to run end-to-end (8 combos x 100
# episodes x 3 passes [cps/static/solo] x 10 arrivals/episode, all through a
# live BlueSky simulation + frozen SAC policy). Do NOT run this casually --
# the already-collected data lives in cps_coordination/data/step10_sanity_sweep/
# (copied there from the original session's /tmp scratch dir specifically so
# this script does not need to be re-run for routine analysis). Only re-run
# this if you need genuinely fresh data (e.g. after a code fix to the
# slot-recycling bug documented in the findings report, to see whether the
# numbers change).
#
# Reconstructed from cps_coordination/testing/run_batch_eval.py's CLI +
# cps_coordination/configs/cps_scale_10k.yaml's cps_eval: defaults (the same
# max_concurrent_aircraft/total_arrivals_per_episode/spawn_window_s the M=10,000
# launch config uses, confirmed by direct inspection of the collected Parquet:
# 5 concurrent slots, 10 arrivals/episode, spawn_window_s=0.0 for THIS sweep
# specifically -- the M=10,000 launch config instead uses spawn_window_s=1800.0,
# see cps_scale_10k.yaml). The exact --run-id (frozen worker checkpoint) used
# for the original 2026-08-03 run is not recorded in any committed doc/config
# found during this analysis -- fill in the correct one below before running.

set -euo pipefail

# --- REQUIRED: fill in before running ---
RUN_ID="${RUN_ID:?Set RUN_ID to the frozen worker run_id under experiments/PathPlanningGoalEnv-v0/SAC/models/ (e.g. 20260610_092116) before running this script.}"
ETA_SURROGATE_PATH="${ETA_SURROGATE_PATH:-cps_coordination/models/eta_surrogate.pkl}"
SAVE_ROOT="${SAVE_ROOT:-cps_coordination/data/step10_sanity_sweep_regenerated}"

echo "This will take ~1.5-2 hours. Ctrl-C now if you did not mean to run this."
sleep 5

uv run python cps_coordination/testing/run_batch_eval.py \
    --run-id "$RUN_ID" \
    --episodes 100 \
    --k-cps-sweep 0 3 \
    --mode-sweep static dynamic \
    --fairness-weight-sweep 0.0 0.3 \
    --max-concurrent-aircraft 5 \
    --total-arrivals-per-episode 10 \
    --spawn-window-s 0.0 \
    --eta-surrogate-path "$ETA_SURROGATE_PATH" \
    --save-path-root "$SAVE_ROOT" \
    --seed-base 0

echo "Done -> $SAVE_ROOT/k<k_cps>_<mode>_fw<fairness_weight>/"
echo "Then: uv run python cps_coordination/testing/step10_deep_analysis.py --sweep-root $SAVE_ROOT"
