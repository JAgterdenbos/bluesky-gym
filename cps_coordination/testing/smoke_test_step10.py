"""
cps_coordination/testing/smoke_test_step10.py
------------------------------------------------
Capped local verification for Phase III roadmap Step 10 (M=10,000
scale-up config/script generation) -- runs a small M=10 sweep through
``run_batch_eval.py``'s real sweep machinery to sanity-check the
generated scripts before anything is handed off to a cluster/runner.

This is NOT a stand-in for the M=10,000 production run (see
``.claude/step10_execution_and_data_collection_plan.md`` for that). Its
job is narrower and specific: ``max_concurrent_aircraft=5,
total_arrivals_per_episode=10, spawn_window_s=1800.0`` deliberately
exercises two code paths that have never been exercised together before
this step -- a rolling arrival stream (total > concurrent) and
time-windowed spawning -- plus the local/global episode-clock conversion
(``MultiAgentPathPlanningGoalEnv.set_tta``/``_get_info``'s ``spawn_time``,
and ``coordination_baseline.py::_run_episode``'s landing-time offset) that
only actually does anything once spawn times are nonzero for the first
time.

Checks performed
-----------------
  1. The sweep (4 combos: k_cps in {0, 3} x mode in {static, dynamic},
     M=10 episodes each) completes with no exceptions -- confirms no
     spawn-schedule/slot-collision bugs under rolling arrivals.
  2. Each combo's two Parquet files load cleanly via
     ``cps_metrics_offline.py``'s own loader and recompute without schema
     errors.
  3. Clock-fix sanity check: the maximum successful ``actual_landing_time``
     across the whole smoke test comfortably exceeds ``spawn_window_s``.
     This would be false under the pre-fix bug, where a late-spawning
     aircraft's logged landing time never reflected its spawn offset (it
     stayed within a single flight's local-duration range, bounded well
     under the 1800s window) -- a concrete, falsifiable regression
     indicator for exactly the bug class this step introduced and fixed.

Run: python cps_coordination/testing/smoke_test_step10.py
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import tempfile

from cps_coordination.testing.run_batch_eval import _DEFAULT_CONFIG, _build_parser, run_sweep
from cps_coordination.testing.cps_metrics_offline import (
    load_recat_matrix,
    load_telemetry,
    recompute_metrics,
)

M_EPISODES = 10
K_CPS_SWEEP = [0, 3]
MODE_SWEEP = ["static", "dynamic"]
FAIRNESS_WEIGHT_SWEEP = [0.0]
MAX_CONCURRENT_AIRCRAFT = 5
TOTAL_ARRIVALS_PER_EPISODE = 10
SPAWN_WINDOW_S = 1800.0


def _find_pretrained_run_id() -> str:
    """Same glob pattern as validate_cps_pipeline.py::_find_pretrained_run_id."""
    candidates = sorted(
        glob.glob("experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip")
    )
    if not candidates:
        raise RuntimeError(
            "No frozen SAC model found under experiments/PathPlanningGoalEnv-v0/SAC/models/"
        )
    return os.path.basename(os.path.dirname(candidates[-1]))


def _build_smoke_args(run_id: str, save_path_root: str) -> argparse.Namespace:
    """Built via run_batch_eval's own argument parser (parsing a synthetic
    argv of just the fields this smoke test cares to override) rather than
    a hand-built argparse.Namespace -- the latter silently broke twice in
    one session (missing --episode-id-offset, then missing --resume)
    every time a new flag was added to run_batch_eval.py, since nothing
    forced it to stay in sync. Parsing through the real parser means every
    other flag (present now or added later) gets its real, correct
    default automatically."""
    argv = [
        "--run-id", run_id,
        "--config", _DEFAULT_CONFIG,
        "--episodes", str(M_EPISODES),
        "--k-cps-sweep", *[str(k) for k in K_CPS_SWEEP],
        "--mode-sweep", *MODE_SWEEP,
        "--fairness-weight-sweep", *[str(fw) for fw in FAIRNESS_WEIGHT_SWEEP],
        "--max-concurrent-aircraft", str(MAX_CONCURRENT_AIRCRAFT),
        "--total-arrivals-per-episode", str(TOTAL_ARRIVALS_PER_EPISODE),
        "--spawn-window-s", str(SPAWN_WINDOW_S),
        "--save-path-root", save_path_root,
        "--chunk-size", "250",
        "--log-every", "5",
    ]
    return _build_parser().parse_args(argv)


def main() -> None:
    run_id = _find_pretrained_run_id()
    save_path_root = tempfile.mkdtemp(prefix="cps_smoke_step10_")
    print(f"Smoke test scratch dir: {save_path_root}")
    print(f"Using pretrained worker run_id: {run_id}")

    args = _build_smoke_args(run_id, save_path_root)

    combo_results = []
    overall_max_landing_time = float("-inf")
    ok = True

    try:
        print("\n=== Running capped sweep (M=10, 4 combos) via run_batch_eval.run_sweep ===")
        run_sweep(args)

        print("\n=== Verifying telemetry round-trips through cps_metrics_offline.py ===")
        recat_matrix = load_recat_matrix()
        for k_cps in K_CPS_SWEEP:
            for mode in MODE_SWEEP:
                for fw in FAIRNESS_WEIGHT_SWEEP:
                    combo_dir = os.path.join(save_path_root, f"k{k_cps}_{mode}_fw{fw:g}")
                    try:
                        aircraft_df, separation_df = load_telemetry(combo_dir)
                        metrics = recompute_metrics(aircraft_df, separation_df, recat_matrix)

                        successful = aircraft_df[aircraft_df["success"]]
                        combo_max_landing = (
                            float(successful["actual_landing_time"].max())
                            if not successful.empty else float("-inf")
                        )
                        overall_max_landing_time = max(overall_max_landing_time, combo_max_landing)

                        n_rows = len(aircraft_df)
                        combo_results.append(
                            (k_cps, mode, fw, "PASS", n_rows, metrics.get("success_rate"), combo_max_landing)
                        )
                        print(
                            f"  k_cps={k_cps}, mode={mode}, fairness_weight={fw:g}: {n_rows} aircraft rows, "
                            f"success_rate={metrics.get('success_rate')}, "
                            f"max_landing_time={combo_max_landing:.1f}s"
                        )
                    except Exception as exc:  # noqa: BLE001 -- smoke test: report, don't hide, any combo's failure
                        ok = False
                        combo_results.append((k_cps, mode, fw, f"FAIL: {exc}", None, None, None))
                        print(f"  k_cps={k_cps}, mode={mode}, fairness_weight={fw:g}: FAIL - {exc}")

        print("\n=== Clock-fix sanity check ===")
        if overall_max_landing_time > SPAWN_WINDOW_S:
            print(
                f"  PASS: max successful actual_landing_time across the smoke test "
                f"({overall_max_landing_time:.1f}s) exceeds spawn_window_s ({SPAWN_WINDOW_S}s) "
                f"-- landing times are correctly offset by spawn time, not left local-only."
            )
        else:
            ok = False
            print(
                f"  FAIL: max successful actual_landing_time ({overall_max_landing_time:.1f}s) "
                f"did not exceed spawn_window_s ({SPAWN_WINDOW_S}s) -- this is the exact "
                f"signature of the pre-fix local/global clock bug (landing times never "
                f"reflect spawn offset). Investigate before trusting the M=10,000 config."
            )
    finally:
        print(f"\nCleaning up scratch dir: {save_path_root}")
        shutil.rmtree(save_path_root, ignore_errors=True)

    print("\n=== Smoke test summary ===")
    for k_cps, mode, fw, status, n_rows, success_rate, max_landing in combo_results:
        print(f"  k_cps={k_cps:<2} mode={mode:<8} fairness_weight={fw:<4g} {status}")

    if ok:
        print("\nOVERALL: PASS")
    else:
        print("\nOVERALL: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
