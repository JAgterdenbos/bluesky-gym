"""
Roadmap steps 3-4 validation: exercise the real ``CPSCoordinationExperiment``
evaluation pipeline (``CPSManager`` + ``MultiAgentPathPlanningGoalEnv`` +
frozen SAC worker) end-to-end, using an actual frozen worker model from
``experiments/PathPlanningGoalEnv-v0/SAC/models/``.

Step 3 (k=0 FCFS, static runway mode): confirm TTA injection visibly changes
worker behaviour by comparing a CPS-coordinated run against a matched-seed
control run where ``set_tta`` is never called (frozen worker flies its own
unconstrained goal).

Step 4 (k>0 CPS, N_a > 2, all aircraft targeting one runway to force real
sequencing): confirm every consecutive pair of landings on that runway
respects the RECAT-EU separation the greedy scheduler is supposed to
guarantee (``TTA_i >= TTA_{i-1} + required_separation``, within a small
tolerance for the worker's own RTA tracking error).

Run: python cps_coordination/testing/validate_cps_pipeline.py
"""

from __future__ import annotations

import glob
import os
from typing import List, Tuple

from bluesky_gym.experiment.config import ExperimentConfig, SessionConfig
from cps_coordination.coordination.cps_manager import CPSManager
from cps_coordination.experiments.config import CPSEnvConfig, CPSEnvKwargsConfig, CPSModelConfig
from cps_coordination.experiments.coordination_baseline import CPSCoordinationExperiment

SEED = 7


def _find_pretrained_run_id() -> str:
    candidates = sorted(
        glob.glob("experiments/PathPlanningGoalEnv-v0/SAC/models/*/final_model.zip")
    )
    if not candidates:
        raise RuntimeError(
            "No frozen SAC model found under experiments/PathPlanningGoalEnv-v0/SAC/models/"
        )
    return os.path.basename(os.path.dirname(candidates[-1]))


def _make_experiment(
    k_cps: int, runway_assignment_mode: str, runways: List[str] | None
) -> CPSCoordinationExperiment:
    run_id = _find_pretrained_run_id()
    cfg = ExperimentConfig(
        model=CPSModelConfig(k_cps=k_cps, runway_assignment_mode=runway_assignment_mode),
        session=SessionConfig(pretrained_run_id=run_id, eval_episodes=1, do_train=False),
        env=CPSEnvConfig(env_kwargs=CPSEnvKwargsConfig(runways=runways)),
    )
    return CPSCoordinationExperiment(cfg)


def _run_with_cps(
    experiment: CPSCoordinationExperiment, model, n_aircraft: int, k_cps: int, mode: str
):
    env = experiment._make_multi_agent_env(n_aircraft)
    recat_matrix = experiment._load_recat_matrix()
    cps_manager = CPSManager(
        k_cps=k_cps,
        recat_matrix=recat_matrix,
        runway_assignment_mode=mode,
        delta_t_plan=120,
        delta_update=1.0,
        available_runways=experiment.cfg.env.env_kwargs.runways,
    )
    records = experiment._run_episode(
        env=env, model=model, cps_manager=cps_manager, surrogate=None,
        deterministic=True, ep_idx=0, seed=SEED,
    )
    env.close()
    return records, recat_matrix


def _run_without_cps(experiment: CPSCoordinationExperiment, model, n_aircraft: int):
    """Matched-seed control: drive the frozen worker with no TTA injection at all."""
    env = experiment._make_multi_agent_env(n_aircraft)
    obs, info_list = env.reset(seed=SEED)
    landing: dict[str, float] = {}
    while not env.is_episode_done():
        actions, _ = model.predict(obs, deterministic=True)
        _obs_t, _rew, terminated, truncated, info_terminal = env.step(actions)
        for row, info in enumerate(info_terminal):
            if terminated[row] or truncated[row]:
                landing[info["acid"]] = float(info.get("sim_time", 0.0))
        obs, info_list = env.get_active_batch()
    env.close()
    return landing


def check_step3_fcfs_static() -> bool:
    n_aircraft = 3
    experiment = _make_experiment(k_cps=0, runway_assignment_mode="static", runways=None)
    model = experiment.make_model(experiment._make_multi_agent_env(n_aircraft))

    cps_records, _ = _run_with_cps(experiment, model, n_aircraft, k_cps=0, mode="static")
    no_cps_landing = _run_without_cps(experiment, model, n_aircraft)

    ok = True
    if len(cps_records) != n_aircraft:
        print(f"FAIL: expected {n_aircraft} CPS episode records, got {len(cps_records)}")
        ok = False

    print("\n--- Step 3: k=0 FCFS static mode ---")
    print(f"{'acid':<10}{'assigned_tta':>14}{'cps_landing':>14}{'no_cps_landing':>16}{'rta_error_cps':>16}")
    any_differs = False
    for rec in cps_records:
        no_cps_t = no_cps_landing.get(rec.acid, float("nan"))
        print(f"{rec.acid:<10}{rec.assigned_tta:>14.1f}{rec.actual_landing_time:>14.1f}"
              f"{no_cps_t:>16.1f}{rec.rta_error_cps:>16.2f}")
        if no_cps_t == no_cps_t and abs(rec.actual_landing_time - no_cps_t) > 1e-6:
            any_differs = True

    if not any_differs:
        print("FAIL: no aircraft's CPS-coordinated landing time differs from its "
              "matched-seed no-injection control — TTA injection had no observable effect")
        ok = False
    else:
        print("PASS: at least one aircraft's landing time changed under CPS TTA "
              "injection vs. the matched-seed no-injection control")
    return ok


def check_step4_k_cps_separation() -> bool:
    n_aircraft = 5
    single_runway = ["27"]
    experiment = _make_experiment(k_cps=3, runway_assignment_mode="static", runways=single_runway)
    model = experiment.make_model(experiment._make_multi_agent_env(n_aircraft))

    records, recat_matrix = _run_with_cps(
        experiment, model, n_aircraft, k_cps=3, mode="static"
    )

    ok = True
    if len(records) != n_aircraft:
        print(f"FAIL: expected {n_aircraft} records, got {len(records)}")
        ok = False

    required_sep = recat_matrix.get("C", {}).get("C", 90.0)
    tolerance_s = 60.0  # worker's own RTA tracking error budget (RTA_TOLERANCE-scale)

    ordered = sorted(records, key=lambda r: r.assigned_tta)
    print("\n--- Step 4: k>0 CPS, single runway, RECAT-EU separation check ---")
    print(f"required separation (C/C) = {required_sep}s, tolerance = {tolerance_s}s")
    print(f"{'acid':<10}{'assigned_tta':>14}{'actual_landing':>16}{'gap_to_prev':>14}")
    violations = []
    prev = None
    for rec in ordered:
        gap = "-" if prev is None else f"{rec.assigned_tta - prev.assigned_tta:.1f}"
        print(f"{rec.acid:<10}{rec.assigned_tta:>14.1f}{rec.actual_landing_time:>16.1f}{gap:>14}")
        if prev is not None:
            actual_gap = rec.assigned_tta - prev.assigned_tta
            if actual_gap < required_sep - tolerance_s:
                violations.append((prev.acid, rec.acid, actual_gap))
        prev = rec

    if violations:
        print(f"FAIL: {len(violations)} TTA pairs violate required RECAT-EU separation "
              f"beyond tolerance: {violations}")
        ok = False
    else:
        print("PASS: every consecutive TTA pair on the shared runway respects "
              "RECAT-EU separation within tolerance")
    return ok


if __name__ == "__main__":
    passed_step3 = check_step3_fcfs_static()
    passed_step4 = check_step4_k_cps_separation()
    raise SystemExit(0 if (passed_step3 and passed_step4) else 1)
