"""
Roadmap steps 3-6 validation: exercise the real ``CPSCoordinationExperiment``
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

Step 5 (dynamic TTA update mid-trajectory): confirm ``env.set_tta()`` called
mid-flight re-injects a new target time without resetting the aircraft's
physical state (position, sim clock, slot/index), touches only the temporal
component of ``desired_goal``, and visibly changes worker behaviour relative
to a matched-seed control that never receives the update.

Step 6 (dynamic runway assignment): confirm ``env.set_runway()`` called
mid-flight re-targets the slot's spatial goal and success-check runway
without resetting any physical/episode state, and that the terminal
``is_success``/``death_cause`` end up evaluated against the *new* runway.

Step 9 (real ETASurrogate genuinely exercised): confirm that passing a real,
non-``None`` ``ETASurrogate`` into ``CPSManager.update_fleet``/
``CPSCoordinationExperiment._run_episode`` produces ``ac.eta``/scheduled TTA
values that measurably diverge from the naive straight-line estimate
(``_estimate_naive_eta``) both (a) directly, in a single ``update_fleet``
call, and (b) end-to-end, comparing two matched-seed "solo" episodes (one
with ``surrogate=None``, one with the real surrogate) run through
``_run_episode``.

Run: python cps_coordination/testing/validate_cps_pipeline.py
"""

from __future__ import annotations

import glob
import math
import os
from typing import List, Optional, Tuple

import bluesky as bs
import numpy as np

from bluesky_gym.envs.pathplanning_goal_env import MAX_TIME
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


def check_step5_dynamic_tta() -> bool:
    """Mid-trajectory TTA update: ``env.set_tta()`` at a decision step well
    into the flight must not reset the aircraft's physical state, sim clock,
    or slot/index — only the temporal component of ``desired_goal`` moves —
    and the change must visibly alter worker behaviour vs. a matched-seed
    control that never receives it.
    """
    n_aircraft = 1
    experiment = _make_experiment(k_cps=0, runway_assignment_mode="static", runways=None)
    env = experiment._make_multi_agent_env(n_aircraft)
    model = experiment.make_model(env)

    inject_step = 10
    tta_delta = 3600.0  # 1 hour later than the naive straight-line estimate

    def _run(apply_injection: bool):
        obs, info_list = env.reset(seed=SEED)
        slot = info_list[0]["slot"]

        naive_eta = CPSCoordinationExperiment._estimate_naive_eta(
            obs["observation"][0], info_list[0], 0.0
        )
        env.set_tta(slot, naive_eta)
        obs, info_list = env.get_active_batch()

        snapshot: dict = {}
        landing_time: Optional[float] = None
        step_i = 0
        while not env.is_episode_done() and step_i < 300:
            if apply_injection and step_i == inject_step:
                pre_acid = env.acid_slots[slot]
                pre_idx = env._acid_to_idx[pre_acid]
                pre_lat = float(bs.traf.lat[pre_idx])
                pre_lon = float(bs.traf.lon[pre_idx])
                pre_simt = env.simt[slot]
                pre_goal_xy = env.goal_vector[slot][:2].copy()

                new_tta = naive_eta + tta_delta
                env.set_tta(slot, new_tta)

                post_acid = env.acid_slots[slot]
                post_idx = env._acid_to_idx.get(post_acid)
                snapshot = {
                    "pre_acid": pre_acid, "post_acid": post_acid,
                    "pre_idx": pre_idx, "post_idx": post_idx,
                    "pre_lat": pre_lat, "post_lat": float(bs.traf.lat[post_idx]),
                    "pre_lon": pre_lon, "post_lon": float(bs.traf.lon[post_idx]),
                    "pre_simt": pre_simt, "post_simt": env.simt[slot],
                    "pre_goal_xy": pre_goal_xy, "post_goal_xy": env.goal_vector[slot][:2].copy(),
                    "post_goal_t": env.goal_vector[slot][2],
                    "expected_goal_t": new_tta / MAX_TIME,
                }
                obs, info_list = env.get_active_batch()

            actions, _ = model.predict(obs, deterministic=True)
            _obs_t, _rew, terminated, truncated, info_terminal = env.step(actions)
            for row, info in enumerate(info_terminal):
                if terminated[row] or truncated[row]:
                    landing_time = float(info.get("sim_time", 0.0))
            obs, info_list = env.get_active_batch()
            step_i += 1

        return snapshot, landing_time, naive_eta

    snapshot, landing_with_injection, naive_eta = _run(apply_injection=True)
    _snapshot_ctrl, landing_control, _naive_eta_ctrl = _run(apply_injection=False)
    env.close()

    ok = True
    print("\n--- Step 5: dynamic TTA update mid-trajectory ---")
    if not snapshot:
        print(f"FAIL: injection never fired — episode ended before step {inject_step}")
        return False

    if snapshot["post_acid"] != snapshot["pre_acid"] or snapshot["post_idx"] != snapshot["pre_idx"]:
        print(f"FAIL: set_tta triggered a respawn/index change "
              f"({snapshot['pre_acid']}@{snapshot['pre_idx']} -> "
              f"{snapshot['post_acid']}@{snapshot['post_idx']})")
        ok = False

    if abs(snapshot["post_lat"] - snapshot["pre_lat"]) > 1e-9 or \
       abs(snapshot["post_lon"] - snapshot["pre_lon"]) > 1e-9:
        print("FAIL: set_tta moved the aircraft's physical position")
        ok = False

    if snapshot["post_simt"] != snapshot["pre_simt"]:
        print(f"FAIL: set_tta perturbed the sim clock "
              f"({snapshot['pre_simt']} -> {snapshot['post_simt']})")
        ok = False

    if not np.allclose(snapshot["post_goal_xy"], snapshot["pre_goal_xy"], atol=1e-12):
        print(f"FAIL: set_tta altered the spatial goal "
              f"({snapshot['pre_goal_xy']} -> {snapshot['post_goal_xy']})")
        ok = False

    if abs(snapshot["post_goal_t"] - snapshot["expected_goal_t"]) > 1e-9:
        print(f"FAIL: temporal goal not updated to the injected TTA "
              f"(got {snapshot['post_goal_t']}, expected {snapshot['expected_goal_t']})")
        ok = False

    print(f"naive_eta={naive_eta:.1f}s, injected_tta={naive_eta + tta_delta:.1f}s "
          f"(delta={tta_delta:.0f}s) at decision step {inject_step}")
    print(f"landing_time_with_injection={landing_with_injection}, "
          f"landing_time_control={landing_control}")

    if landing_with_injection is None or landing_control is None:
        print("FAIL: one of the two runs never terminated within the step cap")
        ok = False
    elif abs(landing_with_injection - landing_control) < 1e-6:
        print("FAIL: mid-flight TTA injection had no observable effect on landing time "
              "vs. the matched-seed control")
        ok = False

    if ok:
        print("PASS: mid-flight set_tta preserved physical/episode state (no reset), "
              "only moved the temporal goal, and visibly changed worker behaviour")
    return ok


def check_step6_dynamic_runway() -> bool:
    """Mid-trajectory runway reassignment: ``env.set_runway()`` must not
    reset any physical/episode state, must move only the spatial goal (the
    temporal target is preserved), and the terminal ``is_success``/
    ``death_cause`` must end up evaluated against the *new* runway.
    """
    n_aircraft = 1
    runway_a, runway_b = "18C", "24"
    reassign_step = 5
    # Seed chosen (see cps_coordination/testing/validate_cps_pipeline.py history)
    # so the frozen worker reliably redirects to runway B's IAF and lands
    # cleanly rather than clipping B's own RESTRICT wedge from an awkward
    # entry angle -- a real geometric risk of *any* mid-flight reassignment
    # this close to the terminal area, independent of whether set_runway()
    # itself behaves correctly. SEED (module-level) also works for this pair
    # but is less robust across reassign_step choices.
    step6_seed = 11

    experiment = _make_experiment(
        k_cps=0, runway_assignment_mode="static", runways=[runway_a, runway_b]
    )
    env = experiment._make_multi_agent_env(n_aircraft)
    model = experiment.make_model(env)

    obs, info_list = env.reset(seed=step6_seed)
    slot = info_list[0]["slot"]

    # Force the initial assignment to runway A regardless of what the
    # random spawn draw picked, so the A -> B reassignment is well defined.
    env.current_runway[slot] = runway_a
    env.non_overlapping_runways[slot] = env._compute_non_overlapping_runways(runway_a)
    env.goal_vector[slot] = env._compute_goal_vector(runway_a)
    obs, info_list = env.get_active_batch()

    ok = True
    snapshot: dict = {}
    final_info: Optional[dict] = None

    print("\n--- Step 6: dynamic runway reassignment mid-flight ---")
    print(f"runway_a={runway_a!r} -> runway_b={runway_b!r} at decision step {reassign_step}")

    step_i = 0
    while not env.is_episode_done() and step_i < 300:
        if step_i == reassign_step:
            pre_acid = env.acid_slots[slot]
            pre_idx = env._acid_to_idx[pre_acid]
            pre_lat = float(bs.traf.lat[pre_idx])
            pre_lon = float(bs.traf.lon[pre_idx])
            pre_simt = env.simt[slot]
            pre_goal_t = env.goal_vector[slot][2]

            env.set_runway(slot, runway_b)

            post_acid = env.acid_slots[slot]
            post_idx = env._acid_to_idx.get(post_acid)
            snapshot = {
                "pre_acid": pre_acid, "post_acid": post_acid,
                "pre_idx": pre_idx, "post_idx": post_idx,
                "pre_lat": pre_lat, "post_lat": float(bs.traf.lat[post_idx]),
                "pre_lon": pre_lon, "post_lon": float(bs.traf.lon[post_idx]),
                "pre_simt": pre_simt, "post_simt": env.simt[slot],
                "pre_goal_t": pre_goal_t, "post_goal_t": env.goal_vector[slot][2],
                "current_runway": env.current_runway[slot],
                "non_overlapping": list(env.non_overlapping_runways[slot]),
            }
            obs, info_list = env.get_active_batch()

        actions, _ = model.predict(obs, deterministic=True)
        _obs_t, _rew, terminated, truncated, info_terminal = env.step(actions)
        for row, info in enumerate(info_terminal):
            if terminated[row] or truncated[row]:
                final_info = info
        obs, info_list = env.get_active_batch()
        step_i += 1

    env.close()

    if not snapshot:
        print(f"FAIL: reassignment never fired — episode ended before step {reassign_step}")
        return False

    if snapshot["post_acid"] != snapshot["pre_acid"] or snapshot["post_idx"] != snapshot["pre_idx"]:
        print(f"FAIL: set_runway triggered a respawn/index change "
              f"({snapshot['pre_acid']}@{snapshot['pre_idx']} -> "
              f"{snapshot['post_acid']}@{snapshot['post_idx']})")
        ok = False

    if abs(snapshot["post_lat"] - snapshot["pre_lat"]) > 1e-9 or \
       abs(snapshot["post_lon"] - snapshot["pre_lon"]) > 1e-9:
        print("FAIL: set_runway moved the aircraft's physical position")
        ok = False

    if snapshot["post_simt"] != snapshot["pre_simt"]:
        print(f"FAIL: set_runway perturbed the sim clock "
              f"({snapshot['pre_simt']} -> {snapshot['post_simt']})")
        ok = False

    if abs(snapshot["post_goal_t"] - snapshot["pre_goal_t"]) > 1e-12:
        print("FAIL: set_runway altered the temporal goal component "
              f"({snapshot['pre_goal_t']} -> {snapshot['post_goal_t']})")
        ok = False

    if snapshot["current_runway"] != runway_b:
        print(f"FAIL: current_runway not updated to {runway_b!r} "
              f"(got {snapshot['current_runway']!r})")
        ok = False

    if runway_a not in snapshot["non_overlapping"]:
        print(f"FAIL: non_overlapping_runways for {runway_b!r} doesn't include "
              f"{runway_a!r} — a stray landing on A's sink wouldn't be flagged "
              f"'wrong_runway' after reassignment: {snapshot['non_overlapping']}")
        ok = False

    if final_info is None:
        print(f"FAIL: episode never terminated within the step cap "
              f"(last runway={snapshot['current_runway']!r})")
        ok = False
    else:
        print(f"terminal: runway={final_info.get('current_runway')!r} "
              f"death_cause={final_info.get('death_cause')!r} "
              f"is_success={final_info.get('is_success')}")
        if final_info.get("current_runway") != runway_b:
            print(f"FAIL: terminal current_runway is {final_info.get('current_runway')!r}, "
                  f"expected {runway_b!r}")
            ok = False
        if final_info.get("death_cause") == "success" and final_info.get("current_runway") != runway_b:
            print("FAIL: 'success' was recorded against a stale runway")
            ok = False
        if final_info.get("death_cause") != "success":
            print(f"FAIL: episode did not land successfully after reassignment "
                  f"(death_cause={final_info.get('death_cause')!r}) — cannot confirm the "
                  "success check tracked runway B's polyline")
            ok = False

    if ok:
        print("PASS: mid-flight set_runway preserved physical/episode state (no reset), "
              "moved only the spatial goal, and the terminal success check tracked runway B")
    return ok


def check_step7_two_pass_solo_baseline() -> bool:
    """Roadmap step 7: genuine two-pass solo baseline, N=5 aircraft x M=10
    episodes.  For each episode, run the CPS-coordinated pass and a matched-
    seed solo pass (unconstrained ETA injected instead of the k-CPS-scheduled
    TTA) through the *same* env, join by ``arrival_index``, and confirm
    ``rta_error_solo`` is no longer a silent copy of ``rta_error_cps`` (the
    bug this step fixes: before the fix, ``rta_error_solo`` always fell back
    to ``rta_error_cps`` itself, so Delta epsilon was structurally ~0).
    """
    n_aircraft = 5
    n_episodes = 10
    k_cps = 2

    experiment = _make_experiment(k_cps=k_cps, runway_assignment_mode="static", runways=None)
    env = experiment._make_multi_agent_env(n_aircraft)
    model = experiment.make_model(env)
    recat_matrix = experiment._load_recat_matrix()

    def _new_manager() -> CPSManager:
        return CPSManager(
            k_cps=k_cps,
            recat_matrix=recat_matrix,
            runway_assignment_mode="static",
            delta_t_plan=120,
            delta_update=1.0,
            available_runways=experiment.cfg.env.env_kwargs.runways,
        )

    cps_manager = _new_manager()
    solo_manager = _new_manager()

    all_records = []
    for ep_idx in range(n_episodes):
        ep_seed = SEED + ep_idx  # matched between the two passes, varied across episodes

        cps_records = experiment._run_episode(
            env=env, model=model, cps_manager=cps_manager, surrogate=None,
            deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="cps",
        )
        cps_manager.reset()

        solo_records = experiment._run_episode(
            env=env, model=model, cps_manager=solo_manager, surrogate=None,
            deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="solo",
        )
        solo_manager.reset()

        all_records.extend(experiment._join_two_pass(cps_records, solo_records))
    env.close()

    ok = True
    print("\n--- Step 7: genuine two-pass solo baseline (N=5 x M=10) ---")
    if not all_records:
        print("FAIL: no joined records produced across all episodes")
        return False

    valid = [r for r in all_records if not math.isnan(r.rta_error_solo)]
    if len(valid) != len(all_records):
        print(f"FAIL: {len(all_records) - len(valid)}/{len(all_records)} records have "
              "no matched solo-pass record (arrival_index join failed)")
        ok = False

    n_differ = sum(
        1 for r in valid if abs(r.rta_error_cps - r.rta_error_solo) > 1e-6
    )
    delta_eps = [abs(r.rta_error_cps) - abs(r.rta_error_solo) for r in valid]
    mean_delta_eps = sum(delta_eps) / len(delta_eps) if delta_eps else float("nan")

    print(f"{'acid':<10}{'rta_error_cps':>16}{'rta_error_solo':>16}{'differs':>10}")
    for r in all_records:
        differs = "yes" if (not math.isnan(r.rta_error_solo)
                             and abs(r.rta_error_cps - r.rta_error_solo) > 1e-6) else "no"
        print(f"{r.acid:<10}{r.rta_error_cps:>16.2f}{r.rta_error_solo:>16.2f}{differs:>10}")

    print(f"\n{n_differ}/{len(valid)} records have rta_error_solo != rta_error_cps "
          f"(mean Delta epsilon = {mean_delta_eps:.2f}s)")

    if n_differ == 0:
        print("FAIL: rta_error_solo never differs from rta_error_cps — "
              "still looks like the silent-fallback bug")
        ok = False
    else:
        print("PASS: rta_error_solo is genuinely computed independently of rta_error_cps")

    return ok


def check_step9_surrogate_exercised() -> bool:
    """Roadmap step 9: confirm a real (non-``None``) ``ETASurrogate`` is
    genuinely exercised end-to-end, not just accepted as a parameter.

    (a) Direct: build a fleet, run one ``CPSManager.update_fleet`` call with
        the real surrogate, and confirm ``ac.eta`` (overwritten in place by
        ``_refresh_etas``) measurably diverges from ``_estimate_naive_eta``'s
        straight-line value for the same aircraft/state.
    (b) End-to-end: run two matched-seed "solo" episodes (``tta_mode="solo"``
        injects ``ac.eta`` directly every step, bypassing k-CPS scheduling so
        the comparison isolates the surrogate's effect) through
        ``_run_episode`` — one with ``surrogate=None``, one with the real
        surrogate — and confirm the resulting scheduled TTAs differ.
    """
    n_aircraft = 5
    experiment = _make_experiment(k_cps=0, runway_assignment_mode="static", runways=None)
    env = experiment._make_multi_agent_env(n_aircraft)
    model = experiment.make_model(env)
    surrogate = experiment._build_surrogate()

    ok = True
    print("\n--- Step 9: real ETASurrogate genuinely exercised (vs. naive straight-line ETA) ---")
    if surrogate is None:
        print("FAIL: _build_surrogate() returned None — no eta_surrogate.pkl found; "
              "cannot validate a real surrogate is exercised")
        env.close()
        return False

    def _new_manager() -> CPSManager:
        return CPSManager(
            k_cps=0,
            recat_matrix=experiment._load_recat_matrix(),
            runway_assignment_mode="static",
            delta_t_plan=120,
            delta_update=1.0,
            available_runways=experiment.cfg.env.env_kwargs.runways,
        )

    # --- (a) direct: does one update_fleet(surrogate=<real>) call actually
    # mutate ac.eta away from the naive fallback it was seeded with? ---
    obs, info_list = env.reset(seed=SEED)
    naive_etas = {ac.acid: ac.eta for ac in experiment._build_fleet(obs, info_list, 0.0)}

    fleet_for_refresh = experiment._build_fleet(obs, info_list, 0.0)
    _new_manager().update_fleet(fleet_for_refresh, current_time=0.0, surrogate=surrogate)
    surrogate_etas = {ac.acid: ac.eta for ac in fleet_for_refresh}

    print(f"{'acid':<10}{'naive_eta':>14}{'surrogate_eta':>16}{'abs_diff':>12}")
    any_direct_diff = False
    for acid in naive_etas:
        s_eta = surrogate_etas.get(acid, float("nan"))
        diff = abs(naive_etas[acid] - s_eta)
        print(f"{acid:<10}{naive_etas[acid]:>14.1f}{s_eta:>16.1f}{diff:>12.1f}")
        if diff > 1.0:
            any_direct_diff = True

    if not any_direct_diff:
        print("FAIL: update_fleet(surrogate=<real>) produced ac.eta values "
              "indistinguishable from the naive straight-line estimate for every aircraft")
        ok = False
    else:
        print("PASS (direct): update_fleet with a real surrogate overwrites ac.eta "
              "with values that measurably diverge from the naive estimate")

    # --- (b) end-to-end: two matched-seed 'solo' episodes, naive vs. real surrogate ---
    naive_records = experiment._run_episode(
        env=env, model=model, cps_manager=_new_manager(), surrogate=None,
        deterministic=True, ep_idx=0, seed=SEED, tta_mode="solo",
    )
    surrogate_records = experiment._run_episode(
        env=env, model=model, cps_manager=_new_manager(), surrogate=surrogate,
        deterministic=True, ep_idx=0, seed=SEED, tta_mode="solo",
    )
    env.close()

    naive_by_arrival = {r.arrival_index: r for r in naive_records}
    surrogate_by_arrival = {r.arrival_index: r for r in surrogate_records}
    common = sorted(set(naive_by_arrival) & set(surrogate_by_arrival))

    print(f"\n{'arrival_idx':<12}{'naive_tta':>14}{'surrogate_tta':>16}{'differs':>10}")
    n_differ = 0
    for idx in common:
        n_rec, s_rec = naive_by_arrival[idx], surrogate_by_arrival[idx]
        differs = abs(n_rec.assigned_tta - s_rec.assigned_tta) > 1.0
        n_differ += int(differs)
        print(f"{idx:<12}{n_rec.assigned_tta:>14.1f}{s_rec.assigned_tta:>16.1f}{'yes' if differs else 'no':>10}")

    if not common:
        print("FAIL: no matching arrival_index between the two runs — cannot compare")
        ok = False
    elif n_differ == 0:
        print("FAIL: no aircraft's end-to-end scheduled TTA differs between the "
              "naive-ETA run and the real-surrogate run")
        ok = False
    else:
        print(f"PASS (end-to-end): {n_differ}/{len(common)} aircraft got a measurably "
              "different scheduled TTA when a real ETASurrogate was exercised end-to-end")

    return ok


if __name__ == "__main__":
    passed_step3 = check_step3_fcfs_static()
    passed_step4 = check_step4_k_cps_separation()
    passed_step5 = check_step5_dynamic_tta()
    passed_step6 = check_step6_dynamic_runway()
    passed_step7 = check_step7_two_pass_solo_baseline()
    passed_step9 = check_step9_surrogate_exercised()
    raise SystemExit(
        0 if (passed_step3 and passed_step4 and passed_step5 and passed_step6
              and passed_step7 and passed_step9)
        else 1
    )
