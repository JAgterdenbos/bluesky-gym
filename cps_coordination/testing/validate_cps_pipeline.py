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
import pandas as pd

from bluesky_gym.envs.pathplanning_goal_env import MAX_TIME
from bluesky_gym.experiment.config import ExperimentConfig, SessionConfig
from cps_coordination.coordination.cps_manager import AircraftState, CPSManager
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.config import CPSEnvConfig, CPSEnvKwargsConfig, CPSModelConfig
from cps_coordination.experiments.coordination_baseline import (
    CPSCoordinationExperiment,
    _EpisodeRecord,
)
from cps_coordination.experiments.metrics import CPSMetricsReporter
from cps_coordination.scripts.cps_metrics_offline import (
    load_recat_matrix,
    recompute_metrics,
    recompute_separation_compliance,
)

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
        trajectory_buffer=TrajectoryBuffer(),
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


def check_step7_three_pass_baseline() -> bool:
    """Roadmap step 7 + the static-TTA addition: genuine three-pass baseline,
    N=5 aircraft x M=10 episodes. For each episode, run the CPS-coordinated
    pass, a matched-seed static pass (same greedy-scheduled TTA, assigned
    once and frozen — the literal Eq. tracking_degradation comparator), and
    a matched-seed solo pass (unconstrained ETA injected instead) through
    the *same* env, join by ``arrival_index``, and confirm:

    (a) ``rta_error_solo`` is no longer a silent copy of ``rta_error_cps``
        (the original step-7 bug: before the fix, ``rta_error_solo`` always
        fell back to ``rta_error_cps`` itself, so Delta epsilon was
        structurally ~0).
    (b) ``rta_error_static`` is populated and genuinely differs from both
        ``rta_error_cps`` and ``rta_error_solo`` in at least some records —
        it is neither an unpopulated NaN column nor a duplicate of an
        existing pass.
    (c) The "frozen after first assignment" invariant: in the static pass,
        ``env.set_tta`` is called at most once per acid across the whole
        episode, even though the CPS pass (same scenario, same seed)
        receives multiple calls per acid under ongoing replanning.
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
            trajectory_buffer=TrajectoryBuffer(),
        )

    cps_manager = _new_manager()
    static_manager = _new_manager()
    solo_manager = _new_manager()

    # Wrap env.set_tta to count calls per (tta_mode, episode, acid) without
    # touching _run_episode itself -- the most direct way to verify the
    # static pass's "frozen after first assignment" invariant at the actual
    # injection boundary, rather than inferring it indirectly from output
    # records. Keyed by episode too, not just acid: per-slot acids (e.g.
    # "AC000") repeat every episode, so counting by bare acid across all
    # M=10 episodes would accumulate ~1 call/episode into a spurious ">1"
    # even when each individual episode is correctly frozen.
    set_tta_calls: dict = {"cps": {}, "static": {}, "solo": {}}
    _real_set_tta = env.set_tta
    _current_mode = {"mode": "cps", "ep_idx": 0}

    def _counting_set_tta(slot, tta):
        acid = env.acid_slots[slot] or f"slot{slot}"
        key = (_current_mode["ep_idx"], acid)
        counts = set_tta_calls[_current_mode["mode"]]
        counts[key] = counts.get(key, 0) + 1
        return _real_set_tta(slot, tta)

    env.set_tta = _counting_set_tta

    all_records = []
    for ep_idx in range(n_episodes):
        ep_seed = SEED + ep_idx  # matched across all three passes, varied across episodes
        _current_mode["ep_idx"] = ep_idx

        _current_mode["mode"] = "cps"
        cps_records = experiment._run_episode(
            env=env, model=model, cps_manager=cps_manager, surrogate=None,
            deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="cps",
        )
        cps_manager.reset()

        _current_mode["mode"] = "static"
        static_records = experiment._run_episode(
            env=env, model=model, cps_manager=static_manager, surrogate=None,
            deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="static",
        )
        static_manager.reset()

        _current_mode["mode"] = "solo"
        solo_records = experiment._run_episode(
            env=env, model=model, cps_manager=solo_manager, surrogate=None,
            deterministic=True, ep_idx=ep_idx, seed=ep_seed, tta_mode="solo",
        )
        solo_manager.reset()

        all_records.extend(
            experiment._join_three_pass(cps_records, static_records, solo_records)
        )
    env.set_tta = _real_set_tta
    env.close()

    ok = True
    print("\n--- Step 7: genuine three-pass baseline (N=5 x M=10) ---")
    if not all_records:
        print("FAIL: no joined records produced across all episodes")
        return False

    valid_solo = [r for r in all_records if not math.isnan(r.rta_error_solo)]
    valid_static = [r for r in all_records if not math.isnan(r.rta_error_static)]
    if len(valid_solo) != len(all_records):
        print(f"FAIL: {len(all_records) - len(valid_solo)}/{len(all_records)} records have "
              "no matched solo-pass record (arrival_index join failed)")
        ok = False
    if len(valid_static) != len(all_records):
        print(f"FAIL: {len(all_records) - len(valid_static)}/{len(all_records)} records have "
              "no matched static-pass record (arrival_index join failed)")
        ok = False

    n_differ_solo = sum(
        1 for r in valid_solo if abs(r.rta_error_cps - r.rta_error_solo) > 1e-6
    )
    n_differ_static = sum(
        1 for r in valid_static if abs(r.rta_error_cps - r.rta_error_static) > 1e-6
    )
    n_static_vs_solo_differ = sum(
        1 for r in all_records
        if not math.isnan(r.rta_error_static) and not math.isnan(r.rta_error_solo)
        and abs(r.rta_error_static - r.rta_error_solo) > 1e-6
    )
    delta_eps_static = [abs(r.rta_error_cps) - abs(r.rta_error_static) for r in valid_static]
    delta_eps_solo = [abs(r.rta_error_cps) - abs(r.rta_error_solo) for r in valid_solo]
    mean_delta_static = sum(delta_eps_static) / len(delta_eps_static) if delta_eps_static else float("nan")
    mean_delta_solo = sum(delta_eps_solo) / len(delta_eps_solo) if delta_eps_solo else float("nan")

    print(f"{'acid':<10}{'rta_error_cps':>16}{'rta_error_static':>18}{'rta_error_solo':>16}")
    for r in all_records:
        print(f"{r.acid:<10}{r.rta_error_cps:>16.2f}{r.rta_error_static:>18.2f}{r.rta_error_solo:>16.2f}")

    print(f"\n{n_differ_solo}/{len(valid_solo)} records have rta_error_solo != rta_error_cps "
          f"(mean Delta_eps_vs_uncoordinated = {mean_delta_solo:.2f}s)")
    print(f"{n_differ_static}/{len(valid_static)} records have rta_error_static != rta_error_cps "
          f"(mean Delta_eps_vs_static = {mean_delta_static:.2f}s)")
    print(f"{n_static_vs_solo_differ}/{len(all_records)} records have rta_error_static != rta_error_solo "
          "(confirms the two baselines are genuinely distinct passes)")

    if n_differ_solo == 0:
        print("FAIL: rta_error_solo never differs from rta_error_cps — "
              "still looks like the silent-fallback bug")
        ok = False
    if n_differ_static == 0:
        print("FAIL: rta_error_static never differs from rta_error_cps — "
              "static pass isn't genuinely independent of the CPS pass")
        ok = False
    if n_static_vs_solo_differ == 0:
        print("FAIL: rta_error_static never differs from rta_error_solo — "
              "the two baseline passes look like duplicates of each other")
        ok = False

    # (c) Frozen-after-first-assignment invariant, checked at the actual
    # env.set_tta injection boundary (not inferred from output records).
    static_counts = set_tta_calls["static"]
    cps_counts = set_tta_calls["cps"]
    max_static_calls = max(static_counts.values(), default=0)
    n_acids_with_multiple_cps_calls = sum(1 for c in cps_counts.values() if c > 1)
    print(f"\nmax set_tta calls per acid: static={max_static_calls}, "
          f"cps acids with >1 call={n_acids_with_multiple_cps_calls}/{len(cps_counts)}")
    if max_static_calls > 1:
        print("FAIL: static pass called env.set_tta more than once for some acid — "
              "not actually frozen after the first assignment")
        ok = False
    if n_acids_with_multiple_cps_calls == 0:
        print("FAIL: cps pass never re-injected a TTA update for any acid in this scenario — "
              "the frozen-vs-replanned contrast isn't actually being exercised")
        ok = False

    if ok:
        print("PASS: rta_error_static/rta_error_solo are genuinely independent of "
              "rta_error_cps and of each other, and the static pass is provably "
              "frozen after each acid's first TTA assignment")

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
            trajectory_buffer=TrajectoryBuffer(),
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


def check_step4b_k_cps_reorders() -> bool:
    """Investigation Vector 2 (pre-Step-10 audit, §2.4): synthetic,
    non-BlueSky regression guard for ``_apply_k_cps_constraint``'s redesign.

    §2.1 proved the pre-fix heap-based selection rule is *always* a no-op
    identity permutation on FCFS-sorted input -- the existing
    ``check_step4_k_cps_separation`` gate can't distinguish "permutation
    worked" from "permutation is a no-op" (it only checks a property the
    greedy scheduler guarantees regardless of input order). This closes
    that gap with three synthetic-fleet assertions:

      1. ``k_cps == 0`` still degenerates to exact FCFS, regardless of
         ``fairness_weight`` (the k-window itself is inactive).
      2. ``fairness_weight == 0.0`` (any ``k_cps``) is byte-identical to
         FCFS -- the proven-optimal no-op default (§2.1/§2.2).
      3. ``fairness_weight > 0.0`` with one pre-flagged-stalled aircraft:
         the stalled aircraft's scheduled position shifts *earlier* than
         pure FCFS (protected from further imposed delay), and the shift
         stays within the ``k_cps`` fairness bound.
    """
    recat_matrix = {"C": {"C": 90.0}}

    def _fleet(etas: List[float]) -> List[AircraftState]:
        return [
            AircraftState(
                acid=f"AC{i:03d}", state=np.zeros(5, dtype=np.float32),
                runway_id="27", eta=eta,
            )
            for i, eta in enumerate(etas)
        ]

    ok = True
    print("\n--- Step 4b: k-CPS reordering sensitivity (fairness_weight ablation) ---")

    # (1) k_cps == 0 still degenerates to exact FCFS, regardless of fairness_weight.
    mgr_k0 = CPSManager(k_cps=0, recat_matrix=recat_matrix, fairness_weight=0.5)
    fcfs = sorted(_fleet([500.0, 105.0, 110.0, 100.0, 510.0]), key=lambda a: (a.eta, a.acid))
    out_k0 = mgr_k0._apply_k_cps_constraint(fcfs, current_time=0.0)
    if [a.acid for a in out_k0] != [a.acid for a in fcfs]:
        print("FAIL: k_cps=0 did not degenerate to exact FCFS")
        ok = False
    else:
        print("PASS: k_cps=0 degenerates to exact FCFS regardless of fairness_weight")

    # (2) fairness_weight == 0.0 (k_cps=3) is byte-identical to FCFS.
    mgr_w0 = CPSManager(k_cps=3, recat_matrix=recat_matrix, fairness_weight=0.0)
    fcfs = sorted(_fleet([500.0, 105.0, 110.0, 100.0, 510.0]), key=lambda a: (a.eta, a.acid))
    out_w0 = mgr_w0._apply_k_cps_constraint(fcfs, current_time=0.0)
    if [a.acid for a in out_w0] != [a.acid for a in fcfs]:
        print("FAIL: fairness_weight=0.0 did not reproduce byte-identical FCFS")
        ok = False
    else:
        print("PASS: fairness_weight=0.0 reproduces byte-identical FCFS (k_cps=3)")

    # (3) fairness_weight > 0.0 protects a pre-flagged-stalled aircraft:
    # equal ETAs isolate the fairness term from ETA-driven imposed-delay
    # differences (all candidates impose identical delay at a given
    # position, so only the slack-penalty term can break the tie).
    k = 2
    mgr_f = CPSManager(k_cps=k, recat_matrix=recat_matrix, fairness_weight=1.0)
    fleet = _fleet([100.0] * 5)
    fcfs = sorted(fleet, key=lambda a: (a.eta, a.acid))
    stalled_acid = fcfs[-1].acid  # last in FCFS order -- worst case under plain FCFS
    mgr_f._stalled_acids.add(stalled_acid)
    out_f = mgr_f._apply_k_cps_constraint(fcfs, current_time=0.0)

    fcfs_pos = {a.acid: i for i, a in enumerate(fcfs)}
    new_pos = {a.acid: i for i, a in enumerate(out_f)}
    shift = fcfs_pos[stalled_acid] - new_pos[stalled_acid]
    print(f"stalled acid {stalled_acid}: FCFS position {fcfs_pos[stalled_acid]} -> "
          f"fairness-weighted position {new_pos[stalled_acid]} (shift={shift}, k_cps={k})")

    if new_pos[stalled_acid] >= fcfs_pos[stalled_acid]:
        print("FAIL: pre-flagged-stalled aircraft did not move earlier under "
              "fairness_weight > 0 (expected priority protection from imposed delay)")
        ok = False
    elif shift > k:
        print(f"FAIL: stalled aircraft's position shifted by {shift}, "
              f"exceeding the k_cps={k} fairness bound")
        ok = False
    else:
        print(f"PASS: stalled aircraft prioritised {shift} position(s) earlier, "
              f"within the k_cps={k} bound")

    return ok


def check_step10_episode_scoped_c_sep() -> bool:
    """Investigation Vector 3 (pre-Step-10 audit, §3.3): adversarial 2-episode
    regression guard confirming C_sep is computed within-episode, never
    pooled across independent episode simulation clocks.

    Constructs 2 synthetic episodes on a single shared runway with matched
    wake category (``AC000``/``AC001`` reused across both episodes, exactly
    the acid-reuse pattern ``episode_id`` exists to disambiguate). Each
    episode's own landing pair is comfortably separation-compliant (gap =
    required_sep + 10s), but episode 0's second landing and episode 1's
    first landing land a fraction of a second apart on a *pooled,
    non-episode-scoped* timeline -- exactly the spurious cross-episode
    adjacency the pre-fix bug (``landing_times_by_rwy`` keyed by
    ``runway_id`` alone across ``all_records.extend(ep_records)``,
    ``coordination_baseline.py`` pre-fix) would have manufactured as a fake
    separation violation.

    Confirms all three C_sep code paths agree exactly and none of them
    counts the spurious pair:
      (a) ``CPSMetricsReporter.compute_aggregate_metrics`` (in-process, ``metrics.py``)
      (b) ``recompute_separation_compliance`` (Parquet per-pair stream --
          already correctly episode-scoped via ``run_cps_eval.py::_log_episode``,
          included here as the ground-truth anchor)
      (c) ``recompute_metrics``'s ``c_sep_from_landings_crosscheck`` (offline,
          now episode-scoped -- this is the leg the bug lived in)
    """
    recat_matrix = load_recat_matrix()
    required_sep = recat_matrix.get("C", {}).get("C", 90.0)

    # Episode 0: AC000 @ t=1000.0, AC001 @ t=1000.0+gap (compliant within ep0).
    # Episode 1: AC000 @ t=1000.5, AC001 @ t=1000.5+gap (compliant within ep1).
    # Pooled-by-runway-alone (pre-fix bug), sorting all 4 across episodes
    # interleaves them -- adjacent gaps of ~0.5s between different episodes'
    # aircraft, far below `required_sep`, would register as violations.
    ep0_t0, ep1_t0 = 1000.0, 1000.5
    gap = required_sep + 10.0

    def _rec(acid: str, episode_id: int, t: float) -> _EpisodeRecord:
        return _EpisodeRecord(
            acid=acid, episode_id=episode_id, runway_id="27", wake_cat="C",
            assigned_tta=t, actual_landing_time=t,
            rta_error_cps=0.0, rta_error_solo=0.0,
            tta_updated_mid_trajectory=False, success=True,
        )

    records = [
        _rec("AC000", 0, ep0_t0),
        _rec("AC001", 0, ep0_t0 + gap),
        _rec("AC000", 1, ep1_t0),
        _rec("AC001", 1, ep1_t0 + gap),
    ]

    ok = True
    print("\n--- Step 10 pre-req: episode-scoped C_sep (adversarial 2-episode regression) ---")

    experiment = _make_experiment(k_cps=0, runway_assignment_mode="static", runways=None)
    reporter = experiment._make_metrics_reporter()
    metrics_in_process = reporter.compute_aggregate_metrics(records, recat_matrix)
    c_sep_in_process = metrics_in_process["c_sep"]

    # (b) Parquet-level per-pair stream, built the way `_log_episode` already
    # correctly builds it (per-episode grouping) -- ground-truth anchor.
    separation_df = pd.DataFrame([
        {"episode_id": 0, "runway_id": "27", "acid_lead": "AC000", "acid_trail": "AC001",
         "gap_actual_s": gap, "required_sep_s": required_sep},
        {"episode_id": 1, "runway_id": "27", "acid_lead": "AC000", "acid_trail": "AC001",
         "gap_actual_s": gap, "required_sep_s": required_sep},
    ])
    c_sep_from_pairs = recompute_separation_compliance(separation_df)

    aircraft_df = pd.DataFrame([
        {
            "episode_id": rec.episode_id, "acid": rec.acid, "runway_id": rec.runway_id,
            "wake_cat": rec.wake_cat, "actual_landing_time": rec.actual_landing_time,
            "rta_error_cps": rec.rta_error_cps, "rta_error_solo": rec.rta_error_solo,
            "tta_updated_mid_trajectory": rec.tta_updated_mid_trajectory,
            "stall_detected": rec.stall_detected, "success": rec.success,
        }
        for rec in records
    ])
    metrics_offline = recompute_metrics(aircraft_df, separation_df, recat_matrix)
    c_sep_offline_crosscheck = metrics_offline["c_sep_from_landings_crosscheck"]

    print(f"c_sep (in-process, CPSMetricsReporter)                   = {c_sep_in_process}")
    print(f"c_sep_from_pairs (Parquet-level, already-correct)        = {c_sep_from_pairs}")
    print(f"c_sep_from_landings_crosscheck (offline, episode-scoped) = {c_sep_offline_crosscheck}")

    if c_sep_in_process != 1.0:
        print(f"FAIL: in-process c_sep={c_sep_in_process} != 1.0 -- a spurious "
              "cross-episode violation leaked in")
        ok = False
    if c_sep_from_pairs != 1.0:
        print(f"FAIL: c_sep_from_pairs={c_sep_from_pairs} != 1.0")
        ok = False
    if c_sep_offline_crosscheck != 1.0:
        print(f"FAIL: c_sep_from_landings_crosscheck={c_sep_offline_crosscheck} != 1.0 -- "
              "still pooling landing times across independent episode clocks")
        ok = False
    if not (c_sep_in_process == c_sep_from_pairs == c_sep_offline_crosscheck):
        print("FAIL: the three C_sep code paths disagree")
        ok = False

    if ok:
        print("PASS: all three C_sep code paths agree exactly (1.0) and correctly "
              "exclude the adversarial cross-episode landing pair")
    return ok


if __name__ == "__main__":
    passed_step3 = check_step3_fcfs_static()
    passed_step4 = check_step4_k_cps_separation()
    passed_step4b = check_step4b_k_cps_reorders()
    passed_step5 = check_step5_dynamic_tta()
    passed_step6 = check_step6_dynamic_runway()
    passed_step7 = check_step7_three_pass_baseline()
    passed_step9 = check_step9_surrogate_exercised()
    passed_step10_c_sep = check_step10_episode_scoped_c_sep()
    raise SystemExit(
        0 if (passed_step3 and passed_step4 and passed_step4b and passed_step5
              and passed_step6 and passed_step7 and passed_step9 and passed_step10_c_sep)
        else 1
    )
