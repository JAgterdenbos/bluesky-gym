"""
cps_coordination/experiments/coordination_baseline.py
------------------------------------------------------
CPSCoordinationExperiment — evaluation baseline that wraps a *frozen*
path-planning worker policy inside the CPS coordination layer.

Overview
--------
  1. Loads a pre-trained ``PathPlanningGoalEnv-v0`` SAC agent from the
     path_planning experiment registry (weights are never updated).
  2. Wraps the worker with a :class:`CPSManager` that schedules arrival
     sequences according to the k-CPS methodology.
  3. Runs M episodes each with N_a independent environment instances
     (one per aircraft), stepping the frozen worker in each env while
     CPSManager continuously updates TTA goals.
  4. Computes and logs the following downstream metrics:

     Metric               Symbol       Description
     ──────────────────── ──────────── ─────────────────────────────────────
     Total throughput      Γ           Aircraft landed per hour (all runways)
     Per-runway throughput Γ_r         Aircraft landed per hour per runway
     Sep. compliance       C_sep       Fraction of consecutive pairs with
                                       actual separation ≥ RECAT-EU minimum
     Tracking degradation  Δε_static   Mean |RTA_error_CPS| − |RTA_error_static|
                                       (Eq. tracking_degradation) — dynamic
                                       replanning vs. the same greedy-scheduled
                                       TTA assigned once and frozen
     Tracking degradation  Δε_uncoord  Mean |RTA_error_CPS| − |RTA_error_solo| —
                                       CPS-coordinated vs. an uncoordinated
                                       reference run under the identical
                                       frozen Worker (secondary metric, NOT
                                       Groot et al.'s published data)
     Recovery success rate R_rec       Fraction of mid-trajectory-updated
                                       flights (M_update) landing within the
                                       RTA tolerance δ_t (Eq. recovery_rate)
     Delay ripple index    ρ_ripple    Lag-1 autocorrelation of the RTA error
                                       sequence (measures delay propagation)

Comparison baselines
---------------------
  Three matched-seed passes are run per episode (:meth:`_run_episode`'s
  ``tta_mode``): ``"cps"`` (real k-CPS coordination, ongoing replanning),
  ``"static"`` (identical greedy-scheduled TTA, assigned once and frozen —
  the literal complement required by Eq. tracking_degradation/RQ2.2's
  question about the cost of replanning itself), and ``"solo"`` (the ETA
  surrogate's raw, unconstrained prediction, no CPS scheduling at all — a
  locally-generated uncoordinated reference under the same frozen Worker,
  *not* a reproduction of Groot et al.'s published dual-runway baseline,
  which this codebase does not have the data to reproduce).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from bluesky_gym.envs.pathplanning_goal_env import (
    ACTION_TIME,
    ALL_RUNWAYS,
    MAX_DISTANCE,
    MAX_TIME as _ENV_MAX_TIME,
    SPEED,
)
from bluesky_gym.envs.multi_agent_pathplanning_env import MultiAgentPathPlanningGoalEnv

from bluesky_gym.experiment import (
    BaseExperiment,
    BaseRegistry,
    EnvConfig,
    EnvKwargsConfig,
    MetricExtractor,
    ModelConfig,
    register_command,
)

from cps_coordination.coordination.cps_manager import AircraftState, CPSManager
from cps_coordination.coordination.eta_surrogate import ETASurrogate
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.config import (
    CPSEnvConfig,
    CPSEnvKwargsConfig,
    CPSModelConfig,
)
from cps_coordination.experiments.metrics import RTA_TOLERANCE_SEC, CPSMetricsReporter

# The frozen worker's actual registered gym id. Deliberately NOT read from
# ``cfg.env.env_name`` — that field is now the experiment-path namespace
# (see CPSEnvConfig docstring), kept distinct from "PathPlanningGoalEnv-v0"
# so CPS eval runs don't scatter empty run-id folders into the trained
# worker's own experiments/PathPlanningGoalEnv-v0/SAC/ directory tree.
_WORKER_ENV_ID = "PathPlanningGoalEnv-v0"


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────


class CPSCoordinationRegistry(BaseRegistry):
    """Persistent CSV registry for CPS coordination experiment runs.

    Tracks the intent, CPS configuration summary, and qualitative outcome
    for each run.  Per-run hyper-parameters are stored in
    ``experiments/.../config.yaml`` as usual.
    """

    def __init__(
        self, filepath: str = "./experiments/cps_registry.csv"
    ) -> None:
        super().__init__(filepath=filepath)

    @property
    def headers(self) -> List[str]:
        return [
            BaseRegistry.run_id,
            BaseRegistry.timestamp,
            "k_cps",
            "mode",
            "frozen_run_id",
            "intent",
            "status",
            "quality",
            "notes",
        ]

    @register_command(
        "Label the outcome of a finished CPS run.",
        status={"choices": ["running", "done", "failed", "abandoned"], "default": "done"},
        quality={"choices": ["good", "bad", "promising", "inconclusive"], "default": "good"},
        notes={"default": ""},
    )
    def label(
        self,
        run_id: str,
        status: str = "done",
        quality: str = "good",
        notes: str = "",
    ) -> None:
        self.update_run(run_id, {"status": status, "quality": quality, "notes": notes})
        print(f"Labelled {run_id}: {status} / {quality}")

    @register_command("Show a summary of all CPS coordination runs.")
    def list(self) -> None:
        rows = self._read_all()
        if not rows:
            return print("Registry is empty.")

        col_w = {"run_id": 20, "k_cps": 6, "mode": 9, "status": 10, "quality": 12}
        header = (
            f"{'RUN ID':<{col_w['run_id']}} | "
            f"{'K_CPS':<{col_w['k_cps']}} | "
            f"{'MODE':<{col_w['mode']}} | "
            f"{'STATUS':<{col_w['status']}} | "
            f"{'QUALITY':<{col_w['quality']}} | "
            "INTENT"
        )
        print(f"\n{header}")
        print("-" * (len(header) + 5))
        for r in rows:
            print(
                f"{r.get('run_id', ''):<{col_w['run_id']}} | "
                f"{r.get('k_cps', ''):<{col_w['k_cps']}} | "
                f"{r.get('mode', ''):<{col_w['mode']}} | "
                f"{r.get('status', ''):<{col_w['status']}} | "
                f"{r.get('quality', ''):<{col_w['quality']}} | "
                f"{r.get('intent', '')}"
            )
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Metric computation, printing, and disk logging live in
# cps_coordination/experiments/metrics.py (CPSMetricsReporter) — see that
# module's docstring for the pre-Step-10-audit Phase D.4 rationale for the
# split.
# ──────────────────────────────────────────────────────────────────────────────
# Episode log entry
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _EpisodeRecord:
    """Accumulator for per-aircraft data within a single episode."""

    acid: str
    episode_id: int                   # ep_idx this record was produced in -- needed
                                       # to scope cross-episode-pooled groupings (e.g.
                                       # separation-compliance landing pairs) to a
                                       # single episode's own simulation clock.
    runway_id: str
    wake_cat: str
    assigned_tta: float               # TTA set by CPSManager at episode end
    actual_landing_time: float        # Sim time of successful landing
    rta_error_cps: float              # |actual_time - assigned_tta|
    rta_error_solo: float             # |actual_time - unconstrained ETA|: locally-
                                       # generated uncoordinated reference under the
                                       # same frozen Worker, NOT Groot et al.'s data.
    tta_updated_mid_trajectory: bool  # Received a TTA update after its initial
                                       # assignment (Eq. recovery_rate's M_update
                                       # membership test). R_rec itself is
                                       # derived from this + rta_error_cps at
                                       # metrics-aggregation time (recomputable
                                       # at any tolerance, like C_sep) rather
                                       # than baked in here.
    success: bool
    stall_detected: bool = False      # CPSManager.is_stalled(acid) at episode
                                       # end -- flagged when distance-to-IAF
                                       # failed to shrink over a rolling window
                                       # (see cps_manager.py's STALL_WINDOW_S).
    rta_error_static: float = float("nan")  # |actual_time - static-pass TTA|
                                       # (Eq. tracking_degradation's ε_static):
                                       # the same greedy-scheduled TTA as the
                                       # CPS pass, assigned once and frozen --
                                       # NOT the same thing as rta_error_solo
                                       # (which has no CPS scheduling at all).
    arrival_index: int = -1           # n-th aircraft to spawn this episode (join key
                                       # for the two-pass baseline — robust to acid
                                       # reuse under a rolling-arrival-stream config,
                                       # unlike joining by acid directly).
    flight_id: str = ""               # f"{acid}#{episode_id}.{arrival_index}" -- a
                                       # human-grep-able, globally unique identity for
                                       # this flight. `acid` alone is NOT unique: it is
                                       # purely slot-derived (bluesky_gym's
                                       # MultiAgentPathPlanningGoalEnv._spawn_into_slot,
                                       # f"AC{slot:03d}") and gets reused both within an
                                       # episode (rolling-arrival-stream slot refills)
                                       # and trivially across every episode (slot
                                       # numbering restarts at each reset). Logging-only
                                       # -- never touches the real BlueSky callsign.
    death_cause: Optional[str] = None
    traj_x: List[float] = field(default_factory=list)  # populated only when
    traj_y: List[float] = field(default_factory=list)  # _run_episode(track_trajectory=True)


# ──────────────────────────────────────────────────────────────────────────────
# CPSCoordinationExperiment
# ──────────────────────────────────────────────────────────────────────────────


class CPSCoordinationExperiment(BaseExperiment):
    """Evaluation experiment that wraps a frozen worker inside the CPS layer.

    Class variables
    ---------------
    model_config_cls : CPSModelConfig
        Controls k-CPS hyper-parameters via CLI/YAML.
    env_config_cls : CPSEnvConfig
        Controls environment kwargs (v_app, runways, sampler path).

    The experiment *never trains* — ``do_train`` is forced to ``False`` and
    the frozen worker model is loaded from ``session.pretrained_run_id`` (or
    ``session.pretrained_model_path``) before the evaluation loop begins.

    Evaluation output
    -----------------
    All per-episode records are written to ``<save_path>/cps_eval_log.csv``.
    Aggregate metrics are printed to stdout and saved to
    ``<save_path>/cps_metrics.yaml``.
    """

    model_config_cls: Type[ModelConfig] = CPSModelConfig  # type: ignore[assignment]
    env_config_cls: Type[EnvConfig] = CPSEnvConfig  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Abstract method implementations (required by BaseExperiment)
    # ------------------------------------------------------------------ #

    def make_env(
        self,
        env_kwargs: dict,
        render_mode: Optional[str] = None,
    ) -> gym.Env:
        """Create a Monitor-wrapped single-agent worker env.

        Used only for framework compatibility (e.g. an "enjoy"/preview
        single-episode path) — the CPS coordination evaluation loop itself
        uses :meth:`_make_multi_agent_env` and never calls this.
        """
        import bluesky_gym

        bluesky_gym.register_envs()

        env = gym.make(_WORKER_ENV_ID, render_mode=render_mode, **env_kwargs)
        env.reset()
        return Monitor(env)

    def _make_multi_agent_env(
        self,
        n_aircraft: int,
        n_aircraft_total: Optional[int] = None,
        spawn_window_s: float = 0.0,
    ) -> MultiAgentPathPlanningGoalEnv:
        """Build the single-BlueSky-instance multi-aircraft env used by the
        CPS coordination evaluation loop.

        ``n_aircraft`` is ``max_concurrent_aircraft``. ``n_aircraft_total``
        defaults to ``n_aircraft`` (one "wave" of N_a aircraft per episode,
        matching the original single-wave-per-episode semantics used by
        every roadmap step through Step 9) — pass a larger value to get a
        genuine rolling arrival stream (Step 10's M=2,000 scale-up config).
        ``spawn_window_s`` defaults to ``0.0`` (every arrival eligible from
        time zero, i.e. today's instant-spawn/instant-refill behavior).

        ``runways`` is deliberately read from ``cfg.env.env_kwargs`` directly,
        NOT from ``cfg.eval_env_kwargs`` (pre-Step-10 audit, discovered live):
        ``ExperimentConfig.eval_env_kwargs``'s ``_inject_groups`` helper
        unconditionally overwrites the group kwarg (``"runways"``, per
        ``CPSEnvKwargsConfig.get_group_kwarg_name()``) with
        ``session.eval_groups`` -- which defaults to ``None`` -- silently
        discarding any explicit ``CPSEnvKwargsConfig(runways=[...])`` before
        it ever reaches the env. Every "restrict to N runways" evaluation in
        this package (``validate_cps_pipeline.py``'s single-runway checks,
        ``diagnose_success_rate.py``'s condition 4, CLI ``--runways``) went
        through this path and was silently running on all 12 runways
        instead. Bypassing ``eval_env_kwargs`` here fixes it locally without
        touching the shared framework property (out of this package's
        scope) -- ``session.eval_groups`` remains the (currently unused,
        for this env) alternate channel that property was designed for.
        """
        runways = self.cfg.env.env_kwargs.runways
        env_kwargs = self.cfg.eval_env_kwargs
        return MultiAgentPathPlanningGoalEnv(
            runways=runways,
            action_mode=env_kwargs.get("action_mode", "hdg"),
            max_concurrent_aircraft=n_aircraft,
            n_aircraft_total=n_aircraft_total if n_aircraft_total is not None else n_aircraft,
            spawn_window_s=spawn_window_s,
        )

    def make_model(self, env: gym.Env):  # type: ignore[override]
        """Load the frozen SAC worker policy; no training is performed.

        The model is loaded from ``cfg.session.pretrained_model_path`` (set
        by the framework from ``pretrained_run_id`` if needed).  The model's
        parameters are *never* modified.

        Parameters
        ----------
        env : gym.Env
            Environment used only for compatibility validation.

        Returns
        -------
        SAC
            Loaded, frozen SAC agent.

        Raises
        ------
        ValueError
            If no pretrained model path is configured.
        """
        path = self.cfg.session.pretrained_model_path
        if not path:
            raise ValueError(
                "CPSCoordinationExperiment requires a pre-trained worker model.\n"
                "Set 'session.pretrained_run_id' or 'session.pretrained_model_path' "
                "in your config / CLI flags."
            )
        print(f"Loading frozen worker policy from: {path}")
        model = SAC.load(path, env=env)
        # Freeze — no gradient updates; SB3 models are eval-only when loaded
        return model

    # ------------------------------------------------------------------ #
    # Metric extractor (info-dict based, used by the framework evaluate())
    # ------------------------------------------------------------------ #

    @classmethod
    def metric_extractor(cls) -> MetricExtractor:
        """Return a MetricExtractor for path-planning worker metrics."""
        return MetricExtractor(
            extractors={
                "flight_time_min": lambda info, ok: (
                    info.get("sim_time", float("nan")) / 60 if ok else float("nan")
                ),
                "on_time": lambda info, _ok: float(
                    info.get("on_time", float("nan"))
                ),
                "correct_runway": lambda info, _ok: float(
                    info.get("correct_runway", float("nan"))
                ),
                "rta_error_s": lambda info, _ok: float(
                    info.get("rta_error", float("nan"))
                ),
            },
            display=["flight_time_min", "on_time", "correct_runway", "rta_error_s"],
        )

    # ------------------------------------------------------------------ #
    # CPS coordination evaluation loop
    # ------------------------------------------------------------------ #

    def evaluate(self, model, deterministic: bool = True) -> dict[str, list[bool]]:  # type: ignore[override]
        """Run the CPS coordination evaluation loop.

        This override replaces the default single-episode evaluation with a
        multi-aircraft coordination loop (one shared ``MultiAgentPathPlanningGoalEnv``
        instance, reused across episodes) that measures all CPS-specific
        metrics.

        Algorithm per episode (see :meth:`_run_episode`)
        -------------------------------------------------
        1. Reset the shared env to spawn this episode's N_a aircraft.
        2. At each decision step: build :class:`AircraftState` records from
           the env's current active aircraft, run ``CPSManager.update_fleet``,
           push any resulting runway/TTA changes into the env, predict with
           the frozen worker in a single batched call, and step.
        3. Repeat until every aircraft scheduled for this episode has landed
           or been truncated.
        4. Compute aggregate metrics across all episodes and save logs.

        Parameters
        ----------
        model : SAC
            Frozen worker policy (loaded by :meth:`make_model`).
        deterministic : bool
            Whether to use deterministic (greedy) actions.

        Returns
        -------
        dict[str, list[bool]]
            Per-runway success lists (compatible with BaseExperiment.run()).
        """
        cfg = self.cfg
        mcfg = cast(CPSModelConfig, cfg.model)

        recat_matrix = self._load_recat_matrix()
        surrogate = self._build_surrogate()

        def _new_cps_manager() -> CPSManager:
            return CPSManager(
                k_cps=mcfg.k_cps,
                recat_matrix=recat_matrix,
                runway_assignment_mode=mcfg.runway_assignment_mode,
                delta_t_plan=mcfg.delta_t_plan,
                delta_update=mcfg.delta_update,
                available_runways=list(cfg.env.env_kwargs.runways or ALL_RUNWAYS),
                # Without this, lag features (delta_atd/cumabs_cte/heading_volatility)
                # are implicitly zero on every ETASurrogate call -- out-of-distribution
                # input for a model trained with those features populated (they
                # survived feature-importance reduction, see
                # cps_coordination/models/surrogate_feature_selection.yaml), which is
                # what was silently degrading ETA accuracy well outside RTA_TOLERANCE.
                trajectory_buffer=TrajectoryBuffer(),
                enable_stall_detection=mcfg.enable_stall_detection,
                enable_cross_cycle_runway_seeding=mcfg.enable_cross_cycle_runway_seeding,
            )

        # Three independent CPSManager instances — one per pass — so that the
        # CPS run's replanning state (_runway_last_committed/_prev_ttas) never
        # leaks into the static/solo runs', or vice versa (see roadmap step 7:
        # every baseline pass must be a genuinely independent rollout, not
        # derived from the CPS trajectory after the fact).
        cps_manager = _new_cps_manager()
        static_manager = _new_cps_manager()
        solo_manager = _new_cps_manager()

        n_episodes = cfg.session.eval_episodes
        n_aircraft = self._get_n_aircraft()
        env = self._make_multi_agent_env(n_aircraft)

        all_records: List[_EpisodeRecord] = []
        results: dict[str, list[bool]] = defaultdict(list)

        print(
            f"\nCPS Coordination Evaluation"
            f"\n  episodes={n_episodes}, aircraft/episode={n_aircraft}"
            f"\n  k_cps={mcfg.k_cps}, mode={mcfg.runway_assignment_mode}"
            f"\n  eta_surrogate={'loaded' if surrogate else 'none (naive ETA estimate)'}"
            f"\n  wake_separation={'reduced x' + str(mcfg.wake_separation_scale) if mcfg.reduced_wake_separation else 'full RECAT-EU'}"
            f"\n  frozen worker: {cfg.session.pretrained_model_path}\n"
        )

        try:
            for ep_idx in range(n_episodes):
                # Matched-seed two-pass baseline (roadmap step 7): the CPS run
                # and the solo/unconstrained control must see the identical
                # spawn/runway draw, or any landing-time difference is
                # confounded with random variation rather than caused by CPS
                # coordination. Re-running the *same* env with the same seed
                # (rather than deriving "solo" from the CPS trajectory after
                # the fact) is required because the two rollouts are causally
                # different from the first decision step onward.
                ep_seed = ep_idx

                cps_records = self._run_episode(
                    env=env,
                    model=model,
                    cps_manager=cps_manager,
                    surrogate=surrogate,
                    deterministic=deterministic,
                    ep_idx=ep_idx,
                    seed=ep_seed,
                    tta_mode="cps",
                )
                cps_manager.reset()

                static_records = self._run_episode(
                    env=env,
                    model=model,
                    cps_manager=static_manager,
                    surrogate=surrogate,
                    deterministic=deterministic,
                    ep_idx=ep_idx,
                    seed=ep_seed,
                    tta_mode="static",
                )
                static_manager.reset()

                solo_records = self._run_episode(
                    env=env,
                    model=model,
                    cps_manager=solo_manager,
                    surrogate=surrogate,
                    deterministic=deterministic,
                    ep_idx=ep_idx,
                    seed=ep_seed,
                    tta_mode="solo",
                )
                solo_manager.reset()

                ep_records = self._join_three_pass(cps_records, static_records, solo_records)
                all_records.extend(ep_records)

                for rec in ep_records:
                    results[rec.runway_id].append(rec.success)
        finally:
            env.close()

        reporter = self._make_metrics_reporter()
        metrics = reporter.compute_aggregate_metrics(all_records, recat_matrix)
        reporter.print_metrics(metrics)
        reporter.save_logs(all_records, metrics)

        return dict(results)

    # ------------------------------------------------------------------ #
    # Per-episode runner
    # ------------------------------------------------------------------ #

    def _run_episode(
        self,
        env: MultiAgentPathPlanningGoalEnv,
        model,
        cps_manager: CPSManager,
        surrogate: Optional[ETASurrogate],
        deterministic: bool,
        ep_idx: int,
        seed: Optional[int] = None,
        tta_mode: str = "cps",
        track_trajectory: bool = False,
    ) -> List[_EpisodeRecord]:
        """Run one episode in the shared multi-aircraft env.

        Per decision step: build :class:`AircraftState` records from the
        env's current active aircraft, run ``CPSManager.update_fleet``,
        push any resulting runway/TTA changes into the env via
        ``set_runway``/``set_tta``, refresh the observation batch, predict
        with the frozen worker in a single batched call, and step. Repeat
        until every aircraft scheduled for this episode has landed or been
        truncated (``env.is_episode_done()``).

        Three-pass baseline (roadmap step 7 + the static-TTA addition)
        -----------------------------------------------------------------
        ``tta_mode`` selects what gets injected via ``env.set_tta`` each
        step, but otherwise runs the identical loop (same fleet-building,
        same ``cps_manager.update_fleet`` call, same dynamic-runway sync):

        - ``"cps"``    — inject ``cps_manager.get_tta(acid)`` (the k-CPS
          greedy-scheduled TTA), gated by ``changed`` (only when the
          scheduled TTA actually shifted) — today's behaviour, ongoing
          replanning for the whole episode.
        - ``"static"`` — inject the *same* greedy-scheduled TTA from the
          *same* scheduler, but only on an acid's first-ever assignment;
          every later replanning cycle is computed (so other, not-yet-
          assigned aircraft are still correctly sequenced) but never
          re-injected into the env for an already-assigned acid. This is
          the literal complement of ``"cps"`` required by
          Eq. tracking_degradation/RQ2.2 — "does replanning itself cost
          tracking accuracy" needs a condition that is identical except for
          the replanning, not a differently-scheduled one.
        - ``"solo"``   — inject ``ac.eta`` directly (the surrogate's raw,
          unconstrained per-aircraft ETA prediction *before* k-CPS
          scheduling), every step, for every active aircraft. This is the
          frozen worker's own unconstrained target, uncontaminated by the
          CPS scheduler's positional shifting — a locally-generated
          uncoordinated reference, not Groot et al.'s published baseline.

        The caller (``evaluate()``) runs all three modes against the *same*
        env with the *same* seed and joins the record sets by
        ``arrival_index`` — never by acid, since a rolling-arrival-stream
        config would reuse acids per-slot and a literal-callsign join could
        pair the wrong physical flights across passes.

        Parameters
        ----------
        env : MultiAgentPathPlanningGoalEnv
            Shared multi-aircraft env (reset here; not created/closed here).
        model : SAC
            Frozen worker policy.
        cps_manager : CPSManager
            Sequence manager for *this pass* (reset before each episode by
            the caller) — must be a separate instance per pass so replanning
            state never leaks between the CPS and solo rollouts.
        surrogate : ETASurrogate or None
            ETA prediction model. ``None`` → ETAs come only from the naive
            straight-line estimate computed at fleet-build time (see
            :meth:`_estimate_naive_eta`) and are never refreshed.
        deterministic : bool
            Whether to use deterministic policy predictions.
        ep_idx : int
            Episode index (used for logging).
        tta_mode : str
            ``"cps"``, ``"static"``, or ``"solo"`` — see above.

        Returns
        -------
        List[_EpisodeRecord]
            One record per aircraft in this episode, in arrival order.
            ``rta_error_solo``/``rta_error_static`` are left as NaN here in
            every mode — the caller fills them in during the join (see
            :meth:`_join_three_pass`).
        """
        if tta_mode not in ("cps", "static", "solo"):
            raise ValueError(f"tta_mode must be 'cps', 'static', or 'solo', got {tta_mode!r}")

        obs, info_list = env.reset(seed=seed)
        sim_time = 0.0
        records: List[_EpisodeRecord] = []
        last_tta: Dict[str, float] = {}
        arrival_order: Dict[str, int] = {}
        # Monotonic counter for `arrival_order` values -- NOT `len(arrival_order)`.
        # Under a rolling-arrival-stream config, `arrival_order` only ever holds
        # `max_concurrent_aircraft` distinct acid keys (one per slot): once every
        # slot has spawned its first occupant, `len(arrival_order)` stops growing,
        # so re-keying a recycled slot's *second* (or later) occupant with
        # `len(arrival_order)` would silently reassign the exact same value every
        # time a slot recycles -- collapsing every non-first occupant of every
        # slot onto one shared `arrival_index`/`flight_id`, and reintroducing the
        # cross-contaminated `_join_three_pass` join this whole detector exists to
        # prevent, just with a different trigger than the original acid-reuse bug.
        next_arrival_order = 0
        # Test A (pre-Step-10 audit §1.2/§1.4): per-episode cache of each
        # acid's first-committed remaining_time_budget value, populated and
        # consulted by _compute_remaining_time_budget only when
        # CPSModelConfig.freeze_remaining_time_budget is set.
        frozen_remaining_time_budget: Dict[str, float] = {}
        trajectories: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
        # Eq. recovery_rate's M_update membership test (cps-mode only): an
        # acid enters `assigned_once` on its first-ever TTA assignment, and
        # `mid_traj_updated` on any *subsequent* one (a genuine mid-trajectory
        # update, not the initial assignment).
        assigned_once: set = set()
        mid_traj_updated: set = set()
        # `acid` is derived purely from slot index
        # (MultiAgentPathPlanningGoalEnv._spawn_into_slot: f"AC{slot:03d}"),
        # so under a rolling-arrival-stream config (total_arrivals_per_episode
        # > max_concurrent_aircraft) the same acid string is reused within a
        # single episode once a slot frees up and a new arrival spawns into
        # it. Every acid-keyed accumulator below must therefore be reset on
        # each new occupancy, or a later arrival silently inherits/overwrites
        # an earlier, unrelated arrival's state (this previously caused
        # `records` -- then a Dict[str, _EpisodeRecord] -- to silently drop
        # every arrival but the last one per slot).
        #
        # A plain "acid not seen last iteration" set-diff is NOT sufficient:
        # MultiAgentPathPlanningGoalEnv._finalize_step refills a freed slot
        # with the next scheduled arrival within the *same* env.step() call
        # that terminated the previous occupant, so the acid string is never
        # actually absent between two consecutive info_list snapshots -- the
        # new occupant is silently missed and inherits the old occupant's
        # bookkeeping. `info["spawn_time"]` (the slot's absolute spawn
        # instant) is unique per physical occupancy of a slot -- it strictly
        # increases across a same-step swap -- so diffing on (acid,
        # spawn_time) instead of acid alone detects same-step refills too.
        prev_slot_spawn_time: Dict[str, float] = {}

        while not env.is_episode_done():
            current_slot_spawn_time = {
                info["acid"]: info.get("spawn_time", 0.0) for info in info_list
            }
            for acid, spawn_time in current_slot_spawn_time.items():
                if prev_slot_spawn_time.get(acid) != spawn_time:
                    arrival_order[acid] = next_arrival_order
                    next_arrival_order += 1
                    trajectories.pop(acid, None)
                    assigned_once.discard(acid)
                    mid_traj_updated.discard(acid)
                    last_tta.pop(acid, None)
                    frozen_remaining_time_budget.pop(acid, None)
            prev_slot_spawn_time = current_slot_spawn_time

            fleet = self._build_fleet(obs, info_list, sim_time, frozen_remaining_time_budget)
            acid_to_slot = {info["acid"]: info["slot"] for info in info_list}

            if track_trajectory:
                for ac in fleet:
                    xs, ys = trajectories[ac.acid]
                    xs.append(float(ac.state[0]))
                    ys.append(float(ac.state[1]))

            # --- CPS coordination step ---
            changed = cps_manager.update_fleet(
                aircraft=fleet,
                current_time=sim_time,
                surrogate=surrogate,
            )

            # Dynamic runway reassignment isn't gated by delta_update, so
            # check every active aircraft, not just `changed`.
            for ac in fleet:
                slot = acid_to_slot[ac.acid]
                if ac.runway_id != env.current_runway[slot]:
                    env.set_runway(slot, ac.runway_id)

            if tta_mode in ("cps", "static"):
                for acid in changed:
                    if tta_mode == "static" and acid in assigned_once:
                        continue  # frozen: never re-inject after the first assignment
                    tta = cps_manager.get_tta(acid)
                    if tta is not None:
                        if acid in assigned_once:
                            mid_traj_updated.add(acid)
                        assigned_once.add(acid)
                        last_tta[acid] = tta
                        env.set_tta(acid_to_slot[acid], tta)
            else:  # "solo" — inject the raw, unconstrained ETA every step.
                for ac in fleet:
                    last_tta[ac.acid] = ac.eta
                    env.set_tta(acid_to_slot[ac.acid], ac.eta)

            # Refresh after set_runway/set_tta before the frozen policy acts.
            obs, info_list = env.get_active_batch()
            actions, _ = model.predict(obs, deterministic=deterministic)
            _obs_terminal, _rewards, terminated, truncated, info_terminal = env.step(actions)

            for row, info in enumerate(info_terminal):
                if terminated[row] or truncated[row]:
                    acid = info["acid"]
                    # "sim_time" is local (elapsed since this aircraft's own
                    # spawn); assigned_tta/cps_manager.get_tta/ac.eta are all
                    # on the episode's global clock (CPSManager sequences
                    # TTAs across aircraft with different spawn times, so
                    # they have to be). Converting to a global landing time
                    # here is a no-op whenever spawn_time == 0 (every
                    # existing single-wave config), and is required for
                    # rta_error_cps/rta_error_solo to compare like with
                    # like, and for actual_landing_time to be meaningfully
                    # comparable *across* aircraft downstream (throughput,
                    # separation compliance, ripple index all sort/diff
                    # landing times between different aircraft).
                    spawn_time = float(info.get("spawn_time", 0.0))
                    landing_time = spawn_time + float(info.get("sim_time", sim_time))
                    success = bool(info.get("is_success", False))
                    assigned_tta = last_tta.get(acid, float("nan"))
                    rta_error_cps = (
                        abs(landing_time - assigned_tta)
                        if not math.isnan(assigned_tta) else float("nan")
                    )
                    tta_updated_mid_trajectory = acid in mid_traj_updated
                    stall_detected = cps_manager.is_stalled(acid)
                    traj_x, traj_y = trajectories[acid] if track_trajectory else ([], [])

                    records.append(_EpisodeRecord(
                        acid=acid,
                        episode_id=ep_idx,
                        runway_id=str(info.get("current_runway", "")),
                        wake_cat="D",
                        assigned_tta=assigned_tta,
                        actual_landing_time=landing_time,
                        rta_error_cps=rta_error_cps,
                        rta_error_solo=float("nan"),
                        rta_error_static=float("nan"),
                        tta_updated_mid_trajectory=tta_updated_mid_trajectory,
                        stall_detected=stall_detected,
                        success=success,
                        arrival_index=arrival_order[acid],
                        flight_id=f"{acid}#{ep_idx}.{arrival_order[acid]}",
                        death_cause=info.get("death_cause"),
                        traj_x=list(traj_x),
                        traj_y=list(traj_y),
                    ))

            sim_time += ACTION_TIME
            obs, info_list = env.get_active_batch()

        return records

    @staticmethod
    def _join_three_pass(
        cps_records: List[_EpisodeRecord],
        static_records: List[_EpisodeRecord],
        solo_records: List[_EpisodeRecord],
    ) -> List[_EpisodeRecord]:
        """Join the CPS/static/solo pass record sets by ``arrival_index``.

        Returns the CPS-pass records (runway/success/assigned_tta/landing
        time all come from the CPS-coordinated rollout) with
        ``rta_error_static``/``rta_error_solo`` filled in from the matching
        static-pass/solo-pass record's own tracking error (``rta_error_cps``
        field of that record, since :meth:`_run_episode` computes "error vs.
        whatever was actually injected" identically in every mode — in
        static mode that's the once-assigned, frozen TTA; in solo mode the
        unconstrained ETA). Joining by ``arrival_index`` rather than ``acid``
        keeps this correct even under a rolling-arrival-stream config where
        per-slot acids repeat.

        An arrival present in one pass but not another (e.g. a truncation
        that changes step timing enough to shift a later spawn) is dropped
        with a printed warning rather than silently producing a NaN Δε.
        """
        static_by_arrival = {rec.arrival_index: rec for rec in static_records}
        solo_by_arrival = {rec.arrival_index: rec for rec in solo_records}

        joined: List[_EpisodeRecord] = []
        for cps_rec in cps_records:
            static_rec = static_by_arrival.get(cps_rec.arrival_index)
            solo_rec = solo_by_arrival.get(cps_rec.arrival_index)
            if static_rec is None or solo_rec is None:
                missing = "static-pass" if static_rec is None else "solo-pass"
                print(
                    f"WARNING: no {missing} match for arrival_index="
                    f"{cps_rec.arrival_index} (acid={cps_rec.acid!r}); "
                    "dropping from this episode's joined records."
                )
                continue
            cps_rec.rta_error_static = static_rec.rta_error_cps
            cps_rec.rta_error_solo = solo_rec.rta_error_cps
            joined.append(cps_rec)
        return joined

    def _build_fleet(
        self,
        obs: dict,
        info_list: List[dict],
        current_time: float,
        frozen_remaining_time_budget: Optional[Dict[str, float]] = None,
    ) -> List[AircraftState]:
        """Construct ``AircraftState`` records from the env's current batch.

        ``eta`` is seeded here with :meth:`_estimate_naive_eta` (a
        straight-line placeholder) for every aircraft — this is what
        ``CPSManager`` sees before its first replanning cycle, and what it
        falls back to permanently when no surrogate is supplied. When
        ``update_fleet(..., surrogate=<real ETASurrogate>)`` is called (as
        :meth:`evaluate` does via :meth:`_build_surrogate`), ``_refresh_etas``
        overwrites every aircraft's ``eta`` in place with
        ``surrogate.predict_eta_fleet(...)`` each planning cycle — see
        ``cps_manager.py``. (x, y) here are the env's Schiphol-centred
        normalised coordinates; ``cartesian_to_polar`` computes the
        surrogate's ``r``/``theta`` features from that same raw frame at both
        training and inference time, so no coordinate reconciliation is
        needed (see the "Coordinate-frame finding" in the Phase III plan and
        ``eta_surrogate.py``'s module docstring).

        Parameters
        ----------
        frozen_remaining_time_budget : Dict[str, float], optional
            Per-episode cache threaded in by :meth:`_run_episode` for Test A
            (pre-Step-10 audit §1.2/§1.4, ``CPSModelConfig.freeze_remaining_time_budget``).
            Callers that don't need the freeze/clamp ablations (e.g. direct
            single-call test harnesses) can omit this -- a fresh, empty dict
            is used internally, which is a no-op unless the config flag is set.
        """
        if frozen_remaining_time_budget is None:
            frozen_remaining_time_budget = {}
        fleet = []
        for row, info in enumerate(info_list):
            x, y = float(obs["observation"][row][0]), float(obs["observation"][row][1])
            elapsed_steps = info["sim_time"] / ACTION_TIME
            heading_deg = float(np.degrees(info["heading"]))
            eta = self._estimate_naive_eta(obs["observation"][row], info, current_time)
            remaining_time_budget = self._compute_remaining_time_budget(
                info, frozen_remaining_time_budget,
            )
            fleet.append(
                AircraftState(
                    acid=info["acid"],
                    state=np.array(
                        [x, y, elapsed_steps, heading_deg, remaining_time_budget],
                        dtype=np.float32,
                    ),
                    runway_id=str(info["current_runway"]),
                    eta=eta,
                    wake_cat="D",
                    spawn_time=float(info.get("spawn_time", 0.0)),
                )
            )
        return fleet

    def _compute_remaining_time_budget(
        self, info: dict, frozen_remaining_time_budget: Dict[str, float],
    ) -> float:
        """Finding-2 feature: the frozen worker's own active temporal target
        minus elapsed time since spawn (mirrors ``rta - t`` in training data).

        ``info["goal_vector"][2]`` is left at its 0.0 spawn placeholder
        (``PathPlanningGoalEnv._compute_goal_vector``: "the temporal
        component is left at 0.0 ... CPSManager is the sole source of
        TTAs, injected separately via set_tta") until CPSManager's first
        ``set_tta()`` call for this aircraft, so a 0.0 value here always
        means "no TTA committed yet" and the fallback below correctly
        returns 0.0 rather than a meaningless ``-elapsed_time``. A nonzero
        value always reflects an already-committed past decision, never
        leakage of a not-yet-decided future one.

        Test A / Test B (pre-Step-10 audit §1.2/§1.4) -- both gated off by
        default (today's behaviour unchanged unless explicitly opted into):

        - ``CPSModelConfig.freeze_remaining_time_budget``: once this acid's
          first nonzero (TTA-committed) value is seen, cache and return it
          for every subsequent call this episode instead of recomputing
          ``tta - t`` -- isolates the surrogate-side channel of the §1.1
          feedback loop by removing the mechanism through which the
          surrogate's own prior-cycle output re-enters as a bigger input.
        - ``CPSModelConfig.remaining_time_budget_cap_s``: cap the (still
          freshly recomputed every cycle) value at this ceiling -- bounds
          the loop without removing the feature entirely. Ignored when
          freeze is also active (freeze already fixes the value).
        """
        goal_t = float(info["goal_vector"][2])
        if goal_t == 0.0:
            return 0.0
        value = goal_t * _ENV_MAX_TIME - float(info["sim_time"])

        mcfg = cast(CPSModelConfig, self.cfg.model)
        acid = info["acid"]
        if mcfg.freeze_remaining_time_budget:
            if acid in frozen_remaining_time_budget:
                return frozen_remaining_time_budget[acid]
            frozen_remaining_time_budget[acid] = value
            return value

        if mcfg.remaining_time_budget_cap_s is not None:
            return min(value, mcfg.remaining_time_budget_cap_s)

        return value

    @staticmethod
    def _estimate_naive_eta(obs_row: np.ndarray, info: dict, current_time: float) -> float:
        """Straight-line ETA estimate: current_time + (distance to the
        assigned runway's IAF, at constant SPEED).

        Used as CPSManager's ETA source whenever a fitted ETASurrogate isn't
        wired in (see :meth:`_build_fleet`'s note on the coordinate-frame gap)
        — good enough to validate the k-CPS sequencing pipeline end-to-end
        (roadmap steps 3-4), not a substitute for the trained surrogate's
        prediction.
        """
        goal_vector = info["goal_vector"]
        dx = float(goal_vector[0]) - float(obs_row[0])
        dy = float(goal_vector[1]) - float(obs_row[1])
        dist_km = math.hypot(dx, dy) * MAX_DISTANCE
        remaining_s = (dist_km * 1000.0) / SPEED
        return current_time + remaining_s

    # ------------------------------------------------------------------ #
    # Metrics reporter factory
    # ------------------------------------------------------------------ #

    def _make_metrics_reporter(self) -> CPSMetricsReporter:
        """Build the :class:`CPSMetricsReporter` for this experiment's config.

        Reads ``cps_eval.separation_tolerance_s``/``cps_eval.rta_tolerance_s``
        from ``cps_base.yaml`` (via :meth:`_cfg_or_default`) once here, rather
        than the reporter re-reading the YAML file on every metrics call.
        """
        cps_eval_cfg = self._cfg_or_default("cps_eval", {})
        return CPSMetricsReporter(
            save_path=self.cfg.save_path,
            separation_tolerance_s=float(cps_eval_cfg.get("separation_tolerance_s", 5.0)),
            rta_tolerance_s=float(cps_eval_cfg.get("rta_tolerance_s", RTA_TOLERANCE_SEC)),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_surrogate(self) -> Optional[ETASurrogate]:
        """Construct an ETASurrogate from CPSModelConfig.eta_surrogate_path.

        Falls back to the canonical ``cps_coordination/models/eta_surrogate.pkl``
        if the field is unset and that file exists; returns ``None`` (naive
        straight-line ETA estimate, see ``_estimate_naive_eta``) otherwise.
        """
        mcfg = cast(CPSModelConfig, self.cfg.model)
        path = mcfg.eta_surrogate_path
        if not path:
            default_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "models", "eta_surrogate.pkl")
            )
            path = default_path if os.path.exists(default_path) else None
        if not path:
            return None
        # Trust the serialized sim_dt rather than overriding it: sim_dt is
        # "seconds per unit of the model's predicted output", and its
        # correct value now depends on the model's own _target — ACTION_TIME
        # (120s) for a target="steps" model (path_planning/rta/collect.py's
        # "step" column, one per env.step() decision step) but 1.0 for a
        # target="seconds" model (Finding 1's continuous time_to_go
        # candidate, which predicts seconds directly). ETASurrogate.load()
        # (unlike from_sampler_path, which unconditionally overrides sim_dt)
        # preserves whatever train_surrogate.py baked in for that target at
        # save time -- overriding here to a single fixed ACTION_TIME would
        # silently 120x-inflate every prediction from a target="seconds"
        # model (this is exactly what promoting
        # eta_surrogate_combined_candidate.pkl to production surfaced).
        surrogate = ETASurrogate.load(path)
        expected_sim_dt = ACTION_TIME if surrogate._target == "steps" else 1.0
        if surrogate.sim_dt != expected_sim_dt:
            raise ValueError(
                f"ETASurrogate at {path!r} has target={surrogate._target!r} but "
                f"sim_dt={surrogate.sim_dt} (expected {expected_sim_dt}) -- "
                "this model was likely saved with a stale/incorrect sim_dt; "
                "refusing to silently mis-scale every ETA prediction."
            )
        return surrogate

    def _load_recat_matrix(self) -> Dict[str, Dict[str, float]]:
        """Load the RECAT-EU matrix from cps_base.yaml, or use a safe default.

        The matrix is stored as a top-level ``recat_eu`` key in the YAML
        config so it is editable without touching Python source.  Falls back
        to a conservative 90-second flat matrix if no YAML is available.

        When ``CPSModelConfig.reduced_wake_separation`` is set, every value is
        scaled by ``wake_separation_scale`` before being returned — this is
        the single source both ``CPSManager``'s greedy scheduler and the
        ``C_sep`` compliance metric read from (``evaluate()`` loads this once
        and passes the same dict to both), so the two stay consistent with
        each other under the reduced-separation scenario.
        """
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "cps_base.yaml",
        )
        config_path = os.path.normpath(config_path)
        if os.path.exists(config_path):
            with open(config_path) as fh:
                data = yaml.safe_load(fh) or {}
            matrix = data.get("recat_eu", {})
            if matrix:
                matrix = {
                    lead: {trail: float(v) for trail, v in row.items()}
                    for lead, row in matrix.items()
                }
            else:
                matrix = None
        else:
            matrix = None
        if matrix is None:
            # Conservative fallback: 90 s for all category pairs
            cats = ["A", "B", "C", "D", "E", "F"]
            matrix = {lead: {trail: 90.0 for trail in cats} for lead in cats}

        mcfg = cast(CPSModelConfig, self.cfg.model)
        if mcfg.reduced_wake_separation:
            scale = mcfg.wake_separation_scale
            matrix = {
                lead: {trail: v * scale for trail, v in row.items()}
                for lead, row in matrix.items()
            }
        return matrix

    def _get_n_aircraft(self) -> int:
        """Read n_aircraft_per_episode from cps_base.yaml or default to 5."""
        config_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "configs", "cps_base.yaml"
            )
        )
        if os.path.exists(config_path):
            with open(config_path) as fh:
                data = yaml.safe_load(fh) or {}
            return int(data.get("cps_eval", {}).get("n_aircraft_per_episode", 5))
        return 5

    def _cfg_or_default(self, section: str, default: Any) -> Any:
        """Read a section from cps_base.yaml, returning *default* on failure."""
        config_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "configs", "cps_base.yaml"
            )
        )
        if os.path.exists(config_path):
            with open(config_path) as fh:
                data = yaml.safe_load(fh) or {}
            return data.get(section, default)
        return default

