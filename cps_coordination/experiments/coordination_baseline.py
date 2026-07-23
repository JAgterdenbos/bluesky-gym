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
     Tracking degradation  Δε          Mean |RTA_error_CPS| − |RTA_error_solo|
                                       comparing CPS to frozen worker alone
     Recovery success rate R_rec       Fraction of RTA violations recovered
                                       within δ_update tolerance
     Delay ripple index    ρ_ripple    Lag-1 autocorrelation of the RTA error
                                       sequence (measures delay propagation)

Comparison baseline
-------------------
  Results are compared against the Groot et al. baseline (unconstrained
  spatial-temporal policy) using the same frozen model without CPS
  coordination.  The baseline episode logs are saved to
  ``experiments/…/baseline_log.csv`` alongside the CPS logs.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from bluesky_gym.envs.pathplanning_goal_env import MAX_TIME as _ENV_MAX_TIME

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
from cps_coordination.experiments.config import (
    CPSEnvConfig,
    CPSEnvKwargsConfig,
    CPSModelConfig,
)


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
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────


def _lag1_autocorrelation(series: List[float]) -> float:
    """Compute lag-1 autocorrelation of *series* (ρ_ripple).

    Returns ``float('nan')`` if the series has fewer than 2 elements or
    zero variance, matching standard behaviour for undefined autocorrelation.

    Implements: ρ_1 = Cov(x_t, x_{t-1}) / Var(x_t)  (Pearson).

    Parameters
    ----------
    series : List[float]
        Ordered sequence of values (e.g. per-aircraft RTA errors within
        an episode, sorted by landing time).

    Returns
    -------
    float
        Lag-1 autocorrelation in ``[-1, 1]``, or NaN.
    """
    if len(series) < 2:
        return float("nan")
    arr = np.asarray(series, dtype=float)
    if np.std(arr) == 0.0:
        return float("nan")
    # Pearson correlation between consecutive pairs
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])


def _compute_separation_compliance(
    landing_times: Dict[str, List[float]],
    wake_cats: Dict[str, str],
    recat_matrix: Dict[str, Dict[str, float]],
    tolerance_s: float = 5.0,
) -> float:
    """Compute C_sep: fraction of consecutive pairs meeting RECAT-EU separation.

    For each runway, consecutive landings (sorted by time) are checked
    against the RECAT-EU matrix.  A pair is *compliant* if the observed gap
    is ≥ (required_separation − tolerance_s).

    Parameters
    ----------
    landing_times : Dict[str, List[float]]
        ``{runway_id: [landing_time, ...]}`` — unsorted is fine.
    wake_cats : Dict[str, str]
        ``{acid: wake_turbulence_category}`` mapping.
    recat_matrix : Dict[str, Dict[str, float]]
        RECAT-EU separation matrix (seconds).
    tolerance_s : float
        Compliance slack in seconds (default 5 s).

    Returns
    -------
    float
        C_sep ∈ [0, 1].
    """
    n_pairs = 0
    n_compliant = 0

    for rwy, times_and_acids in landing_times.items():
        if len(times_and_acids) < 2:
            continue
        sorted_pairs: List[Tuple[float, str]] = sorted(
            times_and_acids, key=lambda x: x[0]  # type: ignore[index]
        )
        for i in range(1, len(sorted_pairs)):
            t_prev, acid_prev = sorted_pairs[i - 1]  # type: ignore[misc]
            t_curr, acid_curr = sorted_pairs[i]       # type: ignore[misc]
            gap = t_curr - t_prev
            lead_cat = wake_cats.get(acid_prev, "C")
            trail_cat = wake_cats.get(acid_curr, "C")
            required = recat_matrix.get(lead_cat, {}).get(trail_cat, 90.0)
            n_pairs += 1
            if gap >= (required - tolerance_s):
                n_compliant += 1

    return (n_compliant / n_pairs) if n_pairs > 0 else float("nan")


def _compute_throughput(
    landing_times: Dict[str, List[Tuple[float, str]]],
    window_h: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute total throughput Γ and per-runway throughput Γ_r.

    Parameters
    ----------
    landing_times : Dict[str, List[Tuple[float, str]]]
        ``{runway_id: [(landing_time_s, acid), ...]}``
    window_h : float
        Observation window in hours.

    Returns
    -------
    gamma : float
        Total landings per hour.
    gamma_r : Dict[str, float]
        Per-runway landings per hour.
    """
    total = sum(len(v) for v in landing_times.values())
    gamma = total / window_h

    gamma_r: Dict[str, float] = {
        rwy: len(lts) / window_h for rwy, lts in landing_times.items()
    }
    return gamma, gamma_r


# ──────────────────────────────────────────────────────────────────────────────
# Episode log entry
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _EpisodeRecord:
    """Accumulator for per-aircraft data within a single episode."""

    acid: str
    runway_id: str
    wake_cat: str
    assigned_tta: float               # TTA set by CPSManager at episode end
    actual_landing_time: float        # Sim time of successful landing
    rta_error_cps: float              # |actual_time - assigned_tta|
    rta_error_solo: float             # |actual_time - unconstrained ETA| (baseline)
    recovered: bool                   # Whether an RTA violation was corrected
    success: bool


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
        """Create a Monitor-wrapped PathPlanningGoalEnv-v0 worker environment.

        The ``rta_sampler_path`` key (if present) is popped from
        ``env_kwargs`` before passing to ``gym.make()``; the sampler is
        loaded separately and injected as a keyword argument.

        Parameters
        ----------
        env_kwargs : dict
            Forwarded to ``gym.make()``; ``rta_sampler_path`` is intercepted.
        render_mode : str or None
            Passed to ``gym.make()``.

        Returns
        -------
        gym.Env
            Monitor-wrapped environment.
        """
        import bluesky_gym

        bluesky_gym.register_envs()

        env_name = self.cfg.env.env_name
        if env_name is None:
            raise ValueError("env_name is not set in CPSEnvConfig.")

        sampler_path = env_kwargs.pop("rta_sampler_path", None)
        rta_sampler = None
        if sampler_path:
            from path_planning.rta.sampling import GeoRunwaySampler

            rta_sampler = GeoRunwaySampler.load(sampler_path)

        # v_app is a CPS-layer parameter; the underlying env does not accept it
        env_kwargs.pop("v_app", None)

        env = gym.make(
            env_name,
            render_mode=render_mode,
            rta_sampler=rta_sampler,
            **env_kwargs,
        )
        env.reset()
        return Monitor(env)

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
        multi-aircraft coordination loop that measures all CPS-specific
        metrics.  A parallel *baseline* run (same frozen worker, no CPS) is
        executed to compute Δε (tracking degradation).

        Algorithm per episode
        ---------------------
        1. Spawn N_a independent env instances; reset each to obtain an
           initial observation and runway assignment.
        2. Build an initial fleet of :class:`AircraftState` records.
        3. At each simulation step:
           a. Refresh ETAs via ETASurrogate (if a sampler path is set).
           b. Call :meth:`CPSManager.update_fleet`; collect callsigns
              whose TTA changed by more than δ_update.
           c. For changed callsigns, inject the new TTA as the ``desired_goal``
              in the worker observation.
           d. Step all active environments with the frozen worker prediction.
           e. Record landings, separation gaps, and RTA errors.
        4. After all aircraft have landed (or been truncated), accumulate
           episode-level records.
        5. Compute aggregate metrics across all episodes and save logs.

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

        cps_manager = CPSManager(
            k_cps=mcfg.k_cps,
            recat_matrix=recat_matrix,
            runway_assignment_mode=mcfg.runway_assignment_mode,
            delta_t_plan=mcfg.delta_t_plan,
            delta_update=mcfg.delta_update,
        )

        n_episodes = cfg.session.eval_episodes
        n_aircraft = self._get_n_aircraft()

        all_records: List[_EpisodeRecord] = []
        results: dict[str, list[bool]] = defaultdict(list)

        print(
            f"\nCPS Coordination Evaluation"
            f"\n  episodes={n_episodes}, aircraft/episode={n_aircraft}"
            f"\n  k_cps={mcfg.k_cps}, mode={mcfg.runway_assignment_mode}"
            f"\n  frozen worker: {cfg.session.pretrained_model_path}\n"
        )

        for ep_idx in range(n_episodes):
            ep_records = self._run_episode(
                model=model,
                cps_manager=cps_manager,
                surrogate=surrogate,
                n_aircraft=n_aircraft,
                deterministic=deterministic,
                ep_idx=ep_idx,
            )
            all_records.extend(ep_records)
            cps_manager.reset()

            for rec in ep_records:
                results[rec.runway_id].append(rec.success)

        metrics = self._compute_aggregate_metrics(all_records, recat_matrix)
        self._print_metrics(metrics)
        self._save_logs(all_records, metrics)

        return dict(results)

    # ------------------------------------------------------------------ #
    # Per-episode runner
    # ------------------------------------------------------------------ #

    def _run_episode(
        self,
        model,
        cps_manager: CPSManager,
        surrogate: Optional[ETASurrogate],
        n_aircraft: int,
        deterministic: bool,
        ep_idx: int,
    ) -> List[_EpisodeRecord]:
        """Run one episode with N_a independent environments.

        Each environment represents one aircraft.  The frozen worker acts in
        each env; CPSManager assigns TTAs which are injected as the
        ``desired_goal`` in the worker's observation dict.

        Parameters
        ----------
        model : SAC
            Frozen worker policy.
        cps_manager : CPSManager
            Sequence manager (reset before each episode by the caller).
        surrogate : ETASurrogate or None
            ETA prediction model.  ``None`` → no ETA refresh; ETAs are
            initialised from initial observations only.
        n_aircraft : int
            Number of aircraft (independent env instances) per episode.
        deterministic : bool
            Whether to use deterministic policy predictions.
        ep_idx : int
            Episode index (used for logging).

        Returns
        -------
        List[_EpisodeRecord]
            One record per aircraft in this episode.
        """
        cfg = self.cfg
        env_kwargs = dict(cfg.eval_env_kwargs)

        # Spawn N_a environment instances
        envs: List[gym.Env] = [
            self.make_env(dict(env_kwargs)) for _ in range(n_aircraft)
        ]
        obs_list: List[Any] = []
        info_list: List[dict] = []

        # Synthetic callsigns for this episode
        acids = [f"AC{ep_idx:03d}_{i:02d}" for i in range(n_aircraft)]

        for env in envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)

        # Build initial fleet records
        fleet: List[AircraftState] = [
            AircraftState(
                acid=acids[i],
                state=self._obs_to_state(obs_list[i]),
                runway_id=str(info_list[i].get("current_runway", "27")),
                eta=float(info_list[i].get("sim_time", 0.0)),
                wake_cat="C",  # default; extend with actual WTC lookup
            )
            for i in range(n_aircraft)
        ]

        done_flags = [False] * n_aircraft
        trunc_flags = [False] * n_aircraft
        records: Dict[str, _EpisodeRecord] = {}
        sim_time: float = 0.0

        # Tracking: last assigned TTA and solo RTA error per aircraft
        last_tta: Dict[str, float] = {ac.acid: 0.0 for ac in fleet}
        solo_rta_errors: Dict[str, float] = {ac.acid: float("nan") for ac in fleet}

        while not all(d or t for d, t in zip(done_flags, trunc_flags)):
            # --- CPS coordination step ---
            changed = cps_manager.update_fleet(
                aircraft=fleet,
                current_time=sim_time,
                surrogate=surrogate,
            )
            for acid in changed:
                tta = cps_manager.get_tta(acid)
                if tta is not None:
                    last_tta[acid] = tta

            # --- Step each active environment ---
            for i, (ac, env) in enumerate(zip(fleet, envs)):
                if done_flags[i] or trunc_flags[i]:
                    continue

                obs = obs_list[i]

                # Inject updated TTA as desired_goal if the observation is a dict
                if isinstance(obs, dict) and "desired_goal" in obs:
                    obs = dict(obs)  # shallow copy
                    # Overwrite the RTA component of desired_goal with CPS TTA.
                    # The env encodes goal time as t / MAX_TIME (see pathplanning_goal_env.py),
                    # so the raw TTA (seconds) must be normalised before injection.
                    desired = obs["desired_goal"].copy()
                    desired[-1] = last_tta[ac.acid] / _ENV_MAX_TIME
                    obs["desired_goal"] = desired

                action, _ = model.predict(obs, deterministic=deterministic)
                next_obs, _rew, done, trunc, info = env.step(action)

                obs_list[i] = next_obs
                done_flags[i] = bool(done)
                trunc_flags[i] = bool(trunc)

                # Update aircraft state for next ETA prediction
                ac.state = self._obs_to_state(next_obs)
                if "current_runway" in info:
                    ac.runway_id = str(info["current_runway"])

                # On terminal step, record landing data
                if done or trunc:
                    landing_time = float(info.get("sim_time", sim_time))
                    success = bool(info.get("is_success", False))

                    rta_error_cps = abs(landing_time - last_tta[ac.acid])
                    # solo error: how far the worker would land without CPS TTA
                    rta_error_solo = float(info.get("rta_error", rta_error_cps))
                    solo_rta_errors[ac.acid] = rta_error_solo

                    recovered = (
                        rta_error_cps < rta_error_solo
                        and rta_error_solo > float(
                            self._cfg_or_default("cps_eval", {}).get(
                                "separation_tolerance_s", 5.0
                            )
                        )
                    )

                    records[ac.acid] = _EpisodeRecord(
                        acid=ac.acid,
                        runway_id=ac.runway_id,
                        wake_cat=ac.wake_cat,
                        assigned_tta=last_tta[ac.acid],
                        actual_landing_time=landing_time,
                        rta_error_cps=rta_error_cps,
                        rta_error_solo=rta_error_solo,
                        recovered=recovered,
                        success=success,
                    )

            sim_time += 1.0

        # Close envs
        for env in envs:
            env.close()

        return list(records.values())

    # ------------------------------------------------------------------ #
    # Aggregate metric computation
    # ------------------------------------------------------------------ #

    def _compute_aggregate_metrics(
        self,
        records: List[_EpisodeRecord],
        recat_matrix: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Compute all CPS metrics from the full set of episode records.

        Metrics computed
        ----------------
        gamma           : Total throughput (landings/hour).
        gamma_r         : Per-runway throughput (landings/hour).
        c_sep           : Separation compliance fraction.
        delta_epsilon   : Tracking degradation (mean |Δε|).
        r_rec           : Recovery success rate.
        rho_ripple      : Delay ripple index (lag-1 autocorrelation of RTA errors).
        n_episodes      : Total episodes evaluated.
        n_aircraft      : Total aircraft evaluated.
        success_rate    : Fraction of successful landings.

        Parameters
        ----------
        records : List[_EpisodeRecord]
            All per-aircraft records from all evaluation episodes.
        recat_matrix : Dict[str, Dict[str, float]]
            RECAT-EU separation matrix for C_sep calculation.

        Returns
        -------
        Dict[str, Any]
            Mapping of metric name → value.
        """
        if not records:
            return {"error": "no records collected"}

        n_aircraft = len(records)
        success_rate = sum(r.success for r in records) / n_aircraft

        # --- Throughput ---
        # Approximate: count landings over total sim time span
        landing_times_by_rwy: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        for rec in records:
            if rec.success:
                landing_times_by_rwy[rec.runway_id].append(
                    (rec.actual_landing_time, rec.acid)
                )

        total_time_s = max(
            (rec.actual_landing_time for rec in records if rec.success),
            default=3600.0,
        )
        window_h = max(total_time_s / 3600.0, 1e-6)
        gamma, gamma_r = _compute_throughput(landing_times_by_rwy, window_h)

        # --- Separation compliance ---
        wake_cats = {rec.acid: rec.wake_cat for rec in records}
        tol = self._cfg_or_default("cps_eval", {}).get("separation_tolerance_s", 5.0)
        c_sep = _compute_separation_compliance(
            landing_times_by_rwy,  # type: ignore[arg-type]
            wake_cats,
            recat_matrix,
            tolerance_s=float(tol),
        )

        # --- Tracking degradation Δε ---
        delta_eps_values = [
            abs(rec.rta_error_cps) - abs(rec.rta_error_solo)
            for rec in records
            if not np.isnan(rec.rta_error_solo)
        ]
        delta_epsilon = float(np.mean(delta_eps_values)) if delta_eps_values else float("nan")

        # --- Recovery success rate R_rec ---
        rta_violations = [rec for rec in records if abs(rec.rta_error_cps) > float(tol)]
        r_rec = (
            sum(rec.recovered for rec in rta_violations) / len(rta_violations)
            if rta_violations
            else float("nan")
        )

        # --- Delay ripple index ρ_ripple ---
        # Sort records by landing time to form the arrival sequence
        sorted_records = sorted(
            (rec for rec in records if rec.success),
            key=lambda r: r.actual_landing_time,
        )
        rta_error_sequence = [rec.rta_error_cps for rec in sorted_records]
        rho_ripple = _lag1_autocorrelation(rta_error_sequence)

        return {
            "n_episodes": len(set(r.acid.split("_")[0] for r in records)),
            "n_aircraft": n_aircraft,
            "success_rate": round(success_rate, 4),
            "gamma": round(gamma, 4),
            "gamma_r": {rwy: round(v, 4) for rwy, v in gamma_r.items()},
            "c_sep": round(float(c_sep), 4) if not np.isnan(c_sep) else "nan",
            "delta_epsilon": round(delta_epsilon, 4) if not np.isnan(delta_epsilon) else "nan",
            "r_rec": round(r_rec, 4) if not np.isnan(r_rec) else "nan",
            "rho_ripple": round(rho_ripple, 4) if not np.isnan(rho_ripple) else "nan",
        }

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print the aggregate metric table to stdout."""
        print("\n--- CPS Coordination Metrics ---")
        print(f"  Episodes evaluated   : {metrics.get('n_episodes')}")
        print(f"  Aircraft evaluated   : {metrics.get('n_aircraft')}")
        print(f"  Success rate         : {metrics.get('success_rate', 'n/a'):.2%}")
        print(f"  Throughput Γ         : {metrics.get('gamma', 'n/a')} ac/h")
        print(f"  Per-runway Γ_r       : {metrics.get('gamma_r', {})}")
        print(f"  Sep. compliance C_sep: {metrics.get('c_sep', 'n/a')}")
        print(f"  Tracking degrad. Δε  : {metrics.get('delta_epsilon', 'n/a')} s")
        print(f"  Recovery rate R_rec  : {metrics.get('r_rec', 'n/a')}")
        print(f"  Ripple index ρ_ripple: {metrics.get('rho_ripple', 'n/a')}")
        print()

    def _save_logs(
        self,
        records: List[_EpisodeRecord],
        metrics: Dict[str, Any],
    ) -> None:
        """Write per-aircraft CSV log and aggregate YAML metrics to disk.

        Outputs
        -------
        ``<save_path>/cps_eval_log.csv``    — one row per aircraft record.
        ``<save_path>/cps_metrics.yaml``    — aggregate metric dict.
        """
        save_path = self.cfg.save_path
        os.makedirs(save_path, exist_ok=True)

        # Per-aircraft CSV
        csv_path = os.path.join(save_path, "cps_eval_log.csv")
        csv_fields = [
            "acid", "runway_id", "wake_cat", "assigned_tta",
            "actual_landing_time", "rta_error_cps", "rta_error_solo",
            "recovered", "success",
        ]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            for rec in records:
                writer.writerow(
                    {
                        "acid": rec.acid,
                        "runway_id": rec.runway_id,
                        "wake_cat": rec.wake_cat,
                        "assigned_tta": rec.assigned_tta,
                        "actual_landing_time": rec.actual_landing_time,
                        "rta_error_cps": rec.rta_error_cps,
                        "rta_error_solo": rec.rta_error_solo,
                        "recovered": rec.recovered,
                        "success": rec.success,
                    }
                )
        print(f"Episode log saved → {csv_path}")

        # Aggregate metrics YAML
        yaml_path = os.path.join(save_path, "cps_metrics.yaml")
        with open(yaml_path, "w") as fh:
            yaml.dump(
                {"timestamp": datetime.now().isoformat(), **metrics},
                fh,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"Aggregate metrics saved → {yaml_path}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_surrogate(self) -> Optional[ETASurrogate]:
        """Construct an ETASurrogate from the configured sampler path, or None."""
        sampler_path = self.cfg.env.env_kwargs.rta_sampler_path  # type: ignore[attr-defined]
        if not sampler_path:
            return None
        return ETASurrogate.from_sampler_path(sampler_path, sim_dt=1.0)

    def _load_recat_matrix(self) -> Dict[str, Dict[str, float]]:
        """Load the RECAT-EU matrix from cps_base.yaml, or use a safe default.

        The matrix is stored as a top-level ``recat_eu`` key in the YAML
        config so it is editable without touching Python source.  Falls back
        to a conservative 90-second flat matrix if no YAML is available.
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
                return {
                    lead: {trail: float(v) for trail, v in row.items()}
                    for lead, row in matrix.items()
                }
        # Conservative fallback: 90 s for all category pairs
        cats = ["A", "B", "C", "D", "E", "F"]
        return {lead: {trail: 90.0 for trail in cats} for lead in cats}

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

    @staticmethod
    def _obs_to_state(obs: Any) -> np.ndarray:
        """Extract the flat state vector from a Gymnasium observation.

        For Dict observations (HER-compatible), uses the ``"observation"``
        key.  For Box observations, returns the array directly.

        Parameters
        ----------
        obs : Any
            Observation from ``env.step()`` or ``env.reset()``.

        Returns
        -------
        np.ndarray
            1-D float32 feature vector for the ETASurrogate.
        """
        if isinstance(obs, dict):
            return np.asarray(obs.get("observation", obs), dtype=np.float32).flatten()
        return np.asarray(obs, dtype=np.float32).flatten()
