"""
base_critic.py
--------------
Reusable toolbox for critic-landscape analysis in GCRL.

Design principles
-----------------
- BaseCriticExperiment knows nothing about DecomposedSAC, reward
  decomposition, or any specific critic architecture. Head selection
  is delegated entirely to the CriticProbe.agg callable.
- ProbeContext decouples *where* to probe from *what* to probe, so
  subclasses can probe at any state, not just episode reset.
- ProbeResult carries pre-computed summary stats so subclasses never
  have to re-derive them.
- run_sweep trains once per parameter set, probes, and saves to CSV.
"""

from __future__ import annotations

import csv
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from stable_baselines3 import SAC

from .base import BasePathPlanningExperiment


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ActionEncoder = Callable[[float], np.ndarray]
ObsAdapter    = Callable[[Dict[str, torch.Tensor], float], Dict[str, torch.Tensor]]

QAggregator = Callable[
    [torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor], 
    Union[float, Dict[str, float]]
]

def _default_agg(q1: torch.Tensor, q2: torch.Tensor, obs: Any, action: Any) -> float:
    """Conservative min of the clipped double-Q pair."""
    return float(torch.min(q1, q2).item())


# ---------------------------------------------------------------------------
# ProbeContext
# ---------------------------------------------------------------------------

@dataclass
class ProbeContext:
    """
    The state from which the critic is probed.

    Separating *where* to probe from *what* to probe lets subclasses
    override build_context() to use seeded resets, mid-episode states, or
    hand-crafted tensors — not just the episode-start default.

    Attributes
    ----------
    obs_t    : observation tensors, already on the correct device.
    model    : trained SAC (or any subclass) to query.
    device   : torch device.
    metadata : free-form dict stored verbatim in every ProbeResult produced
               from this context (e.g. {"goal_reward": 50.0}).
    """
    obs_t:    Dict[str, torch.Tensor]
    model:    SAC
    device:   torch.device
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CriticProbe
# ---------------------------------------------------------------------------

@dataclass
class CriticProbe:
    """
    Defines one axis to sweep across the Q-landscape.

    Parameters
    ----------
    name         : human-readable label (used as a column prefix in CSV).
    sweep_values : 1-D array of values to iterate over.
    encoder      : maps one sweep value to an action array.
    obs_adapter  : optionally mutates the observation to stay coherent with
                   the sweep value (e.g. sync cos/sin heading channels).
    agg          : receives (q1, q2) from model.critic and returns a scalar.
                   Replace this closure to probe any head or composite value
                   without touching BaseCriticExperiment.

    Probing a decomposed head — build the closure in build_probes()
    ---------------------------------------------------------------
    >>> def build_probes(self, model):
    ...     critic = model.critic          # captured by closure
    ...     def aug_agg(q1, q2):
    ...         # q1/q2 are the standard forward() outputs; ignore them and
    ...         # call the decomposed forward instead.
    ...         (a0, a1), _ = critic.forward_decomposed(obs, action)
    ...         return float(torch.min(a0, a1).item())
    ...     return [CriticProbe("heading_aug", sweep, encoder, agg=aug_agg)]
    Note: obs and action must also be captured or passed in — see
    DecomposedCriticExperiment in the examples for the full pattern.
    """
    name:         str
    sweep_values: np.ndarray
    encoder:      ActionEncoder
    obs_adapter:  Optional[ObsAdapter] = None
    agg:          QAggregator          = field(default_factory=lambda: _default_agg)


# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------
@dataclass
class ProbeResult:
    """Output of one sweep along one CriticProbe axis."""

    probe_name:   str
    sweep_values: np.ndarray
    raw_results:  List[Union[float, Dict[str, float]]]
    params:       Dict[str, Any] = field(default_factory=dict)
    context_meta: Dict[str, Any] = field(default_factory=dict)

    # Stores computed stats (e.g. "q_mean", "lambda_max", etc.)
    summary: Dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_results:
            return

        # 1. Normalize: ensure every entry is a dictionary.
        # Scalars are mapped to a default "q" key.
        normalized = [
            r if isinstance(r, dict) else {"q": r} 
            for r in self.raw_results
        ]

        # 2. Extract all unique keys present in the sweep
        all_keys = {k for d in normalized for k in d.keys()}

        # 3. Compute stats for every key found
        for k in all_keys:
            # Create an array for this specific metric across the sweep
            # missing keys are handled as NaNs to prevent alignment shifts
            vals = np.array([d.get(k, np.nan) for d in normalized])
            
            # Formatting logic: 
            # 'q' uses legacy "mean_q" style; others use "key_mean" style.
            prefix = "" if k == "q" else f"{k}_"
            suffix = "_q" if k == "q" else ""
            
            self._compute_stats(vals, prefix=prefix, suffix=suffix)

    def _compute_stats(self, vals: np.ndarray, prefix: str = "", suffix: str = "") -> None:
        """Standard statistical breakdown for a sequence of values."""
        # Use nan-safe versions in case a key was missing in some steps
        self.summary[f"{prefix}mean{suffix}"]   = float(np.nanmean(vals))
        self.summary[f"{prefix}std{suffix}"]    = float(np.nanstd(vals))
        self.summary[f"{prefix}min{suffix}"]    = float(np.nanmin(vals))
        self.summary[f"{prefix}max{suffix}"]    = float(np.nanmax(vals))
        
        if not np.all(np.isnan(vals)):
            idx = int(np.nanargmax(vals))
            self.summary[f"{prefix}argmax{suffix}"] = float(self.sweep_values[idx])

    def as_flat_dict(self) -> Dict[str, Any]:
        """Flattens metadata and computed statistics for CSV logging."""
        return {
            "probe":   self.probe_name,
            **self.summary,
            **self.params,
            **self.context_meta,
        }

# ---------------------------------------------------------------------------
# ProbeReport
# ---------------------------------------------------------------------------

@dataclass
class ProbeReport:
    """
    All ProbeResults from one probe_critic() call, plus shared metadata.
    Passed to on_probe_complete() for a typed data contract.
    """
    results:      List[ProbeResult]
    context_meta: Dict[str, Any] = field(default_factory=dict)

    def by_probe(self, name: str) -> Optional[ProbeResult]:
        return next((r for r in self.results if r.probe_name == name), None)

    def filter(self, prefix: str) -> List[ProbeResult]:
        return [r for r in self.results if r.probe_name.startswith(prefix)]


# ---------------------------------------------------------------------------
# BaseCriticExperiment
# ---------------------------------------------------------------------------

class BaseCriticExperiment(BasePathPlanningExperiment):
    """
    Abstract base for experiments that probe the critic's Q-landscape.

    Subclass contract
    -----------------
    Required:
        build_probes(model) -> List[CriticProbe]
            Receives the trained model so subclasses can inspect its type
            and build architecture-specific agg closures — without this
            base class importing any specific SAC variant.

    Optional:
        build_context(model) -> ProbeContext
            Which observation to probe from.  Default: env.reset().

        apply_env_patches(env, **patch_kwargs) -> env
            Mutate env constants after make_env() (reward weights, etc.).

        _extract_patch_kwargs(env_kwargs) -> dict
            Declare which env_kwargs keys go to apply_env_patches, not gym.make().

        on_probe_complete(report: ProbeReport)
            Post-process results (dominance analysis, plotting, …).
    """

    # ------------------------------------------------------------------ #
    # Env helpers                                                          #
    # ------------------------------------------------------------------ #

    def apply_env_patches(self, env, **patch_kwargs: Any):
        return env

    def make_env(self, env_kwargs=None, render_mode=None):
        patch_kwargs = self._extract_patch_kwargs(env_kwargs or {})
        env = super().make_env(env_kwargs or {}, render_mode)
        return self.apply_env_patches(env, **patch_kwargs)

    def _extract_patch_kwargs(self, env_kwargs: dict) -> dict:
        return {}

    # ------------------------------------------------------------------ #
    # Abstract                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def build_probes(self, model) -> List[CriticProbe]:
        """
        Return the probes to sweep.  Receives `model` so subclasses can
        inspect its type and build architecture-specific agg closures here,
        keeping all critic-variant knowledge out of BaseCriticExperiment.
        """
        ...

    # ------------------------------------------------------------------ #
    # Context                                                              #
    # ------------------------------------------------------------------ #

    def build_context(self, model: SAC, N: int = 1) -> ProbeContext:
        """
        Default: reset a fresh env and use the initial observation.
        Override to probe at a seeded, mid-episode, or hand-crafted state.
        """
        env = self.make_env()
        obs, _ = env.reset()

        observations = []
    
        def to_tensor(o):
            return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in o.items()}

        observations.append(to_tensor(obs))
        for _ in range(N-1):  # Collect N-1 diverse states from a trajectory
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(action)
            # Store a copy of the observation dict
            if term or trunc:
                obs, _ = env.reset()

            observations.append(to_tensor(obs))

        env.close()

        device = model.device
        batch_obs = {
            k: torch.stack([o[k] for o in observations]).to(device)
            for k in observations[0].keys()
        }

        return ProbeContext(obs_t=batch_obs, model=model, device=device)

    # ------------------------------------------------------------------ #
    # Core probing                                                         #
    # ------------------------------------------------------------------ #

    def probe_critic(
        self,
        model: SAC,
        probes: List[CriticProbe],
        context: Optional[ProbeContext] = None,
    ) -> ProbeReport:

        if context is None:
            context = self.build_context(model)

        results = []
        for probe in probes:
            # _sweep_probe now returns a list of whatever the agg returned
            raw_data = self._sweep_probe(context, probe)
            results.append(ProbeResult(
                probe_name=probe.name,
                sweep_values=probe.sweep_values,
                raw_results=raw_data,
                context_meta=context.metadata,
            ))

        return ProbeReport(results=results, context_meta=context.metadata)

    @staticmethod
    def _sweep_probe(context: ProbeContext, probe: CriticProbe) -> List[Any]:
        model = context.model
        obs_t = context.obs_t  # Now a batch of [N, Obs_Dim]
        batch_size = obs_t[next(iter(obs_t))].shape[0]

        results = []
        with torch.no_grad():
            for val in probe.sweep_values:
                # 1. Adapt the batch of observations
                obs = probe.obs_adapter(obs_t, val) if probe.obs_adapter else obs_t
                
                # 2. Create a matching batch of actions
                action_single = probe.encoder(val)
                action_batch = torch.as_tensor(
                    np.tile(action_single, (batch_size, 1)), 
                    device=context.device, 
                    dtype=torch.float32
                )
                
                # 3. Get Q-values for the whole batch of states at this sweep point
                q1, q2 = model.critic(obs, action_batch)
                
                # 4. Aggregator now receives tensors of shape [Batch, 1]
                # It should return the mean across the batch for this specific point
                results.append(probe.agg(q1, q2, obs, action_batch)) 
        return results

    # ------------------------------------------------------------------ #
    # Parameter sweep                                                      #
    # ------------------------------------------------------------------ #

    #TODO: change this so we use run
    def run_sweep(
        self,
        parameter_sets: List[Dict[str, Any]],
        output_path: Optional[str | Path] = None,
        extra_columns: Optional[Sequence[str]] = None,
    ) -> List[ProbeResult]:
        """
        Train once per parameter set, probe the critic, save to CSV.

        Parameters
        ----------
        parameter_sets : each dict updates cfg.env.env_kwargs before training.
        output_path    : destination CSV.
        extra_columns  : forwarded to _save_csv for stable column ordering.
        """
        all_results: List[ProbeResult] = []
        all_rows:    List[Dict[str, Any]] = []

        for params in parameter_sets:
            print(f"--- Sweep: {params} ---")
            self.cfg.env.env_kwargs.__dict__.update(params)
            self.train()

            ctx = self.build_context(self._model)
            ctx.metadata.update(params)

            probes = self.build_probes(self._model)
            report = self.probe_critic(self._model, probes, context=ctx)

            for res in report.results:
                res.params = dict(params)
                all_results.append(res)
                all_rows.append(res.as_flat_dict())

        if output_path is not None:
            self._save_csv(all_rows, output_path, extra_columns)
        return all_results

    # ------------------------------------------------------------------ #
    # CSV                                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _save_csv(
        results: List[Dict[str, Any]],
        output_path: str | Path,
        extra_columns: Optional[Sequence[str]] = None,
    ) -> None:
        if not results:
            print("No results to save.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if extra_columns is not None:
            param_cols = list(extra_columns)
        else:
            first_row_keys = list(results[0].keys())
            probe_stat_prefixes = tuple(
                k.split("_q_")[0] + "_"
                for k in first_row_keys
                if "_q_" in k
            )
            param_cols = [
                k for k in first_row_keys
                if not any(k.startswith(p) for p in probe_stat_prefixes)
            ]

        all_keys = list(dict.fromkeys(
            param_cols + [k for k in results[0] if k not in param_cols]
        ))

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved {len(results)} rows → {output_path}")

    # ------------------------------------------------------------------ #
    # Hook                                                                 #
    # ------------------------------------------------------------------ #

    def on_probe_complete(self, report: ProbeReport) -> None:
        """Override to post-process results (plotting, dominance, logging)."""
        pass

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        import bluesky_gym
        bluesky_gym.register_envs()

        cfg       = self.cfg
        algo_name = cfg.model.algorithm.__name__ if cfg.model.algorithm else "Unspecified"
        print(f"▶️  {cfg.run_id} | env={cfg.env.env_name} | algo={algo_name}")

        if cfg.session.do_train:
            self.train()

        if cfg.session.do_evaluate:
            model = getattr(self, "_model", None) or cfg.model.algorithm.load(
                f"{cfg.save_path}/final_model"
            )
            self.evaluate(model)

        model = getattr(self, "_model", None)
        if model is not None:
            probes = self.build_probes(model)
            report = self.probe_critic(model, probes)
            self.on_probe_complete(report)