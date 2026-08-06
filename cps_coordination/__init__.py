"""
cps_coordination
-----------------
Hierarchical CPS Coordination Layer for 4D RTA path planning in BlueSky-Gym.

This package implements the Constrained Position Shifting (CPS) coordination
algorithm described in the project methodology.  It sits *above* the frozen
path-planning worker agents and assigns Target Times of Arrival (TTAs) that
the workers then optimise against.

Public API
----------
  ETASurrogate             — Self-describing temporal surrogate that wraps a
                             scikit-learn ExtraTreesRegressor to predict
                             T̂_i (remaining simulation steps to IAF).
                             Feature set is determined at training time and
                             stored inside the surrogate (no mode selection).
  CPSManager               — k-CPS sequence manager and TTA assigner.
  TrajectoryBuffer         — Per-aircraft rolling state history for lag
                             feature computation at inference time.
  CPSCoordinationExperiment — BaseExperiment subclass that loads a frozen
                              spatial-temporal policy and runs the CPS
                              coordination evaluation loop.

Utility functions (re-exported for convenience)
-----------------------------------------------
  cartesian_to_polar  — (x, y) → (r, θ) bearing-convention polar coords
  decompose_heading   — heading_deg → (sin ψ, cos ψ) periodic components

Quick start
-----------
  # Train a surrogate (run train_surrogate.py first):
  #   python cps_coordination/scripts/train_surrogate.py data.parquet

  from cps_coordination import ETASurrogate
  import numpy as np

  surrogate = ETASurrogate.load("cps_coordination/models/eta_surrogate.pkl")
  state = np.array([0.3, -0.4, 120.0, 45.0])  # [x, y, elapsed_steps, heading_deg]
  eta = surrogate.predict_eta(state, "27L", current_sim_time=600.0)
"""
from cps_coordination.experiments.config import (
    CPSModelConfig,
    CPSEnvKwargsConfig,
    CPSEnvConfig,
)
from cps_coordination.coordination.eta_surrogate import ETASurrogate, cartesian_to_polar, decompose_heading
from cps_coordination.coordination.cps_manager import AircraftState, CPSManager
from cps_coordination.coordination.trajectory_buffer import TrajectoryBuffer
from cps_coordination.experiments.coordination_baseline import CPSCoordinationExperiment

__all__ = [
    "CPSModelConfig",
    "CPSEnvKwargsConfig",
    "CPSEnvConfig",
    "ETASurrogate",
    "cartesian_to_polar",
    "decompose_heading",
    "AircraftState",
    "CPSManager",
    "TrajectoryBuffer",
    "CPSCoordinationExperiment",
]