"""
cps_coordination/coordination/eta_surrogate.py
-----------------------------------------------
Self-describing ETASurrogate: predicts T̂_i — remaining simulation steps
until an aircraft crosses its assigned Initial Approach Fix (IAF) — for use
by the Hierarchical Coordination Layer.

The surrogate is fully self-describing: it stores its feature recipe
(IAF anchors, surviving feature columns, target transform) during training
via :meth:`from_training` and applies them automatically at inference.
Callers need not specify a feature mode; the model knows what it requires.

Module-level utilities
----------------------
cartesian_to_polar   — (x, y) → (r, θ) in bearing convention
decompose_heading    — heading_deg → (sin ψ, cos ψ)

These are kept at module scope so ``CPSManager`` and ``train_surrogate``
can import them directly for real-time state assembly.

Transform functions
-------------------
Module-level named functions (not lambdas) for forward/inverse transforms
so they survive joblib serialisation.

Canonical feature set (13 columns, in order)
--------------------------------------------
 0  r                   √(x²+y²), from the raw Schiphol-centred (x, y) —
                        NOT IAF-relative (see ``cartesian_to_polar``)
 1  θ                   bearing angle, from the raw Schiphol-centred (x, y) —
                        NOT IAF-relative; only ``along_track_dist``/
                        ``cross_track_error``/``heading_error`` (rows 7-9)
                        use the IAF-relative ``dx = x - iaf_x``
 2  rwy_code            LabelEncoder integer
 3  elapsed_steps       episode step count
 4  sin_ψ               sin(heading_rad)
 5  cos_ψ               cos(heading_rad)
 6  r_sq                x²+y²
 7  along_track_dist    dot((dx,dy), (sin_ah, cos_ah))
 8  cross_track_error   dx·cos_ah − dy·sin_ah
 9  heading_error       (heading_rad − approach_heading + π) % 2π − π
10  delta_atd           ATD[t] − ATD[t − lag_steps]
11  cumabs_cte          rolling(|CTE|, window).sum()
12  heading_volatility  rolling(|Δheading|, window).mean()

At inference the state vector is always ``[x, y, elapsed_steps, heading_deg_bearing]``
(4-dimensional); lag features are passed as a separate optional argument.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Module-level coordinate utilities (public API — imported by CPSManager
# and train_surrogate)
# ---------------------------------------------------------------------------

def cartesian_to_polar(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert normalised Cartesian position to polar coordinates.

    The origin is the raw Schiphol-centred frame (the same ``(x, y)``
    ``_build_feature_matrix`` receives) — NOT the IAF. Both
    ``eta_surrogate.py::_build_feature_matrix`` and
    ``surrogate_data.py::build_feature_matrix`` (used by both
    ``select_surrogate_features.py`` and ``train_surrogate.py``) call this on raw
    Schiphol-centred coordinates identically at training and inference time,
    so there is no train/inference frame mismatch despite the name
    similarity to the IAF-relative features computed elsewhere in the same
    feature vector (``along_track_dist``/``cross_track_error``/
    ``heading_error``, which do use IAF-relative ``dx, dy``). Angles follow
    the bearing convention: θ = 0 points north, increases clockwise.

    Parameters
    ----------
    x, y : float or np.ndarray
        Normalised east–west / north–south coordinates.

    Returns
    -------
    r : np.ndarray
        Radial distance ``√(x²+y²)``.
    theta : np.ndarray
        Bearing angle in radians ``(π/2 − arctan2(y, x)) % 2π``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = (np.pi / 2.0 - np.arctan2(y, x)) % (2.0 * np.pi)
    return r, theta


def decompose_heading(
    psi_deg: Union[float, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose aircraft heading (degrees) into sine/cosine components.

    Periodic encoding prevents the 359° → 0° discontinuity from appearing
    as a large feature jump to tree-based regressors.

    Parameters
    ----------
    psi_deg : float or np.ndarray
        Aircraft heading(s) in degrees [0, 360).

    Returns
    -------
    sin_psi, cos_psi : np.ndarray
    """
    psi_deg = np.asarray(psi_deg, dtype=float)
    psi_rad = np.deg2rad(psi_deg)
    return np.sin(psi_rad), np.cos(psi_rad)


# ---------------------------------------------------------------------------
# Module-level transform functions (picklable by joblib — no lambdas)
# ---------------------------------------------------------------------------

def _fwd_identity(y: np.ndarray) -> np.ndarray:
    return y

def _inv_identity(y: np.ndarray) -> np.ndarray:
    return y

def _fwd_log1p(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)

def _inv_log1p(y: np.ndarray) -> np.ndarray:
    return np.expm1(y)

def _fwd_sqrt(y: np.ndarray) -> np.ndarray:
    return np.sqrt(y)

def _inv_sqrt(y: np.ndarray) -> np.ndarray:
    return np.clip(y, 0.0, None) ** 2


TRANSFORMS: Dict[str, Tuple[Callable, Callable]] = {
    "identity": (_fwd_identity, _inv_identity),
    "log1p":    (_fwd_log1p,    _inv_log1p),
    "sqrt":     (_fwd_sqrt,     _inv_sqrt),
}

# Ordered feature names for the full 13-column set.
ALL_FEATURE_NAMES: List[str] = [
    "r", "theta", "rwy_code", "elapsed_steps",
    "sin_psi", "cos_psi", "r_sq",
    "along_track_dist", "cross_track_error", "heading_error",
    "delta_atd", "cumabs_cte", "heading_volatility",
]

_N_FEATURES_FULL = len(ALL_FEATURE_NAMES)  # 13
_N_LAG = 3                                  # lag columns (indices 10–12)
_LAG_COL_START = 10


# ---------------------------------------------------------------------------
# ETASurrogate
# ---------------------------------------------------------------------------

class ETASurrogate:
    """Self-describing aircraft arrival-time surrogate.

    Wraps a ``sklearn.ensemble.ExtraTreesRegressor`` and stores its full
    feature recipe — IAF anchors, surviving column indices, and target
    transform — so that callers need not know which features the model
    was trained on.

    Construction
    ------------
    Do **not** call ``__init__`` directly to build a production surrogate.
    Use the :meth:`from_training` class method (called by ``train_surrogate.py``),
    or :meth:`load` / :meth:`from_sampler_path` to restore a serialised model.

    State vector format (inference)
    --------------------------------
    All predict methods expect a 4-element state vector per aircraft:

        ``[x, y, elapsed_steps, heading_deg_bearing]``

    Lag features ``[delta_atd, cumabs_cte, heading_volatility]`` are passed
    as a separate optional ``lag_features`` argument (shape ``(n, 3)`` for
    fleet methods).  When ``lag_features=None`` and ``_needs_lag=True``,
    the surrogate substitutes zeros (graceful warm-up degradation).

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest (default 200).
    n_jobs : int
        Parallelism for fitting and predicting (default -1 = all cores).
    random_state : int
        Random seed (default 42).
    sim_dt : float
        Simulation timestep in seconds (default 5.0).  Used to convert
        predicted step counts to absolute arrival-time offsets.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_jobs: int = -1,
        random_state: int = 42,
        sim_dt: float = 5.0,
    ) -> None:
        self.sim_dt = sim_dt
        self._model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._runway_encoder: LabelEncoder = LabelEncoder()
        self._runway_encoder_fitted: bool = False
        self._fitted: bool = False

        # Feature recipe — populated by from_training().
        self._iaf_ref: Dict[str, Tuple[float, float, float]] = {}
        self._feature_names: List[str] = list(ALL_FEATURE_NAMES)
        self._feature_col_indices: List[int] = list(range(_N_FEATURES_FULL))
        self._needs_lag: bool = False
        self._lag_steps: int = 5
        self._window: int = 10
        self._transform_name: str = "identity"
        self._inv_transform: Callable = _inv_identity

    # ------------------------------------------------------------------ #
    # Factory — primary construction path after training
    # ------------------------------------------------------------------ #

    @classmethod
    def from_training(
        cls,
        *,
        model: ExtraTreesRegressor,
        iaf_ref: Dict[str, Tuple[float, float, float]],
        runway_encoder: LabelEncoder,
        feature_names: List[str],
        col_indices: List[int],
        needs_lag: bool,
        lag_steps: int,
        window: int,
        sim_dt: float,
        transform_name: str,
        inv_transform: Callable,
        n_jobs: int = -1,
    ) -> "ETASurrogate":
        """Construct a fully-configured surrogate from training artefacts.

        Parameters
        ----------
        model : ExtraTreesRegressor
            Fitted regressor (in transformed target space).
        iaf_ref : dict
            ``{runway_id: (iaf_x, iaf_y, approach_heading_rad)}``.
        runway_encoder : LabelEncoder
            Fitted encoder for runway identifiers.
        feature_names : list[str]
            Names of surviving features after importance reduction,
            in column order.
        col_indices : list[int]
            Indices into the full 13-column vector that survive reduction.
        needs_lag : bool
            Whether any lag feature survived importance reduction.
        lag_steps, window : int
            Lag parameters used during training (passed to TrajectoryBuffer).
        sim_dt : float
            Simulation timestep in seconds.
        transform_name : str
            Name of the winning target transform (``"identity"``,
            ``"log1p"``, or ``"sqrt"``).
        inv_transform : Callable
            Module-level inverse transform function.
        n_jobs : int
            Parallelism for the underlying regressor predict calls.

        Returns
        -------
        ETASurrogate
        """
        s = cls(n_jobs=n_jobs, sim_dt=sim_dt)
        s._model = model
        s._fitted = True
        s._iaf_ref = iaf_ref
        s._runway_encoder = runway_encoder
        s._runway_encoder_fitted = True
        s._feature_names = list(feature_names)
        s._feature_col_indices = list(col_indices)
        s._needs_lag = needs_lag
        s._lag_steps = lag_steps
        s._window = window
        s._transform_name = transform_name
        s._inv_transform = inv_transform
        return s

    # ------------------------------------------------------------------ #
    # IAF reference fitting
    # ------------------------------------------------------------------ #

    def fit_iaf_reference(self, iaf_terminal_df) -> "ETASurrogate":
        """Compute and store IAF anchors from terminal-state rows.

        Terminal states are rows where ``steps_to_go == 0``, i.e., the
        aircraft has just crossed the IAF.  The method computes the
        mean position and circular-mean approach heading per runway.

        Parameters
        ----------
        iaf_terminal_df : pd.DataFrame
            Must contain columns ``[runway, x, y, heading]`` where
            ``heading`` is in bearing radians.

        Returns
        -------
        self
        """
        ref: Dict[str, Tuple[float, float, float]] = {}
        for rwy, grp in iaf_terminal_df.groupby("runway"):
            iaf_x = float(grp["x"].mean())
            iaf_y = float(grp["y"].mean())
            h = grp["heading"].to_numpy(dtype=float)
            # Circular mean — consistent across the ±π wrap boundary.
            ah = float(np.arctan2(np.sin(h).mean(), np.cos(h).mean()))
            ref[str(rwy)] = (iaf_x, iaf_y, ah)
        self._iaf_ref = ref
        return self

    # ------------------------------------------------------------------ #
    # Runway encoder (kept for direct use in train_surrogate)
    # ------------------------------------------------------------------ #

    def fit_runway_encoder(self, runway_labels: List[str]) -> "ETASurrogate":
        """Fit the internal LabelEncoder on the full set of runway identifiers.

        Parameters
        ----------
        runway_labels : list[str]
            All runway identifiers present in the training data.

        Returns
        -------
        self
        """
        self._runway_encoder.fit(runway_labels)
        self._runway_encoder_fitted = True
        return self

    def encode_runway(self, runway_id: str) -> int:
        """Encode a single runway identifier to its integer code."""
        if not self._runway_encoder_fitted:
            raise RuntimeError("Runway encoder not fitted. Call fit_runway_encoder() first.")
        return int(np.asarray(self._runway_encoder.transform([runway_id]))[0])

    # ------------------------------------------------------------------ #
    # Feature matrix assembly (internal)
    # ------------------------------------------------------------------ #

    def _build_feature_matrix(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        heading_deg_arr: np.ndarray,
        rwy_ids: List[str],
        elapsed_arr: np.ndarray,
        lag_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Assemble the ``(n, n_selected)`` feature matrix from raw inputs.

        Always assembles the full 13-column vector first, then selects
        ``_feature_col_indices`` to match what the model was trained on.

        Parameters
        ----------
        x_arr, y_arr : np.ndarray, shape (n,)
        heading_deg_arr : np.ndarray, shape (n,)
            Aircraft headings in bearing degrees.
        rwy_ids : list[str], length n
        elapsed_arr : np.ndarray, shape (n,)
            Episode step counts.
        lag_features : np.ndarray, shape (n, 3), optional
            ``[delta_atd, cumabs_cte, heading_volatility]`` per aircraft.

        Returns
        -------
        np.ndarray, shape (n, n_selected_features)
        """
        n = len(x_arr)

        # --- Base coordinates ---
        r_arr, theta_arr = cartesian_to_polar(x_arr, y_arr)
        rwy_codes = np.asarray(self._runway_encoder.transform(rwy_ids), dtype=float)
        sin_psi, cos_psi = decompose_heading(heading_deg_arr)
        r_sq = x_arr ** 2 + y_arr ** 2

        # --- IAF-relative geometric features ---
        iaf_data = [self._iaf_ref.get(rwy, (0.0, 0.0, 0.0)) for rwy in rwy_ids]
        iaf_x_arr = np.array([d[0] for d in iaf_data], dtype=float)
        iaf_y_arr = np.array([d[1] for d in iaf_data], dtype=float)
        iaf_ah_arr = np.array([d[2] for d in iaf_data], dtype=float)

        dx = x_arr - iaf_x_arr
        dy = y_arr - iaf_y_arr
        sin_ah = np.sin(iaf_ah_arr)
        cos_ah = np.cos(iaf_ah_arr)
        atd = dx * sin_ah + dy * cos_ah
        cte = dx * cos_ah - dy * sin_ah
        heading_rad = np.deg2rad(heading_deg_arr)
        h_err = (heading_rad - iaf_ah_arr + np.pi) % (2.0 * np.pi) - np.pi

        # --- Lag features ---
        if lag_features is None or not self._needs_lag:
            lag = np.zeros((n, _N_LAG), dtype=float)
        else:
            lag = np.asarray(lag_features, dtype=float)

        # --- Full 13-column matrix in canonical order ---
        X_full = np.column_stack([
            r_arr, theta_arr, rwy_codes, elapsed_arr,
            sin_psi, cos_psi, r_sq,
            atd, cte, h_err,
            lag[:, 0], lag[:, 1], lag[:, 2],
        ])  # (n, 13)

        return X_full[:, self._feature_col_indices]

    # ------------------------------------------------------------------ #
    # sklearn-style fit / predict (used by train_surrogate directly)
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ETASurrogate":
        """Fit the underlying ExtraTreesRegressor on a pre-assembled matrix.

        ``X`` must already be in the reduced column space (apply
        ``_feature_col_indices`` before calling).  ``y`` must be in
        **transformed** target space if a non-identity transform is used.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_selected_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Raw model prediction on a pre-assembled reduced feature matrix.

        Returns predictions in **transformed** space (before ``_inv_transform``).
        Use the high-level ``predict_eta*`` methods for production inference.
        """
        if not self._fitted:
            raise RuntimeError("ETASurrogate is not fitted.")
        return self._model.predict(X)

    # ------------------------------------------------------------------ #
    # High-level prediction API (called by CPSManager)
    # ------------------------------------------------------------------ #

    def predict_steps_to_iaf(
        self,
        state: np.ndarray,
        runway_id: str,
        lag_features: Optional[np.ndarray] = None,
    ) -> float:
        """Predict T̂_i: remaining simulation steps until IAF crossing.

        Parameters
        ----------
        state : np.ndarray, shape (4,)
            ``[x, y, elapsed_steps, heading_deg_bearing]``.
        runway_id : str
        lag_features : np.ndarray, shape (3,), optional

        Returns
        -------
        float
            Predicted step count, clamped to ≥ 0.
        """
        if not self._fitted:
            raise RuntimeError("ETASurrogate is not fitted.")
        state = np.asarray(state, dtype=float).ravel()
        x, y, elapsed, heading = state[0], state[1], state[2], state[3]
        lag = lag_features.reshape(1, -1) if lag_features is not None else None
        X = self._build_feature_matrix(
            np.array([x]), np.array([y]), np.array([heading]),
            [runway_id], np.array([elapsed]), lag,
        )
        raw = self._model.predict(X)
        steps = float(self._inv_transform(raw)[0])
        return max(0.0, steps)

    def predict_eta(
        self,
        state: np.ndarray,
        runway_id: str,
        current_sim_time: float,
        lag_features: Optional[np.ndarray] = None,
    ) -> float:
        """Return absolute ETA at the IAF.

        Parameters
        ----------
        state : np.ndarray, shape (4,)
            ``[x, y, elapsed_steps, heading_deg_bearing]``.
        runway_id : str
        current_sim_time : float
            Current simulation clock in seconds.
        lag_features : np.ndarray, shape (3,), optional

        Returns
        -------
        float
            ``current_sim_time + T̂_i × sim_dt``
        """
        steps = self.predict_steps_to_iaf(state, runway_id, lag_features)
        return current_sim_time + steps * self.sim_dt

    def predict_eta_fleet(
        self,
        states: np.ndarray,
        runway_ids: List[str],
        current_sim_time: float,
        lag_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorised ETA for N aircraft, one runway each.

        This is the 1-to-1 predict method used by ``_refresh_etas`` in
        ``CPSManager``.  All ``n`` feature vectors are assembled in one
        numpy pass; the model is called once.

        Parameters
        ----------
        states : np.ndarray, shape (n, 4)
            Each row: ``[x, y, elapsed_steps, heading_deg_bearing]``.
        runway_ids : list[str], length n
        current_sim_time : float
        lag_features : np.ndarray, shape (n, 3), optional

        Returns
        -------
        np.ndarray, shape (n,)
            Absolute estimated arrival times (seconds).
        """
        if not self._fitted:
            raise RuntimeError("ETASurrogate is not fitted.")
        n = states.shape[0]
        if len(runway_ids) != n:
            raise ValueError(
                f"states has {n} rows but runway_ids has {len(runway_ids)} entries."
            )
        X = self._build_feature_matrix(
            states[:, 0], states[:, 1], states[:, 3],
            runway_ids, states[:, 2], lag_features,
        )
        raw = self._model.predict(X)
        steps = np.maximum(self._inv_transform(raw), 0.0)
        return current_sim_time + steps * self.sim_dt

    def predict_eta_fleet_all_runways(
        self,
        states: np.ndarray,
        runway_ids: List[str],
        current_sim_time: float,
        lag_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorised ETA for every (aircraft, runway) pair.

        Used by ``_assign_runways_dynamic`` in ``CPSManager``.  Builds a
        single ``(n×r, n_selected)`` feature matrix and calls the model once.

        Parameters
        ----------
        states : np.ndarray, shape (n, 4)
        runway_ids : list[str], length r
            Candidate runway identifiers.
        current_sim_time : float
        lag_features : np.ndarray, shape (n, 3), optional

        Returns
        -------
        np.ndarray, shape (n, r)
            ``result[i, j]`` is the ETA for aircraft ``i`` on ``runway_ids[j]``.
        """
        if not self._fitted:
            raise RuntimeError("ETASurrogate is not fitted.")

        n = states.shape[0]
        r = len(runway_ids)

        # Each aircraft row repeated r times → (n*r, 4)
        states_tiled = np.repeat(states, r, axis=0)
        # [rwy0, rwy1, ..., rwy_{r-1}, rwy0, rwy1, ...]
        runway_ids_tiled = runway_ids * n

        if lag_features is not None:
            lag_tiled: Optional[np.ndarray] = np.repeat(lag_features, r, axis=0)
        else:
            lag_tiled = None

        X = self._build_feature_matrix(
            states_tiled[:, 0], states_tiled[:, 1], states_tiled[:, 3],
            runway_ids_tiled, states_tiled[:, 2], lag_tiled,
        )
        raw = self._model.predict(X)
        steps = np.maximum(self._inv_transform(raw), 0.0).reshape(n, r)
        return current_sim_time + steps * self.sim_dt

    def predict_batch(
        self,
        states: np.ndarray,
        runway_ids: List[str],
        current_sim_time: float,
        lag_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Alias for :meth:`predict_eta_fleet` (backwards compatibility)."""
        return self.predict_eta_fleet(states, runway_ids, current_sim_time, lag_features)

    def predict_all_runways(
        self,
        state: np.ndarray,
        runway_ids: List[str],
        current_sim_time: float,
        lag_features: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Predict ETA for one aircraft across all candidate runways.

        Parameters
        ----------
        state : np.ndarray, shape (4,)
        runway_ids : list[str]
        current_sim_time : float
        lag_features : np.ndarray, shape (3,), optional

        Returns
        -------
        dict[str, float]
            ``{runway_id: absolute_eta}``
        """
        return {
            rwy: self.predict_eta(state, rwy, current_sim_time, lag_features)
            for rwy in runway_ids
        }

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def state_layout(self) -> str:
        """Human-readable description of the expected state vector format."""
        return "[x, y, elapsed_steps, heading_deg_bearing]"

    @property
    def n_features(self) -> int:
        """Number of features after importance reduction."""
        return len(self._feature_col_indices)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted surrogate to disk with joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ETASurrogate":
        """Load a serialised ETASurrogate from disk.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        ETASurrogate
        """
        return joblib.load(Path(path))

    @classmethod
    def from_sampler_path(
        cls,
        path: Union[str, Path],
        sim_dt: float = 5.0,
    ) -> "ETASurrogate":
        """Load a serialised ETASurrogate and optionally override ``sim_dt``.

        Parameters
        ----------
        path : str or Path
        sim_dt : float
            Simulation timestep in seconds.  Overrides the stored value.

        Returns
        -------
        ETASurrogate
        """
        surrogate = cls.load(path)
        surrogate.sim_dt = sim_dt
        return surrogate

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        lag_info = f", lag={self._lag_steps}/{self._window}" if self._needs_lag else ""
        return (
            f"ETASurrogate({status}, "
            f"n_features={self.n_features}, "
            f"transform={self._transform_name!r}{lag_info}, "
            f"sim_dt={self.sim_dt}s, "
            f"iaf_runways={sorted(self._iaf_ref)})"
        )
