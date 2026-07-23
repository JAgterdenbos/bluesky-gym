"""
cps_coordination/coordination/trajectory_buffer.py
---------------------------------------------------
TrajectoryBuffer: per-aircraft rolling state history for lag feature
computation at inference time.

Stores raw (x, y, heading_rad) tuples in a fixed-size deque per aircraft
callsign.  Lag features (delta_atd, cumabs_cte, heading_volatility) are
computed on demand given the aircraft's current IAF reference coordinates,
mirroring the ``add_lag_features`` logic used during training.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np


class TrajectoryBuffer:
    """Per-aircraft rolling trajectory history for lag feature computation.

    Each aircraft's history is stored as a ``collections.deque`` of
    ``(x, y, heading_rad)`` tuples in chronological order (oldest first).
    Lag features are computed lazily at query time so that the IAF
    reference can change (e.g. dynamic runway re-assignment) without
    requiring a buffer flush.

    Parameters
    ----------
    maxlen : int
        Maximum states retained per aircraft.  Older entries are evicted
        automatically when the buffer is full.  Must be ≥ ``window + 1``
        to guarantee at least one heading difference.
    lag_steps : int
        Step lag for ``delta_atd``: ``ATD[t] − ATD[t − lag_steps]``.
        Back-filled with the oldest available entry when fewer than
        ``lag_steps + 1`` states exist.
    window : int
        Rolling window size for ``cumabs_cte`` and
        ``heading_volatility``.
    """

    _N_LAG = 3  # [delta_atd, cumabs_cte, heading_volatility]

    def __init__(
        self,
        maxlen: int = 50,
        lag_steps: int = 5,
        window: int = 10,
    ) -> None:
        if maxlen < 2:
            raise ValueError("maxlen must be ≥ 2.")
        self.maxlen = maxlen
        self.lag_steps = lag_steps
        self.window = window
        # acid → deque[(x, y, heading_rad)]
        self._history: Dict[str, deque] = {}

    # ------------------------------------------------------------------ #
    # State ingestion
    # ------------------------------------------------------------------ #

    def push(self, acid: str, x: float, y: float, heading_rad: float) -> None:
        """Append the current aircraft state to the rolling history.

        Parameters
        ----------
        acid : str
            Aircraft callsign (unique identifier).
        x, y : float
            Normalised Cartesian position (same coordinate space as the
            env observation and the surrogate's training data).
        heading_rad : float
            Aircraft heading in bearing radians (0 = north, clockwise).
        """
        if acid not in self._history:
            self._history[acid] = deque(maxlen=self.maxlen)
        self._history[acid].append((float(x), float(y), float(heading_rad)))

    # ------------------------------------------------------------------ #
    # Lag feature computation
    # ------------------------------------------------------------------ #

    def get_lag_features(
        self,
        acid: str,
        iaf_x: float,
        iaf_y: float,
        approach_heading_rad: float,
    ) -> np.ndarray:
        """Compute the three lag features for one aircraft.

        Parameters
        ----------
        acid : str
        iaf_x, iaf_y : float
            IAF anchor coordinates for the aircraft's current runway.
        approach_heading_rad : float
            Approach track direction at the IAF in bearing radians.

        Returns
        -------
        np.ndarray, shape (3,)
            ``[delta_atd, cumabs_cte, heading_volatility]``.
            Returns zeros when no history is available.
        """
        buf = self._history.get(acid)
        if not buf:
            return np.zeros(self._N_LAG, dtype=float)

        entries = list(buf)  # oldest → newest
        n = len(entries)

        xs = np.array([e[0] for e in entries], dtype=float)
        ys = np.array([e[1] for e in entries], dtype=float)
        hs = np.array([e[2] for e in entries], dtype=float)  # bearing radians

        sin_ah = np.sin(approach_heading_rad)
        cos_ah = np.cos(approach_heading_rad)

        dx = xs - iaf_x
        dy = ys - iaf_y

        atds = dx * sin_ah + dy * cos_ah
        ctes = dx * cos_ah - dy * sin_ah

        # delta_atd: ATD[current] − ATD[current − lag_steps], bfill
        lag_idx = max(0, n - 1 - self.lag_steps)
        delta_atd = float(atds[-1] - atds[lag_idx])

        # cumabs_cte: rolling sum of |CTE| over last `window` entries
        win_ctes = ctes[max(0, n - self.window):]
        cumabs_cte = float(np.abs(win_ctes).sum())

        # heading_volatility: mean |wrapped Δheading| over last window steps
        win_hs = hs[max(0, n - self.window - 1):]
        if len(win_hs) >= 2:
            diffs = np.diff(win_hs)
            diffs = (diffs + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [−π, π]
            heading_vol = float(np.abs(diffs).mean())
        else:
            heading_vol = 0.0

        return np.array([delta_atd, cumabs_cte, heading_vol], dtype=float)

    def get_lag_features_batch(
        self,
        acids: List[str],
        iaf_xs: np.ndarray,
        iaf_ys: np.ndarray,
        approach_headings_rad: np.ndarray,
    ) -> np.ndarray:
        """Compute lag features for a fleet of aircraft in one call.

        Parameters
        ----------
        acids : list[str]
            Aircraft callsigns in fleet order, length n.
        iaf_xs, iaf_ys : np.ndarray, shape (n,)
            IAF anchor coordinates for each aircraft's current runway.
        approach_headings_rad : np.ndarray, shape (n,)
            Approach track headings in bearing radians.

        Returns
        -------
        np.ndarray, shape (n, 3)
            Stacked lag feature vectors ``[delta_atd, cumabs_cte, heading_vol]``.
        """
        n = len(acids)
        out = np.zeros((n, self._N_LAG), dtype=float)
        for i, acid in enumerate(acids):
            out[i] = self.get_lag_features(
                acid,
                float(iaf_xs[i]),
                float(iaf_ys[i]),
                float(approach_headings_rad[i]),
            )
        return out

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def evict(self, acid: str) -> None:
        """Remove trajectory history for a departed aircraft.

        Safe to call even if *acid* is not in the buffer.
        """
        self._history.pop(acid, None)

    def reset(self) -> None:
        """Clear all trajectory history.

        Call between episodes so that lag features from the previous
        episode do not bleed into the new one.
        """
        self._history.clear()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def tracked_aircraft(self) -> List[str]:
        """Callsigns of aircraft with at least one stored state."""
        return list(self._history.keys())

    def history_length(self, acid: str) -> int:
        """Number of stored states for *acid* (0 if not tracked)."""
        buf = self._history.get(acid)
        return len(buf) if buf else 0

    def __repr__(self) -> str:
        return (
            f"TrajectoryBuffer("
            f"n_aircraft={len(self._history)}, "
            f"maxlen={self.maxlen}, "
            f"lag_steps={self.lag_steps}, "
            f"window={self.window})"
        )
