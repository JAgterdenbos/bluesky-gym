from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional

import numpy as np

from bluesky_gym.envs.common.base_sampler import BaseSampler
from .registry import SamplerRegistry
from .plot import plot_rta_distribution, PlotKind, CoordSystem


class RTASampler(BaseSampler):
    """
    RTASampler — abstract base class for all RTA distribution samplers.

    Design contract
    ---------------
    * `_sample(runway, X)` receives n-dimensional coordinate arrays.
    * The coordinate system used to display results is a rendering concern and
      is never seen by samplers.
    * Subclasses register themselves automatically via `__init_subclass__`.
    * `fit` / `_fit` follow the same public/private wrapper pattern.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runways: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Auto-registration                                                  #
    # ------------------------------------------------------------------ #

    def __init_subclass__(
        cls, name: Optional[str] = None, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        SamplerRegistry.register(name or cls.__name__)(cls)

    # ------------------------------------------------------------------ #
    # State queries                                                      #
    # ------------------------------------------------------------------ #

    @property
    def runways(self) -> List[str]:
        """Fitted runway identifiers. Empty list if not yet fitted."""
        return self._runways or []

    @property
    def is_fitted(self) -> bool:
        """True once fit() has been called successfully."""
        return self._runways is not None

    def is_runway_fitted(self, runway: str | List[str]) -> bool:
        """Return True if every requested runway has been fitted."""
        if isinstance(runway, str):
            return runway in self.runways
        return bool(runway) and all(r in self.runways for r in runway)

    # ------------------------------------------------------------------ #
    # Public fit / sample interface                                      #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        runways: List[str] | str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Normalise runway input and delegate to _fit."""
        self._runways = [runways] if isinstance(runways, str) else list(runways)
        self._fit(X, y, self._runways, *args, **kwargs)

    def sample(
        self,
        runway: str,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Normalise runway input and delegate to _sample.
        """
        if not self.is_runway_fitted(runway):
            raise ValueError(
                f"Runway '{runway}' has not been fitted. "
                f"Available: {self.runways}"
            )

        return self._sample(runway, X)

    @abstractmethod
    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: List[str], *args: Any, **kwargs: Any) -> None:
        """Fit the model for the given runways. Implemented by subclasses."""

    @abstractmethod
    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        """
        Predict time-to-go for the given coordinates.

        Inputs are provided as unpacked arrays. For 2D samplers, 
        subclasses should implement this as:
        
        def _sample(self, runway, x, y):
            ...
        """

    def plot_distribution(
        self,
        runways: Optional[List[str] | str] = None,
        n_points: int = 10_000,
        kind: PlotKind = PlotKind.CONTOUR,
        coord: CoordSystem = CoordSystem.CARTESIAN,
        save_path: Optional[str] = None,
        *,
        sample_coord: Optional[CoordSystem] = None,
    ) -> None:
        """
        Plot the fitted distribution.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before plot_distribution().")
    

        if runways is None:
            runways = self.runways
        elif isinstance(runways, str):
            runways = [runways]

        # The plotting utility should be updated to handle the *coords signature
        plot_rta_distribution(
            self.sample, runways, n_points, kind, coord, save_path, sample_coord=sample_coord
        )