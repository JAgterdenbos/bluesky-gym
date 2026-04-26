from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional

import numpy as np

from bluesky_gym.envs.common.base_sampler import BaseSampler
from .registry import SamplerRegistry

from .plot import PlotKind, CoordSystem


class DTGSampler(BaseSampler):
    """
    DTGSampler — abstract base class for all Distance-To-Go distribution samplers.

    Predicts P(dist_to_go | x, y, t, runway), where:
        x, y        : normalised aircraft position  [-1, 1]
        t           : normalised elapsed time        [0, 1]
        dist_to_go  : remaining path distance (km) = total_dist_km - path_len

    Including t as a feature future-proofs the sampler for variable-speed
    scenarios: when speed is constant, dist_to_go and rta_remaining are
    perfectly correlated, so t adds no information. When speed varies,
    t decouples schedule progress from physical progress, making it a
    genuinely informative feature.

    Design contract
    ---------------
    * `_sample(runway, X)` receives (N, 3) arrays of [coord1, coord2, t].
    * The coordinate system used to display results is a rendering concern and
      is never seen by samplers.
    * Subclasses register themselves automatically via `__init_subclass__`.
    * `fit` / `_fit` follow the same public/private wrapper pattern.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._runways: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    # Auto-registration                                                    #
    # ------------------------------------------------------------------ #

    def __init_subclass__(
        cls, name: Optional[str] = None, register: bool = True, **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        if register:
            SamplerRegistry.register(name or cls.__name__)(cls)

    # ------------------------------------------------------------------ #
    # State queries                                                        #
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
    # Public fit / sample interface                                        #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        runways: List[str] | str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Fit the sampler.

        Parameters
        ----------
        X : list of (N_i, 3) arrays
            Each array contains [coord1, coord2, t] for runway i.
            coord1/coord2 are in the space defined by the CoordSystem passed
            to fit_and_plot (Cartesian or Polar). t is always normalised [0, 1].
        y : list of (N_i,) arrays
            dist_to_go values (km) for each sample.
        runways : list of str or str
            Runway identifier(s) corresponding to each array in X and y.
        """
        self._runways = [runways] if isinstance(runways, str) else list(runways)
        self._fit(X, y, self._runways, *args, **kwargs)

    def sample(
        self,
        runway: str,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Sample dist_to_go predictions for the given inputs.

        Parameters
        ----------
        runway : str
            The runway identifier.
        X : (N, 3) array
            Columns: [coord1, coord2, t].

        Returns
        -------
        (N,) array of predicted dist_to_go values (km).
        """
        if not self.is_runway_fitted(runway):
            raise ValueError(
                f"Runway '{runway}' has not been fitted. "
                f"Available: {self.runways}"
            )
        return self._sample(runway, X)

    @abstractmethod
    def _fit(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        runways: List[str],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Fit the model for the given runways. Implemented by subclasses."""

    @abstractmethod
    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        """
        Predict dist_to_go for the given coordinate + time inputs.

        Parameters
        ----------
        runway : str
        X : (N, 3) array — [coord1, coord2, t]

        Returns
        -------
        (N,) array of dist_to_go predictions.
        """

    def plot_distribution(
        self,
        runways: Optional[List[str] | str] = None,
        n_points: int = 10_000,
        kind: PlotKind = PlotKind.CONTOUR,
        coord: CoordSystem | str = CoordSystem.CARTESIAN,
        save_path: Optional[str] = None,
        *,
        sample_coord: Optional[CoordSystem | str] = None,
    ) -> None:
        """Plot the fitted DTG distribution."""
        from .plot import plot_dtg_distribution
        if not self.is_fitted:
            raise RuntimeError("Call fit() before plot_distribution().")

        if runways is None:
            runways = self.runways
        elif isinstance(runways, str):
            runways = [runways]

        if isinstance(coord, str):
            coord = CoordSystem.from_str(coord)

        if isinstance(sample_coord, str):
            sample_coord = CoordSystem.from_str(sample_coord)

        plot_dtg_distribution(
            self.sample, runways, n_points, kind, coord, save_path,
            sample_coord=sample_coord,
        )