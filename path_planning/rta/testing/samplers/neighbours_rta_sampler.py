import numpy as np
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

from .rta_sampler import RTASampler

from typing import List, Literal, Optional


class KNNRTASampler(RTASampler, name="KNNRTASampler"):
    """
    k-Nearest-Neighbours predictor for RTA distributions.

    Predictions are the (optionally distance-weighted) average of the k
    closest training samples. Non-parametric and naturally adapts to local
    structure, but memory-intensive for large datasets.
    """

    def __init__(
        self,
        k: int = 50,
        weights: Literal["uniform", "distance"] = "distance",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.k = k
        self.weights = weights
        self._models: dict[str, KNeighborsRegressor] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = KNeighborsRegressor(
                n_neighbors=self.k,
                weights=self.weights,  # type: ignore
            ).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)


class RadiusNeighborsRTASampler(RTASampler, name="RadiusNeighborsRTASampler"):
    """
    Radius-based Nearest-Neighbours predictor for RTA distributions.

    Instead of a fixed number of neighbours, uses all training samples within
    a fixed radius. Prediction density therefore varies with local data density,
    which can be more representative in unevenly sampled airspace regions.

    If no neighbours fall within ``radius`` for a query point, ``outlier_value``
    is returned for that sample (defaults to ``np.nan`` so outliers are visible).
    """

    def __init__(
        self,
        radius: float = 1.0,
        weights: Literal["uniform", "distance"] = "distance",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.weights = weights
        self._models: dict[str, RadiusNeighborsRegressor] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = RadiusNeighborsRegressor(
                radius=self.radius,
                weights=self.weights,  # type: ignore
            ).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)