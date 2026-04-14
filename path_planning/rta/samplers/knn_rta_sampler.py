import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from .rta_sampler import RTASampler

from typing import List, Literal


class KNNRTASampler(RTASampler, name="KNNRTASampler"):
    """
    k-Nearest-Neighbours predictor for RTA distributions.
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
            X_feat = X[i]
            y_target = y[i]

            self._models[rwy] = KNeighborsRegressor(
                n_neighbors=self.k,
                weights=self.weights,  # type: ignore
            ).fit(X_feat, y_target)

    # ------------------------------------------------------------------ #
    # Sample                                                              #
    # ------------------------------------------------------------------ #

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")

        result = model.predict(X)

        return result