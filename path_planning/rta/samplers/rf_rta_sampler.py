import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .rta_sampler import RTASampler

from typing import List, Optional


class RFRTASampler(RTASampler, name="RFRTASampler"):
    """
    Random Forest predictor for RTA (Required Time of Arrival) distributions.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self._models: dict[str, RandomForestRegressor] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            X_feat = X[i]
            y_target = y[i]

            # n_jobs=-1 ensures the forest builds trees in parallel across all CPU cores
            self._models[rwy] = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1
            ).fit(X_feat, y_target)

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")

        # Returns the average prediction from all trees in the forest
        result = model.predict(X)

        return result