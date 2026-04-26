import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor, # Added this
)

from .dtg_sampler import DTGSampler

from typing import List, Optional

class RFDTGSampler(DTGSampler, name="RFDTGSampler"):
    """
    Random Forest predictor for DTG distributions.
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
            self._models[rwy] = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            ).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)


class ETDTGSampler(DTGSampler, name="ETDTGSampler"):
    """
    Extremely Randomised Trees predictor for DTG distributions.
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
        self._models: dict[str, ExtraTreesRegressor] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            ).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)


class GBDTGSampler(DTGSampler, name="GBDTGSampler"):
    """
    Gradient Boosting predictor for DTG distributions.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int = 42,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self._models: dict[str, GradientBoostingRegressor] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                random_state=self.random_state,
            ).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)


class HistGBDTGSampler(DTGSampler, name="HistGBDTGSampler"):
    """
    Histogram-based Gradient Boosting predictor for DTG distributions.

    A much faster alternative to GBDTGSampler for large datasets, 
    integrated into scikit-learn. Very stable and handles missing values natively.
    """

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = None,
        l2_regularization: float = 0.0,
        random_state: int = 42,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Note: HistGradientBoosting uses 'max_iter' instead of 'n_estimators'
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_regularization = l2_regularization
        self.random_state = random_state
        self._models: dict[str, HistGradientBoostingRegressor] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state,
            ).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)