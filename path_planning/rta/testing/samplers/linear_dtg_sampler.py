import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    BayesianRidge,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .dtg_sampler import DTGSampler

from typing import List


class LinearDTGSampler(DTGSampler, name="LinearDTGSampler"):
    """
    Ordinary Least Squares predictor for DTG distributions.

    A fast, interpretable baseline. Features are automatically standardised
    before fitting so that coefficient magnitudes are comparable.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._models: dict[str, Pipeline] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)


class RidgeDTGSampler(DTGSampler, name="RidgeDTGSampler"):
    """
    Ridge (L2-regularised) regression predictor for DTG distributions.

    Adds an L2 penalty to OLS, shrinking coefficients towards zero and
    reducing variance at the cost of a small bias. A good default when
    features are correlated or the dataset is small.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self._models: dict[str, Pipeline] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=self.alpha)),
            ]).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)


class LassoDTGSampler(DTGSampler, name="LassoDTGSampler"):
    """
    Lasso (L1-regularised) regression predictor for DTG distributions.

    Adds an L1 penalty which drives irrelevant feature coefficients exactly to
    zero, performing implicit feature selection. Useful when you suspect only a
    subset of features drive DTG.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 10_000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.max_iter = max_iter
        self._models: dict[str, Pipeline] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=self.alpha, max_iter=self.max_iter)),
            ]).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)


class BayesRidgeDTGSampler(DTGSampler, name="BayesRidgeDTGSampler"):
    """
    Bayesian Ridge regression predictor for DTG distributions.

    Estimates regularisation strength from the data via an evidence-maximisation
    procedure, removing the need to tune ``alpha`` manually. Also provides a
    measure of predictive uncedtginty via ``predict(return_std=True)``, which
    can be useful for downstream risk-aware decision making.
    """

    def __init__(
        self,
        max_iter: int = 300,
        tol: float = 1e-3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_iter = max_iter
        self.tol = tol
        self._models: dict[str, Pipeline] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", BayesianRidge(max_iter=self.max_iter, tol=self.tol)),
            ]).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)