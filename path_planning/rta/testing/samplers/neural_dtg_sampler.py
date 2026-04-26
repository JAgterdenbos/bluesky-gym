import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .dtg_sampler import DTGSampler

from typing import List, Literal, Tuple, Union


class MLPDTGSampler(DTGSampler, name="MLPDTGSampler"):
    """
    Multi-Layer Perceptron predictor for DTG distributions.

    A fully-connected feed-forward neural network implemented via
    ``sklearn.neural_network.MLPRegressor``. Features are automatically
    standardised before fitting.

    Parameters
    ----------
    hidden_layer_sizes:
        Number of neurons in each hidden layer, e.g. ``(128, 64)`` creates
        two hidden layers with 128 and 64 units respectively.
    activation:
        Activation function for hidden layers. ``'relu'`` is a sensible
        default; ``'tanh'`` can help if inputs have strong negative values.
    solver:
        Weight optimisation algorithm. ``'adam'`` works well for large
        datasets; ``'lbfgs'`` can converge faster on small ones.
    alpha:
        L2 regularisation term. Increase to reduce over-fitting.
    learning_rate_init:
        Initial step size for ``'adam'`` or ``'sgd'`` solvers.
    max_iter:
        Maximum number of training epochs.
    early_stopping:
        If ``True``, reserves 10 % of training data for validation and
        halts training when validation score stops improving.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_layer_sizes: Union[Tuple[int, ...], List[int]] = (128, 64),
        activation: Literal["relu", "tanh", "logistic", "identity"] = "relu",
        solver: Literal["adam", "lbfgs", "sgd"] = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = True,
        random_state: int = 42,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self._models: dict[str, Pipeline] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: list[str]) -> None:
        for i, rwy in enumerate(runways):
            self._models[rwy] = Pipeline([
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation,
                    solver=self.solver,
                    alpha=self.alpha,
                    learning_rate_init=self.learning_rate_init,
                    max_iter=self.max_iter,
                    early_stopping=self.early_stopping,
                    random_state=self.random_state,
                )),
            ]).fit(X[i], y[i])

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        model = self._models.get(runway)
        if model is None:
            raise KeyError(f"No fitted model for runway '{runway}'.")
        return model.predict(X)