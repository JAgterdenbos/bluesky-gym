import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels

from .dtg_sampler import DTGSampler
from typing import List, Literal

class KDEDTGSampler(DTGSampler, name="KDEDTGSampler"):
    """
    Kernel Density Estimation (Nadaraya-Watson) regressor for DTG.
    
    Provides a smooth, weighted average of all training samples. 
    The 'bandwidth' parameter controls the smoothness of the distribution.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: Literal["gaussian", "laplacian", "rbf"] = "gaussian",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bandwidth = bandwidth
        self.kernel = kernel
        self._data: dict[str, dict] = {}

    def _fit(self, X: List[np.ndarray], y: List[np.ndarray], runways: List[str]) -> None:
        for i, rwy in enumerate(runways):
            scaler = StandardScaler().fit(X[i])
            # We store the scaled training data to compare against during sampling
            self._data[rwy] = {
                "X": scaler.transform(X[i]),
                "y": y[i],
                "scaler": scaler
            }

    def _sample(self, runway: str, X: np.ndarray) -> np.ndarray:
        entry = self._data.get(runway)
        if entry is None:
            raise KeyError(f"No fitted data for runway '{runway}'.")

        print(f"Sampling for runway '{runway}' with bandwidth={self.bandwidth} and kernel='{self.kernel}'")

        scaler = entry["scaler"]
        X_train = entry["X"]
        y_train = entry["y"]
        
        # 1. Scale query points
        X_query = scaler.transform(X)

        # 2. Compute the Kernel matrix (weights)
        # Gamma is often defined as 1 / (2 * bandwidth^2) for RBF/Gaussian
        gamma = 1.0 / (2.0 * self.bandwidth ** 2)
        weights = pairwise_kernels(X_query, X_train, metric=self.kernel, gamma=gamma)

        # 3. Compute weighted average: (Sum of weight * y) / (Sum of weights)
        weighted_sum = np.dot(weights, y_train)
        sum_of_weights = np.sum(weights, axis=1)
        
        # Avoid division by zero for points very far from data
        return weighted_sum / np.where(sum_of_weights == 0, 1, sum_of_weights)