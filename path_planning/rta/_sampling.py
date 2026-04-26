import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from bluesky_gym.envs.common.base_sampler import BaseSampler

from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from sklearn.neighbors import KDTree

from typing import Any, Optional, TypeVar, Type, Union, Callable

#TODO: look into ProbabilisticResidualForest, use the sources, for now just use Deterministic for testing

RegressorT = TypeVar(
    "RegressorT",
    RandomForestRegressor,
    ExtraTreesRegressor
)

class MinTimeInterpolator:
    """
    Learns the empirical minimum feasible time from training data.
    Uses a KD-tree to find local minimums, then fits an RBF interpolator.
    
    Args:
        k_neighbors: Number of neighbours used to estimate local minimum.
        smoothing: RBF smoothing factor (0 = exact interpolation).
        kernel: RBF kernel type ('thin_plate_spline', 'linear', 'cubic', etc.)
    """
    def __init__(
        self, 
        k_neighbors: int = 10, 
    ):
        self.k_neighbors = k_neighbors
        self._interpolator = None
        self._scaler = StandardScaler()
        self._y_min_global = None  # fallback

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_arr = np.atleast_2d(X)
        y_arr = np.array(y)
        
        # Scale features for fair distance computation
        X_scaled = self._scaler.fit_transform(X_arr)
        
        # For each point, find the minimum y in its local neighbourhood
        tree = KDTree(X_scaled)
        _, indices = tree.query(X_scaled, k=min(self.k_neighbors, len(y_arr)))
        local_mins = np.array([y_arr[idx].min() for idx in indices])
        
        self._y_min_global = float(y_arr.min())  # fallback for extrapolation
        self._interpolator = LinearNDInterpolator(X_scaled, local_mins, fill_value=self._y_min_global)

    def __call__(self, X: np.ndarray) -> np.ndarray | float:
        """Query minimum feasible time for given features."""
        if self._interpolator is None:
            raise RuntimeError("MinTimeInterpolator must be fitted before use.")
        
        X_scaled = self._scaler.transform(np.atleast_2d(X))
        min_times = self._interpolator(X_scaled)

        # RBF can extrapolate negatives in sparse regions — clip to global min
        return np.maximum(min_times, self._y_min_global) # type: ignore

class DeterministicResidualForest(BaseSampler):
    def __init__(self, regressor_cls: Type[RegressorT], n_estimators: int = 100, random_state: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        regressor = regressor_cls(
            n_estimators=n_estimators, 
            random_state=random_state,
            *args,
            **kwargs
        )

        # The core pipeline: Scaler + RF
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', regressor)
        ])

        #gaussian parameters
        self.mu = 0.0
        self.sigma = 0.0

    def fit(self, X: Any, y: Any, **kwargs) -> None:
        self.pipeline.fit(X, y)
        y_hat = self.pipeline.predict(X)

        residuals = y - y_hat
        self.mu = np.mean(residuals)
        self.sigma = np.std(residuals)

    def sample(self, X: Any, **kwargs) -> Any:
        if self.pipeline is None:
            raise RuntimeError("Sampler must be fitted before sampling.")

        # Ensure X is 2D for sklearn pipeline
        X_arr = np.atleast_2d(X)
        
        # Step 1: Get the tree prediction
        # This represents Epistemic Uncertainty (model disagreement)
        rf = self.pipeline.named_steps['rf']
        scaler = self.pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X_arr)

        rf_predictions = rf.predict(X_scaled)

        # Step 2: Add the Gaussian Residual
        # This represents Aleatoric Uncertainty (environmental noise)
        # We include the mean residual (mu) to correct any systemic bias
        noise = self._rng.normal(self.mu, self.sigma, size=rf_predictions.shape)
        
        result = rf_predictions + noise

        return result[0] if np.ndim(X) == 1 else result
        

class ProbabilisticResidualForest(BaseSampler):
    """
    A probabilistic sampler that combines a Random Forest with a Gaussian 
    residual model to capture both model uncertainty and data noise.

    sources:
    - Meinshausen, N. (2006). "Quantile Regression Forests." Journal of Machine Learning Research.
    - Shaker, M. H., & Hüllermeier, E. (2020). "Aleatoric and Epistemic Uncertainty with Random Forests." International Conference on Information Processing and Management of Uncertainty in Knowledge-Based Systems.
    """

    def __init__(self, regressor_cls: Type[RegressorT], n_estimators: int = 100, random_state: Optional[int] = None, *args, **kwargs):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        
        regressor = regressor_cls(
            n_estimators=n_estimators, 
            random_state=random_state,
            *args,
            **kwargs
        )

        # The core pipeline: Scaler + RF
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', regressor)
        ])
        
        # Gaussian parameters
        self.mu = 0.0
        self.sigma = 0.0

    def fit(self, X: Any, y: Any, **kwargs) -> None:
        """
        Fits the pipeline and calculates the distribution of the residuals.
        """
        # 1. Fit the scaled Random Forest
        self.pipeline.fit(X, y)

        # 2. Characterize the residuals (errors)
        # mu: The average bias (ideally near 0)
        # sigma: The spread of the noise
        y_pred = self.pipeline.predict(X)
        residuals = y - y_pred
        
        self.mu = float(np.mean(residuals))
        self.sigma = float(np.std(residuals))

    def sample(self, X: Any, n_subset: Optional[int] = None, **kwargs) -> Any:
        """
        Draws a sample using Randomized Tree Selection + Gaussian Jitter.
        """
        if self.pipeline is None:
            raise RuntimeError("Sampler must be fitted before sampling.")

        # Ensure X is 2D for sklearn pipeline
        X_arr = np.atleast_2d(X)
        
        # Step 1: Get the 'Stochastic Mean' from a random tree
        # This represents Epistemic Uncertainty (model disagreement)
        rf = self.pipeline.named_steps['rf']
        scaler = self.pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X_arr)

        #TODO: check if this also works with extra trees
        k = n_subset if n_subset is not None else self.n_estimators

        indices = self._rng.choice(self.n_estimators, size=k, replace=False)
        # We use a single tree's prediction as our 'mu_stochastic'
        subset_preds = np.array([rf.estimators_[i].predict(X_scaled) for i in indices])
        mu_stochastic = np.mean(subset_preds, axis=0)

        # Step 2: Add the Gaussian Residual
        # This represents Aleatoric Uncertainty (environmental noise)
        # We include the mean residual (mu) to correct any systemic bias
        noise = self._rng.normal(self.mu, self.sigma, size=mu_stochastic.shape)
        
        result = mu_stochastic + noise

        return result[0] if np.ndim(X) == 1 else result

MinTimeFn = Optional[Callable[[np.ndarray, Any], Union[np.ndarray, float]]]

class RunwaySpecificSampler(BaseSampler):
    """
    A coordinator that manages individual samplers for different runways.
    """
    def __init__(self, sampler_type: Union[Type[DeterministicResidualForest], Type[ProbabilisticResidualForest]] = ProbabilisticResidualForest, min_time_fn: MinTimeFn = None, **sampler_kwargs):
        super().__init__()
        self.sampler_type = sampler_type
        self.sampler_kwargs = sampler_kwargs
        self.samplers = {}
        self.min_time_fn = min_time_fn  # fn(X, runway_id) -> min_time

    def fit(self, X: Any, y: Any, runway_ids: Any, **kwargs) -> None:
        """
        Fits a unique sampler for every unique runway, plus a global fallback.
        """
        X_arr = np.array(X)
        y_arr = np.array(y)
        runway_ids = np.array(runway_ids)
        
        unique_runways = np.unique(runway_ids)
        
        # 1. Fit individual runway samplers
        for rwy in unique_runways:
            mask = (runway_ids == rwy)
            sampler = self.sampler_type(**self.sampler_kwargs)
            sampler.fit(X_arr[mask], y_arr[mask])
            self.samplers[rwy] = sampler

    def sample(self, X: Any, runway_id: Any, **kwargs) -> Any:
        """
        Samples from specific runway samplers with support for scalar IDs, 
        broadcasted IDs, or batch arrays.
        """
        # 1. Normalize X to at least 2D (n_samples, n_features)
        X_arr = np.atleast_2d(X)
        n_samples = X_arr.shape[0]

        # 2. Normalize runway_id to an array of length n_samples
        if np.isscalar(runway_id) or isinstance(runway_id, str):
            # Scenario: One runway for one or many X points
            rwy_ids = np.full(n_samples, runway_id, dtype=object)
        else:
            # Scenario: Array of runway IDs
            rwy_ids = np.asarray(runway_id)
            if rwy_ids.shape[0] != n_samples:
                raise ValueError(f"Batch size mismatch: X has {n_samples}, but runway_id has {rwy_ids.shape[0]}")

        # 3. Optimized Grouped Execution
        # Instead of looping over every row, we loop over unique runways 
        # to minimize dictionary lookups and function calls.
        unique_rwys = np.unique(rwy_ids)
        final_results = np.zeros(n_samples)

        for r_id in unique_rwys:
            sampler = self.samplers.get(r_id)
            if sampler is None:
                raise RuntimeError(f"Sampler not fitted for runway {r_id}.")
            
            # Mask for the current runway
            mask = (rwy_ids == r_id)
            X_subset = X_arr[mask]
            
            # Generate samples for this group
            samples = sampler.sample(X_subset, **kwargs)
            
            # Apply physical constraint (min_time_fn)
            if self.min_time_fn is not None:
                min_times = self.min_time_fn(X_subset, r_id)

                samples = np.maximum(samples, min_times)
            
            final_results[mask] = samples.flatten()

        # 4. Match return format to input format
        if np.isscalar(runway_id) and np.ndim(X) == 1:
            return final_results[0]
        return final_results

def create_runway_sampler(
    deterministic: bool = False,
    use_extra_trees: bool = False,
    n_estimators: int = 100,
    random_state: Optional[int] = None,
    min_time_fn: MinTimeFn = None,  # fn(X, runway_id) -> min_time
    **kwargs
) -> RunwaySpecificSampler:
    sampler_type = DeterministicResidualForest if deterministic else ProbabilisticResidualForest
    regressor_cls = ExtraTreesRegressor if use_extra_trees else RandomForestRegressor

    sampler_kwargs = {
        "regressor_cls": regressor_cls,
        "n_estimators": n_estimators,
        "random_state": random_state,
        **kwargs
    }

    return RunwaySpecificSampler(
        sampler_type=sampler_type,
        min_time_fn=min_time_fn,  # lives here now, not in sampler_kwargs
        **sampler_kwargs
    )