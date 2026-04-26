import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from bluesky_gym.envs.common.base_sampler import BaseSampler

from typing import Dict, Optional, Any, Union, List, Callable, TypeVar, Type

T = TypeVar("T", bound="BaseSampler")

#TODO: The tree represents Epistemic Uncertainty, should we add Aleatoric Uncertainty? For prediction of the min_distance to fly to the runway?
class ExtraTreesSampler(BaseSampler):
    def __init__(
            self, 
            n_estimators: int = 100, 
            max_depth: Optional[int] = None, 
            min_samples_leaf: int = 1,
            random_state: Optional[int] = None
        ):
        super().__init__()
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ExtraTreesRegressor(
                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
            ))
        ])

    def fit(self, X: Any, y: Any):
        self.pipeline.fit(X, y)

    def sample(self, X: Any, **kwargs) -> np.ndarray:
        if not hasattr(self.pipeline.named_steps['regressor'], 'estimators_'):
            raise ValueError("The model must be fitted before sampling.")
        
        X_arr = np.atleast_2d(X)

        regressor = self.pipeline.named_steps['regressor']
        scaler = self.pipeline.named_steps['scaler']
        X_scaled = scaler.transform(X_arr)

        return regressor.predict(X_scaled)
    
    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=('zlib', 3))

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object of type {type(obj).__name__} "
                f"is not a subclass of {cls.__name__}"
            )
        return obj
    
class RunwaySpecificSampler(BaseSampler):
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, min_samples_leaf: int = 1, random_state: Optional[int] = None):
        super().__init__()
        self._sampler_kwargs = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }

        self.samplers: Dict[Any, ExtraTreesSampler] = {}
        self.min_fn: Optional[Callable[[Any, Any], Any]] = None

    def fit(self, X: Any, y: Any, runway_ids: Union[List[Any], Any]):
        X_arr, y_arr = np.array(X), np.array(y)
        runway_ids_arr = np.array(runway_ids)

        for rwy in np.unique(runway_ids_arr):
            mask = (runway_ids == rwy)
            sampler = ExtraTreesSampler(**self._sampler_kwargs)
            sampler.fit(X_arr[mask], y_arr[mask])
            self.samplers[rwy] = sampler

    def sample(self, X: Any, runway_id: Any, **kwargs) -> Any:

        X_arr = np.atleast_2d(X)
        n_samples = X_arr.shape[0]

        if np.isscalar(runway_id) or isinstance(runway_id, str):
            # Scenario: One runway for one or many X points
            rwy_ids = np.full(n_samples, runway_id, dtype=object)
        else:
            # Scenario: Array of runway IDs
            rwy_ids = np.asarray(runway_id)
            if rwy_ids.shape[0] != n_samples:
                raise ValueError(f"Batch size mismatch: X has {n_samples}, but runway_id has {rwy_ids.shape[0]}")
            
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
            
            # Apply physical constraint (min_fn)
            if self.min_fn is not None:
                min_values = self.min_fn(X_subset, r_id)

                samples = np.maximum(samples, min_values)
            
            final_results[mask] = samples.flatten()

        # 4. Match return format to input format
        if np.isscalar(runway_id) and np.ndim(X) == 1:
            return final_results[0]
        return final_results
    
    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=('zlib', 3))

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object of type {type(obj).__name__} "
                f"is not a subclass of {cls.__name__}"
            )
        return obj

class UnifiedRunwaySampler(BaseSampler):
    """
    Single Extra Trees model with runway label-encoded as a feature.
    Memory-efficient alternative to per-runway models.
    """
    def __init__(
        self,
        n_estimators: int = 100, 
        max_depth: Optional[int] = None, 
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        known_runways: Optional[List[str]] = None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Fit the label encoder up-front if runway universe is known,
        # so it stays stable across train/inference even if not all
        # runways appear in a training split.
        self.label_encoder = LabelEncoder()
        if known_runways is not None:
            self.label_encoder.fit(sorted(known_runways))
            self._encoder_fitted = True
        else:
            self._encoder_fitted = False

        self.scaler = StandardScaler()
        self.regressor = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf
        )
        self.min_fn: Optional[Callable[[Any, Any], Any]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_runways(self, runway_ids: Any) -> np.ndarray:
        rwy_arr = np.asarray(runway_ids)
        return self.label_encoder.transform(rwy_arr).reshape(-1, 1)

    def _build_features(self, X: Any, runway_ids: Any) -> np.ndarray:
        """Concatenate spatial features with encoded runway column."""
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        rwy_encoded = self._encode_runways(runway_ids)
        return np.hstack([X_arr, rwy_encoded])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any, runway_ids: Union[List[Any], Any]) -> None:
        runway_ids_arr = np.asarray(runway_ids)

        if not self._encoder_fitted:
            self.label_encoder.fit(runway_ids_arr)
            self._encoder_fitted = True

        X_combined = self._build_features(X, runway_ids_arr)
        X_scaled = self.scaler.fit_transform(X_combined)
        self.regressor.fit(X_scaled, np.asarray(y))
        self._fitted = True

    def sample(self, X: Any, runway_id: Any, **kwargs) -> Any:
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling.")

        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        n_samples = X_arr.shape[0]

        # Broadcast scalar runway to all rows
        if np.isscalar(runway_id) or isinstance(runway_id, str):
            rwy_ids = np.full(n_samples, runway_id, dtype=object)
        else:
            rwy_ids = np.asarray(runway_id)
            if rwy_ids.shape[0] != n_samples:
                raise ValueError(
                    f"Batch size mismatch: X has {n_samples} rows "
                    f"but runway_id has {rwy_ids.shape[0]}."
                )

        X_combined = self._build_features(X_arr, rwy_ids)
        X_scaled = self.scaler.transform(X_combined)
        predictions = self.regressor.predict(X_scaled)

        # Apply physical floor constraint if provided
        if self.min_fn is not None:
            min_values = self.min_fn(X_arr, rwy_ids)
            predictions = np.maximum(predictions, min_values)

        # Mirror input shape: scalar runway + 1-D X → scalar out
        if (np.isscalar(runway_id) or isinstance(runway_id, str)) and np.ndim(X) == 1:
            return float(predictions[0])
        return predictions

    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=("zlib", 3))

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object of type {type(obj).__name__} "
                f"is not a subclass of {cls.__name__}"
            )
        return obj


class GeoRunwaySampler(BaseSampler):
    """
    Single Extra Trees model that represents each runway by its physical
    properties (FAF latitude, longitude, and track heading) instead of an
    opaque label encoding.

    This gives the model interpretable, continuous runway features so it can
    potentially generalise to unseen runways that are geometrically similar to
    ones seen during training.

    Feature vector per sample:
        [*spatial_features, faf_lat, faf_lon, sin(track_rad), cos(track_rad)]

    Track is decomposed into sin/cos components to avoid the 359->0
    wrap-around discontinuity that would otherwise confuse the scaler and
    tree splits.
    """

    def __init__(
        self,
        runway_geo: Dict[str, Dict[str, float]],
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        runway_geo:
            Mapping of runway name -> {"lat": ..., "lon": ..., "track": ...}.
            Track is in degrees (0-360, magnetic or true -- be consistent).
        """
        super().__init__()
        self.runway_geo = runway_geo

        # Pre-build a lookup: rwy_name -> np.ndarray([lat, lon, sin(t), cos(t)])
        self._geo_features: Dict[str, np.ndarray] = {}
        for rwy, geo in runway_geo.items():
            track_rad = np.deg2rad(geo["track"])
            self._geo_features[rwy] = np.array(
                [geo["lat"], geo["lon"], np.sin(track_rad), np.cos(track_rad)],
                dtype=np.float32,
            )

        self.scaler = StandardScaler()
        self.regressor = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.min_fn: Optional[Callable[[Any, Any], Any]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, X: np.ndarray, runway_ids: np.ndarray) -> np.ndarray:
        """Return X concatenated with [lat, lon, sin(track), cos(track)]."""
        try:
            geo_block = np.stack(
                [self._geo_features[r] for r in runway_ids], axis=0
            )  # (n, 4)
        except KeyError as e:
            raise KeyError(
                f"Runway {e} not found in runway_geo. "
                f"Known runways: {list(self._geo_features.keys())}"
            ) from e
        return np.hstack([X, geo_block])  # (n, spatial_dim + 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: Any, y: Any, runway_ids: Union[List[Any], Any]) -> None:
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        y_arr = np.asarray(y)
        rwy_arr = np.asarray(runway_ids)

        X_combined = self._build_features(X_arr, rwy_arr)
        X_scaled = self.scaler.fit_transform(X_combined)
        self.regressor.fit(X_scaled, y_arr)
        self._fitted = True

    def sample(self, X: Any, runway_id: Any, **kwargs) -> Any:
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling.")

        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float32))
        n_samples = X_arr.shape[0]

        # Broadcast scalar runway to all rows
        if np.isscalar(runway_id) or isinstance(runway_id, str):
            rwy_ids = np.full(n_samples, runway_id, dtype=object)
        else:
            rwy_ids = np.asarray(runway_id)
            if rwy_ids.shape[0] != n_samples:
                raise ValueError(
                    f"Batch size mismatch: X has {n_samples} rows "
                    f"but runway_id has {rwy_ids.shape[0]}."
                )

        X_combined = self._build_features(X_arr, rwy_ids)
        X_scaled = self.scaler.transform(X_combined)
        predictions = self.regressor.predict(X_scaled)

        # Apply physical floor constraint if provided
        if self.min_fn is not None:
            min_values = self.min_fn(X_arr, rwy_ids.tolist())
            predictions = np.maximum(predictions, min_values)

        # Mirror input shape: scalar runway + 1-D X -> scalar out
        if (np.isscalar(runway_id) or isinstance(runway_id, str)) and np.ndim(X) == 1:
            return float(predictions[0])
        return predictions

    def save(self, path: str) -> None:
        joblib.dump(self, path, compress=("zlib", 3))

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object of type {type(obj).__name__} "
                f"is not a subclass of {cls.__name__}"
            )
        return obj