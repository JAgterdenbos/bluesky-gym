from abc import abstractmethod
from bluesky_gym.envs.common.base_sampler import BaseSampler
from .registry import SamplerRegistry

from typing import Any, Optional, List

class RTASampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._runways: Optional[List[str]] = None

    def __init_subclass__(cls, name: Optional[str] = None, *args, **kwargs):
        """Register the sampler class in the registry."""
        super().__init_subclass__(*args, **kwargs)
        cls_name = name or cls.__name__
        SamplerRegistry.register(cls_name)(cls)

    @property
    def runways(self) -> List[str]:
        """List of available runways. Returns empty list if not fitted."""
        return self._runways or []
    
    @property
    def is_fitted(self) -> bool:
        """Boolean flag indicating if the sampler has been trained."""
        return self._runways is not None
    
    def is_runway_fitted(self, runway: str | List[str]) -> bool:
        """
        Check if a single runway or a list of runways are present in the sampler.
        """
        if isinstance(runway, str):
            return runway in self.runways
        return bool(runway) and all(r in self.runways for r in runway)
    
    def sample(self, runway: str, *args, **kwargs) -> Any:
        """
        Validate input and return a sampled value for the given runway.
        """
        if not self.is_runway_fitted(runway):
            raise ValueError(
                f"Sampler is not fitted for runway '{runway}'. "
                f"Available runways: {self.runways}"
            )
        return self._sample(runway, *args, **kwargs)
    
    @abstractmethod
    def _sample(self, runway: str, *args, **kwargs) -> Any:
        """Internal sampling logic to be implemented by subclasses."""
        pass

    def fit(self, data: Any, runways: List[str] | str, *args, **kwargs) -> None:
        """
        Wrapper to normalise runway inputs and execute fitting logic.
        """
        normalised_runways = [runways] if isinstance(runways, str) else runways
        self._runways = normalised_runways
        self._fit(data, self._runways, *args, **kwargs)

    @abstractmethod
    def _fit(self, data: Any, runways: List[str], *args, **kwargs) -> None:
        """Internal fitting logic to be implemented by subclasses."""
        pass

    def plot_distribution(self, runway: str | List[str] | None = None, n_points: int = 1000):
        if not self.is_fitted:
            raise RuntimeError("Sampler is not fitted yet. Call fit() first.")

        import numpy as np
        import matplotlib.pyplot as plt

        if isinstance(runway, str):
            runways = [runway]
        elif isinstance(runway, list):
            runways = runway
        else:
            runways = self.runways
            
        n = len(runways)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

        # Sample on a regular grid
        xs = np.linspace(-1.5, 1.5, int(np.sqrt(n_points)))
        ys = np.linspace(-1.5, 1.5, int(np.sqrt(n_points)))
        xx, yy = np.meshgrid(xs, ys)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

        for ax, rwy in zip(axes.flat, runways):
            rta_values, valid_x, valid_y = [], [], []

            for x, y in grid_points:
                try:
                    rta = self.sample(rwy, x=float(x), y=float(y))
                    rta_values.append(rta)
                    valid_x.append(x)
                    valid_y.append(y)
                except (ValueError, KeyError):
                    pass  # Outside convex hull or unfitted — skip

            if rta_values:
                sc = ax.scatter(valid_x, valid_y, c=rta_values, cmap="viridis", s=4, alpha=0.8)
                plt.colorbar(sc, ax=ax, label="RTA (s)")
            
            ax.set_title(f"Runway {rwy}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal")

        for ax in axes.flat[n:]:
            ax.set_visible(False)

        plt.suptitle("RTA distribution per runway", y=1.02)
        plt.tight_layout()
        plt.show()