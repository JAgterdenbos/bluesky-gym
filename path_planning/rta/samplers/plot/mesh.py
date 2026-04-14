import numpy as np
from dataclasses import dataclass

from .enums import CoordSystem

# Physical range of the sampling annulus in km.
_DEFAULT_R_MIN_M: float = 55.0
_DEFAULT_R_MAX_M: float = 300.0

@dataclass(frozen=True)
class PlotMesh:
    """
    A polar mesh for plotting.

    The mesh is ALWAYS built in (r, theta) space.
    This is the only representation passed to samplers — coordinate system
    choice is purely a rendering concern and never leaks into sampling.

    Attributes
    ----------
    R       : (side, side) normalised radii  in [r_min, r_max]
    Theta   : (side, side) angles            in [0, 2π)
    r_min   : normalised inner radius (r_min_m / r_max_m)
    r_max   : normalised outer radius (always 1.0)
    r_min_m : inner radius in metres
    r_max_m : outer radius in metres
    """

    R: np.ndarray
    Theta: np.ndarray
    r_min: float
    r_max: float
    r_min_m: float
    r_max_m: float

    @classmethod
    def build(
        cls,
        n_points: int = 10_000,
        r_min_m: float = _DEFAULT_R_MIN_M,
        r_max_m: float = _DEFAULT_R_MAX_M,
    ) -> "PlotMesh":
        """
        Build a uniform polar mesh with approximately `n_points` sample points.

        Parameters
        ----------
        n_points : desired total number of grid points (side² actual)
        r_min_m  : inner radius of the sampling annulus in metres
        r_max_m  : outer radius of the sampling annulus in metres
        """
        if r_min_m <= 0 or r_max_m <= r_min_m:
            raise ValueError(f"Require 0 < r_min_m < r_max_m, got {r_min_m}, {r_max_m}.")

        side = max(2, int(np.sqrt(n_points)))
        r_min = r_min_m / r_max_m   # normalise so outer edge = 1.0
        r_max = 1.0

        rs = np.linspace(r_min, r_max, side)
        thetas = np.linspace(0.0, 2.0 * np.pi, side)
        R, Theta = np.meshgrid(rs, thetas)

        return cls(
            R=R, Theta=Theta,
            r_min=r_min, r_max=r_max,
            r_min_m=r_min_m, r_max_m=r_max_m,
        )
    
    @property
    def X(self) -> np.ndarray:
        return self.R * np.cos(self.Theta) # East = 0°
    
    @property
    def Y(self) -> np.ndarray:
        return self.R * np.sin(self.Theta) 
    
    @property
    def grid_shape(self) -> tuple[int, int]:
        """Returns the (rows, cols) shape of the mesh."""
        return self.R.shape
    
    def reshape_to_grid(self, flat_data: np.ndarray) -> np.ndarray:
        """Converts a flat (N,) sampler output back to (side, side)."""
        return flat_data.reshape(self.grid_shape)
    
    def get_X(self, coord: CoordSystem) -> np.ndarray: # should return (N, 2) but it doesn't
        match coord:
            case CoordSystem.POLAR:
                return np.column_stack([self.R.ravel(), self.Theta.ravel()])
            case CoordSystem.POLAR_NORTH:
                return np.column_stack([self.R.ravel(), self.Theta.ravel()]) # Note: Theta is still mathematically defined (East=0°, CCW), but the renderer will interpret it as North=0°, CW.
            case CoordSystem.CARTESIAN:
                return np.column_stack([self.X.ravel(), self.Y.ravel()])
            case _:
                raise ValueError(f"Unrecognised coordinate system: {coord}")
            
            