import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from .rta_sampler import RTASampler


class LinearRTASampler(RTASampler, name="LinearRTASampler"):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolators = {}

    def _fit(self, data: pd.DataFrame, runways: list[str]):
        cols = ["x", "y"]

        for runway in runways:
            group = data[data["runway"] == runway]
            if group.empty:
                print(f"Warning: no data for runway '{runway}', skipping.")
                continue
            try:
                self._interpolators[runway] = LinearNDInterpolator(
                    group[cols].values,
                    group["rta"].values
                )
            except Exception as e:
                print(f"Warning: could not fit interpolator for '{runway}': {e}")

        print(f"✅ Fitted {len(self._interpolators)} interpolators.")

    def _sample(self, runway: str, x: float, y: float) -> float:
        interp = self._interpolators.get(runway)
        if interp is None:
            raise KeyError(f"No interpolator found for runway '{runway}'")

        point = [[x, y]]
        result = interp(point)

        if np.isnan(result):
            raise ValueError(f"Interpolation returned NaN for runway '{runway}' at {point} — point may be outside convex hull.")

        return float(result[0])