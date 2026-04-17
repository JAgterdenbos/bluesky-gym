from __future__ import annotations

import numpy as np
import pandas as pd

from typing import List, Optional

from .plot import PlotKind, CoordSystem

def transform_coordinates(X: np.ndarray, coord: CoordSystem) -> np.ndarray:
    """
    Transforms an (N, 2) array of Cartesian coordinates (x, y) 
    into the specified coordinate system.
    
    Args:
        X: A numpy array of shape (N, 2) representing (x, y).
        coord: The target CoordSystem enum.
        
    Returns:
        A numpy array of shape (N, 2) with the transformed coordinates.
    """
    # If it's Cartesian, no math is needed. Return a copy to avoid in-place mutations.
    if coord == CoordSystem.CARTESIAN:
        return X.copy()
        
    x_val = X[:, 0]
    y_val = X[:, 1]
    
    # Radius is the same for both polar systems
    r = np.hypot(x_val, y_val)
    
    if coord == CoordSystem.POLAR or coord == CoordSystem.POLAR_NORTH:
        # Standard maths: East = 0°, Counter-Clockwise, polar_north is visual only!
        theta = np.arctan2(y_val, x_val)
    else:
        raise ValueError(f"Unknown coordinate system: {coord}")
        
    return np.column_stack((r, theta))

def prepare_grouped_data(data: pd.DataFrame, coord: CoordSystem):
    # Get the vectorised basics
    runway_categories = data["runway"].astype("category")
    runway_ids = runway_categories.cat.codes.values
    resolved_runways = runway_categories.cat.categories.tolist()

    X_raw = data[["x", "y"]].values
    X_full = transform_coordinates(X_raw, coord)
    y_full = (data["rta"] - data["t"]).values

    # Convert to List of Numpy Arrays
    # We use the unique integer codes to slice the data
    X_list = []
    y_list = []

    for i in range(len(resolved_runways)):
        # Create a boolean mask for the current runway ID
        mask = (runway_ids == i)
        
        # Append the subset of data to our lists
        X_list.append(X_full[mask])
        y_list.append(y_full[mask])

    return X_list, y_list, resolved_runways

def fit_and_plot(
    data_path: str,
    sampler_names: List[str],
    runways: Optional[List[str]] = None,
    n_points: int = 10_000,
    kind: PlotKind = PlotKind.CONTOUR,
    coord: CoordSystem = CoordSystem.POLAR,
    save_path: Optional[str] = None,
    *,
    sample_coord: Optional[CoordSystem] = None,
) -> None:
    """Fit one or more RTASamplers from collected data and plot their distributions."""
    from .registry import SamplerRegistry

    if sample_coord is None:
        sample_coord = coord

    # Load data
    if data_path.endswith(".parquet"):
        data = pd.read_parquet(data_path, engine="pyarrow") # type: ignore
    elif data_path.endswith(".csv"):
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported format: '{data_path}'. Use .csv or .parquet.")

    if runways is not None:
        data = data[data["runway"].isin(runways)]

    data["rta_remaining"] = data["rta"] - data["t"]


    # Prepare data
    X, y, resolved_runways = prepare_grouped_data(data, sample_coord)

    for sampler_name in sampler_names:
        print(f"\n🔧 Fitting '{sampler_name}'...")
        sampler = SamplerRegistry.make(sampler_name)
        sampler.fit(X, y, resolved_runways)

        print(f"📊 Plotting '{sampler_name}'...")
        sampler.plot_distribution(
            runways=resolved_runways,
            n_points=n_points,
            kind=kind,
            coord=coord,
            save_path=save_path,
            sample_coord=sample_coord,
        )


def run_fit_and_plot_cli(experiment_cls) -> None:
    """CLI entry point for fit_and_plot."""
    import argparse
    from .registry import SamplerRegistry

    p = argparse.ArgumentParser(
        description="Fit and plot RTA samplers from collected data."
    )
    p.add_argument("data_path", help="Path to data file (.csv or .parquet).")
    p.add_argument(
        "--samplers", nargs="+", required=True,
        metavar="SAMPLER",
        help=f"Sampler name(s). Available: {SamplerRegistry.list_available()}",
    )
    p.add_argument(
        "--runways", nargs="+", default=None,
        metavar="RUNWAY",
        help="Runways to use. Defaults to all runways in the data.",
    )
    p.add_argument(
        "--n-points", type=int, default=10_000,
        help="Grid resolution (~side^2 sample points). Default: 10 000.",
    )
    p.add_argument(
        "--kind", type=lambda s: PlotKind[s.upper()], default=PlotKind.CONTOUR,
        metavar="KIND",
        help=f"Plot style. Choices: {PlotKind.list_names()}. Default: CONTOUR.",
    )
    p.add_argument(
        "--coord", type=lambda s: CoordSystem[s.upper()], default=CoordSystem.POLAR,
        metavar="COORD",
        help=(
            f"Rendering coordinate system. Choices: {CoordSystem.list_names()}."
            "Default: POLAR."
        ),
    )
    p.add_argument(
        "--sample-coord", type=lambda s: CoordSystem[s.upper()], default=None,
        metavar="SAMPLE_COORD",
        help=(
            f"Coordinate system for sampling. Choices: {CoordSystem.list_names()}."
            "Default: same as --coord. Note: This does NOT affect rendering orientation."
        ),
    )
    p.add_argument("--out", default=None, metavar="PATH", help="Save figure to file.")

    args = p.parse_args()
    fit_and_plot(
        data_path=args.data_path,
        sampler_names=args.samplers,
        runways=args.runways,
        n_points=args.n_points,
        kind=args.kind,
        coord=args.coord,
        save_path=args.out,
        sample_coord=args.sample_coord,
    )