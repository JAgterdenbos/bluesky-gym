"""
plot_rta_distribution — figure-level orchestrator.

Responsibilities:
  1. Build the mesh (once, shared across all runways).
  2. Call sample_fn(runway, X) for each runway — always
     Cartesian kwargs, always the same mesh regardless of CoordSystem.
  3. Compute a shared colour scale across all runways.
  4. Delegate every rendering decision to the AxisRenderer.
  5. Add a shared colorbar and save / show.

Nothing in here branches on PlotKind or CoordSystem — that logic lives
entirely in the renderer chosen by the caller.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional

from .enums import PlotKind, CoordSystem
from .mesh import PlotMesh
from .renderers import AxisRenderer, make_renderer


def plot_rta_distribution(
    sample_fn: Callable[..., np.ndarray],
    runways: List[str] | str,
    n_points: int = 10_000,
    kind: PlotKind = PlotKind.CONTOUR,
    coord: CoordSystem = CoordSystem.POLAR,
    save_path: Optional[str] = None,
    *,
    renderer: Optional[AxisRenderer] = None,
    sample_coord: Optional[CoordSystem] = None,
    r_min_m: float = 55.0,
    r_max_m: float = 300.0,
) -> None:
    """
    Plot the spatial time-to-go distribution for one or more runways.

    Parameters
    ----------
    sample_fn  : callable(runway, *, X) → ndarray — returns n-dimensional
    runways    : runway identifier(s) to plot.
    n_points   : approximate grid resolution (side² actual points).
    kind       : CONTOUR or SURFACE_3D.
    coord      : axis orientation for the rendered plot — does NOT affect sampling.
                   CARTESIAN   → standard x/y axes
                   POLAR       → polar axes, East = 0°, CCW
                   POLAR_NORTH → polar axes, North = 0°, CW  (aviation convention)
    save_path  : if given, saves the figure instead of showing it.
    renderer   : optional custom AxisRenderer; if None, one is built from
                 (kind, coord) via make_renderer().
    r_min_m    : inner radius of sampling annulus in metres.
    r_max_m    : outer radius of sampling annulus in metres.
    """
    runways = [runways] if isinstance(runways, str) else list(runways)
    if not runways:
        raise ValueError("At least one runway must be specified.")

    renderer = renderer or make_renderer(kind, coord)

    if sample_coord is None:
        sample_coord = coord

    # ------------------------------------------------------------------ #
    # 1. Mesh — built once, shared across all runways                     #
    # ------------------------------------------------------------------ #
    mesh = PlotMesh.build(n_points=n_points, r_min_m=r_min_m, r_max_m=r_max_m)

    # ------------------------------------------------------------------ #
    # 2. Sample every runway — always kwargs                      #
    # ------------------------------------------------------------------ #
    grids: dict[str, np.ndarray] = {}
    global_min, global_max = float("inf"), float("-inf")

    for rwy in runways:
        try:
            grid = np.asarray(sample_fn(rwy, mesh.get_X(sample_coord)), dtype=float)
            grid = np.reshape(grid, mesh.R.shape)
        except (ValueError, KeyError) as exc:
            print(f"  ⚠️  Runway '{rwy}' skipped: {exc}")
            grid = np.full(mesh.X.shape, np.nan)

        finite = grid[np.isfinite(grid)]
        if finite.size:
            global_min = min(global_min, float(finite.min()))
            global_max = max(global_max, float(finite.max()))

        grids[rwy] = grid

    if not np.isfinite(global_min):
        raise RuntimeError("All runways returned only NaN — nothing to plot.")        

    # ------------------------------------------------------------------ #
    # 3. Figure layout                                                    #
    # ------------------------------------------------------------------ #
    n = len(runways)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows))

    # ------------------------------------------------------------------ #
    # 4. Render each subplot — fully delegated to the renderer            #
    # ------------------------------------------------------------------ #
    last_mappable = None
    for i, rwy in enumerate(runways):
        ax = renderer.add_subplot(fig, i + 1, nrows, ncols)
        mappable = renderer.render(ax, mesh, grids[rwy], global_min, global_max)
        ax.set_title(f"Runway {rwy}", fontweight="bold", pad=20)
        last_mappable = mappable

    # ------------------------------------------------------------------ #
    # 5. Shared colorbar + title                                          #
    # ------------------------------------------------------------------ #
    fig.tight_layout(rect=(0, 0.03, 0.9, 0.95))
    if last_mappable is not None:
        cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
        fig.colorbar(last_mappable, cax=cbar_ax, label="RTA Remaining")

    plt.suptitle(
        f"Spatial RTA Remaining Distribution ({coord.name})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()