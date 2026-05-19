"""
plot_dtg_distribution — figure-level orchestrator.

Responsibilities:
  1. Build the mesh (once, shared across all runways).
  2. Call sample_fn(runway, X) for each runway — always
     Cartesian kwargs, always the same mesh regardless of CoordSystem.
  3. Compute a shared colour scale across all runways.
  4. Delegate every rendering decision to the AxisRenderer.
  5. Add a shared colorbar and save / show.

Nothing in here branches on PlotKind or CoordSystem — that logic lives
entirely in the renderer chosen by the caller.

Target quantity: dist_to_go = total_dist_km - path_len  (km)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Optional

from .enums import PlotKind, CoordSystem
from .mesh import PlotMesh
from .renderers import AxisRenderer, make_renderer

def plot_dtg_distribution(
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
    t: Optional[float] = None,
) -> None:
    """
    Plot the spatial distance-to-go (DTG) distribution for one or more runways.

    DTG is defined as dist_to_go = total_dist_km − path_len  (km).
    sample_fn must return DTG values in km with shape (N,).
    The array passed to sample_fn is either (N, 2) [coord1, coord2] when
    t is None (independent mode), or (N, 3) [coord1, coord2, t] when t
    is supplied (dependent mode).

    Parameters
    ----------
    sample_fn  : callable(runway, X) → ndarray
                   independent mode (t=None) : X is (N, 2) [coord1, coord2]
                   dependent   mode (t given): X is (N, 3) [coord1, coord2, t]
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
    t          : normalised time fixed across the spatial grid, or None.
                   None → independent mode: X passed as (N, 2), no t column.
                   0.0  → spawn / initial distribution
                   0.5  → mid-flight snapshot
                   1.0  → end-of-episode
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
    # 2. Sample every runway                                              #
    #    t=None  → independent mode: pass (N, 2) spatial coords only     #
    #    t given → dependent   mode: append fixed t column → (N, 3)      #
    # ------------------------------------------------------------------ #
    if t is not None and not (0.0 <= t <= 1.0):
        raise ValueError(f"t must be in [0, 1], got {t}.")

    grids: dict[str, np.ndarray] = {}
    global_min, global_max = float("inf"), float("-inf")

    for rwy in runways:
        try:
            spatial = mesh.get_X(sample_coord)                        # (N, 2)
            if t is None:
                X_query = spatial                                      # (N, 2) independent
            else:
                t_col   = np.full((spatial.shape[0], 1), t)           # (N, 1)
                X_query = np.concatenate([spatial, t_col], axis=1)    # (N, 3) dependent
            grid = np.asarray(sample_fn(rwy, X_query), dtype=float)
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
        fig.colorbar(last_mappable, cax=cbar_ax, label="Distance to Go (km)")

    plt.suptitle(
        f"Spatial Distance-to-Go Distribution ({coord.name})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()