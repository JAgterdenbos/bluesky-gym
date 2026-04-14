"""
Axis renderers for RTA distribution plots.

Each renderer knows how to draw one subplot given a pre-computed grid and mesh.
Adding a new plot style means adding a new class — the orchestrator in figure.py
never needs to change.

Public API
----------
AxisRenderer    - structural Protocol (duck-typed; no inheritance required)
CartesianContourRenderer  - 2-D filled-contour on Cartesian axes
PolarHeatmapRenderer      - pcolormesh on a polar axes
Surface3DRenderer         - 3-D surface on a 3-D axes

CoordSystem controls axis labelling / orientation only, never sampling.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)
from typing import Protocol, runtime_checkable, Any

from .mesh import PlotMesh
from .enums import CoordSystem, PlotKind


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AxisRenderer(Protocol):
    """
    Renders a single subplot for one runway.

    Parameters
    ----------
    fig    : parent Figure (needed for add_subplot)
    index  : 1-based subplot index
    nrows  : total subplot row count
    ncols  : total subplot col count
    mesh   : PlotMesh — carries R, Theta, X, Y and physical bounds
    grid   : (side, side) float array of sampled time-to-go values
    vmin   : global colour-scale minimum across all runways
    vmax   : global colour-scale maximum across all runways

    Returns the matplotlib Mappable (for the shared colorbar).
    """

    def add_subplot(self, fig: Figure, index: int, nrows: int, ncols: int) -> Axes: ...

    def render(
        self,
        ax: Axes,
        mesh: PlotMesh,
        grid: np.ndarray,
        vmin: float,
        vmax: float,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Cartesian filled-contour renderer
# ---------------------------------------------------------------------------

class CartesianContourRenderer(AxisRenderer):
    """
    2-D filled contour plot on standard Cartesian axes.

    Uses contourf directly on the polar mesh's (X, Y) arrays — no regridding,
    no imshow corner artefacts and the inner annulus gap is preserved naturally.
    """

    _LEVELS = 15
    _CONTOUR_LINE_LEVELS = 8

    def __init__(self, coord: CoordSystem = CoordSystem.CARTESIAN) -> None:
        # coord is accepted for API symmetry but has no effect here — Cartesian
        # axes always show x / y.
        self._coord = coord

    def add_subplot(self, fig: Figure, index: int, nrows: int, ncols: int) -> Axes:
        return fig.add_subplot(nrows, ncols, index)

    def render(
        self,
        ax: Axes,
        mesh: PlotMesh,
        grid: np.ndarray,
        vmin: float,
        vmax: float,
    ):
        im = ax.contourf(
            mesh.X, mesh.Y, grid,
            levels=self._LEVELS,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        # Crisp isolines on top
        ax.contour(
            mesh.X, mesh.Y, grid,
            levels=self._CONTOUR_LINE_LEVELS,
            colors="white",
            alpha=0.35,
            linewidths=0.6,
        )
        ax.set_xlabel("Relative X (normalised)")
        ax.set_ylabel("Relative Y (normalised)")
        ax.set_aspect("equal")
        ax.grid(color="white", alpha=0.1)
        return im


# ---------------------------------------------------------------------------
# Polar contour renderer
# ---------------------------------------------------------------------------

class PolarContourRenderer(AxisRenderer):
    """
    For POLAR_NORTH the axes are rotated so North = 0° and angles increase CW,
    matching aviation / ATC convention. For standard POLAR, East = 0° CCW.
    """

    _CONTOUR_LINE_LEVELS = 8

    def __init__(self, coord: CoordSystem = CoordSystem.POLAR) -> None:
        if coord == CoordSystem.CARTESIAN:
            raise ValueError("PolarHeatmapRenderer requires a polar CoordSystem.")
        self._coord = coord

    def add_subplot(self, fig: Figure, index: int, nrows: int, ncols: int) -> Axes:
        return fig.add_subplot(nrows, ncols, index, projection="polar")

    def render(
        self,
        ax: Axes,
        mesh: PlotMesh,
        grid: np.ndarray,
        vmin: float,
        vmax: float,
    ):
        plot_theta = mesh.Theta

        if self._coord == CoordSystem.POLAR_NORTH:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)

            plot_theta = np.pi / 2 - plot_theta  # Transform mathematical Theta (0=East, CCW) to navigational (0=North, CW)

        # Adjust the axis limits to the inner annulus
        ax.set_rorigin(0)
        ax.set_ylim(mesh.r_min, mesh.r_max)

        # Using contourf instead of pcolormesh
        # Note: levels can be an integer or a specific array of values
        im = ax.contourf(
            plot_theta, mesh.R, grid,
            levels=np.linspace(vmin, vmax, 20),
            cmap="viridis",
            extend="both" # Handles values outside vmin/vmax gracefully
        )
        
        # Overlay contours for depth
        ax.contour(
            plot_theta, mesh.R, grid,
            levels=self._CONTOUR_LINE_LEVELS,
            colors="white",
            alpha=0.35,
            linewidths=0.6,
        )
        
        ax.grid(color="white", alpha=0.4, linestyle="--")
        # Ensure the polar plot doesn't show the "pie slice" if data is 360 deg
        ax.set_aspect("equal") 
        
        return im


# ---------------------------------------------------------------------------
# 3-D surface renderer
# ---------------------------------------------------------------------------

class Surface3DRenderer(AxisRenderer):
    """
    3-D surface plot.

    Always plots against (X, Y) so the spatial layout matches Cartesian.
    For polar CoordSystems the floor grid is drawn as concentric rings + spokes
    rather than the default Cartesian panes.
    """

    def __init__(self, coord: CoordSystem = CoordSystem.CARTESIAN) -> None:
        self._coord = coord

    def add_subplot(self, fig: Figure, index: int, nrows: int, ncols: int) -> Axes:
        return fig.add_subplot(nrows, ncols, index, projection="3d")

    def render(
        self,
        ax: Axes,
        mesh: PlotMesh,
        grid: np.ndarray,
        vmin: float,
        vmax: float,
    ):
        surf = ax.plot_surface(
            mesh.X, mesh.Y, grid,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            antialiased=True,
            alpha=0.9,
            linewidth=0,
            shade=True,
        )

        # Floor shadow
        try:
            ax.contourf(
                mesh.X, mesh.Y, grid,
                zdir="z",
                offset=vmin,
                cmap="viridis",
                alpha=0.3,
                levels=15,
            )
        except Exception:
            pass

        ax.set_zlabel("RTA Remaining", labelpad=10) # TODO: move to the left
        ax.set_zlim(vmin, vmax)
        ax.set_box_aspect((1, 1, 0.6))
        ax.view_init(elev=25, azim=-55)

        if self._coord == CoordSystem.CARTESIAN:
            self._style_cartesian_panes(ax)
        else:
            self._draw_cylindrical_floor(ax, mesh, vmin, vmax)

        return surf

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _style_cartesian_panes(ax: Axes) -> None:
        ax.set_xlabel("Relative X", labelpad=10)
        ax.set_ylabel("Relative Y", labelpad=10)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            pane.set_edgecolor("gainsboro")
        ax.grid(color="gainsboro", linestyle=":", linewidth=0.5)

    def _draw_cylindrical_floor(
        self, ax: Axes, mesh: PlotMesh, vmin: float, vmax: float
    ) -> None:
        """Replace default Cartesian panes with a polar floor grid."""
        ax.set_axis_off()
        limit = mesh.r_max

        # Custom Z-axis in the corner
        z_x, z_y = -limit * 1.25, -limit * 1.25
        ax.plot([z_x, z_x], [z_y, z_y], [vmin, vmax], color="gray", linewidth=1.2)
        for z in np.linspace(vmin, vmax, 4):
            ax.plot(
                [z_x, z_x + limit * 0.05], [z_y, z_y - limit * 0.05], [z, z],
                color="gray", linewidth=1.2,
            )
            ax.text(
                z_x - limit * 0.08, z_y - limit * 0.08, z,
                f"{z:.1f}", color="dimgray", va="center", ha="right", fontsize=8,
            )
        ax.text(
            z_x, z_y, vmax + (vmax - vmin) * 0.1,
            "RTA Remaining", color="black", ha="center", fontsize=10, fontweight="bold",
        )

        # Concentric rings
        circle_th = np.linspace(0, 2 * np.pi, 100)
        for r in np.linspace(limit / 4, limit, 4):
            ax.plot(
                r * np.cos(circle_th), r * np.sin(circle_th), vmin,
                color="gray", linestyle="--", linewidth=0.6, alpha=0.5,
            )

        # Spokes + angular labels
        is_north = self._coord == CoordSystem.POLAR_NORTH
        spoke_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        labels = ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]
        for angle, label in zip(spoke_angles, labels):
            sx = limit * (np.sin(angle) if is_north else np.cos(angle))
            sy = limit * (np.cos(angle) if is_north else np.sin(angle))
            ax.plot([0, sx], [0, sy], vmin, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
            ax.text(sx * 1.18, sy * 1.18, vmin, label, ha="center", va="center",
                    color="dimgray", fontsize=9, fontweight="bold")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_renderer(kind: PlotKind, coord: CoordSystem) -> AxisRenderer:  # noqa: F821
    """
    Convenience factory — returns the right renderer for a (kind, coord) pair.

    Importing PlotKind here would be circular; callers can also construct
    renderers directly.
    """
    from .enums import PlotKind  # local import to avoid circular dependency

    if kind == PlotKind.SURFACE_3D:
        return Surface3DRenderer(coord)
    if coord == CoordSystem.CARTESIAN:
        return CartesianContourRenderer(coord)
    return PolarContourRenderer(coord)