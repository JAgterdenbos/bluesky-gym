"""
plot — RTA distribution visualisation package.

Public API
----------
PlotKind              - CONTOUR | SURFACE_3D
CoordSystem           - CARTESIAN | POLAR | POLAR_NORTH  (rendering only)
PlotMesh              - polar mesh projected to (x, y)
AxisRenderer          - Protocol for custom renderers
CartesianContourRenderer
PolarHeatmapRenderer
Surface3DRenderer
make_renderer         - factory(kind, coord) → AxisRenderer
plot_rta_distribution - top-level figure function
"""

from .enums import PlotKind, CoordSystem
from .mesh import PlotMesh
from .renderers import (
    AxisRenderer,
    CartesianContourRenderer,
    PolarContourRenderer,
    Surface3DRenderer,
    make_renderer,
)
from .figure import plot_rta_distribution

__all__ = [
    "PlotKind",
    "CoordSystem",
    "PlotMesh",
    "AxisRenderer",
    "CartesianContourRenderer",
    "PolarContourRenderer",
    "Surface3DRenderer",
    "make_renderer",
    "plot_rta_distribution",
]