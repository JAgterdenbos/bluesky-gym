"""
Shared enumerations for the plot package.

Kept in a dedicated module so that mesh.py, renderers.py, and figure.py can
all import from here without creating circular dependencies.
"""

from __future__ import annotations

from enum import Enum
from typing import List


class PlotKind(Enum):
    """Visual representation of the distribution."""
    CONTOUR   = "contour"     # 2-D filled contour
    SURFACE_3D = "surface_3d"

    @classmethod
    def list_names(cls) -> List[str]:
        return [k.name for k in cls]

class CoordSystem(Enum):
    """
    Axis orientation used by the renderer.
    This is a rendering concern, not a data concern.
    """
    CARTESIAN   = "cartesian"
    POLAR       = "polar"        # East = 0°, CCW  (standard maths convention)
    POLAR_NORTH = "polar_north"  # North = 0°, CW  (aviation / ATC convention)

    @classmethod
    def list_names(cls) -> List[str]:
        return [c.name for c in cls]