from path_planning.rta.testing.samplers import *

from typing import List

def create_plots(samplers: List[str] | str, n_points: int = 10_000, *args, **kwargs):
    fit_and_plot(
        "path_planning/rta/data/rta_data_undeterministic.parquet",
        sampler_names = samplers if isinstance(samplers, list) else [samplers],
        n_points = n_points,
        *args,
        **kwargs
    )

def main():
    from path_planning.rta.testing.samplers.plot import PlotKind, CoordSystem
    samplers = ["RFDTGSampler"]  # was RFRTASampler

    coords = [CoordSystem.CARTESIAN, CoordSystem.POLAR_NORTH]
    sample_coords = CoordSystem.CARTESIAN
    plot_kind = [PlotKind.CONTOUR, PlotKind.SURFACE_3D]

    n_points = 500_000

    for coord in coords:
        for kind in plot_kind:
            create_plots(
                samplers, coord=coord, runways=["18R"], kind=kind,
                n_points=n_points, sample_coord=sample_coords,
            )

    coord = CoordSystem.POLAR_NORTH
    kind = PlotKind.CONTOUR
    save_path = f"path_planning/rta/plots/sampler/{coord.name.lower()}_{kind.name.lower()}.png"
    create_plots(
        samplers, coord=coord, kind=kind, n_points=n_points,
        sample_coord=sample_coords, save_path=save_path,
    )

if __name__ == "__main__":
    main()