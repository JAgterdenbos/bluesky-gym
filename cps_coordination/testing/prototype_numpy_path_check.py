"""
cps_coordination/testing/prototype_numpy_path_check.py
--------------------------------------------------------
PROTOTYPE / VALIDATION HARNESS -- not wired into the pipeline. Answers a
follow-up to the wall-clock investigation in
`.claude/plans/we-re-about-to-launch-delegated-hare.md`: after caching the
`SINK{rwy}`/`RESTRICT{rwy}` `matplotlib.path.Path` objects (see
`multi_agent_pathplanning_env.py::_check_terminal`/`_get_shape_path`), the
remaining ~4.2s/18% of a post-fix M=5 profiling run was
`matplotlib._path.path_intersects_path` itself, called up to ~24x per
aircraft-tick (target sink/restrict + each non-overlapping runway's
sink/restrict).

What `_check_terminal` actually needs, geometrically: does the aircraft's
tiny 2-point movement segment this tick cross any segment of a *static*
open polyline (`SINK{rwy}` = 36-point arc, `RESTRICT{rwy}` = 3-point line,
both built once in `_set_terminal_conditions`)? That's a textbook
segment-vs-many-segments test, vectorizable with numpy cross products
instead of one matplotlib C call per shape.

This file:
  1. Implements that vectorized numpy check (`segment_intersects_polyline`).
  2. Validates it against matplotlib's own `Path.intersects_path` across
     random cases AND deliberately adversarial edge cases (touching
     endpoints, collinear overlap, near-tangent arcs) -- boolean agreement,
     not just headline-metric parity, since a silent mismatch here would
     mean a wrong `death_cause` near a boundary, not a crash.
  3. Benchmarks + cProfile-profiles matplotlib vs numpy on synthetic
     geometry structurally identical to production (36-point arc, 3-point
     line; same call pattern as `_check_terminal`: one fixed static
     polyline, many different tiny query segments).

FINDING (read this before trusting the validation PASS/FAIL below):
`Path.intersects_path(other, filled=True)` -- `filled=True` is the default,
and it is what `_check_terminal` actually calls (no `filled=` kwarg passed
there either) -- returns True not just when the two paths' segments cross,
but ALSO when one path is entirely enclosed by the other (paths treated as
implicitly-closed filled regions). `segment_intersects_polyline` below only
implements plain segment-vs-segment crossing. Validated below: it agrees
with matplotlib 100% when compared against `filled=False`, and disagrees
~4% of the time against the real `filled=True` default on random small
segments near the shape. That ~4% is not a bug in the numpy math -- it's a
missing feature (enclosure/point-in-polygon), which is a materially bigger,
riskier thing to reimplement correctly than segment intersection alone.
Combined with the benchmark result (see bottom of this file), the verdict
is: not worth pursuing further as a per-shape replacement.

Deliberately self-contained (no bluesky_gym/bluesky imports) so it runs in
under a second without booting BlueSky -- the geometry is synthetic but
matches production's point counts and rough shape (see `make_sink_arc`/
`make_restrict_line`), which is what determines both correctness-edge-case
coverage and per-call cost, not the specific runway coordinates.

Run: python cps_coordination/testing/prototype_numpy_path_check.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from typing import List, Tuple

import numpy as np
from matplotlib.path import Path


# --------------------------------------------------------------------- #
# The candidate replacement
# --------------------------------------------------------------------- #

def segment_intersects_polyline(
    p1: np.ndarray, p2: np.ndarray, poly: np.ndarray
) -> bool:
    """Does segment (p1, p2) cross ANY edge of the open polyline `poly`
    (shape (N, 2))? Vectorized numpy equivalent of
    ``Path(np.array([p1, p2])).intersects_path(Path(poly))`` for the
    straight-line (no curves, no closepoly) case `_check_terminal` uses.

    Standard segment-intersection test (orientation via cross products,
    with the collinear/touching-endpoint cases handled explicitly), batched
    across all N-1 edges of `poly` at once instead of looping.
    """
    a = poly[:-1]   # (N-1, 2) edge starts
    b = poly[1:]    # (N-1, 2) edge ends
    p1b = np.broadcast_to(p1, a.shape)
    p2b = np.broadcast_to(p2, a.shape)

    def cross(o, u, v):
        return (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - \
               (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0])

    d1 = cross(a, b, p1b)
    d2 = cross(a, b, p2b)
    d3 = cross(p1b, p2b, a)
    d4 = cross(p1b, p2b, b)

    proper = (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))) & \
             (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))

    def on_segment(u, v, w):
        return (
            (np.minimum(u[..., 0], v[..., 0]) <= w[..., 0]) &
            (w[..., 0] <= np.maximum(u[..., 0], v[..., 0])) &
            (np.minimum(u[..., 1], v[..., 1]) <= w[..., 1]) &
            (w[..., 1] <= np.maximum(u[..., 1], v[..., 1]))
        )

    touching = (
        ((d1 == 0) & on_segment(a, b, p1b)) |
        ((d2 == 0) & on_segment(a, b, p2b)) |
        ((d3 == 0) & on_segment(p1b, p2b, a)) |
        ((d4 == 0) & on_segment(p1b, p2b, b))
    )

    return bool(np.any(proper | touching))


# --------------------------------------------------------------------- #
# Synthetic geometry matching production's shape/point-count profile
# (see multi_agent_pathplanning_env.py::_set_terminal_conditions)
# --------------------------------------------------------------------- #

def make_sink_arc(num_points: int = 36, radius: float = 30.0,
                   center: Tuple[float, float] = (0.0, 0.0),
                   span_deg: float = 60.0) -> np.ndarray:
    """36-point arc, same point count/rough shape as a real SINK{rwy}."""
    angles = np.linspace(-span_deg / 2, span_deg / 2, num_points)
    x = center[0] + radius * np.sin(np.radians(angles))
    y = center[1] + radius * np.cos(np.radians(angles))
    return np.column_stack([x, y])


def make_restrict_line(arc: np.ndarray, apex: Tuple[float, float] = (0.0, 25.0)) -> np.ndarray:
    """3-point line (arc start -> apex -> arc end), same shape as a real
    RESTRICT{rwy} (POLYLINE with the arc's two endpoints + the FAF point)."""
    return np.array([arc[0], apex, arc[-1]])


# --------------------------------------------------------------------- #
# Validation: boolean agreement against matplotlib, not just timing
# --------------------------------------------------------------------- #

def matplotlib_check(p1: np.ndarray, p2: np.ndarray, poly: np.ndarray, filled: bool) -> bool:
    return Path(poly).intersects_path(Path(np.array([p1, p2])), filled=filled)


def run_validation(n_random: int = 20000, seed: int = 0) -> None:
    """Checks numpy agreement against BOTH matplotlib semantics: `filled=False`
    (plain segment crossing -- what `segment_intersects_polyline` implements)
    and `filled=True` (the actual default `_check_terminal` relies on, which
    also fires on enclosure). Reporting both, rather than just one PASS/FAIL,
    is the point -- collapsing this to a single number would hide exactly the
    semantics gap this prototype exists to surface.
    """
    rng = np.random.default_rng(seed)
    arc = make_sink_arc()
    restrict = make_restrict_line(arc)

    mismatches_unfilled = []
    mismatches_filled = []
    n_checked = 0

    for poly, label in [(arc, "SINK-like arc"), (restrict, "RESTRICT-like line")]:
        lo = poly.min(axis=0) - 5.0
        hi = poly.max(axis=0) + 5.0

        # Random small segments (mimics a real aircraft-tick movement: two
        # nearby points, not an arbitrary long line) scattered around the
        # shape's bounding box, biased toward actually crossing it.
        for _ in range(n_random):
            center = rng.uniform(lo, hi)
            step = rng.normal(scale=2.0, size=2)
            p1 = center
            p2 = center + step
            got_np = segment_intersects_polyline(p1, p2, poly)
            got_unfilled = matplotlib_check(p1, p2, poly, filled=False)
            got_filled = matplotlib_check(p1, p2, poly, filled=True)
            n_checked += 1
            if got_np != got_unfilled:
                mismatches_unfilled.append((label, p1.copy(), p2.copy(), got_np, got_unfilled))
            if got_np != got_filled:
                mismatches_filled.append((label, p1.copy(), p2.copy(), got_np, got_filled))

        # Deliberately adversarial edge cases: touching an endpoint exactly,
        # lying exactly on an edge (collinear), grazing a vertex.
        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            mid = (a + b) / 2
            edge_cases = [
                (a, b),                       # exact overlap with the edge
                (a, mid),                     # collinear partial overlap
                (a - (b - a) * 0.01, a),      # touches endpoint `a` exactly
                (mid, mid + np.array([0.0, 0.0])),  # degenerate zero-length segment
            ]
            for p1, p2 in edge_cases:
                got_np = segment_intersects_polyline(p1, p2, poly)
                got_unfilled = matplotlib_check(p1, p2, poly, filled=False)
                got_filled = matplotlib_check(p1, p2, poly, filled=True)
                n_checked += 1
                tag = f"{label} (edge case @ segment {i})"
                if got_np != got_unfilled:
                    mismatches_unfilled.append((tag, p1, p2, got_np, got_unfilled))
                if got_np != got_filled:
                    mismatches_filled.append((tag, p1, p2, got_np, got_filled))

    print(f"=== Validation: {n_checked} cases checked ===")

    print(f"\nvs. matplotlib filled=False (plain segment crossing):")
    if mismatches_unfilled:
        print(f"  MISMATCHES: {len(mismatches_unfilled)}/{n_checked}")
        for label, p1, p2, got_np, got_mpl in mismatches_unfilled[:5]:
            print(f"    [{label}] p1={p1} p2={p2}  numpy={got_np} matplotlib={got_mpl}")
    else:
        print("  PASS: 100% agreement (including adversarial touching/collinear/"
              "degenerate cases) -- segment_intersects_polyline is a correct, exact "
              "replacement for Path.intersects_path(..., filled=False).")

    print(f"\nvs. matplotlib filled=True (the ACTUAL default _check_terminal calls):")
    if mismatches_filled:
        print(f"  MISMATCHES: {len(mismatches_filled)}/{n_checked} "
              f"({100*len(mismatches_filled)/n_checked:.1f}%) -- NOT a numpy math bug, "
              f"see the module docstring: filled=True also matches enclosure, which "
              f"segment_intersects_polyline does not implement.")
    else:
        print("  PASS (unexpected given filled=True's documented semantics -- "
              "re-check if this ever prints for these shapes).")


# --------------------------------------------------------------------- #
# REAL batching: the same query segment vs ALL ~24 shapes _check_terminal
# checks per tick (target sink/restrict + each non-overlapping runway's
# sink/restrict), combined into ONE numpy call instead of 24 separate
# matplotlib calls. This -- not the one-shape-at-a-time version above --
# is where vectorization could plausibly amortize numpy's per-call
# dispatch overhead across many checks instead of paying it once per tiny
# array. Correctness of the underlying cross-product math already
# validated above; this only tests whether concatenating shapes changes
# the *speed* picture, and whether per-shape reduction is still correct.
# --------------------------------------------------------------------- #

def make_all_runway_shapes(n_runways: int = 12, seed: int = 2) -> List[np.ndarray]:
    """12 distinct SINK arcs + 12 RESTRICT lines (24 shapes total) --
    mimics the up-to-24 shapes checked per aircraft-tick in dynamic mode
    with ~11 non-overlapping runways. Rotated/offset per runway so they
    aren't literal duplicates."""
    rng = np.random.default_rng(seed)
    shapes = []
    for _ in range(n_runways):
        center = rng.uniform(-80, 80, size=2)
        rotation = np.radians(rng.uniform(0, 360))
        rot = np.array([[np.cos(rotation), -np.sin(rotation)],
                         [np.sin(rotation), np.cos(rotation)]])
        arc = make_sink_arc(center=(0.0, 0.0))
        arc = arc @ rot.T + center
        restrict = make_restrict_line(arc, apex=tuple(center + rng.uniform(-5, 5, size=2)))
        shapes.append(arc)
        shapes.append(restrict)
    return shapes


def precompute_shape_batch(shapes: List[np.ndarray]):
    """One-time (cacheable, like _get_shape_path) flattening of all shapes'
    edges into contiguous arrays plus a group_id per edge, so a query can
    be tested against all of them in a single vectorized call."""
    starts, ends, group_ids = [], [], []
    for gi, poly in enumerate(shapes):
        starts.append(poly[:-1])
        ends.append(poly[1:])
        group_ids.append(np.full(len(poly) - 1, gi, dtype=np.int64))
    return (np.concatenate(starts, axis=0), np.concatenate(ends, axis=0),
            np.concatenate(group_ids, axis=0), len(shapes))


def batched_hits(p1: np.ndarray, p2: np.ndarray, edge_starts: np.ndarray,
                  edge_ends: np.ndarray, group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """Same cross-product test as segment_intersects_polyline, but over ALL
    shapes' edges concatenated -- one call, then reduced per shape group."""
    a, b = edge_starts, edge_ends
    p1b = np.broadcast_to(p1, a.shape)
    p2b = np.broadcast_to(p2, a.shape)

    def cross(o, u, v):
        return (u[..., 0] - o[..., 0]) * (v[..., 1] - o[..., 1]) - \
               (u[..., 1] - o[..., 1]) * (v[..., 0] - o[..., 0])

    d1 = cross(a, b, p1b)
    d2 = cross(a, b, p2b)
    d3 = cross(p1b, p2b, a)
    d4 = cross(p1b, p2b, b)

    proper = (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))) & \
             (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))

    def on_segment(u, v, w):
        return (
            (np.minimum(u[..., 0], v[..., 0]) <= w[..., 0]) &
            (w[..., 0] <= np.maximum(u[..., 0], v[..., 0])) &
            (np.minimum(u[..., 1], v[..., 1]) <= w[..., 1]) &
            (w[..., 1] <= np.maximum(u[..., 1], v[..., 1]))
        )

    touching = (
        ((d1 == 0) & on_segment(a, b, p1b)) |
        ((d2 == 0) & on_segment(a, b, p2b)) |
        ((d3 == 0) & on_segment(p1b, p2b, a)) |
        ((d4 == 0) & on_segment(p1b, p2b, b))
    )
    edge_hits = proper | touching
    return np.bincount(group_ids, weights=edge_hits.astype(np.int64), minlength=n_groups) > 0


def precompute_shape_batch_with_boundaries(shapes: List[np.ndarray]):
    """Like precompute_shape_batch, but also returns each group's starting
    edge index -- needed for np.logical_or.reduceat in the multi-agent
    batched check below (edges are contiguous per shape by construction)."""
    edge_starts, edge_ends, group_ids, n_groups = precompute_shape_batch(shapes)
    edge_counts = np.array([len(poly) - 1 for poly in shapes])
    boundaries = np.concatenate([[0], np.cumsum(edge_counts)[:-1]])
    return edge_starts, edge_ends, boundaries, n_groups


def batched_hits_multi_agent(p1s: np.ndarray, p2s: np.ndarray, edge_starts: np.ndarray,
                              edge_ends: np.ndarray, boundaries: np.ndarray,
                              n_groups: int) -> np.ndarray:
    """Multiple aircraft (p1s/p2s: (n_agents, 2) each) x all shapes' edges,
    in ONE call -- (n_agents, n_edges) broadcast, then reduced per shape
    group via np.logical_or.reduceat. Returns (n_agents, n_groups) bool."""
    ax, ay = edge_starts[None, :, 0], edge_starts[None, :, 1]     # (1, n_edges)
    bx, by = edge_ends[None, :, 0], edge_ends[None, :, 1]         # (1, n_edges)
    p1x, p1y = p1s[:, None, 0], p1s[:, None, 1]                   # (n_agents, 1)
    p2x, p2y = p2s[:, None, 0], p2s[:, None, 1]                   # (n_agents, 1)

    def cross(ox, oy, ux, uy, vx, vy):
        return (ux - ox) * (vy - oy) - (uy - oy) * (vx - ox)

    d1 = cross(ax, ay, bx, by, p1x, p1y)   # (n_agents, n_edges)
    d2 = cross(ax, ay, bx, by, p2x, p2y)
    d3 = cross(p1x, p1y, p2x, p2y, ax, ay)
    d4 = cross(p1x, p1y, p2x, p2y, bx, by)

    proper = (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))) & \
             (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))

    def on_segment(ux, uy, vx, vy, wx, wy):
        return (np.minimum(ux, vx) <= wx) & (wx <= np.maximum(ux, vx)) & \
               (np.minimum(uy, vy) <= wy) & (wy <= np.maximum(uy, vy))

    touching = (
        ((d1 == 0) & on_segment(ax, ay, bx, by, p1x, p1y)) |
        ((d2 == 0) & on_segment(ax, ay, bx, by, p2x, p2y)) |
        ((d3 == 0) & on_segment(p1x, p1y, p2x, p2y, ax, ay)) |
        ((d4 == 0) & on_segment(p1x, p1y, p2x, p2y, bx, by))
    )
    edge_hits = proper | touching  # (n_agents, n_edges)
    return np.logical_or.reduceat(edge_hits, boundaries, axis=1)  # (n_agents, n_groups)


def run_multi_agent_equivalence_check(n_agents: int = 5, n_random: int = 2000, seed: int = 5) -> None:
    shapes = make_all_runway_shapes()
    edge_starts, edge_ends, boundaries, n_groups = precompute_shape_batch_with_boundaries(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)

    n_mismatched = 0
    for _ in range(n_random):
        centers = rng.uniform(lo, hi, size=(n_agents, 2))
        steps = rng.normal(scale=2.0, size=(n_agents, 2))
        p1s, p2s = centers, centers + steps

        batched = batched_hits_multi_agent(p1s, p2s, edge_starts, edge_ends, boundaries, n_groups)
        looped = np.array([
            [segment_intersects_polyline(p1s[a], p2s[a], poly) for poly in shapes]
            for a in range(n_agents)
        ])
        if not np.array_equal(batched, looped):
            n_mismatched += 1

    print(f"\n=== Multi-agent batched vs. looped agreement: {n_random} ticks, "
          f"{n_agents} aircraft x {n_groups} shapes ===")
    if n_mismatched:
        print(f"  MISMATCHES: {n_mismatched}/{n_random} ticks disagreed -- bug, "
              f"investigate before trusting speed numbers below.")
    else:
        print("  PASS: multi-agent batched_hits agrees with per-aircraft-per-shape "
              "segment_intersects_polyline on every tick tested.")


def _mpl_multi_agent_loop(p1s: np.ndarray, p2s: np.ndarray, shape_paths: List[Path],
                           n_agents: int, n_ticks: int) -> None:
    """Today's real per-tick cost: for each of n_agents active aircraft,
    call _check_terminal, which checks each of the (up to 24) shapes in
    turn via a separate matplotlib call."""
    for t in range(n_ticks):
        for a in range(n_agents):
            line_ac = Path(np.array([p1s[t, a], p2s[t, a]]))
            for sp in shape_paths:
                sp.intersects_path(line_ac)


def _np_multi_agent_loop(p1s: np.ndarray, p2s: np.ndarray, edge_starts: np.ndarray,
                          edge_ends: np.ndarray, boundaries: np.ndarray, n_groups: int,
                          n_ticks: int) -> None:
    for t in range(n_ticks):
        batched_hits_multi_agent(p1s[t], p2s[t], edge_starts, edge_ends, boundaries, n_groups)


def run_multi_agent_benchmark(n_agents: int = 5, n_ticks: int = 50_000, seed: int = 6):
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    edge_starts, edge_ends, boundaries, n_groups = precompute_shape_batch_with_boundaries(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_ticks, n_agents, 2))
    steps = rng.normal(scale=2.0, size=(n_ticks, n_agents, 2))
    p1s, p2s = centers, centers + steps

    t0 = time.perf_counter()
    _mpl_multi_agent_loop(p1s, p2s, shape_paths, n_agents, n_ticks)
    t_mpl = time.perf_counter() - t0

    t0 = time.perf_counter()
    _np_multi_agent_loop(p1s, p2s, edge_starts, edge_ends, boundaries, n_groups, n_ticks)
    t_np = time.perf_counter() - t0

    print(f"\n=== Multi-agent benchmark: {n_ticks} ticks, {n_agents} aircraft/tick, "
          f"{n_groups} shapes/aircraft ({n_agents * n_groups} checks/tick) ===")
    print(f"  matplotlib: {n_agents}x{n_groups}={n_agents*n_groups} separate calls/tick : "
          f"{t_mpl:.3f}s ({t_mpl/n_ticks*1e6:.2f}us/tick)")
    print(f"  numpy: 1 batched call/tick (all agents x all shapes)      : "
          f"{t_np:.3f}s ({t_np/n_ticks*1e6:.2f}us/tick)")
    print(f"  speedup: {t_mpl/t_np:.2f}x  (>1.0 = numpy faster, <1.0 = numpy slower)")
    return t_mpl, t_np


def run_multi_agent_profiling(n_agents: int = 5, n_ticks: int = 5_000, seed: int = 6) -> None:
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    edge_starts, edge_ends, boundaries, n_groups = precompute_shape_batch_with_boundaries(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_ticks, n_agents, 2))
    steps = rng.normal(scale=2.0, size=(n_ticks, n_agents, 2))
    p1s, p2s = centers, centers + steps

    for label, fn, args in [
        ("matplotlib, n_agents x n_shapes calls/tick", _mpl_multi_agent_loop,
         (p1s, p2s, shape_paths, n_agents, n_ticks)),
        ("numpy, 1 batched call/tick (all agents)", _np_multi_agent_loop,
         (p1s, p2s, edge_starts, edge_ends, boundaries, n_groups, n_ticks)),
    ]:
        profiler = cProfile.Profile()
        profiler.enable()
        fn(*args)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(12)
        print(f"\n=== cProfile: {label} ({n_ticks} ticks) ===")
        print(stream.getvalue())


def run_batched_equivalence_check(n_random: int = 5000, seed: int = 3) -> None:
    """Per-shape agreement between the batched call and looping
    segment_intersects_polyline shape-by-shape -- confirms the group_ids
    reduction didn't silently merge/misattribute hits across shapes."""
    shapes = make_all_runway_shapes()
    edge_starts, edge_ends, group_ids, n_groups = precompute_shape_batch(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)

    n_checked = 0
    n_mismatched_rows = 0
    for _ in range(n_random):
        center = rng.uniform(lo, hi)
        step = rng.normal(scale=2.0, size=2)
        p1, p2 = center, center + step

        batched = batched_hits(p1, p2, edge_starts, edge_ends, group_ids, n_groups)
        looped = np.array([segment_intersects_polyline(p1, p2, poly) for poly in shapes])
        n_checked += 1
        if not np.array_equal(batched, looped):
            n_mismatched_rows += 1

    print(f"\n=== Batched vs. looped per-shape agreement: {n_checked} query segments, "
          f"{n_groups} shapes each ===")
    if n_mismatched_rows:
        print(f"  MISMATCHES: {n_mismatched_rows}/{n_checked} query segments had a "
              f"differing per-shape hit pattern -- batching logic bug, investigate before trusting speed numbers below.")
    else:
        print("  PASS: batched_hits agrees with per-shape segment_intersects_polyline "
              "on every query segment tested.")


def _mpl_batch_loop(p1s: np.ndarray, p2s: np.ndarray, shape_paths: List[Path], n_calls: int) -> None:
    for i in range(n_calls):
        for sp in shape_paths:
            sp.intersects_path(Path(np.array([p1s[i], p2s[i]])))


def _np_batch_loop(p1s: np.ndarray, p2s: np.ndarray, edge_starts: np.ndarray,
                    edge_ends: np.ndarray, group_ids: np.ndarray, n_groups: int, n_calls: int) -> None:
    for i in range(n_calls):
        batched_hits(p1s[i], p2s[i], edge_starts, edge_ends, group_ids, n_groups)


def run_batched_benchmark(n_calls: int = 100_000, seed: int = 4):
    """The comparison that actually matters: 24 matplotlib calls (today's
    real per-tick cost) vs. 1 batched numpy call, both over the same set
    of query segments."""
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]  # cached, as _get_shape_path already does
    edge_starts, edge_ends, group_ids, n_groups = precompute_shape_batch(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_calls, 2))
    steps = rng.normal(scale=2.0, size=(n_calls, 2))
    p1s, p2s = centers, centers + steps

    t0 = time.perf_counter()
    _mpl_batch_loop(p1s, p2s, shape_paths, n_calls)
    t_mpl = time.perf_counter() - t0

    t0 = time.perf_counter()
    _np_batch_loop(p1s, p2s, edge_starts, edge_ends, group_ids, n_groups, n_calls)
    t_np = time.perf_counter() - t0

    print(f"\n=== Batched benchmark: {n_calls} aircraft-ticks, {n_groups} shapes/tick "
          f"({len(edge_starts)} total edges) ===")
    print(f"  matplotlib: {n_groups} separate .intersects_path() calls/tick : "
          f"{t_mpl:.3f}s ({t_mpl/n_calls*1e6:.2f}us/tick)")
    print(f"  numpy: 1 batched call/tick                                  : "
          f"{t_np:.3f}s ({t_np/n_calls*1e6:.2f}us/tick)")
    print(f"  speedup: {t_mpl/t_np:.2f}x  (>1.0 = numpy faster, <1.0 = numpy slower)")
    return t_mpl, t_np


def run_batched_profiling(n_calls: int = 20_000, seed: int = 4) -> None:
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    edge_starts, edge_ends, group_ids, n_groups = precompute_shape_batch(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_calls, 2))
    steps = rng.normal(scale=2.0, size=(n_calls, 2))
    p1s, p2s = centers, centers + steps

    for label, fn, args in [
        ("matplotlib, 24 calls/tick", _mpl_batch_loop, (p1s, p2s, shape_paths, n_calls)),
        ("numpy, 1 batched call/tick", _np_batch_loop,
         (p1s, p2s, edge_starts, edge_ends, group_ids, n_groups, n_calls)),
    ]:
        profiler = cProfile.Profile()
        profiler.enable()
        fn(*args)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(12)
        print(f"\n=== cProfile: {label} ({n_calls} ticks) ===")
        print(stream.getvalue())


# --------------------------------------------------------------------- #
# Benchmark: same call pattern as _check_terminal (one static polyline,
# many small per-tick query segments)
# --------------------------------------------------------------------- #

def _mpl_loop(p1s: np.ndarray, p2s: np.ndarray, arc_path: Path, n_calls: int) -> None:
    for i in range(n_calls):
        Path(np.array([p1s[i], p2s[i]])).intersects_path(arc_path)


def _np_loop(p1s: np.ndarray, p2s: np.ndarray, arc: np.ndarray, n_calls: int) -> None:
    for i in range(n_calls):
        segment_intersects_polyline(p1s[i], p2s[i], arc)


def run_benchmark(n_calls: int = 200_000, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    arc = make_sink_arc()
    arc_path = Path(arc)  # cached once, as _get_shape_path already does

    lo = arc.min(axis=0) - 5.0
    hi = arc.max(axis=0) + 5.0
    centers = rng.uniform(lo, hi, size=(n_calls, 2))
    steps = rng.normal(scale=2.0, size=(n_calls, 2))
    p1s = centers
    p2s = centers + steps

    t0 = time.perf_counter()
    _mpl_loop(p1s, p2s, arc_path, n_calls)
    t_mpl = time.perf_counter() - t0

    t0 = time.perf_counter()
    _np_loop(p1s, p2s, arc, n_calls)
    t_np = time.perf_counter() - t0

    print(f"\n=== Benchmark: {n_calls} calls, one static 36-point arc, "
          f"random 2-point query segments ===")
    print(f"  matplotlib Path.intersects_path : {t_mpl:.3f}s ({t_mpl/n_calls*1e6:.2f}us/call)")
    print(f"  numpy segment_intersects_polyline: {t_np:.3f}s ({t_np/n_calls*1e6:.2f}us/call)")
    print(f"  speedup: {t_mpl/t_np:.2f}x  (>1.0 = numpy faster, <1.0 = numpy slower)")
    return t_mpl, t_np


def run_profiling(n_calls: int = 100_000, seed: int = 1) -> None:
    """cProfile breakdown of each implementation's own call, so a slowdown
    (if any) can be attributed to a specific operation rather than just a
    single aggregate number."""
    rng = np.random.default_rng(seed)
    arc = make_sink_arc()
    arc_path = Path(arc)

    lo = arc.min(axis=0) - 5.0
    hi = arc.max(axis=0) + 5.0
    centers = rng.uniform(lo, hi, size=(n_calls, 2))
    steps = rng.normal(scale=2.0, size=(n_calls, 2))
    p1s, p2s = centers, centers + steps

    for label, fn, args in [
        ("matplotlib Path.intersects_path", _mpl_loop, (p1s, p2s, arc_path, n_calls)),
        ("numpy segment_intersects_polyline", _np_loop, (p1s, p2s, arc, n_calls)),
    ]:
        profiler = cProfile.Profile()
        profiler.enable()
        fn(*args)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(12)
        print(f"\n=== cProfile: {label} ({n_calls} calls) ===")
        print(stream.getvalue())


# --------------------------------------------------------------------- #
# PART 4: filling the gaps. matplotlib's actual default is filled=True,
# empirically reverse-engineered here (not guessed from docs) as:
#
#   filled_hit(query, poly) = segment_crosses(query, poly)
#                              OR (point_in_polygon(p1, poly)
#                                  AND point_in_polygon(p2, poly))
#
# i.e. the query segment is also a hit if BOTH its endpoints lie inside
# the shape's implicitly-closed (wraparound) interior -- not just one
# (verified: 0/20000 mismatches for both SINK-arc-like and RESTRICT-line-
# like shapes, isolating the exact false-positive/false-negative directions
# first rather than accepting a plausible-looking guess). All prior PART
# 1-3 numpy functions only implemented the segment_crosses() term, which is
# why they were faster than they should have been relative to matplotlib's
# real (filled=True) cost -- this section adds the missing term and
# re-measures speed honestly with it included.
# --------------------------------------------------------------------- #

def point_in_polygon_single(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-cast even-odd point-in-polygon test, poly implicitly closed
    (wraparound edge poly[-1]->poly[0] included). Reference implementation,
    used to derive/validate the vectorized batched versions below."""
    x, y = pt
    xi, yi = poly[:, 0], poly[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
    crossings = ((yi > y) != (yj > y)) & (x < x_intersect)
    return bool(np.sum(crossings) % 2 == 1)


def segment_intersects_polyline_filled(p1: np.ndarray, p2: np.ndarray, poly: np.ndarray) -> bool:
    """CORRECT single-shape replacement for
    ``Path(poly).intersects_path(Path([p1, p2]), filled=True)`` -- what
    _check_terminal actually calls. Segment crossing OR full enclosure."""
    return (
        segment_intersects_polyline(p1, p2, poly)
        or (point_in_polygon_single(p1, poly) and point_in_polygon_single(p2, poly))
    )


def run_filled_validation(n_random: int = 50_000, seed: int = 7) -> None:
    """Validates segment_intersects_polyline_filled against the REAL
    matplotlib default (filled=True), including the same adversarial
    edge cases used in Part 1's validation."""
    rng = np.random.default_rng(seed)
    arc = make_sink_arc()
    restrict = make_restrict_line(arc)

    mismatches = []
    n_checked = 0
    for poly, label in [(arc, "SINK-like arc"), (restrict, "RESTRICT-like line")]:
        lo, hi = poly.min(axis=0) - 5.0, poly.max(axis=0) + 5.0
        for _ in range(n_random):
            center = rng.uniform(lo, hi)
            step = rng.normal(scale=2.0, size=2)
            p1, p2 = center, center + step
            got = segment_intersects_polyline_filled(p1, p2, poly)
            want = matplotlib_check(p1, p2, poly, filled=True)
            n_checked += 1
            if got != want:
                mismatches.append((label, p1.copy(), p2.copy(), got, want))

        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            mid = (a + b) / 2
            for p1, p2 in [(a, b), (a, mid), (a - (b - a) * 0.01, a),
                           (mid, mid + np.array([0.0, 0.0]))]:
                got = segment_intersects_polyline_filled(p1, p2, poly)
                want = matplotlib_check(p1, p2, poly, filled=True)
                n_checked += 1
                if got != want:
                    mismatches.append((f"{label} (edge case @ {i})", p1, p2, got, want))

    print(f"=== Filled-semantics validation: {n_checked} cases vs. REAL matplotlib default (filled=True) ===")
    if mismatches:
        print(f"  MISMATCHES: {len(mismatches)}/{n_checked}")
        for label, p1, p2, got, want in mismatches[:10]:
            print(f"    [{label}] p1={p1} p2={p2}  numpy={got} matplotlib={want}")
    else:
        print("  PASS: 100% agreement with the REAL matplotlib default "
              "(segment crossing OR both-endpoints-enclosed).")


def precompute_shape_batch_closing(shapes: List[np.ndarray]):
    """Closing (wraparound-included, N edges per shape, not N-1) edge
    arrays for the point-in-polygon term, parallel to
    precompute_shape_batch_with_boundaries's open-edge arrays."""
    starts, ends, group_ids = [], [], []
    for gi, poly in enumerate(shapes):
        starts.append(poly)
        ends.append(np.roll(poly, -1, axis=0))
        group_ids.append(np.full(len(poly), gi, dtype=np.int64))
    edge_starts = np.concatenate(starts, axis=0)
    edge_ends = np.concatenate(ends, axis=0)
    group_ids_arr = np.concatenate(group_ids, axis=0)
    edge_counts = np.array([len(poly) for poly in shapes])
    boundaries = np.concatenate([[0], np.cumsum(edge_counts)[:-1]])
    return edge_starts, edge_ends, boundaries, len(shapes)


def _point_in_polygon_crossings_multi_agent(pts: np.ndarray, close_a: np.ndarray,
                                             close_b: np.ndarray) -> np.ndarray:
    """pts: (n_agents, 2). Returns (n_agents, n_closing_edges) ray-crossing
    booleans, for reduceat-summing per shape group afterward."""
    x, y = pts[:, 0:1], pts[:, 1:2]                       # (n_agents, 1)
    xi, yi = close_a[None, :, 0], close_a[None, :, 1]      # (1, n_edges)
    xj, yj = close_b[None, :, 0], close_b[None, :, 1]      # (1, n_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
    return ((yi > y) != (yj > y)) & (x < x_intersect)      # (n_agents, n_edges)


def filled_hits_multi_agent(
    p1s: np.ndarray, p2s: np.ndarray,
    open_a: np.ndarray, open_b: np.ndarray, open_boundaries: np.ndarray,
    close_a: np.ndarray, close_b: np.ndarray, close_boundaries: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """CORRECT batched multi-agent replacement for matplotlib's real
    default (filled=True): segment crossing (open edges) OR both-endpoints-
    enclosed (closing edges), all agents x all shapes in one shot."""
    seg_hits = batched_hits_multi_agent(p1s, p2s, open_a, open_b, open_boundaries, n_groups)

    cross_p1 = _point_in_polygon_crossings_multi_agent(p1s, close_a, close_b)
    cross_p2 = _point_in_polygon_crossings_multi_agent(p2s, close_a, close_b)
    count_p1 = np.add.reduceat(cross_p1.astype(np.int64), close_boundaries, axis=1)
    count_p2 = np.add.reduceat(cross_p2.astype(np.int64), close_boundaries, axis=1)
    p1_in = (count_p1 % 2) == 1
    p2_in = (count_p2 % 2) == 1

    return seg_hits | (p1_in & p2_in)


def pick_first_by_priority(hit_matrix: np.ndarray) -> np.ndarray:
    """Given (n_agents, n_groups) hits, ordered by priority (group 0 =
    target_sink, 1 = target_restrict, 2.. = other runways' sink/restrict --
    the same order _check_terminal's early-return checks today), return
    per-agent the first hit group index, or -1 if none. Mirrors
    _check_terminal's short-circuit priority (sink > restrict > wrong_runway)
    without actually short-circuiting -- needed because the batched call
    computes every shape's hit regardless, so picking which one 'wins' has
    to be a separate, explicit step to preserve today's death_cause semantics."""
    any_hit = hit_matrix.any(axis=1)
    first_idx = np.argmax(hit_matrix, axis=1)  # first True per row; 0 if none
    return np.where(any_hit, first_idx, -1)


def run_filled_batched_equivalence_check(n_agents: int = 5, n_random: int = 2000, seed: int = 8) -> None:
    shapes = make_all_runway_shapes()
    open_a, open_b, open_bounds, n_groups = precompute_shape_batch_with_boundaries(shapes)
    close_a, close_b, close_bounds, _ = precompute_shape_batch_closing(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)

    n_mismatched = 0
    for _ in range(n_random):
        centers = rng.uniform(lo, hi, size=(n_agents, 2))
        steps = rng.normal(scale=2.0, size=(n_agents, 2))
        p1s, p2s = centers, centers + steps

        batched = filled_hits_multi_agent(p1s, p2s, open_a, open_b, open_bounds,
                                           close_a, close_b, close_bounds, n_groups)
        looped = np.array([
            [segment_intersects_polyline_filled(p1s[a], p2s[a], poly) for poly in shapes]
            for a in range(n_agents)
        ])
        if not np.array_equal(batched, looped):
            n_mismatched += 1

    print(f"\n=== CORRECT batched (filled=True) vs. looped-correct agreement: "
          f"{n_random} ticks, {n_agents} aircraft x {n_groups} shapes ===")
    if n_mismatched:
        print(f"  MISMATCHES: {n_mismatched}/{n_random} -- investigate before trusting speed numbers.")
    else:
        print("  PASS: correct batched filled_hits_multi_agent agrees with the "
              "looped per-shape filled-correct function on every tick tested.")


def run_filled_batched_vs_matplotlib_check(n_agents: int = 5, n_random: int = 2000, seed: int = 9) -> None:
    """The check that actually matters: does the batched numpy result match
    the REAL matplotlib default (filled=True), not just our own looped
    reference? Closes the loop back to ground truth."""
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    open_a, open_b, open_bounds, n_groups = precompute_shape_batch_with_boundaries(shapes)
    close_a, close_b, close_bounds, _ = precompute_shape_batch_closing(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)

    n_mismatched = 0
    for _ in range(n_random):
        centers = rng.uniform(lo, hi, size=(n_agents, 2))
        steps = rng.normal(scale=2.0, size=(n_agents, 2))
        p1s, p2s = centers, centers + steps

        batched = filled_hits_multi_agent(p1s, p2s, open_a, open_b, open_bounds,
                                           close_a, close_b, close_bounds, n_groups)
        mpl = np.array([
            [sp.intersects_path(Path(np.array([p1s[a], p2s[a]]))) for sp in shape_paths]
            for a in range(n_agents)
        ])
        if not np.array_equal(batched, mpl):
            n_mismatched += 1

    print(f"\n=== CORRECT batched vs. REAL matplotlib (ground truth): "
          f"{n_random} ticks, {n_agents} aircraft x {n_groups} shapes ===")
    if n_mismatched:
        print(f"  MISMATCHES: {n_mismatched}/{n_random}")
    else:
        print("  PASS: matches real matplotlib.path.Path.intersects_path (default "
              "filled=True) exactly, batched across agents and shapes.")


def _mpl_multi_agent_loop_filled(p1s: np.ndarray, p2s: np.ndarray, shape_paths: List[Path],
                                  n_agents: int, n_ticks: int) -> None:
    """Same as _mpl_multi_agent_loop -- kept as a separate name for clarity
    in profiling output (matplotlib's default already IS filled=True, so
    this is identical code, just labeled for the Part 4 comparison)."""
    for t in range(n_ticks):
        for a in range(n_agents):
            line_ac = Path(np.array([p1s[t, a], p2s[t, a]]))
            for sp in shape_paths:
                sp.intersects_path(line_ac)


def _np_filled_multi_agent_loop(p1s: np.ndarray, p2s: np.ndarray, open_a, open_b, open_bounds,
                                 close_a, close_b, close_bounds, n_groups: int, n_ticks: int) -> None:
    for t in range(n_ticks):
        filled_hits_multi_agent(p1s[t], p2s[t], open_a, open_b, open_bounds,
                                 close_a, close_b, close_bounds, n_groups)


def run_filled_multi_agent_benchmark(n_agents: int = 5, n_ticks: int = 50_000, seed: int = 10):
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    open_a, open_b, open_bounds, n_groups = precompute_shape_batch_with_boundaries(shapes)
    close_a, close_b, close_bounds, _ = precompute_shape_batch_closing(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_ticks, n_agents, 2))
    steps = rng.normal(scale=2.0, size=(n_ticks, n_agents, 2))
    p1s, p2s = centers, centers + steps

    t0 = time.perf_counter()
    _mpl_multi_agent_loop_filled(p1s, p2s, shape_paths, n_agents, n_ticks)
    t_mpl = time.perf_counter() - t0

    t0 = time.perf_counter()
    _np_filled_multi_agent_loop(p1s, p2s, open_a, open_b, open_bounds,
                                 close_a, close_b, close_bounds, n_groups, n_ticks)
    t_np = time.perf_counter() - t0

    print(f"\n=== CORRECT (filled=True) multi-agent benchmark: {n_ticks} ticks, "
          f"{n_agents} aircraft/tick, {n_groups} shapes/aircraft ===")
    print(f"  matplotlib (real default, {n_agents}x{n_groups}={n_agents*n_groups} calls/tick): "
          f"{t_mpl:.3f}s ({t_mpl/n_ticks*1e6:.2f}us/tick)")
    print(f"  numpy, 1 batched call/tick, segment+enclosure         : "
          f"{t_np:.3f}s ({t_np/n_ticks*1e6:.2f}us/tick)")
    print(f"  speedup: {t_mpl/t_np:.2f}x  (>1.0 = numpy faster, <1.0 = numpy slower)")
    return t_mpl, t_np


def run_filled_multi_agent_profiling(n_agents: int = 5, n_ticks: int = 5_000, seed: int = 10) -> None:
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    open_a, open_b, open_bounds, n_groups = precompute_shape_batch_with_boundaries(shapes)
    close_a, close_b, close_bounds, _ = precompute_shape_batch_closing(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_ticks, n_agents, 2))
    steps = rng.normal(scale=2.0, size=(n_ticks, n_agents, 2))
    p1s, p2s = centers, centers + steps

    for label, fn, args in [
        ("matplotlib (real default filled=True)", _mpl_multi_agent_loop_filled,
         (p1s, p2s, shape_paths, n_agents, n_ticks)),
        ("numpy, batched, segment+enclosure", _np_filled_multi_agent_loop,
         (p1s, p2s, open_a, open_b, open_bounds, close_a, close_b, close_bounds, n_groups, n_ticks)),
    ]:
        profiler = cProfile.Profile()
        profiler.enable()
        fn(*args)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(14)
        print(f"\n=== cProfile: {label} ({n_ticks} ticks) ===")
        print(stream.getvalue())


# --------------------------------------------------------------------- #
# PART 5: further-optimized version of Part 4's CORRECT batched check.
# Part 4's own cProfile showed three concrete, fixable costs: (1) cross/
# on_segment were closures redefined on every single call; (2) point-in-
# polygon ran as two separate calls (once for p1s, once for p2s), each
# paying its own np.errstate context-manager overhead; (3) edge x/y
# components were re-sliced from the same static arrays on every tick
# instead of once at precompute time. All three are fixed here with no
# change in the underlying math -- re-validated against the same ground
# truth (real matplotlib) from scratch, not assumed carried over from Part 4.
# --------------------------------------------------------------------- #

def _cross_batch(ox, oy, ux, uy, vx, vy):
    return (ux - ox) * (vy - oy) - (uy - oy) * (vx - ox)


def _on_segment_batch(ux, uy, vx, vy, wx, wy):
    return (np.minimum(ux, vx) <= wx) & (wx <= np.maximum(ux, vx)) & \
           (np.minimum(uy, vy) <= wy) & (wy <= np.maximum(uy, vy))


def precompute_shape_batch_fast(shapes: List[np.ndarray]):
    """Like precompute_shape_batch_with_boundaries, but also pre-splits
    edge x/y components once -- avoids re-slicing the same static arrays
    on every per-tick call."""
    edge_starts, edge_ends, boundaries, n_groups = precompute_shape_batch_with_boundaries(shapes)
    ax, ay = edge_starts[None, :, 0], edge_starts[None, :, 1]
    bx, by = edge_ends[None, :, 0], edge_ends[None, :, 1]
    return ax, ay, bx, by, boundaries, n_groups


def precompute_shape_batch_closing_fast(shapes: List[np.ndarray]):
    edge_starts, edge_ends, boundaries, n_groups = precompute_shape_batch_closing(shapes)
    ax, ay = edge_starts[None, :, 0], edge_starts[None, :, 1]
    bx, by = edge_ends[None, :, 0], edge_ends[None, :, 1]
    return ax, ay, bx, by, boundaries, n_groups


def batched_hits_multi_agent_fast(p1s: np.ndarray, p2s: np.ndarray, ax, ay, bx, by,
                                   boundaries: np.ndarray, n_groups: int) -> np.ndarray:
    """Same result as batched_hits_multi_agent -- module-level cross/
    on_segment (no per-call closure creation) and pre-split ax/ay/bx/by
    (no per-call re-slicing) are the only differences."""
    p1x, p1y = p1s[:, None, 0], p1s[:, None, 1]
    p2x, p2y = p2s[:, None, 0], p2s[:, None, 1]

    d1 = _cross_batch(ax, ay, bx, by, p1x, p1y)
    d2 = _cross_batch(ax, ay, bx, by, p2x, p2y)
    d3 = _cross_batch(p1x, p1y, p2x, p2y, ax, ay)
    d4 = _cross_batch(p1x, p1y, p2x, p2y, bx, by)

    proper = (((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0))) & \
             (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0)))

    touching = (
        ((d1 == 0) & _on_segment_batch(ax, ay, bx, by, p1x, p1y)) |
        ((d2 == 0) & _on_segment_batch(ax, ay, bx, by, p2x, p2y)) |
        ((d3 == 0) & _on_segment_batch(p1x, p1y, p2x, p2y, ax, ay)) |
        ((d4 == 0) & _on_segment_batch(p1x, p1y, p2x, p2y, bx, by))
    )
    edge_hits = proper | touching
    return np.logical_or.reduceat(edge_hits, boundaries, axis=1)


def _point_in_polygon_crossings_fast(pts: np.ndarray, ax, ay, bx, by) -> np.ndarray:
    """pts: (n_pts, 2) -- may be p1s and p2s STACKED together (see
    filled_hits_multi_agent_fast) so one call does the work of two."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_intersect = (bx - ax) * (y - ay) / (by - ay) + ax
    return ((ay > y) != (by > y)) & (x < x_intersect)


def filled_hits_multi_agent_fast(
    p1s: np.ndarray, p2s: np.ndarray,
    open_ax, open_ay, open_bx, open_by, open_boundaries: np.ndarray,
    close_ax, close_ay, close_bx, close_by, close_boundaries: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Same result as filled_hits_multi_agent, optimized: p1s/p2s are
    stacked into one (2*n_agents, 2) array so the point-in-polygon ray-cast
    (and its np.errstate context) runs ONCE per tick instead of twice."""
    seg_hits = batched_hits_multi_agent_fast(p1s, p2s, open_ax, open_ay, open_bx, open_by,
                                              open_boundaries, n_groups)

    n_agents = p1s.shape[0]
    stacked = np.concatenate([p1s, p2s], axis=0)  # (2*n_agents, 2)
    crossings = _point_in_polygon_crossings_fast(stacked, close_ax, close_ay, close_bx, close_by)
    counts = np.add.reduceat(crossings.astype(np.int64), close_boundaries, axis=1)
    inside = (counts % 2) == 1  # (2*n_agents, n_groups)
    p1_in, p2_in = inside[:n_agents], inside[n_agents:]

    return seg_hits | (p1_in & p2_in)


def run_fast_equivalence_check(n_agents: int = 5, n_random: int = 2000, seed: int = 12) -> None:
    """Confirms the Part 5 optimizations didn't change the result -- checked
    against Part 4's own (already matplotlib-validated) filled_hits_multi_agent,
    not re-derived from scratch, since Part 4 is already ground-truthed."""
    shapes = make_all_runway_shapes()
    open_a, open_b, open_bounds, n_groups = precompute_shape_batch_with_boundaries(shapes)
    close_a, close_b, close_bounds, _ = precompute_shape_batch_closing(shapes)
    open_ax, open_ay, open_bx, open_by, _, _ = precompute_shape_batch_fast(shapes)
    close_ax, close_ay, close_bx, close_by, _, _ = precompute_shape_batch_closing_fast(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)

    n_mismatched = 0
    for _ in range(n_random):
        centers = rng.uniform(lo, hi, size=(n_agents, 2))
        steps = rng.normal(scale=2.0, size=(n_agents, 2))
        p1s, p2s = centers, centers + steps

        slow = filled_hits_multi_agent(p1s, p2s, open_a, open_b, open_bounds,
                                        close_a, close_b, close_bounds, n_groups)
        fast = filled_hits_multi_agent_fast(p1s, p2s, open_ax, open_ay, open_bx, open_by, open_bounds,
                                             close_ax, close_ay, close_bx, close_by, close_bounds, n_groups)
        if not np.array_equal(slow, fast):
            n_mismatched += 1

    print(f"\n=== Part 5 (optimized) vs. Part 4 (ground-truthed) agreement: "
          f"{n_random} ticks, {n_agents} aircraft x {n_groups} shapes ===")
    if n_mismatched:
        print(f"  MISMATCHES: {n_mismatched}/{n_random} -- optimization changed the result, do not trust speed numbers.")
    else:
        print("  PASS: Part 5's optimized implementation is bit-identical to Part 4's "
              "already matplotlib-validated implementation on every tick tested.")


def run_fast_vs_matplotlib_check(n_agents: int = 5, n_random: int = 2000, seed: int = 13) -> None:
    """Direct re-check against real matplotlib (not just against Part 4) --
    closes the loop from scratch for the version that would actually ship."""
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    open_ax, open_ay, open_bx, open_by, open_bounds, n_groups = precompute_shape_batch_fast(shapes)
    close_ax, close_ay, close_bx, close_by, close_bounds, _ = precompute_shape_batch_closing_fast(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)

    n_mismatched = 0
    for _ in range(n_random):
        centers = rng.uniform(lo, hi, size=(n_agents, 2))
        steps = rng.normal(scale=2.0, size=(n_agents, 2))
        p1s, p2s = centers, centers + steps

        fast = filled_hits_multi_agent_fast(p1s, p2s, open_ax, open_ay, open_bx, open_by, open_bounds,
                                             close_ax, close_ay, close_bx, close_by, close_bounds, n_groups)
        mpl = np.array([
            [sp.intersects_path(Path(np.array([p1s[a], p2s[a]]))) for sp in shape_paths]
            for a in range(n_agents)
        ])
        if not np.array_equal(fast, mpl):
            n_mismatched += 1

    print(f"\n=== Part 5 (optimized) vs. REAL matplotlib (ground truth, checked from scratch): "
          f"{n_random} ticks, {n_agents} aircraft x {n_groups} shapes ===")
    if n_mismatched:
        print(f"  MISMATCHES: {n_mismatched}/{n_random}")
    else:
        print("  PASS: matches real matplotlib.path.Path.intersects_path (default "
              "filled=True) exactly.")


def _np_fast_multi_agent_loop(p1s: np.ndarray, p2s: np.ndarray,
                               open_ax, open_ay, open_bx, open_by, open_bounds,
                               close_ax, close_ay, close_bx, close_by, close_bounds,
                               n_groups: int, n_ticks: int) -> None:
    for t in range(n_ticks):
        filled_hits_multi_agent_fast(p1s[t], p2s[t], open_ax, open_ay, open_bx, open_by, open_bounds,
                                      close_ax, close_ay, close_bx, close_by, close_bounds, n_groups)


def run_fast_multi_agent_benchmark(n_agents: int = 5, n_ticks: int = 50_000, seed: int = 14):
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    open_ax, open_ay, open_bx, open_by, open_bounds, n_groups = precompute_shape_batch_fast(shapes)
    close_ax, close_ay, close_bx, close_by, close_bounds, _ = precompute_shape_batch_closing_fast(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_ticks, n_agents, 2))
    steps = rng.normal(scale=2.0, size=(n_ticks, n_agents, 2))
    p1s, p2s = centers, centers + steps

    t0 = time.perf_counter()
    _mpl_multi_agent_loop_filled(p1s, p2s, shape_paths, n_agents, n_ticks)
    t_mpl = time.perf_counter() - t0

    t0 = time.perf_counter()
    _np_fast_multi_agent_loop(p1s, p2s, open_ax, open_ay, open_bx, open_by, open_bounds,
                               close_ax, close_ay, close_bx, close_by, close_bounds, n_groups, n_ticks)
    t_np = time.perf_counter() - t0

    print(f"\n=== Part 5 (optimized) multi-agent benchmark: {n_ticks} ticks, "
          f"{n_agents} aircraft/tick, {n_groups} shapes/aircraft ===")
    print(f"  matplotlib (real default, {n_agents}x{n_groups}={n_agents*n_groups} calls/tick): "
          f"{t_mpl:.3f}s ({t_mpl/n_ticks*1e6:.2f}us/tick)")
    print(f"  numpy, optimized batched call/tick                    : "
          f"{t_np:.3f}s ({t_np/n_ticks*1e6:.2f}us/tick)")
    print(f"  speedup: {t_mpl/t_np:.2f}x  (>1.0 = numpy faster, <1.0 = numpy slower)")
    return t_mpl, t_np


def run_fast_multi_agent_profiling(n_agents: int = 5, n_ticks: int = 5_000, seed: int = 14) -> None:
    shapes = make_all_runway_shapes()
    shape_paths = [Path(s) for s in shapes]
    open_ax, open_ay, open_bx, open_by, open_bounds, n_groups = precompute_shape_batch_fast(shapes)
    close_ax, close_ay, close_bx, close_by, close_bounds, _ = precompute_shape_batch_closing_fast(shapes)

    all_pts = np.concatenate(shapes, axis=0)
    lo, hi = all_pts.min(axis=0) - 5.0, all_pts.max(axis=0) + 5.0
    rng = np.random.default_rng(seed)
    centers = rng.uniform(lo, hi, size=(n_ticks, n_agents, 2))
    steps = rng.normal(scale=2.0, size=(n_ticks, n_agents, 2))
    p1s, p2s = centers, centers + steps

    for label, fn, args in [
        ("matplotlib (real default filled=True)", _mpl_multi_agent_loop_filled,
         (p1s, p2s, shape_paths, n_agents, n_ticks)),
        ("numpy, Part 5 optimized batched", _np_fast_multi_agent_loop,
         (p1s, p2s, open_ax, open_ay, open_bx, open_by, open_bounds,
          close_ax, close_ay, close_bx, close_by, close_bounds, n_groups, n_ticks)),
    ]:
        profiler = cProfile.Profile()
        profiler.enable()
        fn(*args)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(14)
        print(f"\n=== cProfile: {label} ({n_ticks} ticks) ===")
        print(stream.getvalue())


if __name__ == "__main__":
    print("#" * 70)
    print("# PART 1: one-shape-at-a-time replacement (naive vectorization)")
    print("#" * 70)
    run_validation(n_random=100_000)
    t_mpl_1, t_np_1 = run_benchmark(n_calls=500_000)
    run_profiling(n_calls=100_000)

    print("\n" + "#" * 70)
    print("# PART 2: all ~24 shapes/tick batched into ONE numpy call")
    print("#" * 70)
    run_batched_equivalence_check(n_random=5000)
    t_mpl_2, t_np_2 = run_batched_benchmark(n_calls=100_000)
    run_batched_profiling(n_calls=20_000)

    print("\n" + "#" * 70)
    print("# PART 3: 5 aircraft x ~24 shapes/tick, ALL batched into ONE call")
    print("#" * 70)
    run_multi_agent_equivalence_check(n_agents=5, n_random=2000)
    t_mpl_3, t_np_3 = run_multi_agent_benchmark(n_agents=5, n_ticks=50_000)
    run_multi_agent_profiling(n_agents=5, n_ticks=5_000)

    print("\n" + "#" * 70)
    print("# PART 4: filling the gaps -- correct (filled=True) + batched")
    print("#" * 70)
    run_filled_validation(n_random=50_000)
    run_filled_batched_equivalence_check(n_agents=5, n_random=2000)
    run_filled_batched_vs_matplotlib_check(n_agents=5, n_random=2000)
    t_mpl_4, t_np_4 = run_filled_multi_agent_benchmark(n_agents=5, n_ticks=50_000)
    run_filled_multi_agent_profiling(n_agents=5, n_ticks=5_000)

    # Demonstrate the priority-order pick (death_cause equivalent) on one
    # batch, purely to show it composes with the batched hit matrix --
    # not a benchmark, just a correctness/API demonstration.
    demo_shapes = make_all_runway_shapes()
    demo_open_a, demo_open_b, demo_open_bounds, demo_n_groups = precompute_shape_batch_with_boundaries(demo_shapes)
    demo_close_a, demo_close_b, demo_close_bounds, _ = precompute_shape_batch_closing(demo_shapes)
    demo_rng = np.random.default_rng(11)
    demo_pts = np.concatenate(demo_shapes, axis=0)
    demo_lo, demo_hi = demo_pts.min(axis=0) - 5.0, demo_pts.max(axis=0) + 5.0
    demo_p1s = demo_rng.uniform(demo_lo, demo_hi, size=(5, 2))
    demo_p2s = demo_p1s + demo_rng.normal(scale=2.0, size=(5, 2))
    demo_hits = filled_hits_multi_agent(demo_p1s, demo_p2s, demo_open_a, demo_open_b, demo_open_bounds,
                                         demo_close_a, demo_close_b, demo_close_bounds, demo_n_groups)
    demo_winners = pick_first_by_priority(demo_hits)
    print(f"\n=== Priority-pick demo (5 aircraft, group 0=target_sink, "
          f"1=target_restrict, 2..=other runways) ===")
    print(f"  per-aircraft any_hit: {demo_hits.any(axis=1)}")
    print(f"  per-aircraft winning shape index (-1 = no hit): {demo_winners}")

    print("\n" + "#" * 70)
    print("# PART 5: further-optimized version of Part 4 (same math, less overhead)")
    print("#" * 70)
    run_fast_equivalence_check(n_agents=5, n_random=2000)
    run_fast_vs_matplotlib_check(n_agents=5, n_random=2000)
    t_mpl_5, t_np_5 = run_fast_multi_agent_benchmark(n_agents=5, n_ticks=50_000)
    run_fast_multi_agent_profiling(n_agents=5, n_ticks=5_000)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        f"Speedup by version (>1.0 = numpy faster than matplotlib's REAL default):\n"
        f"  Part 1 -- 1 shape/call, segment-only (INCOMPLETE)          : {t_mpl_1/t_np_1:.2f}x\n"
        f"  Part 2 -- ~24 shapes/tick batched, segment-only (INCOMPLETE): {t_mpl_2/t_np_2:.2f}x\n"
        f"  Part 3 -- 5 aircraft x 24 shapes batched, segment-only (INCOMPLETE): {t_mpl_3/t_np_3:.2f}x\n"
        f"  Part 4 -- 5 aircraft x 24 shapes batched, CORRECT (filled=True): {t_mpl_4/t_np_4:.2f}x\n"
        f"  Part 5 -- Part 4, optimized (hoisted closures, precomputed x/y,\n"
        f"            single combined p1+p2 point-in-polygon call)     : {t_mpl_5/t_np_5:.2f}x\n"
    )
    print(
        "Parts 1-3 only implemented segment-crossing (filled=False), which is\n"
        "cheaper than matplotlib's real filled=True default -- their speedups were\n"
        "measured against the right matplotlib baseline but an incomplete numpy\n"
        "side, so they overstate what a CORRECT replacement would achieve. Part 4\n"
        "adds the missing enclosure term (reverse-engineered and validated to 0\n"
        "mismatches against real matplotlib, both standalone and batched). Part 5\n"
        "is the same math as Part 4, re-validated against Part 4 AND against real\n"
        "matplotlib from scratch (not assumed carried over), with three concrete\n"
        "overheads removed that Part 4's own cProfile identified: per-call closure\n"
        "creation, per-call edge x/y re-slicing, and a doubled point-in-polygon\n"
        "call (once for p1, once for p2) each paying its own np.errstate cost.\n"
        "Part 5's number is the one to trust as 'what this can actually do.'\n\n"
        "Remaining gap, now closed at the math level, still open at the\n"
        "integration level: pick_first_by_priority (demoed above) shows the\n"
        "batched hit matrix composes cleanly with _check_terminal's sink > "
        "restrict > other-runway priority order for death_cause -- but wiring\n"
        "this into the real stepping loop still means restructuring\n"
        "_check_terminal from a per-slot function into a per-tick-for-all-\n"
        "active-slots one (Part 3's finding), which is genuine surgery on the\n"
        "env's control flow, not a drop-in swap. Given the verification bar this\n"
        "session has held everything else to (bit-for-bit validate_multiagent_env.py,\n"
        "full M=30/8-combo compare), that restructuring would need the same\n"
        "treatment before touching the real pipeline -- worth doing only if the\n"
        "Part 5 speedup above justifies it against that cost."
    )
