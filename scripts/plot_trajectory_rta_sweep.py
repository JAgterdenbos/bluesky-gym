"""
scripts/plot_trajectory_rta_sweep.py
-------------------------------------
SF-17 (supervisor comments #30/#31): trajectories flown by the frozen Phase II
winner (No-HER, without heading) from several fixed spawn points, at several
RTA slack targets, to show holding/vectoring behaviour emerging as the target
tightens. Reuses the model's own trained slack distribution (Eq. 14 in the
paper: goal_dist = sampler-predicted d_hat + slack, slack in [0, 100] km) by
monkeypatching the two private env methods that draw spawn/slack randomly,
rather than inventing a new distribution.

Usage
-----
  python scripts/plot_trajectory_rta_sweep.py
  python scripts/plot_trajectory_rta_sweep.py --n-spawns 2 --slack-km 0 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrow
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import MultipleLocator
import numpy as np

import bluesky as bs
import bluesky_gym
import bluesky_gym.envs.common.functions as fn
from bluesky_gym.envs.pathplanning_goal_env import (
    SCHIPHOL,
    RUNWAYS_SCHIPHOL_FAF,
    MAX_DISTANCE,
    MAX_TIME,
    SPEED,
    NM2KM,
    FAF_DISTANCE,
    IAF_DISTANCE,
    IAF_ANGLE,
)
from path_planning.experiment import BasePathPlanningExperiment
from path_planning.rta.collect import create_env_and_model
from path_planning.rta.testing.spatial_visitation_analysis import (
    cmap_population,
    draw_population,
)

# rcParams reused from spatial_visitation_analysis.py for visual consistency
# with the rest of the results chapter's figures.
matplotlib.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
})


# ─────────────────────────────────────────────────────────────
# Coordinate helper: physical East/North km from Schiphol
# ─────────────────────────────────────────────────────────────

def latlon_to_km(lat, lon):
    """Local East/North Cartesian offset (km) from Schiphol, same bearing/
    distance convention the env itself uses for its own (x, y) observation."""
    brg, dis = bs.tools.geo.kwikqdrdist(SCHIPHOL[0], SCHIPHOL[1], lat, lon)
    brg = np.radians(brg)
    dis_km = dis * NM2KM
    return dis_km * np.sin(brg), dis_km * np.cos(brg)


# ─────────────────────────────────────────────────────────────
# Deterministic spawn / RTA-slack overrides
# ─────────────────────────────────────────────────────────────

def install_fixed_spawn(env_unwrapped, lat, lon, heading):
    env_unwrapped._get_spawn = lambda: (lat, lon, heading)


_ORIGINAL_UPDATE_REWARD = None


def install_trajectory_recorder(env_unwrapped, points):
    """Hooks `_update_reward`, which the env already calls every SIM_DT (5s)
    simulation tick inside a single macro `env.step()` call, to record a
    position at that same fine resolution instead of only once per 120s
    action. Always wraps the true class-level original (captured once) so
    repeated installs across rollouts don't stack wrappers."""
    global _ORIGINAL_UPDATE_REWARD
    if _ORIGINAL_UPDATE_REWARD is None:
        _ORIGINAL_UPDATE_REWARD = type(env_unwrapped)._update_reward

    def _update_reward():
        _ORIGINAL_UPDATE_REWARD(env_unwrapped)
        points.append(latlon_to_km(bs.traf.lat[0], bs.traf.lon[0]))

    env_unwrapped._update_reward = _update_reward


def install_fixed_goal(env_unwrapped, slack_km):
    """Reproduces `_compute_goal_vector` verbatim, replacing the random slack
    draw with a fixed value so identical (spawn, slack) pairs are repeatable."""

    def _compute_goal_vector(runway):
        self = env_unwrapped
        rwy_info = RUNWAYS_SCHIPHOL_FAF[runway]

        iaf_lat, iaf_lon = fn.get_point_at_distance(
            rwy_info["lat"], rwy_info["lon"],
            FAF_DISTANCE + IAF_DISTANCE,
            rwy_info["track"] - 180,
        )

        goal_brg, goal_dis = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], iaf_lat, iaf_lon
        )
        goal_brg = np.radians(goal_brg)
        goal_dis = goal_dis * NM2KM / MAX_DISTANCE

        goal_x = np.sin(goal_brg) * goal_dis
        goal_y = np.cos(goal_brg) * goal_dis

        ac_brg, ac_dis = bs.tools.geo.kwikqdrdist(
            SCHIPHOL[0], SCHIPHOL[1], self.lat, self.lon
        )
        ac_brg = np.radians(ac_brg)
        ac_dis = ac_dis * NM2KM / MAX_DISTANCE

        goal_t = 0.0
        if self.use_rta:
            goal_dist = self._rta_sampler.sample(np.array([ac_dis, ac_brg]), runway)
            goal_dist = 1000 * (goal_dist + slack_km)  # km -> m, fixed slack
            goal_t = goal_dist / SPEED / MAX_TIME

        return np.array([goal_x, goal_y, goal_t], dtype=np.float64)

    env_unwrapped._compute_goal_vector = _compute_goal_vector


# ─────────────────────────────────────────────────────────────
# Spawn grid
# ─────────────────────────────────────────────────────────────

def build_spawn_points(r0_km, n_spawns, start_bearing_deg=0.0):
    """Evenly distributes n_spawns around the full 360 deg circle at radius r0_km."""
    bearings = start_bearing_deg + np.linspace(0, 360, n_spawns, endpoint=False)
    points = []
    for brg in bearings:
        lat, lon = fn.get_point_at_distance(SCHIPHOL[0], SCHIPHOL[1], r0_km, brg)
        heading = (brg + 180) % 360
        points.append((lat, lon, heading))
    return points, bearings


# ─────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────

def rollout(env, model, spawn_points, slack_values):
    """Returns {(spawn_idx, slack_km): (xs_km, ys_km, success)}."""
    trajectories = {}
    for spawn_idx, (lat, lon, heading) in enumerate(spawn_points):
        for slack_km in slack_values:
            install_fixed_spawn(env.unwrapped, lat, lon, heading)
            install_fixed_goal(env.unwrapped, slack_km)

            obs, info = env.reset()
            points = [latlon_to_km(bs.traf.lat[0], bs.traf.lon[0])]
            install_trajectory_recorder(env.unwrapped, points)

            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)

            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            success = bool(info.get("is_success", False))
            trajectories[(spawn_idx, slack_km)] = (xs, ys, success)
            print(
                f"  spawn {spawn_idx} (brg-relative) | slack {slack_km:>5.1f} km "
                f"| points {len(points):>4d} | success={success}"
            )
    return trajectories


# ─────────────────────────────────────────────────────────────
# Population basemap (physical km, NOT the normalised [-1, 1] convention)
# ─────────────────────────────────────────────────────────────

def load_population_km(pop_path, x_path, y_path):
    Z = np.genfromtxt(pop_path, delimiter=" ")
    X = np.genfromtxt(x_path, delimiter=" ") / 1000.0  # m -> km
    Y = np.genfromtxt(y_path, delimiter=" ") / 1000.0
    return X, Y, Z


# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────

SLACK_STYLES = [":", "-.", "--", "-", (0, (1, 1, 3, 1))]

# Okabe & Ito (2008) colour-blind-safe qualitative palette — distinguishable
# under deuteranopia/protanopia/tritanopia, unlike a hue-wheel cycle (which
# passes straight through the red-green confusion band). The stock palette's
# pure yellow (#F0E442) is nearly invisible as a thin line against a light
# basemap, colour-blind or not — swapped for a darker, still-distinct gold.
SPAWN_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#B8960C",
                "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def spawn_color(spawn_idx):
    return SPAWN_COLORS[spawn_idx % len(SPAWN_COLORS)]


def draw_map_layers(target_ax, X, Y, Z, cone_pts, rwy_km, trajectories, spawn_points,
                     bearings, slack_sorted, style_map, show_labels=True, linewidth=1.0):
    """Draws the basemap + EHAM + cone + trajectories onto any axes — used for
    both the main map and the zoomed holding-pattern inset, so they never
    drift out of sync with each other."""
    draw_population(target_ax, X, Y, Z, cmap_population())

    target_ax.scatter([0], [0], marker="*", s=220 if show_labels else 90,
                       color="#264653", zorder=6, edgecolors="white", linewidths=0.8)
    if show_labels:
        target_ax.annotate("EHAM", (0, 0), textcoords="offset points", xytext=(7, 6),
                            fontsize=8.5, color="#264653", fontweight="bold", zorder=6)

    cone = Polygon(cone_pts, closed=True, facecolor="#f4a261", edgecolor="#e76f51",
                    alpha=0.30, linewidth=1.0, zorder=2)
    target_ax.add_patch(cone)

    for spawn_idx, ((lat, lon, heading), brg) in enumerate(zip(spawn_points, bearings)):
        color = spawn_color(spawn_idx)
        if show_labels:
            sx, sy = latlon_to_km(lat, lon)
            target_ax.scatter([sx], [sy], marker="o", s=32, color=color, zorder=5,
                               edgecolors="white", linewidths=0.6)
            target_ax.annotate(f"S{spawn_idx + 1}", (sx, sy), textcoords="offset points",
                                xytext=(5, 5), fontsize=7.5, color="black",
                                fontweight="bold", zorder=5)
        for slack_km in slack_sorted:
            xs, ys, success = trajectories[(spawn_idx, slack_km)]
            target_ax.plot(xs, ys, color=color, linestyle=style_map[slack_km],
                            linewidth=linewidth, alpha=0.9 if success else 0.35, zorder=4)
            if not success and show_labels:
                target_ax.scatter([xs[-1]], [ys[-1]], marker="x", s=40,
                                   color=color, zorder=7)


def plot_sweep(trajectories, spawn_points, bearings, runway, slack_values, out_path,
               pop_data, x_grid, y_grid, inset_xlim=(-150, -50), inset_ylim=(-50, 50)):
    fig, ax = plt.subplots(figsize=(9.3, 8.0))

    X, Y, Z = load_population_km(pop_data, x_grid, y_grid)

    # ── runway + IAF/FAF approach cone geometry ─────────────
    rwy_info = RUNWAYS_SCHIPHOL_FAF[runway]
    rwy_km = latlon_to_km(rwy_info["lat"], rwy_info["lon"])

    faf_lat, faf_lon = fn.get_point_at_distance(
        rwy_info["lat"], rwy_info["lon"], FAF_DISTANCE, rwy_info["track"] - 180
    )
    faf_km = latlon_to_km(faf_lat, faf_lon)

    # True circular-sector cone, matching the env's own SINK-polygon geometry
    # in `_set_terminal_conditions` (FAF apex + an arc of points at IAF_DISTANCE,
    # spanning IAF_ANGLE degrees) instead of a straight-edged triangle.
    cw_bound = ((rwy_info["track"] - 180 + 360) % 360) + (IAF_ANGLE / 2)
    ccw_bound = ((rwy_info["track"] - 180 + 360) % 360) - (IAF_ANGLE / 2)
    arc_angles = np.linspace(cw_bound, ccw_bound, 36)
    arc_lat, arc_lon = fn.get_point_at_distance(faf_lat, faf_lon, IAF_DISTANCE, arc_angles)
    arc_km = [latlon_to_km(la, lo) for la, lo in zip(arc_lat, arc_lon)]
    cone_pts = [faf_km] + arc_km

    slack_sorted = sorted(slack_values)
    style_map = {s: SLACK_STYLES[i % len(SLACK_STYLES)] for i, s in enumerate(slack_sorted)}

    draw_map_layers(ax, X, Y, Z, cone_pts, rwy_km, trajectories, spawn_points, bearings,
                     slack_sorted, style_map, show_labels=True, linewidth=1.0)

    # ── full spawn envelope: the whole [-300, 300] km x [-300, 300] km map
    # (MAX_DISTANCE), not cropped to the trajectories/corridor alone — plus a
    # small margin so spawn labels at r=R0 aren't clipped at the plot border.
    margin = 25.0
    xmin, xmax = -MAX_DISTANCE - margin, MAX_DISTANCE + margin
    ymin, ymax = -MAX_DISTANCE - margin, MAX_DISTANCE + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ── zoomed inset: placed outside the axes, in the same right-hand margin
    # as the legend — not floated on top of the map, and with no connector
    # lines back to the main map (just a plain rectangle marking the region).
    rect = plt.Rectangle((inset_xlim[0], inset_ylim[0]),
                          inset_xlim[1] - inset_xlim[0], inset_ylim[1] - inset_ylim[0],
                          fill=False, edgecolor="black", linewidth=1.0, zorder=9)
    ax.add_patch(rect)

    axins = ax.inset_axes([1.08, 0.04, 0.34, 0.42])
    draw_map_layers(axins, X, Y, Z, cone_pts, rwy_km, trajectories, spawn_points, bearings,
                     slack_sorted, style_map, show_labels=False, linewidth=1.3)
    axins.set_xlim(*inset_xlim)
    axins.set_ylim(*inset_ylim)
    axins.set_aspect("equal")
    axins.set_title("Manoeuvring detail", fontsize=8.5, fontweight="bold", pad=4)
    axins.xaxis.set_major_locator(MultipleLocator(25))
    axins.yaxis.set_major_locator(MultipleLocator(25))
    axins.grid(True, linestyle=":", linewidth=0.4, color="#888888", alpha=0.35, zorder=0)
    axins.tick_params(labelsize=6, length=2)
    axins.set_xlabel("Easting (km)", fontsize=6.5, labelpad=2)
    axins.set_ylabel("Northing (km)", fontsize=6.5, labelpad=2)
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    # ── extra cartographic cues ──────────────────────────────
    bar_x0 = xmin + 0.06 * (xmax - xmin)
    bar_y0 = ymin + 0.06 * (ymax - ymin)
    ax.plot([bar_x0, bar_x0 + 100], [bar_y0, bar_y0], color="black", linewidth=2, zorder=8)
    ax.annotate("100 km", (bar_x0 + 50, bar_y0), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=7.5, zorder=8)

    arrow_x = xmax - 0.06 * (xmax - xmin)
    arrow_y0 = ymin + 0.06 * (ymax - ymin)
    ax.add_patch(FancyArrow(arrow_x, arrow_y0, 0, 0.08 * (ymax - ymin),
                             width=1.2, head_width=6, head_length=6,
                             color="black", zorder=8))
    ax.annotate("N", (arrow_x, arrow_y0 + 0.10 * (ymax - ymin)), ha="center",
                fontsize=8.5, fontweight="bold", zorder=8)

    # ── legend ───────────────────────────────────────────────
    # Spawn identity is already given directly on the map (colour + S# label at
    # each origin), so the only legend needed is the slack line-style key —
    # one clean box instead of two.
    slack_handles = [
        Line2D([0], [0], color="#333333", linestyle=style_map[s], linewidth=1.3,
               label=f"{s:g} km")
        for s in slack_sorted
    ]
    # Placed just outside the axes (not in a corner) so it never sits on top
    # of a spawn point regardless of how many spawns/bearings are used.
    ax.legend(handles=slack_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=8, framealpha=0.92, title="$\\delta_\\text{slack}$",
              title_fontsize=8.5, borderpad=0.7, labelspacing=0.5)

    # ── axes ─────────────────────────────────────────────────
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.grid(True, linestyle=":", linewidth=0.5, color="#888888", alpha=0.35, zorder=0)
    ax.set_xlabel("Easting from EHAM (km)", fontsize=9)
    ax.set_ylabel("Northing from EHAM (km)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.set_title(
        f"Frozen No-HER Policy: Trajectory Response to RTA Slack\n"
        f"{len(spawn_points)} spawn points around EHAM, runway {runway}",
        fontsize=11.5, fontweight="bold", linespacing=1.5,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[+] Saved {out_path}")
    return fig


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _get_args():
    p = argparse.ArgumentParser(
        description="Trajectory-vs-RTA-slack sweep for the frozen No-HER Phase II winner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-id", type=str, default="20260615_095840")
    p.add_argument("--runway", type=str, default="18R")
    p.add_argument("--r0-km", type=float, default=280.0)
    p.add_argument("--n-spawns", type=int, default=8)
    p.add_argument("--start-bearing-deg", type=float, default=0.0,
                    help="Phase offset for the evenly-spaced full-circle spawn grid.")
    p.add_argument("--slack-km", type=float, nargs="+", default=[0.0, 33.0, 67.0, 100.0])
    p.add_argument("--out", type=str, default="figures/spatio_temporal/trajectory_rta_sweep.png")
    p.add_argument("--pop-data", type=str, default="bluesky_gym/envs/data/population_1km.csv")
    p.add_argument("--x-grid", type=str, default="bluesky_gym/envs/data/x_array.csv")
    p.add_argument("--y-grid", type=str, default="bluesky_gym/envs/data/y_array.csv")
    return p.parse_args()


def main():
    args = _get_args()
    bluesky_gym.register_envs()

    print(f"Loading run {args.run_id} ...")
    env, model, success_key = create_env_and_model(
        experiment_cls=BasePathPlanningExperiment,
        run_id=args.run_id,
        runways=[args.runway],
    )
    print(f"  success_key={success_key}, use_rta={env.unwrapped.use_rta}")

    spawn_points, bearings = build_spawn_points(
        args.r0_km, args.n_spawns, args.start_bearing_deg
    )
    print(f"Spawn bearings (deg, relative to EHAM): {np.round(bearings, 1).tolist()}")

    print(f"Rolling out {args.n_spawns} spawns x {len(args.slack_km)} slack targets ...")
    trajectories = rollout(env, model, spawn_points, args.slack_km)
    env.close()

    n_fail = sum(1 for *_, success in trajectories.values() if not success)
    if n_fail:
        print(f"[!] {n_fail}/{len(trajectories)} rollouts did NOT reach is_success=True.")
    else:
        print(f"[+] All {len(trajectories)} rollouts succeeded.")

    plot_sweep(
        trajectories, spawn_points, bearings, args.runway, args.slack_km, args.out,
        args.pop_data, args.x_grid, args.y_grid,
    )


if __name__ == "__main__":
    main()
