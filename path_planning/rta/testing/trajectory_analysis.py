import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import argparse
import re
import os

# ── 1. Style & Helper Functions (Thesis Light Mode) ───────────────────────────
BG         = "#ffffff"  # Crisp white background for print/thesis
GRID       = "#e5e5e5"  # Faint gray for clean grid lines
TEXT_DARK  = "#222222"  # Off-black for optimal text readability
TEXT_MUTED = "#555555"  # Soft dark gray for labels and ticks

SUCCESS_C  = "#1a8754"  # Professional academic green
FAIL_C     = "#b51d1d"  # Deep crimson red for high-contrast failures
FAIL_CMAP  = "Reds"     # Smooth high-contrast gradient for white background

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")  # Clean, light border lines
    ax.xaxis.label.set_color(TEXT_MUTED)
    ax.yaxis.label.set_color(TEXT_MUTED)
    ax.set_title(title, color=TEXT_DARK, fontsize=10, fontweight="semibold", pad=6)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(color=GRID, linewidth=0.5)

def add_eham(ax, label=True):
    # Professional dark blue crosshair for the airport location
    ax.scatter(0, 0, color="#0056b3", s=100, zorder=10, marker="+", linewidths=2)
    if label:
        ax.annotate("EHAM", (0,0), color="#0056b3", fontsize=7, fontweight="bold",
                    xytext=(0.03, 0.03), textcoords="data")

def add_centreline(ax, rwy_name, heading_deg):
    angle_rad = np.radians(90 - heading_deg)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    ax.axline((0, 0), (dx, dy), color=TEXT_MUTED, linewidth=0.6, linestyle="--",
              alpha=0.4, zorder=1, label=f"{rwy_name} centreline")

def heading_error(h, target_heading):
    err = (h - target_heading) % 360
    return err if err <= 180 else err - 360

def get_default_heading(rwy_name):
    """Dynamically calculates heading based on standard runway naming (e.g., 36R -> 360)."""
    match = re.match(r"(\d{2})[LCR]?", rwy_name)
    if not match:
        raise ValueError(f"Invalid runway format: '{rwy_name}'. Cannot auto-derive heading.")
    heading = int(match.group(1)) * 10
    return 360 if heading == 0 else heading

# ── 2. Main Plotting Routine ──────────────────────────────────────────────────
def plot_runway_analysis(rwy, algo, data_path, heading=None, scale=300, out_dir=".", dpi=150):
    """
    Plots trajectory analysis driven by CLI arguments.
    """
    # ── Configuration Resolution ──
    rwy_heading = heading if heading is not None else get_default_heading(rwy)
    
    # ── Load & filter ──
    if not os.path.exists(data_path):
        print(f"Error: Could not find data file at {data_path}")
        return

    print(f"Loading data from {data_path}...")
    
    _, ext = os.path.splitext(data_path.lower())
    if ext in ['.parquet', '.pq']:
        df = pd.read_parquet(data_path, engine='pyarrow')
    else:
        df = pd.read_csv(data_path)
        
    df = df[df["runway"] == rwy]

    if df.empty:
        print(f"Warning: No data found for runway '{rwy}' in this dataset.")
        return

    success_eps = df[df["is_success"] == True]["episode"].unique()
    failure_eps = df[df["is_success"] == False]["episode"].unique()
    print(f"[{rwy} | {algo}] Successes: {len(success_eps)}, Failures: {len(failure_eps)}")

    # ── Pre-compute per-episode features ──
    episode_max_step = df.groupby("episode")["step"].max()
    global_max_step  = df["step"].max()

    first_steps = df.groupby("episode").first().reset_index()
    last_steps  = df.groupby("episode").last().reset_index()

    def bearing_deg(row):
        return np.degrees(np.arctan2(row["x"], row["y"])) % 360

    first_steps["bearing"] = first_steps.apply(bearing_deg, axis=1)

    def final_heading(ep_id):
        t = df[df["episode"] == ep_id].sort_values("step")
        if len(t) < 2:
            return np.nan
        dx = t["x"].iloc[-1] - t["x"].iloc[-2]
        dy = t["y"].iloc[-1] - t["y"].iloc[-2]
        return np.degrees(np.arctan2(dx, dy)) % 360

    ep_meta = df.groupby("episode").agg(
        total_dist_km=("total_dist_km", "first"),
        rta=("rta", "first"),
        is_success=("is_success", "first"),
    ).reset_index()
    
    ep_meta = ep_meta.merge(first_steps[["episode","bearing"]], on="episode")
    ep_meta["max_step"]    = ep_meta["episode"].map(episode_max_step)
    ep_meta["progress"]    = ep_meta["max_step"] / global_max_step
    ep_meta["final_hdg"]   = ep_meta["episode"].apply(final_heading)
    ep_meta["hdg_error"]   = ep_meta["final_hdg"].apply(
                                 lambda h: heading_error(h, rwy_heading) if not np.isnan(h) else np.nan)

    ep_coords = first_steps[["episode", "x", "y"]].rename(columns={"x": "sx", "y": "sy"})
    ep_coords = ep_coords.merge(
        last_steps[["episode", "x", "y"]].rename(columns={"x": "ex", "y": "ey"}),
        on="episode"
    )
    ep_meta = ep_meta.merge(ep_coords, on="episode")

    ep_meta["chord_km"] = np.sqrt(
        (ep_meta["ex"] - ep_meta["sx"])**2 + 
        (ep_meta["ey"] - ep_meta["sy"])**2
    ) * scale

    ep_meta["tortuosity"] = ep_meta["total_dist_km"] / ep_meta["chord_km"].replace(0, np.nan)
    ep_meta["tortuosity"] = ep_meta["tortuosity"].clip(lower=1.0)
    ep_meta["bearing_bin"] = (ep_meta["bearing"] // 30 * 30).astype(int)

    rta_bins = np.linspace(ep_meta["rta"].min(), ep_meta["rta"].max(), 9)
    ep_meta["rta_bin"] = pd.cut(ep_meta["rta"], bins=rta_bins,
                                 labels=np.round(rta_bins[:-1], 3))

    ep_s = ep_meta[ep_meta["is_success"] == True]
    ep_f = ep_meta[ep_meta["is_success"] == False]

    # ── Figure layout ──
    fig = plt.figure(figsize=(26, 22), facecolor=BG)
    fig.suptitle(f"{rwy} Trajectory Analysis — {algo}",
                 color=TEXT_DARK, fontsize=17, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(
        3, 3, figure=fig, hspace=0.45, wspace=0.35,  # Slightly opened up spacing
        left=0.05, right=0.97, top=0.96, bottom=0.05
    )

    ax_traj  = fig.add_subplot(gs[0, 0])
    ax_term  = fig.add_subplot(gs[0, 1])
    ax_polar = fig.add_subplot(gs[0, 2], projection="polar")
    ax_steps = fig.add_subplot(gs[1, 0])
    ax_pathl = fig.add_subplot(gs[1, 1])
    ax_rta   = fig.add_subplot(gs[1, 2])
    ax_hdg   = fig.add_subplot(gs[2, 0])
    ax_heat  = fig.add_subplot(gs[2, 1])
    ax_tort  = fig.add_subplot(gs[2, 2])

    # Shared academic legend style configuration
    leg_style = dict(facecolor="#f8f9fa", labelcolor=TEXT_DARK, edgecolor="#e5e5e5", fontsize=7)

    # 1. All trajectories
    style_ax(ax_traj, f"All Trajectories ({rwy})", f"Normalised x (×{scale} km)", f"Normalised y (×{scale} km)")
    add_centreline(ax_traj, rwy, rwy_heading)

    for ep in success_eps:
        t = df[df["episode"] == ep]
        ax_traj.plot(t["x"], t["y"], color=SUCCESS_C, alpha=0.03, linewidth=0.4)
    for ep in failure_eps:
        t = df[df["episode"] == ep]
        ax_traj.plot(t["x"], t["y"], color=FAIL_C, alpha=0.55, linewidth=1.0)

    add_eham(ax_traj)
    ax_traj.set_xlim(-1, 1); ax_traj.set_ylim(-1, 1)
    ax_traj.legend(handles=[
        Line2D([0],[0], color=SUCCESS_C, alpha=0.7, label=f"Success (n={len(success_eps)})"),
        Line2D([0],[0], color=FAIL_C,    alpha=0.8, label=f"Failure  (n={len(failure_eps)})"),
    ], loc="upper right", **leg_style)

    # 2. Termination points
    style_ax(ax_term, f"Termination Points ({rwy})", f"Normalised x (×{scale} km)", f"Normalised y (×{scale} km)")
    add_centreline(ax_term, rwy, rwy_heading)

    term_s = last_steps[last_steps["episode"].isin(success_eps)]
    term_f = last_steps[last_steps["episode"].isin(failure_eps)]

    if not term_s.empty:
        jitter_s = np.random.normal(0, 0.003, size=len(term_s))
        ax_term.scatter(term_s["x"], term_s["y"] + jitter_s, c=SUCCESS_C, alpha=0.08, s=6,  
                        label=f"Success (n={len(term_s)})", clip_on=False)
        
    if not term_f.empty:
        jitter_f = np.random.normal(0, 0.003, size=len(term_f))
        ax_term.scatter(term_f["x"], term_f["y"] + jitter_f, c=FAIL_C, alpha=0.85, s=30, 
                        label=f"Failure (n={len(term_f)})", edgecolors="white", linewidths=0.3, clip_on=False)

    add_eham(ax_term)

    all_x = pd.concat([term_s["x"], term_f["x"]]) if not term_s.empty or not term_f.empty else pd.Series([0])
    all_y = pd.concat([term_s["y"], term_f["y"]]) if not term_s.empty or not term_f.empty else pd.Series([0])
    max_val = max(all_x.abs().max(), all_y.abs().max())
    limit = min(1.0, max(0.25, max_val * 1.1))
    ax_term.set_xlim(-limit, limit)
    ax_term.set_ylim(-limit, limit)
    
    ax_term.legend(loc="lower left", **leg_style)

    # 3. Polar approach-direction
    fs_s = first_steps[first_steps["episode"].isin(success_eps)]
    fs_f = first_steps[first_steps["episode"].isin(failure_eps)]

    bins_pol = np.linspace(0, 2*np.pi, 37)
    centres  = (bins_pol[:-1] + bins_pol[1:]) / 2
    width    = bins_pol[1] - bins_pol[0]

    s_cnt, _ = np.histogram(np.radians(fs_s["bearing"]), bins=bins_pol)
    f_cnt, _ = np.histogram(np.radians(fs_f["bearing"]), bins=bins_pol)
    s_den = s_cnt / s_cnt.sum() if s_cnt.sum() else s_cnt
    f_den = f_cnt / f_cnt.sum() if f_cnt.sum() else f_cnt

    ax_polar.set_facecolor(BG)
    ax_polar.bar(centres, s_den, width=width, color=SUCCESS_C, alpha=0.5, label="Success", zorder=2)
    ax_polar.bar(centres, f_den, width=width, color=FAIL_C,    alpha=0.7, label="Failure",  zorder=3)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.tick_params(colors=TEXT_MUTED, labelsize=7)
    ax_polar.set_title("Spawn Direction from EHAM", color=TEXT_DARK, fontsize=10, pad=14)
    ax_polar.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), **leg_style)
    ax_polar.grid(color=GRID, linewidth=0.5)

    # 4. Failures coloured by episode progress (With Success Background Context)
    style_ax(ax_steps, "Failures — coloured by episode progress", f"Normalised x (×{scale} km)", f"Normalised y (×{scale} km)")
    add_centreline(ax_steps, rwy, rwy_heading)

    # REFINEMENT: Faint context lines for successes so plot doesn't look empty
    for ep in success_eps:
        t = df[df["episode"] == ep]
        ax_steps.plot(t["x"], t["y"], color="#ececec", alpha=0.01, linewidth=0.3, zorder=1)

    cmap_fail = plt.get_cmap(FAIL_CMAP)
    for ep in failure_eps:
        t = df[df["episode"] == ep]
        progress = episode_max_step[ep] / global_max_step
        ax_steps.plot(t["x"], t["y"], color=cmap_fail(1 - progress), alpha=0.7, linewidth=1.1, zorder=2)

    add_eham(ax_steps)
    ax_steps.set_xlim(-1, 1); ax_steps.set_ylim(-1, 1)

    sm = ScalarMappable(cmap=cmap_fail, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_steps, fraction=0.035, pad=0.02)
    cbar.set_label("Episode progress (0=early fail, 1=late fail)", color=TEXT_MUTED, fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_MUTED)
    cbar.outline.set_edgecolor("#cccccc")

    # 5. Path length distribution
    style_ax(ax_pathl, "Path Length Distribution", "Total distance (km)", "Density")

    bins_pl = np.linspace(ep_meta["total_dist_km"].min(), ep_meta["total_dist_km"].max(), 50)
    ax_pathl.hist(ep_s["total_dist_km"], bins=bins_pl, density=True, color=SUCCESS_C, alpha=0.5, label="Success")
    ax_pathl.hist(ep_f["total_dist_km"], bins=bins_pl, density=True, color=FAIL_C,    alpha=0.7, label="Failure")
    if not ep_s.empty:
        ax_pathl.axvline(ep_s["total_dist_km"].mean(), color=SUCCESS_C, linestyle="--", linewidth=1, label=f"Mean success: {ep_s['total_dist_km'].mean():.0f} km")
    if not ep_f.empty:
        ax_pathl.axvline(ep_f["total_dist_km"].mean(), color="#ff8800", linestyle="--", linewidth=1, label=f"Mean failure: {ep_f['total_dist_km'].mean():.0f} km")
    ax_pathl.legend(loc="best", **leg_style)  # REFINEMENT: dynamic positioning

    # 6. RTA distribution
    style_ax(ax_rta, "RTA Distribution", "Required Time of Arrival (normalised)", "Density")

    bins_rta = np.linspace(ep_meta["rta"].min(), ep_meta["rta"].max(), 40)
    ax_rta.hist(ep_s["rta"], bins=bins_rta, density=True, color=SUCCESS_C, alpha=0.5, label="Success")
    ax_rta.hist(ep_f["rta"], bins=bins_rta, density=True, color=FAIL_C,    alpha=0.7, label="Failure")
    if not ep_s.empty:
        ax_rta.axvline(ep_s["rta"].mean(), color=SUCCESS_C, linestyle="--", linewidth=1, label=f"Mean success: {ep_s['rta'].mean():.3f}")
    if not ep_f.empty:
        ax_rta.axvline(ep_f["rta"].mean(), color="#ff8800", linestyle="--", linewidth=1, label=f"Mean failure: {ep_f['rta'].mean():.3f}")
    ax_rta.legend(loc="best", **leg_style)  # REFINEMENT: dynamic positioning

    # 7. Final approach heading error
    style_ax(ax_hdg, f"Final Approach Heading Error ({rwy} = {rwy_heading}°)", "Heading error (degrees)", "Density")

    hdg_s = ep_s["hdg_error"].dropna()
    hdg_f = ep_f["hdg_error"].dropna()
    bins_hdg = np.linspace(-180, 180, 60)

    ax_hdg.hist(hdg_s, bins=bins_hdg, density=True, color=SUCCESS_C, alpha=0.5, label="Success")
    ax_hdg.hist(hdg_f, bins=bins_hdg, density=True, color=FAIL_C,    alpha=0.7, label="Failure")
    ax_hdg.axvline(0, color=TEXT_MUTED, linewidth=1.0, linestyle="-", alpha=0.4, label=f"Perfect alignment ({rwy_heading}°)")
    if not hdg_s.empty:
        ax_hdg.axvline(hdg_s.mean(), color=SUCCESS_C, linestyle="--", linewidth=1, label=f"Mean success: {hdg_s.mean():.1f}°")
    if not hdg_f.empty:
        ax_hdg.axvline(hdg_f.mean(), color="#ff8800", linestyle="--", linewidth=1, label=f"Mean failure: {hdg_f.mean():.1f}°")
    ax_hdg.legend(loc="best", **leg_style)  # REFINEMENT: dynamic positioning

    # 8. RTA × bearing heatmap
    style_ax(ax_heat, "Failure Rate: Spawn Bearing × RTA", "Spawn bearing bin (°)", "RTA (normalised)")

    bear_bins = np.arange(0, 391, 30)
    pivot_fail  = ep_f.groupby(["rta_bin","bearing_bin"]).size().unstack(fill_value=0)
    pivot_total = ep_meta.groupby(["rta_bin","bearing_bin"]).size().unstack(fill_value=0)

    pivot_fail  = pivot_fail.reindex(index=pivot_total.index, columns=pivot_total.columns, fill_value=0)
    pivot_rate  = (pivot_fail / pivot_total.replace(0, np.nan)).fillna(0)

    im = ax_heat.imshow(
        pivot_rate.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
        extent=[bear_bins[0], bear_bins[-2],
                float(str(pivot_rate.index[-1])), float(str(pivot_rate.index[0]))],
        origin="upper"
    )
    
    # REFINEMENT: Enforce standardized decimal lengths for y-axis formatting
    ax_heat.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    cbar2 = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar2.set_label("Failure rate", color=TEXT_MUTED, fontsize=7)
    cbar2.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelsize=7)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color=TEXT_MUTED)
    cbar2.outline.set_edgecolor("#cccccc")
    ax_heat.set_xticks(bear_bins[::2])
    ax_heat.set_xticklabels([f"{b}°" for b in bear_bins[::2]], fontsize=7)

    # 9. Path tortuosity
    style_ax(ax_tort, "Path Tortuosity (actual / straight-line dist)", "Tortuosity ratio", "Density")

    tort_s = ep_s["tortuosity"].dropna()
    tort_f = ep_f["tortuosity"].dropna()

    t_max = min(max(tort_s.max() if not tort_s.empty else 0, tort_f.max() if not tort_f.empty else 0), 10)
    t_min = min(tort_s.min() if not tort_s.empty else 1, tort_f.min() if not tort_f.empty else 1)
    
    if t_max > t_min:
        bins_tort = np.linspace(t_min, t_max, 50)
        ax_tort.hist(tort_s, bins=bins_tort, density=True, color=SUCCESS_C, alpha=0.5, label="Success")
        ax_tort.hist(tort_f, bins=bins_tort, density=True, color=FAIL_C,    alpha=0.7, label="Failure")
        
        if not tort_s.empty:
            ax_tort.axvline(tort_s.mean(), color=SUCCESS_C, linestyle="--", linewidth=1, label=f"Mean success: {tort_s.mean():.2f}×")
        if not tort_f.empty:
            ax_tort.axvline(tort_f.mean(), color="#ff8800", linestyle="--", linewidth=1, label=f"Mean failure: {tort_f.mean():.2f}×")
            
        ax_tort.legend(loc="best", **leg_style)  # REFINEMENT: dynamic positioning

    # ── Save ──
    os.makedirs(out_dir, exist_ok=True)
    # Changing output extension to .png makes a vector format that is infinitely zoomable in LaTeX
    filename = os.path.join(out_dir, f"{rwy}_trajectory_analysis_{algo}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved publication-quality vector plot: {filename}")

# ── 3. CLI Execution ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate runway trajectory analysis plots.")
    
    parser.add_argument("-r", "--rwy", type=str, required=True, help="Runway identifier.")
    parser.add_argument("-a", "--algo", type=str, required=True, help="Algorithm name.")
    parser.add_argument("-d", "--data", type=str, required=True, help="Full data file path.")
    parser.add_argument("--heading", type=int, default=None, help="Target runway heading.")
    parser.add_argument("--scale", type=int, default=300, help="Scale factor (default: 300 km).")
    parser.add_argument("-o", "--out-dir", type=str, default=".", help="Output directory.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI resolution for rendering.")

    args = parser.parse_args()

    plot_runway_analysis(
        rwy=args.rwy, algo=args.algo, data_path=args.data, heading=args.heading,
        scale=args.scale, out_dir=args.out_dir, dpi=args.dpi
    )