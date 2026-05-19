"""
Spatial Visitation Analysis — v3
=================================
Clean, high-contrast dark-theme figure.
Architecture is ready for 3-subplot use (two visitation maps + difference).

Usage (single plot):
    python spatial_visitation_analysis_v3.py

Usage (3-subplot, uncomment at bottom):
    Call `make_three_panel(df_a, df_b, X, Y, Z)`
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, median_filter, zoom
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'figure.dpi':        180,
    'savefig.dpi':       300,
    'font.family':       'monospace',
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
})

MAX_DISTANCE = 300_000   # metres
BG           = '#08090d'
PANEL_BG     = '#0c0e14'

# ─────────────────────────────────────────────────────────────
# Colormaps
# ─────────────────────────────────────────────────────────────

def _make_cmap(stops, name, N=512):
    pos  = [s[0] for s in stops]
    rgba = np.array([s[1] for s in stops], dtype=float)
    out  = np.zeros((N, 4))
    for ch in range(4):
        out[:, ch] = np.interp(np.linspace(0, 1, N), pos, rgba[:, ch])
    return mcolors.ListedColormap(out, name=name)


def cmap_visitation():
    """Black → violet → fuchsia → orange → cream — perceptually bright."""
    stops = [
        (0.00, [0.00, 0.00, 0.00, 0.00]),   # transparent
        (0.04, [0.03, 0.01, 0.08, 0.85]),   # near-black purple
        (0.20, [0.18, 0.02, 0.42, 0.95]),   # deep violet
        (0.42, [0.65, 0.05, 0.60, 1.00]),   # fuchsia
        (0.65, [0.97, 0.38, 0.05, 1.00]),   # vivid orange
        (0.82, [0.99, 0.78, 0.20, 1.00]),   # amber
        (1.00, [1.00, 0.97, 0.88, 1.00]),   # cream white
    ]
    return _make_cmap(stops, 'visitation')


def cmap_difference():
    """Diverging: electric teal → dark neutral → vivid coral. 
    Zero-crossing is a visible cool-grey, not black."""
    stops = [
        (0.00, [0.00, 0.75, 0.80, 1.00]),   # bright teal
        (0.30, [0.00, 0.30, 0.38, 1.00]),   # mid teal
        (0.46, [0.18, 0.22, 0.28, 1.00]),   # cool grey-blue (near zero)
        (0.50, [0.26, 0.28, 0.32, 1.00]),   # visible neutral (zero)
        (0.54, [0.30, 0.18, 0.18, 1.00]),   # cool grey-red (near zero)
        (0.70, [0.55, 0.10, 0.08, 1.00]),   # mid coral
        (1.00, [1.00, 0.35, 0.10, 1.00]),   # vivid coral
    ]
    return _make_cmap(stops, 'difference')

def cmap_population():
    """
    6-Stop Ultra-Tactical Palette:
    Ocean -> Coastline -> Rural -> Urban -> Metro -> Core Hub
    """
    stops = [
        (0.00, [0.02, 0.03, 0.05, 1.00]),   # 1. Deep Void (Ocean)
        (0.12, [0.05, 0.08, 0.12, 1.00]),   # 2. Ghostly Trace (Landmass Edge)
        (0.35, [0.10, 0.18, 0.28, 1.00]),   # 3. Midnight Blue (Low Density/Rural)
        (0.60, [0.20, 0.35, 0.50, 1.00]),   # 4. Steel Blue (Suburban/Urban)
        (0.85, [0.40, 0.65, 0.85, 1.00]),   # 5. Electric Sky (High Density Metro)
        (1.00, [0.80, 0.95, 1.00, 1.00]),   # 6. Ice White (Central Business Districts)
    ]
    return _make_cmap(stops, 'population')

# ─────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────

def filter_rwy(df, runways: list[str] | None = None):
    """
    Filters the dataframe for specific runways. 
    Matches the 'runway' column as defined in the data contract.
    """
    if df is None or runways is None or not runways:
        return df
    
    # Ensure we only filter for runways actually present in the data to avoid empty returns
    return df[df['runway'].isin(runways)]

def filter_data(df_a, df_b, runways):
        df_a = filter_rwy(df_a, runways)
        
        # Verify dataset isn't empty after filtering
        if df_a.empty:
            print(f"Warning: Primary dataset is empty after filtering!")

        if df_b is None:
            return df_a, None
            
        df_b = filter_rwy(df_b, runways)
        if df_b.empty:
            print(f"[!] Warning: Secondary dataset is empty after filtering!")

        return df_a, df_b


def load_population(pop_path, x_path, y_path):
    Z = np.genfromtxt(pop_path, delimiter=' ')
    X = np.genfromtxt(x_path,  delimiter=' ') / MAX_DISTANCE
    Y = np.genfromtxt(y_path,  delimiter=' ') / MAX_DISTANCE
    X = np.clip(X, -1, 1)
    Y = np.clip(Y, -1, 1)
    return X, Y, Z


def load_visitation(path):
    return pd.read_parquet(path, engine='pyarrow')


def build_heatmap(df, bins=1000, x_range=(-1, 1), y_range=(-1, 1),
                  sigma_fine=0.7, sigma_glow=2.8, glow_weight=0.35, vmin_percentile=50, use_median_filter=True):
    """
    Returns a smoothed log10 density array and its (vmin, vmax).
    Also returns raw H for difference maps.
    """
    H, xe, ye = np.histogram2d(
        df['x'], df['y'], bins=bins,
        range=[x_range, y_range]
    )
    H = H.T


    if use_median_filter:
        cross_footprint = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        
        # Use footprint= instead of size=
        H = median_filter(H, footprint=cross_footprint)

    H_fine  = gaussian_filter(H.astype(float), sigma=sigma_fine)
    H_glow  = gaussian_filter(H.astype(float), sigma=sigma_glow)
    H_blend = H_fine + glow_weight * H_glow

    # Normalise so colourmap sees [0, 1] in log space
    H_log           = np.full_like(H_blend, np.nan)
    mask            = H_blend > 0
    H_log[mask]     = np.log10(H_blend[mask])

    vmin = np.nanpercentile(H_log, vmin_percentile)   # cut sparse shot noise
    vmax = np.nanmax(H_log)

    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    return H_blend, H_log, vmin, vmax, extent, H


def build_difference_map(H_a, H_b, sigma=20.0, noise_threshold=0.05):
    """
    Normalised signed difference with noise masking.
    Increase sigma to suppress granular scatter.
    """
    eps  = 1e-6
    H_af = gaussian_filter(H_a.astype(float), sigma=sigma)
    H_bf = gaussian_filter(H_b.astype(float), sigma=sigma)
    diff = (H_af - H_bf) / (H_af + H_bf + eps)

    # Mask bins where BOTH conditions have negligible visitation
    visited = (H_af + H_bf) > np.percentile(H_af + H_bf, 85)
    diff[~visited] = np.nan   # will render as background
    return diff

def compute_information_metrics(Ha, Hb = None):
    """
    Calculates Shannon Entropy for both distributions and 
    the KL Divergence D_KL(Ha || Hb).
    """
    from scipy.stats import entropy
    eps = 1e-12  # To avoid log(0)
    
    p = Ha.flatten() + eps
    p /= p.sum()

    h_a = entropy(p, base=2)

    if Hb is None:
        return h_a, None, None

    q = Hb.flatten() + eps
    q /= q.sum()

    h_b = entropy(q, base=2)
    
    # Calculate KL Divergence (bits)
    kl_div = entropy(p, q, base=2)
    
    return h_a, h_b, kl_div

def compute_tortuosity(df):
    """
    Calculates the average tortuosity (L/C) across all trajectories.
    L = total path length, C = straight-line displacement.
    """
    def get_path_metrics(group):
        dx = group['x'].diff()
        dy = group['y'].diff()
        path_length = np.sum(np.sqrt(dx**2 + dy**2))
        chord = np.sqrt((group['x'].iloc[-1] - group['x'].iloc[0])**2 + 
                        (group['y'].iloc[-1] - group['y'].iloc[0])**2)
        return path_length / chord if chord > 0 else 1.0

    # Assuming 'episode' column exists in your parquet
    tortuosities = df.groupby('episode').apply(get_path_metrics)
    return tortuosities.mean()

def compute_mean_population_exposure(df, X, Y, Z):
    """
    Maps (x, y) coordinates to the population grid Z and calculates the mean.
    Uses actual X and Y grid bounds to ensure perfect alignment.
    """
    rows, cols = Z.shape
    
    # 1. Handle both 1D arrays and 2D meshgrids (safeguard based on how genfromtxt loads)
    x_ax = X[0, :] if X.ndim == 2 else X
    y_ax = Y[:, 0] if Y.ndim == 2 else Y
    
    # 2. Get true boundaries instead of assuming exactly [-1, 1]
    x_start, x_end = x_ax[0], x_ax[-1]
    y_start, y_end = y_ax[0], y_ax[-1]
    
    # 3. Map values to grid indices based on the true grid span and round to nearest integer
    x_idx = ((df['x'] - x_start) / (x_end - x_start) * (cols - 1)).round().astype(int)
    y_idx = ((df['y'] - y_start) / (y_end - y_start) * (rows - 1)).round().astype(int)
    
    # 4. Clip to ensure we don't index out of bounds
    x_idx = x_idx.clip(0, cols - 1)
    y_idx = y_idx.clip(0, rows - 1)
    
    # 5. Extract population values at these points
    exposed_pop = Z[y_idx, x_idx]
    return np.mean(exposed_pop)

# ─────────────────────────────────────────────────────────────
# Panel rendering
# ─────────────────────────────────────────────────────────────

def _style_ax(ax, title, subtitle=None):
    """Common axis styling."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors='#3a4a5a', labelsize=7, length=3, width=0.4)
    ax.set_xlabel('Normalised x  (x300 km)', fontsize=7.5, color='#3a4a5a', labelpad=4)
    ax.set_ylabel('Normalised y  (x300 km)', fontsize=7.5, color='#3a4a5a', labelpad=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2633')
        spine.set_linewidth(0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)

    # Panel title
    ax.text(0.02, 0.985, title, transform=ax.transAxes,
            fontsize=9, fontweight='bold', color='#dde8f0',
            va='top', ha='left', fontfamily='monospace',
            path_effects=[pe.withStroke(linewidth=2.5, foreground=PANEL_BG)])
    if subtitle:
        ax.text(0.02, 0.958, subtitle, transform=ax.transAxes,
                fontsize=6.5, color='#4a6070',
                va='top', ha='left', fontfamily='monospace')

def draw_population(ax, X, Y, Z, pop_cmap):
    """Subtle but defined geography with a clear coastline."""
    # Ensure we don't log(0)
    floor = 1  # 10 people per km² is a reasonable "near zero" for population density
    Z_safe = np.clip(Z, floor, None)
    
    # 1. The Landmass Fill
    norm = LogNorm(vmin=floor, vmax=Z.max(), clip=True)
    ax.pcolormesh(X, Y, Z_safe, cmap=pop_cmap, norm=norm,
                  alpha=0.45, shading='gouraud', zorder=1, rasterized=True)

    # 2. The Coastline (The "Zero-Edge")
    # This draws a subtle line wherever population starts, effectively showing the coast
    ax.contour(X, Y, Z, levels=[floor * 2], 
               colors='#3a5a72', linewidths=0.6, alpha=0.4, zorder=2)

    # 3. Urban Density "Isobars"
    # Subtle internal contours to show city clusters
    lvls = np.logspace(np.log10(10), np.log10(Z.max()), 4)
    ax.contour(X, Y, Z_safe, levels=lvls,
               colors='#4a90b8', linewidths=0.2, alpha=0.2, zorder=2)

def draw_visitation(ax, H_log, vmin, vmax, extent, vis_cmap):
    im = ax.imshow(
        H_log, origin='lower', extent=extent,
        cmap=vis_cmap, vmin=vmin, vmax=vmax,
        alpha=0.88, # Dropped slightly for transparency
        aspect='auto', interpolation='bilinear',
        zorder=3, rasterized=True,
    )
    return im


def draw_difference(ax, diff, extent, diff_cmap, abs_max=None, H_a=None, H_b=None):
    if abs_max is None:
        abs_max = np.nanpercentile(np.abs(diff), 95)
    im = ax.imshow(
        diff, origin='lower', extent=extent,
        cmap=diff_cmap, vmin=-abs_max, vmax=abs_max,
        alpha=0.95, aspect='auto', interpolation='bilinear',
        zorder=3, rasterized=True,
    )

    # Magnitude contours: use raw arrays if provided, else fall back to |diff|
    if H_a is not None and H_b is not None:
        sigma = 3.0
        abs_diff = np.abs(gaussian_filter(H_a.astype(float), sigma=sigma) -
                          gaussian_filter(H_b.astype(float), sigma=sigma))
    else:
        abs_diff = np.abs(np.nan_to_num(diff))

    nonzero = abs_diff[abs_diff > 0]
    if nonzero.size > 0:
        levels_alphas = zip(
            np.percentile(nonzero, [80, 92, 99]),
            [0.08, 0.15, 0.30],
        )
        for level, alpha in levels_alphas:
            ax.contour(abs_diff, levels=[level], colors=['#ffffff'],
                       alpha=alpha, linewidths=0.4,
                       extent=extent, origin='lower', zorder=4)

    # Zero-crossing contour — boundary between the two conditions
    ax.contour(np.nan_to_num(diff), levels=[0], colors=['#ffffff'],
               linewidths=0.6, alpha=0.5,
               extent=extent, origin='lower', zorder=5)

    return im


def add_origin(ax, x=0.0, y=0.0, label='EHAM'):
    """Minimal crosshair origin marker."""
    size = 0.018
    ax.plot([x - size, x + size], [y, y], color='#FFD060',
            lw=0.8, alpha=0.9, zorder=10)
    ax.plot([x, x], [y - size, y + size], color='#FFD060',
            lw=0.8, alpha=0.9, zorder=10)
    ring = plt.Circle((x, y), size * 1.6, color='#FFD060', fill=False,
                       lw=0.7, alpha=0.6, zorder=10)
    ax.add_patch(ring)
    ax.text(x + 0.03, y + 0.03, label, fontsize=6.5, color='#FFD060',
            fontfamily='monospace', zorder=11,
            path_effects=[pe.withStroke(linewidth=1.8, foreground=PANEL_BG)])


def add_range_rings(ax, radii=(0.5, 1.0), labels=True):
    """Minimal range rings — just two, very subtle."""
    label_map = {0.5: '150 km', 1.0: '300 km'}
    for r in radii:
        ring = plt.Circle((0, 0), r, color="#ffffff", fill=False,
                           lw=0.35, alpha=0.3, linestyle='--', zorder=2)
        ax.add_patch(ring)
        if labels and r in label_map:
            ax.text(r * 0.707 + 0.02, r * 0.707 + 0.01, label_map[r],
                    fontsize=5.5, color="#ffffff", alpha=0.7, fontfamily='monospace', zorder=3)


def add_colorbar(fig, ax, im, label, side='right', shrink=0.65, pad=0.012):
    cb = fig.colorbar(im, ax=ax, orientation='vertical',
                      shrink=shrink, pad=pad, aspect=35,
                      location=side)
    cb.set_label(label, fontsize=7, color='#7899aa', labelpad=6)
    cb.ax.tick_params(labelsize=6, colors='#4a6070', length=2, width=0.4)
    cb.outline.set_edgecolor('#1e2633')
    cb.outline.set_linewidth(0.4)
    return cb


# ─────────────────────────────────────────────────────────────
# Single-panel figure (current use-case)
# ─────────────────────────────────────────────────────────────

def make_single_panel(df, X, Y, Z, bins=1000):
    vis_cmap = cmap_visitation()
    pop_cmap = cmap_population()

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(BG)

    H_blend, H_log, vmin, vmax, extent, Ha = build_heatmap(df, bins=bins)

    h_a, _, _ = compute_information_metrics(Ha)
    mean_tur = compute_tortuosity(df)
    pop_score = compute_mean_population_exposure(df, X, Y, Z)

    subtitle = (
        f"Agent flight paths · population density context · 300 km radius · bins={bins}\n"
        f"Shannon Entropy: {h_a:.2f} bits · Mean Tortuosity: {mean_tur:.3f} · Mean Pop. Exposure: {pop_score:.1f}"
    )

    draw_population(ax, X, Y, Z, pop_cmap)
    im = draw_visitation(ax, H_log, vmin, vmax, extent, vis_cmap)
    add_range_rings(ax)
    add_origin(ax)
    _style_ax(ax, 'SPATIAL VISITATION ANALYSIS', subtitle)

    add_colorbar(fig, ax, im, 'Visitation Frequency (log10)')

    # Figure-level suptitle
    fig.text(0.06, 0.97, 'SPATIAL VISITATION', fontsize=16,
             fontweight='bold', color='#dde8f0', fontfamily='monospace',
             va='top')
    fig.text(0.06, 0.95, 'ANALYSIS', fontsize=16,
             fontweight='bold', color='#4a90b8', fontfamily='monospace',
             va='top')

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ─────────────────────────────────────────────────────────────
# Three-panel figure  (df_a vs df_b + difference)
# ─────────────────────────────────────────────────────────────

def make_three_panel(df_a, df_b, X, Y, Z, bins=1000,
                     label_a='Condition A', label_b='Condition B'):
    vis_cmap  = cmap_visitation()
    diff_cmap = cmap_difference()
    pop_cmap  = cmap_population()

    fig = plt.figure(figsize=(22, 8))
    fig.patch.set_facecolor(BG)

    gs = GridSpec(1, 3, figure=fig,
                  left=0.04, right=0.97, bottom=0.08, top=0.88,
                  wspace=0.06)

    ax_a    = fig.add_subplot(gs[0])
    ax_b    = fig.add_subplot(gs[1])
    ax_diff = fig.add_subplot(gs[2])

    # Build arrays
    Ha, Ha_log, vmin_a, vmax_a, extent, H = build_heatmap(df_a, bins=bins)
    Hb, Hb_log, vmin_b, vmax_b, _, H     = build_heatmap(df_b, bins=bins)
    # Shared colour scale for fair comparison
    vmin = min(vmin_a, vmin_b)
    vmax = max(vmax_a, vmax_b)

    ratio = 0.25
    Ha_low = zoom(Ha, ratio)
    Hb_low = zoom(Hb, ratio)

    diff   = build_difference_map(Ha_low, Hb_low, sigma=5)

    h_a, h_b, kl_ab = compute_information_metrics(Ha, Hb)

    tur_a, tur_b = compute_tortuosity(df_a), compute_tortuosity(df_b)
    pop_a, pop_b = compute_mean_population_exposure(df_a, X, Y, Z), compute_mean_population_exposure(df_b, X, Y, Z)

    delta_tur = tur_a - tur_b
    delta_pop = pop_a - pop_b

    subtitle_a = (f'Visitation Frequency · bins={bins}\n'
                  f'Entropy: {h_a:.2f} bits · Tortuosity: {tur_a:.3f} · Pop: {pop_a:.1f}')
    
    subtitle_b = (f'Visitation Frequency · bins={bins}\n'
                  f'Entropy: {h_b:.2f} bits · Tortuosity: {tur_b:.3f} · Pop: {pop_b:.1f}')
    
    subtitle_diff = (f'KL Div: {kl_ab:.2f} · Δ Tortuosity: {delta_tur:+.3f} · '
                     f'Δ Pop: {delta_pop:+.1f}')

    for ax, H_log, label, sub in [
        (ax_a,    Ha_log, label_a, subtitle_a),
        (ax_b,    Hb_log, label_b, subtitle_b),
    ]:
        draw_population(ax, X, Y, Z, pop_cmap)
        im = draw_visitation(ax, H_log, vmin, vmax, extent, vis_cmap)
        add_range_rings(ax, labels=(ax is ax_a))
        add_origin(ax)
        _style_ax(ax, label, sub)

    # Shared colorbar for A and B
    add_colorbar(fig, ax_b, im, 'Visitation Frequency (log10)',
                 side='right', shrink=0.65)

    # Difference panel
    draw_population(ax_diff, X, Y, Z, pop_cmap)
    im_diff = draw_difference(ax_diff, diff, extent, diff_cmap, H_a=Ha, H_b=Hb)
    add_range_rings(ax_diff, labels=False)
    add_origin(ax_diff)
    _style_ax(ax_diff, f'DIFFERENCE  ({label_a} - {label_b})',
              f'normalised signed difference  ∈ [-1, +1] · 300 km radius · bins={bins*ratio}\n{subtitle_diff}')  # KL Divergence in subtitle
    add_colorbar(fig, ax_diff, im_diff,
                 f'← {label_a} |  {label_b} →',
                 side='right', shrink=0.65)

    # Suptitle
    fig.text(0.03, 0.955, 'SPATIAL VISITATION ANALYSIS',
             fontsize=14, fontweight='bold', color='#dde8f0',
             fontfamily='monospace', va='top')
    fig.text(0.03, 0.925,
             f'Agent flight paths · population density context · 300 km radius',
             fontsize=8, color='#3a5a72', fontfamily='monospace', va='top')

    return fig


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def setup_args():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        description="Spatial Visitation Analysis v3 - High-contrast trajectory visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path Group
    paths = parser.add_argument_group("Data Paths")
    paths.add_argument("--data_a", type=Path, required=True, 
                        help="Path to the primary .parquet visitation data.")
    paths.add_argument("--data_b", type=Path, default=None, 
                        help="Path to secondary .parquet data for comparison.")
    paths.add_argument("--pop_data", type=Path, default="bluesky_gym/envs/data/population_1km.csv",
                        help="Path to population density CSV.")
    paths.add_argument("--x_grid", type=Path, default="bluesky_gym/envs/data/x_array.csv",
                        help="Path to X grid CSV.")
    paths.add_argument("--y_grid", type=Path, default="bluesky_gym/envs/data/y_array.csv",
                        help="Path to Y grid CSV.")

    # Params Group
    params = parser.add_argument_group("Analysis Parameters")
    params.add_argument("--bins", type=int, default=1000, 
                        help="Resolution of the heatmap.")
    params.add_argument("--runways", type=str, nargs='+', default=None,
                        help="Filter data by specific runways (e.g., --runways 04 06 09).")
    params.add_argument("--label_a", type=str, default="Condition A", help="Label for plot A.")
    params.add_argument("--label_b", type=str, default="Condition B", help="Label for plot B.")
    
    # Output Group
    output = parser.add_argument_group("Output Options")
    output.add_argument("--output", type=Path, default="spatial_analysis.png",
                        help="Filename for the saved figure.")
    output.add_argument("--no_show", action="store_true", 
                        help="Save the figure but do not open the GUI window.")

    return parser.parse_args()

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    args = setup_args()
    plt.style.use('dark_background')

    # 1. Load Population Context
    try:
        print(f"[*] Loading population context from {args.pop_data}...")
        X, Y, Z = load_population(args.pop_data, args.x_grid, args.y_grid)
    except FileNotFoundError as e:
        print(f"[!] Error: Population data files not found: {e}")
        return

    # 2. Load Visitation Data
    print(f"[*] Loading primary dataset: {args.data_a}")
    df_a = load_visitation(args.data_a)
    
    df_b = None
    if args.data_b:
        print(f"[*] Loading secondary dataset: {args.data_b}")
        df_b = load_visitation(args.data_b)

    # 3. Filter by Runway
    if args.runways:
        print(f"[*] Filtering for runways: {args.runways}")
        df_a, df_b = filter_data(df_a, df_b, args.runways)

    # 4. Generate Figures
    if df_b is None:
        print(f"[*] Generating single-panel plot for {args.data_a.name}...")
        fig = make_single_panel(df_a, X, Y, Z, bins=args.bins)
    else:
        print(f"[*] Generating comparison plot: {args.label_a} vs {args.label_b}...")
        fig = make_three_panel(
            df_a, df_b, X, Y, Z, 
            bins=args.bins, 
            label_a=args.label_a, 
            label_b=args.label_b
        )

    # 5. Save and Show
    # Ensure output directory exists if user provided a path
    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(args.output, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"[+] Success: Figure saved to {args.output}")

    if not args.no_show:
        print("[*] Opening GUI window...")
        plt.show()

if __name__ == '__main__':
    main()