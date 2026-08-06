"""
Spatial Visitation Analysis — Publication Ready Edition
======================================================
Clean, professional, high-contrast light-theme figure optimized for academic papers.
Architecture is ready for 3-subplot use (two visitation maps + difference).

Usage (single plot):
    python spatial_visitation_analysis.py --data_a path/to/data.parquet

Usage (3-subplot comparison):
    python spatial_visitation_analysis.py --data_a data_a.parquet --data_b data_b.parquet
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, median_filter
import warnings
warnings.filterwarnings('ignore')

# Publication-grade rcParams configuration
matplotlib.rcParams.update({
    'figure.dpi':        180,
    'savefig.dpi':       300,
    'font.family':       'serif',       # Standard serif font for academic papers
    'axes.linewidth':    0.6,           # Crisp border lines
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
})

MAX_DISTANCE = 300_000   # metres
BG           = '#ffffff'   # Paper white
PANEL_BG     = '#ffffff'   # Paper white

# ─────────────────────────────────────────────────────────────
# Academic Colormaps
# ─────────────────────────────────────────────────────────────

def _make_cmap(stops, name, N=512):
    pos  = [s[0] for s in stops]
    rgba = np.array([s[1] for s in stops], dtype=float)
    out  = np.zeros((N, 4))
    for ch in range(4):
        out[:, ch] = np.interp(np.linspace(0, 1, N), pos, rgba[:, ch])
    return mcolors.ListedColormap(out, name=name)


def cmap_visitation():
    """Light-themed sequential map: white/transparent → light blue → deep indigo → black."""
    stops = [
        (0.00, [1.00, 1.00, 1.00, 0.00]),   # Completely transparent
        (0.05, [0.90, 0.92, 0.98, 0.70]),   # Faint blue-grey tint
        (0.25, [0.65, 0.55, 0.85, 0.85]),   # Soft lavender
        (0.50, [0.40, 0.20, 0.70, 0.95]),   # Rich purple
        (0.75, [0.15, 0.10, 0.50, 1.00]),   # Deep indigo
        (1.00, [0.02, 0.02, 0.15, 1.00]),   # Near-black navy (Peak density)
    ]
    return _make_cmap(stops, 'visitation')


def cmap_difference():
    """Diverging for light theme: crisp teal → white neutral (zero) → vivid dark red."""
    stops = [
        (0.00, [0.00, 0.45, 0.55, 1.00]),   # Deep teal
        (0.30, [0.30, 0.65, 0.75, 1.00]),   # Mid teal
        (0.46, [0.90, 0.93, 0.95, 1.00]),   # Very light cool grey
        (0.50, [0.97, 0.97, 0.97, 1.00]),   # Clean near-white neutral (Zero line)
        (0.54, [0.95, 0.91, 0.90, 1.00]),   # Very light warm grey
        (0.70, [0.80, 0.35, 0.30, 1.00]),   # Mid coral/red
        (1.00, [0.65, 0.05, 0.05, 1.00]),   # Deep rich red
    ]
    return _make_cmap(stops, 'difference')


def cmap_population():
    """
    Ultra-Minimalist Monochromatic Palette:
    Translucent Silver → Light Platinum → Muted Steel → Dark Asphalt → Deep Graphite.
    Completely eliminates chromatic competition, letting purple tracks be the only color.
    """
    stops = [
        (0.00, [0.93, 0.94, 0.96, 1.00]),   # 1. Translucent Silver
        (0.30, [0.81, 0.83, 0.86, 1.00]),   # 2. Light Platinum
        (0.60, [0.56, 0.58, 0.62, 1.00]),   # 3. Muted Steel
        (0.85, [0.32, 0.35, 0.39, 1.00]),   # 4. Dark Asphalt
        (1.00, [0.14, 0.16, 0.18, 1.00]),   # 5. Deep Graphite Core
    ]
    return _make_cmap(stops, 'population_platinum_graphite')

# ─────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────

def filter_rwy(df, runways: list[str] | None = None):
    if df is None or runways is None or not runways:
        return df
    return df[df['runway'].isin(runways)]

def filter_data(df_a, df_b, runways):
    df_a = filter_rwy(df_a, runways)
    if df_a.empty:
        print(f"Warning: Primary dataset is empty after filtering!")
    if df_b is None:
        return df_a, None
    df_b = filter_rwy(df_b, runways)
    if df_b.empty:
        print(f"[!] Warning: Secondary dataset is empty after filtering!")
    return df_a, df_b

def compute_optimal_bins(df, x_range=(-1, 1), y_range=(-1, 1)):
    n = len(df)
    if n == 0:
        return 1000, 1000
    iqr_x = np.percentile(df['x'], 75) - np.percentile(df['x'], 25)
    iqr_y = np.percentile(df['y'], 75) - np.percentile(df['y'], 25)
    if iqr_x == 0: iqr_x = np.std(df['x']) * 1.349
    if iqr_y == 0: iqr_y = np.std(df['y']) * 1.349
    h_x = 2 * iqr_x * (n ** (-1/3))
    h_y = 2 * iqr_y * (n ** (-1/3))
    if h_x == 0: h_x = 0.01
    if h_y == 0: h_y = 0.01
    span_x = x_range[1] - x_range[0]
    span_y = y_range[1] - y_range[0]
    bins_x = int(np.ceil(span_x / h_x))
    bins_y = int(np.ceil(span_y / h_y))
    return [min(bins_x, 3000), min(bins_y, 3000)]

def load_population(pop_path, x_path, y_path):
    Z = np.genfromtxt(pop_path, delimiter=' ')
    X = np.genfromtxt(x_path,  delimiter=' ') / MAX_DISTANCE
    Y = np.genfromtxt(y_path,  delimiter=' ') / MAX_DISTANCE
    X = np.clip(X, -1, 1)
    Y = np.clip(Y, -1, 1)
    return X, Y, Z

def load_visitation(path):
    df = pd.read_parquet(path, engine="pyarrow")
    if "is_success" in df.columns:
        df = df[df["is_success"] == True]
    return df

def build_heatmap(df, bins=1000, x_range=(-1, 1), y_range=(-1, 1),
                  sigma_fine=0.7, sigma_glow=2.8, glow_weight=0.35, vmin_percentile=50, use_median_filter=True):
    H, xe, ye = np.histogram2d(df['x'], df['y'], bins=bins, range=[x_range, y_range])
    H = H.T
    if use_median_filter:
        cross_footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        H = median_filter(H, footprint=cross_footprint)
    H_fine  = gaussian_filter(H.astype(float), sigma=sigma_fine)
    H_glow  = gaussian_filter(H.astype(float), sigma=sigma_glow)
    H_blend = H_fine + glow_weight * H_glow
    H_log   = np.full_like(H_blend, np.nan)
    mask    = H_blend > 0
    H_log[mask] = np.log10(H_blend[mask])
    vmin = np.nanpercentile(H_log, vmin_percentile)
    vmax = np.nanmax(H_log)
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    return H_blend, H_log, vmin, vmax, extent, H

def build_difference_map(H_a, H_b, sigma=20.0, noise_threshold=0.05):
    eps  = 1e-6
    H_af = gaussian_filter(H_a.astype(float), sigma=sigma)
    H_bf = gaussian_filter(H_b.astype(float), sigma=sigma)
    diff = (H_af - H_bf) / (H_af + H_bf + eps)
    visited = (H_af + H_bf) > np.percentile(H_af + H_bf, 85)
    diff[~visited] = np.nan
    return diff

def compute_information_metrics(Ha, Hb = None):
    from scipy.stats import entropy
    eps = 1e-12
    p = Ha.flatten() + eps
    p /= p.sum()
    h_a = entropy(p, base=2)
    if Hb is None:
        return h_a, None, None
    q = Hb.flatten() + eps
    q /= q.sum()
    h_b = entropy(q, base=2)
    kl_div = entropy(p, q, base=2)
    return h_a, h_b, kl_div

def compute_tortuosity(df):
    grouped = df.groupby('episode')
    # Per-row step length (NaN on each group's first row, skipped by .sum()
    # the same way np.sum(group['x'].diff()...) skipped it per-group above).
    step_length = np.sqrt(grouped['x'].diff()**2 + grouped['y'].diff()**2)
    path_length = step_length.groupby(df['episode']).sum()

    firsts = grouped[['x', 'y']].first()
    lasts = grouped[['x', 'y']].last()
    chord = np.sqrt((lasts['x'] - firsts['x'])**2 + (lasts['y'] - firsts['y'])**2)

    tortuosities = (path_length / chord).where(chord > 0, 1.0)
    return tortuosities.mean()

def compute_mean_population_exposure(df, X, Y, Z):
    rows, cols = Z.shape
    x_ax = X[0, :] if X.ndim == 2 else X
    y_ax = Y[:, 0] if Y.ndim == 2 else Y
    x_start, x_end = x_ax[0], x_ax[-1]
    y_start, y_end = y_ax[0], y_ax[-1]
    x_idx = ((df['x'] - x_start) / (x_end - x_start) * (cols - 1)).round().astype(int)
    y_idx = ((df['y'] - y_start) / (y_end - y_start) * (rows - 1)).round().astype(int)
    x_idx = x_idx.clip(0, cols - 1)
    y_idx = y_idx.clip(0, rows - 1)
    exposed_pop = Z[y_idx, x_idx]
    return np.mean(exposed_pop)

# ─────────────────────────────────────────────────────────────
# Panel rendering
# ─────────────────────────────────────────────────────────────

def _style_ax(ax, title, subtitle=None):
    """Common axis styling optimized for academic publication layout."""
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors='#222222', labelsize=8, length=3.5, width=0.5)
    ax.set_xlabel('Normalised x  (x300 km)', fontsize=8.5, color='#222222', labelpad=4)
    ax.set_ylabel('Normalised y  (x300 km)', fontsize=8.5, color='#222222', labelpad=4)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
        spine.set_linewidth(0.6)
    ax.set_aspect('equal')
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)

    # Panel title
    ax.text(0.02, 0.985, title, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color='#111111',
            va='top', ha='left', fontfamily='serif',
            path_effects=[pe.withStroke(linewidth=2.5, foreground=PANEL_BG)])
    if subtitle:
        ax.text(0.02, 0.952, subtitle, transform=ax.transAxes,
                fontsize=7, color='#555555',
                va='top', ha='left', fontfamily='serif')

def draw_population(ax, X, Y, Z, pop_cmap):
    """
    Maintains perfect island separation via binary masking and replaces the 
    muddy grey land texture with a premium tactical midnight-blue canvas.
    """
    # 1. Generate a strict binary land mask (1 where land exists, 0 for sea)
    land_mask = (Z > 0).astype(float)
    
    # 2. Sub-pixel anti-aliasing strictly for the coastlines.
    # Tracing the 0.5 midpoint keeps all islands perfectly separated and razor-sharp.
    Z_coast = gaussian_filter(land_mask, sigma=0.5)
    
    # 3. PREMIUM TACTICAL FILL: Replaces the unappealing grey landmass texture.
    # This uses a deep, solid midnight-indigo that contrasts beautifully 
    # with the purple flight tracks without creating a muddy overlay.
    ax.contourf(X, Y, Z_coast, levels=[0.01, 1.0], 
                colors=["#dee2ea"], alpha=0.95, zorder=1)
    
    # 4. Crisp, luminous geographic coastline
    ax.contour(X, Y, Z_coast, levels=[0.01], 
               colors="#000000", linewidths=0.8, alpha=0.7, zorder=3.5)
    
    # 5. Smooth the population data heavily *only* for the urban hotspots
    Z_urban = gaussian_filter(Z, sigma=4.0)
    
    # 6. Smooth metropolitan glow shapes UNDER the flight paths.
    # We skip low rural densities completely (>150) so the rural land stays clean.
    urban_levels = [150, 500, 1500, np.max(Z_urban)]
    norm = LogNorm(vmin=150, vmax=Z.max(), clip=True)
    
    ax.contourf(X, Y, Z_urban, levels=urban_levels, cmap=pop_cmap, norm=norm,
                alpha=0.45, zorder=1.5)


def draw_visitation(ax, H_log, vmin, vmax, extent, vis_cmap):
    im = ax.imshow(
        H_log, origin='lower', extent=extent,
        cmap=vis_cmap, vmin=vmin, vmax=vmax,
        alpha=0.9, # Dropped from 0.88 to let underlying population blend through
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
            [0.15, 0.25, 0.40],
        )
        for level, alpha in levels_alphas:
            ax.contour(abs_diff, levels=[level], colors=['#333333'],
                       alpha=alpha, linewidths=0.4,
                       extent=extent, origin='lower', zorder=4)

    # Zero-crossing contour (dark track split line)
    ax.contour(np.nan_to_num(diff), levels=[0], colors=['#222222'],
               linewidths=0.6, alpha=0.6,
               extent=extent, origin='lower', zorder=5)

    return im

def overlay_delta_entropy_contours(ax, Ha, Hb, extent, eps = 1e-12):
    pa = Ha / (Ha.sum() + eps)
    pb = Hb / (Hb.sum() + eps)
    
    ha_local = -pa * np.log2(pa, out=np.zeros_like(pa), where=(pa > 0))
    hb_local = -pb * np.log2(pb, out=np.zeros_like(pb), where=(pb > 0))
    delta_h_matrix = ha_local - hb_local
    
    pos_mask = delta_h_matrix > 0
    neg_mask = delta_h_matrix < 0

    legend_handles = []
    
    # Orange-red contours for dispersion increases
    if np.any(pos_mask):
        pos_max = np.max(delta_h_matrix[pos_mask])
        ax.contour(delta_h_matrix, levels=[pos_max * 0.3, pos_max * 0.6, pos_max * 0.9], 
                   colors=['#d95f02'], linestyles='dashed', linewidths=0.5, 
                   extent=extent, origin='lower', zorder=6)
        legend_handles.append(
            Line2D([0], [0], color='#d95f02', linestyle='dashed', linewidth=0.8, 
                   label='Δh > 0 : Tactical Path Dispersal (Dynamic Vectoring)')
        )
                        
    # Blue contours for flow channelisation
    if np.any(neg_mask):
        neg_min = np.min(delta_h_matrix[neg_mask])
        ax.contour(delta_h_matrix, levels=[neg_min * 0.9, neg_min * 0.6, neg_min * 0.3], 
                   colors=['#1f78b4'], linestyles='dashed', linewidths=0.5, 
                   extent=extent, origin='lower', zorder=6)
        legend_handles.append(
            Line2D([0], [0], color='#1f78b4', linestyle='dashed', linewidth=0.8, 
                   label='Δh < 0 : Flow Channelisation (Highly Structured Corridors)')
        )

    legend_handles.append(
        Line2D([0], [0], color='#222222', alpha=0.6, linewidth=0.6, 
               label='Volumetric Traffic Equilibrium Axis')
    )

    legend_handles.append(
        Line2D([0], [0], color='#333333', alpha=0.3, linewidth=0.5, 
               label='Airspace Reconfiguration Envelopes (80% / 92% / 99% Shift)')
    )

    ax.legend(
        handles=legend_handles,
        loc='lower left',
        frameon=True,
        facecolor='#ffffff',      
        edgecolor='#cccccc',      
        fontsize=6.5,
        labelcolor='#222222',     
        handlelength=2.5,
        borderpad=0.8,
        handletextpad=0.6
    )

def add_origin(ax, x=0.0, y=0.0, label='EHAM'):
    """Minimal paper-ready crosshair origin marker (Burgundy/Deep Red)."""
    size = 0.018
    ax.plot([x - size, x + size], [y, y], color='#990000',
            lw=0.8, alpha=0.9, zorder=10)
    ax.plot([x, x], [y - size, y + size], color='#990000',
            lw=0.8, alpha=0.9, zorder=10)
    ring = plt.Circle((x, y), size * 1.6, color='#990000', fill=False,
                       lw=0.7, alpha=0.6, zorder=10)
    ax.add_patch(ring)
    ax.text(x + 0.03, y + 0.03, label, fontsize=7, color='#990000',
            fontfamily='serif', fontweight='bold', zorder=11,
            path_effects=[pe.withStroke(linewidth=1.8, foreground=PANEL_BG)])


def add_range_rings(ax, radii=(0.5, 1.0), labels=True):
    """Subtle dashed charcoal range rings."""
    label_map = {0.5: '150 km', 1.0: '300 km'}
    for r in radii:
        ring = plt.Circle((0, 0), r, color="#444444", fill=False,
                           lw=0.4, alpha=0.4, linestyle='--', zorder=2)
        ax.add_patch(ring)
        if labels and r in label_map:
            ax.text(r * 0.707 + 0.02, r * 0.707 + 0.01, label_map[r],
                    fontsize=6, color="#444444", alpha=0.8, fontfamily='serif', zorder=3)


def add_colorbar(fig, ax, im, label, side='right', shrink=0.65, pad=0.012):
    cb = fig.colorbar(im, ax=ax, orientation='vertical',
                      shrink=shrink, pad=pad, aspect=35,
                      location=side)
    cb.set_label(label, fontsize=8, color='#222222', labelpad=6)
    cb.ax.tick_params(labelsize=7, colors='#444444', length=2, width=0.4)
    cb.outline.set_edgecolor('#cccccc')
    cb.outline.set_linewidth(0.5)
    return cb


# ─────────────────────────────────────────────────────────────
# Publication Figures Layout Architecture
# ─────────────────────────────────────────────────────────────

def make_single_panel(df, X, Y, Z, bins=1000):
    vis_cmap = cmap_visitation()
    pop_cmap = cmap_population()

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(BG)

    H_blend, H_log, vmin, vmax, extent, Ha = build_heatmap(df, bins=bins)

    h_a, _, _ = compute_information_metrics(Ha)
    mean_tur = compute_tortuosity(df)
    pop_score = compute_mean_population_exposure(df, X, Y, Z)

    bin_str = f"{bins[0]}x{bins[1]}" if isinstance(bins, list) else str(bins)

    subtitle = (
        f"Agent flight paths · population density context · 300 km radius · bins={bin_str}\n"
        f"Shannon Entropy: {h_a:.2f} bits · Mean Tortuosity: {mean_tur:.3f} · Mean Pop. Exposure: {pop_score:.1f}"
    )

    draw_population(ax, X, Y, Z, pop_cmap)
    im = draw_visitation(ax, H_log, vmin, vmax, extent, vis_cmap)
    add_range_rings(ax)
    add_origin(ax)
    _style_ax(ax, 'SPATIAL VISITATION ANALYSIS', subtitle)

    add_colorbar(fig, ax, im, 'Visitation Frequency (log10)')

    fig.text(0.06, 0.96, 'SPATIAL VISITATION ANALYSIS', fontsize=14,
             fontweight='bold', color='#111111', fontfamily='serif', va='top')

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    return fig

def display_metrics_table(kl_ab, delta_tur, delta_pop, delta_H, label_a="Condition A", label_b="Condition B"):
    if kl_ab >= 0.5:
        kl_interpret = "Substantial, non-trivial structural shift in routing"
    elif kl_ab >= 0.1:
        kl_interpret = "Moderate structural shift in routing"
    else:
        kl_interpret = "Negligible structural variation between policies"

    if delta_tur >= 0:
        tur_interpret = f"{delta_tur*100:.1f}% average path elongation (stretching)"
    else:
        tur_interpret = f"{abs(delta_tur)*100:.1f}% average path compression (shorter paths)"

    if delta_pop <= 0:
        pop_interpret = f"Strategic noise-abatement routing discovered ({abs(delta_pop):.1f} lower exposure)"
    else:
        pop_interpret = f"Increased population exposure footprint (+{delta_pop:.1f} exposure score)"

    if delta_H <= 0:
        ent_interpret = "Macro-scale traffic flow channelisation (more orderly tracks)"
    else:
        ent_interpret = "Increased trajectory randomness and structural dispersion"

    print("\n" + "="*115)
    print(f"               SUMMARY OF EMPIRICAL METRICS ({label_a} vs {label_b})".center(115))
    print("="*115)
    print(f"  {'Metric':<35} | {'Value':<12} | {'Operational Interpretation':<60}")
    print("-"*115)
    print(f"  KL Divergence (DKL)                 | {kl_ab:<7.2f} bits | {kl_interpret:<60}")
    print(f"  Δ Tortuosity (Δτ)                   | {delta_tur:<+7.3f}      | {tur_interpret:<60}")
    print(f"  Δ Population Exposure (ΔPop)        | {delta_pop:<+7.1f}      | {pop_interpret:<60}")
    print(f"  Δ Global Shannon Entropy (ΔH)       | {delta_H:<+7.2f} bits | {ent_interpret:<60}")
    print("="*115 + "\n")


def make_three_panel(df_a, df_b, X, Y, Z, bins=1000, label_a='Condition A', label_b='Condition B'):
    vis_cmap  = cmap_visitation()
    diff_cmap = cmap_difference()
    pop_cmap  = cmap_population()

    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor(BG)

    gs = GridSpec(1, 3, figure=fig, left=0.04, right=0.96, bottom=0.08, top=0.86, wspace=0.06)

    ax_a    = fig.add_subplot(gs[0])
    ax_b    = fig.add_subplot(gs[1])
    ax_diff = fig.add_subplot(gs[2])

    Ha, Ha_log, vmin_a, vmax_a, extent, _ = build_heatmap(df_a, bins=bins)
    Hb, Hb_log, vmin_b, vmax_b, _, _     = build_heatmap(df_b, bins=bins)
    
    vmin = min(vmin_a, vmin_b)
    vmax = max(vmax_a, vmax_b)

    bx, by = bins if isinstance(bins, list) else (bins, bins)
    diff_bins_str = f"{bx}x{by}"
    sigma_diff = (20.0 * bx / 1000.0, 20.0 * by / 1000.0)

    diff = build_difference_map(Ha, Hb, sigma=sigma_diff)
    h_a, h_b, kl_ab = compute_information_metrics(Ha, Hb)

    tur_a, tur_b = compute_tortuosity(df_a), compute_tortuosity(df_b)
    pop_a, pop_b = compute_mean_population_exposure(df_a, X, Y, Z), compute_mean_population_exposure(df_b, X, Y, Z)

    delta_tur = tur_a - tur_b
    delta_pop = pop_a - pop_b
    delta_H = h_a - h_b

    display_metrics_table(kl_ab=kl_ab, delta_tur=delta_tur, delta_pop=delta_pop, delta_H=delta_H, label_a=label_a, label_b=label_b)

    bin_str = f"{bins[0]}x{bins[1]}" if isinstance(bins, list) else str(bins)

    subtitle_a = (f'Visitation Frequency · bins={bin_str}\n'
                  f'Entropy: {h_a:.2f} bits · Tortuosity: {tur_a:.3f} · Pop: {pop_a:.1f}')
    subtitle_b = (f'Visitation Frequency · bins={bin_str}\n'
                  f'Entropy: {h_b:.2f} bits · Tortuosity: {tur_b:.3f} · Pop: {pop_b:.1f}')
    subtitle_diff = (f'KL Div: {kl_ab:.2f} · Δ Tortuosity: {delta_tur:+.3f} · '
                     f'Δ Pop: {delta_pop:+.1f} · Δ Entropy: {delta_H:+.3f} bits')

    for ax, H_log, label, sub in [
        (ax_a,    Ha_log, label_a, subtitle_a),
        (ax_b,    Hb_log, label_b, subtitle_b),
    ]:
        draw_population(ax, X, Y, Z, pop_cmap)
        im = draw_visitation(ax, H_log, vmin, vmax, extent, vis_cmap)
        add_range_rings(ax, labels=(ax is ax_a))
        add_origin(ax)
        _style_ax(ax, label, sub)

    add_colorbar(fig, ax_b, im, 'Visitation Frequency (log10)', side='right', shrink=0.65)
    cb_dummy = add_colorbar(fig, ax_a, im, '', side='right', shrink=0.65)
    cb_dummy.ax.set_visible(False)

    # Difference panel
    draw_population(ax_diff, X, Y, Z, pop_cmap)
    im_diff = draw_difference(ax_diff, diff, extent, diff_cmap, H_a=Ha, H_b=Hb)
    overlay_delta_entropy_contours(ax_diff, Ha, Hb, extent)
    add_range_rings(ax_diff, labels=False)
    add_origin(ax_diff)

    _style_ax(ax_diff, f'DIFFERENCE  ({label_a} - {label_b})',
              f'Normalised signed difference  ∈ [-1, +1] · 300 km radius · bins={diff_bins_str}\n{subtitle_diff}')
    add_colorbar(fig, ax_diff, im_diff, f'← {label_b} |  {label_a} →', side='right', shrink=0.65)

    # Global Figure Header
    fig.text(0.03, 0.95, 'SPATIAL VISITATION ANALYSIS', fontsize=14, fontweight='bold', color='#111111', fontfamily='serif', va='top')
    fig.text(0.03, 0.92, f'Agent flight paths · population density context · 300 km radius', fontsize=8.5, color='#555555', fontfamily='serif', va='top')

    return fig


def setup_args():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        description="Spatial Visitation Analysis - Light-theme academic publication layout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    paths = parser.add_argument_group("Data Paths")
    paths.add_argument("--data_a", type=Path, required=True, help="Path to primary .parquet data.")
    paths.add_argument("--data_b", type=Path, default=None, help="Path to secondary .parquet data.")
    paths.add_argument("--pop_data", type=Path, default="bluesky_gym/envs/data/population_1km.csv")
    paths.add_argument("--x_grid", type=Path, default="bluesky_gym/envs/data/x_array.csv")
    paths.add_argument("--y_grid", type=Path, default="bluesky_gym/envs/data/y_array.csv")

    params = parser.add_argument_group("Analysis Parameters")
    params.add_argument("--bins", type=int, default=None, help="Map grid resolution.")
    params.add_argument("--runways", type=str, nargs='+', default=None, help="Filter by runways.")
    params.add_argument("--label_a", type=str, default="Condition A")
    params.add_argument("--label_b", type=str, default="Condition B")
    
    output = parser.add_argument_group("Output Options")
    output.add_argument("--output", type=Path, default="spatial_analysis.png")
    output.add_argument("--no_show", action="store_true")

    return parser.parse_args()


def main():
    args = setup_args()
    plt.style.use('default')  # Switched to default white background profile

    # Load Context
    try:
        X, Y, Z = load_population(args.pop_data, args.x_grid, args.y_grid)
    except FileNotFoundError as e:
        print(f"[!] Error loading grid data: {e}")
        return

    # Load Data
    df_a = load_visitation(args.data_a)
    df_b = load_visitation(args.data_b) if args.data_b else None

    if args.runways:
        df_a, df_b = filter_data(df_a, df_b, args.runways)

    calc_bins = args.bins if args.bins is None else compute_optimal_bins(df_a)
    if args.bins is None:
        calc_bins = compute_optimal_bins(df_a)
    else:
        calc_bins = args.bins

    # Figure Type Dispatch
    if df_b is None:
        fig = make_single_panel(df_a, X, Y, Z, bins=calc_bins)
    else:
        fig = make_three_panel(df_a, df_b, X, Y, Z, bins=calc_bins, label_a=args.label_a, label_b=args.label_b)

    if args.output.parent:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(args.output, dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"[+] Success: Academic-ready figure saved to {args.output}")

    if not args.no_show:
        plt.show()

if __name__ == '__main__':
    main()