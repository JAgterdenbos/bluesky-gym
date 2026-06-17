import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# ── 1. Style & Publication Configuration (Thesis Mode) ───────────────────────
matplotlib.rcParams.update({
    'figure.dpi': 300,           # High-resolution for print
    'savefig.dpi': 300,
    'font.family': 'serif',       # Formal academic serif
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'axes.linewidth': 0.7,       # Crisp axis lines
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
})

# Professional Academic Palette
BG_WHITE = "#ffffff"
GRID_COLOR = "#ececec"
TEXT_MAIN = "#111111"
TEXT_MUTED = "#444444"

# Tolerance Zone Fills (Soft, desaturated tints)
TOLERANCE_FILL = "#f4f9f4"  # Soft green tint for inside tolerance
VIOLATION_FILL = "#fdf2f2"  # Soft red/pink tint for outside tolerance

# Dataset Distinct Line Colors (Colorblind-friendly publication cycle)
DATASET_COLORS = ["#004488", "#bb5566", "#44aa99", "#ddaacc", "#999933"]

DELAY_THRESHOLD = 60.0  # 1 minute represented in seconds
MAX_TIME = 3600 * 6     # 6 hours scale factor


def clean_axis(ax: plt.Axes) -> None:
    """Applies modern minimalist academic formatting to an axis."""
    ax.set_facecolor(BG_WHITE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, direction='out')
    
    # Remove top and right spines to reduce visual clutter
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor("#bbbbbb")
    ax.spines['bottom'].set_edgecolor("#bbbbbb")
    
    ax.xaxis.label.set_color(TEXT_MAIN)
    ax.yaxis.label.set_color(TEXT_MAIN)
    ax.grid(True, linestyle="--", linewidth=0.5, color=GRID_COLOR, zorder=0)


def ceil_to_nearest_half(val: float) -> float:
    """Helper to round up to the nearest 0.5 for clean axis boundaries."""
    return float(np.ceil(val * 2) / 2)


def process_dataset(data_path: Path, rwy: Optional[str] = None) -> np.ndarray:
    """Loads and extracts final episode delays from a given file."""
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
        
    if rwy and 'runway' in df.columns:
        df = df[df['runway'] == rwy]
        
    if 'episode' in df.columns:
        episode_summary = df.groupby('episode').last().reset_index()
    else:
        episode_summary = df

    if 'delay' not in episode_summary.columns:
        raise KeyError(f"Could not find 'delay' column in {data_path}")

    delays_sec = episode_summary['delay'].dropna().to_numpy(np.float32) * MAX_TIME
    return delays_sec


def render_metrics_table(ax_table: plt.Axes, table_rows: List[List[str]]) -> None:
    """Generates a structured table matrix below the plot with a booktabs design."""
    # Hide the bounding framework of the table container axis completely
    ax_table.axis('off')
    
    col_headers = [
        "Dataset Source", "Total Samples (N)", "On-Time Rate", 
        "Delayed Rate", "MAE", "Max Abs Delay", "Clipped (%)"
    ]
    
    metrics_table = ax_table.table(
        cellText=table_rows,
        colLabels=col_headers,
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0] # Force table to tightly fill the allocated axes layout safely
    )
    
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(8.5)
    
    num_rows = max(idx[0] for idx in metrics_table.get_celld().keys()) + 1
    num_cols = max(idx[1] for idx in metrics_table.get_celld().keys()) + 1

    # Apply a high-end journal "booktabs" design look to the metrics matrix
    for (row_idx, col_idx), cell in metrics_table.get_celld().items():
        cell.set_text_props(color=TEXT_MAIN, fontfamily='serif')
        cell.set_facecolor(BG_WHITE)
        
        # Add padding to prevent text from overlapping the vertical borders
        cell.PAD = 0.05
        
        # Determine which borders to draw
        edges = ''
        
        # Horizontal lines for structure
        if row_idx == 0:
            cell.set_text_props(fontweight='bold', color=TEXT_MAIN)
            edges += 'T'  # Top of header
        if row_idx == 1:
            edges += 'T'  # Bottom of header
        if row_idx == num_rows - 1:
            edges += 'B'  # Bottom of table
            
        # Vertical lines for the outer frame boundaries
        if col_idx == 0:
            edges += 'L'  # Leftmost column edge
        if col_idx == num_cols - 1:
            edges += 'R'  # Rightmost column edge
            
        # Apply the explicit boundaries with a uniform line weight so corners perfectly meet
        cell.visible_edges = edges
        cell.set_linewidth(1.0)
        cell.set_edgecolor(TEXT_MAIN)


def generate_multi_plot(
    data_inputs: Dict[str, Path],
    out_dir: Path = Path("."),
    rwy: Optional[str] = None,
    figure_title: Optional[str] = None,
    show_zone_labels: bool = True,
) -> None:
    """Generates a single overlaid axis plot with an elegant structured metrics table below."""
    if not figure_title:
        figure_title = "Arrival Delay Distribution Analysis"

    # Create the base figure window canvas
    fig = plt.figure(figsize=(11.0, 6.2), facecolor=BG_WHITE)
    
    # ── 2. Absolute Positioning Layout Control ───────────────────────────────
    # Coordinates format: [left, bottom, width, height] as percentages (0.0 to 1.0)
    # The plot is constrained to the left to leave room for the right legend
    ax = fig.add_axes([0.06, 0.34, 0.68, 0.52])
    
    # The table axis spans nearly the full width of the entire image canvas
    ax_table = fig.add_axes([0.06, 0.06, 0.88, 0.16])
    
    clean_axis(ax)
        
    threshold_min = DELAY_THRESHOLD / 60.0
    per_dataset_95ths = []
    dataset_data_cached = []

    # First Pass: Load all records to establish uniform robust layout ranges
    for label, path in data_inputs.items():
        delays_sec = process_dataset(path, rwy=rwy)
        delays_min = delays_sec / 60.0
        dataset_data_cached.append((label, delays_sec, delays_min))
        per_dataset_95ths.append(np.percentile(np.abs(delays_min), 95))

    # Robust scaling logic: base sizing on the median 95th percentile across groups
    robust_max = np.median(per_dataset_95ths)
    max_val = max(robust_max, threshold_min * 5) # Ensure it spans at least 5x threshold minimums
    x_limit = ceil_to_nearest_half(max_val)

    # Core background zones (Drawn once inside the single axis)
    ax.axvspan(-threshold_min, threshold_min, facecolor=TOLERANCE_FILL, alpha=0.6, zorder=1)
    ax.axvspan(-x_limit, -threshold_min, facecolor=VIOLATION_FILL, alpha=0.5, zorder=1)
    ax.axvspan(threshold_min, x_limit, facecolor=VIOLATION_FILL, alpha=0.5, zorder=1)
    
    ax.axvline(threshold_min, color="#ccaaaa", linestyle=":", linewidth=1.0, zorder=2)
    ax.axvline(-threshold_min, color="#ccaaaa", linestyle=":", linewidth=1.0, zorder=2)

    if show_zone_labels:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        label_y = 0.93
        ax.text(0, label_y, "ON-TIME", transform=trans, color="#2e7d32", 
                fontsize=8, fontweight='bold', ha='center', va='top', alpha=0.6)
        ax.text((-x_limit - threshold_min) / 2, label_y, "EARLY", transform=trans, color="#c62828", 
                fontsize=8, fontweight='bold', ha='center', va='top', alpha=0.6)
        ax.text((x_limit + threshold_min) / 2, label_y, "LATE", transform=trans, color="#c62828", 
                fontsize=8, fontweight='bold', ha='center', va='top', alpha=0.6)

    global_max_y = 0.5
    table_rows = []

    # Second Pass: Overlay distributions and extract performance summaries
    for idx, (label, delays_sec, delays_min) in enumerate(dataset_data_cached):
        abs_delays_sec = np.abs(delays_sec)
        is_delayed = abs_delays_sec > DELAY_THRESHOLD
        delay_rate = np.mean(is_delayed) * 100
        on_time_rate = 100.0 - delay_rate
        mae_min = np.mean(abs_delays_sec) / 60.0
        max_abs_delay_min = np.max(np.abs(delays_min))
        
        # Calculate exactly how much data is clipped off the sides of the plot range
        is_out_of_bounds = np.abs(delays_min) > x_limit
        out_of_bounds_count = np.sum(is_out_of_bounds)
        out_of_bounds_rate = (out_of_bounds_count / len(delays_min)) * 100
        
        # Append stats directly to the master list container for the summary table
        table_rows.append([
            label, 
            f"{len(delays_sec):,}", 
            f"{on_time_rate:.1f}%", 
            f"{delay_rate:.1f}%", 
            f"{mae_min:.2f} min",
            f"{max_abs_delay_min:.1f} min",
            f"{out_of_bounds_rate:.1f}%"
        ])
        
        color = DATASET_COLORS[idx % len(DATASET_COLORS)]
        
        if len(delays_min) > 1:
            kde = gaussian_kde(delays_min)
            x_vals = np.linspace(-x_limit, x_limit, 500)
            y_vals = kde(x_vals)
            ax.plot(x_vals, y_vals, color=color, linewidth=1.75, label=label, zorder=4)
            ax.fill_between(x_vals, 0, y_vals, color=color, alpha=0.12, zorder=3)
            global_max_y = max(global_max_y, np.max(y_vals))
        else:
            counts, _, _ = ax.hist(delays_min, bins=30, range=(-x_limit, x_limit), density=True, 
                                   color=color, alpha=0.4, zorder=4, label=label)
            if len(counts) > 0:
                global_max_y = max(global_max_y, np.max(counts))

    ax.set_ylim(0, global_max_y * 1.25)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylabel("Density", fontsize=9, color=TEXT_MUTED)
    ax.set_xlabel(r"Arrival Delay $\Delta t$ [minutes]", fontsize=10)

    # ── 3. Structural Legend Modification (Right Side Space) ─────────────────
    legend_elements = [
        Patch(facecolor=TOLERANCE_FILL, edgecolor='#cccccc', linestyle=':', label=f'Within Tolerance ($|\\Delta t| \\leq {threshold_min:g}$m)'),
        Patch(facecolor=VIOLATION_FILL, edgecolor='#cccccc', linestyle=':', label=f'Outside Tolerance ($|\\Delta t| > {threshold_min:g}$m)'),
    ]
    for idx, (label, _, _) in enumerate(dataset_data_cached):
        color = DATASET_COLORS[idx % len(DATASET_COLORS)]
        legend_elements.append(Patch(facecolor=color, alpha=0.7, edgecolor=color, label=label))

    # Anchored right outside the main plot's bounding box
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5),
              frameon=False, fontsize=8, labelcolor=TEXT_MUTED)

    # Render metrics table into its wider container axis
    render_metrics_table(ax_table, table_rows)

    # ── 4. Centered Title Presentation ───────────────────────────────────────
    # Setting x=0.5 and ha='center' centers the title over the entire figure width
    fig.suptitle(figure_title, fontsize=12, fontweight='bold', color=TEXT_MAIN, y=0.94, x=0.5, ha='center')
    
    # Save Image Target File
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{rwy}" if rwy else ""
    filename = out_dir / f"multi_dataset_delay_analysis{suffix}.png"
    
    # Note: bbox_inches="tight" is avoided here since we manual-positioned everything perfectly
    plt.savefig(filename, dpi=300, facecolor=BG_WHITE)
    print(f"[✓] Saved streamlined single-plot table layout to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-grade comparative delay distributions.")
    parser.add_argument("-d", "--datasets", nargs='+', required=True, help="Space-separated paths to dataset files.")
    parser.add_argument("-l", "--labels", nargs='+', default=None, help="Space-separated custom names.")
    parser.add_argument("-t", "--title", type=str, default=None, help="Master title text string.")
    parser.add_argument("-r", "--rwy", type=str, default=None, help="Optional runway filter.")
    parser.add_argument("-o", "--out-dir", type=str, default=".", help="Output directory path.")
    parser.add_argument("--hide-zones", action="store_true", help="Disable zone annotations.")
    
    args = parser.parse_args()
    input_dict = {}
    dataset_paths = [Path(p) for p in args.datasets]
    
    if args.labels:
        if len(args.labels) != len(dataset_paths):
            parser.error("The number of items in --labels must exactly match the number of --datasets.")
        input_dict = dict(zip(args.labels, dataset_paths))
    else:
        for path in dataset_paths:
            label = path.stem.replace('_', ' ').title()
            input_dict[label] = path
            
    generate_multi_plot(
        data_inputs=input_dict, 
        out_dir=Path(args.out_dir), 
        rwy=args.rwy,
        figure_title=args.title,
        show_zone_labels=not args.hide_zones
    )