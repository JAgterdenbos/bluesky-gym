import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm, skew, kurtosis

matplotlib.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
})

BG_WHITE = "#ffffff"
GRID_COLOR = "#ececec"
TEXT_MAIN = "#111111"
TEXT_MUTED = "#444444"
TOLERANCE_FILL = "#f4f9f4"
VIOLATION_FILL = "#fdf2f2"
FLAG_COLOR = "#a83232"
DATASET_COLORS = ["#004488", "#bb5566", "#44aa99", "#d95f02", "#999933"]

DELAY_THRESHOLD = 60.0   # seconds = 1 minute tolerance
MAX_TIME = 3600 * 6      # normalised window: 6 hours

TAIL_PERCENTILE = 99.0   # x-axis sized to capture this much mass of the WIDEST dataset
TAIL_HEADROOM = 1.15
Y_AXIS_FLOOR = 1e-3       # log-scale floor for density axis

SKEW_FLAG = 1.0           # |skew| beyond this is flagged as non-normal
KURTOSIS_FLAG = 3.0       # |excess kurtosis| beyond this is flagged

SYMLOG_LINSCALE = 1.5     # higher = more screen width given to the linear (within-tolerance) zone


def clean_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(BG_WHITE)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, direction='out')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor("#bbbbbb")
    ax.spines['bottom'].set_edgecolor("#bbbbbb")
    ax.xaxis.label.set_color(TEXT_MAIN)
    ax.yaxis.label.set_color(TEXT_MAIN)
    ax.grid(True, linestyle="--", linewidth=0.5, color=GRID_COLOR, zorder=0)


def ceil_to_nearest_half(val: float) -> float:
    return float(np.ceil(val * 2) / 2)


def process_dataset(data_path: Path, rwy: Optional[str] = None) -> np.ndarray:
    df = pd.read_parquet(data_path) if data_path.suffix == '.parquet' else pd.read_csv(data_path)

    if rwy and 'runway' in df.columns:
        df = df[df['runway'] == rwy]

    episode_summary = df.groupby('episode').last().reset_index() if 'episode' in df.columns else df

    if 'delay' not in episode_summary.columns:
        raise KeyError(f"Could not find 'delay' column in {data_path}")

    return episode_summary['delay'].dropna().to_numpy(np.float32) * MAX_TIME


def render_metrics_table(ax_table: plt.Axes, table_rows: List[List[str]], flag_cells: set) -> None:
    ax_table.axis('off')
    col_headers = [
        "Dataset Source", "N", "On-Time Rate", "Delayed Rate",
        "MAE", "Max Abs Delay", "Clipped (%)", "KDE BW ($h$)", "Skewness ($S$)", "Ex. Kurtosis ($K$)"
    ]
    # Size columns by their longest entry so the dataset-name column doesn't
    # get squeezed to the same width as a 4-character number column.
    raw_widths = [max(len(str(col_headers[c])), max(len(str(row[c])) for row in table_rows))
                  for c in range(len(col_headers))]
    total_width = sum(raw_widths)
    col_widths = [w / total_width for w in raw_widths]

    table = ax_table.table(cellText=table_rows, colLabels=col_headers, cellLoc='center',
                            loc='center', bbox=[0.0, 0.0, 1.0, 1.0], colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    num_rows = max(i[0] for i in table.get_celld().keys()) + 1
    num_cols = max(i[1] for i in table.get_celld().keys()) + 1

    for (r, c), cell in table.get_celld().items():
        cell.set_facecolor(BG_WHITE)
        cell.PAD = 0.03
        if c == 0:
            cell.PAD = 0.02
            cell.get_text().set_ha('left')
        edges = ''
        if r == 0:
            cell.set_text_props(fontweight='bold', color=TEXT_MAIN, fontfamily='serif')
            edges += 'T'
        if r == 1:
            edges += 'T'
        if r == num_rows - 1:
            edges += 'B'
        if c == 0:
            edges += 'L'
        if c == num_cols - 1:
            edges += 'R'
        cell.visible_edges = edges
        cell.set_linewidth(1.0)
        cell.set_edgecolor(TEXT_MAIN)

        if (r, c) in flag_cells:
            cell.set_text_props(color=FLAG_COLOR, fontweight='bold', fontfamily='serif')
        elif r > 0:
            cell.set_text_props(color=TEXT_MAIN, fontfamily='serif')

    ax_table.text(0.0, -0.1, f"Bold red: |S| > {SKEW_FLAG:g} or |K| > {KURTOSIS_FLAG:g} (deviation from normality).",
                  transform=ax_table.transAxes, fontsize=7, color=TEXT_MUTED, ha='left', style='italic')


def generate_multi_plot(
    data_inputs: Dict[str, Path],
    out_dir: Path = Path("."),
    rwy: Optional[str] = None,
    figure_title: Optional[str] = None,
    show_zone_labels: bool = True,
) -> None:
    if not figure_title:
        figure_title = "Arrival Delay Distribution Analysis with Normality Diagnostics"

    fig = plt.figure(figsize=(12.0, 7.4), facecolor=BG_WHITE)
    ax = fig.add_axes([0.07, 0.42, 0.66, 0.46])
    ax_table = fig.add_axes([0.05, 0.04, 0.90, 0.27])
    clean_axis(ax)

    threshold_min = DELAY_THRESHOLD / 60.0
    dataset_cache = []
    tail_targets = []

    for label, path in data_inputs.items():
        delays_sec = process_dataset(path, rwy=rwy)
        delays_min = delays_sec / 60.0
        dataset_cache.append((label, delays_sec, delays_min))
        # Size the axis to the WIDEST dataset's tail, not the median dataset.
        # Using median here would let a single high-variance dataset get
        # silently clipped while narrower datasets fit comfortably.
        tail_targets.append(np.percentile(np.abs(delays_min), TAIL_PERCENTILE) * TAIL_HEADROOM)

    x_limit = ceil_to_nearest_half(max(max(tail_targets), threshold_min * 5))

    # Sample density on a grid matching the symlog axis: linear (dense) inside
    # the tolerance band, log-spaced in the tails. A plain linspace over the
    # full range would barely sample the now visually-expanded ±1min zone.
    inner_x = np.linspace(-threshold_min, threshold_min, 300)
    outer_pos = np.geomspace(threshold_min, x_limit, 200)[1:]
    x_vals = np.concatenate([-outer_pos[::-1], inner_x, outer_pos])

    ax.axvspan(-threshold_min, threshold_min, facecolor=TOLERANCE_FILL, alpha=0.6, zorder=1)
    ax.axvspan(-x_limit, -threshold_min, facecolor=VIOLATION_FILL, alpha=0.5, zorder=1)
    ax.axvspan(threshold_min, x_limit, facecolor=VIOLATION_FILL, alpha=0.5, zorder=1)
    ax.axvline(threshold_min, color="#ccaaaa", linestyle=":", linewidth=1.0, zorder=2)
    ax.axvline(-threshold_min, color="#ccaaaa", linestyle=":", linewidth=1.0, zorder=2)

    if show_zone_labels:
        # On a symlog axis the visual midpoint of the outer zone is the
        # geometric mean of its bounds, not the arithmetic mean.
        outer_label_x = float(np.sqrt(x_limit * threshold_min))
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0, 0.95, "ON-TIME", transform=trans, color="#2e7d32",
                fontsize=8, fontweight='bold', ha='center', va='top', alpha=0.6)
        ax.text(-outer_label_x, 0.95, "EARLY", transform=trans, color="#c62828",
                fontsize=8, fontweight='bold', ha='center', va='top', alpha=0.6)
        ax.text(outer_label_x, 0.95, "LATE", transform=trans, color="#c62828",
                fontsize=8, fontweight='bold', ha='center', va='top', alpha=0.6)

    global_max_y = Y_AXIS_FLOOR * 10
    table_rows = []
    flag_cells = set()

    for idx, (label, delays_sec, delays_min) in enumerate(dataset_cache):
        abs_delays_sec = np.abs(delays_sec)
        abs_min = np.abs(delays_min)
        is_delayed = abs_delays_sec > DELAY_THRESHOLD
        delay_rate = np.mean(is_delayed) * 100
        on_time_rate = 100.0 - delay_rate
        mae_min = np.mean(abs_delays_sec) / 60.0
        max_abs_delay_min = float(np.max(abs_min))
        clipped_pct = float(np.mean(abs_min > x_limit) * 100)

        sk = float(skew(delays_min)) if len(delays_min) > 1 else 0.0
        kt = float(kurtosis(delays_min)) if len(delays_min) > 1 else 0.0

        color = DATASET_COLORS[idx % len(DATASET_COLORS)]
        bw_str = "N/A"

        if len(delays_min) > 1:
            kde = gaussian_kde(delays_min)
            bw = kde.factor * np.std(delays_min, ddof=1)
            bw_str = f"{bw:.2f}"

            y_vals = np.clip(kde(x_vals), Y_AXIS_FLOOR, None)

            ax.plot(x_vals, y_vals, color=color, linewidth=1.5, label=label, zorder=4)
            ax.fill_between(x_vals, Y_AXIS_FLOOR, y_vals, color=color, alpha=0.06, zorder=3)

            mu, sigma = np.mean(delays_min), np.std(delays_min)
            y_normal = np.clip(norm.pdf(x_vals, mu, sigma), Y_AXIS_FLOOR, None)
            ax.plot(x_vals, y_normal, color=color, linestyle=(0, (3, 4)), linewidth=0.8, alpha=0.5, zorder=3)

            global_max_y = max(global_max_y, float(np.max(y_vals)), float(np.max(y_normal)))
        else:
            counts, _, _ = ax.hist(delays_min, bins=30, range=(-x_limit, x_limit), density=True,
                                    color=color, alpha=0.4, zorder=4, label=label)
            if len(counts) > 0:
                global_max_y = max(global_max_y, float(np.max(counts)))

        row_idx = idx + 1
        if abs(sk) > SKEW_FLAG:
            flag_cells.add((row_idx, 8))
        if abs(kt) > KURTOSIS_FLAG:
            flag_cells.add((row_idx, 9))

        table_rows.append([
            label,
            f"{len(delays_sec):,}",
            f"{on_time_rate:.1f}%",
            f"{delay_rate:.1f}%",
            f"{mae_min:.2f} min",
            f"{max_abs_delay_min:.1f} min",
            f"{clipped_pct:.3f}%",
            bw_str,
            f"{sk:.2f}",
            f"{kt:.2f}",
        ])

    ax.set_yscale('log')
    ax.set_ylim(Y_AXIS_FLOOR, global_max_y * 2.2)

    ax.set_xscale('symlog', linthresh=threshold_min, linscale=SYMLOG_LINSCALE)
    ax.set_xlim(-x_limit, x_limit)

    outer_tick_candidates = [2, 5, 10, 20, 30, 40, 50, 60, 90, 120]
    outer_ticks = [v for v in outer_tick_candidates if threshold_min < v < x_limit]
    ticks = sorted(set(
        [-threshold_min, -threshold_min / 2, 0, threshold_min / 2, threshold_min]
        + [-v for v in outer_ticks] + outer_ticks
    ))
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))

    ax.set_ylabel("Density (log scale)", fontsize=9, color=TEXT_MUTED)
    ax.set_xlabel(r"Arrival Delay $\Delta t$ [minutes] (symlog scale)", fontsize=10)

    legend_elements = [
        Patch(facecolor=TOLERANCE_FILL, edgecolor='#cccccc', linestyle=':', label=f'Within Tolerance ($|\\Delta t| \\leq {threshold_min:g}$m)'),
        Patch(facecolor=VIOLATION_FILL, edgecolor='#cccccc', linestyle=':', label=f'Outside Tolerance ($|\\Delta t| > {threshold_min:g}$m)'),
        plt.Line2D([0], [0], color=TEXT_MUTED, linestyle='-', linewidth=1.5, label='Empirical KDE Curve'),
        plt.Line2D([0], [0], color=TEXT_MUTED, linestyle=(0, (3, 4)), linewidth=1.0, label='Theoretical Normal Fit'),
    ]
    for idx, (label, _, _) in enumerate(dataset_cache):
        color = DATASET_COLORS[idx % len(DATASET_COLORS)]
        legend_elements.append(Patch(facecolor=color, alpha=0.7, edgecolor=color, label=label))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5),
              frameon=False, fontsize=8, labelcolor=TEXT_MUTED)

    render_metrics_table(ax_table, table_rows, flag_cells)

    fig.suptitle(figure_title, fontsize=12, fontweight='bold', color=TEXT_MAIN, y=0.96, x=0.5, ha='center')

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{rwy}" if rwy else ""
    filename = out_dir / f"multi_dataset_delay_analysis{suffix}.png"
    plt.savefig(filename, dpi=300, facecolor=BG_WHITE)
    print(f"[OK] Saved to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-grade comparative delay distributions.")
    parser.add_argument("-d", "--datasets", nargs='+', required=True, help="Space-separated paths to dataset files.")
    parser.add_argument("-l", "--labels", nargs='+', default=None, help="Space-separated custom names.")
    parser.add_argument("-t", "--title", type=str, default=None, help="Master title text string.")
    parser.add_argument("-r", "--rwy", type=str, default=None, help="Optional runway filter.")
    parser.add_argument("-o", "--out-dir", type=str, default=".", help="Output directory path.")
    parser.add_argument("--hide-zones", action="store_true", help="Disable zone annotations.")

    args = parser.parse_args()
    dataset_paths = [Path(p) for p in args.datasets]

    if args.labels:
        if len(args.labels) != len(dataset_paths):
            parser.error("The number of items in --labels must exactly match the number of --datasets.")
        input_dict = dict(zip(args.labels, dataset_paths))
    else:
        input_dict = {p.stem.replace('_', ' ').title(): p for p in dataset_paths}

    generate_multi_plot(
        data_inputs=input_dict,
        out_dir=Path(args.out_dir),
        rwy=args.rwy,
        figure_title=args.title,
        show_zone_labels=not args.hide_zones,
    )