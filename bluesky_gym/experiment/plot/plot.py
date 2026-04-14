"""
Plotting API and Command-Line Interface.

This module serves as the primary entry point for generating visual analytics 
across both training and evaluation phases. It parses user commands to locate 
log files (CSV/YAML) and routes the data to the appropriate plotting engines 
defined in `train_plots.py` and `eval_plots.py`.
"""

from __future__ import annotations

import argparse
import os

from .data import _find_training_csv, _find_all_training_csvs, _find_eval_files, _load_eval_csv, _load_eval_yaml, _load_merged_csv, _load_training_csv, _resolve_eval_files, _list_eval_files
from .train_plots import plot_training_curves, plot_comparison_grid
from .eval_plots import plot_eval_summary, plot_eval_episodes, plot_eval_dashboard


def plot(
    command:      str,
    runs:         list[str] | None = None,
    discover_all: bool = False,
    from_csv:     str | None = None,
    files:        list[str] | None = None,
    run_ids:      list[str] | None = None,
    metric:       str = "success_rate",
    labels:       list[str] | None = None,
    eval_indices: list[int] | None = None,
    list_evals:   bool = False,
    out:          str | None = None,
    title:        str | None = None,
    smooth:       int = 1,
    grid:         bool = False,
) -> None:
    """
    Programmatic entry point for plotting.

    Parameters
    ----------
    from_csv : str, optional
        Path to a merged comparison CSV (output of compare_runs).  When
        provided for the 'training' command, run IDs and data are loaded
        directly from it — no individual training_evals.csv files needed.
    grid : bool
        When True and command='training', render the richer 2x2 comparison
        grid instead of the simple side-by-side panels.
    run_ids : list[str], optional
        One or more run IDs for eval subcommands. Each run resolves to one
        eval file (selected by eval_indices).
    eval_indices : list[int], optional
        Per-run file indices. eval_indices[i] selects which file to use for
        run_ids[i] when multiple eval files exist. Defaults to 0 per run.
    list_evals : bool
        When True, print the discovered eval files for each run and exit
        without plotting. Useful for finding the right --eval-indices values.
    """
    if command == "training":
        # ── Resolve data source ──────────────────────────────────────────────
        if from_csv:
            resolved_run_ids, all_rows = _load_merged_csv(from_csv)
            print(f"📂 Loaded {len(resolved_run_ids)} run(s) from {from_csv}")
        elif discover_all:
            discovered = _find_all_training_csvs()
            resolved_run_ids = [r for r, _ in discovered]
            all_rows = [_load_training_csv(p) for _, p in discovered]
        else:
            if not runs:
                print("❌ Error: Provide --runs, --all, or --from-csv for the training command.")
                return
            resolved_run_ids = runs
            all_rows = [_load_training_csv(_find_training_csv(r)) for r in resolved_run_ids]

        # Map labels
        plot_labels = labels if labels and len(labels) == len(resolved_run_ids) else resolved_run_ids
        if labels and len(labels) != len(resolved_run_ids):
            print(f"⚠️  Warning: Provided {len(labels)} labels for {len(resolved_run_ids)} runs. Defaulting to Run IDs.")

        if grid or len(resolved_run_ids) > 1:
            plot_comparison_grid(plot_labels, all_rows, out, smooth, title)
        else:
            plot_training_curves(plot_labels, all_rows, out, smooth, title)

    elif command == "eval-dashboard":
        # ── --list-evals dry run ─────────────────────────────────────────────
        if list_evals:
            if not run_ids:
                print("❌ Error: --list-evals requires --run-ids.")
                return
            print("📋 Available eval CSV files:")
            _list_eval_files(run_ids, "csv")
            return

        # Single-run dashboard: one CSV (and optionally one YAML) per run
        csv_files = _resolve_eval_files(run_ids, files, eval_indices, "csv", command)
        if not csv_files:
            print("❌ Error: Provide --files <CSV> or --run-ids for eval-dashboard.")
            return
        dash_labels = labels or [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
        for i, (label, csv_path) in enumerate(zip(dash_labels, csv_files)):
            rows   = _load_eval_csv(csv_path)
            # For YAML, attempt to find one alongside the CSV's run if run_ids given
            yaml_d = None
            if run_ids and i < len(run_ids):
                yaml_files = _find_eval_files(run_ids[i], "yaml")
                if yaml_files:
                    yaml_idx = (eval_indices[i] if eval_indices and i < len(eval_indices) else 0)
                    yaml_idx = yaml_idx if yaml_idx < len(yaml_files) else 0
                    yaml_d   = _load_eval_yaml(yaml_files[yaml_idx])
            plot_eval_dashboard(label, rows, yaml_d, out, title)

    elif command in ["eval-summary", "eval-episodes"]:
        ext = "yaml" if command == "eval-summary" else "csv"

        # ── --list-evals dry run ─────────────────────────────────────────────
        if list_evals:
            if not run_ids:
                print("❌ Error: --list-evals requires --run-ids.")
                return
            print(f"📋 Available eval {ext.upper()} files:")
            _list_eval_files(run_ids, ext)
            return

        eval_files = _resolve_eval_files(run_ids, files, eval_indices, ext, command)
        if not eval_files:
            print(f"❌ Error: Provide --files or --run-ids with existing {ext} files.")
            return

        plot_labels = labels or [os.path.basename(f) for f in eval_files]
        if labels and len(labels) != len(eval_files):
            print(f"⚠️  Warning: Provided {len(labels)} labels for {len(eval_files)} files. Defaulting to filenames.")
            plot_labels = [os.path.basename(f) for f in eval_files]

        if command == "eval-summary":
            yaml_data = [_load_eval_yaml(f) for f in eval_files]
            plot_eval_summary(plot_labels, yaml_data, metric, out, title)
        else:
            csv_data = [_load_eval_csv(f) for f in eval_files]
            plot_eval_episodes(plot_labels, csv_data, out, title)

    else:
        print(f"❌ Unknown command: {command}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate plots for training and evaluation results. "
            "Choose a subcommand to plot training reward curves, "
            "evaluation summaries, per-episode breakdowns, or a "
            "full single-run dashboard."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Training\n"
            "  %(prog)s training --runs run_001 run_002 --smooth 5 --grid\n"
            "  %(prog)s training --from-csv merged.csv --labels Baseline Tuned\n"
            "\n"
            "  # List available eval files before plotting\n"
            "  %(prog)s eval-summary --run-ids run_001 run_002 --list-evals\n"
            "\n"
            "  # Use specific eval files by index\n"
            "  %(prog)s eval-summary --run-ids run_001 run_002 --eval-indices 0 2\n"
            "  %(prog)s eval-episodes --run-ids run_001 run_002 --eval-indices 1 0\n"
            "\n"
            "  # Explicit file paths\n"
            "  %(prog)s eval-episodes --files eval_run1.csv eval_run2.csv\n"
            "  %(prog)s eval-dashboard --run-ids run_001 --out ./plots\n"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── training subcommand ──────────────────────────────────────────────────
    tr = sub.add_parser(
        "training",
        help="Plot training reward and success-rate curves.",
        description=(
            "Plot mean reward (with ±1 std bands) and success-rate curves "
            "over environment steps. Supports single-run and multi-run "
            "comparisons. With --grid, renders a richer 2×2 panel that also "
            "shows reward std dev over time and a per-step reward gap (for "
            "two-run comparisons) or a min–max range band (for three or more)."
        ),
    )
    src = tr.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--runs",
        nargs="+",
        metavar="RUN_ID",
        help=(
            "One or more run IDs to plot. Each ID is resolved to its "
            "training_evals.csv via the data module's lookup logic."
        ),
    )
    src.add_argument(
        "--all",
        action="store_true",
        help=(
            "Auto-discover and plot all runs found by the data module. "
            "Mutually exclusive with --runs and --from-csv."
        ),
    )
    src.add_argument(
        "--from-csv",
        metavar="PATH",
        help=(
            "Path to a merged comparison CSV produced by compare_runs. "
            "Run IDs and data are loaded directly from this file — no "
            "individual training_evals.csv files are needed. "
            "Mutually exclusive with --runs and --all."
        ),
    )
    tr.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help=(
            "Custom legend labels, one per run, in the same order as the "
            "runs are resolved. If omitted, run IDs are used. The count "
            "must match the number of runs or labels will be ignored."
        ),
    )
    tr.add_argument(
        "--smooth",
        type=int,
        default=1,
        help=(
            "Rolling-average window size applied to reward and success-rate "
            "curves before plotting. 1 means no smoothing (default: 1)."
        ),
    )
    tr.add_argument(
        "--grid",
        action="store_true",
        help=(
            "Render the richer 2×2 comparison grid (mean reward, success "
            "rate, reward std dev, and reward gap / range) instead of the "
            "default side-by-side panels. Automatically enabled when more "
            "than one run is provided."
        ),
    )
    tr.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Directory where the output PNG will be saved. "
            "If omitted, the plot is displayed interactively."
        ),
    )
    tr.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override the default figure title.",
    )

    # ── shared factory for eval subcommands ─────────────────────────────────
    def _add_eval_args(sub_parser, *, file_ext: str, file_metavar: str, files_help: str) -> None:
        """
        Attach the standard eval arguments to an eval subparser.

        Enforces mutual exclusivity between --files and --run-ids at the
        argparse level so users get a clear error instead of silent
        misbehaviour.

        Parameters
        ----------
        sub_parser    : the subparser to attach arguments to.
        file_ext      : extension shown in --list-evals output ('csv'/'yaml').
        file_metavar  : displayed metavar for --files (e.g. 'CSV_PATH').
        files_help    : help text for --files, specific to this subcommand.
        """
        src = sub_parser.add_mutually_exclusive_group(required=True)
        src.add_argument(
            "--files",
            nargs="+",
            metavar=file_metavar,
            help=files_help,
        )
        src.add_argument(
            "--run-ids",
            nargs="+",
            metavar="RUN_ID",
            help=(
                "One or more run IDs to plot. For each run the data module "
                "discovers all available eval files; use --eval-indices to "
                "select which file to use per run (default: 0, the first). "
                "Run --list-evals first to see what is available. "
                "Mutually exclusive with --files."
            ),
        )
        sub_parser.add_argument(
            "--eval-indices",
            nargs="+",
            type=int,
            metavar="N",
            default=None,
            help=(
                f"Per-run index selecting which {file_ext} file to use when a "
                "run has multiple eval files. Supply one integer per run in "
                "the same order as --run-ids (e.g. --eval-indices 0 2 1). "
                "Runs without a matching index default to 0. "
                "Use --list-evals to see available files and their indices. "
                "Ignored when --files is used."
            ),
        )
        sub_parser.add_argument(
            "--list-evals",
            action="store_true",
            default=False,
            help=(
                f"List all available {file_ext} eval files (with their indices) "
                "for each run in --run-ids, then exit without plotting. "
                "Use this to find the right values for --eval-indices."
            ),
        )
        sub_parser.add_argument(
            "--labels",
            nargs="+",
            default=None,
            help=(
                "Custom legend labels, one per resolved file, in the same "
                "order as --files or --run-ids. Defaults to bare filenames."
            ),
        )
        sub_parser.add_argument(
            "--out",
            type=str,
            default=None,
            help=(
                "Directory where the output PNG will be saved. "
                "If omitted, the plot is displayed interactively."
            ),
        )
        sub_parser.add_argument(
            "--title",
            type=str,
            default=None,
            help="Override the default figure title.",
        )

    # ── eval-summary subcommand ──────────────────────────────────────────────
    es = sub.add_parser(
        "eval-summary",
        help="Plot a cross-run evaluation summary for a single metric.",
        description=(
            "Reads per-group metric values from one or more evaluation YAML "
            "files and renders a grouped bar chart, with overall-mean dashed "
            "lines per run. Use --metric to select which aggregated field to "
            "visualise (default: success_rate). Pass --run-ids with multiple "
            "IDs to compare runs side by side; use --list-evals to inspect "
            "which YAML files are available per run before committing to "
            "--eval-indices."
        ),
    )
    _add_eval_args(
        es,
        file_ext="yaml",
        file_metavar="YAML_PATH",
        files_help=(
            "Explicit paths to evaluation YAML files (one per run). "
            "Mutually exclusive with --run-ids."
        ),
    )
    es.add_argument(
        "--metric",
        type=str,
        default="success_rate",
        help=(
            "Name of the metric to plot from the YAML 'per_group' and "
            "'overall' sections (e.g. 'success_rate', 'mean_total_reward'). "
            "Default: success_rate."
        ),
    )

    # ── eval-episodes subcommand ─────────────────────────────────────────────
    ep = sub.add_parser(
        "eval-episodes",
        help="Plot a cross-run episode-level evaluation comparison.",
        description=(
            "Reads per-episode CSV data from one or more evaluation runs and "
            "renders a 2×2 comparison grid: reward boxplots by group, success "
            "rate bars, episode-timeline rolling means, and overlapping reward "
            "histograms. Pass multiple --run-ids to compare runs side by side; "
            "use --list-evals to inspect available CSVs per run, then "
            "--eval-indices to select the right one for each run."
        ),
    )
    _add_eval_args(
        ep,
        file_ext="csv",
        file_metavar="CSV_PATH",
        files_help=(
            "Explicit paths to evaluation episode CSV files (one per run). "
            "Mutually exclusive with --run-ids."
        ),
    )

    # ── eval-dashboard subcommand ────────────────────────────────────────────
    ed = sub.add_parser(
        "eval-dashboard",
        help="Generate a single-run evaluation dashboard.",
        description=(
            "Produces a comprehensive dashboard for one evaluation run at a "
            "time. Always includes a reward violin + jitter plot by group, a "
            "success-rate horizontal bar chart, and an episode-timeline "
            "scatter with rolling mean. Additional panels are generated "
            "dynamically for each extra numeric metric found in the CSV "
            "(cross-referenced against the YAML when available); if no extras "
            "exist, a fallback reward histogram is shown instead. "
            "Use --list-evals to inspect available CSVs for a run, then "
            "--eval-indices to select the correct one."
        ),
    )
    _add_eval_args(
        ed,
        file_ext="csv",
        file_metavar="CSV_PATH",
        files_help=(
            "One or more evaluation episode CSV files. A separate dashboard "
            "is produced for each file. Mutually exclusive with --run-ids."
        ),
    )

    return p


def run_plot_cli(experiment_cls=None) -> None:
    """Standalone CLI entry point."""
    args = _build_parser().parse_args()

    if args.command == "training":
        plot(
            command="training",
            runs=getattr(args, "runs", None),
            discover_all=getattr(args, "all", False),
            from_csv=getattr(args, "from_csv", None),
            smooth=args.smooth,
            grid=getattr(args, "grid", False),
            labels=getattr(args, "labels", None),
            out=args.out,
            title=args.title,
        )
    elif args.command in ["eval-summary", "eval-episodes", "eval-dashboard"]:
        plot(
            command=args.command,
            files=getattr(args, "files", None),
            run_ids=getattr(args, "run_ids", None),
            eval_indices=getattr(args, "eval_indices", None),
            list_evals=getattr(args, "list_evals", False),
            metric=getattr(args, "metric", "success_rate"),
            labels=getattr(args, "labels", None),
            out=args.out,
            title=args.title,
        )