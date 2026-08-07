#!/usr/bin/env bash
# cps_coordination/scripts/step10_progress.sh
# --------------------------------------------------------------
# Progress snapshot for a Step 10 batch-eval sweep launched via
# launch_step10_dedicated_terminal.sh / run_step10_scale10k.sh.
#
# Two sources, merged:
#   - The launch log's "[N/M] episodes logged" lines (printed every
#     --log-every episodes, default 100) -- the ONLY live signal available
#     for a combo that's still running. run_batch_eval.py's
#     ParquetDataCollector (path_planning/rta/collect.py) keeps a single
#     ParquetWriter open across chunk flushes and only calls .close() (which
#     writes the Parquet footer) when a combo finishes or the run is
#     cleanly interrupted -- so the Parquet file is genuinely unreadable
#     for that combo's entire in-progress duration, not just briefly.
#   - Each combo's cps_eval_aircraft.parquet once it IS readable (combo
#     finished, or a clean stop via step10_stop.sh closed the writer) --
#     exact episode/success counts, since it's the same durable source
#     --resume itself reads.
#
# Usage:
#   ./cps_coordination/scripts/step10_progress.sh                    # auto-detect the newest SAVE_ROOT
#   ./cps_coordination/scripts/step10_progress.sh <save_root>         # explicit SAVE_ROOT
#   ./cps_coordination/scripts/step10_progress.sh --watch             # refresh every INTERVAL seconds (Ctrl-C to stop)
#   ./cps_coordination/scripts/step10_progress.sh --watch --notify    # also: fire a macOS notification (+ sound) the
#                                                                      # moment the run stops (finished OR crashed/
#                                                                      # interrupted), then exit -- so you don't have
#                                                                      # to babysit the terminal for an 8-20h run.
#
# Env vars:
#   EPISODES   Target M per combo, for the percent-complete column (default 2000).
#   INTERVAL   Refresh interval in seconds for --watch (default 30).

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, regardless of where this is invoked from

WATCH=0
NOTIFY=0
SAVE_ROOT=""
for arg in "$@"; do
    case "$arg" in
        --watch|-w) WATCH=1 ;;
        --notify|-n) NOTIFY=1 ;;
        *) SAVE_ROOT="$arg" ;;
    esac
done

if [ -z "$SAVE_ROOT" ]; then
    SAVE_ROOT="$(ls -dt experiments/cps_eval/scale_10k_* 2>/dev/null | head -n 1 || true)"
    if [ -z "$SAVE_ROOT" ]; then
        echo "No experiments/cps_eval/scale_10k_* directory found -- pass a SAVE_ROOT explicitly." >&2
        exit 1
    fi
fi

EPISODES="${EPISODES:-2000}"
INTERVAL="${INTERVAL:-30}"

notify() {
    # $1 = title, $2 = message. Best-effort: osascript is macOS-only and
    # this is explicitly a macOS Terminal.app workflow (see README).
    osascript -e "display notification \"$2\" with title \"$1\" sound name \"Glass\"" >/dev/null 2>&1 || true
}

format_duration() {
    # $1 = total seconds -> "XhYmZs" (omits leading zero units).
    local total="$1" h m s
    h=$((total / 3600))
    m=$(((total % 3600) / 60))
    s=$((total % 60))
    if [ "$h" -gt 0 ]; then
        printf "%dh%dm%ds" "$h" "$m" "$s"
    elif [ "$m" -gt 0 ]; then
        printf "%dm%ds" "$m" "$s"
    else
        printf "%ds" "$s"
    fi
}

# Sets globals RUNNING (0/1) and VERDICT (RUNNING/COMPLETE/STOPPED_EARLY/NONE).
show_once() {
    echo "=================================================================="
    echo "Step 10 progress -- $SAVE_ROOT"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="

    # --- is a run_batch_eval.py process for this SAVE_ROOT still alive? ---
    PIDS="$(pgrep -f "run_batch_eval.py.*$SAVE_ROOT" 2>/dev/null || true)"
    if [ -n "$PIDS" ]; then
        RUNNING=1
        echo "Process: RUNNING (pid: $(echo "$PIDS" | tr '\n' ' '))"
    else
        RUNNING=0
        echo "Process: NOT RUNNING (finished, not started yet, or launched with a different SAVE_ROOT)"
    fi

    # --- most recent launch log: staleness + elapsed since launch ---
    LOG="$(ls -t cps_coordination/data/step10_launch_*.log 2>/dev/null | head -n 1 || true)"
    if [ -n "$LOG" ]; then
        LOG_AGE=$(( $(date +%s) - $(date -r "$LOG" +%s) ))
        echo "Log:     $LOG (last write ${LOG_AGE}s ago)"
        # The launch script names this file step10_launch_<YYYYMMDD_HHMMSS>.log
        # at the moment it's created (right before the actual launch) -- parse
        # that back out as the run's start time rather than relying on
        # filesystem birth-time semantics, which aren't portable/reliable.
        LOG_TS="$(basename "$LOG" .log | sed -E 's/^step10_launch_//')"
        LAUNCH_EPOCH="$(date -j -f "%Y%m%d_%H%M%S" "$LOG_TS" +%s 2>/dev/null || true)"
        if [ -n "$LAUNCH_EPOCH" ]; then
            ELAPSED=$(( $(date +%s) - LAUNCH_EPOCH ))
            echo "Elapsed: $(format_duration "$ELAPSED") since launch ($LOG_TS)"
        fi
    else
        echo "Log:     none found under cps_coordination/data/"
    fi
    echo ""

    # --- per-combo progress: log-parsed (live) + Parquet (final) merged ---
    PYOUT="$(uv run python3 - "$SAVE_ROOT" "$EPISODES" "$LOG" <<'PY'
import glob, os, re, sys
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

save_root, episodes, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

# --- parse the log for the live (pre-close) progress signal ---
combo_start_re = re.compile(
    r"^--- combo k_cps=(\d+), mode=(\w+), fairness_weight=([\d.]+)( done)? -> (\S+) ---$"
)
progress_re = re.compile(r"^\s*\[(\d+)/(\d+)\] episodes logged$")

log_progress = {}   # combo_key -> latest episodes-attempted seen in the log
log_done = set()
current_combo = None

if log_path and os.path.exists(log_path):
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = combo_start_re.match(line.rstrip("\n"))
            if m:
                k_cps, mode, fw, done = m.group(1), m.group(2), float(m.group(3)), m.group(4)
                key = f"k{k_cps}_{mode}_fw{fw:g}"
                if done:
                    log_done.add(key)
                    current_combo = None
                else:
                    current_combo = key
                continue
            m = progress_re.match(line.rstrip("\n"))
            if m and current_combo is not None:
                log_progress[current_combo] = int(m.group(1))

# --- enumerate combos: union of on-disk dirs and anything seen in the log ---
combo_dirs = {os.path.basename(d): d for d in glob.glob(os.path.join(save_root, "k*_fw*"))}
combo_keys = set(combo_dirs) | set(log_progress) | log_done

if not combo_keys:
    print(f"No combos reached yet under {save_root}.")
    print("SUMMARY: seen=0 complete=0")
    sys.exit(0)


def sort_key(key):
    m = re.match(r"^k(\d+)_(static|dynamic)_fw", key)
    if not m:
        return (2, 0, key)
    return (0 if m.group(2) == "static" else 1, int(m.group(1)), key)


n_complete = 0
print(f"{'combo':<22} {'episodes':>14} {'%':>7}  {'success_rate':>12}  {'source':>26}")
for key in sorted(combo_keys, key=sort_key):
    d = combo_dirs.get(key)
    path = os.path.join(d, "cps_eval_aircraft.parquet") if d else None
    table = None
    if path and os.path.exists(path):
        try:
            table = pq.read_table(path, columns=["episode_id", "success"])
        except Exception:
            table = None  # writer still open for this combo -- not readable yet

    if table is not None:
        n_rows = table.num_rows
        if n_rows == 0:
            print(f"{key:<22} {'0/' + str(episodes):>14} {'0.0%':>7}  {'':>12}  {'parquet (final)':>26}")
            continue
        n_episodes = pc.count_distinct(table.column("episode_id")).as_py()
        success_rate = pc.mean(table.column("success").cast(pa.float64())).as_py()
        pct = 100.0 * n_episodes / episodes
        if n_episodes >= episodes:
            n_complete += 1
        print(f"{key:<22} {str(n_episodes) + '/' + str(episodes):>14} {pct:>6.1f}%  {success_rate:>11.1%}  {'parquet (final)':>26}")
    elif key in log_progress:
        n_attempted = log_progress[key]
        pct = 100.0 * n_attempted / episodes
        print(f"{key:<22} {str(n_attempted) + '/' + str(episodes):>14} {pct:>6.1f}%  {'n/a':>12}  {'log, attempted (live)':>26}")
    elif key in log_done:
        print(f"{key:<22} {'done (closing)':>14} {'':>7}  {'':>12}  {'log: done, writer closing':>26}")
    else:
        print(f"{key:<22} {'(started, no data)':>14}")

print(f"SUMMARY: seen={len(combo_keys)} complete={n_complete}")
PY
)"
    echo "$PYOUT" | grep -v '^SUMMARY:'
    echo ""

    SEEN="$(echo "$PYOUT" | grep '^SUMMARY:' | sed -E 's/.*seen=([0-9]+).*/\1/')"
    COMPLETE="$(echo "$PYOUT" | grep '^SUMMARY:' | sed -E 's/.*complete=([0-9]+).*/\1/')"
    if [ "$RUNNING" = "1" ]; then
        VERDICT="RUNNING"
    elif [ -n "$SEEN" ] && [ "$SEEN" -gt 0 ] && [ "$SEEN" = "$COMPLETE" ]; then
        VERDICT="COMPLETE"
    elif [ -n "$SEEN" ] && [ "$SEEN" -gt 0 ]; then
        VERDICT="STOPPED_EARLY ($COMPLETE/$SEEN combos fully complete)"
    else
        VERDICT="NONE (no combos reached yet)"
    fi
    echo "Verdict: $VERDICT"
}

if [ "$WATCH" = "1" ]; then
    WAS_RUNNING=""
    while true; do
        clear 2>/dev/null || true   # no-op (not an error) if TERM isn't set, e.g. a non-interactive test run
        show_once
        if [ "$WAS_RUNNING" = "1" ] && [ "$RUNNING" = "0" ]; then
            echo ""
            echo "Run stopped (verdict: $VERDICT)."
            if [ "$NOTIFY" = "1" ]; then
                notify "Step 10" "Sweep stopped -- $VERDICT"
            fi
            break
        fi
        WAS_RUNNING="$RUNNING"
        REMAINING="$INTERVAL"
        while [ "$REMAINING" -gt 0 ]; do
            printf "\rNext refresh in %3ds -- Ctrl-C to stop.  " "$REMAINING"
            sleep 1
            REMAINING=$((REMAINING - 1))
        done
        printf "\r"
    done
else
    show_once
fi
