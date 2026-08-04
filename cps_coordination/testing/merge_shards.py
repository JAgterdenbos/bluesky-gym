"""
cps_coordination/testing/merge_shards.py
------------------------------------------
Merges one combo's sharded Parquet output (see run_step10_scale10k.sh's
SHARDS/SHARD_INDEX support) into the single combo directory the rest of
the tooling (cps_metrics_offline.py, step10_deep_analysis.py) expects.

Each shard writes to its own `{save_root}/shard_{i}of{N}/{combo}/` directory
rather than a shared one -- Parquet has no true row-group append
(ParquetWriter always opens in write/truncating mode; see
run_batch_eval.py::_merge_resume_delta's docstring for the same finding in
the --resume context), so N concurrently-running shard processes writing to
the same file would corrupt it. This script does the merge as a separate,
offline step once all shards have finished, verifying there are no
colliding episode_ids across shards before writing anything (sharding's own
--episode-id-offset math guarantees disjoint ranges, but this checks the
actual data rather than assuming the math held).

Usage
-----
  uv run python cps_coordination/testing/merge_shards.py \\
      --save-root experiments/cps_eval/scale_10k_20260801_000000 \\
      --combo k3_dynamic_fw0.3 \\
      --shards 4
"""
from __future__ import annotations

import argparse
import os
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

_FILES = ("cps_eval_aircraft.parquet", "cps_eval_separation.parquet")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-root", type=str, required=True,
                   help="The SAVE_ROOT passed to run_step10_scale10k.sh (without any "
                        "shard_*of* suffix -- that's added per shard automatically).")
    p.add_argument("--combo", type=str, required=True,
                   help="Combo directory name, e.g. k3_dynamic_fw0.3.")
    p.add_argument("--shards", type=int, required=True, help="Total number of shards.")
    p.add_argument("--force", action="store_true", default=False,
                   help="Overwrite the destination combo directory's Parquet files if "
                        "they already exist (default: refuse, to avoid silently "
                        "clobbering an earlier merge or a non-sharded run's data).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    shard_dirs = [
        os.path.join(args.save_root, f"shard_{i}of{args.shards}", args.combo)
        for i in range(args.shards)
    ]
    missing = [d for d in shard_dirs if not os.path.exists(os.path.join(d, "cps_eval_aircraft.parquet"))]
    if missing:
        print(f"ERROR: {len(missing)}/{args.shards} shard(s) missing cps_eval_aircraft.parquet -- "
              f"not all shards have finished yet:")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    dest_dir = os.path.join(args.save_root, args.combo)
    for filename in _FILES:
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path) and not args.force:
            print(f"ERROR: {dest_path} already exists -- pass --force to overwrite "
                  f"(refusing by default to avoid silently clobbering existing data).")
            sys.exit(1)

    # Safety check: shard episode_id ranges must be disjoint (sharding's own
    # --episode-id-offset math guarantees this if used correctly, but a
    # config mistake -- e.g. two shards launched with the same SHARD_INDEX --
    # would silently corrupt c_sep/other metrics that group by episode_id,
    # so verify the actual data rather than trust the math held).
    aircraft_tables = []
    episode_id_ranges = []
    for i, shard_dir in enumerate(shard_dirs):
        table = pq.read_table(os.path.join(shard_dir, "cps_eval_aircraft.parquet"))
        ids = pc.unique(table.column("episode_id"))
        lo, hi = int(pc.min(ids).as_py()), int(pc.max(ids).as_py())
        episode_id_ranges.append((i, set(ids.to_pylist())))
        aircraft_tables.append(table)
        print(f"shard {i}/{args.shards}: {table.num_rows} rows, episode_id range [{lo}, {hi}]")

    collisions = []
    for a in range(len(episode_id_ranges)):
        ia, ids_a = episode_id_ranges[a]
        for b in range(a + 1, len(episode_id_ranges)):
            ib, ids_b = episode_id_ranges[b]
            overlap = ids_a & ids_b
            if overlap:
                collisions.append((ia, ib, sorted(overlap)[:10]))

    if collisions:
        print(f"\nERROR: {len(collisions)} shard pair(s) have colliding episode_ids -- "
              f"refusing to merge (this would silently corrupt c_sep and other metrics "
              f"that group by episode_id):")
        for ia, ib, sample in collisions:
            print(f"  shard {ia} <-> shard {ib}: e.g. episode_ids {sample}")
        sys.exit(1)

    print(f"\nNo episode_id collisions across {args.shards} shards -- merging.")

    os.makedirs(dest_dir, exist_ok=True)
    merged_aircraft = pa.concat_tables(aircraft_tables)
    pq.write_table(merged_aircraft, os.path.join(dest_dir, "cps_eval_aircraft.parquet"))
    print(f"Wrote {merged_aircraft.num_rows} rows -> {dest_dir}/cps_eval_aircraft.parquet")

    separation_tables = [
        pq.read_table(os.path.join(shard_dir, "cps_eval_separation.parquet"))
        for shard_dir in shard_dirs
    ]
    merged_separation = pa.concat_tables(separation_tables)
    pq.write_table(merged_separation, os.path.join(dest_dir, "cps_eval_separation.parquet"))
    print(f"Wrote {merged_separation.num_rows} rows -> {dest_dir}/cps_eval_separation.parquet")

    print(f"\nDone. Merged combo -> {dest_dir}/")


if __name__ == "__main__":
    main()
