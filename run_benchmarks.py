import subprocess
import sys


# Define the experiments as a list of dictionaries for clean organization
experiments = [
    {
        "data": "path_planning/rta/data/spatial/no_HER/rta_data_deterministic.parquet",
        "alg": "No_HER",
    },
    {
        "data": "path_planning/rta/data/spatial/HER/rta_data_deterministic.parquet",
        "alg": "HER",
    },
    {
        "data": "path_planning/rta/data/spatial/no_HER_hdg/rta_data_deterministic.parquet",
        "alg": "No_HER_hdg",
    },
    {
        "data": "path_planning/rta/data/spatial/HER/hdg_rta_data_deterministic.parquet",
        "alg": "HER_hdg",
    },
]

for i, exp in enumerate(experiments, 1):
    print(f"\n" + "=" * 60)
    print(f"Benchmarking DTG for experiment {i}/{len(experiments)}")
    print(f"Algorithm: {exp['alg']}")
    print(f"=" * 60 + "\n")

    # Construct the command as a list of arguments
    cmd = [
        "uv",
        "run",
        "path_planning/rta/testing/benchmark.py",
        exp["data"],
        "--polar",
        "--min_dist",
        "--no-plot"
    ]

    # Run the command and stream output directly to the console
    result = subprocess.run(cmd, check=False)

    # If a run fails, stop the entire script
    if result.returncode != 0:
        print(
            f"\n[ERROR] Experiment {i} failed with exit code {result.returncode}. Exiting."
        )
        sys.exit(result.returncode)

print("\nAll data gathering sessions completed successfully!")