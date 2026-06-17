import subprocess
import sys

episodes = 100_000
stochastic = False
fresh_start = False
use_parquet = True

chunk = 10_000
verbose_frequency = 10_000
verbose_store = True

out_path = "path_planning/rta/data/temporal"

training_timestamps = "350k"

def get_filepath(alg: str) -> str:
    ext = "parquet" if use_parquet else "csv"
    filename = f"{training_timestamps}_training_rta_data.{ext}"
    return f"{out_path}/{alg}/{filename}"


# Define the experiments as a list of dictionaries for clean organization
experiments = [
    {
        "run_id": "20260602_165242",
        "alg": "No_HER",
    },
    {
        "run_id": "20260602_221818",
        "alg": "HER",
    },
    {
        "run_id": "20260603_035602",
        "alg": "No_HER_hdg",
    },
    {
        "run_id": "20260603_092624",
        "alg": "HER_hdg",
    },
]

for i, exp in enumerate(experiments, 1):
    print(f"\n" + "=" * 60)
    print(f"Gathering data for experiment {i}/{len(experiments)}")
    print(f"Run ID: {exp['run_id']}")
    print(f"Algorithm: {exp['alg']}")
    print(f"Episodes: {episodes}")
    print(f"Chunk: {chunk}")
    print(f"Verbose Frequency: {verbose_frequency}")
    print(f"Verbose Store: {verbose_store}")
    print(f"=" * 60 + "\n")

    # Construct the command as a list of arguments
    cmd = [
        "uv",
        "run",
        "path_planning",
        "collect-rta",
        exp["run_id"],
        "--episodes",
        str(episodes),
        "--out",
        get_filepath(exp["alg"]),
        "--chunk",
        str(chunk),
        "--verbose_frequency",
        str(verbose_frequency),
    ]

    if stochastic:
        cmd.append("--stochastic")
    if not fresh_start:
        cmd.append("--no-fresh-start")
    if verbose_store:
        cmd.append("--verbose-store")

    # Run the command and stream output directly to the console
    result = subprocess.run(cmd, check=False)

    # If a run fails, stop the entire script
    if result.returncode != 0:
        print(
            f"\n[ERROR] Experiment {i} failed with exit code {result.returncode}. Exiting."
        )
        sys.exit(result.returncode)

print("\nAll data gathering sessions completed successfully!")