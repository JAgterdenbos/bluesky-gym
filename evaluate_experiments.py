import subprocess
import sys

episodes = 10_000
no_render = True

# Define the experiments as a list of dictionaries for clean organization
experiments = [
    {
        "run_id": "20260602_165242",
    },
    {
        "run_id": "20260602_221818",
    },
    {
        "run_id": "20260603_035602",
    },
    {
        "run_id": "20260603_092624",
    },
]

for i, exp in enumerate(experiments, 1):
    print(f"\n" + "=" * 60)
    print(f"Evaluating experiment {i}/{len(experiments)}")
    print(f"Run ID: {exp['run_id']}")
    print(f"=" * 60 + "\n")

    # Construct the command as a list of arguments
    cmd = [
        "uv",
        "run",
        "path_planning",
        "evaluate",
        "--run-id",
        exp["run_id"],
        "--episodes",
        str(episodes),
    ]

    if no_render:
        cmd.append("--no-render")

    # Run the command and stream output directly to the console
    result = subprocess.run(cmd, check=False)

    # If a run fails, stop the entire script
    if result.returncode != 0:
        print(
            f"\n[ERROR] Experiment {i} failed with exit code {result.returncode}. Exiting."
        )
        sys.exit(result.returncode)

print("\nAll evaluation sessions completed successfully!")