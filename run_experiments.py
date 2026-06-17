import subprocess
import sys

# Define the experiments as a list of dictionaries for clean organization
experiments = [
    {
        "config": "path_planning/configs/temporal_comparison/extended/no_her.yaml",
    },
    {
        "config": "path_planning/configs/temporal_comparison/extended/her.yaml",
    },
    {
        "config": "path_planning/configs/temporal_comparison/extended/no_her_hdg.yaml",
    },
    {
        "config": "path_planning/configs/temporal_comparison/extended/her_hdg.yaml",
    },
]

for i, exp in enumerate(experiments, 1):
    print(f"\n" + "=" * 60)
    print(f"Running experiment {i}/{len(experiments)}")
    print(f"Config: {exp['config']}")
    print(f"=" * 60 + "\n")

    # Construct the command as a list of arguments
    cmd = [
        "uv",
        "run",
        "path_planning",
        "train",
        "--config",
        exp["config"],
    ]

    # Run the command and stream output directly to the console
    result = subprocess.run(cmd, check=False)

    # If a run fails, stop the entire script
    if result.returncode != 0:
        print(
            f"\n[ERROR] Experiment {i} failed with exit code {result.returncode}. Exiting."
        )
        sys.exit(result.returncode)

print("\nAll training sessions completed successfully!")