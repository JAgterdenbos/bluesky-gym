from bluesky_gym.experiment import BaseRegistry, register_command

class PathPlanningRegistry(BaseRegistry):
    """
    High-level metadata registry for PathPlanning experiments.
    Per-run hyperparameters live in each run's config.yaml.
    This registry tracks intent, priority and qualitative outcome only.

    Headers
    -------
    run_id    : Unique run identifier, auto-set by the framework (e.g. 20260407_153017).
    timestamp : Auto-set when the run is added to the registry.
    intent    : Free-text description of what you were testing (e.g. "HER with future strategy on 27/18R").
    priority  : How important this run is to follow up on: "high" | "medium" | "low".
    status    : Lifecycle state of the run: "running" | "done" | "failed" | "abandoned".
    quality   : Qualitative outcome after reviewing results: "good" | "bad" | "promising" | "inconclusive".
    notes     : Free-text post-run observations (e.g. "converged early, noise reward collapsed").
    """

    @property
    def headers(self):
        return [
            "run_id",
            "timestamp",
            "intent",
            "priority",
            "status",
            "quality",
            "notes",
        ]

    @register_command(
        "Mark the outcome of a finished run.",
        status={"choices": ["running", "done", "failed", "abandoned"]},
        quality={"choices": ["good", "bad", "promising", "inconclusive"]},
        notes={"default": ""},
    )
    def label(self, run_id: str, status: str, quality: str, notes: str = ""):
        self.update_run(run_id, {"status": status, "quality": quality, "notes": notes})
        print(f"✅ Labelled {run_id}: {status} / {quality}")

    @register_command(
        "Set the priority of a run.",
        priority={"choices": ["high", "medium", "low"]},
    )
    def prioritise(self, run_id: str, priority: str):
        self.update_run(run_id, {"priority": priority})
        print(f"🔖 {run_id} → priority: {priority}")

    @register_command("Show a summary table of all runs.")
    def list(self):
        rows = self._read_all()
        if not rows:
            return print("Registry is empty.")

        col_w = {"run_id": 20, "status": 10, "priority": 8, "quality": 12, "intent": 35}
        header = (
            f"{'RUN ID':<{col_w['run_id']}} | "
            f"{'STATUS':<{col_w['status']}} | "
            f"{'PRIO':<{col_w['priority']}} | "
            f"{'QUALITY':<{col_w['quality']}} | "
            f"INTENT"
        )
        print(f"\n{header}")
        print("-" * len(header))
        for r in rows:
            print(
                f"{r.get('run_id', ''):<{col_w['run_id']}} | "
                f"{r.get('status', ''):<{col_w['status']}} | "
                f"{r.get('priority', ''):<{col_w['priority']}} | "
                f"{r.get('quality', ''):<{col_w['quality']}} | "
                f"{r.get('intent', '')}"
            )
        print()

    @register_command(
        "Filter runs by quality.",
        quality={"choices": ["good", "bad", "promising", "inconclusive"]},
    )
    def filter(self, quality: str):
        rows = self._read_all()
        matched = [r for r in rows if r.get("quality") == quality]
        if not matched:
            return print(f"No runs with quality='{quality}'.")
        for r in matched:
            print(f"{r.get('run_id', '')} | {r.get('status', '')} | {r.get('notes', '')}")