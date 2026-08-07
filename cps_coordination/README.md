# cps_coordination

Hierarchical Constrained Position Shifting (CPS) coordination layer for 4D RTA path planning in BlueSky-Gym. It sits above a frozen single-aircraft `path_planning` worker policy and sequences multiple concurrent aircraft onto shared runways, assigning each a Target Time of Arrival (TTA) that respects RECAT-EU wake-turbulence separation.

It is a `uv` workspace member (`cps-coordination` package, console script `cps-sim`) depending on `bluesky-gym` and `path-planning`.

## Layout

```
cps_coordination/
├── main.py                          CLI entry point (train / evaluate / enjoy / registry)
├── coordination/
│   ├── cps_manager.py                 CPSManager — k-CPS sequencing + greedy TTA scheduler
│   ├── eta_surrogate.py               ETASurrogate — self-describing ETA prediction model
│   └── trajectory_buffer.py           TrajectoryBuffer — per-aircraft rolling state history for lag features
├── experiments/
│   ├── config.py                      CPSModelConfig / CPSEnvConfig / CPSEnvKwargsConfig
│   └── coordination_baseline.py       CPSCoordinationExperiment + CPSCoordinationRegistry
├── testing/                          validation/regression gates only (see below)
├── scripts/                           offline surrogate-training, Step 10 batch-eval, and paper-reporting scripts
├── configs/
│   ├── cps_base.yaml                  default experiment config (k-CPS params, RECAT-EU matrix, eval params)
│   └── cps_scale_10k.yaml             Step 10 scale-up config (rolling arrival stream, higher aircraft density)
├── models/                           fitted ETASurrogate .pkl artifacts
└── figures/                          output figures from surrogate_analyse.py / step10_deep_analysis.py
```

## Core coordination algorithm — `CPSManager`

`coordination/cps_manager.py` implements the k-CPS coordination algorithm:

1. Maintain and update an ordered fleet of `N_a` incoming aircraft (`AircraftState` records).
2. Compute the FCFS reference sequence by ascending absolute ETA (`t + T̂_i`), where `T̂_i` is predicted by `ETASurrogate`.
3. Apply the k-CPS constraint: no aircraft may shift more than `k` positions from its FCFS rank.
4. Run the greedy forward scheduler: `TTA_i = max(ETA_i, TTA_{i-1} + ΔT_sep)`, where `ΔT_sep` is read from the RECAT-EU wake-turbulence separation matrix.
5. Support `static` (sector-committed IAFs) and `dynamic` (minimum-feasible-TTA selection) runway assignment modes.
6. Re-evaluate every `delta_t_plan` simulation seconds; propagate goal updates to the worker only when the shift exceeds `delta_update` seconds.

An optional `TrajectoryBuffer` (per-aircraft rolling `(x, y, heading_rad)` deque) supplies lag features (`delta_atd`, `cumabs_cte`, `heading_volatility`) to the surrogate; without one, lag features default to zero for a gracefully degraded prediction.

## `ETASurrogate`

`coordination/eta_surrogate.py` predicts `T̂_i` — remaining simulation steps until an aircraft crosses its assigned Initial Approach Fix (IAF) — from a 13-column feature set (polar position, runway code, elapsed steps, heading decomposition, along-track/cross-track/heading error, plus lag features). It is **self-describing**: `from_training()` bakes in the IAF anchors, surviving feature columns, and target transform, so callers don't need to specify a feature mode at inference.

Training pipeline (`scripts/train_surrogate.py`):
1. Load parquet RTA data (produced by `path_planning`'s `collect-rta`), filter successful episodes, compute steps-to-go.
2. Derive exact IAF anchors from `PathPlanningGoalEnv` constants.
3. Engineer static geometric features, then lag features.
4. 5-fold group cross-validation for OOF metrics.
5. Feature reduction via a scout `ExtraTreesRegressor`; select the best target transform (identity/log1p/sqrt) by OOF RMSE.
6. Fit the final model and package it via `ETASurrogate.from_training()`; save with `joblib`.

`scripts/surrogate_analyse.py` runs exploratory analysis (feature/coordinate-system justification) and writes figures to `cps_coordination/figures/`.

## `CPSCoordinationExperiment`

`experiments/coordination_baseline.py` subclasses `bluesky_gym.experiment.BaseExperiment` as an **evaluation-only** experiment that wraps a frozen worker inside the CPS layer:

- `do_train` is forced `False`; the frozen worker is loaded from `session.pretrained_run_id` (or `pretrained_model_path`) before evaluation.
- `evaluate()` replaces the default single-episode loop with a multi-aircraft coordination loop over one shared `MultiAgentPathPlanningGoalEnv` instance, reused across episodes:
  1. Reset the shared env to spawn the episode's `N_a` aircraft.
  2. Each decision step: build `AircraftState` records from the env's active aircraft, run `CPSManager.update_fleet`, push resulting runway/TTA changes into the env, predict with the frozen worker in a single batched call, and step.
  3. Repeat until every aircraft for the episode has landed or been truncated.
  4. Compute aggregate metrics across episodes and save logs.
- Per-episode records go to `<save_path>/cps_eval_log.csv`; aggregate metrics to `<save_path>/cps_metrics.yaml`.
- `CPSCoordinationRegistry` (subclass of `bluesky_gym.experiment.BaseRegistry`) is a persistent CSV registry tracking CPS coordination run metadata.

`experiments/config.py` defines `CPSModelConfig` (k-CPS hyperparameters: `k_cps`, `delta_t_plan`, `delta_update`, `runway_assignment_mode`, `eta_surrogate_path`), `CPSEnvConfig`, and `CPSEnvKwargsConfig` (env kwargs: `v_app`, `runways`, sampler path) — all auto-exposed as `--model-*` / `--env-*` CLI flags by the shared framework runner.

## `testing/` — validation/regression gates

Genuine validation/regression scripts only — everything else (production launch
drivers, offline analysis, paper-reporting) lives in `scripts/`, below.

- **`validate_multiagent_env.py`** — regression gates for `MultiAgentPathPlanningGoalEnv` itself: (1) at `max_concurrent_aircraft=1, n_aircraft_total=1` it must reproduce `PathPlanningGoalEnv-v0` bit-for-bit given the same seed/actions; (2) at `max_concurrent_aircraft=2, n_aircraft_total=3` it verifies the acid→traffic-index remap survives a mid-episode delete/respawn with no observation "jumping" between aircraft.
- **`validate_cps_pipeline.py`** — exercises the real `CPSCoordinationExperiment` evaluation pipeline end-to-end with an actual frozen SAC worker: confirms TTA injection changes worker behaviour vs. an uncoordinated control run (k=0 FCFS), and that every consecutive landing pair on a forced-shared runway respects RECAT-EU separation (k>0, `N_a > 2`).
- **`validate_surrogate.py`** — held-out CV + end-to-end condition-3 gate for `ETASurrogate` (imports the training/analysis pipeline from `scripts/`).
- **`smoke_test_step10.py`** — capped local sanity check of `scripts/run_batch_eval.py`'s real sweep machinery before a full/cluster run.
- **`telemetry.py`** — shared Parquet telemetry collector, used by both validation gates and the `scripts/` production pipeline.

## `scripts/` — production pipelines, offline analysis, paper reporting

- **`train_surrogate.py`** / **`surrogate_data.py`** / **`select_surrogate_features.py`** / **`surrogate_analyse.py`** — `ETASurrogate` training pipeline (feature selection → training → exploratory analysis); see above.
- **`diagnose_success_rate.py`** — standalone success-rate diagnostic, also exercised by `testing/validate_surrogate.py`'s end-to-end gate.

### Step 10 scale-up evaluation path

The production evaluation flow for the Phase III Step 10 scale-up (rolling arrival stream, `configs/cps_scale_10k.yaml`) lives in `scripts/`, layered on top of the validation gates in `testing/`:

- **`run_batch_eval.py`** — production batch driver: sweeps `k_cps` × `runway_assignment_mode` × `fairness_weight`, streaming telemetry to Parquet via `testing/telemetry.py`.
- **`run_cps_eval.py`** — single-combo (non-sweep) evaluation driver; `run_batch_eval.py` reuses its `_log_episode`.
- **`run_step10_scale10k.sh`** — the actual launch script wrapping `run_batch_eval.py` with the current production combo grid and per-mode `fairness_weight` values.
- **`regenerate_step10_sanity_sweep.sh`** — regenerates the M=100 sanity-sweep dataset used to validate exit criterion #6.
- **`merge_shards.py`** — merges sharded (`SHARDS`/`SHARD_INDEX`-split) Parquet output back into the unsharded per-combo layout.
- **`cps_metrics_offline.py`** → **`summarize_batch_sweep.py`** → **`step10_deep_analysis.py`** → **`analyze_fairness_weight_offline.py`** — a layered offline-metrics pipeline: base per-combo metric recomputation, sweep-wide tabulation, deep collision/stall/tortuosity diagnostics, and the `fairness_weight` calibration analysis, each building on the one before it rather than duplicating it.
- **`generate_paper_report.py`** — single consolidated script producing every LaTeX table/figure the Phase III thesis chapter needs, wrapping (not reimplementing) the four scripts above; output lands in `cps_coordination/figures/paper_report/`. See `.claude/plans/phase3_cps_coordination_plan.md` for the session that built it.

### Running Step 10 in a dedicated terminal (macOS)

`run_step10_scale10k.sh` is sized for a cluster/runner, not a quick local check — the full 4-combo grid at M=2,000 measures ~8.2h wall-clock sequentially (~2.1h/combo). Three scripts wrap it for unattended local use, each doing one job:

| Script | Purpose |
|---|---|
| `scripts/launch_step10_dedicated_terminal.sh` | Start a run, safely, from Terminal.app |
| `scripts/step10_progress.sh` | Check (or watch) how far along a run is |
| `scripts/step10_stop.sh` | Stop a run cleanly and resumably |

#### Starting a run

```bash
./cps_coordination/scripts/launch_step10_dedicated_terminal.sh
```

Run it from the repo root (it also `cd`s there itself, so it works from anywhere). It:

1. Auto-resolves the frozen worker `run_id` (latest run with a `final_model.zip`) unless `RUN_ID` is set.
2. Asserts `cps_scale_10k.yaml`'s runway scope still resolves to `18R 27` and **aborts instead of launching** if it doesn't — `run_step10_scale10k.sh` never passes `--runways` itself, it relies entirely on that YAML field, so this catches a silent drift back to all-12-runways before an 8-hour run rather than after.
3. Prints the full resolved config (runway scope, episodes/combo, worker checkpoint path, per-mode `fairness_weight`, save root) for you to review.
4. Runs the capped M=10 smoke test (skip with `SKIP_SMOKE=1` — not recommended except when resuming a `SAVE_ROOT` already smoke-tested this session).
5. Asks for a `y/N` confirmation.
6. Launches with `PYTHONUNBUFFERED=1` (Python block-buffers stdout when it's not a TTY, i.e. redirected to a file — without this, progress prints wouldn't reach the log until an internal buffer happened to fill), wrapped in `nohup caffeinate -i ... &` + `disown` so it survives closing the Terminal window and blocks idle sleep, logging to a timestamped file under `cps_coordination/data/`.

It then prints exactly what to run next:
```
Progress:                  ./cps_coordination/scripts/step10_progress.sh <save_root>
Watch + notify on stop:    ./cps_coordination/scripts/step10_progress.sh <save_root> --watch --notify
Stop cleanly (resumable):  ./cps_coordination/scripts/step10_stop.sh <save_root>
Resume later:              RESUME=1 SAVE_ROOT=<save_root> ./cps_coordination/scripts/launch_step10_dedicated_terminal.sh
```

Useful env var overrides (see the script header for the full list): `RUN_ID`, `SAVE_ROOT`, `COMBO="k_cps:mode:fw"` (single combo instead of the full sweep — e.g. `COMBO="3:dynamic:0.5"`), `RESUME=1`, `STATIC_FW`/`DYNAMIC_FW`.

`caffeinate -i` only blocks *idle* sleep, not a manually closed laptop lid, so keep the machine plugged in with the lid open (or set `caffeinate -s` yourself if unsure about your sleep-on-lid-close setting).

#### Checking progress

```bash
./cps_coordination/scripts/step10_progress.sh                    # newest SAVE_ROOT, one-shot
./cps_coordination/scripts/step10_progress.sh <save_root> --watch --notify
```

Per combo, it shows exact episode/success-rate counts once a combo's Parquet file is durably readable (`source: parquet (final)`), or the latest `[N/M] episodes logged` line from the log while a combo is still in progress (`source: log, attempted (live)`) — `run_batch_eval.py`'s Parquet writer stays open for a combo's *entire* run and is only closed when that combo finishes or is cleanly stopped, so it's genuinely unreadable the whole time in between, not just briefly. A trailing `Verdict:` line summarizes `RUNNING` / `COMPLETE` / `STOPPED_EARLY (x/y combos complete)`.

`--watch --notify` refreshes every `INTERVAL` seconds (default 30) and, the moment it detects the process has stopped — whether it finished normally or died/was interrupted — fires a native macOS notification with the verdict and exits, so an 8-20h run doesn't need a terminal babysat the whole time.

#### Stopping a run

```bash
./cps_coordination/scripts/step10_stop.sh <save_root>
```

Don't `kill` a PID directly. `run_step10_scale10k.sh`'s "full sequential" mode runs static and dynamic mode as two *separate* `run_batch_eval.py` invocations back to back — signaling only the current worker just makes the wrapper move on and start the next mode's worker fresh, and `$!` from a `nohup caffeinate -i ... &` launch doesn't reliably resolve to the process that can stop that chain either. `step10_stop.sh` finds the actual live worker for a `SAVE_ROOT`, walks up its process ancestry to the `run_step10_scale10k.sh` wrapper (which installs its own signal trap), and signals that — stopping the current combo gracefully (finishes the in-flight episode, flushes and closes telemetry) *and* preventing any further combo/mode from starting. Safe to resume afterward with `RESUME=1`.

**Alternative: `screen`** (ships with macOS) if you'd rather reattach to a live session than tail a log file — the launcher's nohup approach still applies underneath, this just changes how you observe it:

```bash
screen -S step10
# inside the screen session:
./cps_coordination/scripts/launch_step10_dedicated_terminal.sh
# detach: Ctrl-A then D  (leaves it running)
# reattach later: screen -r step10
```

Once a combo (or the whole sweep) finishes: `uv run python cps_coordination/scripts/step10_deep_analysis.py --sweep-root <save_root>`.

## CLI usage

```bash
# Evaluate with defaults
python -m cps_coordination train

# Load from YAML, override one field
python -m cps_coordination train --config cps_coordination/configs/cps_base.yaml --model-k-cps 5

# Override CPS model fields
python -m cps_coordination train --model-k-cps 3
python -m cps_coordination train --model-delta-t-plan 120
python -m cps_coordination train --model-runway-assignment-mode static

# Override env fields
python -m cps_coordination train --env-v-app 135.0
python -m cps_coordination train --env-runways 27 18R

# Point at a pre-trained worker policy
python -m cps_coordination train --session-pretrained-run-id 20260301_120000

# Evaluate / watch a saved CPS run
python -m cps_coordination evaluate --run-id 20260301_120000
python -m cps_coordination enjoy    --run-id 20260301_120000

# Full flag list
python -m cps_coordination --help

# Validation scripts (not CLI subcommands — run directly)
python cps_coordination/testing/validate_multiagent_env.py
python cps_coordination/testing/validate_cps_pipeline.py
```

Despite the subcommand name, `train` runs an **evaluation-only** experiment by default (`configs/cps_base.yaml` sets `session.total_timesteps: 0`, `do_train: false`, `do_evaluate: true`) — there is no CPS-layer training, only sequencing evaluation of a frozen worker.

### `configs/cps_base.yaml`

Priority chain: dataclass defaults → this YAML → CLI flags. Key sections:

- `model` — k-CPS hyperparameters (`k_cps`, `delta_t_plan`, `delta_update`, `runway_assignment_mode`, `eta_surrogate_path`)
- `session` — evaluation-only session settings, `pretrained_run_id`/`pretrained_model_path` for the frozen worker, `eval_episodes`
- `env` — `env_name: CPSCoordination` (experiment-path namespace only, kept distinct from the frozen worker's own `PathPlanningGoalEnv-v0` so eval runs land under `experiments/CPSCoordination/...` instead of scattering into the worker's training-run directory tree), `group_key`, `success_key`, `env_kwargs` (`action_mode`, `runways`)
- `recat_eu` — RECAT-EU time-based separation matrix (seconds) by leading/trailing wake-turbulence category (A–F)
- `cps_eval` — evaluation parameters read by `CPSCoordinationExperiment.evaluate()`: `n_aircraft_per_episode`, `separation_tolerance_s`, `ripple_lag`, `throughput_window_h`

## Relationship to path_planning and bluesky_gym

- Depends on `bluesky_gym` for `MultiAgentPathPlanningGoalEnv`, the experiment framework (`BaseExperiment`, `ExperimentConfig`, `BaseRegistry`), and `PathPlanningGoalEnv` constants (IAF anchors).
- Depends on `path_planning` for the frozen worker policy (trained via `path_planning/main.py`) and the RTA parquet data (`path_planning`'s `collect-rta`) used to train `ETASurrogate`.
- Conceptually: `path_planning` trains the **tactical, single-aircraft 4D worker**; `cps_coordination` adds the **strategic, multi-aircraft sequencing layer** on top, without retraining the worker.

## Relationship to the thesis (`docs/paper/Thesis_Paper_draft.pdf`)

This package implements the strategic/coordination half of the thesis's Goal-Conditioned Hierarchical Reinforcement Learning (GCHRL) framework for the Trajectory Based Aircraft Landing Problem (TBALP), which the paper frames as a rule-based coordinator wrapped around a learned tactical policy (as opposed to the fully learned multi-agent baselines it cites, e.g. Groot et al.'s multi-agent hierarchical RL, MSTAGNN-MARL):

- **`coordination/cps_manager.py` (`CPSManager`)** implements the paper's **"CPS Manager"** — the Constrained Position Shifting scheduler, grounded in the classical Aircraft Landing Problem / CPS literature (Balakrishnan & Chandran, Beasley et al.'s Heathrow population heuristic, MILP ALP formulations) but applied here as an online k-CPS + greedy-forward-scheduling heuristic rather than an offline MILP solve.
- **`coordination/eta_surrogate.py` (`ETASurrogate`)** implements the paper's **"ETA Surrogate Model"**, likewise an Extra Trees (ET) Regressor via scikit-learn. The paper explicitly flags this as an **unvalidated architecture-transfer analogy** from `path_planning`'s DTG (Distance-To-Go) sampler — same modeling approach, but retargeted from spatial distance-to-go to a time-domain ETA prediction at the IAF, informed by related ETA-prediction literature (Dhief et al., "Predicting Aircraft Landing Time in Extended-TMA").
- **`CPSManager` + `ETASurrogate` + the frozen `path_planning` worker (the paper's "4D-Worker")** together form the paper's **"Manager-Surrogate-Worker TBO-ALP Bridge Loop"** — the closed loop by which the strategic CPS layer reads ETA predictions, computes TTAs, and injects them as goals into the tactical worker each replanning cycle. `CPSCoordinationExperiment.evaluate()` is the code-level implementation of that loop.
- The wider GCHRL framework this package sits under also connects to the paper's framing of **Trajectory Based Operations (TBO)** and the SESAR ATM Master Plan, the **runway overload phenomenon** at Amsterdam Schiphol's multi-runway airspace (the paper's motivating case study), and the **Goal-Conditioned MDP** formalism shared with `path_planning`'s worker.
- See `path_planning`'s README for the corresponding "4D-Worker" / "DTG Sampler" mapping on the tactical side.
