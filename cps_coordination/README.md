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
├── testing/                          validation & offline surrogate-training scripts
├── configs/
│   └── cps_base.yaml                  default experiment config (k-CPS params, RECAT-EU matrix, eval params)
├── models/                           fitted ETASurrogate .pkl artifacts
└── figures/                          output figures from surrogate_analyse.py
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

Training pipeline (`testing/train_surrogate.py`):
1. Load parquet RTA data (produced by `path_planning`'s `collect-rta`), filter successful episodes, compute steps-to-go.
2. Derive exact IAF anchors from `PathPlanningGoalEnv` constants.
3. Engineer static geometric features, then lag features.
4. 5-fold group cross-validation for OOF metrics.
5. Feature reduction via a scout `ExtraTreesRegressor`; select the best target transform (identity/log1p/sqrt) by OOF RMSE.
6. Fit the final model and package it via `ETASurrogate.from_training()`; save with `joblib`.

`testing/surrogate_analyse.py` runs exploratory analysis (feature/coordinate-system justification) and writes figures to `cps_coordination/figures/`.

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

## `testing/` — validation scripts

- **`validate_multiagent_env.py`** — regression gates for `MultiAgentPathPlanningGoalEnv` itself: (1) at `max_concurrent_aircraft=1, n_aircraft_total=1` it must reproduce `PathPlanningGoalEnv-v0` bit-for-bit given the same seed/actions; (2) at `max_concurrent_aircraft=2, n_aircraft_total=3` it verifies the acid→traffic-index remap survives a mid-episode delete/respawn with no observation "jumping" between aircraft.
- **`validate_cps_pipeline.py`** — exercises the real `CPSCoordinationExperiment` evaluation pipeline end-to-end with an actual frozen SAC worker: confirms TTA injection changes worker behaviour vs. an uncoordinated control run (k=0 FCFS), and that every consecutive landing pair on a forced-shared runway respects RECAT-EU separation (k>0, `N_a > 2`).
- **`train_surrogate.py`** / **`surrogate_analyse.py`** — see above.

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
- `env` — `env_name: PathPlanningGoalEnv-v0`, `group_key`, `success_key`, `env_kwargs` (`action_mode`, `runways`)
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
