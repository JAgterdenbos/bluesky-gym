# path_planning

Research sub-package for runway-specific 4D path planning with Required Time of Arrival (RTA) in BlueSky-Gym. It trains and evaluates a single-aircraft "worker" policy (SAC + HER) on the `PathPlanningGoalEnv-v0` / `-heading` environments, and provides the RTA sampler pipeline that the worker uses to sample achievable arrival-time goals.

It is a `uv` workspace member (`path-planning` package) that depends on `bluesky-gym`. It is in turn a dependency of [`cps_coordination`](../cps_coordination/README.md), which reuses the trained worker as a frozen policy inside a multi-aircraft coordination layer.

## Layout

```
path_planning/
├── main.py                  CLI entry point (train / evaluate / enjoy / rta subcommands)
├── experiment/
│   ├── base.py               PathPlanningExperiment + its Model/Env config subclasses
│   ├── base_critic.py         BaseCriticExperiment / CriticProbe for critic ablations
│   └── registry.py           PathPlanningRegistry (per-run intent/priority/quality tracking)
├── rta/                      RTA goal-sampler pipeline
│   ├── collect.py             BaseDataCollector — streams per-step data from successful episodes to CSV
│   ├── sampling.py            ExtraTreesSampler / GeoRunwaySampler (sklearn-based BaseSampler)
│   ├── train_sampler.py       fits a sampler from collected data
│   ├── test_sampler.py        evaluation utilities
│   └── testing/                sampler benchmarking & analysis (multiple sampler variants, plotting)
├── critic/                   Critic/reward-shaping ablation experiments (DecomposedSAC, TTG/reward sweeps)
└── configs/                  YAML experiment configs (base, long_run, main_model, spatial/temporal comparisons)
```

## Experiment framework

`path_planning/experiment/base.py` is the only file needed to plug an environment into `bluesky_gym`'s experiment framework:

- `PathPlanningModelConfig` — adds `net_arch`, `policy_kwargs`, and HER-specific fields on top of `bluesky_gym.experiment.ModelConfig`
- `PathPlanningEnvKwargsConfig` — `gym.make()` kwargs: `action_mode`, `use_rta`, `runways`
- `PathPlanningEnvConfig` — wraps the above, sets `env_name`, `group_key`, `success_key`
- `PathPlanningExperiment` (subclass of `bluesky_gym.experiment.BaseExperiment`) — implements `make_env`, `make_model`, and the `MetricExtractor`

`PathPlanningRegistry` (subclass of `bluesky_gym.experiment.BaseRegistry`) tracks per-run `intent`, `priority`, `status`, and `quality` in a CSV; per-run hyperparameters live in each run's `config.yaml` instead.

Every field on the config dataclasses is auto-exposed as a `--{section}-{field}` CLI flag by the shared `bluesky_gym` runner (`run_experiment`).

## RTA sampler pipeline

`path_planning/rta/` implements a collect → fit → evaluate pipeline for modeling the distribution of achievable arrival times given aircraft state and runway:

1. **`collect.py`** (`BaseDataCollector`) streams per-step data from successful training episodes to CSV, chunk-flushed to bound memory.
2. **`sampling.py`** (`ExtraTreesSampler`, `GeoRunwaySampler`) wraps a scikit-learn `ExtraTreesRegressor` behind `bluesky_gym.envs.common.base_sampler.BaseSampler`'s `fit`/`sample` interface.
3. **`train_sampler.py`** / **`test_sampler.py`** fit and evaluate the sampler.
4. **`testing/`** holds a wider registry of alternative sampler implementations (`dtg_sampler.py`, `kde_dtg_sampler.py`, `linear_dtg_sampler.py`, `neighbours_dtg_sampler.py`, `neural_dtg_sampler.py`, `tree_dtg_sampler.py`, via `SamplerRegistry`), plus benchmarking (`benchmark.py`), trajectory/spatial analysis, and a `plot/` sub-package (mesh/contour/surface renderers) for comparing sampler variants.

A fitted sampler (e.g. `runway_sampler_deterministic_polar_main_15.joblib`) is loaded by the worker's env config to sample RTA goals during training, and later reused by `cps_coordination`'s `ETASurrogate` training data.

## Critic experiments

`path_planning/critic/` runs ablation studies on the SAC critic and reward shaping:

- `continouos_critic.py` — `DecomposedSAC`, exact gradient computation, dominance metrics
- `reward_experiment.py` — `RewardDecompositionWrapper`, reward-scale sweep configs/experiments
- `ttg_experiment.py` — time-to-go dominance experiment
- `run_reward_sweep.py` / `run_ttg_sweep.py` / `plot_reward_sweep.py` / `plot_ttg_sweep.py` — sweep runners and plotting

## CLI usage

```bash
# Train with dataclass defaults
python path_planning/main.py

# Load from YAML config, override one field on top
python path_planning/main.py --config path_planning/configs/base.yaml --session-total-timesteps 500000

# Session overrides
python path_planning/main.py --session-total-timesteps 500000
python path_planning/main.py --session-train-groups 27 18R 06
python path_planning/main.py --session-eval-groups 27 18R
python path_planning/main.py --session-eval-episodes 20
python path_planning/main.py --session-no-do-evaluate
python path_planning/main.py --session-track-training-evals

# Model overrides
python path_planning/main.py --model-learning-rate 1e-4
python path_planning/main.py --model-no-use-her
python path_planning/main.py --model-her-n-sampled-goal 8
python path_planning/main.py --model-her-goal-selection-strategy future
python path_planning/main.py --model-algorithm TD3

# Env overrides
python path_planning/main.py --env-action-mode wpt
python path_planning/main.py --env-use-rta
python path_planning/main.py --env-runways 27 18R

# Evaluate / watch a saved model
python path_planning/main.py evaluate --run-id 20260331_134059
python path_planning/main.py enjoy    --run-id 20260331_134059 --groups 27

# RTA sampler pipeline
python path_planning/main.py collect-rta      # Collect RTA data step-by-step per successful episode
python path_planning/main.py analyse-rta      # Analyse RTA data
python path_planning/main.py benchmark-rta    # Benchmark samplers on RTA data
python path_planning/main.py fit-and-plot-rta # Fit and plot RTA data

# Full flag list
python path_planning/main.py --help
```

### Configs

- `configs/base.yaml` — default config: SAC+HER, 100k timesteps, `PathPlanningGoalEnv-v0`
- `configs/long_run.yaml` — 500k timestep SAC+HER config
- `configs/main_model/spatial.yaml` — phase-1 (spatial): SAC no-HER, 500k timesteps, 500 eval episodes
- `configs/main_model/temporal.yaml` — phase-2 (temporal): resumes from the phase-1 pretrained run, adds the fitted RTA sampler
- `configs/spatial_comparison/` and `configs/temporal_comparison/` — HER × environment-variant ablation grids (`her`, `her_hdg`, `no_her`, `no_her_hdg`, with an `extended`-duration variant under `temporal_comparison/`)

Priority chain: dataclass defaults → YAML config → CLI flags.

## Relationship to bluesky_gym and cps_coordination

- Depends on `bluesky_gym` for the environments (`PathPlanningGoalEnv-v0`, `-heading`), the experiment framework (`BaseExperiment`, `ExperimentConfig`, `BaseRegistry`), and `BaseSampler`.
- The worker policy trained here (`experiments/PathPlanningGoalEnv-v0/SAC/models/<run_id>/final_model.zip`) is loaded as a **frozen** policy by `cps_coordination.experiments.coordination_baseline.CPSCoordinationExperiment` via `session.pretrained_run_id` / `pretrained_model_path`, and driven inside `bluesky_gym`'s `MultiAgentPathPlanningGoalEnv` under CPS coordination.
- RTA training data collected here also feeds `cps_coordination`'s `ETASurrogate` training pipeline (`cps_coordination/testing/train_surrogate.py` consumes the same parquet output as `collect-rta`).

## Relationship to the thesis (`docs/paper/Thesis_Paper_draft.pdf`)

This package implements the tactical layer of the thesis's Goal-Conditioned Hierarchical Reinforcement Learning (GCHRL) framework for the Trajectory Based Aircraft Landing Problem (TBALP):

- **`experiment/base.py` (`PathPlanningExperiment`, SAC + HER)** implements the paper's **"4D-Worker"** — the tactical RL agent trained with Hindsight Experience Replay (HER) on Soft Actor-Critic (SAC) via Stable-Baselines3. The paper evaluates it using metrics computed from the fields this package logs: RTA tracking error (ε_RTA), path tortuosity (T), spatial visitation entropy (H), and success rate (S).
- **`configs/main_model/spatial.yaml` then `temporal.yaml`** implement the paper's **"Two-Stage Spatial-to-Spatial-Temporal Training Pipeline"**: phase 1 (spatial) trains the worker without RTA goals; phase 2 (temporal) resumes from that checkpoint and adds RTA-conditioned goals sampled by the fitted DTG sampler.
- **`rta/` (the DTG/Distance-To-Go sampler pipeline, `ExtraTreesSampler` / `SamplerRegistry`)** implements the paper's **"DTG Sampler"** — modeled with an Extra Trees (ET) Regressor (Geurts et al.) and validated against competing sampler families (KNN, linear, ridge, lasso, MLP, gradient-boosted, etc. in `rta/testing/samplers/`) using the Diebold-Mariano and Wilcoxon signed-rank tests. The paper documents this as a **future-proofing** choice — modeling distance-to-go rather than the more constrained time-to-go, so the same sampler adapts to variable aircraft speed and wind — and notes the memory-cost trade-off between a 100-estimator and a slimmer 15-estimator forest for deployment.
- The paper also uses this DTG sampler's architecture as an **(explicitly flagged, unvalidated) analogy** for `cps_coordination`'s `ETASurrogate` — see that package's README for the corresponding "ETA Surrogate Model" mapping.
