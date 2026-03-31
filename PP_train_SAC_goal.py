"""
HER training script for PathPlanningGoalEnv.
This script trains a SAC + HER agent on the PathPlanningGoalEnv, with configurable options for training runways, logging frequencies, and checkpointing. It includes a smoke test to verify that the environment and replay buffer are working correctly before committing to a long training run. After training, it evaluates the trained model on a set of episodes and prints per-runway success rates.
"""

import math
import gymnasium as gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

import bluesky_gym
import bluesky_gym.envs
from bluesky_gym.utils import logger

bluesky_gym.register_envs()


# ── config ────────────────────────────────────────────────────────────────────

ENV_NAME  = "PathPlanningGoalEnv-v0"
ALGORITHM = SAC

TRAIN         = True          # set to True to train, False to skip straight to evaluation
EVAL_EPISODES = 10
TIMESTEPS     = 1         # set to 2_000_000 for a real run
BOOTSTRAP     = False           # load from checkpoint before training

TRAIN_RUNWAYS = ["27", "18R"]
EVAL_RUNWAYS  = TRAIN_RUNWAYS

TRAIN_RUNWAYS = None  # set to None to train on all runways (not recommended — much harder)

# HER
HER_N_SAMPLED_GOAL         = 4
HER_GOAL_SELECTION_STRATEGY = "future"

# paths
LOG_DIR   = f"./logs/{ENV_NAME}/"
SAVE_PATH = f"./pathplanning_models/{ENV_NAME}/{ALGORITHM.__name__}_HER"

# network
NET_ARCH      = dict(pi=[256, 256, 256], qf=[256, 256, 256])
POLICY_KWARGS = dict(net_arch=NET_ARCH)

# logging / saving frequencies (auto-scaled later in main)
SAVE_FREQ     = max(5_000, TIMESTEPS // 200)   # ~200 checkpoints across the run
LOG_FREQ      = 5_000                             # episodes between per-runway prints
DIAG_FREQ     = 5_000                          # steps between reward/action stats


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_log_interval(timesteps: int, n_events: int = 10) -> int:
    """Return a step interval that gives ~n_events log lines across the run."""
    return max(1_000, int(timesteps / (n_events * math.log10(max(timesteps, 10)))))


# ── env factory ───────────────────────────────────────────────────────────────

def make_env(runways: list[str] | None = None, render_mode: str | None = None) -> gym.Env:
    """
    Create a Monitor-wrapped PathPlanningGoalEnv.

    Monitor injects an "episode" key into info dicts at true episode
    boundaries, which is required for GoalSuccessLoggerCallback to fire
    correctly when HerReplayBuffer is in use.
    """
    env = gym.make(ENV_NAME, render_mode=render_mode, runways=runways, action_mode="hdg")
    return Monitor(env)


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(env: gym.Env, checkpoint_path: str | None = None) -> SAC:
    """
    Construct a fresh SAC model with HerReplayBuffer.

    If checkpoint_path is provided the weights are loaded from that file
    (note: algorithm.load() returns a *new* object — the return value must be
    used, which is why BOOTSTRAP was silently broken in the original script).
    """
    model = ALGORITHM(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        policy_kwargs=POLICY_KWARGS,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=HER_N_SAMPLED_GOAL,
            goal_selection_strategy=HER_GOAL_SELECTION_STRATEGY,
        ),
    )

    if checkpoint_path is not None:
        print(f"Bootstrapping from {checkpoint_path} ...")
        model = ALGORITHM.load(checkpoint_path, env=env)   # reassign — critical!

    return model


# ── callbacks ─────────────────────────────────────────────────────────────────

class SaveModelCallback(BaseCallback):
    """Periodically saves a checkpoint zip."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}/model_checkpoint.zip"
            self.model.save(path)
            if self.verbose:
                print(f"[Save] checkpoint → {path}  (step {self.n_calls})")
        return True


class GoalSuccessLoggerCallback(BaseCallback):
    """
    Tracks per-runway success rate.

    Relies on Monitor injecting info["episode"] at true episode boundaries.
    With HerReplayBuffer the raw done / info locals are not reliable episode
    signals; the "episode" key from Monitor is.

    SB3 sets info["is_success"] = True when compute_reward returns 0
    (i.e. the agent reached the FAF).
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq    = log_freq
        self.ep_count    = 0
        self.runway_wins: dict[str, int] = {}
        self.runway_eps:  dict[str, int] = {}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:   # Monitor injects this at true episode end
                continue

            self.ep_count += 1
            rwy = info.get("current_runway", "unknown")
            self.runway_eps[rwy]  = self.runway_eps.get(rwy, 0) + 1

            if info.get("is_success", False):
                self.runway_wins[rwy] = self.runway_wins.get(rwy, 0) + 1

            if self.ep_count % self.log_freq == 0:
                self._print_rates(f"episode {self.ep_count}")

        return True

    def _print_rates(self, label: str = "") -> None:
        header = f"[GoalLogger] Per-runway success rate"
        if label:
            header += f" @ {label}"
        print(f"\n{header}:")
        for r in sorted(self.runway_eps):
            eps  = self.runway_eps[r]
            wins = self.runway_wins.get(r, 0)
            print(f"  {r:>4s}: {wins}/{eps}  ({100 * wins / eps:.1f} %)")

    def display_final_results(self) -> None:
        if self.ep_count == 0:
            return
        print("\n── Final per-runway success rates (training) ──")
        self._print_rates()


class DiagnosticsCallback(BaseCallback):
    """
    Logs reward and action distribution statistics from the replay buffer.

    Catches reward hacking, collapsed policies, and broken reward signals
    early — before hours of compute are wasted.
    """

    def __init__(self, log_freq: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        buf = self.model.replay_buffer
        n   = buf.size()
        if n < 100:
            return True

        rewards = buf.rewards[:n].flatten()
        actions = buf.actions[:n]
        print(
            f"\n[Diag @ {self.n_calls:,}]"
            f"  reward  mean={rewards.mean():.3f}  std={rewards.std():.3f}"
            f"  min={rewards.min():.3f}  max={rewards.max():.3f}"
            f"  |  action  mean={actions.mean():.3f}  std={actions.std():.3f}"
        )
        return True


def make_callbacks(
    save_path:   str,
    log_dir:     str,
    eval_env:    gym.Env,
    timesteps:   int,
    save_freq:   int = 10_000,
    log_freq:    int = 100,
    diag_freq:   int = 5_000,
) -> tuple[CallbackList, GoalSuccessLoggerCallback]:
    """
    Assemble the full CallbackList used during training.

    Returns the list and the GoalSuccessLoggerCallback separately so the
    caller can call display_final_results() after training.
    """
    file_name          = f"{ENV_NAME}_{ALGORITHM.__name__}_HER.csv"
    csv_logger         = logger.CSVLoggerCallback(log_dir, file_name)
    save_cb            = SaveModelCallback(save_freq=save_freq, save_path=save_path, verbose=1)
    goal_logger        = GoalSuccessLoggerCallback(log_freq=log_freq, verbose=1)
    diagnostics        = DiagnosticsCallback(log_freq=diag_freq, verbose=1)
    eval_cb            = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_dir,
        eval_freq=max(10_000, timesteps // 20),   # ~20 evals across the run
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callback = CallbackList([save_cb, goal_logger, diagnostics, eval_cb, csv_logger])
    return callback, goal_logger


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_test(env: gym.Env, model: SAC, steps: int = 500) -> None:
    print(f"\n── Smoke test ({steps} steps) ──")

    verbose = model.verbose  # save and suppress SB3's own logs
    model.verbose = 0   # suppress SB3's own logs during the smoke test
    model.learn(total_timesteps=steps)  # use SB3's own collection loop
    model.verbose = verbose  # restore original verbosity

    buf = model.replay_buffer
    if buf is None:
        print("  replay buffer is None — smoke test failed")
        return
    
    n   = buf.size()
    mean_rew = buf.rewards[:n].mean() if n > 0 else float("nan")

    print(f"  buffer size : {n}")
    print(f"  mean reward : {mean_rew:.3f}")

    if n == 0:
        raise RuntimeError("Smoke test failed: replay buffer is empty after training steps. "
                           "Check env.step() returns and HER buffer config.")
    if abs(mean_rew) < 1e-6:
        raise RuntimeError("Smoke test failed: mean reward is exactly 0 — "
                           "compute_reward may be returning success for everything.")

    print("── Smoke test passed ──\n")


# ── training ──────────────────────────────────────────────────────────────────

def train(
    model:       SAC,
    callback:    CallbackList,
    goal_logger: GoalSuccessLoggerCallback,
    timesteps:   int,
    save_path:   str,
) -> None:
    """Run model.learn() then save the final model."""
    log_interval = compute_log_interval(timesteps)
    print(f"Training {ALGORITHM.__name__} + HER on {ENV_NAME} for {timesteps:,} steps ...")
    print(f"  log_interval = {log_interval:,}  (auto-scaled)\n")

    model.learn(
        total_timesteps=int(timesteps),
        callback=callback,
        log_interval=log_interval,
    )

    final_path = f"{save_path}/model"
    model.save(final_path)
    print(f"\nModel saved → {final_path}.zip")

    goal_logger.display_final_results()


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    eval_env:      gym.Env,
    model_path:    str,
    eval_runways:  list[str] | None = None,
    n_episodes:    int = 10,
) -> None:
    """
    Load the saved model and run deterministic evaluation episodes.
    Prints per-episode reward and a per-runway summary.
    """
    print(f"\nLoading model from {model_path}.zip for evaluation ...")
    model   = ALGORITHM.load(model_path, env=eval_env)
    results = {rwy: [] for rwy in eval_runways} if eval_runways is not None else {}

    for i in range(n_episodes):
        done = truncated = False
        obs, info        = eval_env.reset()
        active_runway    = info.get("current_runway", "?")
        total_reward     = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action[()])
            total_reward += reward

        results[active_runway].append(total_reward)
        print(f"  Episode {i + 1:>2d} | runway: {active_runway:>4s} | reward: {total_reward:.3f}")

    print("\n── Per-runway mean reward ──")
    for rwy, rews in results.items():
        if rews:
            print(f"  {rwy:>4s}: {np.mean(rews):.3f}  (n={len(rews)})")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── training env ──────────────────────────────────────────────────────────
    env = make_env(runways=TRAIN_RUNWAYS)

    # ── eval env (headless during training, used by EvalCallback) ─────────────
    # A separate env instance is required — SB3's EvalCallback must not share
    # state with the training env.
    eval_env_headless = make_env(runways=EVAL_RUNWAYS)

    # ── model ─────────────────────────────────────────────────────────────────
    checkpoint = f"{SAVE_PATH}/model_checkpoint.zip" if BOOTSTRAP else None
    model      = build_model(env, checkpoint_path=checkpoint)

    # ── callbacks ─────────────────────────────────────────────────────────────
    callback, goal_logger = make_callbacks(
        save_path  = SAVE_PATH,
        log_dir    = LOG_DIR,
        eval_env   = eval_env_headless,
        timesteps  = TIMESTEPS,
        save_freq  = SAVE_FREQ,
        log_freq   = LOG_FREQ,
        diag_freq  = DIAG_FREQ,
    )

    # ── smoke test ────────────────────────────────────────────────────────────
    smoke_test(env, model, steps=500)
    model = build_model(env, checkpoint_path=checkpoint)   # rebuild to clear replay buffer after smoke test

    # ── train ─────────────────────────────────────────────────────────────────
    if TRAIN:
        train(
            model       = model,
            callback    = callback,
            goal_logger = goal_logger,
            timesteps   = TIMESTEPS,
            save_path   = SAVE_PATH,
        )

    env.close()
    eval_env_headless.close()

    EVAL_EPISODES = 1

    # ── evaluate ──────────────────────────────────────────────────────────────
    eval_env = make_env(runways=EVAL_RUNWAYS, render_mode="human")
    evaluate(
        eval_env     = eval_env,
        model_path   = f"{SAVE_PATH}/model",
        eval_runways = EVAL_RUNWAYS,
        n_episodes   = EVAL_EPISODES,
    )
    eval_env.close()


if __name__ == "__main__":
    main()