"""
enjoy.py
--------
Watch or record a previously trained model.

The config is loaded from the config.json saved alongside the model,
so enjoy.py always uses the exact same settings the run was trained with
(env name, action mode, eval runways, etc.) — no hardcoded defaults.

Usage
-----
  python enjoy.py --run-id 20260331_134059
  python enjoy.py --run-id 20260331_134059 --episodes 10
  python enjoy.py --run-id 20260331_134059 --record
  python enjoy.py --run-id 20260331_134059 --runways 27 18R
"""

from __future__ import annotations

import argparse

import bluesky_gym
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch or record a trained PathPlanning model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-id", type=str, required=True,
        help="Run ID to load (e.g. 20260331_134059).",
    )
    p.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to run.",
    )
    p.add_argument(
        "--record", action="store_true",
        help="Record episodes to video (saves to <save_path>/videos/).",
    )
    p.add_argument(
        "--runways", nargs="+", default=None,
        metavar="RWY",
        help="Override the eval runways from the saved config.",
    )
    p.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic actions (default: True).",
    )
    return p.parse_args()


def enjoy(
    run_id:       str,
    episodes:     int  = 5,
    record:       bool = False,
    runways:      list | None = None,
    deterministic: bool = True,
) -> None:
    bluesky_gym.register_envs()

    # Reconstruct the exact config from the saved JSON — no guesswork
    cfg = ExperimentConfig.load(run_id)
    print(f"📺 Run: {run_id}  |  env={cfg.session.env_name}"
          f"  |  algo={cfg.model.algorithm.__name__}")

    # Allow caller to override the runway list (e.g. test on a new runway)
    eval_kwargs = cfg.eval_env_kwargs
    if runways is not None:
        eval_kwargs["runways"] = runways

    render_mode = "rgb_array" if record else "human"
    env = gym.make(cfg.session.env_name, render_mode=render_mode, **eval_kwargs)

    if record:
        video_dir = f"{cfg.save_path}/videos"
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda _: True)
        print(f"🎥 Recording to {video_dir}")

    model_path = f"{cfg.save_path}/final_model.zip"
    model = cfg.model.algorithm.load(model_path, env=env)
    print(f"✅ Model loaded from {model_path}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            # Unwrap 0-d numpy array that some envs return for single-action spaces
            if hasattr(action, "shape") and action.shape == ():
                action = action[()]
            obs, _, done, truncated, _ = env.step(action)
        print(f"  Episode {ep}/{episodes} done.")

    env.close()
    print("✅ Done.")


if __name__ == "__main__":
    args = parse_args()
    enjoy(
        run_id=args.run_id,
        episodes=args.episodes,
        record=args.record,
        runways=args.runways,
        deterministic=args.deterministic,
    )