import argparse
import gymnasium as gym
import bluesky_gym
from gymnasium.wrappers import RecordVideo
from config import ExperimentConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Watch or record a trained model.")
    parser.add_argument("--run-id", type=str, required=True, help="Run ID to load (e.g. 20260331_134059)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--record", action="store_true", help="Record episodes to video")
    return parser.parse_args()


def enjoy(run_id: str, episodes: int = 5, record: bool = False):
    bluesky_gym.register_envs()

    cfg = ExperimentConfig(run_id=run_id)
    model_path = f"{cfg.save_path}/final_model.zip"

    print(f"📺 Loading Model from Run: {run_id}")

    render_mode = "rgb_array" if record else "human"
    env = gym.make(
        cfg.session.env_name,
        render_mode=render_mode,
        runways=cfg.session.eval_runways,
        action_mode="hdg"
    )

    if record:
        video_dir = f"{cfg.save_path}/videos"
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda _: True)
        print(f"🎥 Recording videos to {video_dir}")

    model = cfg.model.algorithm.load(model_path, env=env)

    for i in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[()])
        print(f"Finished episode {i + 1}")

    env.close()
    print("✅ Done.")

if __name__ == "__main__":
    args = parse_args()
    enjoy(run_id=args.run_id, episodes=args.episodes, record=args.record)