import gymnasium as gym
import bluesky_gym
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from config import ExperimentConfig
from callbacks import get_callbacks


#TODO: Make this more generic and find a way to share code between different experiments. Maybe we can have a base Experiment class that handles the common logic and then specific ones for each type of experiment (e.g., PathPlanningExperiment, etc.) that inherit from it and implement specific methods if needed. For now, this is simple and works well for our current needs. 

#TODO: Find a way to make learn log frequency be dynamic based on the number of timesteps and not too many logs. Maybe we can have it log every 1% of the total timesteps or something like that.

DO_EVALUATE = True  # Set to False if you want to skip evaluation after training. Evaluation can be time-consuming, so this allows for quicker iterations during development.

def make_env(env_id, runways=None, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode, runways=runways, action_mode="hdg")
    return Monitor(env)

def compute_log_interval(total_timesteps):
    # Log every 1% of total timesteps, but at least every 1000 steps
    return max(1000, total_timesteps // 100)

def train(cfg: ExperimentConfig, verbose: int = 1):
    # Setup Envs
    env = make_env(cfg.session.env_name, runways=cfg.session.train_runways)
    eval_env = make_env(cfg.session.env_name, runways=cfg.session.eval_runways)

    # Build Model
    model = cfg.model.algorithm(
        "MultiInputPolicy", 
        env, 
        learning_rate=cfg.model.learning_rate,
        policy_kwargs=cfg.model.policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=cfg.model.her_n_sampled_goal,
            goal_selection_strategy=cfg.model.her_goal_selection_strategy
        ),
        verbose=verbose
    )

    log_interval = compute_log_interval(cfg.session.total_timesteps)

    callbacks, _ = get_callbacks(cfg, eval_env)
    model.learn(total_timesteps=cfg.session.total_timesteps, callback=callbacks, log_interval=log_interval)
    model.save(f"{cfg.save_path}/final_model")
    print(f"✅ Training Complete. Model saved to {cfg.save_path}")

    env.close()
    eval_env.close()

def evaluate(cfg: ExperimentConfig):
    print(f"\nLoading model from {cfg.save_path}/final_model.zip for evaluation ...")
    eval_env = make_env(cfg.session.env_name, runways=cfg.session.eval_runways, render_mode="human")
    model = cfg.model.algorithm.load(f"{cfg.save_path}/final_model", env=eval_env)
    
    results = {rwy: [] for rwy in cfg.session.eval_runways} if cfg.session.eval_runways is not None else {}
    
    for i in range(cfg.session.eval_episodes):
        done = truncated = False
        obs, info = eval_env.reset()
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
        
        rwy = info.get("current_runway", "unknown")
        win = info.get("is_success", False)
        results[rwy].append(win)
    
    # Print success rates
    print("\nEvaluation Results:")
    for rwy, outcomes in results.items():
        success_rate = sum(outcomes) / len(outcomes) if outcomes else 0.0
        print(f"Runway {rwy}: Success Rate = {success_rate:.2%} ({sum(outcomes)}/{len(outcomes)})")
    
    eval_env.close()

def main():
    bluesky_gym.register_envs()
    
    # Instantiate the master config
    cfg = ExperimentConfig()
    print(f"▶️ Starting Experiment: {cfg.run_id}")

    # Train the model
    if cfg.session.do_train:
        print("\nTraining Model ...")
        train(cfg)

    # Evaluate the model
    if DO_EVALUATE:
        evaluate(cfg)

if __name__ == "__main__":
    main()