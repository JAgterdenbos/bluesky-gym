import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Type
from stable_baselines3 import SAC


# Configuration classes for the Path Planning experiment
# TODO: Refactor maybe or think about how to make this more flexible or more generic for other experiments. Maybe we can have a base ExperimentConfig and then specific ones for each type of experiment (e.g., PathPlanningConfig, etc.) that inherit from it and add specific fields if needed. For now, this is simple and works well for our current needs. Maybe not everything should be a Config dataclass. 

@dataclass
class ModelConfig:
    algorithm: Type = SAC
    net_arch: Dict[str, List[int]] = field(default_factory=lambda: dict(pi=[256, 256, 256], qf=[256, 256, 256]))
    learning_rate: float = 3e-4
    her_n_sampled_goal: int = 4
    her_goal_selection_strategy: str = "future"

    @property
    def policy_kwargs(self) -> Dict[str, Any]:
        return dict(net_arch=self.net_arch)

@dataclass
class SessionConfig:
    env_name: str = "PathPlanningGoalEnv-v0"
    total_timesteps: int = 250_000
    train_runways: Optional[List[str]] = field(default_factory=lambda: None)  # None means all runways
    eval_runways: Optional[List[str]] = field(default_factory=lambda: ["27", "18R"])
    eval_episodes: int = 10
    do_train: bool = True

@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    
    # Run ID is generated once per experiment
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self):
        # Construct paths based on the model and session info
        base_path = f"./experiments/{self.session.env_name}/{self.model.algorithm.__name__}"
        self.log_dir = f"{base_path}/logs/{self.run_id}/"
        self.save_path = f"{base_path}/models/{self.run_id}/"
        self.save_freq = max(5_000, self.session.total_timesteps // 100)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.save_path, exist_ok=True)