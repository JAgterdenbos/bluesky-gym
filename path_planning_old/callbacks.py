from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from bluesky_gym.utils import logger as b_logger

from config import ExperimentConfig

# TODO: add an experiment callback, which saves data for each experiment run

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}/model_checkpoint.zip")
        return True

class SuccessRateLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.runway_stats = {}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                rwy = info.get("current_runway", "unknown")
                win = info.get("is_success", False)
                stats = self.runway_stats.get(rwy, {"wins": 0, "eps": 0})
                stats["eps"] += 1
                if win: 
                    stats["wins"] += 1
                self.runway_stats[rwy] = stats
        return True
    
    def _on_training_end(self):
        print("\nFinal Success Rates by Runway:")
        for rwy, stats in self.runway_stats.items():
            success_rate = stats["wins"] / stats["eps"] if stats["eps"] > 0 else 0.0
            print(f"Runway {rwy}: Success Rate = {success_rate:.2%} ({stats['wins']}/{stats['eps']})")

def get_callbacks(cfg: ExperimentConfig, eval_env):
    csv_log = b_logger.CSVLoggerCallback(cfg.log_dir)
    checkpoint = CheckpointCallback(cfg.save_freq, cfg.save_path)
    success_log = SuccessRateLogger()
    
    eval_cb = EvalCallback(
        eval_env, 
        best_model_save_path=cfg.save_path,
        log_path=cfg.log_dir,
        eval_freq=5000,
        deterministic=True
    )
    
    return CallbackList([csv_log, checkpoint, success_log, eval_cb]), success_log