"""
callbacks.py
------------
SB3 callbacks for experiment monitoring.

Classes
-------
  CheckpointCallback   - periodic model save
  SuccessRateLogger    - per-runway win/loss tracking printed at end of training

Factories
---------
  get_callbacks()      - assembles the full CallbackList for a given config
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from bluesky_gym.utils import logger as b_logger

from config import ExperimentConfig


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class CheckpointCallback(BaseCallback):
    """Saves a checkpoint zip at a fixed step interval.

    The file is always written to the same name (checkpoint_model.zip)
    so only the *latest* checkpoint is kept on disk, avoiding unbounded
    disk usage during long runs.  If you need full history, increment the
    name here.
    """

    def __init__(self, save_freq: int, save_path: str) -> None:
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}/checkpoint_model")
        return True


# ---------------------------------------------------------------------------
# Per-runway success tracking
# ---------------------------------------------------------------------------

class SuccessRateLogger(BaseCallback):
    """Tracks per-runway success rates across all training episodes.

    Reads `current_runway` and `is_success` from the step info dict.
    Both keys are optional — missing values are handled gracefully
    (runway defaults to 'unknown', success defaults to False).

    The summary is printed at the end of training via _on_training_end().
    """

    def __init__(self) -> None:
        super().__init__()
        # { runway_id: {"wins": int, "eps": int} }
        self.runway_stats: dict[str, dict[str, int]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # Only record at episode boundaries
            if "episode" not in info:
                continue
            rwy  = info.get("current_runway", "unknown")
            win  = bool(info.get("is_success", False))
            stat = self.runway_stats.setdefault(rwy, {"wins": 0, "eps": 0})
            stat["eps"] += 1
            if win:
                stat["wins"] += 1
        return True

    def _on_training_end(self) -> None:
        if not self.runway_stats:
            return
        print("\n📈 Final Training Success Rates by Runway:")
        for rwy, stat in sorted(self.runway_stats.items()):
            eps  = stat["eps"]
            wins = stat["wins"]
            rate = wins / eps if eps > 0 else 0.0
            print(f"  Runway {rwy}: {wins}/{eps}  ({rate:.1%})")

    @property
    def overall_success_rate(self) -> float:
        """Convenience accessor — total wins / total episodes."""
        total_eps  = sum(s["eps"]  for s in self.runway_stats.values())
        total_wins = sum(s["wins"] for s in self.runway_stats.values())
        return total_wins / total_eps if total_eps > 0 else 0.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_callbacks(cfg: ExperimentConfig, eval_env) -> tuple[CallbackList, SuccessRateLogger]:
    """Assemble and return the full callback list for a training run.

    Returns
    -------
    callbacks : CallbackList
        Ready to pass to model.learn().
    success_logger : SuccessRateLogger
        Returned separately so callers can inspect stats after training.
    """
    csv_logger     = b_logger.CSVLoggerCallback(cfg.log_dir)
    checkpoint     = CheckpointCallback(cfg.save_freq, cfg.save_path)
    success_logger = SuccessRateLogger()

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg.save_path,
        log_path=cfg.log_dir,
        eval_freq=cfg.session.eval_freq,
        deterministic=True,
        verbose=0,   # suppresses per-eval spam; summary still logged to CSV
    )

    return CallbackList([csv_logger, checkpoint, success_logger, eval_cb]), success_logger