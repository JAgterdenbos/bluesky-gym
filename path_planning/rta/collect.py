import csv
from pathlib import Path
from abc import ABC, abstractmethod
from bluesky_gym.experiment.config import ExperimentConfig

from typing import Optional

class BaseDataCollector(ABC):
    """Abstract base class for streaming data collectors."""
    
    def __init__(self, output_path: str, chunk_size: int = 10, fresh_start: bool = True):
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.buffer = []
        self.current_episode_steps = []
        self.successful_count = 0
        
        # Ensure parent directories exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if fresh_start and self.output_path.exists():
            self.output_path.unlink()

    def collect_step(self, **data_dict):
        """Accepts arbitrary keyword arguments to easily collect dynamic data."""
        self.current_episode_steps.append(data_dict)

    def finalise_episode(self, success: bool, backfill: Optional[dict] = None):
        if success:
            if backfill is not None:
                for step in self.current_episode_steps:
                    step.update(backfill)
            self.buffer.extend(self.current_episode_steps)
            self.successful_count += 1
        
        self.current_episode_steps = []

        if self.successful_count >= self.chunk_size:
            self._flush()

    @abstractmethod
    def _flush(self):
        """Must be implemented by subclasses to handle writing to disk."""
        pass

    def _clear_buffer(self):
        self.buffer.clear()
        self.successful_count = 0

    def close(self):
        """Flush remaining data and close any open resources."""
        if self.buffer:
            self._flush()
            
        self._on_close() # Hook for subclasses
        print(f"\n✅ Collection complete. File: {self.output_path}")

    def _on_close(self):
        """Optional hook for subclasses to handle closing file handles (e.g., ParquetWriter)."""
        pass


class CSVDataCollector(BaseDataCollector):
    """Dedicated collector for writing to CSV."""
    
    def _flush(self):
        if not self.buffer: 
            return
        
        file_exists = self.output_path.exists()
        keys = self.buffer[0].keys()

        # Handle complex types for CSV (stringify lists/dicts)
        csv_buffer = []
        for row in self.buffer:
            csv_buffer.append({k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})

        with open(self.output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not file_exists:
                writer.writeheader()
            writer.writerows(csv_buffer)
        
        self._clear_buffer()


class ParquetDataCollector(BaseDataCollector):
    """Dedicated collector for writing to Parquet format."""
    
    def __init__(self, output_path: str, chunk_size: int = 10, fresh_start: bool = True):
        # Initialize the base class first
        super().__init__(output_path, chunk_size, fresh_start)
        
        self._writer = None
        
        # Fail fast: check for required library
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            self.pa = pa
            self.pq = pq
        except ImportError:
            raise ImportError("To save as Parquet, you must install pyarrow: 'pip install pyarrow'")

    def _flush(self):
        if not self.buffer: 
            return
        
        # PyArrow natively handles lists, no stringification needed
        table = self.pa.Table.from_pylist(self.buffer)
        
        if self._writer is None:
            self._writer = self.pq.ParquetWriter(self.output_path, table.schema)
        
        self._writer.write_table(table)
        self._clear_buffer()

    def _on_close(self):
        """Close the parquet writer safely."""
        if self._writer is not None:
            self._writer.close()

class VerboseDataCollector(BaseDataCollector):
    """
    Stores all episodes regardless of success, 
    adding an 'is_success' column to the data.
    """
    
    def finalise_episode(self, success: bool, backfill: Optional[dict] = None):
        # 1. Prepare the update dictionary
        update_data = backfill.copy() if backfill is not None else {}
        update_data["is_success"] = success 

        # 2. Apply updates (including success status) to every step
        for step in self.current_episode_steps:
            step.update(update_data)
        
        # 3. Always extend the buffer (ignoring the success gate)
        self.buffer.extend(self.current_episode_steps)
        
        # 4. Increment success count only for chunking/flushing logic
        self.successful_count += 1
        
        self.current_episode_steps = []

        if self.successful_count >= self.chunk_size:
            self._flush()

class VerboseCSVCollector(VerboseDataCollector, CSVDataCollector):
    """Combines verbose logic with CSV writing."""
    pass

class VerboseParquetCollector(VerboseDataCollector, ParquetDataCollector):
    """Combines verbose logic with Parquet writing."""
    pass

def get_collector(output_path: str, chunk_size: int, fresh_start: bool = True, is_verbose: bool = False):
    """
    Factory function to return the correct collector.
    
    Args:
        output_path: Path to the save file.
        chunk_size: How many successful episodes to buffer before flushing.
        fresh_start: If True, deletes existing file at output_path.
        is_verbose: If True, stores all episodes (including failures).
    """
    ext = Path(output_path).suffix.lower()
    
    # Mapping table for easy extension
    mapping = {
        ".csv": {
            True: VerboseCSVCollector,
            False: CSVDataCollector
        },
        ".parquet": {
            True: VerboseParquetCollector,
            False: ParquetDataCollector
        }
    }

    if ext not in mapping:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .parquet")

    # Select the class based on extension and verbosity
    collector_cls = mapping[ext][is_verbose]
    
    return collector_cls(output_path, chunk_size, fresh_start)

def _get_args():
    import argparse

    p = argparse.ArgumentParser(description="Collect rta data step-by-step per successful episode.")
    p.add_argument("run_id", type=str, help="The ID of the run to collect data from.")
    p.add_argument("--episodes", type=int, default=100, help="Number of successful episodes to collect.")
    p.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions.")
    p.add_argument("--no-fresh-start", action="store_true", default=False, help="Append to existing data.")
    p.add_argument("--out", type=str, default="rta_data.csv", help="Output file path.")
    p.add_argument("--chunk", type=int, default=25, help="Number of episodes to collect per file flush.")
    p.add_argument("--verbose_frequency", type=int, default=100, help="Print progress every N episodes.")
    p.add_argument("--verbose-store", action="store_true", default=False, help="Store all episodes, including failures.")
    p.add_argument("--runways", type=str, nargs="*", default=None, help="Optional list of specific runways to collect (e.g. --runways 18R 36L)")
    return p.parse_args()

def _path_length_km(info: dict) -> float:
    """Helper function to calculate path length in kilometers from episode info."""
    plw = info.get("path_length_weight", 0.0)
    path_rew = info.get("average_path_rew", 0.0)
    
    if plw == 0:
        return 0.0
        
    return float((path_rew / plw) * 1.852) # NM -> km

def collect(env, model, collector, max_episodes, success_key, stochastic=False, verbose_frequency=100):
    success_count = 0
    total_attempts = 0  # Unique ID for every episode attempt

    is_spatial = not env.unwrapped.use_rta #Note: this assumes that env is wrapped by a gym Monitor
    rta_key = "observation" if is_spatial else "desired_goal"
    
    print(f"🏃 Collecting {max_episodes} successful episodes... (mode: {"spatial" if is_spatial else "spatial-temporal"})")
    print(f"Progress: [{success_count}/{max_episodes}] successful episodes (Total attempts: 0)", end="", flush=True)
    
    while success_count < max_episodes:
        obs, info = env.reset()
        total_attempts += 1 
        done = truncated = False
        step = 0

        while not (done or truncated):
            # We use total_attempts for the 'episode' column
            collector.collect_step(
                episode  = total_attempts,
                step     = step,
                x        = float(obs["observation"][0]),
                y        = float(obs["observation"][1]),
                t        = float(obs["observation"][2]),
                runway   = info["current_runway"],
                path_len = _path_length_km(info),
                heading  = info.get("heading", 0)
            )
            
            action, _ = model.predict(obs, deterministic=not stochastic)
            obs, reward, done, truncated, info = env.step(action)
            step += 1
        
        t = float(obs["observation"][2])

        # Record the final state of the episode
        collector.collect_step(
            episode  = total_attempts,
            step     = step,
            x        = float(obs["observation"][0]),
            y        = float(obs["observation"][1]),
            t        = t,
            runway   = info["current_runway"],
            path_len = _path_length_km(info),
            heading  = info.get("heading", 0)
        )

        is_success = info.get(success_key, False)
        rta = float(obs[rta_key][2])
        total_dist_km = _path_length_km(info)

        backfill = {"rta": rta, "total_dist_km": total_dist_km}

        if not is_spatial:
            backfill["delay"] = rta - t

        collector.finalise_episode(
            success=is_success,
            backfill = backfill,
        )

        if is_success:
            success_count += 1

        if success_count % verbose_frequency == 0:
            print(f"\rProgress: [{success_count}/{max_episodes}] successful episodes (Total attempts: {total_attempts})", end="")
    
    print(f"\n✅ Done. Total episodes run: {total_attempts}")

def create_env_and_model(experiment_cls, run_id, runways = None, model_name="final_model.zip"):
    from os.path import join

    cfg = ExperimentConfig.load(
        run_id, 
        model_config_cls=experiment_cls.model_config_cls,
        env_config_cls=experiment_cls.env_config_cls
    )

    if runways is not None:
        cfg.env.env_kwargs.runways = runways

    exp = experiment_cls(cfg)
    env = exp.make_env(cfg.eval_env_kwargs)

    model_path = join(cfg.save_path, model_name)
    model = cfg.model.get_algorithm().load(model_path, env=env)

    return env, model, cfg.env.success_key

def run_collection_cli(experiment_cls):
    """Entry point for CLI-based data collection."""
    import bluesky_gym
    bluesky_gym.register_envs()

    args = _get_args() # Retrieve CLI arguments

    env, model, succes_key = create_env_and_model(
        experiment_cls=experiment_cls,
        run_id=args.run_id,
        runways=args.runways,
        model_name="final_model.zip"
    )
    
    # 2. Setup Collector
    collector = get_collector(
            args.out, 
            args.chunk, 
            fresh_start=not args.no_fresh_start,
            is_verbose=args.verbose_store
        )    
    
    # 3. Execute Collection with Resource Safety
    try:
        collect(
            env=env,
            model=model,
            collector=collector,
            max_episodes=args.episodes,
            success_key=succes_key,
            stochastic=args.stochastic,
            verbose_frequency=args.verbose_frequency
        )
    finally:
        # Ensure files are flushed and environments are closed
        collector.close()
        env.close()