import csv
from pathlib import Path
from abc import ABC, abstractmethod
import argparse
from bluesky_gym.experiment.config import ExperimentConfig

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

    def finalise_episode(self, success: bool):
        if success:
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

def get_collector(output_path: str, chunk_size: int, fresh_start: bool = True):
    """Factory function to return the correct collector based on file extension."""
    if output_path.endswith(".parquet"):
        return ParquetDataCollector(output_path, chunk_size, fresh_start)
    elif output_path.endswith(".csv"):
        return CSVDataCollector(output_path, chunk_size, fresh_start)
    else:
        raise ValueError(f"Unsupported file format for: {output_path}. Use .csv or .parquet")

def _get_args():
    p = argparse.ArgumentParser(description="Collect rta data step-by-step per successful episode.")
    p.add_argument("--run-id", type=str, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", type=str, default="rta_data.csv")
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--verbose_frequency", type=int, default=100)
    return p.parse_args()

def run_collection(experiment_cls):
    from os.path import join
    args = _get_args()

    # 1. Load Config & Model
    cfg = ExperimentConfig.load(
        args.run_id, 
        model_config_cls=experiment_cls.model_config_cls,
        env_config_cls=experiment_cls.env_config_cls
    )
    
    exp = experiment_cls(cfg)
    env = exp.make_env(cfg.eval_env_kwargs)

    model_path = join(cfg.save_path, "final_model.zip")
    model = cfg.model.get_algorithm().load(model_path, env=env)
    
    # 2. Setup Collector
    collector = get_collector(args.out, args.chunk, fresh_start=True)
    
    # 3. Main Loop
    success_count, max_episodes = 0, args.episodes
    success_key = cfg.env.success_key
    
    print(f"🏃 Collecting {max_episodes} successful episodes...")
    while success_count < max_episodes:
        obs, info = env.reset()
        done = truncated = False
        step = 0
        
        while not (done or truncated):
            action, info = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
            # Simple conversion of numpy arrays to lists for serialization
            act_val = action.tolist() if hasattr(action, 'tolist') else action
            
            #TODO: rewrate so we collect the necessary info for the rta!
            collector.collect_step(
                episode = success_count + 1,
                step = step,
                reward = float(reward),
                action = str(act_val)
            )
            step += 1
        
        is_success = info.get(success_key, False)
        collector.finalise_episode(success=is_success)
        
        if is_success:
            success_count += 1

        if success_count % args.verbose_frequency == 0:
            print(f"\rProgress: [{success_count}/{max_episodes}] episodes", end="")
    
    collector.close()
    env.close()