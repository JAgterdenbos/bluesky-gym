import numpy as np
import torch

from typing import Dict

def heading_encoder(heading: float) -> np.ndarray:
    """Maps heading (rad) to (sin, cos) action."""
    return np.array([np.sin(heading), np.cos(heading)])

def heading_obs_adapter(obs: Dict[str, torch.Tensor], val: float) -> Dict[str, torch.Tensor]:
    """Syncs the cos/sin components in the observation with the sweep value."""
    # Assumes Hdg env: (x, y, t, cos, sin)
    if obs["observation"].shape[-1] == 5:
        new_obs = {k: v.clone() for k, v in obs.items()}
        new_obs["observation"][:, 3] = np.cos(val)
        new_obs["observation"][:, 4] = np.sin(val)
        return new_obs
    return obs