from bluesky_gym.envs.common.base_sampler import BaseSampler
from .registry import SamplerRegistry

from typing import Any, Optional

class RTASampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, name: Optional[str] = None, *args, **kwargs):
        """Register the sampler class in the registry."""
        super().__init_subclass__(*args, **kwargs)

        cls_name = name or cls.__name__
        SamplerRegistry.register(cls_name)(cls)

    def plot_distribution(self, *args, **kwargs):
        pass
