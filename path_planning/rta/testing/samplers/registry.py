from typing import TYPE_CHECKING, Dict, Type, List

if TYPE_CHECKING:
    from .dtg_sampler import DTGSampler

class SamplerRegistry:
    """A standalone manager that maps strings to Sampler classes."""
    _catalog: Dict[str, Type["DTGSampler"]] = {}

    @classmethod
    def register(cls, name: str):
        """A decorator to add a sampler to the catalog."""
        def wrapper(sampler_cls):
            cls._catalog[name] = sampler_cls
            return sampler_cls
        return wrapper

    @classmethod
    def make(cls, name: str, **kwargs) -> "DTGSampler":
        """Factory method to create a sampler instance."""
        if name not in cls._catalog:
            raise KeyError(f"Sampler '{name}' not found in registry.")
        return cls._catalog[name](**kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._catalog.keys())