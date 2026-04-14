from abc import ABC, abstractmethod
import pickle
from typing import Any

class BaseSampler(ABC):
    """
    Abstract base class for sampling values (like RTA) 
    to be used across different BlueSky environments.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialise sampler parameters or seeds. 
        """
        pass

    @abstractmethod
    def sample(self, X: Any, *args, **kwargs) -> Any:
        """
        Return a single sampled value.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def fit(self, X: Any, y: Any, *args, **kwargs) -> None:
        """
        Train the sampler based on provided data.
        Must be implemented. If the sampler is static, simply use 'pass'.
        """
        pass

    def save(self, path: str) -> None:
        """
        Save the sampler state to a file using pickle.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """
        Load the sampler state from a file.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
            for key, value in state.items():
                setattr(self, key, value)