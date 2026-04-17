from abc import ABC, abstractmethod
import pickle
from typing import Any, Type, TypeVar

T = TypeVar("T", bound="BaseSampler")

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
            pickle.dump(self, f)

    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """
        Load the sampler object from a file and return it.
        """
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            
            # This check passes if obj is the specific class 
            # OR any subclass of the class 'load' was called on.
            if not isinstance(obj, cls):
                raise TypeError(
                    f"Loaded object of type {type(obj).__name__} "
                    f"is not a subclass of {cls.__name__}"
                )
                
            return obj