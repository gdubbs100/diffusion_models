from abc import ABC, abstractmethod
import torch

class NoiseScheduler(ABC):

    @abstractmethod
    def __call__(
            self, 
            t: torch.Tensor
        ) -> torch.Tensor:
        ...

class LinearNoiseScheduler(NoiseScheduler):

    def __call__(
            self, 
            t: torch.Tensor
        ) -> torch.Tensor:
        return t, 1 - t
    
## TODO: GaussianNoiseScheduler
