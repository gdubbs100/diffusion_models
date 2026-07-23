from abc import ABC, abstractmethod
import torch
from torch import Tensor

class NoiseScheduler(ABC):

    @abstractmethod
    def __call__(
            self, 
            t: Tensor
        ) -> Tensor:
        ...

class LinearNoiseScheduler(NoiseScheduler):

    def __call__(
            self, 
            t: Tensor
        ) -> Tensor:
        return t, 1 - t


class DDPMNoiseScheduler(NoiseScheduler):

    def __init__(self, beta: Tensor):
        self.beta = beta
        self.alpha_bar = (1 - self.beta).cumprod(dim = 0)
        self.T = beta.size(0) # indexed 0..T-1

    def __call__(
            self,
            t: Tensor
        ) -> Tensor:
        # convert 0-1 representation of t in trainer to int
        idx = self.t_to_index(t) 
        alpha_bar_t = self.alpha_bar[idx].reshape_as(t)
        return torch.sqrt(alpha_bar_t), torch.sqrt(1 - alpha_bar_t)
    
    def t_to_index(self, t: Tensor) -> Tensor:
        """
        Converts t to index and maintains t=0 being the data dist
        """
        return ((1 - t).flatten() * self.T).int()

    def index_to_t(self, idx: Tensor) -> Tensor:
        """
        Inverse of t_to_index
        """
        return 1 - idx/self.T
    




