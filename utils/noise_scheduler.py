from abc import ABC, abstractmethod
import torch
from torch import Tensor

class NoiseScheduler(ABC):

    @classmethod
    @abstractmethod
    def from_args(cls, args) -> "NoiseScheduler":
        """Build the scheduler, pulling whatever it needs out of the CLI args."""
        ...

    @abstractmethod
    def __call__(
            self,
            t: Tensor
        ) -> Tensor:
        ...

class LinearNoiseScheduler(NoiseScheduler):

    @classmethod
    def from_args(cls, args) -> "LinearNoiseScheduler":
        return cls()

    def __call__(
            self,
            t: Tensor
        ) -> Tensor:
        return t, 1 - t


class DDPMNoiseScheduler(NoiseScheduler):

    @classmethod
    def from_args(cls, args) -> "DDPMNoiseScheduler":
        return cls(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)

    def __init__(self, beta_min: float, beta_max: float, T: int):
        self.beta = torch.arange(start = beta_min, end = beta_max, step = (beta_max - beta_min)/T)
        self.alpha_bar = (1 - self.beta).cumprod(dim = 0)
        self.T = T 
        self.beta_min, self.beta_max = beta_min, beta_max

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
    
    def calc_simulation_weights(self, t: Tensor) -> Tensor:
        idx = self.t_to_index(t)
        drift_weight = (1 / torch.sqrt(1 - self.beta[idx])).reshape_as(t)
        noise_model_weight = (self.beta[idx] / (torch.sqrt(1 - self.alpha_bar[idx]))).reshape_as(t)
        return drift_weight, noise_model_weight
        
    




