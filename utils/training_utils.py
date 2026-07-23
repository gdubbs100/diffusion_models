import torch
import torch.distributions as dist

from torch import Tensor
from torch.utils.data import Dataset

from utils.noise_scheduler import NoiseScheduler

def add_noise(
        x: Tensor, 
        t: Tensor, 
        noise: Tensor,
        noise_schedule: NoiseScheduler
    ) -> Tensor:
    alpha, beta = noise_schedule(t)
    x = alpha * x + beta * noise
    return x

def generate_circle_sample(dims: tuple) -> Tensor:
    p_target = dist.Uniform(low = torch.zeros((1,)), high = 2.0*torch.pi)
    target_samples = p_target.sample(dims)
    x = torch.cos(target_samples)
    y = torch.sin(target_samples)
    return torch.cat([x, y], dim=-1)

class DiffusionDataset(Dataset):
    def __init__(self, target_samples: Tensor):
        self.target_samples = target_samples
        
    def __len__(self):
        return self.target_samples.size(0)
    
    def __getitem__(self, idx):
        target_sample = self.target_samples[idx, ...]
        return target_sample