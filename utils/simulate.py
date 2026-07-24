import torch
from torch.nn import Module
from torch import Tensor
from typing import Callable
from utils.noise_scheduler import NoiseScheduler, DDPMNoiseScheduler

from abc import ABC, abstractmethod

## A "to_velocity" turns whatever the model predicts into the velocity dXt/dt of the
## linear interpolant, so a single Euler sampler can drive every model type.


## noise prediction: with x_t = t*x0 + (1-t)*noise the velocity is (x_t - noise)/t.
noise_to_velocity = lambda out, Xt, t: (Xt - out) / t

## --- velocities for the other model types (uncomment alongside their trainer) -----
## velocity / flow-matching: the model already predicts the velocity.
## velocity_to_velocity: ToVelocity = lambda out, Xt, t: out
##
## sample / x0 prediction: velocity is (x0 - x_t)/(1-t).
## sample_to_velocity: ToVelocity = lambda out, Xt, t: (out - Xt) / (1 - t)
##
## score prediction: convert score -> velocity, e.g. roughly Xt + sigma(t)**2 * score
## style terms -- needs the same sigma(t) you add to the noise scheduler for training.
class Simulator(ABC):

    @classmethod
    @abstractmethod
    def from_args(cls, args, noise_scheduler: NoiseScheduler) -> "Simulator":
        """Build the simulator from the CLI args and the scheduler used for training."""
        ...

    @abstractmethod
    def simulate(self):
        ...


class NoiseModelSimulator(Simulator):
    """
    Basic class designed to predict noise.
    Requires a specified velocity function
    default is: (x_t - noise)/t.
    """

    def __init__(
            self,
            velocity_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = noise_to_velocity
        ):
        self.velocity_fn = velocity_fn

    @classmethod
    def from_args(cls, args, noise_scheduler: NoiseScheduler) -> "NoiseModelSimulator":
        return cls()

    @torch.no_grad()
    def simulate(
            self, 
            X0: Tensor,
            model: Module,
            num_steps: int,
            t_start: float = 0.05,
    ) -> Tensor:
        """
        Euler-integrate the samples from t_start up to t=1 and return the whole
        trajectory, shape (num_steps + 1, N, dims), so it can be animated.
        """
        t = torch.tensor([t_start]).expand(X0.size(0), X0.size(1), 1)
        h = torch.tensor([(1.0 - t_start) / num_steps])
        Xt = X0.clone()
        trajectory = [Xt.clone()]
        for _ in range(num_steps):
            Xt = Xt + h * self.velocity_fn(model(Xt, t), Xt, t)
            t = t + h
            trajectory.append(Xt.clone())
        return torch.stack(trajectory)

class DDPMSimulator(Simulator):

    def __init__(
            self, 
            diffusion_coefficient: float, 
            noise_schedule: DDPMNoiseScheduler
        ):

        self.diffusion_coefficient = diffusion_coefficient
        self.noise_schedule = noise_schedule

    @classmethod
    def from_args(cls, args, noise_scheduler: NoiseScheduler) -> "DDPMSimulator":
        return cls(diffusion_coefficient=args.diffusion_coef, noise_schedule=noise_scheduler)

    @torch.no_grad
    def simulate(
            self,
            X0: Tensor,
            model: Module,
            num_steps: int, # TODO: I think this should match noise_schedule.T
        ) -> Tensor:
        Xt = X0.clone()
        t = torch.tensor([self.noise_schedule.beta_min]).expand(X0.size(0), X0.size(1), 1)
        h = (self.noise_schedule.beta_max - self.noise_schedule.beta_min) / self.noise_schedule.T
        trajectory = [Xt.clone()]

        for _ in range(num_steps):

            z = torch.randn_like(Xt)
            drift_weight, noise_model_weight = self.noise_schedule.calc_simulation_weights(t)
            Xt = drift_weight * (Xt - noise_model_weight * model(Xt, t)) + z * self.diffusion_coefficient
            t = t + h
            trajectory.append(Xt.clone())
        return torch.stack(trajectory)
