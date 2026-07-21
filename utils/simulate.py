import torch
from torch.nn import Module
from torch import Tensor
from typing import Callable

## A "to_velocity" turns whatever the model predicts into the velocity dXt/dt of the
## linear interpolant, so a single Euler sampler can drive every model type.
ToVelocity = Callable[[Tensor, Tensor, Tensor], Tensor]

## noise prediction: with x_t = t*x0 + (1-t)*noise the velocity is (x_t - noise)/t.
noise_to_velocity: ToVelocity = lambda out, Xt, t: (Xt - out) / t

## --- velocities for the other model types (uncomment alongside their trainer) -----
## velocity / flow-matching: the model already predicts the velocity.
## velocity_to_velocity: ToVelocity = lambda out, Xt, t: out
##
## sample / x0 prediction: velocity is (x0 - x_t)/(1-t).
## sample_to_velocity: ToVelocity = lambda out, Xt, t: (out - Xt) / (1 - t)
##
## score prediction: convert score -> velocity, e.g. roughly Xt + sigma(t)**2 * score
## style terms -- needs the same sigma(t) you add to the noise scheduler for training.

@torch.no_grad()
def simulate(
        X0: Tensor,
        model: Module,
        to_velocity: ToVelocity,
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
        Xt = Xt + h * to_velocity(model(Xt, t), Xt, t)
        t = t + h
        trajectory.append(Xt.clone())
    return torch.stack(trajectory)

@torch.no_grad()
def simulate_noise_model(X0: Tensor, model: Module, num_steps: int) -> Tensor:
    """Thin wrapper: final sample of a noise-prediction model."""
    return simulate(X0, model, noise_to_velocity, num_steps)[-1]
