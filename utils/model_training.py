import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader

from abc import ABC, abstractmethod
from typing import Callable
from utils.noise_scheduler import NoiseScheduler
from utils.training_utils import add_noise

class Trainer(ABC):

    def __init__(
            self,
            model: Module,
            dataloader: DataLoader,
            optimizer: Optimizer,
            criterion: Callable[[Tensor, Tensor], Tensor],
            noise_scheduler: NoiseScheduler
        ):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.noise_scheduler = noise_scheduler

    @abstractmethod
    def prediction_and_target(self, target_sample: Tensor) -> tuple[Tensor, Tensor]:
        """
        The only bit that changes between model types: noise the sample, run the
        model, and return (prediction, target) to feed the loss. Everything else
        (the optimisation loop below) is shared.
        """
        ...

    def run(self, num_epochs: int) -> list:
        losses = []
        for epoch in range(num_epochs):
            self.model.train()

            for target_sample in self.dataloader:
                self.optimizer.zero_grad()
                prediction, target = self.prediction_and_target(target_sample)

                loss = self.criterion(prediction, target)
                loss.backward()
                losses.append(loss.detach().numpy())
                self.optimizer.step()
            if epoch % 100 == 0:
                print(f"Iter {epoch + 1}: Loss={loss}...")
        return losses

class NoisePredictionTrainer(Trainer):
    """
    Predicts the noise that was mixed into the sample (the worked example).
    """

    def prediction_and_target(self, target_sample: Tensor) -> tuple[Tensor, Tensor]:
        t = torch.rand(target_sample.size(0), 1, 1)
        noise = torch.randn_like(target_sample)
        noised_target_sample = add_noise(target_sample, t, noise, self.noise_scheduler)
        return self.model(noised_target_sample, t), noise

## --- Add new model types here, then register them in utils/cli.py ---------------
##
## Each one is the same shape as NoisePredictionTrainer: sample t, draw noise, build
## the noised sample with add_noise, and return (model output, target). Pair it with
## a matching to_velocity() in utils/simulate.py so it can be sampled from.
##
## Velocity / flow-matching -- target is the interpolant velocity x0 - noise:
##
## class VelocityTrainer(Trainer):
##     def prediction_and_target(self, target_sample):
##         t = torch.rand(target_sample.size(0), 1, 1)
##         noise = torch.randn_like(target_sample)
##         noised = add_noise(target_sample, t, noise, self.noise_scheduler)
##         return self.model(noised, t), target_sample - noise
##
## Sample / x0 prediction -- target is the clean data:
##
## class SamplePredictionTrainer(Trainer):
##     def prediction_and_target(self, target_sample):
##         t = torch.rand(target_sample.size(0), 1, 1)
##         noise = torch.randn_like(target_sample)
##         noised = add_noise(target_sample, t, noise, self.noise_scheduler)
##         return self.model(noised, t), target_sample
##
## Score prediction -- trickier: the target is the score of the perturbation kernel,
## roughly -noise / sigma(t), so it needs the noise level sigma(t) which the current
## LinearNoiseScheduler doesn't expose yet. To add it you'd likely:
##   - give NoiseScheduler a sigma(t) (std of the noise at time t),
##   - target -noise / sigma(t) here (or use a sigma-weighted / denoising-score loss
##     rather than plain MSE),
##   - and convert score -> velocity in the sampler accordingly.
