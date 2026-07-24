import argparse
from dataclasses import dataclass

from utils.model_training import Trainer, NoisePredictionTrainer
from utils.simulate import Simulator, NoiseModelSimulator, DDPMSimulator
from utils.noise_scheduler import NoiseScheduler, LinearNoiseScheduler, DDPMNoiseScheduler

@dataclass
class ModelSpec:
    """Everything that makes a model type its own thing: how it's trained, and how
    its predictions are simulated."""
    trainer: type[Trainer]
    simulator: type[Simulator]
    noise_scheduler: type[NoiseScheduler]

## The single place model types live. Add a new one here (after writing its Trainer
## in utils/model_training.py and its to_velocity in utils/simulate.py) and it shows
## up as a --model choice automatically.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "noise": ModelSpec(NoisePredictionTrainer, NoiseModelSimulator, LinearNoiseScheduler),
    "ddpm": ModelSpec(NoisePredictionTrainer , DDPMSimulator, DDPMNoiseScheduler)
    ## "velocity": ModelSpec(VelocityTrainer, velocity_to_velocity),
    ## "sample":   ModelSpec(SamplePredictionTrainer, sample_to_velocity),
    ## "score":    ModelSpec(ScoreTrainer, score_to_velocity),  # see notes in those two files
}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a diffusion model on the toy circle dataset.")

    p.add_argument("--model", choices=list(MODEL_REGISTRY), default="noise",
                   help="which diffusion parameterisation to train")
    
    ## DDPM noise schedule args
    p.add_argument("--beta_min", type=float, default=1.0e-4, help="min value of DDPM noise schedule")
    p.add_argument("--beta_max", type=float, default=0.02, help="max value of DDPM noise schedule")
    p.add_argument("--T", type=int, default = 100, help="length of DDPM noise schedule")

    ## DDPM simulation args
    p.add_argument("--diffusion_coef", type=float, default=1.0, help="diffusion coefficient for DDPM simulation")

    ## data
    p.add_argument("--num-samples", type=int, default=256 * 4, help="number of point clouds")
    p.add_argument("--points-per-sample", type=int, default=100, help="points in each cloud")
    p.add_argument("--batch-size", type=int, default=256)

    ## optimisation
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)

    ## sampling / video
    p.add_argument("--sim-steps", type=int, default=100, help="Euler steps when sampling")
    p.add_argument("--sim-particles", type=int, default=400, help="points in the simulated cloud")

    ## logging
    p.add_argument("--out-dir", default="runs", help="where run folders are written")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args(argv)
