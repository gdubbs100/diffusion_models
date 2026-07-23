import os
## this machine ships two OpenMP runtimes; let them coexist so torch can import
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.cli import parse_args, MODEL_REGISTRY
from utils.networks import Net
from utils.noise_scheduler import LinearNoiseScheduler
from utils.training_utils import DiffusionDataset, generate_circle_sample
from utils.simulate import simulate
from utils.viz import save_loss_plot, save_simulation_gif

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    spec = MODEL_REGISTRY[args.model]

    ## data: a bunch of point clouds sampled from the unit circle
    target_samples = generate_circle_sample(dims=(args.num_samples, args.points_per_sample))
    dataloader = DataLoader(DiffusionDataset(target_samples), batch_size=args.batch_size)

    ## model + training pieces
    model = Net(input_dim=2, output_dim=2)
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    noise_schedule = LinearNoiseScheduler()

    trainer = spec.trainer(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        noise_scheduler=noise_schedule,
    )

    print(f"Training '{args.model}' model for {args.epochs} epochs...")
    losses = trainer.run(args.epochs)

    ## everything for this run lands in its own timestamped folder
    run_dir = Path(args.out_dir) / f"{args.model}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_loss_plot(losses, str(run_dir / "loss.png"))
    torch.save(model.state_dict(), run_dir / "model.pt")

    ## sample one cloud from a Gaussian and watch it flow into the circle
    model.eval()
    X0 = dist.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample((1, args.sim_particles))
    trajectory = simulate(X0, model, spec.to_velocity, num_steps=args.sim_steps)
    save_simulation_gif(trajectory.squeeze(1), str(run_dir / "simulation.gif"))  # drop the single-cloud batch dim

    print(f"Done. Wrote loss.png, model.pt and simulation.gif to {run_dir}")

if __name__ == "__main__":
    main()
