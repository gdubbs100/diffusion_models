# diffusion_models
create some basic diffusion models and learn what they are about

## Running

Train a model with `main.py`, e.g.:

```bash
python main.py --model noise --epochs 500 --lr 1e-3 --sim-steps 100
```

This trains on the toy circle dataset and drops a timestamped folder in `runs/`
with the loss plot (`loss.png`), the weights (`model.pt`), and a GIF of the
sampling process (`simulation.gif`). Run `python main.py --help` for all the
flags (dataset size, batch size, learning rate, sim steps, seed, ...).

## Adding a new model type

Every model type shares the same interpolant `x_t = t*x0 + (1-t)*noise`, so it
only differs in two small ways: **what it's trained to predict**, and **how that
prediction is turned into a velocity for sampling**. Adding one is three steps —
skeletons for each live as comments next to the worked `noise` example.

**1. A trainer** in `utils/model_training.py` — subclass `Trainer` and implement
`prediction_and_target`, which noises the sample, runs the model, and returns
`(prediction, target)` for the loss:

```python
class VelocityTrainer(Trainer):
    def prediction_and_target(self, target_sample):
        t = torch.rand(target_sample.size(0), 1, 1)
        noise = torch.randn_like(target_sample)
        noised = add_noise(target_sample, t, noise, self.noise_scheduler)
        return self.model(noised, t), target_sample - noise  # target = velocity
```

**2. A `to_velocity`** in `utils/simulate.py` — maps the model's output to the
velocity `dXt/dt` that the sampler integrates:

```python
velocity_to_velocity: ToVelocity = lambda out, Xt, t: out  # already a velocity
```

**3. Register it** in `utils/cli.py` by adding one line to `MODEL_REGISTRY`:

```python
"velocity": ModelSpec(VelocityTrainer, velocity_to_velocity),
```

That's it — `--model velocity` now works and shows up in `--help`.

> **Score prediction** is the one that needs a bit more: the target is roughly
> `-noise / sigma(t)`, so you'll first want to give `NoiseScheduler` a `sigma(t)`
> (the noise std at time `t`), possibly swap the plain MSE for a sigma-weighted
> loss, and convert score → velocity in its `to_velocity`. See the notes in
> `utils/model_training.py` and `utils/simulate.py`.
