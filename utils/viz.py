import numpy as np
import matplotlib
matplotlib.use("Agg")  # write files, no display window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torch import Tensor

## a small dark palette so the plot and the video read as one thing
BG = "#0f1116"
INK = "#c9d1d9"
MUTED = "#6e7681"
ACCENT = "#4c9be8"

def save_loss_plot(losses: list, path: str) -> None:
    losses = np.asarray(losses)
    steps = np.arange(len(losses))

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=BG)
    ax.set_facecolor(BG)

    ## raw loss sits in the background, a smoothed line carries the trend
    ax.plot(steps, losses, color=ACCENT, alpha=0.25, linewidth=1)
    window = max(1, len(losses) // 100)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, color=ACCENT, linewidth=2, label=f"smoothed (w={window})")
        ax.legend(facecolor=BG, edgecolor=MUTED, labelcolor=INK, loc="upper right")

    ax.set_yscale("log")
    ax.set_xlabel("training step (batch)", color=INK)
    ax.set_ylabel("MSE loss", color=INK)
    ax.set_title("Training loss", color=INK)
    _style_axes(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)

def save_simulation_gif(trajectory: Tensor, path: str, t_start: float = 0.05, fps: int = 20, hold: float = 1.5) -> None:
    """
    trajectory: (num_frames, N, 2) tensor from utils.simulate.simulate.
    Points flow from a Gaussian blob into the target circle; each particle is
    coloured by the angle it lands at so the ring structure emerges as it forms.
    """
    traj = trajectory.detach().cpu().numpy()
    num_frames = traj.shape[0]

    ## colour every particle by its final angle, through a cyclic colormap
    final = traj[-1]
    angles = np.arctan2(final[:, 1], final[:, 0])
    colors = plt.cm.twilight((angles + np.pi) / (2 * np.pi))

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    _style_axes(ax)

    ## faint reference unit circle
    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(ring), np.sin(ring), color=MUTED, linewidth=1, linestyle="--", alpha=0.5)

    trail_len = 6  # a few fading ghosts behind each point
    trails = [ax.scatter(traj[0][:, 0], traj[0][:, 1], s=10, c=colors) for _ in range(trail_len)]
    head = ax.scatter(traj[0][:, 0], traj[0][:, 1], s=22, c=colors, edgecolor=BG, linewidth=0.4)
    title = ax.set_title("", color=INK)

    h = (1.0 - t_start) / (num_frames - 1)

    def draw(frame):
        for k, trail in enumerate(trails):
            past = max(0, frame - (trail_len - k))
            trail.set_offsets(traj[past])
            trail.set_alpha(0.06 * (k + 1))
        head.set_offsets(traj[frame])
        title.set_text(f"sampling   step {frame}/{num_frames - 1}   t = {t_start + frame * h:.2f}")
        return (*trails, head, title)

    ## play through, then linger on the final frame for `hold` seconds before looping
    frames = list(range(num_frames)) + [num_frames - 1] * int(hold * fps)
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def _style_axes(ax) -> None:
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=MUTED, alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
        spine.set_alpha(0.4)
