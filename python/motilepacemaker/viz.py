"""Plotting: field snapshots, kymographs, trajectories, movies."""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def plot_field_snapshot(u, positions, L, ax=None, vmin=None, vmax=None, title=""):
    """`u`: (nx, ny) field snapshot. `positions`: (N, 2) source positions."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    im = ax.imshow(u.T, origin="lower", extent=[0, L, 0, L], cmap="Reds", vmin=vmin, vmax=vmax)
    ax.scatter(positions[:, 0], positions[:, 1], c="grey", s=80, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax, im


def kymograph(u_field_stack, y_index, width=4):
    """Average `u` over a strip of `width` grid rows centered at `y_index`,
    for every saved frame -> (T_saved, nx) array."""
    lo = max(int(y_index - width / 2), 0)
    hi = int(y_index + width / 2) + 1
    return u_field_stack[:, :, lo:hi].mean(axis=2)


def plot_kymograph(kymo, ax=None, cmap="RdBu"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    vmax = np.max(np.abs(kymo)) or 1.0
    ax.imshow(kymo.T, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xlabel("saved frame")
    ax.set_ylabel("x index")
    return fig, ax


def plot_trajectories(x_unwrapped, ax=None, times=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    for traj in x_unwrapped:
        ax.plot(times if times is not None else np.arange(len(traj)), traj)
    ax.set_xlabel("time")
    ax.set_ylabel("x position")
    return fig, ax


def field_frame_to_source_index(field_frame_idx, save_every, source_save_every, n_source_frames):
    """Map a 0-indexed field-snapshot frame to the corresponding column index
    into the source trajectory arrays (`x_source`/`y_source`/etc.).

    The field is saved every `save_every` steps and the sources every
    `source_save_every` steps -- two independent cadences (see
    `SimulationConfig.source_save_every` on the Julia side) -- so the column
    index is *not* simply `field_frame_idx * save_every`; that only happens
    to be a valid index when source_save_every == 1, and silently clamps to
    the last column (freezing displayed positions) otherwise.
    """
    step = field_frame_idx * save_every
    idx = round(step / source_save_every)
    return min(idx, n_source_frames - 1)


def make_movie(u_field_stack, x_source, y_source, L, save_every, source_save_every, path,
                fps=5, dpi=80, vmin=None, vmax=None):
    """`x_source`/`y_source` are (N, n_source_frames) trajectories saved every
    `source_save_every` steps; the field is saved every `save_every` steps --
    `field_frame_to_source_index` aligns the two cadences per frame."""
    n_frames = u_field_stack.shape[0]
    n_source_frames = x_source.shape[1]
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(t):
        ax.clear()
        idx = field_frame_to_source_index(t, save_every, source_save_every, n_source_frames)
        positions = np.stack([x_source[:, idx], y_source[:, idx]], axis=1)
        plot_field_snapshot(u_field_stack[t], positions, L, ax=ax, vmin=vmin, vmax=vmax,
                             title=f"frame {t}")
        return ()

    ani = animation.FuncAnimation(fig, update, frames=range(n_frames), blit=False)
    ani.save(path, dpi=dpi, fps=fps)
    plt.close(fig)
    return path
