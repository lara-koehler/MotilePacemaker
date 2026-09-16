"""Plotting: field snapshots, kymographs, trajectories, movies."""

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from . import metrics


def plot_field_snapshot(u, positions, L, ax=None, vmin=None, vmax=None, title="", markersize=20,
                         cmap="Reds", colors="grey"):
    """`u`: (nx, ny) field snapshot. `positions`: (N, 2) source positions.

    `colors`, if given, overrides the default uniform "grey" markers -- e.g.
    an (N, 4) array of per-source RGBA colors, fixed across calls, to track
    individual source identity (see `make_movie`'s `show_cell_identity`)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    im = ax.imshow(u.T, origin="lower", extent=[0, L, 0, L], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=markersize, edgecolor="white", linewidths=0.5, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax, im


def plot_forces(u, positions, forces, L, ax=None, vmin=None, vmax=None, title="",
                 markersize=20, force_scale=None, cmap="Reds", colors="grey"):
    """Field background + source positions (as `plot_field_snapshot`), with
    an arrow per source showing its net mechanical force -- e.g. from
    `metrics.compute_mechanical_forces`.

    `force_scale`, if given, is passed through to `quiver`'s `scale` (larger
    -> shorter arrows for the same force magnitude); by default (`None`)
    matplotlib autoscales arrow length to the plotted data on each call, so
    arrow lengths aren't directly comparable across different frames/runs --
    pass an explicit shared value for that. `cmap`/`colors` are forwarded to
    `plot_field_snapshot`."""
    fig, ax, im = plot_field_snapshot(u, positions, L, ax=ax, vmin=vmin, vmax=vmax,
                                       title=title, markersize=markersize, cmap=cmap, colors=colors)
    ax.quiver(positions[:, 0], positions[:, 1], forces[:, 0], forces[:, 1],
              color="black", scale=force_scale, width=0.003, zorder=3)
    return fig, ax, im


def kymograph(u_field_stack, y_index, width=4):
    """Average `u` over a strip of `width` grid rows centered at `y_index`,
    for every saved frame -> (T_saved, nx) array."""
    lo = max(int(y_index - width / 2), 0)
    hi = int(y_index + width / 2) + 1
    return u_field_stack[:, :, lo:hi].mean(axis=2)


def plot_kymograph(kymo, ax=None, cmap="RdBu", vmax=None):
    """`vmax`, if given, overrides the default per-call auto-scaling (`max(abs(kymo))`)
    -- pass a shared value across multiple calls for a consistent color scale
    (e.g. a grid of kymographs, one per subplot)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    vmax = vmax if vmax is not None else (np.max(np.abs(kymo)) or 1.0)
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


def plot_trajectories_2d(x_source, y_source, L=None, times=None, t_window=None, frac=1.0,
                          rng=None, ax=None, mark_start=True, mark_end=True, alpha=0.8):
    """Plot the 2D (x, y) path of every source over a time window, for one
    simulation. `x_source`/`y_source`: (N_sources, T) position arrays, same
    shape/orientation as `io.load_run`/`load_run_lite` return (raw, i.e.
    periodic-wrapped into [0, L)).

    `frac` (default 1.0): randomly plot only this fraction of sources
    (rounded to the nearest count, at least 1), useful when `N_sources` is
    large enough that plotting all of them is slow or visually cluttered.
    `rng`, if given (a `numpy.random.Generator`), makes the selection
    reproducible; otherwise a fresh one is used each call.

    Drawn as a single `LineCollection` (one segment per consecutive pair of
    samples, all sources at once) rather than a per-source Python loop of
    `ax.plot` calls, since `T` can be large. With `L` given, segments whose
    endpoints jump by more than `L/2` in x or y -- i.e. a wrap around the
    periodic boundary, not real motion -- are dropped from the collection
    (vectorized boolean mask, no explicit loop) instead of being drawn as a
    spurious straight line across the box. If you'd rather see the true
    unwrapped path extend beyond the box, run `x_source`/`y_source` through
    `metrics.unwrap_periodic_trajectories` first and pass `L=None` here (no
    wrap segments to drop, since the coordinates no longer wrap).

    `t_window = (t_min, t_max)` restricts the plot to that time range: with
    `times` given (the T time values matching each column, e.g. `data["times"]`),
    it's interpreted in simulation time units; without `times`, as a pair of
    column indices instead. Omit `t_window` to plot the full trajectories.

    `mark_start`/`mark_end`: scatter a marker ("o"/"x", matching each
    trajectory's line color) at its first/last plotted position, to show
    direction of travel at a glance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    if frac < 1.0:
        rng = np.random.default_rng() if rng is None else rng
        n_select = max(1, round(frac * x_source.shape[0]))
        idx = rng.choice(x_source.shape[0], size=n_select, replace=False)
        x_source = x_source[idx]
        y_source = y_source[idx]

    if t_window is not None:
        if times is not None:
            lo = np.searchsorted(times, t_window[0])
            hi = np.searchsorted(times, t_window[1], side="right")
        else:
            lo, hi = t_window
        x_source = x_source[:, lo:hi]
        y_source = y_source[:, lo:hi]

    n_sources, n_steps = x_source.shape
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = np.array([cycle[i % len(cycle)] for i in range(n_sources)])

    if n_steps > 1:
        points = np.stack([x_source, y_source], axis=-1)  # (N, T, 2)
        segments = np.stack([points[:, :-1], points[:, 1:]], axis=2)  # (N, T-1, 2, 2)
        seg_colors = np.repeat(colors, n_steps - 1)  # (N*(T-1),), same flattening order as below
        segments = segments.reshape(-1, 2, 2)

        if L is not None:
            jump = (np.abs(segments[:, 1, 0] - segments[:, 0, 0]) > L / 2) | \
                   (np.abs(segments[:, 1, 1] - segments[:, 0, 1]) > L / 2)
            segments = segments[~jump]
            seg_colors = seg_colors[~jump]

        ax.add_collection(LineCollection(segments, colors=seg_colors, alpha=alpha, linewidth=1))

    if mark_start:
        ax.scatter(x_source[:, 0], y_source[:, 0], c=colors, marker="o", s=16, zorder=3)
    if mark_end:
        ax.scatter(x_source[:, -1], y_source[:, -1], c=colors, marker="x", s=36, zorder=3)

    if L is not None:
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
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
                dt=None, fps=5, dpi=80, vmin=None, vmax=None, extra_title=None,
                show_forces=False, mech=None, force_scale=None, show_cell_identity=False):
    """`x_source`/`y_source` are (N, n_source_frames) trajectories saved every
    `source_save_every` steps; the field is saved every `save_every` steps --
    `field_frame_to_source_index` aligns the two cadences per frame.

    `dt`, if given, adds the actual simulation time for each frame to its
    title alongside the frame index -- `t = (1 + frame*save_every) * dt`,
    the same step-numbering convention as `io.field_stats_over_time` (field
    frame 0 is saved at step 1, not step 0). `save_every` here should already
    account for any frame `stride` (i.e. `stride`-th-frame subsampling), same
    as `field_frame_to_source_index` expects, so the reported time is correct
    even when only every `stride`-th saved frame is rendered.

    `extra_title`, if given, is appended below the per-frame title on every
    frame (e.g. a scan's swept-parameter values).

    `show_forces`: draw each source's net mechanical force as an arrow every
    frame, via `metrics.compute_mechanical_forces` (recomputed per frame from
    that frame's field + positions -- not read from the simulation itself,
    which doesn't save forces). Requires `mech` (a run's resolved
    `[mechanics]` config dict, e.g. `params["mechanics"]`); `force_scale` is
    forwarded to `plot_forces`.

    `show_cell_identity`: give each source a fixed color (from `tab20`,
    cycling every 20 sources), by source index, held constant across every
    frame -- so the same source can be visually tracked over time -- and
    switch the field colormap to greyscale ("Greys") instead of "Reds", so
    the fixed marker colors stay visually distinct from the field."""
    if show_forces and mech is None:
        raise ValueError("show_forces=True requires mech (the run's [mechanics] config dict)")

    n_frames = u_field_stack.shape[0]
    n_sources, n_source_frames = x_source.shape
    fig, ax = plt.subplots(figsize=(6, 6))

    cmap = "Greys" if show_cell_identity else "Reds"
    cell_colors = None
    if show_cell_identity:
        tab20 = plt.get_cmap("tab20")
        cell_colors = np.array([tab20(i % 20) for i in range(n_sources)])

    def update(t):
        ax.clear()
        idx = field_frame_to_source_index(t, save_every, source_save_every, n_source_frames)
        positions = np.stack([x_source[:, idx], y_source[:, idx]], axis=1)
        title = f"frame {t}"
        if dt is not None:
            title += f", t = {(1 + t * save_every) * dt:.3g}"
        if extra_title:
            title = f"{title}\n{extra_title}"
        colors = cell_colors if show_cell_identity else "grey"
        if show_forces:
            forces = metrics.compute_mechanical_forces(positions, u_field_stack[t], L, mech, dt=dt)
            plot_forces(u_field_stack[t], positions, forces, L, ax=ax, vmin=vmin, vmax=vmax,
                        title=title, cmap=cmap, colors=colors, force_scale=force_scale)
        else:
            plot_field_snapshot(u_field_stack[t], positions, L, ax=ax, vmin=vmin, vmax=vmax,
                                 title=title, cmap=cmap, colors=colors)
        return ()

    def progress(current_frame, total_frames):
        print(f"\rmake_movie: rendering frame {current_frame + 1}/{total_frames}", end="", flush=True)

    ani = animation.FuncAnimation(fig, update, frames=range(n_frames), blit=False)
    ani.save(path, dpi=dpi, fps=fps, progress_callback=progress)
    print()
    plt.close(fig)
    return path
