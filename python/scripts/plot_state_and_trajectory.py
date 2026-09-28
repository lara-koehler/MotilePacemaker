#!/usr/bin/env python3
"""
Usage: python scripts/plot_state_and_trajectory.py <output.h5> <t> [out_dir] [out_filename]
           [--percentage_traj 0.5] [--traj-window t_min t_max | --traj-duration d]

Produces one figure, one panel: the field + source positions at the saved
field frame closest to simulation time `t` (field in "Reds", sources in
grey -- as `plot_forces.py`/`plot_summary.py`), with the 2D (x, y)
trajectories of a random subset of sources over a time window drawn on top,
in royalblue instead of the usual per-source color cycle.

Saved as `<out_dir>/<out_filename>` (`out_dir` default:
`data/processed/<run_name>/`, via `io.run_name`; `out_filename` default:
`state_trajectory_<t>.png`, where `<t>` is the actual simulation time of the
matched state frame -- not necessarily exactly the requested `t`, since only
saved frames are available).

`--percentage_traj` (default 0.5): only plot a randomly selected fraction of
sources' trajectories (e.g. 0.2 -> 20%) -- useful when `N_sources` is large
enough that plotting all of them is slow/cluttered. The number actually
plotted is shown in that panel's title.

`--traj-window t_min t_max` restricts the trajectory panel to that
simulation-time range (default: the full run); it has no effect on the
state panel. Mutually exclusive with `--traj-duration`.

`--traj-duration d` restricts the trajectory panel to the `d`-long window
ending at the plotted state's own time (i.e. `[t_actual - d, t_actual]`, so
the trajectories always trail up to the frame shown), instead of specifying
the window's absolute bounds directly. Mutually exclusive with
`--traj-window`.

Only the one matched field frame is read from `<output.h5>` (via an HDF5
hyperslab selection, `io.load_field_frame`), not the full u_field_stack --
source trajectories are read in full (`x_source`/`y_source` only, much
smaller than the field).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from motilepacemaker import io, viz


def parse_args(argv):
    positional = []
    percentage_traj = 0.5
    traj_window = None
    traj_duration = None
    i = 0
    while i < len(argv):
        if argv[i] == "--percentage_traj":
            percentage_traj = float(argv[i + 1])
            i += 2
        elif argv[i] == "--traj-window":
            traj_window = (float(argv[i + 1]), float(argv[i + 2]))
            i += 3
        elif argv[i] == "--traj-duration":
            traj_duration = float(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    if traj_window is not None and traj_duration is not None:
        raise ValueError("--traj-window and --traj-duration are mutually exclusive")
    return positional, percentage_traj, traj_window, traj_duration


def main():
    positional, percentage_traj, traj_window, traj_duration = parse_args(sys.argv[1:])
    if len(positional) < 2:
        print("Usage: plot_state_and_trajectory.py <output.h5> <t> [out_dir] [out_filename] "
              "[--percentage_traj 0.5] [--traj-window t_min t_max | --traj-duration d]")
        sys.exit(1)

    h5path = positional[0]
    t_requested = float(positional[1])
    out_dir = Path(positional[2]) if len(positional) > 2 else Path("../data/processed") / io.run_name(h5path)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx, t_actual, params = io.nearest_field_frame_index(h5path, t_requested)
    if traj_duration is not None:
        traj_window = (t_actual - traj_duration, t_actual)
    u = io.load_field_frame(h5path, frame_idx)
    L = params["grid"]["L"]
    save_every = params["time"]["save_every"]
    source_save_every = params["time"].get("source_save_every", 1)

    data = io.load_run(h5path, keys=["x_source", "y_source"], verbose=True)
    x_source, y_source = data["x_source"], data["y_source"]
    source_idx = viz.field_frame_to_source_index(
        frame_idx, save_every, source_save_every, x_source.shape[1]
    )
    positions = np.stack([x_source[:, source_idx], y_source[:, source_idx]], axis=1)

    out_filename = positional[3] if len(positional) > 3 else f"state_trajectory_{t_actual:g}.png"

    fig, ax = plt.subplots(figsize=(6, 6))

    vmax = float(np.abs(u).max())
    fig, ax, im = viz.plot_field_snapshot(
        u, positions, L, ax=ax, vmin=-vmax, vmax=vmax,
        title=f"state at t = {t_actual:.3g} (frame {frame_idx})",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="u")

    # drawn on the same axes, above the field but below the source markers
    # (zorder=1, between plot_field_snapshot's imshow at 0 and scatter at 2);
    # no start/end markers here, since the current positions are already
    # shown by the field snapshot's own source markers
    viz.plot_trajectories_2d(x_source, y_source, L=L, times=data["times"],
                              t_window=traj_window, frac=percentage_traj,
                              ax=ax, color="royalblue", mark_start=False, mark_end=False)
    window_label = f"t in [{traj_window[0]}, {traj_window[1]}]" if traj_window else "full run"
    n_sources = x_source.shape[0]
    n_selected = max(1, round(percentage_traj * n_sources))
    ax.set_title(
        f"state at t = {t_actual:.3g} (frame {frame_idx})\n"
        f"trajectories ({window_label}, {n_selected}/{n_sources} sources = "
        f"{100 * percentage_traj:.0f}%)"
    )

    fig.tight_layout()
    out_path = out_dir / out_filename
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
