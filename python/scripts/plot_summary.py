#!/usr/bin/env python3
"""
Usage: python scripts/plot_summary.py <output.h5> [out.png] [--time-window t_min t_max]
           [--percentage_traj 0.5]

Produces one figure: final field snapshot, kymograph, unwrapped x
trajectories, and 2D (x, y) trajectories, for a quick look at a single run.
Without an explicit `out.png`, saved to `data/processed/<run_name>/summary.png`
(`io.run_name` -- the raw file's parent directory name for the single-run
`output.h5` convention, else the file's own stem for per-task scan files).

`--time-window t_min t_max` restricts only the 2D trajectory panel to that
simulation-time range (default: the full run); it has no effect on the
other three panels or on the output filename -- the time window is shown
in that panel's own title instead.

`--percentage_traj` (default 0.5): only plot a randomly selected fraction of
sources' trajectories in the 2D panel (e.g. 0.2 -> 20%) -- useful when
`N_sources` is large enough that plotting all of them is slow/cluttered.
The number actually plotted is shown in that panel's title.

Uses `io.load_run_lite`, which never materializes the full `u_field_stack`
(the dominant contributor to file size) in memory -- only the last saved
frame and a y=L/2 kymograph strip are read from it, via HDF5 hyperslab
selections. This matters on a large raw per-task file (e.g. from a cluster
scan) sitting on a slow/network-mounted path, where reading the whole field
stack can otherwise take minutes. One consequence: the final-state panel's
color scale is derived from the last frame and the kymograph strip, not the
true max over the whole run/domain (which would require the full array) --
in practice indistinguishable for a roughly spatially-homogeneous field.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from motilepacemaker import io, metrics, viz


def parse_args(argv):
    positional = []
    time_window = None
    percentage_traj = 0.5
    i = 0
    while i < len(argv):
        if argv[i] == "--time-window":
            time_window = (float(argv[i + 1]), float(argv[i + 2]))
            i += 3
        elif argv[i] == "--percentage_traj":
            percentage_traj = float(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return positional, time_window, percentage_traj


def main():
    positional, time_window, percentage_traj = parse_args(sys.argv[1:])
    if len(positional) < 1:
        print("Usage: plot_summary.py <output.h5> [out.png] [--time-window t_min t_max] "
              "[--percentage_traj 0.5]")
        sys.exit(1)

    h5path = positional[0]
    if len(positional) > 1:
        out_path = Path(positional[1])
    else:
        out_path = Path("../data/processed") / io.run_name(h5path) / "summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = io.load_run_lite(h5path, verbose=True)
    params = data["params"]
    L = params["grid"]["L"]
    save_every = params["time"]["save_every"]
    source_save_every = params["time"].get("source_save_every", 1)

    x_source = data["x_source"]
    y_source = data["y_source"]
    final_u_field = data["final_u_field"]
    kymo = data["kymo"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    last_field_idx = data["n_field_saved"] - 1
    step_idx = viz.field_frame_to_source_index(
        last_field_idx, save_every, source_save_every, x_source.shape[1]
    )
    positions_final = np.stack([x_source[:, step_idx], y_source[:, step_idx]], axis=1)
    vmax = max(np.max(np.abs(final_u_field)), np.max(np.abs(kymo)))
    viz.plot_field_snapshot(final_u_field, positions_final, L, ax=axes[0, 0],
                             vmin=-vmax, vmax=vmax, title="final field state")

    viz.plot_kymograph(kymo, ax=axes[0, 1])
    axes[0, 1].set_title("kymograph at y = L/2")

    x_unwrapped = metrics.unwrap_periodic_trajectories(x_source, L)
    viz.plot_trajectories(x_unwrapped, ax=axes[1, 0], times=data["times"])
    axes[1, 0].set_title("unwrapped x trajectories")

    viz.plot_trajectories_2d(x_source, y_source, L=L, times=data["times"],
                              t_window=time_window, frac=percentage_traj, ax=axes[1, 1])
    window_label = f"t in [{time_window[0]}, {time_window[1]}]" if time_window else "full run"
    n_sources = x_source.shape[0]
    n_selected = max(1, round(percentage_traj * n_sources))
    axes[1, 1].set_title(
        f"2D trajectories ({window_label}, {n_selected}/{n_sources} sources = "
        f"{100 * percentage_traj:.0f}%)"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
