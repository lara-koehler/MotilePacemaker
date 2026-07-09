#!/usr/bin/env python3
"""
Usage: python scripts/plot_summary.py <output.h5> [out.png]

Produces one figure: final field snapshot, kymograph, and unwrapped x
trajectories, for a quick look at a single run.

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


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_summary.py <output.h5> [out.png]")
        sys.exit(1)

    h5path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(h5path).with_name("summary.png"))

    data = io.load_run_lite(h5path, verbose=True)
    params = data["params"]
    L = params["grid"]["L"]
    save_every = params["time"]["save_every"]
    source_save_every = params["time"].get("source_save_every", 1)

    x_source = data["x_source"]
    y_source = data["y_source"]
    final_u_field = data["final_u_field"]
    kymo = data["kymo"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    last_field_idx = data["n_field_saved"] - 1
    step_idx = viz.field_frame_to_source_index(
        last_field_idx, save_every, source_save_every, x_source.shape[1]
    )
    positions_final = np.stack([x_source[:, step_idx], y_source[:, step_idx]], axis=1)
    vmax = max(np.max(np.abs(final_u_field)), np.max(np.abs(kymo)))
    viz.plot_field_snapshot(final_u_field, positions_final, L, ax=axes[0],
                             vmin=-vmax, vmax=vmax, title="final field state")

    viz.plot_kymograph(kymo, ax=axes[1])
    axes[1].set_title("kymograph at y = L/2")

    x_unwrapped = metrics.unwrap_periodic_trajectories(x_source, L)
    viz.plot_trajectories(x_unwrapped, ax=axes[2], times=data["times"])
    axes[2].set_title("unwrapped x trajectories")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
