#!/usr/bin/env python3
"""
Usage: python scripts/plot_summary.py <output.h5> [out.png]

Produces one figure: final field snapshot, kymograph, and unwrapped x
trajectories, for a quick look at a single run.
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

    data = io.load_run(h5path)
    params = data["params"]
    L = params["grid"]["L"]
    save_every = params["time"]["save_every"]
    source_save_every = params["time"].get("source_save_every", 1)

    x_source = data["x_source"]
    y_source = data["y_source"]
    u_field_stack = data["u_field_stack"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    last_field_idx = u_field_stack.shape[0] - 1
    step_idx = viz.field_frame_to_source_index(
        last_field_idx, save_every, source_save_every, x_source.shape[1]
    )
    positions_final = np.stack([x_source[:, step_idx], y_source[:, step_idx]], axis=1)
    vmax = np.max(np.abs(u_field_stack))
    viz.plot_field_snapshot(u_field_stack[last_field_idx], positions_final, L, ax=axes[0],
                             vmin=-vmax, vmax=vmax, title="final field state")

    ny = u_field_stack.shape[2]
    kymo = viz.kymograph(u_field_stack, ny // 2)
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
