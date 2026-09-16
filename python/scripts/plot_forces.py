#!/usr/bin/env python3
"""
Usage: python scripts/plot_forces.py <output.h5> <t> [out_dir]

Plots the field background, source positions, and each source's net
mechanical force (WCA + field-gated, via `metrics.compute_mechanical_forces`
-- a Python port of the Julia force law, for diagnostics only) as arrows, at
the saved field frame closest to simulation time `t`.

Saved as `<out_dir>/forces_<t>.png` (`out_dir` default:
`data/processed/<run_name>/`, via `io.run_name`), where `<t>` is the actual
simulation time of the matched frame -- not necessarily exactly the
requested `t`, since only saved frames are available.

Only the one matched field frame is read from `<output.h5>` (via an HDF5
hyperslab selection, `io.load_field_frame`), not the full u_field_stack --
source trajectories are read in full (`x_source`/`y_source` only, much
smaller than the field).
"""
import sys
from pathlib import Path

import numpy as np

from motilepacemaker import io, metrics, viz


def main():
    if len(sys.argv) < 3:
        print("Usage: plot_forces.py <output.h5> <t> [out_dir]")
        sys.exit(1)

    h5path = sys.argv[1]
    t_requested = float(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("../data/processed") / io.run_name(h5path)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx, t_actual, params = io.nearest_field_frame_index(h5path, t_requested)
    u = io.load_field_frame(h5path, frame_idx)
    L = params["grid"]["L"]
    save_every = params["time"]["save_every"]
    source_save_every = params["time"].get("source_save_every", 1)
    dt = params["time"]["dt"]

    data = io.load_run(h5path, keys=["x_source", "y_source"], verbose=True)
    x_source, y_source = data["x_source"], data["y_source"]
    source_idx = viz.field_frame_to_source_index(
        frame_idx, save_every, source_save_every, x_source.shape[1]
    )
    positions = np.stack([x_source[:, source_idx], y_source[:, source_idx]], axis=1)

    forces = metrics.compute_mechanical_forces(positions, u, L, params["mechanics"], dt=dt)

    vmax = float(np.abs(u).max())
    fig, ax, im = viz.plot_forces(
        u, positions, forces, L, vmin=-vmax, vmax=vmax,
        title=f"forces at t = {t_actual:.3g} (frame {frame_idx})",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="u")

    out_path = out_dir / f"forces_{t_actual:g}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
