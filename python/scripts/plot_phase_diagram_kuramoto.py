#!/usr/bin/env python3
"""
Usage: python scripts/plot_phase_diagram_kuramoto.py <aggregate.h5> <scan_config_dir> <key1> <key2>
           [--replicate N] [--fix key=value ...] [--n-last 10] [out_dir]

A true scalar phase diagram: same (key1, key2) grid selection as
plot_phase_diagram.py (key1 -> rows, key2 -> columns, one replicate per
cell), but instead of tiling final-state images, plots a single heatmap of
the Kuramoto order parameter averaged over the last `--n-last` saved
timesteps of each selected task.

Whole-scan plot, saved with a leading underscore per convention:
data/processed/<scan_name>/_phase_diagram_kuramoto_<key1>_<key2>.png
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from motilepacemaker import scan


def parse_args(argv):
    positional = []
    replicate = 1
    n_last = 10
    fixed = {}
    i = 0
    while i < len(argv):
        if argv[i] == "--replicate":
            replicate = int(argv[i + 1])
            i += 2
        elif argv[i] == "--n-last":
            n_last = int(argv[i + 1])
            i += 2
        elif argv[i] == "--fix":
            k, v = argv[i + 1].split("=", 1)
            fixed[k] = scan.coerce_value(v)
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return positional, replicate, n_last, fixed


def main():
    positional, replicate, n_last, fixed = parse_args(sys.argv[1:])
    if len(positional) < 4:
        print("Usage: plot_phase_diagram_kuramoto.py <aggregate.h5> <scan_config_dir> <key1> <key2> "
              "[--replicate N] [--fix key=value ...] [--n-last 10] [out_dir]")
        sys.exit(1)

    agg_path = Path(positional[0])
    scan_config_dir = Path(positional[1])
    key1, key2 = positional[2], positional[3]
    scan_name = agg_path.parent.name
    out_dir = Path(positional[4]) if len(positional) > 4 else Path("../data/processed") / scan_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = scan.load_scan_index(scan_config_dir)
    key1_values, key2_values, task_grid = scan.select_task_grid(rows, key1, key2, replicate, fixed)
    n_rows, n_cols = len(key1_values), len(key2_values)

    heatmap = np.full((n_rows, n_cols), np.nan)
    with h5py.File(agg_path, "r") as f:
        for i in range(n_rows):
            for j in range(n_cols):
                task_index = task_grid[i][j]
                if task_index is None:
                    continue
                kuramoto_r = f[f"task_{task_index}"]["kuramoto_r"][()]
                heatmap[i, j] = np.mean(kuramoto_r[-n_last:])

    fig, ax = plt.subplots(figsize=(1.0 * n_cols + 2, 0.8 * n_rows + 2))
    im = ax.imshow(heatmap, cmap="viridis", vmin=0, vmax=1, aspect="auto", origin="upper")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(key2_values, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(key1_values)
    ax.set_xlabel(key2)
    ax.set_ylabel(key1)
    fig.colorbar(im, ax=ax, label=f"Kuramoto r (mean of last {n_last} saved steps)")
    fig.suptitle(f"{scan_name}: Kuramoto order parameter, replicate {replicate}")
    fig.tight_layout()

    out_path = out_dir / f"_phase_diagram_kuramoto_{key1}_{key2}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
