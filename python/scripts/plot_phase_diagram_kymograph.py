#!/usr/bin/env python3
"""
Usage: python scripts/plot_phase_diagram_kymograph.py <aggregate.h5> <scan_config_dir> <key1> <key2>
           [--replicate N] [--fix key=value ...] [out_dir]

Tiles the kymograph of one task per (key1, key2) combination into one big
grid figure -- exactly like plot_phase_diagram.py, but each cell shows that
task's kymograph (space-time plot at y=L/2) instead of its final-state field
snapshot. key1's distinct values become rows, key2's become columns. Picks
one replicate per combination (default 1); if the scan sweeps more than 2
keys, pin the others with repeated --fix key=value flags or cell selection
is ambiguous (raises an error listing what's unresolved).

Whole-scan plot (not tied to one task), saved with a leading underscore per
convention: data/processed/<scan_name>/_phase_diagram_kymograph_<key1>_<key2>.png
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from motilepacemaker import scan, viz


def parse_args(argv):
    positional = []
    replicate = 1
    fixed = {}
    i = 0
    while i < len(argv):
        if argv[i] == "--replicate":
            replicate = int(argv[i + 1])
            i += 2
        elif argv[i] == "--fix":
            k, v = argv[i + 1].split("=", 1)
            fixed[k] = scan.coerce_value(v)
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    return positional, replicate, fixed


def main():
    positional, replicate, fixed = parse_args(sys.argv[1:])
    if len(positional) < 4:
        print("Usage: plot_phase_diagram_kymograph.py <aggregate.h5> <scan_config_dir> <key1> <key2> "
              "[--replicate N] [--fix key=value ...] [out_dir]")
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

    with h5py.File(agg_path, "r") as f:
        cells = [[None] * n_cols for _ in range(n_rows)]
        all_values = []
        for i in range(n_rows):
            for j in range(n_cols):
                task_index = task_grid[i][j]
                if task_index is None:
                    continue
                kymo = f[f"task_{task_index}"]["kymograph"][()]
                cells[i][j] = kymo
                all_values.append(kymo)

        vmax = max(np.max(np.abs(k)) for k in all_values) if all_values else 1.0

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows), squeeze=False)
        for i in range(n_rows):
            for j in range(n_cols):
                ax = axes[i, j]
                if cells[i][j] is None:
                    ax.axis("off")
                    continue
                viz.plot_kymograph(cells[i][j], ax=ax, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                # bare values per cell (not "key=value") -- the key names are
                # shown once for the whole figure via supxlabel/supylabel
                # below, since repeating a long dotted key on every column
                # header doesn't fit and overlaps
                ax.set_ylabel(str(key1_values[i]) if j == 0 else "")
                if i == 0:
                    ax.set_title(str(key2_values[j]))

    fig.suptitle(f"{scan_name}: kymograph, replicate {replicate}")
    fig.supxlabel(key2)
    fig.supylabel(key1)
    fig.tight_layout()

    out_path = out_dir / f"_phase_diagram_kymograph_{key1}_{key2}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
