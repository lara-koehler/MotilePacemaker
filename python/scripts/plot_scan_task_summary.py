#!/usr/bin/env python3
"""
Usage: python scripts/plot_scan_task_summary.py <aggregate.h5> <task_index> [out_dir]

Renders the same 3-panel summary as plot_summary.py (final field state,
kymograph, unwrapped x trajectories) but for a *single, specific* task drawn
from a scan's aggregate file (see aggregate_scan.py) instead of a full-size
local run. Always plots exactly the one task_index given -- never a batch.

plot_summary.py itself is untouched; this is a separate script (rather than a
flag on plot_summary.py) since the data source and indexing are different
enough to warrant it.

Defaults out_dir to data/processed/<scan_name>/, where scan_name is the
aggregate file's parent directory name (matching aggregate_scan.py's
<output_dir>/<scan_name>_aggregate.h5 convention). Saves as
<out_dir>/<task_index>_summary.png.

If ../configs/scans/<scan_name>/manifest.toml can be found (i.e. run from
python/ in a checkout with that scan's config still present), the swept
parameters' actual values for this task -- read from the task's own
config_toml, stored in the aggregate by aggregate_scan.py -- are shown in
the figure title, same convention as make_movie.py. Silently skipped if the
manifest isn't found.
"""
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from motilepacemaker import metrics, scan, viz


def swept_params_label(config_toml, scan_dir):
    """Best-effort: if `scan_dir`'s manifest.toml can be found, return a
    "key=value, key=value" string of this task's actual swept-parameter
    values, read from its own resolved config_toml (so it's always accurate
    for this task). Returns None otherwise -- never raises."""
    try:
        manifest = scan.load_manifest(scan_dir)
    except (FileNotFoundError, OSError):
        return None
    if isinstance(config_toml, bytes):
        config_toml = config_toml.decode("utf-8")
    params = tomllib.loads(config_toml)
    parts = []
    for key in manifest["keys"]:
        try:
            parts.append(f"{key}={scan.dotted_get(params, key)}")
        except KeyError:
            continue
    return ", ".join(parts) if parts else None


def main():
    if len(sys.argv) < 3:
        print("Usage: plot_scan_task_summary.py <aggregate.h5> <task_index> [out_dir]")
        sys.exit(1)

    agg_path = Path(sys.argv[1])
    task_index = int(sys.argv[2])
    scan_name = agg_path.parent.name
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("../data/processed") / scan_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(agg_path, "r") as f:
        group = f[f"task_{task_index}"]
        final_u_field = group["final_u_field"][()]
        final_positions = group["final_positions"][()]
        kymo = group["kymograph"][()]
        trajectory_x = group["trajectory_x"][()]
        times = group["times"][()]
        L = group.attrs["L"]
        config_toml = group.attrs["config_toml"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = max(abs(final_u_field.min()), abs(final_u_field.max()))
    viz.plot_field_snapshot(final_u_field, final_positions, L, ax=axes[0],
                             vmin=-vmax, vmax=vmax, title="final field state")

    viz.plot_kymograph(kymo, ax=axes[1])
    axes[1].set_title("kymograph at y = L/2")

    x_unwrapped = metrics.unwrap_periodic_trajectories(trajectory_x, L)
    viz.plot_trajectories(x_unwrapped, ax=axes[2], times=times)
    axes[2].set_title(f"unwrapped x trajectories ({trajectory_x.shape[0]} of full population)")

    scan_dir = Path("../configs/scans") / scan_name
    label = swept_params_label(config_toml, scan_dir)
    title = f"{scan_name} -- task {task_index}"
    if label:
        title += f" ({label})"
    fig.suptitle(title)
    fig.tight_layout()

    out_path = out_dir / f"{task_index}_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
