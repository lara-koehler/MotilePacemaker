#!/usr/bin/env python3
"""
Usage: python scripts/aggregate_scan.py <scan_config_dir> <raw_results_dir> <output_dir> [--n-traj 20]

Distills every task's full-size raw .h5 (as written by run_scan.jl) down to
just what's needed for plotting -- a few source trajectories, one kymograph,
the final field/positions, and the Kuramoto order parameter over time -- and
combines all tasks into a single small aggregate .h5. Meant to run on the
cluster (or wherever the raw per-task files live), against files far too
large to rsync down individually; the aggregate is typically 2-3 orders of
magnitude smaller and safe to copy normally.

<scan_config_dir>, e.g. configs/scans/260708ScanChemistry -- read for
scan_index.csv (task count). scan_name = Path(scan_config_dir).name.

<raw_results_dir> -- directory containing <scan_name>_<task_index>.h5 files
(e.g. the cluster's $results/<scan_name>/, or a local copy/mount of it).

Writes <output_dir>/<scan_name>_aggregate.h5, with one HDF5 group per task
(task_<k>), gzip-compressed datasets, and each task's own config_toml as a
group attribute for provenance. Missing/unreadable task files are skipped
with a warning rather than aborting the whole run, so this can also be used
to check progress on a scan that's still in flight.

Resumable: the aggregate file is opened in append mode (created fresh if it
doesn't exist yet), and any task already present with its `complete` marker
attribute set is skipped rather than re-read from the (typically much
larger, possibly slow/network-mounted) raw file -- so re-running this
script against a scan that's partway aggregated, or one where more tasks
have since finished, only does work for the tasks not yet aggregated. The
file is flushed to disk after every task, so an interrupted connection (or
job timeout) only loses whatever task was mid-read at that moment, not the
tasks already aggregated in that run or earlier ones. To force a specific
task to be re-aggregated (e.g. its raw file changed), delete its
`task_<k>` group from the aggregate first (or delete the whole aggregate
file to start over).
"""
import sys
from pathlib import Path

import h5py
import numpy as np

from motilepacemaker import io, metrics, scan, viz


def aggregate_task(raw_h5_path, n_traj):
    """Extract the small summary pieces from one full-size raw run."""
    data = io.load_run(raw_h5_path)
    params = data["params"]
    L = params["grid"]["L"]
    save_every = params["time"]["save_every"]
    source_save_every = params["time"].get("source_save_every", 1)

    u_field_stack = data["u_field_stack"]
    x_source = data["x_source"]
    y_source = data["y_source"]
    n_sources = x_source.shape[0]

    last_field_idx = u_field_stack.shape[0] - 1
    final_idx = viz.field_frame_to_source_index(
        last_field_idx, save_every, source_save_every, x_source.shape[1]
    )
    final_u_field = u_field_stack[last_field_idx]
    final_positions = np.stack([x_source[:, final_idx], y_source[:, final_idx]], axis=1)

    ny = u_field_stack.shape[2]
    kymo = viz.kymograph(u_field_stack, ny // 2)

    # Kuramoto needs the *whole* population -- compute before subsetting/discarding
    kuramoto_r = metrics.kuramoto_order_parameter_over_time(data["u_source"], data["v_source"])

    n_traj = min(n_traj, n_sources)
    traj_idx = np.linspace(0, n_sources - 1, n_traj, dtype=int)

    return {
        "final_u_field": final_u_field,
        "final_positions": final_positions,
        "kymograph": kymo,
        "trajectory_x": x_source[traj_idx, :],
        "trajectory_y": y_source[traj_idx, :],
        "times": data["times"],
        "kuramoto_r": kuramoto_r,
        "L": L,
        "config_toml": data["config_toml"],
    }


def write_task_group(h5file, task_index, summary):
    group_name = f"task_{task_index}"
    if group_name in h5file:
        # a partial group left over from a run interrupted mid-write -- drop
        # it and rewrite from scratch rather than leaving stale/incomplete
        # datasets behind
        del h5file[group_name]
    group = h5file.create_group(group_name)
    for key in ("final_u_field", "kymograph", "trajectory_x", "trajectory_y"):
        group.create_dataset(key, data=summary[key], compression="gzip", compression_opts=4)
    for key in ("final_positions", "times", "kuramoto_r"):
        group.create_dataset(key, data=summary[key])
    group.attrs["L"] = summary["L"]
    group.attrs["config_toml"] = summary["config_toml"]
    # set only once every dataset above has been written successfully, so a
    # crash/interruption mid-write leaves this task without the marker --
    # `is_task_cached` below then correctly treats it as not yet done
    group.attrs["complete"] = True


def is_task_cached(h5file, task_index):
    group_name = f"task_{task_index}"
    return group_name in h5file and bool(h5file[group_name].attrs.get("complete", False))


def main():
    if len(sys.argv) < 4:
        print("Usage: aggregate_scan.py <scan_config_dir> <raw_results_dir> <output_dir> [--n-traj N]")
        sys.exit(1)

    scan_config_dir = Path(sys.argv[1])
    raw_results_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    n_traj = 20
    if "--n-traj" in sys.argv:
        n_traj = int(sys.argv[sys.argv.index("--n-traj") + 1])

    scan_name = scan_config_dir.name
    rows = scan.load_scan_index(scan_config_dir)
    n_tasks = len(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{scan_name}_aggregate.h5"

    n_ok, n_skipped, n_cached = 0, 0, 0
    with h5py.File(output_path, "a") as agg:
        for row in rows:
            task_index = row["task_index"]
            if is_task_cached(agg, task_index):
                n_cached += 1
                continue
            raw_path = raw_results_dir / f"{scan_name}_{task_index}.h5"
            try:
                summary = aggregate_task(raw_path, n_traj)
                write_task_group(agg, task_index, summary)
                agg.flush()  # commit this task to disk now, not just at process exit
                n_ok += 1
            except (FileNotFoundError, OSError, KeyError) as e:
                print(f"  [skip] task {task_index} ({raw_path.name}): {e}")
                n_skipped += 1
            if task_index % 10 == 0 or task_index == n_tasks:
                print(f"  {task_index}/{n_tasks} processed "
                      f"({n_ok} new, {n_cached} cached, {n_skipped} skipped)")

    raw_size = sum(f.stat().st_size for f in raw_results_dir.glob(f"{scan_name}_*.h5"))
    agg_size = output_path.stat().st_size
    print(f"Wrote {output_path} ({n_ok} new, {n_cached} already cached, {n_skipped} skipped)")
    if raw_size:
        print(f"Aggregate size: {agg_size / 1e6:.1f} MB vs raw {raw_size / 1e9:.2f} GB "
              f"({raw_size / max(agg_size, 1):.0f}x reduction)")


if __name__ == "__main__":
    main()
