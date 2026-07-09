#!/usr/bin/env python3
"""
Usage: python scripts/make_movie.py <output.h5> [out.mp4] [stride]

`stride` (default 1): only read/render every `stride`-th saved field frame --
cuts both the HDF5 read and the rendering time roughly proportionally, useful
for a quick preview of a large raw run before committing to the full movie.

If `<output.h5>`'s filename matches the `<scan_name>_<task_index>.h5`
convention written by `run_scan.jl`, and `../configs/scans/<scan_name>/
manifest.toml` exists (i.e. run from `python/` in a checkout with that scan's
config still present), the swept parameters' actual values for this task are
shown on every frame automatically -- no flag needed, and silently skipped
for a plain (non-scan) run.

Requires ffmpeg to be installed and on PATH.
"""
import sys
from pathlib import Path

from motilepacemaker import io, scan, viz


def detect_swept_params_label(h5path, params, scans_dir=Path("../configs/scans")):
    """Best-effort: if `h5path`'s filename looks like a scan output
    (`<scan_name>_<task_index>.h5`) and that scan's `manifest.toml` can be
    found, return a "key=value, key=value" string of this task's actual
    swept-parameter values (read from its own resolved `params`, so it's
    always accurate for this file). Returns None otherwise -- never raises,
    since a plain single run has no manifest at all."""
    # split "<scan_name>_<task_index>" off the filename stem; bail out if it
    # doesn't fit that shape (e.g. a plain single run like "minimal")
    scan_name, _, task_str = Path(h5path).stem.rpartition("_")
    if not scan_name or not task_str.isdigit():
        return None
    scan_dir = scans_dir / scan_name
    try:
        # only exists for scan outputs -- this is what makes the whole
        # function a no-op for a plain run instead of erroring
        manifest = scan.load_manifest(scan_dir)
    except (FileNotFoundError, OSError):
        return None
    parts = []
    for key in manifest["keys"]:
        try:
            # read straight from this task's own resolved config, not
            # scan_index.csv, so it can't go stale relative to the file
            parts.append(f"{key}={scan.dotted_get(params, key)}")
        except KeyError:
            continue
    return ", ".join(parts) if parts else None


def main():
    if len(sys.argv) < 2:
        print("Usage: make_movie.py <output.h5> [out.mp4] [stride]")
        sys.exit(1)

    h5path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(h5path).with_name("movie.mp4"))
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # create the output folder if it doesn't exist yet, so out_path can point
    # anywhere without a separate mkdir step
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # only load the 3 arrays a movie actually needs (skips v_field_stack/
    # u_source/v_source), and only every `stride`-th field frame
    data = io.load_run(h5path, keys=["u_field_stack", "x_source", "y_source"],
                        verbose=True, field_stride=stride)
    params = data["params"]
    L = params["grid"]["L"]
    # strided frames are `stride` saved-snapshots apart, i.e. stride*save_every
    # simulation steps apart -- scale save_every so source positions (sampled
    # every source_save_every steps, independently) still line up per frame
    save_every = params["time"]["save_every"] * stride
    source_save_every = params["time"].get("source_save_every", 1)

    # auto-detect swept-parameter values for scan outputs; None (and hence a
    # no-op below) for a plain single run
    swept_label = detect_swept_params_label(h5path, params)
    if swept_label:
        print(f"Detected scan parameters: {swept_label}")

    vmax = float(abs(data["u_field_stack"]).max())
    viz.make_movie(data["u_field_stack"], data["x_source"], data["y_source"], L,
                    save_every, source_save_every, out_path, vmin=-vmax, vmax=vmax,
                    extra_title=swept_label)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
