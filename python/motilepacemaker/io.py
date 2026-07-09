"""Read simulation output written by the Julia side (`MotilePacemaker.save_run`)."""

import sys
import time

import h5py
import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ARRAY_KEYS = ["u_field_stack", "v_field_stack", "x_source", "y_source", "u_source", "v_source"]


def load_run(h5path, keys=None, verbose=False, field_stride=1):
    """Load one run's fields, source trajectories, and config.

    Returns a dict with keys: times, u_field_stack, v_field_stack, x_source,
    y_source, u_source, v_source (all numpy arrays, shaped as on the Julia
    side -- see note below), and `params` (the run's TOML config, parsed into
    nested dicts).

    `keys`: optional subset of `ARRAY_KEYS` to actually load (default: all
    six) -- skip arrays a caller doesn't need to save I/O. Raw per-task
    files can be ~1 GB; e.g. `make_movie.py` only ever touches
    `u_field_stack`/`x_source`/`y_source`, so it passes those three instead
    of paying for `v_field_stack`/`u_source`/`v_source` too.

    `field_stride`: subsample `u_field_stack`/`v_field_stack` to every
    `field_stride`-th saved frame, via an HDF5 hyperslab selection (so it
    cuts the bytes actually transferred, not just the array returned after
    the fact). Source arrays are untouched -- they're already sampled at
    their own, independent `source_save_every` cadence.

    `verbose`: print each array's size and load time as it's read -- useful
    on a slow/network-mounted `h5path` (e.g. a mounted cluster results
    drive), where a full load otherwise blocks silently with no feedback
    for however long the I/O takes.

    HDF5.jl stores arrays with axes fully reversed relative to Julia's
    `size()` (Julia is column-major, HDF5's C API/h5py are row-major), so
    every array read here is transposed (`.T`, which reverses all axes) to
    recover the shape it had in Julia: `u_field_stack` as
    `(n_saved, nx, ny)`, `x_source`/`u_source`/etc. as `(n_sources, n_steps)`.
    """
    array_keys = keys if keys is not None else ARRAY_KEYS
    with h5py.File(h5path, "r") as f:
        data = {"times": f["times"][()]}
        for key in array_keys:
            dset = f[key]
            strided = field_stride > 1 and key in ("u_field_stack", "v_field_stack")
            if strided:
                # on-disk axes are the FULL reverse of Julia's -- u_field_stack/
                # v_field_stack are (n_saved, nx, ny) in Julia, so the frame
                # axis is the LAST on-disk axis, not the first. Striding the
                # first on-disk axis instead would subsample ny (a spatial
                # axis) while silently leaving every frame in place.
                n_full = dset.shape[-1]
                n_frames = len(range(0, n_full, field_stride))
                size_mb = (dset.size / n_full) * n_frames * dset.dtype.itemsize / 1e6
            else:
                size_mb = dset.size * dset.dtype.itemsize / 1e6
            if verbose:
                suffix = f", every {field_stride} frames" if strided else ""
                print(f"load_run: reading {key} ({size_mb:.0f} MB{suffix})...", end="", flush=True)
                t0 = time.time()
            data[key] = (dset[:, :, ::field_stride] if strided else dset[()]).T
            if verbose:
                print(f" done ({time.time() - t0:.1f}s)")
        data["config_toml"] = f.attrs["config_toml"]
    if isinstance(data["config_toml"], bytes):
        data["config_toml"] = data["config_toml"].decode("utf-8")
    data["params"] = tomllib.loads(data["config_toml"])
    return data


def load_run_lite(h5path, kymo_width=4, verbose=False):
    """Read only what `plot_summary.py`'s 3-panel figure needs, without ever
    materializing the full `u_field_stack` (the dominant contributor to file
    size, easily ~1 GB for a large raw run) in memory: the last saved field
    frame and a thin y=L/2 kymograph strip, both via HDF5 hyperslab
    selections on the field dataset, plus the (much smaller) source
    trajectories in full.

    Returns a dict: `final_u_field` (nx, ny), `kymo` (n_field_saved, nx) --
    same shape/orientation `viz.kymograph` would produce -- `n_field_saved`,
    `x_source`, `y_source` (N, n_source_saved), `times`, `params`.

    Trade-off vs. `load_run` + `viz.plot_field_snapshot`'s usual `vmax`: the
    final-state panel's color scale can no longer be based on the max |u|
    over the *entire* run (that needs the full array) -- callers should
    scale it from `final_u_field` and/or `kymo` instead, which covers all
    saved frames but only at the y=L/2 strip.
    """
    with h5py.File(h5path, "r") as f:
        u_dset = f["u_field_stack"]  # on-disk shape (ny, nx, n_saved) -- see load_run's docstring
        ny, nx, n_field_saved = u_dset.shape
        lo = max(int(ny // 2 - kymo_width / 2), 0)
        hi = int(ny // 2 + kymo_width / 2) + 1

        if verbose:
            print("load_run_lite: reading final field frame...", end="", flush=True)
            t0 = time.time()
        final_u_field = u_dset[:, :, -1].T
        if verbose:
            print(f" done ({time.time() - t0:.1f}s)")

        if verbose:
            size_mb = (hi - lo) * nx * n_field_saved * u_dset.dtype.itemsize / 1e6
            print(f"load_run_lite: reading kymograph strip, y rows {lo}:{hi} ({size_mb:.0f} MB)...",
                  end="", flush=True)
            t0 = time.time()
        strip = u_dset[lo:hi, :, :]  # (hi-lo, nx, n_field_saved)
        kymo = strip.mean(axis=0).T  # -> (n_field_saved, nx)
        if verbose:
            print(f" done ({time.time() - t0:.1f}s)")

        data = {"final_u_field": final_u_field, "kymo": kymo, "n_field_saved": n_field_saved,
                "times": f["times"][()]}
        for key in ("x_source", "y_source"):
            data[key] = f[key][()].T
        config_toml = f.attrs["config_toml"]
    if isinstance(config_toml, bytes):
        config_toml = config_toml.decode("utf-8")
    data["params"] = tomllib.loads(config_toml)
    return data


def field_stats_over_time(h5path, key="u_field_stack", verbose=False):
    """Stream the spatial mean and std of a field dataset (default
    `u_field_stack`) one saved frame at a time, without ever holding more
    than a single (nx, ny) frame in memory -- a cheap proxy for how
    spatially uniform/quiescent (std near 0) vs. structured (std elevated,
    e.g. a wave or burst) the field is at each point in time, over an entire
    run, without paying for the full `(n_saved, nx, ny)` array.

    Returns `(field_times, mean_per_frame, std_per_frame)`. `field_times` is
    computed from `[time].dt`/`save_every` in the run's own config, using the
    same step-numbering convention as the Julia simulation loop (`step =
    1, 1+save_every, 1+2*save_every, ...`). Note this is a *different*
    cadence/array than `times` in `load_run`'s output, which is sampled at
    the source save cadence, not the field's.
    """
    with h5py.File(h5path, "r") as f:
        dset = f[key]  # on-disk shape (ny, nx, n_saved) -- see load_run's docstring
        n_saved = dset.shape[-1]
        config_toml = f.attrs["config_toml"]
        if isinstance(config_toml, bytes):
            config_toml = config_toml.decode("utf-8")
        params = tomllib.loads(config_toml)
        dt = params["time"]["dt"]
        save_every = params["time"]["save_every"]

        means = np.empty(n_saved)
        stds = np.empty(n_saved)
        report_every = max(1, n_saved // 20)
        for i in range(n_saved):
            if verbose and i % report_every == 0:
                print(f"\rfield_stats_over_time: frame {i}/{n_saved}", end="", flush=True)
            frame = dset[:, :, i]
            means[i] = frame.mean()
            stds[i] = frame.std()
        if verbose:
            print(f"\rfield_stats_over_time: frame {n_saved}/{n_saved}")

    field_times = (1 + np.arange(n_saved) * save_every) * dt
    return field_times, means, stds
