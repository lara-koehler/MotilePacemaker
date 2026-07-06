"""Read simulation output written by the Julia side (`MotilePacemaker.save_run`)."""

import sys

import h5py

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_run(h5path):
    """Load one run's fields, source trajectories, and config.

    Returns a dict with keys: times, u_field_stack, v_field_stack, x_source,
    y_source, u_source, v_source (all numpy arrays, shaped as on the Julia
    side -- see note below), and `params` (the run's TOML config, parsed into
    nested dicts).

    HDF5.jl stores arrays with axes fully reversed relative to Julia's
    `size()` (Julia is column-major, HDF5's C API/h5py are row-major), so
    every array read here is transposed (`.T`, which reverses all axes) to
    recover the shape it had in Julia: `u_field_stack` as
    `(n_saved, nx, ny)`, `x_source`/`u_source`/etc. as `(n_sources, n_steps)`.
    """
    array_keys = ["u_field_stack", "v_field_stack", "x_source", "y_source", "u_source", "v_source"]
    with h5py.File(h5path, "r") as f:
        data = {"times": f["times"][()]}
        for key in array_keys:
            data[key] = f[key][()].T
        data["config_toml"] = f.attrs["config_toml"]
    if isinstance(data["config_toml"], bytes):
        data["config_toml"] = data["config_toml"].decode("utf-8")
    data["params"] = tomllib.loads(data["config_toml"])
    return data
