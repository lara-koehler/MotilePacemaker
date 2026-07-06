"""Parameter-scan sweep specs: sweep spec TOML -> parameters_array.txt + manifest.toml,
consumed by julia/scripts/run_scan.jl on the cluster (or locally)."""

import itertools
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_sweep_spec(path):
    """Load a sweep spec: `{"base_config": str, "sweep": [{"key": str, "values": [...]}, ...]}`."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def generate_grid(spec):
    """Cartesian product of all swept values, in declaration order.

    Returns `(keys, rows)`: `keys` is the list of dotted TOML keys (one per
    column, e.g. `"mechanics.mobility"`), `rows` is a list of tuples (one per
    parameter combination, values in the same order as `keys`).
    """
    keys = [s["key"] for s in spec["sweep"]]
    value_lists = [s["values"] for s in spec["sweep"]]
    rows = list(itertools.product(*value_lists))
    return keys, rows


def _toml_string_literal(s):
    escaped = str(s).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_manifest_toml(base_config_relpath, keys):
    """Hand-format a minimal manifest.toml (`base_config` + `keys`). Python's
    stdlib `tomllib` is read-only and this manifest is simple enough not to
    need a TOML-writer dependency."""
    keys_toml = ", ".join(_toml_string_literal(k) for k in keys)
    return (
        f"base_config = {_toml_string_literal(base_config_relpath)}\n"
        f"keys = [{keys_toml}]\n"
    )
