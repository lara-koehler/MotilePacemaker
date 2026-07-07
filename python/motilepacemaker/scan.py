"""Parameter-scan sweep specs: sweep spec TOML -> parameters_array.txt + manifest.toml,
consumed by julia/scripts/run_scan.jl on the cluster (or locally)."""

import itertools
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_sweep_spec(path):
    """Load a sweep spec: `{"base_config": str, "repeats": int (optional,
    default 1), "sweep": [{"key": str, "values": [...]}, ...]}`."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def generate_grid(spec):
    """Cartesian product of all swept values, in declaration order, each
    combination repeated `spec.get("repeats", 1)` times.

    Returns `(keys, rows, replicate_ids)`: `keys` is the list of dotted TOML
    keys (one per column, e.g. `"mechanics.mobility"`); `rows` is a list of
    tuples (one per simulation, values in the same order as `keys`), with
    each distinct combination's repeats grouped consecutively; `replicate_ids`
    is a parallel list of 1-indexed replicate numbers (1, 2, ..., repeats) for
    each row, all 1 when `repeats` is omitted/1.

    Repeats exist to run the exact same parameters multiple times with
    different initial conditions: `run_scan.jl` seeds each run from its
    (1-indexed) line number, so identical rows produced here automatically
    get different seeds -- nothing else needs to vary them.
    """
    keys = [s["key"] for s in spec["sweep"]]
    value_lists = [s["values"] for s in spec["sweep"]]
    repeats = spec.get("repeats", 1)

    rows = []
    replicate_ids = []
    for combo in itertools.product(*value_lists):
        for r in range(1, repeats + 1):
            rows.append(combo)
            replicate_ids.append(r)
    return keys, rows, replicate_ids


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
