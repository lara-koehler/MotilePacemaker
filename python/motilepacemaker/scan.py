"""Parameter-scan sweep specs: sweep spec TOML -> parameters_array.txt + manifest.toml,
consumed by julia/scripts/run_scan.jl on the cluster (or locally). Also reads
back scan_index.csv for post-processing (aggregate_scan.py, the phase-diagram
scripts)."""

import csv
import itertools
import sys
from pathlib import Path

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


def load_manifest(scan_dir):
    """Read `<scan_dir>/manifest.toml` (written by `generate_param_scan.py`):
    `{"base_config": str, "keys": [dotted swept-key strings]}`."""
    with open(Path(scan_dir) / "manifest.toml", "rb") as f:
        return tomllib.load(f)


def dotted_get(d, dotted_key):
    """Look up a dotted TOML key (e.g. `"chemistry.b"`) in a nested dict,
    such as the `params` dict returned by `io.load_run`."""
    value = d
    for part in dotted_key.split("."):
        value = value[part]
    return value


def coerce_value(s):
    """Type-infer a scan_index.csv cell: int, then float, else leave as string."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_scan_index(scan_dir):
    """Read `<scan_dir>/scan_index.csv` (written by `generate_param_scan.py`)
    into a list of dicts: `task_index`/`replicate` as `int`, swept-key
    columns type-inferred (`int`/`float`/`str`), keyed by their exact column
    name (e.g. `"chemistry.b"`)."""
    path = Path(scan_dir) / "scan_index.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw_row in reader:
            row = {"task_index": int(raw_row["task_index"]), "replicate": int(raw_row["replicate"])}
            for key, value in raw_row.items():
                if key in ("task_index", "replicate"):
                    continue
                row[key] = coerce_value(value)
            rows.append(row)
    return rows


def task_params(rows, task_index):
    """Look up the swept-key values for one `task_index` from `load_scan_index`'s
    rows -- the reverse lookup of `select_task_grid` (parameter values -> task
    index): here we go from a task index back to its parameter values."""
    for r in rows:
        if r["task_index"] == task_index:
            return {k: v for k, v in r.items() if k not in ("task_index", "replicate")}
    raise ValueError(f"task_index {task_index} not found in scan index")


def task_indices_for_params(rows, params):
    """Return every `task_index` in `rows` matching all key=value pairs in
    `params` (e.g. `{"chemistry.I0": 1.2, "chemistry.b": 0.5}`) -- the reverse
    of `task_params`. Unlike `select_task_grid`, this doesn't filter by
    replicate or error on multiple matches: getting one task_index per
    replicate back is the normal, expected result."""
    return [r["task_index"] for r in rows if all(r.get(k) == v for k, v in params.items())]


def select_task_grid(rows, key1, key2, replicate=1, fixed=None):
    """Build a task-index grid for a 2-key phase diagram from `load_scan_index`'s
    rows: `key1`'s distinct values become grid rows, `key2`'s become columns.

    `fixed` (optional `{other_key: value}`) pins any additional swept keys --
    only needed if the scan sweeps more than 2 keys. Raises `ValueError` if,
    after filtering by `replicate`/`fixed`, more than one task still matches
    a given cell (ambiguous -- add more `fixed` constraints) rather than
    silently picking one.

    Returns `(key1_values, key2_values, task_index_grid)`: the first two are
    sorted lists of distinct values; `task_index_grid` is
    `len(key1_values) x len(key2_values)`, entries `None` for missing combos.
    """
    fixed = fixed or {}
    filtered = [
        r for r in rows
        if r["replicate"] == replicate and all(r.get(k) == v for k, v in fixed.items())
    ]

    key1_values = sorted({r[key1] for r in filtered})
    key2_values = sorted({r[key2] for r in filtered})

    task_index_grid = []
    for v1 in key1_values:
        row_tasks = []
        for v2 in key2_values:
            matches = [r for r in filtered if r[key1] == v1 and r[key2] == v2]
            if len(matches) > 1:
                other_keys = [k for k in filtered[0] if k not in ("task_index", "replicate", key1, key2)]
                raise ValueError(
                    f"Ambiguous cell ({key1}={v1}, {key2}={v2}): {len(matches)} tasks match "
                    f"(task_indices={[m['task_index'] for m in matches]}). This scan sweeps "
                    f"additional keys ({other_keys}) -- pin them via the `fixed=` argument."
                )
            row_tasks.append(matches[0]["task_index"] if matches else None)
        task_index_grid.append(row_tasks)

    return key1_values, key2_values, task_index_grid
