#!/usr/bin/env python3
"""
Usage: python scripts/generate_param_scan.py <sweep_spec.toml> [out_dir]

Reads a sweep spec (a base config + one or more swept dotted-key parameters,
see configs/scans/example_epsilon_width.toml), writes:
  <out_dir>/parameters_array.txt  -- one line per combination, whitespace-separated values
  <out_dir>/manifest.toml         -- base_config path + ordered dotted keys, for run_scan.jl
  <out_dir>/base_config.toml      -- a copy of the base config (so the scan directory is
                                      self-contained and immune to later edits of the original)
  <out_dir>/scan_index.csv        -- task_index,<key1>,<key2>,... lookup table (1-indexed,
                                      matching SLURM's array convention and run_scan.jl's
                                      default per-task seed)
and prints the combination count, for `#SBATCH --array=1-N`.

Defaults out_dir to a sibling directory named after the spec file, e.g.
configs/scans/example_epsilon_width.toml -> configs/scans/example_epsilon_width/.
"""
import csv
import shutil
import sys
from pathlib import Path

from motilepacemaker import scan


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_param_scan.py <sweep_spec.toml> [out_dir]")
        sys.exit(1)

    spec_path = Path(sys.argv[1])
    spec = scan.load_sweep_spec(spec_path)

    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else spec_path.parent / spec_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, rows = scan.generate_grid(spec)

    base_config_src = (spec_path.parent / spec["base_config"]).resolve()
    shutil.copy(base_config_src, out_dir / "base_config.toml")

    with open(out_dir / "parameters_array.txt", "w") as f:
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")

    with open(out_dir / "manifest.toml", "w") as f:
        f.write(scan.format_manifest_toml("base_config.toml", keys))

    with open(out_dir / "scan_index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_index", *keys])
        for i, row in enumerate(rows, start=1):
            writer.writerow([i, *row])

    print(f"Wrote {len(rows)} combinations to {out_dir}/parameters_array.txt")
    print(f"Set #SBATCH --array=1-{len(rows)}")


if __name__ == "__main__":
    main()
