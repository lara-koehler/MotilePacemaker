# Cluster workflow

Running parameter scans on a SLURM cluster, and analyzing results directly
there. For the model physics, layout, and the local single-run / basic
Python analysis workflow, see [README.md](README.md).

## Cluster parameter scans (SLURM)

For sweeping one or more parameters over many combinations as a SLURM job
array, following the `parameters_array.txt` convention (one line per array
task, `sed -n -e "$SLURM_ARRAY_TASK_ID p"` extracts that task's values):

1. **One-time setup**: push this repo to a private GitHub/GitLab remote,
   then on the cluster:
   ```bash
   git clone <your-remote-url> MotilePacemaker
   cd MotilePacemaker/julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```
   To update later, just `git pull` on the cluster (and re-run
   `Pkg.instantiate()` if `julia/Manifest.toml` changed).

   `aggregate_scan.py` (step 5 below) also needs the `motilepacemaker`
   package, so set up Python on the cluster too (this is a separate install
   from your laptop's `python/.venv` -- it doesn't get created by `git
   clone`/`git pull`). **Use a conda environment, not a plain `venv`**: on at
   least one cluster, the system Python used to build a plain `venv` turned
   out to be compiled without OpenSSL support (`pip install` fails with "the
   ssl module in Python is not available", and no amount of venv/pip
   fiddling fixes it -- it's baked into that Python build). A conda-provided
   Python normally ships with working SSL out of the box:
   ```bash
   module load anaconda   # module name may differ -- check `module avail anaconda`
   conda create -n motilepacemaker python=3.10 numpy scipy matplotlib h5py -y
   conda activate motilepacemaker
   cd MotilePacemaker/python
   pip install -e . --no-deps   # just registers the local package; deps already satisfied via conda
   ```
   Usually best run on a login node (some clusters block internet access,
   needed here for `conda create`/`pip install`, from compute nodes).
   Re-run `pip install -e . --no-deps` after a `git pull` if
   `python/pyproject.toml` changed.

   **Every new session/job, before running any Python script here**, you
   need to re-activate this environment (creating it is one-time, but
   activation is per-shell and doesn't persist across logins):
   ```bash
   module load anaconda
   conda activate motilepacemaker
   ```
   Don't also `source .venv/bin/activate` a plain venv in the same shell if
   you have one lying around from experimenting -- an already-active venv on
   `PATH` can shadow conda's `python`/`pip` and bring back the SSL error even
   though conda itself is fine. `which python`/`which pip` should show a path
   containing `envs/motilepacemaker` when set up correctly.

2. **Write a sweep spec** naming a base config and the dotted TOML keys to
   scan (any section: `mechanics.*`, `coupling.*`, `sources.*`, ...), e.g.
   `configs/scans/example_epsilon_width.toml`:
   ```toml
   base_config = "../wave.toml"
   repeats = 1   # optional, default 1 -- see step 4 below

   [[sweep]]
   key = "mechanics.epsilon_LJ"
   values = [0.5, 1.0, 2.0]

   [[sweep]]
   key = "coupling.pacemaker_width"
   values = [0.01, 0.02, 0.05]
   ```

3. **Generate the scan** (cartesian product of all `values` lists, each
   combination repeated `repeats` times):
   ```bash
   module load anaconda && conda activate motilepacemaker
   cd python
   python scripts/generate_param_scan.py ../configs/scans/example_epsilon_width.toml
   ```
   Writes `configs/scans/example_epsilon_width/{parameters_array.txt,
   manifest.toml, base_config.toml, scan_index.csv}` and prints the total
   simulation count (combinations x repeats) -- use it for
   `#SBATCH --array=1-N` in `cluster/launch_scan.sh` (also set `--job-name`
   to the scan's name, which `launch_scan.sh` uses to find the right
   `configs/scans/<name>/` directory and to name outputs `<name>_<task_id>.h5`).

4. **Submit**: `sbatch cluster/launch_scan.sh` (after editing the `project`
   path and `--array`/`--job-name` for your scan, per the comments in the
   script). Each task writes straight to that job's `$scratch`, then copies
   to `/data/.../Results/<scan_name>/<scan_name>_<task_id>.h5`.

   To run the same parameter combination multiple times with different
   initial conditions, set `repeats` in the sweep spec (or manually duplicate
   a line in `parameters_array.txt` for a one-off repeat) -- `run_scan.jl`
   uses the SLURM array task ID as the run's default random seed, so
   otherwise-identical lines automatically get different initial conditions
   (unless the sweep spec explicitly includes `"time.seed"` as a swept key,
   in which case that value wins instead).

5. **Retrieve results**: raw per-task outputs can be large (e.g. ~900 MB for
   an nx=300, 180k-step run) -- a full scan can easily reach tens or hundreds
   of GB, too much to `rsync` down wholesale. Instead, **aggregate on the
   cluster first**, against the full-size files there, then copy down only
   the small result:
   ```bash
   # on the cluster, with the conda env from step 1 activated
   module load anaconda && conda activate motilepacemaker
   cd MotilePacemaker/python
   python scripts/aggregate_scan.py \
       ../configs/scans/<scan_name> /data/biophys/<username>/MotilePacemaker/<scan_name> \
       ../data/raw/scans/<scan_name>
   ```
   This distills each task's raw `.h5` down to just a kymograph, the final
   field/positions, a handful of source trajectories, and the Kuramoto order
   parameter over time (computed from the full source population before
   discarding it) into one `data/raw/scans/<scan_name>/<scan_name>_aggregate.h5`
   -- typically a ~100-300x size reduction, small enough to `rsync` normally:
   ```bash
   rsync -avz your-cluster:.../data/raw/scans/<scan_name>/ data/raw/<scan_name>/
   ```
   Missing/failed task files are skipped with a warning rather than aborting,
   so this can also be run against a scan that's still in progress. Every
   raw `<scan_name>_<k>.h5` still works with the existing Python pipeline
   unchanged if you do want to pull one down individually, or even just
   point straight at it over a mounted path without copying it down at all
   (`motilepacemaker.io.load_run`/`load_run_lite`, `plot_summary.py`,
   `make_movie.py`, `metrics.py`, ...) -- each file's `config_toml` attribute
   records that task's actual resolved parameters (including the per-task
   seed), and `scan_index.csv` (generated alongside the scan) is a quick
   human-readable lookup from task index to swept values (also queryable
   from Python: `motilepacemaker.scan.load_scan_index`, `task_params`,
   `task_indices_for_params`).

6. **Plot from the aggregate**:
   ```bash
   module load anaconda && conda activate motilepacemaker
   cd python

   # one task's 3-panel summary (final field state, kymograph, trajectories)
   python scripts/plot_scan_task_summary.py ../data/raw/scans/<scan_name>/<scan_name>_aggregate.h5 7

   # grid of final-state images: key1 -> rows, key2 -> columns, one replicate per cell
   python scripts/plot_phase_diagram.py ../data/raw/scans/<scan_name>/<scan_name>_aggregate.h5 \
       ../configs/scans/<scan_name> chemistry.I0 chemistry.b

   # same grid, but each cell is that task's kymograph instead of its final state
   python scripts/plot_phase_diagram_kymograph.py ../data/raw/scans/<scan_name>/<scan_name>_aggregate.h5 \
       ../configs/scans/<scan_name> chemistry.I0 chemistry.b

   # scalar phase diagram: Kuramoto order parameter averaged over the last few steps
   python scripts/plot_phase_diagram_kuramoto.py ../data/raw/scans/<scan_name>/<scan_name>_aggregate.h5 \
       ../configs/scans/<scan_name> chemistry.I0 chemistry.b --n-last 10
   ```
   All three phase-diagram scripts take `--replicate N` (default 1, since a
   scan with `repeats > 1` has several tasks per parameter combination) and
   `--fix key=value` (repeatable; only needed if the scan sweeps more than
   the 2 keys being plotted, to resolve which task a cell should show).

   All output PNGs are saved to `data/processed/<scan_name>/`: per-task
   plots as `<task_index>_summary.png`, whole-scan plots with a leading
   underscore, e.g. `_phase_diagram_chemistry.I0_chemistry.b.png`.
