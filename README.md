# Motile Pacemaker

Minimal mechanochemical model of motile, mechanically-interacting pacemaker
sources coupled to an excitable FitzHugh–Nagumo reaction–diffusion field —
a toy model for mitotic waves organizing nuclei in the syncytial *Drosophila*
embryo.

Simulations run in Julia; data processing and plotting happen in Python.
The two sides talk to each other through HDF5 files under `data/`.

## Physics

- **Field**: two coupled 2D reaction–diffusion fields `u` (activator), `v`
  (recovery), evolved with a semi-implicit pseudospectral solver on a
  periodic domain.
- **Sources**: discrete, motile "nuclei" that locally raise the excitation
  parameter of the field (pacemakers), and that interact mechanically via
  a short-range WCA repulsion plus a longer-range force whose sign is gated
  by the local field value (attractive when `u` is high, repulsive
  otherwise).
- **Reciprocity as a control knob**: the field-gated force can be run in a
  `:nonreciprocal` mode (force on nucleus *i* gated by the field at *xi*,
  as literally written in the model) or a `:reciprocal` mode (gated by the
  symmetrized field at the pair), selectable per run.
- **Kinetics are swappable**: the reaction term is behind a small interface
  (`AbstractKinetics`); FHN is the only kinetics implemented today, but a
  different (e.g. bistable/relaxation-oscillator) kinetics can be dropped in
  without touching the PDE solver or the particle code.
- **Noise** (additive field noise, positional jitter on nuclei) is a
  first-class, opt-in config option, off by default.

## Scaling to more sources

The code was originally written/tested with O(10) sources; it now scales to
hundreds–low thousands:

- **`pacemaker_field!`** only evaluates a local window around each source
  (sized to `[coupling].cutoff_factor * pacemaker_width`, default 4 widths)
  instead of the whole grid, which is the dominant per-step cost at
  low-to-moderate source counts. This also fixed a latent bug: the pacemaker
  forcing previously did *not* wrap across the periodic boundary (unlike the
  mechanical forces), so a source near one edge silently didn't influence the
  field near the opposite edge.
- **`NeighborList`** (`julia/src/neighbor_list.jl`) is a cell list + Verlet
  list that replaces the O(N²) all-pairs mechanical force loop with an O(N)
  one once the source density is roughly uniform. It's rebuilt automatically
  (tracking each source's displacement since the last rebuild; a rebuild
  triggers once two sources could plausibly have closed the `skin` buffer
  distance between them) -- no config needed beyond `[mechanics]`'s optional
  `verlet_skin` (defaults to `0.5 * interaction_range`). **It only helps when
  `interaction_range` is short relative to the domain `L`** -- if the force
  cutoff (`rcut_far_factor * interaction_range`, default `5x`) ends up
  comparable to `L`, there's no local structure to exploit and it degenerates
  to (slightly slower than) brute force. `wave.toml`/`wave_2d.toml`'s
  `interaction_range = 0.2` is in that regime; `chaotic.toml` and the denser
  configs are not.
- **`source_save_every`** (new, optional `[time]` key, default `1`) lets
  trajectory arrays (`x_source`/`y_source`/`u_source`/`v_source`/`times`) be
  saved at a coarser cadence than every step, independent of the field
  snapshots' `save_every` -- important once `N x n_steps` trajectory storage
  starts to dominate memory at high source counts.

## Layout

```
configs/            run configuration files (TOML)
  scans/              parameter-scan sweep specs + generated scan directories
julia/               simulation code (Julia package `MotilePacemaker`)
  src/                library code
  scripts/            entry-point scripts, e.g. run_minimal.jl, run_scan.jl
  test/               unit / regression tests
python/               analysis & plotting (package `motilepacemaker`)
  motilepacemaker/    library code
  scripts/            entry-point scripts, e.g. plot_summary.py, generate_param_scan.py
cluster/             SLURM job-array tooling for parameter scans
data/
  raw/                simulation outputs (HDF5), gitignored
  processed/          derived data/figures, gitignored
```

## Running a simulation (Julia)

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # first time only
julia --project=. test/runtests.jl                     # run tests
julia --project=. scripts/run_minimal.jl ../configs/minimal.toml
```

This writes `data/raw/minimal/output.h5`.

## Analysis (Python)

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/plot_summary.py ../data/raw/minimal/output.h5
```

**This local workflow (`run_minimal.jl` + a single `configs/*.toml` file) is unaffected by the cluster tooling below** -- it's still the fastest way to run one simulation.

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
   cd python && source .venv/bin/activate
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

5. **Retrieve results**:
   ```bash
   rsync -avz your-cluster:/data/biophys/<username>/Results/<scan_name>/ data/raw/scans/<scan_name>/
   ```
   After that, every `<scan_name>_<k>.h5` works with the existing Python
   pipeline exactly like any other run (`motilepacemaker.io.load_run`,
   `plot_summary.py`, `metrics.py`, ...) -- each file's `config_toml`
   attribute records that task's *actual* resolved parameters (including the
   per-task seed), so no separate bookkeeping is needed to know what a given
   result was run with (`scan_index.csv`, generated alongside the scan, is a
   quicker human-readable lookup from task index to swept values).
