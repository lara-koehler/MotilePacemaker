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
julia --project=. scripts/run_minimal.jl ../configs/minimal.toml ../data/raw/minimal/output.h5
```

**Always pass the output path explicitly** (second argument) -- `run_minimal.jl`
does not derive it from the config filename, so e.g. running with
`../configs/chaotic.toml` but no second argument still writes to
`data/raw/minimal/output.h5` (silently overwriting any previous run left
there) rather than `data/raw/chaotic/output.h5`.

## Analysis (Python)

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/plot_summary.py ../data/raw/minimal/output.h5

# animate the field + source positions over time (requires ffmpeg on PATH)
python scripts/make_movie.py ../data/raw/minimal/output.h5 ../data/processed/minimal_movie.mp4

# quick preview: only read/render every 4th saved field frame
python scripts/make_movie.py ../data/raw/minimal/output.h5 ../data/processed/minimal_preview.mp4 4
```

**This local workflow (`run_minimal.jl` + a single `configs/*.toml` file) is unaffected by the cluster tooling in [README_cluster.md](README_cluster.md)** -- it's still the fastest way to run one simulation.

Both scripts avoid reading a raw run's full `u_field_stack` -- easily the
dominant contributor to file size (~1 GB for a large run) -- since that
matters most when `<output.h5>` sits on a slow/network-mounted path (e.g. a
mounted cluster results drive, see [README_cluster.md](README_cluster.md)'s
step 5):
- `plot_summary.py` (via `motilepacemaker.io.load_run_lite`) only reads the
  last saved field frame and a thin y=L/2 kymograph strip, both via HDF5
  hyperslab selections, plus the source trajectories in full. One
  consequence: the final-state panel's color scale is derived from that
  frame + strip rather than the true max over the whole run (which would
  need the full array) -- in practice indistinguishable for a roughly
  spatially-homogeneous field.
- `make_movie.py`'s optional `stride` argument (default 1) reads/renders
  only every `stride`-th saved field frame, via the same kind of hyperslab
  selection -- a fast preview before committing to a full-resolution movie.
  If `<output.h5>`'s filename matches the `<scan_name>_<task_index>.h5`
  convention `run_scan.jl` writes, and `configs/scans/<scan_name>/
  manifest.toml` can be found, the swept parameters' actual values for that
  task are shown on every frame automatically (no flag needed, silently
  skipped for a plain non-scan run).

## Cluster usage

Running parameter scans on a SLURM cluster (including cluster-specific
Python environment setup) and analyzing results directly there, e.g. via
`aggregate_scan.py`, `plot_scan_task_summary.py`, or the `plot_phase_diagram*`
scripts, is covered in [README_cluster.md](README_cluster.md).
