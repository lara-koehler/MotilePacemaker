#!/usr/bin/env julia
"""
    julia --project=julia julia/scripts/run_scan.jl <manifest.toml> <task_index> <output_path> <value1> [value2 ...]

Run one simulation from a parameter-scan manifest (see
python/scripts/generate_param_scan.py), applying <value1>, <value2>, ... as
overrides onto the manifest's base config, in the order the manifest's
`keys` are listed. Mirrors the calling convention of SLURM array scripts that
extract one line of a `parameters_array.txt` via
`sed -n -e "\${SLURM_ARRAY_TASK_ID}p"` and pass it as trailing, unquoted,
shell-expanded positional args -- this script never needs to know in advance
how many parameters are being scanned.

`task_index` (typically `\$SLURM_ARRAY_TASK_ID`) becomes the run's random
seed by default -- applied *before* the scan overrides, so if the swept keys
happen to include `"time.seed"` explicitly, that value wins instead. This
means duplicating a `parameters_array.txt` line (to repeat the same
parameters with a different initial condition) gets a different seed for
each occurrence for free, since each is a different array task.

The saved output's `config_toml` records the actual overridden config
(including the resolved seed), not the template -- each result is
self-describing.
"""

using MotilePacemaker
using Random
using TOML

length(ARGS) >= 3 ||
    error("usage: run_scan.jl <manifest.toml> <task_index> <output_path> <value1> [value2 ...]")

manifest_path = ARGS[1]
task_index = parse(Int, ARGS[2])
output_path = ARGS[3]
value_args = ARGS[4:end]

manifest = TOML.parsefile(manifest_path)
base_config_path = joinpath(dirname(manifest_path), manifest["base_config"])
keys_to_override = manifest["keys"]

length(value_args) == length(keys_to_override) ||
    error("expected $(length(keys_to_override)) values (one per key in $(keys_to_override)), " *
          "got $(length(value_args)): $(value_args)")

raw = TOML.parsefile(base_config_path)

# default seed = task index, applied before overrides so an explicit
# "time.seed" sweep column (if present) takes precedence
apply_override!(raw, "time.seed", task_index)

overrides = Dict{String,Any}()
for (key, valstr) in zip(keys_to_override, value_args)
    value = parse_scan_value(valstr)
    apply_override!(raw, key, value)
    overrides[key] = value
end

cfg, sourcescfg = build_simulation_config(raw)

rng = MersenneTwister(cfg.seed)
u0, v0 = initialize_field(cfg.nx, cfg.ny, rng)
positions0, n_sources, spacing_for_diagnostics = initialize_positions(sourcescfg, cfg.L, rng)

println("Scan overrides: ", overrides)
println("Resolved seed: ", cfg.seed)
println("Running $(cfg.n_steps) steps on a $(cfg.nx)x$(cfg.ny) grid with $(n_sources) sources...")
println("(chi, Gamma) = ", chi_gamma(cfg.model, cfg.mech, spacing_for_diagnostics))
flush(stdout)

progress_every = max(1, cfg.n_steps ÷ 100)
result = run_simulation(cfg, u0, v0, positions0; progress_every=progress_every)

config_text = sprint(TOML.print, raw)
save_run(output_path, result, config_text)
println("Saved to $(output_path)")
