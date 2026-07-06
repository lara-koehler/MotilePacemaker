#!/usr/bin/env julia
"""
    julia --project=julia julia/scripts/run_minimal.jl [config_path] [output_path]

Run one simulation from a TOML config and write the result to an HDF5 file.
Defaults: config_path = configs/minimal.toml, output_path = data/raw/minimal/output.h5
(paths resolved relative to the repo root).
"""

using MotilePacemaker
using Random

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(repo_root, "configs", "minimal.toml")
output_path = length(ARGS) >= 2 ? ARGS[2] : joinpath(repo_root, "data", "raw", "minimal", "output.h5")

cfg, sourcescfg, _ = load_config(config_path)

rng = MersenneTwister(cfg.seed)
u0, v0 = initialize_field(cfg.nx, cfg.ny, rng)

positions0, n_sources, spacing_for_diagnostics = initialize_positions(sourcescfg, cfg.L, rng)

println("Running $(cfg.n_steps) steps on a $(cfg.nx)x$(cfg.ny) grid with $(n_sources) sources...")
println("(chi, Gamma) = ", chi_gamma(cfg.model, cfg.mech, spacing_for_diagnostics))
flush(stdout)

progress_every = max(1, cfg.n_steps ÷ 100)  # ~1% increments in the log
result = run_simulation(cfg, u0, v0, positions0; progress_every=progress_every)

save_run(output_path, result, read(config_path, String))
println("Saved to $(output_path)")
