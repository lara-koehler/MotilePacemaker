using Random

"""
    SimulationConfig

Full specification of one run: grid, timestepping, physical model, coupling,
mechanics, and (optional) noise. Construct with keyword arguments; see
`load_config` for building one from a TOML file.
"""
Base.@kwdef struct SimulationConfig
    L::Float64
    nx::Int
    ny::Int
    dt::Float64
    n_steps::Int
    save_every::Int
    source_save_every::Int = 1
    model::ReactionDiffusionModel
    coupling::PacemakerCoupling
    mech::MechanicalParams
    field_noise::AbstractNoise = NoNoise()
    position_noise::Union{PositionalNoise,Nothing} = nothing
    seed::Int = 1
end

"""
    SimulationResult

Output of `run_simulation`: field snapshots (saved every `save_every` steps)
and source trajectories/field values (saved every `source_save_every` steps;
`times` is sampled at the same cadence, so it always matches the second
dimension of `x_source` etc.).
"""
struct SimulationResult
    times::Vector{Float64}
    u_field_stack::Array{Float64,3}
    v_field_stack::Array{Float64,3}
    x_source::Matrix{Float64}
    y_source::Matrix{Float64}
    u_source::Matrix{Float64}
    v_source::Matrix{Float64}
end

"""
    run_simulation(cfg, u0, v0, positions0; progress_every=nothing) -> SimulationResult

Run the coupled field/source dynamics for `cfg.n_steps` steps of size `cfg.dt`,
starting from field state `(u0, v0)` and source `positions0` (an `N x 2` matrix).

If `progress_every` is set to an integer, a one-line progress report (step
count, percent complete, elapsed and estimated-remaining time) is printed
every `progress_every` steps and flushed immediately, so it shows up live in
a redirected log file (e.g. via `tail -f`) rather than only appearing once
the process exits or its output buffer fills.
"""
function run_simulation(cfg::SimulationConfig, u0::Matrix{Float64}, v0::Matrix{Float64},
                         positions0::Matrix{Float64}; progress_every::Union{Int,Nothing}=nothing)
    start_time = time()
    rng = MersenneTwister(cfg.seed)
    grid = make_grid(cfg.L, cfg.nx, cfg.ny)

    u = copy(u0)
    v = copy(v0)
    positions = copy(positions0)
    N = size(positions, 1)

    cutoff = interaction_cutoff(cfg.mech)
    neighbor_list = NeighborList(cfg.L, cutoff, cfg.mech.verlet_skin, positions)

    # length(1:every:n_steps), i.e. exactly how many times step in 1:n_steps
    # satisfies (step-1) % every == 0 below -- NOT n_steps ÷ every + 1, which
    # overcounts by one whenever n_steps is an exact multiple of every.
    n_field_saved = length(1:cfg.save_every:cfg.n_steps)
    u_field_stack = Array{Float64}(undef, n_field_saved, cfg.nx, cfg.ny)
    v_field_stack = Array{Float64}(undef, n_field_saved, cfg.nx, cfg.ny)

    n_source_saved = length(1:cfg.source_save_every:cfg.n_steps)
    x_source = Matrix{Float64}(undef, N, n_source_saved)
    y_source = Matrix{Float64}(undef, N, n_source_saved)
    u_source = Matrix{Float64}(undef, N, n_source_saved)
    v_source = Matrix{Float64}(undef, N, n_source_saved)
    times = Vector{Float64}(undef, n_source_saved)

    I_forcing = zeros(Float64, cfg.nx, cfg.ny)
    field_save_idx = 1
    source_save_idx = 1

    for step in 1:cfg.n_steps
        pacemaker_field!(I_forcing, positions, cfg.coupling, grid)
        step_fields!(u, v, cfg.model, I_forcing, cfg.dt, grid, cfg.field_noise, rng)

        if N > 1
            update_positions!(positions, u, cfg.dt, cfg.mech, grid, cfg.position_noise, rng, neighbor_list)
        end

        if (step - 1) % cfg.source_save_every == 0
            for k in 1:N
                ix = local_field_index(positions[k, 1], cfg.L, cfg.nx)
                iy = local_field_index(positions[k, 2], cfg.L, cfg.ny)
                x_source[k, source_save_idx] = positions[k, 1]
                y_source[k, source_save_idx] = positions[k, 2]
                u_source[k, source_save_idx] = u[ix, iy]
                v_source[k, source_save_idx] = v[ix, iy]
            end
            times[source_save_idx] = step * cfg.dt
            source_save_idx += 1
        end

        if (step - 1) % cfg.save_every == 0
            u_field_stack[field_save_idx, :, :] = u
            v_field_stack[field_save_idx, :, :] = v
            field_save_idx += 1
        end

        if progress_every !== nothing && (step % progress_every == 0 || step == cfg.n_steps)
            elapsed_min = (time() - start_time) / 60
            frac = step / cfg.n_steps
            eta_min = elapsed_min / frac - elapsed_min
            println("step $(step)/$(cfg.n_steps) ($(round(100 * frac, digits=1))%) -- ",
                     "elapsed $(round(elapsed_min, digits=1)) min, ETA $(round(eta_min, digits=1)) min")
            flush(stdout)
        end
    end

    return SimulationResult(times, u_field_stack, v_field_stack, x_source, y_source, u_source, v_source)
end

"""
    chi_gamma(model, mech, l0) -> (χ, Γ)

Dimensionless control numbers comparing chemical and mechanical timescales:
`χ = μF1 / sqrt(D*τu)` (nuclear speed vs. wave speed) and
`Γ = τv*μF1 / l0` (recovery time vs. mechanical relaxation time), for a
source spacing `l0`.
"""
function chi_gamma(model::ReactionDiffusionModel, mech::MechanicalParams, l0::Real)
    τu = model.kinetics.τu
    τv = model.kinetics.τv
    v_wave = sqrt(model.Du / τu)
    v_particles = mech.mobility * mech.interaction_strength

    χ = v_particles / v_wave
    Γ = τv * mech.mobility * mech.interaction_strength / l0
    return χ, Γ
end
