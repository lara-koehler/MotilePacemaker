"""
    initialize_field(nx, ny, rng) -> (u0, v0)

Small random perturbation around a resting state, matching the prototype's
initial condition.
"""
function initialize_field(nx::Int, ny::Int, rng)
    u = -1.0 .+ 0.5 .* randn(rng, nx, ny)
    v = 0.5 .+ 0.5 .* randn(rng, nx, ny)
    return u, v
end

"""
    step_fields!(u, v, model, I_forcing, dt, grid, noise, rng)

Advance the (u, v) fields by one timestep using a semi-implicit pseudospectral
scheme: the reaction term (from `reaction_rhs`, generic over `model.kinetics`)
is explicit, diffusion is implicit in Fourier space. Mutates `u`, `v` in place.
Optional additive field noise is applied after the deterministic step.
"""
function step_fields!(u::Matrix{Float64}, v::Matrix{Float64},
                       model::ReactionDiffusionModel, I_forcing::Matrix{Float64},
                       dt::Real, grid::Grid, noise::AbstractNoise, rng)
    du, dv = reaction_rhs(model.kinetics, u, v, I_forcing)

    u_hat = grid.fft_plan * complex.(u)
    v_hat = grid.fft_plan * complex.(v)
    fu_hat = grid.fft_plan * complex.(du)
    fv_hat = grid.fft_plan * complex.(dv)

    u_hat = (u_hat .+ dt .* fu_hat) ./ (1 .+ dt .* model.Du .* grid.k2)
    v_hat = (v_hat .+ dt .* fv_hat) ./ (1 .+ dt .* model.Dv .* grid.k2)

    u .= real.(grid.ifft_plan * u_hat)
    v .= real.(grid.ifft_plan * v_hat)

    apply_field_noise!(u, v, noise, dt, rng)

    return u, v
end
