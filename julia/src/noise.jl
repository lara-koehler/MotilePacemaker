"""
Pluggable, opt-in noise. Everything defaults to `NoNoise`/`nothing` so a run is
fully deterministic unless noise is explicitly requested in the config.
"""
abstract type AbstractNoise end

struct NoNoise <: AbstractNoise end

"""
    AdditiveFieldNoise(σu, σv)

Additive white noise on the field update, applied (Euler–Maruyama) after the
deterministic field step: `u += σu * sqrt(dt) * randn()` and similarly for `v`.
"""
struct AdditiveFieldNoise{T<:Real} <: AbstractNoise
    σu::T
    σv::T

    function AdditiveFieldNoise(σu::Real, σv::Real)
        σu, σv = promote(σu, σv)
        new{typeof(σu)}(σu, σv)
    end
end

apply_field_noise!(u, v, ::NoNoise, dt, rng) = nothing

function apply_field_noise!(u, v, noise::AdditiveFieldNoise, dt, rng)
    sq = sqrt(dt)
    if noise.σu > 0
        u .+= noise.σu * sq .* randn(rng, size(u))
    end
    if noise.σv > 0
        v .+= noise.σv * sq .* randn(rng, size(v))
    end
    return nothing
end

"""
    PositionalNoise(σ)

Independent Gaussian jitter added to each nucleus's displacement every step
(thermal-like noise on the overdamped motion), on top of the deterministic
mechanical forces.
"""
struct PositionalNoise{T<:Real}
    σ::T
end

apply_position_noise!(positions, ::Nothing, dt, rng) = nothing

function apply_position_noise!(positions, noise::PositionalNoise, dt, rng)
    if noise.σ > 0
        positions .+= noise.σ * sqrt(dt) .* randn(rng, size(positions))
    end
    return nothing
end
