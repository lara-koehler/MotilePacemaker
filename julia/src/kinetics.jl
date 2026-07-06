"""
Reaction kinetics for the two-field (activator `u`, recovery `v`) reaction–diffusion
system. The field solver only ever calls `reaction_rhs` — it has no knowledge of the
concrete kinetics. FHN is a smooth normal form, not the real bistable Cdk1/APC-C
switch; swapping in a different two-field kinetics (e.g. a relaxation oscillator)
means adding a new subtype + `reaction_rhs` method here, nothing else.
"""
abstract type AbstractKinetics end

"""
    FHNKinetics(τu, τv, a, b, I0)

FitzHugh–Nagumo / Van der Pol type kinetics:
    du/dt = (u - u^3 - v + I0 + I_forcing) / τu
    dv/dt = (u + a - b*v) / τv
"""
struct FHNKinetics{T<:Real} <: AbstractKinetics
    τu::T
    τv::T
    a::T
    b::T
    I0::T

    function FHNKinetics(τu::Real, τv::Real, a::Real, b::Real, I0::Real)
        τu, τv, a, b, I0 = promote(τu, τv, a, b, I0)
        new{typeof(τu)}(τu, τv, a, b, I0)
    end
end

"""
    reaction_rhs(kinetics, u, v, I_forcing) -> (du, dv)

Pointwise reaction terms (no diffusion). `I_forcing` is an additional, spatially
varying excitation (e.g. the pacemaker forcing field) added on top of `I0`.
"""
function reaction_rhs(k::FHNKinetics, u, v, I_forcing)
    du = (u .- u .^ 3 .- v .+ k.I0 .+ I_forcing) ./ k.τu
    dv = (u .+ k.a .- k.b .* v) ./ k.τv
    return du, dv
end

"""
    ReactionDiffusionModel(kinetics, Du, Dv)

Bundles a kinetics instance with the diffusion coefficients of `u` and `v`.
"""
struct ReactionDiffusionModel{K<:AbstractKinetics,T<:Real}
    kinetics::K
    Du::T
    Dv::T

    function ReactionDiffusionModel(kinetics::AbstractKinetics, Du::Real, Dv::Real)
        Du, Dv = promote(Du, Dv)
        new{typeof(kinetics),typeof(Du)}(kinetics, Du, Dv)
    end
end
