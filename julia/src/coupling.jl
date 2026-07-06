"""
    PacemakerCoupling(strength, width, cutoff_factor=4.0)

Each source contributes a Gaussian bump of the given `strength` and `width` to
the excitation-parameter forcing field `I(x) = I0 + sum_i strength * exp(-|x-xi|^2/2w^2)`.

`cutoff_factor` truncates each bump beyond `cutoff_factor * width` (default 4,
i.e. `exp(-8) ~= 3e-4` of the peak amplitude -- numerically negligible), so
`pacemaker_field!` only evaluates a local window around each source instead of
the whole grid, which is the dominant per-step cost at low-to-moderate source
counts.
"""
struct PacemakerCoupling{T<:Real}
    strength::T
    width::T
    cutoff_factor::T

    function PacemakerCoupling(strength::Real, width::Real, cutoff_factor::Real=4.0)
        strength, width, cutoff_factor = promote(strength, width, cutoff_factor)
        new{typeof(strength)}(strength, width, cutoff_factor)
    end
end

"""
    pacemaker_field!(field, positions, coupling, grid)

Fill `field` in place with the sum of Gaussian pacemaker bumps centered at
`positions`, under periodic (minimum-image) boundary conditions, evaluated
only within `coupling.cutoff_factor * coupling.width` of each source rather
than over the whole grid.

Note: earlier versions of this function (and the Python prototype it was
ported from) did *not* apply periodic wrap here, unlike the mechanical forces
-- a source near one edge of the domain would not influence the field near the
opposite edge, even though the field itself is periodic. Restricting to a
local window forces this to be handled explicitly, so it is fixed here too.
"""
function pacemaker_field!(field::Matrix{Float64}, positions::Matrix{Float64},
                           coupling::PacemakerCoupling, grid::Grid)
    fill!(field, 0.0)
    N = size(positions, 1)
    inv_two_w2 = 1.0 / (2 * coupling.width^2)
    r_cut = coupling.cutoff_factor * coupling.width
    # capped at nx-1 (resp. ny-1): any run of >= nx consecutive integers
    # already covers every residue mod nx at least once, so more offsets
    # than that add nothing but duplicates for `unique` to discard. Capping
    # any tighter (e.g. at nx/2, as an earlier version of this function did)
    # can leave a whole row/column uncovered when nx is even -- see
    # test_coupling.jl's "clamped to (near) the whole grid" case.
    half_win_x = min(ceil(Int, r_cut / grid.dx), grid.nx - 1)
    half_win_y = min(ceil(Int, r_cut / grid.dy), grid.ny - 1)
    L = grid.L

    @inbounds for k in 1:N
        xk = positions[k, 1]
        yk = positions[k, 2]
        ic = round(Int, xk / grid.dx)
        jc = round(Int, yk / grid.dy)

        ix_list = unique(mod.(ic .+ (-half_win_x:half_win_x), grid.nx) .+ 1)
        iy_list = unique(mod.(jc .+ (-half_win_y:half_win_y), grid.ny) .+ 1)

        # the Gaussian is separable: exp(-(dx^2+dy^2)/2w^2) = exp(-dx^2/2w^2)*exp(-dy^2/2w^2),
        # so precompute the 1D factors once instead of calling exp() per grid point
        gx = Vector{Float64}(undef, length(ix_list))
        for (n, i) in enumerate(ix_list)
            dx = grid.X[i, 1] - xk
            dx -= L * round(dx / L)
            gx[n] = exp(-dx^2 * inv_two_w2)
        end
        gy = Vector{Float64}(undef, length(iy_list))
        for (m, j) in enumerate(iy_list)
            dy = grid.Y[1, j] - yk
            dy -= L * round(dy / L)
            gy[m] = exp(-dy^2 * inv_two_w2)
        end

        for (m, j) in enumerate(iy_list), (n, i) in enumerate(ix_list)
            field[i, j] += coupling.strength * gx[n] * gy[m]
        end
    end
    return field
end
