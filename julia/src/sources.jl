"""
    initialize_sources_line(n_sources, L, spacing; x0=0.5, y0=0.5) -> positions

`n_sources` sources centered at `(x0*L, y0*L)`, evenly spaced by `spacing`
along x, matching the prototype's `initialize_sources_line`.
"""
function initialize_sources_line(n_sources::Int, L::Real, spacing::Real; x0::Real=0.5, y0::Real=0.5)
    positions = zeros(Float64, n_sources, 2)
    positions[:, 2] .= y0 * L
    center = x0 * L
    offsets = ((1:n_sources) .- (n_sources + 1) / 2) .* spacing
    positions[:, 1] .= mod.(center .+ offsets, L)
    return positions
end

"""
    initialize_sources_random(n_sources, L, rng) -> positions
"""
function initialize_sources_random(n_sources::Int, L::Real, rng)
    return rand(rng, n_sources, 2) .* L
end

"""
    initialize_sources_grid(n_rows, n_cols, L, spacing; x0=0.5, y0=0.5) -> positions

`n_rows x n_cols` sources on a regular 2D lattice, centered at `(x0*L, y0*L)`
and spaced by `spacing` in both x and y (row-major order: row 1 first, its
columns left to right, then row 2, ...).
"""
function initialize_sources_grid(n_rows::Int, n_cols::Int, L::Real, spacing::Real;
                                  x0::Real=0.5, y0::Real=0.5)
    positions = zeros(Float64, n_rows * n_cols, 2)
    cx = x0 * L
    cy = y0 * L
    col_offsets = ((1:n_cols) .- (n_cols + 1) / 2) .* spacing
    row_offsets = ((1:n_rows) .- (n_rows + 1) / 2) .* spacing

    idx = 1
    for r in 1:n_rows, c in 1:n_cols
        positions[idx, 1] = mod(cx + col_offsets[c], L)
        positions[idx, 2] = mod(cy + row_offsets[r], L)
        idx += 1
    end
    return positions
end

"""
    initialize_sources_grid_fill(n_rows, n_cols, L) -> positions

`n_rows x n_cols` sources evenly tiling the *entire* periodic domain --
spacing `L/n_cols` in x and `L/n_rows` in y, one source centered in each
tile -- unlike [`initialize_sources_grid`](@ref), which places a compact
block of a given spacing centered at a chosen point. Use this when the source
count is large enough that a centered block would leave most of the domain
empty (e.g. hundreds of sources).
"""
function initialize_sources_grid_fill(n_rows::Int, n_cols::Int, L::Real)
    spacing_x = L / n_cols
    spacing_y = L / n_rows
    positions = zeros(Float64, n_rows * n_cols, 2)

    idx = 1
    for r in 0:(n_rows-1), c in 0:(n_cols-1)
        positions[idx, 1] = (c + 0.5) * spacing_x
        positions[idx, 2] = (r + 0.5) * spacing_y
        idx += 1
    end
    return positions
end

"""
    initialize_positions(sourcescfg::Dict, L::Real, rng) -> (positions, n_sources, spacing_for_diagnostics)

Build initial source positions from a config's raw `[sources]` table,
dispatching on `sourcescfg["arrangement"]` (`"line"`, `"random"`, `"grid"`, or
`"grid_fill"`). `spacing_for_diagnostics` is a representative spacing passed
on to [`chi_gamma`](@ref) for logging -- not used in the dynamics itself.

Shared by `run_minimal.jl` and `run_scan.jl` so the arrangement dispatch only
lives in one place.
"""
function initialize_positions(sourcescfg::Dict, L::Real, rng)
    arrangement = get(sourcescfg, "arrangement", "line")
    if arrangement == "line"
        spacing = sourcescfg["line_spacing"]
        n = sourcescfg["n_sources"]
        pos = initialize_sources_line(
            n, L, spacing;
            x0=get(sourcescfg, "line_x0", 0.5), y0=get(sourcescfg, "line_y0", 0.5),
        )
        return pos, n, spacing
    elseif arrangement == "random"
        n = sourcescfg["n_sources"]
        pos = initialize_sources_random(n, L, rng)
        return pos, n, get(sourcescfg, "line_spacing", 0.1)
    elseif arrangement == "grid"
        spacing = sourcescfg["grid_spacing"]
        rows, cols = sourcescfg["grid_rows"], sourcescfg["grid_cols"]
        pos = initialize_sources_grid(
            rows, cols, L, spacing;
            x0=get(sourcescfg, "grid_x0", 0.5), y0=get(sourcescfg, "grid_y0", 0.5),
        )
        return pos, rows * cols, spacing
    elseif arrangement == "grid_fill"
        rows, cols = sourcescfg["grid_rows"], sourcescfg["grid_cols"]
        pos = initialize_sources_grid_fill(rows, cols, L)
        return pos, rows * cols, min(L / cols, L / rows)
    else
        error("Unknown source arrangement: $(arrangement)")
    end
end

"""
    MechanicalParams(mobility, sigma, epsilon_LJ, interaction_strength,
                      interaction_range, u_threshold, reciprocity;
                      rcut_far_factor=5.0, rcut_near_factor=0.05,
                      verlet_skin=0.5*interaction_range, max_force=Inf)

`reciprocity` is `:nonreciprocal` (default in the model as literally written:
the force felt by nucleus i is gated by the field at *xi*, independently of
what nucleus j feels) or `:reciprocal` (gated by the pair-symmetrized field
`(u(xi)+u(xj))/2`, giving standard equal-and-opposite pair forces). WCA
repulsion does not depend on the field and is always reciprocal.

`rcut_far_factor`/`rcut_near_factor` set the field-gated force's truncation
distances (as multiples of `interaction_range`) -- kept here rather than as
bare defaults on `gated_force_components` so [`interaction_cutoff`](@ref) (and
therefore [`NeighborList`](@ref)) reads the true cutoff from a single source
of truth. `verlet_skin` is the extra buffer radius used when building a
`NeighborList` for this `mech`.

`max_force` caps each pair's combined WCA+gated force magnitude (default:
uncapped). This matters because WCA repulsion diverges steeply at short
range; with the explicit-Euler position update, a stiff enough pairwise force
can produce a single-step displacement large enough to "tunnel" straight
through the repulsive wall instead of being smoothly decelerated by it,
occasionally landing two sources within the `dist2 < 1e-24`
division-by-zero guard used in `wca_force_components`/`gated_force_components`
-- which then permanently zeroes their mutual force, leaving them to drift
together (bit-identical) for the rest of the simulation. `load_config`
computes a sensible automatic default from `dt`/`mobility`/`sigma` when not
set explicitly in a TOML config's `[mechanics]` table.
"""
struct MechanicalParams{T<:Real}
    mobility::T
    sigma::T
    epsilon_LJ::T
    interaction_strength::T
    interaction_range::T
    u_threshold::T
    reciprocity::Symbol
    rcut_far_factor::T
    rcut_near_factor::T
    verlet_skin::T
    max_force::T

    function MechanicalParams(mobility::Real, sigma::Real, epsilon_LJ::Real,
                               interaction_strength::Real, interaction_range::Real,
                               u_threshold::Real, reciprocity::Symbol;
                               rcut_far_factor::Real=5.0, rcut_near_factor::Real=0.05,
                               verlet_skin::Real=0.5 * interaction_range, max_force::Real=Inf)
        reciprocity in (:reciprocal, :nonreciprocal) ||
            error("reciprocity must be :reciprocal or :nonreciprocal, got $(reciprocity)")
        mobility, sigma, epsilon_LJ, interaction_strength, interaction_range, u_threshold,
            rcut_far_factor, rcut_near_factor, verlet_skin, max_force =
            promote(mobility, sigma, epsilon_LJ, interaction_strength, interaction_range,
                    u_threshold, rcut_far_factor, rcut_near_factor, verlet_skin, max_force)
        new{typeof(mobility)}(mobility, sigma, epsilon_LJ, interaction_strength,
                               interaction_range, u_threshold, reciprocity,
                               rcut_far_factor, rcut_near_factor, verlet_skin, max_force)
    end
end

"""
    interaction_cutoff(mech) -> Float64

The largest distance at which any mechanical force (WCA or field-gated) on
`mech` is nonzero -- the radius a [`NeighborList`](@ref) must use to avoid
missing interactions.
"""
interaction_cutoff(mech::MechanicalParams) =
    max(2^(1 / 6) * mech.sigma, mech.rcut_far_factor * mech.interaction_range)

"""
    wca_force_components(dx, dy, epsilon, sigma) -> (fx, fy)

Weeks-Chandler-Andersen short-range repulsion along the separation `(dx, dy)`,
vanishing beyond the cutoff `2^(1/6) * sigma`.
"""
function wca_force_components(dx::Float64, dy::Float64, epsilon::Float64, sigma::Float64)
    dist2 = dx^2 + dy^2
    dist2 < 1e-24 && return 0.0, 0.0

    rc2 = (2^(1 / 6) * sigma)^2
    dist2 >= rc2 && return 0.0, 0.0

    inv_r2 = 1.0 / dist2
    inv_r6 = sigma^6 * inv_r2^3
    inv_r12 = inv_r6^2
    f_over_r = 24 * epsilon * inv_r2 * (2 * inv_r12 - inv_r6)
    return f_over_r * dx, f_over_r * dy
end

"""
    gated_force_components(dx, dy, strength, r0, sign) -> (fx, fy)

Exponentially decaying, finite-range force along `(dx, dy)`, positive `sign`
giving attraction and negative `sign` giving repulsion. Truncated at short and
long distances (fractions of `r0`).
"""
function gated_force_components(dx::Float64, dy::Float64, strength::Float64, r0::Float64,
                                 sign::Float64; rcut_far_factor::Float64=5.0,
                                 rcut_near_factor::Float64=0.05)
    dist2 = dx^2 + dy^2
    dist2 < 1e-24 && return 0.0, 0.0

    rcut_far2 = (rcut_far_factor * r0)^2
    rcut_near2 = (rcut_near_factor * r0)^2
    (dist2 >= rcut_far2 || dist2 <= rcut_near2) && return 0.0, 0.0

    dist = sqrt(dist2)
    mag = sign * strength * exp(-dist / r0) / dist
    return mag * dx, mag * dy
end

"""
    cap_force_magnitude(fx, fy, max_force) -> (fx, fy)

Clamp the force vector `(fx, fy)` to magnitude at most `max_force`, preserving
direction. See [`MechanicalParams`](@ref)'s `max_force` docs for why this
matters (preventing explicit-Euler "tunneling" through the stiff WCA wall).
"""
@inline function cap_force_magnitude(fx::Float64, fy::Float64, max_force::Float64)
    isinf(max_force) && return fx, fy
    mag2 = fx^2 + fy^2
    mag2 <= max_force^2 && return fx, fy
    scale = max_force / sqrt(mag2)
    return fx * scale, fy * scale
end

"""
    pair_raw_force(dx, dy, mech, sign) -> (fx, fy)

Sum of WCA repulsion and the field-gated interaction along `(dx, dy) = pos_j - pos_i`,
capped to `mech.max_force` in magnitude.
"""
function pair_raw_force(dx::Float64, dy::Float64, mech::MechanicalParams, sign::Float64)
    fx, fy = wca_force_components(dx, dy, mech.epsilon_LJ, mech.sigma)
    gx, gy = gated_force_components(dx, dy, mech.interaction_strength, mech.interaction_range, sign;
                                     rcut_far_factor=mech.rcut_far_factor,
                                     rcut_near_factor=mech.rcut_near_factor)
    return cap_force_magnitude(fx + gx, fy + gy, mech.max_force)
end

@inline function minimum_image(dx::Float64, dy::Float64, L::Float64)
    return dx - L * round(dx / L), dy - L * round(dy / L)
end

@inline function local_field_index(pos_component::Float64, L::Float64, n::Int)
    return mod(round(Int, pos_component / L * n), n) + 1
end

"""
    local_field_values(positions, u_field, grid) -> Vector{Float64}

Value of `u_field` at the grid cell nearest to each source position.
"""
function local_field_values(positions::Matrix{Float64}, u_field::Matrix{Float64}, grid::Grid)
    N = size(positions, 1)
    values = Vector{Float64}(undef, N)
    for i in 1:N
        ix = local_field_index(positions[i, 1], grid.L, grid.nx)
        iy = local_field_index(positions[i, 2], grid.L, grid.ny)
        values[i] = u_field[ix, iy]
    end
    return values
end

"""
    accumulate_pair_force!(forces, positions, local_u, i, j, mech, L)

Add the (i, j) pair's mechanical force contribution into `forces[i, :]` and
`forces[j, :]`, dispatching on `mech.reciprocity`. Shared by the brute-force
[`compute_forces`](@ref) and the [`NeighborList`](@ref)-accelerated one, so
the two can never drift apart.
"""
@inline function accumulate_pair_force!(forces::Matrix{Float64}, positions::Matrix{Float64},
                                         local_u::Vector{Float64}, i::Int, j::Int,
                                         mech::MechanicalParams, L::Float64)
    dx = positions[j, 1] - positions[i, 1]
    dy = positions[j, 2] - positions[i, 2]
    dx, dy = minimum_image(dx, dy, L)

    if mech.reciprocity == :reciprocal
        gate = 0.5 * (local_u[i] + local_u[j]) > mech.u_threshold ? 1.0 : -1.0
        fx, fy = pair_raw_force(dx, dy, mech, gate)
        forces[i, 1] -= fx
        forces[i, 2] -= fy
        forces[j, 1] += fx
        forces[j, 2] += fy
    else # :nonreciprocal
        sign_i = local_u[i] > mech.u_threshold ? 1.0 : -1.0
        sign_j = local_u[j] > mech.u_threshold ? 1.0 : -1.0

        fx_i, fy_i = pair_raw_force(dx, dy, mech, sign_i)
        forces[i, 1] -= fx_i
        forces[i, 2] -= fy_i

        fx_j, fy_j = pair_raw_force(-dx, -dy, mech, sign_j)
        forces[j, 1] -= fx_j
        forces[j, 2] -= fy_j
    end
    return nothing
end

"""
    compute_forces(positions, u_field, mech, grid) -> Matrix{Float64}

Pairwise mechanical forces (WCA + field-gated) on every source, under periodic
(minimum-image) boundary conditions -- brute-force all-pairs, O(N^2). Dispatches
on `mech.reciprocity`. See the [`NeighborList`](@ref)-accelerated method for
larger `N`; both share [`accumulate_pair_force!`](@ref) so they agree exactly.
"""
function compute_forces(positions::Matrix{Float64}, u_field::Matrix{Float64},
                         mech::MechanicalParams, grid::Grid)
    N = size(positions, 1)
    forces = zeros(Float64, N, 2)
    L = grid.L
    local_u = local_field_values(positions, u_field, grid)

    @inbounds for i in 1:N, j in (i+1):N
        accumulate_pair_force!(forces, positions, local_u, i, j, mech, L)
    end

    return forces
end

"""
    update_positions!(positions, u_field, dt, mech, grid, position_noise, rng)

Overdamped Euler step: `x += dt * mobility * F(x)`, plus optional positional
noise, then wraps into the periodic domain `[0, L)`. Uses the brute-force
`compute_forces` -- O(N^2), fine for small `N`; see the `NeighborList` method
for larger source counts.
"""
function update_positions!(positions::Matrix{Float64}, u_field::Matrix{Float64}, dt::Real,
                            mech::MechanicalParams, grid::Grid, position_noise, rng)
    forces = compute_forces(positions, u_field, mech, grid)
    positions .+= dt .* mech.mobility .* forces
    apply_position_noise!(positions, position_noise, dt, rng)
    positions .= mod.(positions, grid.L)
    return positions
end
