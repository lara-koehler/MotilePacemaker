"""
    NeighborList

Cell list + Verlet list acceleration structure for the pairwise mechanical
forces, so `compute_forces` doesn't have to check all O(N^2) pairs at every
step.

Cells are sized `cutoff + skin` so that, under periodic boundary conditions,
any pair within `cutoff` of each other is guaranteed to be found by checking
a particle's own cell and its (deduplicated) 3x3 neighborhood of cells -- this
holds regardless of how few cells the domain has (it degenerates gracefully
to an all-pairs search when the domain is small relative to the cutoff).

`neighbors[i]` stores only `j > i` (matching the convention of the brute-force
`for i in 1:N, j in (i+1):N` loop it replaces), rebuilt via `rebuild!`. Between
rebuilds, `ref_positions` tracks where each particle was *at* the last
rebuild; `needs_rebuild` compares the largest single-particle displacement
since then against `skin` -- if two particles have each drifted up to that far
towards each other, a pair that was previously farther than `cutoff + skin`
apart could now be within `cutoff`, so a rebuild is triggered before that can
silently go undetected.
"""
mutable struct NeighborList
    cutoff::Float64
    skin::Float64
    cell_size::Float64
    n_cells_x::Int
    n_cells_y::Int
    cells::Matrix{Vector{Int}}
    neighbors::Vector{Vector{Int}}
    ref_positions::Matrix{Float64}
end

@inline function periodic_cell_neighbors(c::Int, n_cells::Int)
    return unique(mod.((c - 1) .+ (-1:1), n_cells) .+ 1)
end

@inline function cell_of(pos_component::Float64, cell_size::Float64, n_cells::Int)
    return clamp(floor(Int, pos_component / cell_size), 0, n_cells - 1) + 1
end

"""
    NeighborList(L, cutoff, skin, positions) -> NeighborList

Build a neighbor list for a periodic domain of side `L`, given the force
`cutoff` (see [`interaction_cutoff`](@ref)), a `skin` buffer, and the current
source `positions` (`N x 2`).
"""
function NeighborList(L::Real, cutoff::Real, skin::Real, positions::Matrix{Float64})
    cell_size = cutoff + skin
    n_cells = max(floor(Int, L / cell_size), 1)
    N = size(positions, 1)

    nl = NeighborList(
        Float64(cutoff), Float64(skin), L / n_cells, n_cells, n_cells,
        [Int[] for _ in 1:n_cells, _ in 1:n_cells],
        [Int[] for _ in 1:N],
        zeros(Float64, N, 2),
    )
    rebuild!(nl, positions, L)
    return nl
end

"""
    rebuild!(nl, positions, L)

Recompute the cell assignments and Verlet list from scratch, and reset the
displacement-tracking reference to `positions`.
"""
function rebuild!(nl::NeighborList, positions::Matrix{Float64}, L::Real)
    N = size(positions, 1)

    for cy in 1:nl.n_cells_y, cx in 1:nl.n_cells_x
        empty!(nl.cells[cx, cy])
    end
    for i in 1:N
        empty!(nl.neighbors[i])
    end

    cell_x = Vector{Int}(undef, N)
    cell_y = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        cx = cell_of(positions[i, 1], nl.cell_size, nl.n_cells_x)
        cy = cell_of(positions[i, 2], nl.cell_size, nl.n_cells_y)
        push!(nl.cells[cx, cy], i)
        cell_x[i], cell_y[i] = cx, cy
    end

    r2_max = (nl.cutoff + nl.skin)^2
    @inbounds for cy in 1:nl.n_cells_y, cx in 1:nl.n_cells_x
        members = nl.cells[cx, cy]
        isempty(members) && continue

        ncxs = periodic_cell_neighbors(cx, nl.n_cells_x)
        ncys = periodic_cell_neighbors(cy, nl.n_cells_y)

        for i in members
            xi, yi = positions[i, 1], positions[i, 2]
            for ncy in ncys, ncx in ncxs
                for j in nl.cells[ncx, ncy]
                    j <= i && continue
                    dx = positions[j, 1] - xi
                    dy = positions[j, 2] - yi
                    dx, dy = minimum_image(dx, dy, Float64(L))
                    if dx * dx + dy * dy <= r2_max
                        push!(nl.neighbors[i], j)
                    end
                end
            end
        end
    end

    nl.ref_positions .= positions
    return nl
end

"""
    needs_rebuild(nl, positions, L) -> Bool

`true` if any particle has moved more than `skin/2` since the last rebuild --
i.e. two particles could together have closed the `skin` buffer distance.
"""
function needs_rebuild(nl::NeighborList, positions::Matrix{Float64}, L::Real)
    N = size(positions, 1)
    max_disp2 = 0.0
    @inbounds for i in 1:N
        dx = positions[i, 1] - nl.ref_positions[i, 1]
        dy = positions[i, 2] - nl.ref_positions[i, 2]
        dx, dy = minimum_image(dx, dy, Float64(L))
        d2 = dx * dx + dy * dy
        d2 > max_disp2 && (max_disp2 = d2)
    end
    return 2 * sqrt(max_disp2) > nl.skin
end

"""
    maybe_rebuild!(nl, positions, L) -> Bool

Rebuild `nl` in place if [`needs_rebuild`](@ref) says so. Returns whether a
rebuild happened.
"""
function maybe_rebuild!(nl::NeighborList, positions::Matrix{Float64}, L::Real)
    if needs_rebuild(nl, positions, L)
        rebuild!(nl, positions, L)
        return true
    end
    return false
end

"""
    compute_forces(positions, u_field, mech, grid, neighbor_list) -> Matrix{Float64}

Same result as the brute-force `compute_forces`, but only checking pairs in
`neighbor_list.neighbors` -- O(N) instead of O(N^2) once the source density is
roughly uniform. `neighbor_list` is not mutated here; call `maybe_rebuild!`
(done automatically by the `update_positions!` method below) to keep it valid
as particles move.
"""
function compute_forces(positions::Matrix{Float64}, u_field::Matrix{Float64},
                         mech::MechanicalParams, grid::Grid, neighbor_list::NeighborList)
    N = size(positions, 1)
    forces = zeros(Float64, N, 2)
    L = grid.L
    local_u = local_field_values(positions, u_field, grid)

    @inbounds for i in 1:N
        for j in neighbor_list.neighbors[i]
            accumulate_pair_force!(forces, positions, local_u, i, j, mech, L)
        end
    end

    return forces
end

"""
    update_positions!(positions, u_field, dt, mech, grid, position_noise, rng, neighbor_list)

Same as the brute-force `update_positions!`, but using `neighbor_list` to
compute forces, and rebuilding it afterwards if particles have drifted enough
to risk missing a newly-close pair (see [`maybe_rebuild!`](@ref)).
"""
function update_positions!(positions::Matrix{Float64}, u_field::Matrix{Float64}, dt::Real,
                            mech::MechanicalParams, grid::Grid, position_noise, rng,
                            neighbor_list::NeighborList)
    forces = compute_forces(positions, u_field, mech, grid, neighbor_list)
    positions .+= dt .* mech.mobility .* forces
    apply_position_noise!(positions, position_noise, dt, rng)
    positions .= mod.(positions, grid.L)
    maybe_rebuild!(neighbor_list, positions, grid.L)
    return positions
end
