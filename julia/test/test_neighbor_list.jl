using Random

function make_mech_nl(reciprocity; interaction_range=0.1)
    MechanicalParams(0.1, 0.01, 1.0, 0.1, interaction_range, 0.0, reciprocity)
end

@testset "neighbor list" begin
    @testset "matches brute force on random configurations" begin
        grid = make_grid(1.0, 40, 40)
        for reciprocity in (:reciprocal, :nonreciprocal), N in (5, 30, 80)
            rng = MersenneTwister(N + (reciprocity == :reciprocal ? 0 : 1000))
            positions = rand(rng, N, 2)
            u_field = -1.0 .+ 2.0 .* rand(rng, 40, 40)  # mix of signs so both gates are exercised
            mech = make_mech_nl(reciprocity)

            forces_brute = compute_forces(positions, u_field, mech, grid)

            cutoff = interaction_cutoff(mech)
            nl = NeighborList(1.0, cutoff, mech.verlet_skin, positions)
            forces_nl = compute_forces(positions, u_field, mech, grid, nl)

            @test forces_nl ≈ forces_brute atol = 1e-10
        end
    end

    @testset "degenerates correctly when cutoff is large relative to the domain" begin
        # forces a small (possibly 1-cell) cell grid, exercising the dedup logic
        grid = make_grid(1.0, 40, 40)
        rng = MersenneTwister(42)
        N = 12
        positions = rand(rng, N, 2)
        u_field = fill(1.0, 40, 40)
        mech = make_mech_nl(:reciprocal; interaction_range=0.6)  # cutoff = 5*0.6 = 3.0 >> L=1

        forces_brute = compute_forces(positions, u_field, mech, grid)
        cutoff = interaction_cutoff(mech)
        nl = NeighborList(1.0, cutoff, mech.verlet_skin, positions)
        @test nl.n_cells_x == 1  # cutoff+skin >> L, must collapse to one cell
        forces_nl = compute_forces(positions, u_field, mech, grid, nl)

        @test forces_nl ≈ forces_brute atol = 1e-10
    end

    @testset "neighbor list only contains pairs within cutoff + skin" begin
        rng = MersenneTwister(7)
        N = 50
        positions = rand(rng, N, 2)
        cutoff, skin = 0.08, 0.02
        nl = NeighborList(1.0, cutoff, skin, positions)
        r2_max = (cutoff + skin)^2

        for i in 1:N, j in nl.neighbors[i]
            dx = positions[j, 1] - positions[i, 1]
            dy = positions[j, 2] - positions[i, 2]
            dx, dy = MotilePacemaker.minimum_image(dx, dy, 1.0)
            @test dx^2 + dy^2 <= r2_max + 1e-12
        end

        # and every true-brute-force pair within cutoff+skin IS present
        for i in 1:N, j in (i+1):N
            dx = positions[j, 1] - positions[i, 1]
            dy = positions[j, 2] - positions[i, 2]
            dx, dy = MotilePacemaker.minimum_image(dx, dy, 1.0)
            if dx^2 + dy^2 <= r2_max
                @test j in nl.neighbors[i]
            end
        end
    end

    @testset "needs_rebuild triggers on the skin/2 threshold" begin
        positions = [0.2 0.2; 0.8 0.8]
        nl = NeighborList(1.0, 0.1, 0.04, positions)  # skin = 0.04 -> threshold disp = 0.02

        moved = copy(positions)
        moved[1, 1] += 0.019  # below skin/2
        @test !needs_rebuild(nl, moved, 1.0)

        moved2 = copy(positions)
        moved2[1, 1] += 0.021  # above skin/2
        @test needs_rebuild(nl, moved2, 1.0)
    end

    @testset "maybe_rebuild! updates ref_positions and neighbors" begin
        rng = MersenneTwister(3)
        N = 20
        positions = rand(rng, N, 2)
        nl = NeighborList(1.0, 0.1, 0.05, positions)

        moved = copy(positions)
        moved[1, :] .= mod.(moved[1, :] .+ 0.5, 1.0)  # big jump, definitely triggers rebuild
        rebuilt = maybe_rebuild!(nl, moved, 1.0)

        @test rebuilt
        @test nl.ref_positions ≈ moved
    end
end
