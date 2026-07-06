@testset "forces" begin
    @testset "WCA repulsion" begin
        # deep inside the repulsive core (dist < sigma): kernel points along +(dx,dy)
        fx, fy = MotilePacemaker.wca_force_components(0.005, 0.0, 1.0, 0.01)
        @test fx > 0.0
        @test fy == 0.0

        # beyond cutoff 2^(1/6)*sigma, force vanishes
        fx_far, fy_far = MotilePacemaker.wca_force_components(1.0, 0.0, 1.0, 0.01)
        @test fx_far == 0.0
        @test fy_far == 0.0
    end

    @testset "gated force sign" begin
        strength, r0 = 0.1, 0.2
        fx_attr, _ = MotilePacemaker.gated_force_components(0.1, 0.0, strength, r0, 1.0)
        fx_rep, _ = MotilePacemaker.gated_force_components(0.1, 0.0, strength, r0, -1.0)
        @test fx_attr ≈ -fx_rep
        @test sign(fx_attr) != sign(fx_rep)

        # zero outside [near, far] cutoffs
        fx_near, _ = MotilePacemaker.gated_force_components(1e-4 * r0, 0.0, strength, r0, 1.0)
        @test fx_near == 0.0
        fx_far, _ = MotilePacemaker.gated_force_components(10 * r0, 0.0, strength, r0, 1.0)
        @test fx_far == 0.0
    end

    function make_mech(reciprocity)
        MechanicalParams(0.1, 0.01, 1.0, 0.1, 0.2, 0.0, reciprocity)
    end

    @testset "reciprocal mode: equal and opposite forces" begin
        mech = make_mech(:reciprocal)
        grid = make_grid(1.0, 50, 50)
        positions = [0.4 0.5; 0.55 0.5]

        u_field = fill(-2.0, 50, 50)  # both particles see u < threshold by default
        # give the two particles very different local field values directly
        ix1 = MotilePacemaker.local_field_index(positions[1, 1], grid.L, grid.nx)
        iy1 = MotilePacemaker.local_field_index(positions[1, 2], grid.L, grid.ny)
        ix2 = MotilePacemaker.local_field_index(positions[2, 1], grid.L, grid.nx)
        iy2 = MotilePacemaker.local_field_index(positions[2, 2], grid.L, grid.ny)
        u_field[ix1, iy1] = 5.0
        u_field[ix2, iy2] = -5.0

        forces = compute_forces(positions, u_field, mech, grid)
        @test forces[1, :] ≈ -forces[2, :]
    end

    @testset "non-reciprocal mode: forces need not be equal and opposite" begin
        mech = make_mech(:nonreciprocal)
        grid = make_grid(1.0, 50, 50)
        positions = [0.4 0.5; 0.55 0.5]

        u_field = fill(-2.0, 50, 50)
        ix1 = MotilePacemaker.local_field_index(positions[1, 1], grid.L, grid.nx)
        iy1 = MotilePacemaker.local_field_index(positions[1, 2], grid.L, grid.ny)
        ix2 = MotilePacemaker.local_field_index(positions[2, 1], grid.L, grid.nx)
        iy2 = MotilePacemaker.local_field_index(positions[2, 2], grid.L, grid.ny)
        u_field[ix1, iy1] = 5.0   # particle 1 sees high u -> attractive gate
        u_field[ix2, iy2] = -5.0  # particle 2 sees low u -> repulsive gate

        forces = compute_forces(positions, u_field, mech, grid)
        @test !(forces[1, :] ≈ -forces[2, :])
    end

    @testset "non-reciprocal mode reduces to reciprocal when field is uniform" begin
        grid = make_grid(1.0, 50, 50)
        positions = [0.4 0.5; 0.55 0.5]
        u_field = fill(5.0, 50, 50)  # same value everywhere: both particles see the same gate

        forces_recip = compute_forces(positions, u_field, make_mech(:reciprocal), grid)
        forces_nonrecip = compute_forces(positions, u_field, make_mech(:nonreciprocal), grid)
        @test forces_recip ≈ forces_nonrecip
        @test forces_nonrecip[1, :] ≈ -forces_nonrecip[2, :]
    end

    @testset "cap_force_magnitude" begin
        # under the cap: unchanged
        fx, fy = MotilePacemaker.cap_force_magnitude(3.0, 4.0, 10.0)
        @test (fx, fy) == (3.0, 4.0)

        # over the cap: scaled down to exactly max_force, direction preserved
        fx, fy = MotilePacemaker.cap_force_magnitude(3.0, 4.0, 2.0)
        @test isapprox(hypot(fx, fy), 2.0)
        @test isapprox(fy / fx, 4.0 / 3.0)

        # Inf means uncapped, regardless of magnitude
        fx, fy = MotilePacemaker.cap_force_magnitude(1e10, 0.0, Inf)
        @test (fx, fy) == (1e10, 0.0)
    end

    @testset "max_force prevents an explicit-Euler pair from tunneling past each other" begin
        # deep inside the WCA core, the raw force is astronomically large;
        # capped, a single Euler step should move sources by at most ~max_force*dt*mobility
        mech_uncapped = MechanicalParams(0.5, 0.02, 1.0, 0.5, 0.2, 0.0, :reciprocal)
        mech_capped = MechanicalParams(0.5, 0.02, 1.0, 0.5, 0.2, 0.0, :reciprocal; max_force=20.0)

        dx, dy = 0.001, 0.0  # well inside sigma=0.02 -> huge raw WCA force
        fx_uncapped, _ = MotilePacemaker.pair_raw_force(dx, dy, mech_uncapped, 1.0)
        fx_capped, _ = MotilePacemaker.pair_raw_force(dx, dy, mech_capped, 1.0)

        @test abs(fx_uncapped) > 1000 * abs(fx_capped)  # uncapped really is huge
        @test isapprox(abs(fx_capped), 20.0; atol=1e-8)

        dt = 0.001
        step_capped = dt * mech_capped.mobility * abs(fx_capped)
        @test step_capped <= 0.5 * mech_capped.sigma + 1e-12  # half the WCA radius, by construction
    end
end
