function pacemaker_field_bruteforce(positions::Matrix{Float64}, coupling::PacemakerCoupling, grid::Grid)
    field = zeros(grid.nx, grid.ny)
    inv_two_w2 = 1.0 / (2 * coupling.width^2)
    L = grid.L
    for k in 1:size(positions, 1)
        xk, yk = positions[k, 1], positions[k, 2]
        for j in 1:grid.ny, i in 1:grid.nx
            dx = grid.X[i, j] - xk
            dy = grid.Y[i, j] - yk
            dx, dy = MotilePacemaker.minimum_image(dx, dy, L)
            field[i, j] += coupling.strength * exp(-(dx^2 + dy^2) * inv_two_w2)
        end
    end
    return field
end

@testset "pacemaker coupling" begin
    grid = make_grid(1.0, 60, 60)

    @testset "localized field matches full-grid (periodic) reference for a large cutoff" begin
        coupling = PacemakerCoupling(2.0, 0.03, 8.0)  # 8 sigma: truncation error ~exp(-32), negligible
        for positions in (
            [0.5 0.5],           # interior source
            [0.01 0.5],          # near x=0 boundary -- exercises wraparound
            [0.5 0.995],         # near y=L boundary
            [0.002 0.998],       # near a corner
        )
            field = zeros(grid.nx, grid.ny)
            pacemaker_field!(field, positions, coupling, grid)
            reference = pacemaker_field_bruteforce(positions, coupling, grid)
            @test field ≈ reference atol = 1e-8
        end
    end

    @testset "small cutoff_factor actually truncates (localization has an effect)" begin
        coupling_full = PacemakerCoupling(2.0, 0.03, 8.0)
        coupling_tight = PacemakerCoupling(2.0, 0.03, 1.0)
        positions = [0.5 0.5]

        field_full = zeros(grid.nx, grid.ny)
        pacemaker_field!(field_full, positions, coupling_full, grid)
        field_tight = zeros(grid.nx, grid.ny)
        pacemaker_field!(field_tight, positions, coupling_tight, grid)

        @test sum(field_tight) < sum(field_full)
        # far from the source, the tight window must be exactly zero
        @test field_tight[1, 1] == 0.0
    end

    @testset "no double-counting when the window is clamped to (near) the whole grid" begin
        small_grid = make_grid(1.0, 10, 10)
        coupling = PacemakerCoupling(2.0, 0.5, 8.0)  # cutoff = 4.0 >> L=1, forces clamping
        positions = [0.5 0.5]

        field = zeros(small_grid.nx, small_grid.ny)
        pacemaker_field!(field, positions, coupling, small_grid)
        reference = pacemaker_field_bruteforce(positions, coupling, small_grid)

        @test field ≈ reference atol = 1e-8
    end

    @testset "multiple sources superpose" begin
        coupling = PacemakerCoupling(2.0, 0.03, 8.0)
        positions = [0.3 0.3; 0.7 0.7]
        field = zeros(grid.nx, grid.ny)
        pacemaker_field!(field, positions, coupling, grid)
        reference = pacemaker_field_bruteforce(positions, coupling, grid)
        @test field ≈ reference atol = 1e-8
    end
end
