using Random

@testset "source initial conditions" begin
    @testset "initialize_sources_grid" begin
        L = 1.0
        rows, cols = 3, 4
        spacing = 0.1
        positions = initialize_sources_grid(rows, cols, L, spacing; x0=0.5, y0=0.5)

        @test size(positions) == (rows * cols, 2)
        @test all(0.0 .<= positions .< L)  # wrapped into the periodic domain

        # unique x values (one per column) and y values (one per row), up to
        # floating point, confirming a genuine 2D lattice rather than a line
        @test length(unique(round.(positions[:, 1], digits=8))) == cols
        @test length(unique(round.(positions[:, 2], digits=8))) == rows

        # centered on (x0*L, y0*L)
        @test isapprox(sum(positions[:, 1]) / size(positions, 1), 0.5 * L; atol=1e-8)
        @test isapprox(sum(positions[:, 2]) / size(positions, 1), 0.5 * L; atol=1e-8)
    end

    @testset "initialize_sources_grid reduces to a row when n_rows == 1" begin
        positions = initialize_sources_grid(1, 5, 1.0, 0.1)
        @test all(positions[:, 2] .== 0.5)
        @test length(unique(positions[:, 1])) == 5
    end

    @testset "initialize_sources_grid_fill spans the whole domain" begin
        L = 1.0
        rows, cols = 5, 100
        positions = initialize_sources_grid_fill(rows, cols, L)

        @test size(positions) == (rows * cols, 2)
        @test all(0.0 .<= positions .< L)

        # one distinct value per column/row, and they tile [0, L) evenly
        xs = sort(unique(round.(positions[:, 1], digits=8)))
        ys = sort(unique(round.(positions[:, 2], digits=8)))
        @test length(xs) == cols
        @test length(ys) == rows
        @test isapprox(xs[1], 0.5 * (L / cols); atol=1e-8)        # first tile center
        @test isapprox(xs[end], L - 0.5 * (L / cols); atol=1e-8)  # last tile center
        @test isapprox(ys[1], 0.5 * (L / rows); atol=1e-8)
        @test isapprox(ys[end], L - 0.5 * (L / rows); atol=1e-8)

        # unlike initialize_sources_grid, sources actually fill the domain,
        # not a small block centered on one point
        @test maximum(positions[:, 1]) - minimum(positions[:, 1]) > 0.9 * L
        @test maximum(positions[:, 2]) - minimum(positions[:, 2]) > 0.7 * L
    end

    @testset "initialize_sources_grid_fill has no duplicate positions" begin
        positions = initialize_sources_grid_fill(20, 25, 1.0)
        @test size(positions, 1) == 500
        @test size(unique(positions, dims=1), 1) == 500
    end

    @testset "initialize_positions dispatches on arrangement" begin
        rng = MersenneTwister(1)
        L = 1.0

        pos, n, spacing = initialize_positions(
            Dict("arrangement" => "line", "n_sources" => 6, "line_spacing" => 0.1), L, rng,
        )
        @test size(pos) == (6, 2)
        @test n == 6
        @test spacing == 0.1

        pos, n, _ = initialize_positions(Dict("arrangement" => "random", "n_sources" => 8), L, rng)
        @test size(pos) == (8, 2)
        @test n == 8

        pos, n, spacing = initialize_positions(
            Dict("arrangement" => "grid", "grid_rows" => 2, "grid_cols" => 3, "grid_spacing" => 0.1), L, rng,
        )
        @test size(pos) == (6, 2)
        @test n == 6
        @test spacing == 0.1

        pos, n, _ = initialize_positions(
            Dict("arrangement" => "grid_fill", "grid_rows" => 4, "grid_cols" => 5), L, rng,
        )
        @test size(pos) == (20, 2)
        @test n == 20

        @test_throws ErrorException initialize_positions(Dict("arrangement" => "bogus"), L, rng)
    end
end
