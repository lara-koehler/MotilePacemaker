import MotilePacemaker: reaction_rhs, AbstractKinetics
using Random

struct ZeroKineticsFS <: AbstractKinetics end
reaction_rhs(::ZeroKineticsFS, u, v, _I_forcing) = (zero(u), zero(v))

@testset "field_solver" begin
    @testset "single Fourier mode decays exactly per the semi-implicit update rule" begin
        nx, ny = 32, 32
        L = 1.0
        grid = make_grid(L, nx, ny)
        Du = 0.05
        model = ReactionDiffusionModel(ZeroKineticsFS(), Du, 0.0)

        # isolate one non-trivial Fourier mode
        mode = (3, 1)
        u_hat0 = zeros(ComplexF64, nx, ny)
        u_hat0[mode...] = 1.0 + 0.0im
        # keep the field real: also populate the conjugate-symmetric partner
        ci = mod(nx - (mode[1] - 1), nx) + 1
        cj = mod(ny - (mode[2] - 1), ny) + 1
        u_hat0[ci, cj] = 1.0 + 0.0im

        u = real.(grid.ifft_plan * u_hat0)
        v = zeros(nx, ny)
        I_forcing = zeros(nx, ny)

        dt = 0.001
        n_steps = 25
        rng = MersenneTwister(1)
        for _ in 1:n_steps
            step_fields!(u, v, model, I_forcing, dt, grid, NoNoise(), rng)
        end

        u_hat_final = grid.fft_plan * complex.(u)
        k2_mode = grid.k2[mode...]
        expected = 1.0 / (1 + dt * Du * k2_mode)^n_steps

        @test isapprox(abs(u_hat_final[mode...]), expected; rtol=1e-8)
        @test all(v .== 0.0)
    end

    @testset "zero diffusion, zero reaction: field is unchanged" begin
        nx, ny = 16, 16
        grid = make_grid(1.0, nx, ny)
        model = ReactionDiffusionModel(ZeroKineticsFS(), 0.0, 0.0)
        rng = MersenneTwister(2)
        u0, v0 = initialize_field(nx, ny, rng)
        u, v = copy(u0), copy(v0)
        I_forcing = zeros(nx, ny)

        step_fields!(u, v, model, I_forcing, 0.01, grid, NoNoise(), rng)

        @test u ≈ u0 atol = 1e-10
        @test v ≈ v0 atol = 1e-10
    end
end
