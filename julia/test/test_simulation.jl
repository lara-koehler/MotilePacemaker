@testset "simulation" begin
    @testset "source_save_every subsamples trajectories consistently with a full-resolution run" begin
        kinetics = FHNKinetics(1.0, 12.5, 0.2, 0.2, -1.6)
        model = ReactionDiffusionModel(kinetics, 0.001, 0.0)
        coupling = PacemakerCoupling(2.0, 0.05)
        mech = MechanicalParams(0.1, 0.01, 1.0, 0.1, 0.2, 0.0, :reciprocal)

        n_steps = 40
        every = 5

        base_kwargs = (L=1.0, nx=20, ny=20, dt=0.01, n_steps=n_steps,
                        save_every=n_steps, model=model, coupling=coupling, mech=mech, seed=3)

        cfg_full = SimulationConfig(; base_kwargs..., source_save_every=1)
        cfg_sub = SimulationConfig(; base_kwargs..., source_save_every=every)

        rng = MersenneTwister(1)
        u0, v0 = initialize_field(20, 20, rng)
        positions0 = initialize_sources_line(4, 1.0, 0.1)

        result_full = run_simulation(cfg_full, u0, v0, positions0)
        result_sub = run_simulation(cfg_sub, u0, v0, positions0)

        strided_idx = 1:every:n_steps

        @test size(result_full.x_source, 2) == n_steps
        @test size(result_sub.x_source, 2) == length(strided_idx)

        # the subsampled run must exactly match every `every`-th column of the
        # full-resolution run (both start from the same seed/ICs, so the
        # trajectories themselves are identical -- only the save cadence differs)
        @test result_sub.x_source ≈ result_full.x_source[:, strided_idx]
        @test result_sub.times ≈ result_full.times[strided_idx]
    end

    @testset "source_save_every defaults to 1 (backward compatible)" begin
        cfg = SimulationConfig(
            L=1.0, nx=10, ny=10, dt=0.01, n_steps=7, save_every=7,
            model=ReactionDiffusionModel(FHNKinetics(1.0, 12.5, 0.2, 0.2, -1.6), 0.001, 0.0),
            coupling=PacemakerCoupling(2.0, 0.05),
            mech=MechanicalParams(0.1, 0.01, 1.0, 0.1, 0.2, 0.0, :reciprocal),
        )
        @test cfg.source_save_every == 1
    end

    @testset "progress_every prints step/percent updates without changing results" begin
        cfg = SimulationConfig(
            L=1.0, nx=10, ny=10, dt=0.01, n_steps=20, save_every=20,
            model=ReactionDiffusionModel(FHNKinetics(1.0, 12.5, 0.2, 0.2, -1.6), 0.001, 0.0),
            coupling=PacemakerCoupling(2.0, 0.05),
            mech=MechanicalParams(0.1, 0.01, 1.0, 0.1, 0.2, 0.0, :reciprocal),
            seed=5,
        )
        rng = MersenneTwister(1)
        u0, v0 = initialize_field(10, 10, rng)
        positions0 = initialize_sources_line(3, 1.0, 0.1)

        result_quiet = run_simulation(cfg, u0, v0, positions0)

        pipe = Pipe()
        Base.link_pipe!(pipe; reader_supports_async=true, writer_supports_async=true)
        result_verbose = redirect_stdout(pipe) do
            run_simulation(cfg, u0, v0, positions0; progress_every=5)
        end
        close(pipe.in)
        output = read(pipe.out, String)

        # same deterministic dynamics regardless of whether progress is reported
        @test result_verbose.u_field_stack ≈ result_quiet.u_field_stack
        @test result_verbose.x_source ≈ result_quiet.x_source

        # printed at steps 5, 10, 15, 20 (n_steps), each with a step count and a percent
        @test occursin("5/20", output)
        @test occursin("10/20", output)
        @test occursin("15/20", output)
        @test occursin("20/20", output)
        @test occursin("100.0%", output)
    end
end
