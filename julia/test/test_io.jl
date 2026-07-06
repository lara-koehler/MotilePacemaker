using TOML

const _TEST_TOML_BASE = """
[grid]
L = 1.0
nx = 10
ny = 10

[time]
dt = 0.001
n_steps = 10
save_every = 10

[chemistry]
tau_u = 1.0
tau_v = 12.5
a = 0.2
b = 0.2
I0 = -1.6
Du = 0.001
Dv = 0.0

[coupling]
pacemaker_strength = 2.0
pacemaker_width = 0.05

[mechanics]
mobility = 0.5
sigma = 0.02
epsilon_LJ = 1.0
interaction_strength = 0.5
interaction_range = 0.2
u_threshold = 0.0
reciprocity = "nonreciprocal"
%s

[sources]
n_sources = 2
arrangement = "random"
"""

@testset "io" begin
    @testset "load_config computes an automatic max_force default" begin
        path = joinpath(@__DIR__, "_scratch_test_config.toml")
        write(path, replace(_TEST_TOML_BASE, "%s" => ""))
        try
            cfg, _, _ = load_config(path)
            expected = 0.5 * 0.02 / (0.001 * 0.5)  # 0.5*sigma / (dt*mobility)
            @test isapprox(cfg.mech.max_force, expected)
        finally
            rm(path, force=true)
        end
    end

    @testset "load_config respects an explicit max_force override" begin
        path = joinpath(@__DIR__, "_scratch_test_config.toml")
        write(path, replace(_TEST_TOML_BASE, "%s" => "max_force = 999.0"))
        try
            cfg, _, _ = load_config(path)
            @test cfg.mech.max_force == 999.0
        finally
            rm(path, force=true)
        end
    end

    @testset "apply_override!" begin
        @testset "sets a leaf key that already exists" begin
            raw = Dict{String,Any}("mechanics" => Dict{String,Any}("mobility" => 0.1))
            apply_override!(raw, "mechanics.mobility", 0.9)
            @test raw["mechanics"]["mobility"] == 0.9
        end

        @testset "creates intermediate Dicts for a path that doesn't exist yet" begin
            raw = Dict{String,Any}()
            apply_override!(raw, "mechanics.mobility", 0.5)
            @test raw["mechanics"]["mobility"] == 0.5
        end

        @testset "adds a new leaf without disturbing sibling keys" begin
            raw = Dict{String,Any}("mechanics" => Dict{String,Any}("mobility" => 0.1))
            apply_override!(raw, "mechanics.max_force", 20.0)
            @test raw["mechanics"]["mobility"] == 0.1
            @test raw["mechanics"]["max_force"] == 20.0
        end
    end

    @testset "parse_scan_value" begin
        @test parse_scan_value("0.5") === 0.5
        @test parse_scan_value("17") === 17
        @test parse_scan_value("1e-3") === 1e-3
        @test parse_scan_value("true") === true
        @test parse_scan_value("False") === false
        @test parse_scan_value("nonreciprocal") == "nonreciprocal"
        @test parse_scan_value("nonreciprocal") isa String
    end

    @testset "apply_override! + build_simulation_config integration" begin
        path = joinpath(@__DIR__, "_scratch_test_config.toml")
        write(path, replace(_TEST_TOML_BASE, "%s" => ""))
        try
            raw = TOML.parsefile(path)
            apply_override!(raw, "mechanics.mobility", parse_scan_value("0.9"))
            apply_override!(raw, "mechanics.reciprocity", parse_scan_value("reciprocal"))
            apply_override!(raw, "time.seed", parse_scan_value("42"))

            cfg, sourcescfg = build_simulation_config(raw)

            @test cfg.mech.mobility == 0.9
            @test cfg.mech.reciprocity == :reciprocal
            @test cfg.seed == 42
            @test sourcescfg["arrangement"] == "random"  # untouched keys survive
        finally
            rm(path, force=true)
        end
    end
end
