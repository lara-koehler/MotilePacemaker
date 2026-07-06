import MotilePacemaker: reaction_rhs, AbstractKinetics

# A minimal second kinetics, used only here, to prove the field solver and
# simulation code have no FHN-specific assumptions: swapping the kinetics
# requires nothing beyond a new subtype + reaction_rhs method. (struct
# definitions must live at file top-level, not inside a @testset block.)
struct ZeroKinetics <: AbstractKinetics end
reaction_rhs(::ZeroKinetics, u, v, _I_forcing) = (zero(u), zero(v))

@testset "kinetics" begin
    @testset "FHNKinetics reaction_rhs" begin
        k = FHNKinetics(1.0, 12.5, 0.2, 0.2, -1.6)
        u = [0.0 1.0; -1.0 0.5]
        v = [0.0 0.0; 0.0 0.0]
        I_forcing = zeros(2, 2)

        du, dv = reaction_rhs(k, u, v, I_forcing)

        @test size(du) == size(u)
        @test size(dv) == size(v)
        # at u=0, v=0, I=0: du = (0 - 0 - 0 + I0)/tau_u = I0/tau_u
        @test du[1, 1] ≈ k.I0 / k.τu
        # at u=1: du = (1 - 1 - 0 + I0)/tau_u = I0/tau_u
        @test du[1, 2] ≈ k.I0 / k.τu
        @test dv[1, 1] ≈ (u[1, 1] + k.a - k.b * v[1, 1]) / k.τv
    end

    @testset "constructors promote mixed Int/Float64" begin
        k = FHNKinetics(1, 12.5, 0, 0.2, -1.6)  # τu, a as Int literals
        @test k.τu isa Float64
        @test k.a isa Float64

        model = ReactionDiffusionModel(k, 1, 0.0)  # Du as Int literal
        @test model.Du isa Float64
    end

    @testset "kinetics interface is swappable" begin
        k = ZeroKinetics()
        u = rand(3, 3)
        v = rand(3, 3)
        du, dv = reaction_rhs(k, u, v, zeros(3, 3))
        @test all(du .== 0.0)
        @test all(dv .== 0.0)

        model = ReactionDiffusionModel(k, 0.001, 0.0)
        @test model.kinetics isa ZeroKinetics
    end
end
