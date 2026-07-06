using Test
using MotilePacemaker

@testset "MotilePacemaker" begin
    include("test_kinetics.jl")
    include("test_forces.jl")
    include("test_field_solver.jl")
    include("test_sources_init.jl")
    include("test_neighbor_list.jl")
    include("test_coupling.jl")
    include("test_simulation.jl")
    include("test_io.jl")
end
