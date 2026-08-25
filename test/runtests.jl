using ReactiveKernels
using Test

@testset "ReactiveKernels" begin
    include("test_stateless.jl")
    include("test_composition_cse.jl")
    include("test_reactive.jl")
    include("test_adversarial.jl")
    include("test_eight_schools_example.jl")
end
