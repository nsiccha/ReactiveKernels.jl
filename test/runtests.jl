using ReactiveKernels
using Test

@testset "ReactiveKernels" begin
    benchmark_only = ARGS == ["benchmark"]
    if !benchmark_only
        include("test_stateless.jl")
        include("test_authoring.jl")
        include("test_nonallocating_core.jl")
        include("test_composition_cse.jl")
        include("test_reactive.jl")
        include("test_stateful.jl")
        include("test_visualization.jl")
        include("test_adversarial.jl")
        include("test_eight_schools_example.jl")
        include("test_hmc.jl")
        include("test_preexisting_examples.jl")
        include("test_corrected_core_examples.jl")
    end
    include("test_handwritten_benchmarks.jl")
end
