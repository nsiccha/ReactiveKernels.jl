using ReactiveKernels
using Test

@testset "ReactiveKernels" begin
    benchmark_only = ARGS == ["benchmark"]
    if !benchmark_only
        include("test_stateless.jl")
        include("test_composition_cse.jl")
        include("test_reactive.jl")
        include("test_preexisting_examples.jl")
    end
    include("test_handwritten_benchmarks.jl")
end
