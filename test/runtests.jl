using ReactiveKernels
using Test

@testset "ReactiveKernels" begin
    benchmark_only = ARGS == ["benchmark"]
    distributions_only = ARGS == ["distributions"]
    if distributions_only
        include("test_distributions_example.jl")
    elseif !benchmark_only
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
        include("test_reactive_sampler_baseline.jl")
        include("test_reactive_nuts.jl")
        include("test_reactive_adaptation.jl")
        include("test_reactive_facade.jl")
        include("test_reactive_facade_ca9.jl")
        include("test_online_stats_example.jl")
        include("test_distributions_example.jl")
        include("test_preexisting_examples.jl")
        include("test_corrected_core_examples.jl")
    end
    distributions_only || include("test_handwritten_benchmarks.jl")
end
