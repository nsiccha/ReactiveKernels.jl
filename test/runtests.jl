using ReactiveKernels
using Test

# Assert the package loads without sampler/compiler domain code before opting into
# the external executable exemplar used by the remaining NUTS/HMC acceptance tests.
include("test_package_boundary.jl")
include(joinpath(@__DIR__, "..", "examples", "nuts_runtime.jl"))
using .ReactiveKernelsNUTSExample

@testset "ReactiveKernels" begin
    benchmark_only = ARGS == ["benchmark"]
    distributions_only = ARGS == ["distributions"]
    if distributions_only
        include("test_distributions_example.jl")
    elseif !benchmark_only
        include("test_stateless.jl")
        include("test_authoring.jl")
        include("test_plate.jl")
        include("test_replica.jl")
        include("test_kernel_stateful.jl")
        include("test_kernel_methodir.jl")
        include("test_kernel_factory.jl")
        include("test_kernel_lowering.jl")
        include("test_kernel_codegen.jl")
        include("test_kernel_control.jl")
        include("test_kernel_control_regressions.jl")
        include("test_kernel_nuts.jl")
        include("test_kernel_nuts_native.jl")
        include("test_nuts_docs_fixture.jl")
        include("test_compiler_docs.jl")
        include("test_kernel_adaptation.jl")
        include("test_nonallocating_core.jl")
        include("test_composition_cse.jl")
        include("test_deterministic_ast.jl")
        include("test_reactive.jl")
        include("test_stateful.jl")
        include("test_visualization.jl")
        include("test_adversarial.jl")
        include("test_eight_schools_example.jl")
        include("test_linear_regression_example.jl")
        include("test_beta_binomial_example.jl")
        include("test_poisson_gamma_example.jl")
        include("test_dugongs_example.jl")
        include("test_arma11_example.jl")
        include("test_gaussian_mixture_example.jl")
        include("test_ppl_docs_source_authority.jl")
        include("test_hmc.jl")
        include("test_reactive_sampler_baseline.jl")
        include("test_reactive_nuts.jl")
        include("test_reactive_adaptation.jl")
        include("test_reactive_facade.jl")
        include("test_reactive_facade_ca9.jl")
        include("test_benchmark_smoke.jl")
        include("test_online_stats_example.jl")
        include("test_distributions_example.jl")
        include("test_batched_example.jl")
        include("test_hmc_example.jl")
        include("test_preexisting_examples.jl")
        include("test_corrected_core_examples.jl")
    end
    distributions_only || include("test_handwritten_benchmarks.jl")
end
