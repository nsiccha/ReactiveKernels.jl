using Pkg

const _REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ROOT_PROJECT = joinpath(_REPOSITORY_ROOT, "Project.toml")
const _PPL_TEST_PACKAGE_PATHS = (
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsDistributionKernels"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsPPLExamples"),
)

# `Pkg.test` evaluates this file in a temporary test environment. Julia 1.10
# ignores the nested projects' `[sources]` entries, so materialize the local
# packages used by the cross-package HMC acceptance before loading the suite.
# A direct `julia --project=. test/runtests.jl` keeps using the explicitly
# prepared `packages/` environment and must not mutate the root Project.toml.
if !samefile(Base.active_project(), _ROOT_PROJECT)
    Pkg.develop([PackageSpec(path = path) for path in _PPL_TEST_PACKAGE_PATHS])
end

using ReactiveKernels
using Test

# Assert the package loads without sampler/compiler domain code before opting into
# the external executable exemplar used by the remaining NUTS/HMC acceptance tests.
include("test_package_boundary.jl")
include(joinpath(@__DIR__, "..", "examples", "nuts_runtime.jl"))
using .ReactiveKernelsNUTSExample

@testset "ReactiveKernels" begin
    benchmark_only = ARGS == ["benchmark"]
    ad_only = ARGS == ["ad"]
    if ad_only
        include("test_ad.jl")
    elseif !benchmark_only
        include("test_stateless.jl")
        include("test_ad.jl")
        include("test_authoring.jl")
        include("test_readable_expr.jl")
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
        include("test_reactivehmc_algorithm_corpus.jl")
        include("test_walnuts_external_corpus.jl")
        include("test_walnuts_docs_fixture.jl")
        include("test_compiler_docs.jl")
        include("test_kernel_adaptation.jl")
        include("test_nonallocating_core.jl")
        include("test_composition_cse.jl")
        include("test_deterministic_ast.jl")
        include("test_reactive.jl")
        include("test_stateful.jl")
        include("test_visualization.jl")
        include("test_adversarial.jl")
        include("test_hmc.jl")
        include("test_pathfinder.jl")
        include("test_pathfinder_docs.jl")
        include("test_reactive_sampler_baseline.jl")
        include("test_reactive_nuts.jl")
        include("test_reactive_adaptation.jl")
        include("test_reactive_facade.jl")
        include("test_reactive_facade_ca9.jl")
        include("test_benchmark_smoke.jl")
        include("test_online_stats_example.jl")
        include("test_nutpie_diagonal_adaptation.jl")
    end
    ad_only || include("test_handwritten_benchmarks.jl")
end
