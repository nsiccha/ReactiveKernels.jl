using Pkg

const _REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ROOT_PROJECT = joinpath(_REPOSITORY_ROOT, "Project.toml")
const _ROOT_TEST_PACKAGE_PATHS = (
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsCompatibilityExamples"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsDistributionKernels"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsPPLExamples"),
)

# `Pkg.test` evaluates this file in a temporary test environment. Julia 1.10
# ignores the nested projects' `[sources]` entries, so materialize the local
# packages used by the cross-package HMC acceptance before loading the suite.
# A direct `julia --project=. test/runtests.jl` keeps using the explicitly
# prepared `packages/` environment and must not mutate the root Project.toml.
if !samefile(Base.active_project(), _ROOT_PROJECT)
    Pkg.develop([PackageSpec(path = path) for path in _ROOT_TEST_PACKAGE_PATHS])
end

using ReactiveKernels
using Test

const _MATRIX_CORE_TESTS = (
    "test_stateless.jl",
    "test_ad.jl",
    "test_authoring.jl",
    "test_readable_expr.jl",
    "test_plate.jl",
    "test_replica.jl",
    "test_kernel_stateful.jl",
    "test_kernel_methodir.jl",
    "test_kernel_factory.jl",
    "test_kernel_lowering.jl",
    "test_kernel_codegen.jl",
    "test_kernel_control.jl",
    "test_kernel_control_regressions.jl",
    "test_kernel_nuts.jl",
    "test_kernel_nuts_native.jl",
    "test_nuts_docs_fixture.jl",
    "test_reactivehmc_algorithm_corpus.jl",
    "test_reactivehmc_corpus_docs.jl",
    "test_walnuts_external_corpus.jl",
    "test_walnuts_docs_fixture.jl",
    "test_reactivehmc_rke_fixture.jl",
    "test_reactivehmc_rke_compiler.jl",
    "test_reactivehmc_hmc_fixture.jl",
)

const _MATRIX_ACCEPTANCE_TESTS = (
    "test_reactivehmc_hmc_compiler.jl",
    "test_finite_structural_container.jl",
    "test_compiler_docs.jl",
    "test_kernel_adaptation.jl",
    "test_nonallocating_core.jl",
    "test_composition_cse.jl",
    "test_deterministic_ast.jl",
    "test_reactive.jl",
    "test_stateful.jl",
    "test_visualization.jl",
    "test_adversarial.jl",
    "test_hmc.jl",
    "test_pathfinder.jl",
    "test_pathfinder_docs.jl",
    "test_reactive_sampler_baseline.jl",
    "test_reactive_nuts.jl",
    "test_reactive_adaptation.jl",
    "test_benchmark_smoke.jl",
    "test_online_stats_example.jl",
    "test_nutpie_diagonal_adaptation.jl",
    "test_reactivehmc_phasepoint_receipt.jl",
    "test_reactivehmc_integrator_fixture.jl",
    "test_reactivehmc_endpoint_specs.jl",
    "test_reactivehmc_statistics_receipt.jl",
    "test_reactivehmc_statistics_fixture.jl",
    "test_reactivehmc_statistics_compiler.jl",
)

const _FULL_TESTS = (_MATRIX_CORE_TESTS..., _MATRIX_ACCEPTANCE_TESTS...)

const _TEST_MODE = if isempty(ARGS)
    :full
elseif ARGS == ["benchmark"]
    :benchmark
elseif ARGS == ["ad"]
    :ad
elseif ARGS == ["core"]
    :core
elseif ARGS == ["acceptance"]
    :acceptance
else
    throw(ArgumentError(
        "expected no test argument or one of benchmark, ad, core, acceptance; got $(repr(ARGS))"))
end

@assert isempty(intersect(Set(_MATRIX_CORE_TESTS), Set(_MATRIX_ACCEPTANCE_TESTS)))
@assert all(isfile(joinpath(@__DIR__, file)) for file in _FULL_TESTS)

# Assert the package loads without sampler/compiler domain code before opting into
# the external executable exemplar used by the remaining NUTS/HMC acceptance tests.
# The matrix core shard owns this assertion; an ordinary full/ad/benchmark run keeps
# the historical behavior unchanged.
_TEST_MODE == :acceptance || include("test_package_boundary.jl")
include(joinpath(@__DIR__, "..", "examples", "nuts_runtime.jl"))
using .ReactiveKernelsNUTSExample

@testset "ReactiveKernels" begin
    _TEST_MODE == :acceptance || include("test_ci_compiled_modules.jl")
    if _TEST_MODE == :ad
        include("test_ad.jl")
    elseif _TEST_MODE != :benchmark
        tests = _TEST_MODE == :core ? _MATRIX_CORE_TESTS :
            _TEST_MODE == :acceptance ? _MATRIX_ACCEPTANCE_TESTS : _FULL_TESTS
        foreach(include, tests)
    end
    _TEST_MODE in (:full, :benchmark, :acceptance) &&
        include("test_handwritten_benchmarks.jl")
end
