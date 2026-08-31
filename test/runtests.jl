using Pkg

const _REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const _ROOT_PROJECT = joinpath(_REPOSITORY_ROOT, "Project.toml")
const _ROOT_TEST_PACKAGE_PATHS = (
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsCompatibilityExamples"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsDistributionKernels"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsPPLExamples"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsNUTSExamples"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsStreamingStats"),
    joinpath(_REPOSITORY_ROOT, "packages", "ReactiveKernelsHMCDiagnostics"),
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
import ReactiveKernelsNUTSExamples
using Test

const _MATRIX_CORE_TESTS = (
    "test_runtests_selector.jl",
    "test_stateless.jl",
    "test_ad.jl",
    "test_authoring.jl",
    "test_kernel_objects.jl",
    "test_readable_expr.jl",
    "test_plate.jl",
    "test_authored_plate.jl",
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
    "test_mutation_profile_b.jl",
    "test_nuts_docs_fixture.jl",
    "test_reactivehmc_algorithm_corpus.jl",
    "test_reactivehmc_corpus_docs.jl",
    "test_walnuts_external_corpus.jl",
    "test_walnuts_docs_fixture.jl",
    "test_reactivehmc_rke_fixture.jl",
    "test_reactivehmc_rke_compiler.jl",
    "test_reactivehmc_hmc_fixture.jl",
)

const _MATRIX_ACCEPTANCE_COMPILER_TESTS = (
    "test_effect_boundary.jl",
    "test_reactivehmc_hmc_compiler.jl",
    "test_finite_structural_container.jl",
)

const _MATRIX_ACCEPTANCE_RUNTIME_TESTS = (
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
)

const _MATRIX_ACCEPTANCE_SAMPLERS_TESTS = (
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

const _MATRIX_ACCEPTANCE_BENCHMARK_TESTS = ("test_handwritten_benchmarks.jl",)

const _MATRIX_ACCEPTANCE_SHARD_GROUPS = (
    "acceptance-compiler", "acceptance-runtime", "acceptance-samplers",
    "acceptance-benchmarks",
)
const _MATRIX_ACCEPTANCE_SHARDS = (
    _MATRIX_ACCEPTANCE_COMPILER_TESTS,
    _MATRIX_ACCEPTANCE_RUNTIME_TESTS,
    _MATRIX_ACCEPTANCE_SAMPLERS_TESTS,
    _MATRIX_ACCEPTANCE_BENCHMARK_TESTS,
)
const _MATRIX_ACCEPTANCE_TESTS = (
    _MATRIX_ACCEPTANCE_COMPILER_TESTS...,
    _MATRIX_ACCEPTANCE_RUNTIME_TESTS...,
    _MATRIX_ACCEPTANCE_SAMPLERS_TESTS...,
)

const _FULL_TESTS = (_MATRIX_CORE_TESTS..., _MATRIX_ACCEPTANCE_TESTS...)

const _TEST_FILE_ORDER = (
    "test_package_boundary.jl",
    "test_ci_compiled_modules.jl",
    _FULL_TESTS...,
    "test_handwritten_benchmarks.jl",
)

const _TEST_GROUPS = (
    "core", "acceptance", _MATRIX_ACCEPTANCE_SHARD_GROUPS..., "ad", "benchmark",
)

function _test_group_files(selector::String)
    selector == "core" && return (
        "test_package_boundary.jl", "test_ci_compiled_modules.jl",
        _MATRIX_CORE_TESTS...,
    )
    selector == "acceptance" && return (
        _MATRIX_ACCEPTANCE_TESTS..., "test_handwritten_benchmarks.jl",
    )
    selector == "acceptance-compiler" && return _MATRIX_ACCEPTANCE_COMPILER_TESTS
    selector == "acceptance-runtime" && return _MATRIX_ACCEPTANCE_RUNTIME_TESTS
    selector == "acceptance-samplers" && return _MATRIX_ACCEPTANCE_SAMPLERS_TESTS
    selector == "acceptance-benchmarks" && return _MATRIX_ACCEPTANCE_BENCHMARK_TESTS
    selector == "ad" && return (
        "test_package_boundary.jl", "test_ci_compiled_modules.jl", "test_ad.jl",
    )
    selector == "benchmark" && return (
        "test_package_boundary.jl", "test_ci_compiled_modules.jl",
        "test_handwritten_benchmarks.jl",
    )
    nothing
end

function _test_file_selector(selector::String)
    basename(selector) == selector || return nothing
    file = endswith(selector, ".jl") ? selector : selector * ".jl"
    file in _TEST_FILE_ORDER ? file : nothing
end

function _select_test_files(selectors::AbstractVector{<:AbstractString})
    isempty(selectors) && return _TEST_FILE_ORDER

    requested = Set{String}()
    unknown = String[]
    for raw_selector in selectors
        selector = String(raw_selector)
        group = _test_group_files(selector)
        if group !== nothing
            union!(requested, group)
            continue
        end
        file = _test_file_selector(selector)
        if file === nothing
            push!(unknown, selector)
        else
            push!(requested, file)
        end
    end

    if !isempty(unknown)
        unknown = sort!(unique!(unknown))
        throw(ArgumentError(
            "unknown test selector(s): $(join(repr.(unknown), ", ")). " *
            "Known groups: $(join(_TEST_GROUPS, ", ")). " *
            "Known files: $(join(_TEST_FILE_ORDER, ", "))"))
    end

    Tuple(file for file in _TEST_FILE_ORDER if file in requested)
end

@assert isempty(intersect(Set(_MATRIX_CORE_TESTS), Set(_MATRIX_ACCEPTANCE_TESTS)))
@assert Tuple(Iterators.flatten(_MATRIX_ACCEPTANCE_SHARDS)) ==
    (_MATRIX_ACCEPTANCE_TESTS..., _MATRIX_ACCEPTANCE_BENCHMARK_TESTS...)
@assert all(
    isempty(intersect(Set(_MATRIX_ACCEPTANCE_SHARDS[i]),
                      Set(_MATRIX_ACCEPTANCE_SHARDS[j])))
    for i in eachindex(_MATRIX_ACCEPTANCE_SHARDS)
    for j in (i + 1):length(_MATRIX_ACCEPTANCE_SHARDS)
)
@assert all(isfile(joinpath(@__DIR__, file)) for file in _FULL_TESTS)
@assert _select_test_files(["core", "acceptance"]) == _TEST_FILE_ORDER
@assert _select_test_files(collect(_MATRIX_ACCEPTANCE_SHARD_GROUPS)) ==
    _test_group_files("acceptance")

const _SELECTED_TEST_FILES = _select_test_files(ARGS)

# Assert the package loads without sampler/compiler domain code before opting into
# the external executable exemplar used by the remaining NUTS/HMC acceptance tests.
# The matrix core shard and the historical ad/benchmark groups own this assertion.
"test_package_boundary.jl" in _SELECTED_TEST_FILES &&
    include("test_package_boundary.jl")
include(joinpath(@__DIR__, "..", "examples", "nuts_runtime.jl"))
using .ReactiveKernelsNUTSExample
include(joinpath(@__DIR__, "fixtures", "reactivehmc_algorithm_corpus_setup.jl"))

@testset "ReactiveKernels" begin
    foreach(include, filter(!=("test_package_boundary.jl"), _SELECTED_TEST_FILES))
end
