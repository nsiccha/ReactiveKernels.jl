using ReactiveKernels
using LinearAlgebra
using Random
import SHA
import TOML
using Test

# Keeps the checked-in benchmark scripts from rotting outside the (external-dep)
# benchmark runs: every script must PARSE, and the in-repo microbenchmark's parity
# gates + 0-B receipts + typed/LLVM hard gate must still hold on a tiny build.

const _BENCH_DIR = joinpath(@__DIR__, "..", "benchmark")
const _REPOSITORY_ROOT = normpath(joinpath(_BENCH_DIR, ".."))

include(joinpath(_BENCH_DIR, "mnist_logistic_matrix_spec.jl"))
import .MNISTLogisticMatrixSpec
include(joinpath(_BENCH_DIR, "eight_schools_matrix_spec.jl"))

function _load_benchmark_validator(path)
    validator = Module(gensym(:BenchmarkValidator), true, true)
    Core.eval(validator,
        :(include(path::AbstractString) = Base.include($validator, path)))
    Base.include(validator, path)
    validator
end

function _validator_fixture_root(relative_paths)
    root = mktempdir()
    for relative in relative_paths
        source = joinpath(_REPOSITORY_ROOT, relative)
        destination = joinpath(root, relative)
        mkpath(dirname(destination))
        cp(source, destination; force = true)
    end
    repository_git = joinpath(_REPOSITORY_ROOT, ".git")
    fixture_git = joinpath(root, ".git")
    if isdir(repository_git)
        symlink(realpath(repository_git), fixture_git)
    else
        write(fixture_git, read(repository_git, String))
    end
    root
end

function _reject_current_source_mutation(path, mutation, check)
    saved = read(path, String)
    mutated = mutation(saved)
    mutated === nothing && error("negative mutation anchor was not exact in $path")
    try
        write(path, mutated)
        errors = check()
        @test !isempty(errors)
        @test any(error -> occursin("current source exceeds its supported delta", error), errors)
    finally
        write(path, saved)
    end
end

_replace_one_anchor(text, replacement) =
    length(findall(first(replacement), text)) == 1 ?
        replace(text, replacement) : nothing

function _parses(path)
    ex = Meta.parseall(read(path, String); filename = path)
    !(ex isa Expr && ex.head === :error) &&
        !any(a -> a isa Expr && a.head === :error, ex.args)
end

@testset "benchmark scripts parse (anti-rot)" begin
    for name in ("nuts_comparison.jl", "nuts_comparison_body.jl",
                 "nuts_microbench.jl", "nuts_microbench_ca9.jl",
                 "_ca9_microbench_body.jl", "_repro_guard.jl",
                 "distributions_comparison.jl",
                 "scalar_distribution_gallery_comparison.jl",
                 "structured_distributions_comparison.jl",
                 "distribution_gradients.jl",
                 "_ad_comparison_support.jl",
                 "_comparison_source_attestation.jl",
                 "model_benchmark_matrix_spec.jl",
                 "eight_schools_matrix_spec.jl",
                 "eight_schools_ad_comparison.jl",
                 "eight_schools_ad_comparison_body.jl",
                 "mnist_logistic_comparison.jl",
                 "mnist_logistic_comparison_body.jl",
                 "mnist_logistic_matrix_spec.jl",
                 "mnist_logistic_ad_comparison.jl",
                 "mnist_logistic_ad_comparison_body.jl",
                 "eight_schools_reactant_comparison.jl",
                 "eight_schools_reactant_comparison_body.jl",
                 "eight_schools_reactant_ad_comparison.jl",
                 "eight_schools_reactant_ad_comparison_body.jl",
                 "sum_to_zero_comparison.jl",
                 "sum_to_zero_comparison_body.jl",
                 "sum_to_zero_reactant_comparison.jl",
                 "sum_to_zero_reactant_comparison_body.jl",
                 "mnist_reactant_comparison.jl",
                 "mnist_reactant_comparison_body.jl",
                 "mnist_reactant_wren_pca40_comparison_body.jl",
                 "mnist_reactant_ad_comparison.jl",
                 "mnist_reactant_ad_comparison_body.jl",
                 "mnist_reactant_ad_wren_pca40_comparison_body.jl",
                 "_mnist_dataset_profiles.jl",
                 "probprog_mcmc_comparison.jl",
                 "probprog_mcmc_comparison_body.jl",
                 joinpath("receipts", "validate_probprog_mcmc.jl"),
                 "practicalbayes_comparison.jl",
                 "practicalbayes_comparison_body.jl",
                 joinpath("receipts", "validate_practicalbayes.jl"),
                 "nuts_reactant_comparison.jl",
                 "nuts_reactant_comparison_body.jl",
                 "eval_throughput_comparison.jl",
                 "eval_throughput_comparison_body.jl",
                 "partial_evaluation_comparison.jl",
                 "partial_evaluation_comparison_body.jl",
                 joinpath("receipts", "validate_partial_evaluation.jl"),
                 joinpath("receipts", "validate_nuts_reactant.jl"),
                 joinpath("receipts", "validate_eval_throughput.jl"),
                 joinpath("receipts", "validate_distributions.jl"),
                 joinpath("receipts", "validate_scalar_gallery_distributions.jl"),
                 joinpath("receipts", "validate_structured_distributions.jl"),
                 joinpath("receipts", "validate_distribution_gradients.jl"),
                 joinpath("receipts", "validate_eight_schools_ad.jl"),
                 joinpath("receipts", "validate_eight_schools_primal.jl"),
                 joinpath("receipts", "validate_mnist_logistic.jl"),
                 joinpath("receipts", "validate_mnist_logistic_ad.jl"),
                 joinpath("receipts", "validate_mnist_logistic_suite.jl"),
                 joinpath("receipts", "validate_eight_schools_suite.jl"),
                 joinpath("receipts", "validate_ppl_model_suites.jl"),
                 joinpath("receipts", "validate_eight_schools_reactant.jl"),
                 joinpath("receipts", "validate_eight_schools_reactant_ad.jl"),
                 joinpath("receipts", "validate_sum_to_zero_native.jl"),
                 joinpath("receipts", "validate_sum_to_zero_reactant.jl"),
                 joinpath("receipts", "_validate_mnist_dataset_profile.jl"),
                 joinpath("receipts", "validate_mnist_reactant.jl"),
                 joinpath("receipts", "validate_mnist_reactant_ad.jl"))
        path = joinpath(_BENCH_DIR, name)
        @test isfile(path)
        @test _parses(path)
    end
end

@testset "sum-to-zero benchmark receipts validate" begin
    for name in (
        "validate_sum_to_zero_native.jl",
        "validate_sum_to_zero_reactant.jl",
    )
        _load_benchmark_validator(joinpath(_BENCH_DIR, "receipts", name))
    end
    @test true
end

@testset "MNIST Reactant dataset routes stay reproducible" begin
    for (wrapper, frozen_body) in (
            "mnist_reactant_comparison.jl" =>
                "mnist_reactant_wren_pca40_comparison_body.jl",
            "mnist_reactant_ad_comparison.jl" =>
                "mnist_reactant_ad_wren_pca40_comparison_body.jl",
        )
        source = read(joinpath(_BENCH_DIR, wrapper), String)
        @test occursin("--dataset=wren-pca40", source)
        @test occursin(frozen_body, source)
        @test isfile(joinpath(_BENCH_DIR, frozen_body))
    end
end

@testset "Eight Schools benchmark capability matrix is explicit" begin
    @test EightSchoolsMatrixSpec.EIGHT_SCHOOLS_MODELS == ("centered",)
    @test EightSchoolsMatrixSpec.EIGHT_SCHOOLS_BOUNDARIES ==
        ("packed_unconstrained", "constrained_parameters", "minimal_likelihood")
    @test EightSchoolsMatrixSpec.EIGHT_SCHOOLS_OUTCOMES ==
        ("joint", "prior", "likelihood", "pointwise")
    @test length(EightSchoolsMatrixSpec.EIGHT_SCHOOLS_RK_CONFIGURATIONS) == 10
    @test length(EightSchoolsMatrixSpec.headline_cells()) == 10
    @test all(cell -> cell.state == "supported",
              EightSchoolsMatrixSpec.headline_cells())

    nonallocating = only(filter(
        configuration -> configuration.id == "primal_nonallocating_bound",
        EightSchoolsMatrixSpec.EIGHT_SCHOOLS_RK_CONFIGURATIONS))
    @test EightSchoolsMatrixSpec.matrix_support(
        nonallocating, "packed_unconstrained", "prior")[1] == "not_applicable"
    @test EightSchoolsMatrixSpec.matrix_support(
        nonallocating, "constrained_parameters", "joint")[1] == "unsupported"

    ad = only(filter(
        configuration -> configuration.id == "ad_native",
        EightSchoolsMatrixSpec.EIGHT_SCHOOLS_RK_CONFIGURATIONS))
    @test EightSchoolsMatrixSpec.matrix_support(
        ad, "packed_unconstrained", "pointwise")[1] == "supported"
    @test EightSchoolsMatrixSpec.matrix_support(
        ad, "constrained_parameters", "joint")[1] == "supported"
    @test EightSchoolsMatrixSpec.matrix_support(
        ad, "minimal_likelihood", "joint")[1] == "unsupported"

    reactant_ad = only(filter(
        configuration -> configuration.id == "ad_reactant",
        EightSchoolsMatrixSpec.EIGHT_SCHOOLS_RK_CONFIGURATIONS))
    @test EightSchoolsMatrixSpec.matrix_support(
        reactant_ad, "packed_unconstrained", "pointwise")[1] == "unsupported"
    @test EightSchoolsMatrixSpec.matrix_support(
        reactant_ad, "constrained_parameters", "joint")[1] == "unsupported"
end

include(joinpath(_BENCH_DIR, "_comparison_source_attestation.jl"))
using .ComparisonSourceAttestation

@testset "historical source pins and current-source deltas stay explicit" begin
    root = normpath(joinpath(_BENCH_DIR, ".."))
    receipts = (
        "mnist-logistic-ad-v2.toml" => (
            "model_source" => nothing,
            "primal_comparator_source" =>
                "get(ENV, \"RK_MNIST_DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n",
        ),
        "eight-schools-ad-v2.toml" => (
            "model_source" => nothing,
            "primal_comparator_source" =>
                "get(ENV, \"RK_EIGHT_SCHOOLS_DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n",
        ),
    )

    for (receipt_name, pins) in receipts
        receipt = TOML.parsefile(joinpath(_BENCH_DIR, "receipts", receipt_name))
        for (key, guard) in pins
            pin = receipt["pins"][key]
            label = "$receipt_name/$key"
            @test isempty(historical_source_pin_errors(root, pin; label = label))
            current = pin["current"]
            @test current["path"] == pin["path"]
            @test isempty(recorded_current_source_pin_errors(
                root, current; label = "$label recorded current"))
            actual = read(joinpath(root, pin["path"]), String)
            published = ComparisonSourceAttestation._git_blob_text(
                root, pin["git_blob"])
            recorded = ComparisonSourceAttestation._git_blob_text(
                root, current["git_blob"])
            if key == "model_source"
                if startswith(receipt_name, "mnist")
                    @test mnist_model_source_preserves_published_authority(
                        actual, published)
                    @test mnist_model_source_matches_recorded_current(
                        actual, recorded)
                else
                    @test eight_schools_model_source_preserves_published_authority(
                        actual, published)
                    @test eight_schools_model_source_matches_recorded_current(
                        actual, recorded)
                end
            else
                @test comparator_source_matches_current_delta(
                    actual, published, guard)
            end
        end
    end

    mnist_pin = TOML.parsefile(joinpath(
        _BENCH_DIR, "receipts", "mnist-logistic-ad-v2.toml"))["pins"]["model_source"]
    mnist_published = ComparisonSourceAttestation._git_blob_text(
        root, mnist_pin["git_blob"])
    mnist_recorded = ComparisonSourceAttestation._git_blob_text(
        root, mnist_pin["current"]["git_blob"])
    mnist_current = read(joinpath(root, mnist_pin["path"]), String)
    @test !mnist_model_source_preserves_published_authority(replace(
        mnist_current,
        "normal(0.0, 1.0).logpdf(coefficient)" =>
            "normal(0.0, 2.0).logpdf(coefficient)",
    ), mnist_published)
    @test !mnist_model_source_preserves_published_authority(replace(
        mnist_current,
        "function evaluate_mnist_logistic_source(; model_only::Bool = false)" =>
            "function evaluate_mnist_logistic_source_changed()",
    ), mnist_published)
    @test !mnist_model_source_matches_recorded_current(replace(
        mnist_current,
        "function evaluate_mnist_logistic_source(; model_only::Bool = false)" =>
            "function evaluate_mnist_logistic_source_changed()",
    ), mnist_recorded)
    @test !mnist_model_source_matches_recorded_current(replace(
        mnist_current,
        "compose(_MNIST_LOGISTIC_OPTIMIZED_GRAPH_TEMPLATE[])" =>
            "compose(_MNIST_LOGISTIC_GRAPH_TEMPLATE[])",
    ), mnist_recorded)
    @test !mnist_model_source_matches_recorded_current(replace(
        mnist_current,
        "const NUM_CLASSES = 10" => "const NUM_CLASSES = 11",
    ), mnist_recorded)
    @test !mnist_model_source_preserves_published_authority(replace(
        mnist_current,
        "    nonreference_logits = W * transpose(X) .+ b\n" =>
            "    nonreference_logits = W * transpose(X) .+ b .+ 1.0\n",
    ), mnist_recorded)

    eight_pin = TOML.parsefile(joinpath(
        _BENCH_DIR, "receipts", "eight-schools-ad-v2.toml"))["pins"]["model_source"]
    eight_published = ComparisonSourceAttestation._git_blob_text(
        root, eight_pin["git_blob"])
    eight_current = read(joinpath(root, eight_pin["path"]), String)
    @test !eight_schools_model_source_preserves_published_authority(replace(
        eight_current,
        "eight_schools-eight_schools_centered" => "eight_schools-changed_model",
    ), eight_published)
    @test !eight_schools_model_source_preserves_published_authority(replace(
        eight_current,
        "    posterior::Float64 = constrained_logdensity + log_jacobian\n" =>
            "    posterior::Float64 = constrained_logdensity + 2.0 * log_jacobian\n",
    ), eight_published)

    sum_receipt = TOML.parsefile(joinpath(
        _BENCH_DIR, "receipts", "sum-to-zero-native-v1.toml"))
    sum_pin = sum_receipt["pins"]["model_source"]
    @test isempty(historical_source_pin_errors(
        root, sum_pin; commit = sum_receipt["pins"]["reactivekernels_sha"],
        label = "sum-to-zero/model_source"))
    sum_published = ComparisonSourceAttestation._git_blob_text(
        root, sum_pin["git_blob"])
    sum_current = read(joinpath(root, sum_pin["path"]), String)
    @test sum_to_zero_model_source_preserves_published_authority(
        sum_current, sum_published)
    @test !sum_to_zero_model_source_preserves_published_authority(replace(
        sum_current,
        "    log_τ::Float64 = unconstrained[2]\n" =>
            "    log_τ::Float64 = unconstrained[3]\n",
    ), sum_published)
    @test !sum_to_zero_model_source_preserves_published_authority(replace(
        sum_current,
        "    sum_to_zero_log_jacobian::Float64 = 0.0\n" =>
            "    sum_to_zero_log_jacobian::Float64 = 1.0\n",
    ), sum_published)

    bad_history = Dict(mnist_pin)
    bad_history["git_blob"] = "0"^40
    @test "mnist-logistic-ad-v2.toml/model_source published Git blob mismatch" in
        historical_source_pin_errors(root, bad_history;
            label = "mnist-logistic-ad-v2.toml/model_source")

    published = "model prefix\nturing definition\nmanual definition\nrun_comparison()\n"
    guard = "get(ENV, \"DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n"
    current = """
    model prefix
    # DOCS-BASELINE-BEGIN: turing
    turing definition
    # DOCS-BASELINE-END: turing
    # DOCS-BASELINE-BEGIN: manual
    manual definition
    # DOCS-BASELINE-END: manual
    $(guard)"""
    @test comparator_source_matches_current_delta(current, published, guard)
    @test !comparator_source_matches_current_delta(
        replace(current, "# DOCS-BASELINE-END: manual\n" => ""),
        published, guard)
    @test comparator_source_matches_current_delta(
        replace(current, "model prefix" => "expanded matrix implementation"),
        published, guard)

    optimized = replace(current,
        "# DOCS-BASELINE-END: turing\n" =>
            "# DOCS-BASELINE-END: turing\n" *
            "# DOCS-BASELINE-BEGIN: turing-optimized\noptimized definition\n" *
            "# DOCS-BASELINE-END: turing-optimized\n")
    @test comparator_source_matches_current_delta(optimized, published, guard)
    @test !comparator_source_matches_current_delta(
        replace(optimized, "turing definition" => "changed baseline"),
        published, guard)
end

@testset "MNIST logistic primal benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_mnist_logistic.jl")
    validation = _load_benchmark_validator(validator)
    for receipt_name in (
            "mnist-logistic-primal-v3.toml",
            "mnist-logistic-wren-pca40-v1.toml",
        )
        receipt = joinpath(_BENCH_DIR, "receipts", receipt_name)
        @test isfile(receipt)
        @test isempty(validation.validate_mnist_logistic_receipt(receipt))
    end
end

@testset "MNIST logistic AD benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_mnist_logistic_ad.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-logistic-ad-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_mnist_logistic_ad_receipt(receipt))
    wren_receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-logistic-ad-wren-pca40-v1.toml")
    @test isfile(wren_receipt)
    @test isempty(validation.validate_mnist_logistic_ad_receipt(wren_receipt))

    comparator_relative = "benchmark/mnist_logistic_comparison_body.jl"
    model_relative =
        "packages/ReactiveKernelsPPLExamples/src/mnist_logistic.jl"
    original_comparator = read(joinpath(_REPOSITORY_ROOT, comparator_relative), String)
    original_model = read(joinpath(_REPOSITORY_ROOT, model_relative), String)
    fixture = _validator_fixture_root((
        comparator_relative, model_relative,
        "benchmark/receipts/mnist-logistic-primal-v3.toml"))
    try
        comparator_path = joinpath(fixture, comparator_relative)
        model_path = joinpath(fixture, model_relative)
        check() = validation.validate_mnist_logistic_ad_receipt(receipt; root = fixture)
        @test isempty(check())
        _reject_current_source_mutation(comparator_path,
            text -> _replace_one_anchor(text,
                "outcome == \"joint\" ? :density : Symbol(outcome)" =>
                    "outcome == \"joint\" ? :prior : Symbol(outcome)"),
            check)
        _reject_current_source_mutation(model_path,
            text -> _replace_one_anchor(
                text, "const NUM_CLASSES = 10" => "const NUM_CLASSES = 11"),
            check)
        _reject_current_source_mutation(model_path,
            text -> _replace_one_anchor(text,
                "compose(_MNIST_LOGISTIC_OPTIMIZED_GRAPH_TEMPLATE[])" =>
                    "compose(_MNIST_LOGISTIC_GRAPH_TEMPLATE[])"),
            check)

        function optimized_only_mutation(text)
            marker = "# DOCS-BASELINE-BEGIN: turing-optimized\n"
            pieces = split(text, marker; limit = 2)
            length(pieces) == 2 || return nothing
            from = "W ~ filldist(Normal(), C - 1, D)"
            length(findall(from, pieces[1])) == 1 &&
                length(findall(from, pieces[2])) == 1 || return nothing
            pieces[1] * marker * replace(
                pieces[2], from => "W ~ filldist(Normal(0, 2), C - 1, D)"; count = 1)
        end
        pin = TOML.parsefile(receipt)["pins"]["primal_comparator_source"]
        published = read(`git -C $fixture cat-file blob $(pin["git_blob"])`, String)
        guard = "get(ENV, \"RK_MNIST_DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n"
        optimized_only = optimized_only_mutation(read(comparator_path, String))
        @test comparator_source_matches_current_delta(
            optimized_only, published, guard)
        _reject_current_source_mutation(comparator_path, optimized_only_mutation, check)
    finally
        @test read(joinpath(_REPOSITORY_ROOT, comparator_relative), String) ==
            original_comparator
        @test read(joinpath(_REPOSITORY_ROOT, model_relative), String) ==
            original_model
        rm(fixture; recursive = true, force = true)
    end

    impossible_receipt = TOML.parsefile(wren_receipt)
    impossible_receipt["pins"]["primal_comparator_source"]["commit"] = "0"^40
    mktemp() do impossible_path, io
        TOML.print(io, impossible_receipt)
        flush(io)
        errors = validation.validate_mnist_logistic_ad_receipt(
            impossible_path; root = normpath(joinpath(_BENCH_DIR, "..")))
        @test "primal_comparator_source published source commit is unavailable" in errors
    end
end

@testset "Eight Schools primal benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_eight_schools_primal.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "eight-schools-primal-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_eight_schools_primal_receipt(receipt))
end

@testset "Eight Schools AD benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_eight_schools_ad.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "eight-schools-ad-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_eight_schools_ad_receipt(receipt))

    comparator_relative = "benchmark/eight_schools_primal_comparison_body.jl"
    original_comparator = read(joinpath(_REPOSITORY_ROOT, comparator_relative), String)
    model_relative = "packages/ReactiveKernelsPPLExamples/src/eight_schools.jl"
    original_model = read(joinpath(_REPOSITORY_ROOT, model_relative), String)
    fixture = _validator_fixture_root((
        comparator_relative,
        model_relative))
    try
        comparator_path = joinpath(fixture, comparator_relative)
        check() = validation.validate_eight_schools_ad_receipt(receipt; root = fixture)
        @test isempty(check())
        _reject_current_source_mutation(comparator_path,
            text -> _replace_one_anchor(text, "μ ~ Normal(0, 5)" => "μ ~ Normal(0, 2)"),
            check)
        _reject_current_source_mutation(comparator_path,
            text -> _replace_one_anchor(text,
                "outcome == \"joint\" ? :posterior : Symbol(outcome == \"prior\" ?" =>
                    "outcome == \"joint\" ? :constrained_logdensity : Symbol(outcome == \"prior\" ?"),
            check)
    finally
        @test read(joinpath(_REPOSITORY_ROOT, comparator_relative), String) ==
            original_comparator
        @test read(joinpath(_REPOSITORY_ROOT, model_relative), String) ==
            original_model
        rm(fixture; recursive = true, force = true)
    end
end

@testset "Eight Schools Reactant benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_eight_schools_reactant.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "eight-schools-reactant-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_eight_schools_reactant_receipt(receipt))
end

@testset "Eight Schools Reactant-compiled-AD benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_eight_schools_reactant_ad.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "eight-schools-reactant-ad-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_eight_schools_reactant_ad_receipt(receipt))
end

@testset "MNIST Reactant benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_mnist_reactant.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-reactant-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_mnist_reactant_receipt(receipt))
    wren_receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-reactant-wren-pca40-v1.toml")
    @test isfile(wren_receipt)
    @test isempty(validation.validate_mnist_reactant_receipt(wren_receipt))
end

@testset "MNIST Reactant-compiled-AD benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_mnist_reactant_ad.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-reactant-ad-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_mnist_reactant_ad_receipt(receipt))
    wren_receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-reactant-ad-wren-pca40-v1.toml")
    @test isfile(wren_receipt)
    @test isempty(validation.validate_mnist_reactant_ad_receipt(wren_receipt))
end

@testset "complete PPL model benchmark suites validate" begin
    validation = _load_benchmark_validator(joinpath(
        _BENCH_DIR, "receipts", "validate_ppl_model_suites.jl"))
    @test isempty(validation.validate_ppl_model_suites())
end

@testset "partial-evaluation benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_partial_evaluation.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "partial-evaluation-mnist-v1.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_partial_evaluation_receipt(receipt))
end

@testset "MNIST benchmark capability matrix is explicit" begin
    @test MNISTLogisticMatrixSpec.MNIST_MODELS == ("idiomatic", "vcat_free")
    @test MNISTLogisticMatrixSpec.MNIST_BOUNDARIES ==
        ("packed_unconstrained", "structured_parameters")
    @test MNISTLogisticMatrixSpec.MNIST_OUTCOMES ==
        ("joint", "prior", "likelihood", "pointwise")
    @test length(MNISTLogisticMatrixSpec.MNIST_RK_CONFIGURATIONS) == 10
    @test length(MNISTLogisticMatrixSpec.ModelBenchmarkMatrixSpec.
        RK_UNAVAILABLE_CONFIGURATION_COMBINATIONS) == 3
    @test all(combination -> !isempty(combination.reason),
        MNISTLogisticMatrixSpec.ModelBenchmarkMatrixSpec.
            RK_UNAVAILABLE_CONFIGURATION_COMBINATIONS)
    @test length(MNISTLogisticMatrixSpec.headline_cells()) ==
        length(MNISTLogisticMatrixSpec.MNIST_MODELS) *
        length(MNISTLogisticMatrixSpec.MNIST_RK_CONFIGURATIONS)
    @test all(cell -> cell.state == "supported",
              MNISTLogisticMatrixSpec.headline_cells())

    nonallocating = only(filter(
        configuration -> configuration.id == "primal_nonallocating_bound",
        MNISTLogisticMatrixSpec.MNIST_RK_CONFIGURATIONS))
    @test MNISTLogisticMatrixSpec.matrix_support(
        nonallocating, "packed_unconstrained", "prior")[1] == "not_applicable"

    ad = only(filter(
        configuration -> configuration.id == "ad_native",
        MNISTLogisticMatrixSpec.MNIST_RK_CONFIGURATIONS))
    @test MNISTLogisticMatrixSpec.matrix_support(
        ad, "packed_unconstrained", "pointwise")[1] == "unsupported"
    @test MNISTLogisticMatrixSpec.matrix_support(
        ad, "structured_parameters", "joint")[1] == "unsupported"

    @test Set(comparator.id for comparator in
              MNISTLogisticMatrixSpec.MNIST_COMPARATORS) == Set((
        "manual_primal", "manual_ad",
        "turing_idiomatic_primal", "turing_idiomatic_ad",
        "turing_vcat_free_primal", "turing_vcat_free_ad",
        "practicalbayes_idiomatic_primal", "practicalbayes_idiomatic_ad",
        "practicalbayes_vcat_free_primal", "practicalbayes_vcat_free_ad",
    ))
end

@testset "Eight Schools receipt text digests are checkout-line-ending invariant" begin
    mktempdir() do dir
        lf_path = joinpath(dir, "lf.txt")
        crlf_path = joinpath(dir, "crlf.txt")
        text = "alpha\nβeta\n"
        write(lf_path, text)
        write(crlf_path, replace(text, "\n" => "\r\n"))
        expected = bytes2hex(SHA.sha256(text))
        digest_specs = (
            ("validate_eight_schools_ad.jl", :_eight_schools_ad_text_sha256),
            ("validate_mnist_logistic_ad.jl", :_mnist_ad_text_sha256),
            ("validate_eight_schools_reactant.jl",
             :_eight_schools_reactant_text_sha256),
            ("validate_eight_schools_reactant_ad.jl",
             :_eight_schools_reactant_ad_text_sha256),
            ("validate_mnist_reactant.jl", :_mnist_reactant_text_sha256),
            ("validate_mnist_reactant_ad.jl",
             :_mnist_reactant_ad_text_sha256),
        )
        for (filename, function_name) in digest_specs
            validation = _load_benchmark_validator(joinpath(
                _BENCH_DIR, "receipts", filename))
            digest = getfield(validation, function_name)
            @test Base.invokelatest(digest, lf_path) == expected
            @test Base.invokelatest(digest, crlf_path) == expected
        end
    end
end

@testset "ProbProg MCMC sampling receipt validates" begin
    validator = joinpath(_BENCH_DIR, "receipts", "validate_probprog_mcmc.jl")
    receipt = joinpath(_BENCH_DIR, "receipts", "probprog-mcmc-v1.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(Base.invokelatest(
        validation.validate_probprog_mcmc_receipt, receipt))
end

@testset "PracticalBayes PPL comparator receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_practicalbayes.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "practicalbayes-comparison-v1.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(Base.invokelatest(
        validation.validate_practicalbayes_receipt, receipt))
end

@testset "adaptive Reactant NUTS benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_nuts_reactant.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "nuts-reactant-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_nuts_reactant_receipt(receipt))
end

@testset "evaluation throughput benchmark receipt validates" begin
    validator = joinpath(_BENCH_DIR, "receipts", "validate_eval_throughput.jl")
    receipt = joinpath(_BENCH_DIR, "receipts", "eval-throughput-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_eval_throughput_receipt(receipt))
end

@testset "structured distribution benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_structured_distributions.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "structured-distribution-logdensity-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_structured_distribution_receipt(receipt))
end

@testset "scalar gallery benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_scalar_gallery_distributions.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "scalar-distribution-gallery-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_scalar_gallery_distribution_receipt(receipt))
end

@testset "distribution benchmark receipt validates" begin
    validator = joinpath(_BENCH_DIR, "receipts", "validate_distributions.jl")
    receipt = joinpath(_BENCH_DIR, "receipts", "distribution-logdensity-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_distribution_receipt(receipt))
end

@testset "distribution gradient benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_distribution_gradients.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "distribution-gradient-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_distribution_gradient_receipt(receipt))
end

@testset "reproducibility guard: attached rejected, detached accepted, dirty rejected" begin
    include(joinpath(_BENCH_DIR, "_repro_guard.jl"))
    mktempdir() do dir
        repo = joinpath(dir, "repo")
        run(`git init -q $repo`)
        run(`git -C $repo config user.email t@example.com`)
        run(`git -C $repo config user.name tester`)
        write(joinpath(repo, "f.txt"), "x")
        run(`git -C $repo add -A`)
        run(`git -C $repo commit -q -m init`)
        sha = readchomp(`git -C $repo rev-parse HEAD`)
        # Attached-branch worktree is REJECTED even though it is tracked-clean.
        att = joinpath(dir, "att")
        run(`git -C $repo worktree add -q -b br $att $sha`)
        @test_throws ErrorException _require_clean_detached_candidate(att)
        # Clean detached worktree is ACCEPTED and returns the pinned SHA.
        det = joinpath(dir, "det")
        run(`git -C $repo worktree add -q --detach $det $sha`)
        @test _require_clean_detached_candidate(det) == sha
        # A dirty detached worktree is REJECTED.
        write(joinpath(det, "f.txt"), "y")
        @test_throws ErrorException _require_clean_detached_candidate(det)
    end
end

@testset "in-repo microbench smoke (parity + 0-B + typed/LLVM)" begin
    # Define the microbench functions without running the full (slow) benchmark.
    include(joinpath(_BENCH_DIR, "nuts_microbench.jl"))
    evidence = microbench_smoke(3)          # runs the parity gates + hard gates
    @test evidence.return_concrete
    @test evidence.any_typed_slots == 0
    @test evidence.dynamic_calls == 0
    @test isempty(evidence.llvm_forbidden_symbols)
end
