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
                 "mnist_reactant_comparison.jl",
                 "mnist_reactant_comparison_body.jl",
                 "mnist_reactant_wren_pca40_comparison_body.jl",
                 "mnist_reactant_ad_comparison.jl",
                 "mnist_reactant_ad_comparison_body.jl",
                 "mnist_reactant_ad_wren_pca40_comparison_body.jl",
                 "_mnist_dataset_profiles.jl",
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
                 joinpath("receipts", "_validate_mnist_dataset_profile.jl"),
                 joinpath("receipts", "validate_mnist_reactant.jl"),
                 joinpath("receipts", "validate_mnist_reactant_ad.jl"))
        path = joinpath(_BENCH_DIR, name)
        @test isfile(path)
        @test _parses(path)
    end
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

@testset "AD comparator current-source delta stays narrow" begin
    for (receipt_name, guard) in (
            "mnist-logistic-ad-v2.toml" =>
                "get(ENV, \"RK_MNIST_DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n",
            "eight-schools-ad-v2.toml" =>
                "get(ENV, \"RK_EIGHT_SCHOOLS_DEFINITIONS_ONLY\", \"\") == \"1\" || run_comparison()\n",
        )
        receipt = TOML.parsefile(joinpath(_BENCH_DIR, "receipts", receipt_name))
        pin = receipt["pins"]["primal_comparator_source"]
        published = read(
            `git -C $_BENCH_DIR show $(pin["commit"]):$(pin["path"])`, String)
        current = read(joinpath(dirname(_BENCH_DIR), pin["path"]), String)
        @test comparator_source_matches_current_delta(current, published, guard)
    end

    model_pin = TOML.parsefile(joinpath(
        _BENCH_DIR, "receipts", "mnist-logistic-ad-v2.toml"))["pins"]["model_source"]
    published_model = read(
        `git -C $_BENCH_DIR show $(model_pin["commit"]):$(model_pin["path"])`, String)
    current_model = read(joinpath(dirname(_BENCH_DIR), model_pin["path"]), String)
    @test mnist_model_source_preserves_published_authority(
        current_model, published_model)
    @test !mnist_model_source_preserves_published_authority(
        replace(current_model, "@kernel model(" => "@kernel changed_model("),
        published_model)

    eight_pin = TOML.parsefile(joinpath(
        _BENCH_DIR, "receipts", "eight-schools-ad-v2.toml"))["pins"]["model_source"]
    published_eight = read(
        `git -C $_BENCH_DIR show $(eight_pin["commit"]):$(eight_pin["path"])`, String)
    current_eight = read(joinpath(dirname(_BENCH_DIR), eight_pin["path"]), String)
    @test eight_schools_model_source_preserves_published_authority(
        current_eight, published_eight)
    @test !eight_schools_model_source_preserves_published_authority(
        replace(current_eight, "@kernel model(" => "@kernel changed_model("),
        published_eight)

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
    receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-logistic-primal-v3.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_mnist_logistic_receipt(receipt))
end

@testset "MNIST logistic AD benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_mnist_logistic_ad.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "mnist-logistic-ad-v2.toml")
    @test isfile(receipt)
    validation = _load_benchmark_validator(validator)
    @test isempty(validation.validate_mnist_logistic_ad_receipt(receipt))
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
