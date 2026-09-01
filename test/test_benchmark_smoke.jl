using ReactiveKernels
using LinearAlgebra
using Random
import SHA
using Test

# Keeps the checked-in benchmark scripts from rotting outside the (external-dep)
# benchmark runs: every script must PARSE, and the in-repo microbenchmark's parity
# gates + 0-B receipts + typed/LLVM hard gate must still hold on a tiny build.

const _BENCH_DIR = joinpath(@__DIR__, "..", "benchmark")

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
                 "eight_schools_ad_comparison.jl",
                 "eight_schools_ad_comparison_body.jl",
                 "eight_schools_reactant_comparison.jl",
                 "eight_schools_reactant_comparison_body.jl",
                 "eight_schools_reactant_ad_comparison.jl",
                 "eight_schools_reactant_ad_comparison_body.jl",
                 "nuts_reactant_comparison.jl",
                 "nuts_reactant_comparison_body.jl",
                 "eval_throughput_comparison.jl",
                 "eval_throughput_comparison_body.jl",
                 joinpath("receipts", "validate_nuts_reactant.jl"),
                 joinpath("receipts", "validate_eval_throughput.jl"),
                 joinpath("receipts", "validate_distributions.jl"),
                 joinpath("receipts", "validate_scalar_gallery_distributions.jl"),
                 joinpath("receipts", "validate_structured_distributions.jl"),
                 joinpath("receipts", "validate_distribution_gradients.jl"),
                 joinpath("receipts", "validate_eight_schools_ad.jl"),
                 joinpath("receipts", "validate_eight_schools_reactant.jl"),
                 joinpath("receipts", "validate_eight_schools_reactant_ad.jl"))
        path = joinpath(_BENCH_DIR, name)
        @test isfile(path)
        @test _parses(path)
    end
end

@testset "Eight Schools AD benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_eight_schools_ad.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "eight-schools-ad-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_eight_schools_ad_receipt(receipt))
end

@testset "Eight Schools Reactant benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_eight_schools_reactant.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "eight-schools-reactant-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_eight_schools_reactant_receipt(receipt))
end

@testset "Eight Schools Reactant-compiled-AD benchmark receipt validates" begin
    validator = joinpath(
        _BENCH_DIR, "receipts", "validate_eight_schools_reactant_ad.jl")
    receipt = joinpath(
        _BENCH_DIR, "receipts", "eight-schools-reactant-ad-v1.toml")
    @test isfile(receipt)
    include(validator)
    @test isempty(validate_eight_schools_reactant_ad_receipt(receipt))
end

@testset "Eight Schools receipt text digests are checkout-line-ending invariant" begin
    mktempdir() do dir
        lf_path = joinpath(dir, "lf.txt")
        crlf_path = joinpath(dir, "crlf.txt")
        text = "alpha\nβeta\n"
        write(lf_path, text)
        write(crlf_path, replace(text, "\n" => "\r\n"))
        expected = bytes2hex(SHA.sha256(text))
        for digest in (
                _eight_schools_ad_text_sha256,
                _eight_schools_reactant_text_sha256,
                _eight_schools_reactant_ad_text_sha256,
            )
            @test digest(lf_path) == expected
            @test digest(crlf_path) == expected
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
