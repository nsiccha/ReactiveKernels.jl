using ReactiveKernels
using LinearAlgebra
using Random
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
                 joinpath("receipts", "validate_distributions.jl"))
        path = joinpath(_BENCH_DIR, name)
        @test isfile(path)
        @test _parses(path)
    end
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
