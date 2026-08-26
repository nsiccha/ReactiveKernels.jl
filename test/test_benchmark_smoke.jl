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
                 "_ca9_microbench_body.jl")
        path = joinpath(_BENCH_DIR, name)
        @test isfile(path)
        @test _parses(path)
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
