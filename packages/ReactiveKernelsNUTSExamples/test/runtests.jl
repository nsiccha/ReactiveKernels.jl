using LinearAlgebra
using Random
using ReactiveKernels
using ReactiveKernelsNUTSExamples
using Test

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(REPOSITORY_ROOT, "test", "nuts_source_oracle.jl"))

@testset "package-owned NUTS runtime" begin
    @test !isdefined(ReactiveKernels, :_NutsFrame)
    @test isdefined(ReactiveKernelsNUTSExamples, :_NutsFrame)
    @test isfile(joinpath(pkgdir(ReactiveKernelsNUTSExamples), "src",
                          "nuts_runtime", "kernel_nuts_native.jl"))

    position = [1.0, 2.0]
    momentum = [3.0, 4.0]
    metric = [2.0 0.0; 0.0 2.0]
    potential_gradient!(gradient, point) =
        (gradient .= 2 .* point; sum(abs2, point))

    group = reactive_nuts_group(
        potential_gradient!, metric, copy(position), copy(momentum))
    sampler = nuts_state(
        group;
        rng=Xoshiro(42),
        step_f=partial(leapfrog!; stepsize=0.1),
        max_depth=3,
    )
    chain = sample!(sampler, 1)

    oracle = NutsSourceOracle.State(
        pos=position, mom=momentum, metric=metric,
        stepsize=0.1, max_depth=3,
    )
    NutsSourceOracle.transition!(oracle, Xoshiro(42))
    expected = NutsSourceOracle.snapshot(oracle)
    observed = only(chain.diagnostics)

    @test only(eachcol(chain.samples)) == expected.pos
    @test observed.depth == expected.reached_depth
    @test observed.n_steps == expected.n_steps
    @test observed.acceptance_rate == expected.acceptance_rate
    @test observed.diverged == expected.diverged
    @test observed.energy_error == expected.dham
end

include(joinpath(REPOSITORY_ROOT, "test", "test_nuts_docs_fixture.jl"))
