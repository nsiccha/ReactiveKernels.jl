module SMCThreadedTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using LinearAlgebra, Test, Serialization, MutatingFunctions
include("window_fixture.jl")
using .SMCWindowFixture

const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)

function target(; nonallocating=false, shift=0.0)
    X = [sin(i*j/11) for i in 1:36, j in 1:5]
    y = [cos(i/5)+shift for i in 1:36]
    source = quote
        theta[1:5] .~ Normal.(0, 1)
        y .~ Normal.(X * theta, 1)
    end
    data = (; X, y)
    plan = bind_data(lower_rkppl(source, data; conditioned=(:y,)), data)
    built = build_kernel(plan)
    t = prepare_smc(built, plan, zeros(5); backend=BACKEND,
        model_id="threaded-gaussian", binding_id="shift-$shift", nonallocating)
    return t, X, y
end

function same(a, b)
    @test a.positions == b.positions
    @test a.weights == b.weights
    @test a.history == b.history
    @test a.log_normalizer == b.log_normalizer
    @test a.lineage == b.lineage && a.ancestors == b.ancestors
    @test rand(copy(a.rng), UInt64, 8) == rand(copy(b.rng), UInt64, 8)
end

@testset "Concurrent SMC target gradients own their scratch" begin
    Threads.nthreads() > 1 && @test isdefined(Enzyme.Compiler, :RestoreSafepointEffectsPass)
    for nonallocating in (false, true)
        t, X, y = target(; nonallocating)
        copies = [EXT._worker_target(t) for _ in 1:Threads.nthreads()]
        original_X, original_y = copy(X), copy(y)
        # Warm one derivative, then exercise concurrent calls at different
        # positions against the independent Gaussian gradient.
        EXT._gradient(t, zeros(5), zeros(5))
        tasks = map(eachindex(copies)) do worker
            Threads.@spawn begin
                g = zeros(5)
                for k in 1:64
                    u = fill((worker-k)/100, 5)
                    value = EXT._gradient(copies[worker], g, u)
                    expected = -5log(2pi)/2-sum(abs2,u)/2 -
                        length(y)*log(2pi)/2-sum(abs2,y-X*u)/2
                    isapprox(value, expected; rtol=1e-12) || return false
                    isapprox(g, -u+X'*(y-X*u); rtol=1e-11, atol=1e-11) || return false
                    yield()
                end
                true
            end
        end
        @test all(fetch, tasks)
        @test X == original_X && y == original_y
        @test all(q -> q.query !== t.query && q.query.ad !== t.query.ad, copies)
    end
end

@testset "Threaded mutation, uneven particle counts and checkpoint replay" begin
    t, X, y = target()
    rng = Xoshiro(315)
    saved_rng = copy(rng)
    Random.seed!(619)
    default_rng = copy(Random.default_rng())
    options(w) = SMCOptions(nthreads=w, moves=2, stepsize=0.15, max_tree_depth=5)
    serial = smc(t; rng, particles=37, options=options(1))
    for workers in unique((1, min(2,Threads.nthreads()), Threads.nthreads()))
        start = initialize_smc(t; rng, particles=37, options=options(workers))
        saved = copy(start.positions)
        partial = smc_step(start)
        cp = smc_checkpoint(partial)
        buffer = IOBuffer(); serialize(buffer,cp); seekstart(buffer)
        final = smc(partial)
        same(final, smc(resume_smc(t,deserialize(buffer))))
        same(final, smc(start))
        @test final.positions == serial.positions
        @test all(h.nthreads == min(workers,37) for h in final.history)
        @test all(h.moves == 74 for h in final.history)
        @test sum(h.evaluations for h in final.history) == sum(h.evaluations for h in serial.history)
        @test start.positions == saved
        next_target, _, _ = target(; shift=0.2)
        next = update_smc(final,next_target; stages=1)
        resumed = resume_smc(next_target,smc_checkpoint(next); previous=t)
        same(smc(next), smc(resumed))
    end
    @test rand(copy(rng),UInt64,8) == rand(saved_rng,UInt64,8)
    @test rand(copy(Random.default_rng()),UInt64,8) == rand(default_rng,UInt64,8)
    @test_throws ArgumentError SMCOptions(nthreads=0)
    @test_throws ArgumentError SMCOptions(nthreads=1.5)
end

@testset "Carried-window mutation owns likelihood and prior scratch" begin
    t = SMCWindowFixture.window()
    fork = EXT._worker_target(t)
    @test fork.query !== t.query && fork.query.ad !== t.query.ad
    @test fork.prior !== t.prior && fork.prior.ad !== t.prior.ad
    @test fork.signature === t.signature
    options(w) = SMCOptions(nthreads=w, moves=2, stepsize=0.15, max_tree_depth=5)
    serial = smc(t; rng=Xoshiro(911), particles=37, options=options(1))
    initial = initialize_smc(t; rng=Xoshiro(911), particles=37,
        options=options(Threads.nthreads()))
    partial = smc_step(initial)
    final = smc(partial)
    @test final.positions == serial.positions
    @test final.log_normalizer == serial.log_normalizer
    @test sum(h.evaluations for h in final.history) == sum(h.evaluations for h in serial.history)
    same(final, smc(resume_smc(t, smc_checkpoint(partial))))
end
end
