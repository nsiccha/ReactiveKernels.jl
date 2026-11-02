module PreparedSMCTests
using ReactiveKernelsPPL, LogDensityProblems, Random, Test, Statistics, Serialization
include("prepared_fixture.jl")
const F = PreparedSMCFixture
const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)

@testset "Prepared density, supplied gradients and owned workspaces" begin
    @test EXT !== nothing
    factory = F.GaussianFactory()
    t = F.target(factory)
    @test length(factory.instances) == 1
    @test t.problem.evaluations == t.problem.gradients == 0
    u = [0.7,-0.2]
    g = zeros(2)
    @test EXT._density(t,u) ≈ F.value(t.problem,u)
    @test EXT._gradient(t,g,u) ≈ F.value(t.problem,u)
    @test g ≈ -(u-factory.location)./factory.scale.^2
    bridge = EXT.BridgeTarget(nothing,t,1.0,2)
    _, first_gradient = LogDensityProblems.logdensity_and_gradient(bridge,u)
    saved = copy(first_gradient)
    LogDensityProblems.logdensity_and_gradient(bridge,-u)
    @test first_gradient == saved
    @test first_gradient !== t.problem.gradient

    a = initialize_smc(t; rng=Xoshiro(18),particles=31,options=F.options())
    b = initialize_smc(t; rng=Xoshiro(18),particles=31,options=F.options())
    restored = resume_smc(t, smc_checkpoint(a))
    @test all(p !== t.problem for p in (a.target.problem,b.target.problem,restored.target.problem))
    @test a.target.problem !== b.target.problem !== restored.target.problem
    @test a.target.problem.location === factory.location # Read-only data shared.
    @test a.target.problem.gradient !== b.target.problem.gradient
    @test F.snapshot(smc(a)) == F.snapshot(smc(b)) == F.snapshot(smc(restored))

    # Concurrent evaluations must not overwrite another worker's returned buffer.
    workers = [EXT._worker_target(t) for _ in 1:4]
    tasks = map(enumerate(workers)) do (i, worker)
        Threads.@spawn begin
            for j in 1:16
                x = [i/10,j/20]
                gradient = zeros(2)
                EXT._gradient(worker,gradient,x)
                gradient ≈ -(x-factory.location)./factory.scale.^2 || return false
            end
            true
        end
    end
    @test all(fetch, tasks)
    @test length(unique(objectid(p.gradient) for p in factory.instances)) == length(factory.instances)
end

@testset "Prepared Gaussian posterior, serial/threaded mutation and replay" begin
    factory = F.GaussianFactory()
    t = F.target(factory)
    rng = Xoshiro(991)
    saved_rng = copy(rng)
    Random.seed!(375)
    default_rng = copy(Random.default_rng())
    initial = initialize_smc(t; rng,particles=1800,options=F.options())
    partial = smc_step(initial)
    @test 0 < partial.temperature < 1
    final = smc(partial)
    @test vec(mean(final.positions;dims=2)) ≈ factory.location atol=0.06
    @test vec(var(final.positions;dims=2)) ≈ factory.scale.^2 atol=0.06
    @test final.log_normalizer ≈ factory.logmass atol=0.08
    @test F.snapshot(final) == F.snapshot(smc(resume_smc(t,smc_checkpoint(partial))))
    @test rand(copy(rng),UInt64,8) == rand(saved_rng,UInt64,8)
    @test rand(copy(Random.default_rng()),UInt64,8) == rand(default_rng,UInt64,8)
    @test initial.temperature == 0
    @test t.problem.evaluations == t.problem.gradients == 0

    serial = smc(t; rng=Xoshiro(134), particles=37,options=F.options())
    threaded = smc(t; rng=Xoshiro(134),particles=37,options=F.options(Threads.nthreads()))
    @test serial.positions == threaded.positions
    @test serial.log_normalizer == threaded.log_normalizer
    @test all(h.nthreads == min(Threads.nthreads(),37) for h in threaded.history)
    @test sum(h.evaluations for h in serial.history) == sum(h.evaluations for h in threaded.history)

    next = F.target(F.GaussianFactory(;location=[0.6,-0.3]);binding_id="new-data")
    updating = update_smc(serial,next;stages=0)
    @test updating.source.problem !== serial.target.problem
    @test updating.target.problem !== next.problem
    @test F.snapshot(smc(updating)) == F.snapshot(smc(resume_smc(next,
        smc_checkpoint(updating);previous=t)))
    carried = carry_smc(final;calibration_id="gaussian-summary")
    @test carried.frame == EXT._frame(t)
    @test carried.reference.location == vec(final.positions*final.weights)
    @test_throws ArgumentError carry_smc(final;calibration_id="named",parameters=(:x,))
end

@testset "Independent prepared-density pilots and fixed production tuning" begin
    factory = F.GaussianFactory()
    t = F.target(factory)
    production = initialize_smc(t;rng=Xoshiro(9),particles=37,options=F.options())
    before = F.snapshot(production)
    count = length(factory.instances)
    init, rng = zeros(2), Xoshiro(817)
    rng_before = copy(rng)
    schedule = tune_smc(t; rng,init,pilot_steps=12,temperatures=(0.0,0.5,1.0),
        options=F.options())
    @test length(factory.instances) == count+1
    @test factory.instances[end].gradients > 0
    @test production.target.problem.gradients == 0
    @test t.problem.gradients == 0
    @test F.snapshot(production) == before
    @test init == zeros(2)
    @test rand(copy(rng),UInt64,8) == rand(rng_before,UInt64,8)
    options = SMCOptions(tuning=schedule,nthreads=Threads.nthreads())
    partial = smc(t;rng=Xoshiro(911),particles=97,options,stages=1)
    @test all(h.kernel == schedule.kernels[h.tuning_index] for h in partial.history)
    @test F.snapshot(smc(partial)) == F.snapshot(smc(resume_smc(t,smc_checkpoint(partial))))
    next = F.target(F.GaussianFactory(;scale=[0.8,1.0]);binding_id="next")
    updates = tune_smc(next;previous=t,rng=Xoshiro(891),init,
        pilot_steps=8,temperatures=(0.0,1.0),options=F.options())
    @test smc_reference(updates) === nothing
    complete = smc(production)
    updating = update_smc(complete,next;options=SMCOptions(tuning=updates,nthreads=1),stages=0)
    @test F.snapshot(smc(updating)) == F.snapshot(smc(resume_smc(next,
        smc_checkpoint(updating);previous=t)))
    @test_throws ArgumentError initialize_smc(next;rng=Xoshiro(1),options)

    adaptive = tune_smc(t; rng=Xoshiro(987),init,pilot=:adaptive,reference=:pilot,
        temperatures=(0.0,1.0),pilot_steps=16,pilot_evaluations=100,
        recording_target=16,max_reflections=2,options=F.options())
    @test adaptive.pilot.method == :adaptive
    @test adaptive.pilot.reference_work.retained_draws >= 16
    @test t.problem.gradients == 0
    @test all(k -> length(k.metric.reflections) <= 2, adaptive.kernels)
    adapted = smc(t;rng=Xoshiro(887),particles=37,stages=1,
        options=SMCOptions(tuning=adaptive,nthreads=1))
    @test F.snapshot(smc(adapted)) == F.snapshot(smc(resume_smc(t,smc_checkpoint(adapted))))
end

struct ValueOnly end
LogDensityProblems.capabilities(::Type{ValueOnly}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::ValueOnly) = 2
struct HigherOrder end
LogDensityProblems.capabilities(::Type{HigherOrder}) = LogDensityProblems.LogDensityOrder{2}()
LogDensityProblems.dimension(::HigherOrder) = 2

@testset "Prepared identity, dimensions and nonfinite diagnostics" begin
    t = F.target()
    state = smc(t;rng=Xoshiro(7),particles=31,options=F.options())
    checkpoint = smc_checkpoint(state)
    for keywords in ((;model_id="changed-approximation"),(;binding_id="changed-data"),
            (;frame_id="reordered"))
        other = F.target(;keywords...)
        @test_throws ArgumentError resume_smc(other,checkpoint)
        if !haskey(keywords,:binding_id)
            @test_throws ArgumentError update_smc(state,other)
        end
    end
    changed = F.target(F.GaussianFactory(;logmass=1.7))
    @test_throws ArgumentError resume_smc(changed,checkpoint)
    @test_throws DimensionMismatch F.target(;u0=zeros(3))
    for keywords in ((;model_id=""),(;binding_id=""),(;frame_id=""))
        @test_throws ArgumentError F.target(;keywords...)
    end
    args = (;backend=F.BACKEND,model_id="m",binding_id="b",frame_id="f")
    @test_throws ArgumentError prepare_smc(()->ValueOnly(),zeros(2);args...)
    @test prepare_smc(()->HigherOrder(),zeros(2);args...) isa EXT.PreparedSMCTarget
    factory = F.GaussianFactory()
    mutable_dimension = F.target(factory)
    push!(factory.location,0)
    @test_throws DimensionMismatch initialize_smc(mutable_dimension;rng=Xoshiro(3))

    for fault in (:density,:gradient)
        failed = initialize_smc(F.target(F.GaussianFactory(;fault));
            rng=Xoshiro(22),particles=13,options=F.options(Threads.nthreads()))
        before = F.snapshot(failed)
        err = try smc_step(failed); nothing catch e; e end
        @test err isa SMCNonfiniteError
        @test err.quantity == fault
        @test F.snapshot(failed) == before
    end
    unsupported = F.target(F.GaussianFactory(;fault=:support))
    bridge = EXT.BridgeTarget(nothing,unsupported,1.0,2)
    @test first(LogDensityProblems.logdensity_and_gradient(bridge,zeros(2))) == -Inf
    wrongshape = F.target(F.GaussianFactory(;fault=:shape))
    @test_throws DimensionMismatch EXT._gradient(wrongshape,zeros(2),zeros(2))
    fixed = F.target(F.GaussianFactory(;location=Float64[],scale=Float64[]))
    fixedfit = smc(fixed;rng=Xoshiro(66),particles=10,options=F.options())
    @test size(fixedfit.positions) == (0,10)
    @test fixedfit.log_normalizer ≈ 0.7
    @test all(h.moves == 0 for h in fixedfit.history)
end

@testset "Prepared target checkpoints rebuild in a fresh process" begin
    mktempdir() do directory
        path = joinpath(directory,"prepared.bin")
        fixture = joinpath(@__DIR__,"prepared_process.jl")
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $fixture`
        @test success(pipeline(`$cmd write $path`;stdout=stdout,stderr=stderr))
        @test success(pipeline(`$cmd read $path`;stdout=stdout,stderr=stderr))
    end
end
end
