module SMCParallelTuningTests
include("tuning_lifecycle_fixture.jl")
include("window_fixture.jl")
using .SMCTuningLifecycleFixture, ReactiveKernelsPPL, Random, Serialization, Test
using Treebars: with_progress, request_interrupt!, render_text
const F = SMCTuningLifecycleFixture
const EXT = F.EXT
options(n) = SMCOptions(nthreads=n, moves=1, max_tree_depth=4, history_limit=2)
tune(t=F.target(); nthreads=Threads.nthreads(), kwargs...) = F.tune(t;
    grid_start=:independent, options=options(nthreads), kwargs...)
same(a, b) = EXT._same(F.signature(a), F.signature(b))

@testset "Independent grid seeds and kernels do not depend on workers" begin
    t = F.target()
    # Five jobs leave uneven work for two or four workers.
    grid = (0.0, 0.01, 0.1, 0.5, 1.0)
    serial = tune(t; nthreads=1, temperatures=grid)
    @test serial.pilot.grid_start === :independent
    for nthreads in unique((1, 2, Threads.nthreads(), Threads.nthreads()+1))
        @test same(tune(t; nthreads, temperatures=grid), serial)
    end
    @test F.tune(t).pilot.grid_start === :sequential
    @test_throws ArgumentError F.tune(t; grid_start=:typo)
    @test_throws ArgumentError F.tune(t; pilot=:continuous, grid_start=:independent)
    init, rng = [0.1,-0.2], Xoshiro(1731)
    before = F.bytes((init, rng))
    @test same(tune(t; init, rng, temperatures=grid), serial)
    @test F.bytes((init, rng)) == before

    # Each knot starts at the caller's init, rather than a preceding pilot's
    # final position. The seeds are exactly the serially reserved stream.
    partial = tune(t; reference=nothing, callback=(p,b) -> b === :init)
    cp = partial.checkpoint
    expected_rng = copy(cp.initial_rng)
    @test [p.seed for p in cp.pilots] == [rand(expected_rng, UInt64) for _ in cp.pilots]
    @test all(p -> p.warmup === nothing ||
        p.warmup.position_and_gradient.q == init, cp.pilots)
end

@testset "Independent pilots join, checkpoint and change worker counts" begin
    t = F.target()
    expected = tune(t)
    # Stop at the reference, before the grid, and inside a grid pilot.
    for (phase,boundary) in ((:reference,:window), (:reference,:pilot), (:grid,:init), (:grid,:window))
        partial = tune(t; callback=(p,b) -> p.phase === phase && b === boundary)
        @test partial isa SMCTuningPartial
        before = F.bytes(partial)
        @test same(tune(t; nthreads=1, resume=partial), expected)
        @test same(tune(t; nthreads=4, resume=partial), expected)
        @test F.bytes(partial) == before
        @test length(partial.checkpoint.history) <= 2
        @test_throws ArgumentError F.tune(t; resume=partial)
        @test_throws ArgumentError tune(F.target(binding_id="changed"); resume=partial)
    end
    @testset "Several active continuations and serialized observers" begin
        entered = Set{Int}()
        callback_busy = Ref(false)
        partial = tune(t; reference=nothing, pilot_steps=64,
            recording_target=64, pilot_evaluations=16, callback=(p,b) -> begin
                @test !callback_busy[]
                callback_busy[] = true
                yield()
                b === :init && push!(entered,p.knot)
                callback_busy[] = false
                length(entered) >= min(2,Threads.nthreads())
            end)
        @test partial isa SMCTuningPartial
        @test count(p -> p.warmup !== nothing, partial.checkpoint.pilots) >= min(2,Threads.nthreads())
        expected_many = tune(t; nthreads=1, reference=nothing, pilot_steps=64,
            recording_target=64, pilot_evaluations=16)
        @test same(tune(t; nthreads=1, reference=nothing, pilot_steps=64,
            recording_target=64, pilot_evaluations=16, resume=partial), expected_many)
    end
    mktempdir() do directory
        callback = (p,b) -> p.phase === :grid && b === :init && error("observer failed")
        @test_throws TaskFailedException tune(t; checkpoint_dir=directory, callback)
        path = joinpath(directory,"tuning_latest.jls")
        saved = read(path)
        # All workers have joined: no delayed checkpoint writes after failure.
        yield()
        @test read(path) == saved
        @test_throws ArgumentError tune(t; checkpoint_dir=directory, resume=true, pilot_steps=5)
        @test read(path) == saved
        @test same(tune(t; checkpoint_dir=directory, resume=true, nthreads=1), expected)
        @test deserialize(path).phase === :complete
        @test all(p -> p.warmup === nothing, deserialize(path).pilots)
        @test same(tune(t; checkpoint_dir=directory, resume=true,
            callback=(p,b) -> error("completed pilot was rerun")), expected)
        @test readdir(directory) == ["tuning_latest.jls"]
    end
    rendered = String[]
    partial = with_progress(:state) do root
        tune(t; reference=nothing, progress=root, callback=(p,b) -> begin
            push!(rendered,render_text(root))
            request_interrupt!(root)
            false
        end)
    end
    @test partial isa SMCTuningPartial
    @test occursin("Grid pilot",first(rendered))
    @test same(tune(t; reference=nothing, resume=partial), tune(t; reference=nothing))
end

@testset "Independent tuning owns prepared, update and window evaluators" begin
    factory = F.F.GaussianFactory()
    t = F.F.target(factory)
    expected = tune(t; nthreads=1)
    @test same(tune(t; nthreads=4), expected)
    @test all(p -> !p.busy, factory.instances)
    invalid = F.F.GaussianFactory(fault=:gradient)
    @test_throws SMCNonfiniteError tune(F.F.target(invalid); reference=nothing)
    @test all(p -> !p.busy, invalid.instances)
    previous = F.target()
    current = F.F.target(F.F.GaussianFactory(location=[0.7,-0.3]); binding_id="data-v2")
    @test same(tune(current; previous, reference=nothing, nthreads=1),
        tune(current; previous, reference=nothing, nthreads=4))
    window = SMCWindowFixture.window()
    @test same(tune(window; reference=nothing, nthreads=1),
        tune(window; reference=nothing, nthreads=4))
end
end
