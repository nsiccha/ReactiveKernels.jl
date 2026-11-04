module SMCTuningLifecycleTests
include("tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture, ReactiveKernelsPPL, Random, Serialization, Test
using Treebars: with_progress, render_text, request_interrupt!
const F = SMCTuningLifecycleFixture
const EXT = F.EXT
@test EXT !== nothing

@testset "Adaptive tuning owns interrupted work and resumes exactly" begin
    t = F.target()
    expected = F.tune(t)
    @test expected isa SMCTuningSchedule
    for (phase, boundary, knot) in ((:reference, :init, 0),
            (:reference, :window, 0), (:grid, :init, 2), (:grid, :window, 2))
        interrupted = F.tune(t; callback=(p, b) -> p.phase == phase && b == boundary && p.knot == knot)
        @test interrupted isa SMCTuningPartial
        @test interrupted.progress.phase == phase
        @test interrupted.progress.boundary == boundary
        @test interrupted.progress.knot == knot
        boundary == :init && @test interrupted.progress.retained_draws == 0
        @test length(interrupted.checkpoint.history) <= 2
        @test length(interrupted.checkpoint.kernels) == (phase == :reference ? 0 : knot - 1)
        @test_throws ArgumentError SMCOptions(tuning=interrupted)
        before = F.bytes(interrupted)
        continued = NamedTuple[]
        resumed = F.tune(t; resume=interrupted, rng=Xoshiro(999),
            callback=(p,b) -> (push!(continued,p); nothing))
        @test first(continued) != interrupted.progress
        @test EXT._same(F.signature(resumed), F.signature(expected))
        @test F.bytes(interrupted) == before
    end
    # Stop after the fitted reference was committed, before the first grid seed.
    between = F.tune(t; callback=(p,b) -> p.phase == :reference && b == :pilot)
    @test between isa SMCTuningPartial
    @test between.checkpoint.phase == :grid
    @test between.checkpoint.warmup === nothing
    @test between.checkpoint.reference_work !== nothing
    seen = Symbol[]
    resumed = F.tune(t; resume=between, callback=(p,b) -> (push!(seen,p.phase); false))
    @test all(==(:grid), seen)
    @test EXT._same(F.signature(resumed), F.signature(expected))

    @testset "Validate before advancing the saved chain" begin
        @test_throws ArgumentError F.tune(F.target(binding_id="different"); resume=between)
        @test_throws ArgumentError F.tune(F.target(frame_id="different"); resume=between)
        @test_throws ArgumentError F.tune(t; resume=between, pilot_steps=5)
        @test_throws ArgumentError F.tune(t; resume=between, temperatures=(0.0,1.0))
        changed_density = F.F.target(F.F.GaussianFactory(location=[0.5,-0.6]))
        @test_throws ArgumentError F.tune(changed_density; resume=between)
        changed = merge(between.checkpoint, (; versions=merge(between.checkpoint.versions, (; nuts=v"999.0.0"))))
        @test_throws ArgumentError F.tune(t; resume=SMCTuningPartial(changed, between.progress))
        @test_throws ArgumentError F.tune(t; resume=true)
        @test_throws ArgumentError F.tune(t; resume=between, overwrite=true)
    end

    @testset "Atomic latest checkpoint and callback failure recovery" begin
        mktempdir() do directory
            marker = joinpath(directory,"caller-owned.txt")
            write(marker, "preserve")
            callback = (p,b) -> b == :init ? error("consumer stopped unexpectedly") : false
            @test_throws ErrorException F.tune(t; checkpoint_dir=directory, callback)
            path = joinpath(directory,"tuning_latest.jls")
            saved = read(path)
            @test deserialize(path).warmup !== nothing
            @test_throws ArgumentError F.tune(t; checkpoint_dir=directory)
            @test read(path) == saved
            @test_throws ArgumentError F.tune(t; checkpoint_dir=directory, resume=true, pilot_steps=5)
            @test read(path) == saved
            result = F.tune(t; checkpoint_dir=directory, resume=true)
            @test EXT._same(F.signature(result), F.signature(expected))
            cp = deserialize(path)
            @test cp.phase == :complete
            @test cp.warmup === nothing
            @test length(cp.history) <= 2
            @test sort(readdir(directory)) == ["caller-owned.txt", "tuning_latest.jls"]
            @test read(marker,String) == "preserve"
            # A completed checkpoint returns its schedule without another callback.
            again = F.tune(t; checkpoint_dir=directory, resume=true,
                callback=(p,b) -> error("completed work was rerun"))
            @test EXT._same(F.signature(again), F.signature(expected))
            replacement = F.tune(t; checkpoint_dir=directory, overwrite=true,
                callback=(p,b) -> b == :init)
            @test replacement isa SMCTuningPartial
            @test read(marker,String) == "preserve"
        end
    end

    @testset "Nested progress and cooperative subtree interruption" begin
        observed = String[]
        partial = with_progress(:state; description="Caller") do root
            result = F.tune(t; progress=root, callback=(p,b) -> begin
                push!(observed,render_text(root))
                request_interrupt!(root)
                false
            end)
            push!(observed,render_text(root))
            result
        end
        @test partial isa SMCTuningPartial
        @test occursin("SMC tuning", first(observed))
        @test occursin("Reference pilot", first(observed))
        @test occursin("MCMC", first(observed))
        @test occursin("partial", last(observed))
        # The phase label remains, but the transient pilot subtree is gone.
        @test count("Reference pilot", last(observed)) == 1
        @test !occursin("MCMC", last(observed))
        @test EXT._same(F.signature(F.tune(t; resume=partial)), F.signature(expected))
        pending = with_progress(:state) do root
            request_interrupt!(root)
            F.tune(t; progress=root)
        end
        @test pending isa SMCTuningPartial
        @test pending.progress.boundary == :pending
        @test pending.checkpoint.seed === nothing
        @test EXT._same(F.signature(F.tune(t; resume=pending)), F.signature(expected))
    end

    # Observers and checkpointing preserve caller inputs and the seed stream.
    init, rng = [0.1,-0.2], Xoshiro(1731)
    init_before, rng_before = copy(init), copy(rng)
    partial = F.tune(t; init, rng, callback=(p,b) -> true)
    @test init == init_before
    @test rand(copy(rng),UInt64,16) == rand(rng_before,UInt64,16)
    @test partial isa SMCTuningPartial
end

@testset "Later warm-up windows keep adaptation and recorded draws" begin
    config = (; pilot_steps=64, recording_target=64, pilot_evaluations=16)
    expected = F.signature(F.tune(; config...))
    for phase in (:reference, :grid)
        partial = F.tune(; config..., callback=(p,b) ->
            p.phase == phase && b == :window && p.window >= 2 &&
                (phase == :reference || p.knot == 2))
        @test partial isa SMCTuningPartial
        @test partial.progress.window >= 2
        @test length(partial.checkpoint.history) <= 2
        @test EXT._same(F.signature(F.tune(; config..., resume=partial)), expected)
    end
end

@testset "Adaptive update tuning restores both endpoints" begin
    previous = F.target()
    target = F.F.target(F.F.GaussianFactory(location=[0.7,-0.3]); binding_id="data-v2")
    expected = F.tune(target; previous, reference=nothing)
    @test smc_reference(expected) === nothing
    partial = F.tune(target; previous, reference=nothing,
        callback=(p,b) -> p.knot == 2 && b == :window)
    @test partial isa SMCTuningPartial
    @test partial.checkpoint.reference_work === nothing
    @test length(partial.checkpoint.kernels) == 1
    @test EXT._same(F.signature(F.tune(target; previous, reference=nothing,
        resume=partial)), F.signature(expected))
    @test_throws ArgumentError F.tune(target; previous=F.target(binding_id="wrong"),
        reference=nothing, resume=partial)
    @test_throws ArgumentError F.tune(target; previous=F.F.target(
        F.F.GaussianFactory(location=[0.5,-0.6])), reference=nothing, resume=partial)
end
end
