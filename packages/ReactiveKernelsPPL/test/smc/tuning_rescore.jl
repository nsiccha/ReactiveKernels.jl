module SMCTuningRescoreTests
include("tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture, ReactiveKernelsPPL, Random, Serialization, Test
using LogDensityProblems, WarmupHMC
const F = SMCTuningLifecycleFixture
const EXT = F.EXT

function moved(partial)
    cp = partial.checkpoint
    s, t = copy(cp.source_scores), copy(cp.target_scores)
    s[1] = nextfloat(s[1], 2)
    t[end] = prevfloat(t[end])
    SMCTuningPartial(merge(cp, (; source_scores=s, target_scores=t)), partial.progress)
end
message(f) = try f(); "" catch e; e isa ArgumentError || rethrow(); e.msg end
same(a, b) = EXT._same(F.signature(a), F.signature(b))
roundtrip(x) = deserialize(IOBuffer(F.bytes(x)))

@testset "Tuning re-anchoring preserves exact same-host continuation" begin
    t = F.target()
    for strategy in (:sequential, :independent)
        kwargs = (; grid_start=strategy)
        expected = F.tune(t; kwargs...)
        for (phase, boundary) in ((:reference, :init), (:reference, :window),
                (:reference, :pilot), (:grid, :init), (:grid, :window))
            partial = F.tune(t; kwargs..., callback=(p,b) -> p.phase == phase && b == boundary)
            drifted = moved(partial)
            before = F.bytes(drifted)
            refusal = message(() -> F.tune(t; kwargs..., resume=drifted))
            @test occursin("2 of", refusal)
            @test occursin("at probe", refusal)
            @test occursin("rescore_smc_tuning_checkpoint", refusal)
            anchored = rescore_smc_tuning_checkpoint(t, drifted)
            @test F.bytes(drifted) == before
            @test anchored.progress == partial.progress
            @test anchored.checkpoint.source_scores == partial.checkpoint.source_scores
            @test anchored.checkpoint.target_scores == partial.checkpoint.target_scores
            @test only(anchored.checkpoint.rescored).changed == 2
            @test 0 < only(anchored.checkpoint.rescored).max_rel < 1e-14
            @test same(F.tune(t; kwargs..., resume=roundtrip(anchored)), expected)
            payloads = Any[anchored.checkpoint.warmup]
            append!(payloads, (p.warmup for p in anchored.checkpoint.pilots))
            @test all(p -> p === nothing || length(p.rescored) == 1, payloads)
            again = rescore_smc_tuning_checkpoint(t, anchored; rtol=0)
            @test length(again.checkpoint.rescored) == 2
            @test last(again.checkpoint.rescored).changed == 0
        end
    end
end

@testset "Tuning guards and atomic directory replacement" begin
    t = F.target()
    partial = F.tune(t; callback=(p,b) -> b == :window)
    drifted = moved(partial)
    @test occursin("more than rtol=0.0", message(() -> rescore_smc_tuning_checkpoint(t, drifted; rtol=0)))
    for rtol in (-1.0, Inf, NaN)
        @test_throws ArgumentError rescore_smc_tuning_checkpoint(t, drifted; rtol)
    end
    @test_throws ArgumentError rescore_smc_tuning_checkpoint(F.target(binding_id="changed"), drifted)
    @test_throws ArgumentError rescore_smc_tuning_checkpoint(F.target(frame_id="changed"), drifted)
    changed = F.F.target(F.F.GaussianFactory(location=[0.5, -0.6]))
    @test_throws ArgumentError rescore_smc_tuning_checkpoint(changed, drifted)
    cp = drifted.checkpoint
    for edit in ((; schema_version=99), (; phase=:invalid),
            (; versions=merge(cp.versions, (; nuts=v"999.0.0"))),
            (; source_scores=cp.source_scores[1:end-1]))
        @test_throws ArgumentError rescore_smc_tuning_checkpoint(t,
            SMCTuningPartial(merge(cp, edit), drifted.progress))
    end
    mktempdir() do input
        mktempdir() do output
            source_path = joinpath(input, "tuning_latest.jls")
            dest_path = joinpath(output, "tuning_latest.jls")
            serialize(source_path, cp)
            write(dest_path, "existing destination")
            write(joinpath(input, "caller.txt"), "source marker")
            write(joinpath(output, "caller.txt"), "destination marker")
            before, dest_before = read(source_path), read(dest_path)
            @test_throws ArgumentError rescore_smc_tuning_checkpoint(changed, input; output_dir=output)
            @test read(source_path) == before && read(dest_path) == dest_before
            anchored = rescore_smc_tuning_checkpoint(t, input; output_dir=output)
            @test read(source_path) == before
            @test read(dest_path) == F.bytes(anchored.checkpoint)
            expected = F.tune(t; resume=partial)
            @test same(F.tune(t; checkpoint_dir=output, resume=true), expected)
            completed = rescore_smc_tuning_checkpoint(t, output)
            @test completed.checkpoint.phase == :complete
            @test length(completed.checkpoint.rescored) == 2
            @test same(F.tune(t; checkpoint_dir=output, resume=true,
                callback=(p,b) -> error("completed work reran")), expected)
            @test_throws ArgumentError F.tune(t; resume=anchored, pilot_steps=5)
            rescore_smc_tuning_checkpoint(t, input)
            @test read(source_path) != before
            @test read(joinpath(input, "caller.txt"), String) == "source marker"
            @test read(joinpath(output, "caller.txt"), String) == "destination marker"
            @test sort(readdir(input)) == ["caller.txt", "tuning_latest.jls"]
            @test sort(readdir(output)) == ["caller.txt", "tuning_latest.jls"]
        end
    end
    mktempdir() do missing
        @test_throws ArgumentError rescore_smc_tuning_checkpoint(t, missing)
    end
end

@testset "Every active independent pilot is re-anchored at its own temperature" begin
    entered = Set{Int}()
    t = F.target()
    kwargs = (; grid_start=:independent, reference=nothing,
        temperatures=(0.0, 0.7, 1.0), pilot_steps=64, recording_target=64,
        pilot_evaluations=16, options=SMCOptions(nthreads=Threads.nthreads(),
            moves=1, max_tree_depth=4, history_limit=2))
    partial = F.tune(t; kwargs..., callback=(p,b) -> begin
        b == :init && push!(entered, p.knot)
        length(entered) >= min(2, Threads.nthreads())
    end)
    active = findall(p -> p.warmup !== nothing, partial.checkpoint.pilots)
    @test length(active) >= min(2, Threads.nthreads())
    anchored = rescore_smc_tuning_checkpoint(t, moved(partial))
    @test all(i -> length(anchored.checkpoint.pilots[i].warmup.rescored) == 1, active)
    @test same(F.tune(t; kwargs..., resume=anchored), F.tune(t; kwargs..., resume=partial))
end

# Simulate another CPU's value/gradient rounding through the public target
# protocol, without editing a WarmupHMC payload's cached evaluations.
struct RoundedProblem{P}
    inner::P
    ulps::Int
    gradient_scale::Float64
end
LogDensityProblems.capabilities(::Type{<:RoundedProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::RoundedProblem) = LogDensityProblems.dimension(p.inner)
LogDensityProblems.logdensity(p::RoundedProblem, q) = nextfloat(LogDensityProblems.logdensity(p.inner, q), p.ulps)
function LogDensityProblems.logdensity_and_gradient(p::RoundedProblem, q)
    value, gradient = LogDensityProblems.logdensity_and_gradient(p.inner, q)
    return nextfloat(value, p.ulps), nextfloat.(gradient .* p.gradient_scale, p.ulps)
end
function rounded_target(; ulps=0, gradient_scale=1.0)
    factory = () -> RoundedProblem(F.F.GaussianFactory()(), ulps, gradient_scale)
    return prepare_smc(factory, zeros(2); backend=F.F.BACKEND,
        model_id="roundoff-gaussian", binding_id="data", frame_id="x-y")
end

@testset "Nested pilot re-scoring uses the producer contract" begin
    writer, reader = rounded_target(), rounded_target(ulps=2)
    for strategy in (:sequential, :independent)
        partial = F.tune(writer; grid_start=strategy, reference=nothing,
            temperatures=(0.0, 0.7, 1.0), callback=(p,b) -> b == :window && p.temperature > 0)
        before = F.bytes(partial)
        @test_throws ArgumentError F.tune(reader; grid_start=strategy, reference=nothing,
            temperatures=(0.0, 0.7, 1.0), resume=partial)
        anchored = rescore_smc_tuning_checkpoint(reader, partial)
        @test F.bytes(partial) == before
        @test only(anchored.checkpoint.rescored).changed > 0
        @test F.tune(reader; grid_start=strategy, reference=nothing,
            temperatures=(0.0, 0.7, 1.0), resume=roundtrip(anchored)) isa SMCTuningSchedule
        @test_throws ArgumentError rescore_smc_tuning_checkpoint(reader, partial; rtol=0)
        # Values agree exactly; only nested gradient checks can reject this.
        bad_gradient = rounded_target(gradient_scale=1.1)
        @test_throws ArgumentError rescore_smc_tuning_checkpoint(bad_gradient, partial)
        @test F.bytes(partial) == before
        mktempdir() do directory
            path = joinpath(directory, "tuning_latest.jls")
            serialize(path, partial.checkpoint)
            saved = read(path)
            @test_throws ArgumentError rescore_smc_tuning_checkpoint(bad_gradient, directory)
            @test read(path) == saved
        end
    end
    previous = F.target()
    target = F.F.target(F.F.GaussianFactory(location=[0.7, -0.3]); binding_id="data-v2")
    partial = F.tune(target; previous, reference=nothing, callback=(p,b) -> b == :window)
    @test_throws ArgumentError rescore_smc_tuning_checkpoint(target, partial)
    anchored = rescore_smc_tuning_checkpoint(target, moved(partial); previous)
    @test same(F.tune(target; previous, reference=nothing, resume=anchored),
        F.tune(target; previous, reference=nothing, resume=partial))
end
end
