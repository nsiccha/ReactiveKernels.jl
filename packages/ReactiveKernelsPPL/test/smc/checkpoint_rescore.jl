# Runnable alone in the optional SMC integration environment, or from runtests.
# Re-anchoring a checkpoint whose stored endpoint scores differ in their last
# bits, as one written on a host with different CPU code generation does.
module SMCCheckpointRescoreTests
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, Test, Serialization

const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)
const DATA = [0.9, 1.6, 0.2, 1.3, 0.6, 1.1]
const GAUSSIAN = quote
    mu ~ Normal(0, 1)
    y .~ Normal.(mu, 1)
end

function target_for(data; binding_id, built=nothing)
    plan = bind_data(lower_rkppl(GAUSSIAN, data; conditioned=(:y,)), data)
    built === nothing && (built = build_kernel(plan))
    target = prepare_smc(built, plan, zeros(built.layout.total);
        backend=BACKEND, model_id="gaussian-v1", binding_id)
    return (; target, built)
end

snapshot(s) = (; s.positions, s.weights, s.temperature, s.log_normalizer,
    s.ancestors, s.lineage, s.history, rng=rand(copy(s.rng), UInt64, 8))
message(f) = try f(); "" catch e; e isa ArgumentError || rethrow(); e.msg end

# Stand-in for another host's roundoff: move a few stored scores by 1-2 ulp.
function drift(cp; source=(3,), target=(1, 2, 7))
    s, t = copy(cp.source_scores), copy(cp.target_scores)
    for i in source; s[i] = nextfloat(s[i], 2); end
    for i in target; t[i] = prevfloat(t[i], 1 + isodd(i)); end
    return merge(cp, (; source_scores=s, target_scores=t))
end

@testset "SMC checkpoint re-anchoring across code generation" begin
    f = target_for((; y=DATA); binding_id="six")
    partial = smc(initialize_smc(f.target; rng=Xoshiro(121), particles=200,
        reference=SMCReference([0.3], [1.2])); stages=1)
    @test 0 < partial.temperature < 1
    cp = smc_checkpoint(partial)
    expected = snapshot(smc(partial))
    moved = drift(cp)
    stored = (copy(moved.source_scores), copy(moved.target_scores))

    # Plain resume stays bit-exact and names the deviation and the remedy.
    refusal = message(() -> resume_smc(f.target, moved))
    @test startswith(refusal, "SMC checkpoint endpoint densities differ")
    @test occursin("4 of 400 stored endpoint scores differ", refusal)
    @test occursin("rescore_smc_checkpoint", refusal)

    anchored = rescore_smc_checkpoint(f.target, moved)
    @test (moved.source_scores, moved.target_scores) == stored
    @test !haskey(moved, :rescored)
    @test anchored.source_scores == cp.source_scores
    @test anchored.target_scores == cp.target_scores
    record = only(anchored.rescored)
    @test record.changed == 4
    @test 0 < record.max_abs && 0 < record.max_rel < 1e-15
    @test record.rtol == 1e-10 && record.cpu == Sys.CPU_NAME
    buffer = IOBuffer()
    serialize(buffer, anchored)
    seekstart(buffer)
    @test snapshot(smc(resume_smc(f.target, deserialize(buffer)))) == expected

    # A second crossing appends its record; an exact checkpoint records zero.
    again = rescore_smc_checkpoint(f.target, drift(anchored; source=(), target=(5,)))
    @test length(again.rescored) == 2 && again.rescored[1] == record
    @test again.rescored[2].changed == 1
    @test only(rescore_smc_checkpoint(f.target, cp).rescored).changed == 0

    # The tolerance is explicit and bounded.
    @test occursin("by more than rtol=0.0", message(() -> rescore_smc_checkpoint(f.target, moved; rtol=0)))
    @test rescore_smc_checkpoint(f.target, moved; rtol=1e-14).target_scores == cp.target_scores
    @test_throws ArgumentError rescore_smc_checkpoint(f.target, moved; rtol=-1e-10)
    @test_throws ArgumentError rescore_smc_checkpoint(f.target, moved; rtol=NaN)

    # Changed data under the same ids is still refused, without the roundoff hint.
    wrongdata = target_for((; y=DATA .+ 1); binding_id="six", built=f.built)
    refusal = message(() -> rescore_smc_checkpoint(wrongdata.target, moved))
    @test startswith(refusal, "SMC checkpoint endpoint densities differ by more than rtol=1.0e-10")
    @test occursin("at particle", refusal)
    @test !occursin("rescore_smc_checkpoint", message(() -> resume_smc(wrongdata.target, cp)))

    # Every other resume guard runs first.
    @test_throws ArgumentError rescore_smc_checkpoint(f.target, merge(moved, (; schema_version=2)))
    @test_throws ArgumentError rescore_smc_checkpoint(f.target,
        merge(moved, (; versions=merge(moved.versions, (; ppl=v"0.0.1")))))
    @test_throws ArgumentError rescore_smc_checkpoint(f.target, moved; previous=f.target)
    @test_throws ArgumentError rescore_smc_checkpoint(f.target,
        merge(moved, (; target_scores=moved.target_scores[1:end-1])))

    # A data-update checkpoint needs its previous target here too.
    old = target_for((; y=DATA[1:2]); binding_id="prefix-2", built=f.built)
    previous = smc(old.target; rng=Xoshiro(343), particles=200)
    updating = smc_step(update_smc(previous, f.target; stages=0))
    update = drift(smc_checkpoint(updating))
    @test_throws ArgumentError rescore_smc_checkpoint(f.target, update)
    @test_throws ArgumentError resume_smc(f.target, update; previous=old.target)
    reanchored = rescore_smc_checkpoint(f.target, update; previous=old.target)
    @test snapshot(smc(resume_smc(f.target, reanchored; previous=old.target))) ==
        snapshot(smc(updating))
end

end
