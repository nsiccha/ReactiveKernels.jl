# The emitted namespace deliberately does not exist in the reading process.
# A named law lives in its owner's module, just as a BRM law does; its authored
# arguments call a helper resolved in the fresh model module.
include("tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture, ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface, Enzyme, Random, Serialization, Test
const F = SMCTuningLifecycleFixture
const PPL = ReactiveKernelsPPL

module GeneratedSMCGeometry
using ReactiveKernelsPPL
struct NormalVector{T}
    location::T
end
ReactiveKernelsPPL.sampling_logdensity(d::NormalVector, x) =
    -length(x)*log(2pi)/2 - sum(abs2, x .- d.location)/2
to_model(u, shape, location) = copy(u)
from_model(x, shape, location) = copy(x)
logjac(u, shape, location) = 0.0
ReactiveKernelsPPL.sampling_geometry(::Type{NormalVector}, args, shape) =
    ParameterGeometry(shape, :(length($(args[1]))); support=:real,
        constrain=to_model, unconstrain=from_model, logjac)
end

function generated_target(; shift=0.25, binding_id="rows-v1")
    mod = Module(gensym(:RKEmittedModels))
    Core.eval(mod, :(const NormalVector = $(GeneratedSMCGeometry.NormalVector)))
    Core.eval(mod, :(center(x) = x .+ $shift))
    source = quote
        a[1:2] ~ NormalVector(center(zeros(2)))
        y .~ Normal.(a[1] .+ a[2] .* x, 1.0)
    end
    data = (; x=[-1.0, 0.0, 1.0], y=[0.2, 0.4, 0.1])
    plan = bind_data(lower_rkppl(source, data; conditioned=(:y,), mod), data)
    built = build_kernel(plan)
    target = prepare_smc(built, plan, [0.1,-0.2]; backend=AutoEnzyme(mode=Enzyme.Reverse),
        model_id="generated-helper-model", binding_id)
    (; target, built, plan, mod)
end

function work_snapshot(partial)
    cp, w = partial.checkpoint, partial.checkpoint.warmup
    (; phase=cp.phase, knot=cp.knot, seed=cp.seed, history=cp.history,
        outer_rng=rand(copy(cp.rng), UInt64, 16),
        sampler_rng=rand(copy(w.rng), UInt64, 16),
        position=w.position_and_gradient.q, gradient=w.position_and_gradient.∇ℓq,
        posterior=copy(w.posterior_position), halo=copy(w.halo_position),
        evaluations=w.total_evaluation_counter, window=w.outer_counter)
end

mode, directory = ARGS
if mode == "write"
    @testset "Generated helper checkpoint producer" begin
        f = generated_target()
        expected = F.signature(F.tune(f.target))
        serialize(joinpath(directory,"expected.jls"), expected)
        partial = F.tune(f.target; checkpoint_dir=directory,
            callback=(p,b) -> b == :init)
        @test partial isa SMCTuningPartial
        @test partial.progress.boundary == :init
        @test partial.progress.retained_draws == 0
        serialize(joinpath(directory,"partial.jls"), partial)
        # Preserve a sample of the old identity representation. Reading it
        # needs the vanished namespace; neither reader aliases nor migration
        # of that legacy representation are part of this repair.
        serialize(joinpath(directory,"legacy-rhs.jls"), only(f.built.layout.entries).sampling.rhs)
        next = F.tune(generated_target().target; resume=partial,
            callback=(p,b) -> b == :window)
        @test next isa SMCTuningPartial
        @test next.progress.boundary == :window
        @test next.progress.window > 0
        @test next.progress.evaluations > 0
        serialize(joinpath(directory,"next-window.jls"), work_snapshot(next))
        @test F.EXT._same(F.signature(F.tune(f.target; resume=next)), expected)

        carry = carry_smc(f.target, SMCReference(zeros(2), ones(2)); calibration_id="pilot-v1")
        serialize(joinpath(directory,"carry.jls"), carry)
        batch = initialize_smc(f.target; rng=Xoshiro(191), particles=8)
        serialize(joinpath(directory,"batch.jls"), smc_checkpoint(batch))
        records = PPL._smc_coordinate_records(f.built.layout)
        serialize(joinpath(directory,"coordinates.jls"), records)
        @test F.EXT._same(records, PPL._smc_coordinate_records(generated_target().built.layout))
    end
elseif mode == "read"
    @testset "Generated helper fresh-process checkpoint consumer" begin
        # Deserialize before constructing any replacement model module.
        partial = deserialize(joinpath(directory,"partial.jls"))
        expected = deserialize(joinpath(directory,"expected.jls"))
        carry = deserialize(joinpath(directory,"carry.jls"))
        batch = deserialize(joinpath(directory,"batch.jls"))
        records = deserialize(joinpath(directory,"coordinates.jls"))
        @test_throws UndefVarError deserialize(joinpath(directory,"legacy-rhs.jls"))
        f = generated_target()
        before = F.bytes(partial)
        next = F.tune(f.target; resume=partial, rng=Xoshiro(999),
            callback=(p,b) -> b == :window)
        @test F.EXT._same(work_snapshot(next), deserialize(joinpath(directory,"next-window.jls")))
        @test F.bytes(partial) == before
        @test F.EXT._same(F.signature(F.tune(f.target; resume=next)), expected)
        @test F.EXT._same(F.signature(F.tune(f.target; checkpoint_dir=directory, resume=true)), expected)
        @test_throws ArgumentError F.tune(generated_target(binding_id="changed").target; resume=partial)
        @test_throws ArgumentError F.tune(generated_target(shift=0.5).target; resume=partial)
        @test_throws ArgumentError F.tune(f.target; resume=partial, pilot_steps=5)
        old = merge(partial.checkpoint, (; versions=merge(partial.checkpoint.versions, (; ppl=v"0.1.97"))))
        @test_throws ArgumentError F.tune(f.target; resume=SMCTuningPartial(old, partial.progress))
        @test F.EXT._same(records, PPL._smc_coordinate_records(f.built.layout))
        @test F.EXT._same(carry.frame, F.EXT._frame(f.target))
        restored = resume_smc(f.target, batch)
        @test restored.positions == batch.positions
        @test rand(copy(restored.rng), UInt64, 16) == rand(copy(batch.rng), UInt64, 16)
    end
else
    error("expected write or read")
end
