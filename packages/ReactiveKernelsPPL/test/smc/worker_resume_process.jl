include("tuning_lifecycle_fixture.jl")
using .SMCTuningLifecycleFixture, ReactiveKernelsPPL, Random, Serialization, Test
const F = SMCTuningLifecycleFixture
tune(;kwargs...) = F.tune(; grid_start=:independent, kwargs...)
cloud(s) = (; s.positions,s.weights,s.temperature,s.log_normalizer,s.ancestors,
    s.lineage,rng=rand(copy(s.rng),UInt64,16))
mode, directory = ARGS
if mode == "write"
    serialize(joinpath(directory,"expected.jls"), F.signature(tune()))
    partial = tune(; checkpoint_dir=joinpath(directory,"grid"),
        callback=(p,b) -> p.phase === :grid && b === :window)
    @test partial isa SMCTuningPartial
    serialize(joinpath(directory,"partial.jls"),partial)
    state = initialize_smc(F.target(); rng=Xoshiro(891), particles=37,
        options=SMCOptions(nthreads=Threads.nthreads(), stepsize=0.15, moves=2))
    serialize(joinpath(directory,"production.jls"),smc_checkpoint(state))
    serialize(joinpath(directory,"cloud.jls"),cloud(smc(state)))
elseif mode == "read"
    expected = deserialize(joinpath(directory,"expected.jls"))
    partial = deserialize(joinpath(directory,"partial.jls"))
    before = F.bytes(partial)
    @test F.EXT._same(F.signature(tune(;resume=partial)),expected)
    @test F.bytes(partial) == before
    @test F.EXT._same(F.signature(tune(;checkpoint_dir=joinpath(directory,"grid"),resume=true)),expected)
    cp = deserialize(joinpath(directory,"production.jls"))
    before = F.bytes(cp)
    @test resume_smc(F.target(),cp).options.nthreads == cp.options.nthreads
    resumed = resume_smc(F.target(),cp; nthreads=Threads.nthreads())
    @test resumed.options.nthreads == Threads.nthreads()
    for name in fieldnames(SMCOptions)
        name === :nthreads && continue
        @test F.EXT._same(getfield(resumed.options,name),getfield(cp.options,name))
    end
    final = smc(resumed)
    @test F.EXT._same(cloud(final),deserialize(joinpath(directory,"cloud.jls")))
    @test all(h -> h.nthreads == Threads.nthreads(),final.history)
    @test F.bytes(cp) == before
    for invalid in (0,-1,1.5,:auto)
        @test_throws ArgumentError resume_smc(F.target(),cp; nthreads=invalid)
    end
else
    error("expected write or read")
end
