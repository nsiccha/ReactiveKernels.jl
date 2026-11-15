# Run by runtests.jl, once per cloud: `pool_writer.jl <seed> <file>` finishes
# one cloud of an @rkppl target and one of a prepared density, and saves their
# checkpoints for `pool_process.jl`.
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, Serialization
include("prepared_fixture.jl")
using .PreparedSMCFixture
const F = PreparedSMCFixture

data = (; y=[0.9, 1.6, 0.2, 1.3, 0.6, 1.1])
source = quote
    mu ~ Normal(0, 1)
    y .~ Normal.(mu, 1)
end
plan = bind_data(lower_rkppl(source, data; conditioned=(:y,)), data)
built = build_kernel(plan)
model = prepare_smc(built, plan, zeros(built.layout.total);
    backend=AutoEnzyme(mode=Enzyme.Reverse), model_id="pool-gaussian", binding_id="six")
seed = parse(Int, ARGS[1])
clouds = (; built=smc(model; rng=Xoshiro(seed), particles=200),
    prepared=smc(F.target(); rng=Xoshiro(seed), particles=200, options=F.options()))
serialize(ARGS[2], map(smc_checkpoint, clouds))
