using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, Serialization, Test

source = quote
    mu ~ Normal(0,1)
    y .~ Normal.(mu,1)
end
data = (; y=[0.9,1.2,0.8])
plan = bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
built = build_kernel(plan)
target = prepare_smc(built,plan,[0.0]; backend=AutoEnzyme(mode=Enzyme.Reverse),
    model_id="tuning-process-v1",binding_id="three")

function signature(state)
    (; state.positions, state.weights, state.log_normalizer, state.temperature,
       state.history, state.ancestors, state.lineage,
       rng_draws=rand(copy(state.rng),UInt64,16),
       pilot_draws=rand(copy(state.options.tuning.pilot.final_rng),UInt64,16))
end

mode, path = ARGS
if mode == "write"
    tuning = tune_smc(target; rng=Xoshiro(927), init=[0.1],
        temperatures=(0.0,0.5,1.0),pilot_steps=16)
    state = smc(target; rng=Xoshiro(71), particles=64,
        options=SMCOptions(tuning=tuning), stages=1)
    serialize(path,(; checkpoint=smc_checkpoint(state), expected=signature(smc(state))))
elseif mode == "read"
    payload = deserialize(path)
    state = resume_smc(target,payload.checkpoint)
    @test signature(smc(state)) == payload.expected
else
    error("expected write or read")
end
