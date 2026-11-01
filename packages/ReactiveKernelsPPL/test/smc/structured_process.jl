using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, Serialization, Test

source = quote
    a ~ Normal(0,1)
    b ~ Normal(a,0.2)
    y .~ Normal.(b,1)
end
data = (; y=[0.9,1.2,0.8])
plan = bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
built = build_kernel(plan)
target = prepare_smc(built,plan,zeros(2); backend=AutoEnzyme(mode=Enzyme.Reverse),
    model_id="structured-process-v1",binding_id="three")

function signature(state)
    reference = smc_reference(state)
    (; state.positions, state.weights, state.log_normalizer, state.temperature,
       state.history, state.ancestors, state.lineage,
       rng_draws=rand(copy(state.rng),UInt64,16),
       reference=(reference.location,reference.scale))
end

mode, path = ARGS
if mode == "write"
    scale=(; diagonal=[sqrt(2.0),sqrt(0.02)],
        indices=[[1,2]], reflections=[[cos(pi/8),sin(pi/8)]])
    state = initialize_smc(target; rng=Xoshiro(71), particles=64,
        reference=SMCReference([0.1,-0.1],scale),
        options=SMCOptions(metric=scale,stepsize=0.3))
    tuning = tune_smc(target; rng=Xoshiro(910), init=zeros(2),
        pilot=:adaptive, reference=:pilot, temperatures=(0.0, 1.0),
        pilot_steps=32, pilot_evaluations=100, recording_target=32,
        max_reflections=2, options=SMCOptions(moves=1, max_tree_depth=4))
    tuned = initialize_smc(target; rng=Xoshiro(911), particles=64,
        options=SMCOptions(tuning=tuning))
    serialize(path,(; checkpoint=smc_checkpoint(state), expected=signature(smc(state)),
        tuned_checkpoint=smc_checkpoint(tuned), tuned_expected=signature(smc(tuned)),
        reference_work=tuning.pilot.reference_work))
elseif mode == "read"
    payload = deserialize(path)
    state = resume_smc(target,payload.checkpoint)
    @test signature(smc(state)) == payload.expected
    tuned = resume_smc(target, payload.tuned_checkpoint)
    @test tuned.options.tuning.pilot.reference_work == payload.reference_work
    @test signature(smc(tuned)) == payload.tuned_expected
else
    error("expected write or read")
end
