module SMCWindowFixture
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using LinearAlgebra, Statistics, Test, Serialization
const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)
const SOURCE = quote
    a ~ Normal(100, 0.1)
    b ~ Normal(a, 0.2)
    y .~ Normal.(a + b, 1)
end
const MODEL_ID = "carried-window-v1"
const SCALE = (; diagonal=[sqrt(2.0),sqrt(0.02)],
    indices=[[1,2]], reflections=[[cos(pi/8),sin(pi/8)]])
const LOCATION = [0.2,-0.1]
const COVARIANCE = [1.01 0.99; 0.99 1.01]
function build(y)
    data = (; y=Float64.(y))
    plan = bind_data(lower_rkppl(SOURCE,data;conditioned=(:y,)),data)
    (; built=build_kernel(plan),plan)
end
function calibration(; n=0, calibration_id="calibration-1")
    f = build(zeros(n))
    target = prepare_smc(f.built,f.plan,zeros(2); backend=BACKEND,
        model_id=MODEL_ID,binding_id="old-$n")
    carry_smc(target,SMCReference(LOCATION,SCALE);calibration_id)
end
function window(y=[0.9,1.2,0.8]; previous=calibration(), model_id=MODEL_ID,
        binding_id="new-window")
    f = build(y)
    prepare_smc_window(f.built,f.plan,zeros(2); previous,backend=BACKEND,
        model_id,binding_id)
end
function signature(state)
    (; state.positions,state.weights,state.log_normalizer,state.temperature,
        state.history,state.ancestors,state.lineage,
        rng_draws=rand(copy(state.rng),UInt64,16))
end
end
