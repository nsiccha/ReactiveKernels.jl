module SMCSelectiveFixture
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using LinearAlgebra, Statistics, Test, Serialization
const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)
const SOURCE = quote
    z[1:nb] .~ Normal.(0, 1)
    y .~ Normal.(z[block], 1)
end
normal(x,m,s) = -(log(2pi) + 2log(s) + abs2((x-m)/s))/2
function build(nb, block, y; source=SOURCE)
    data = (; nb, block=Int.(block), y=Float64.(y))
    plan = bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
    (; built=build_kernel(plan), plan)
end
function calibration(f, reference; model_id="selective-v1")
    target = prepare_smc(f.built,f.plan,zeros(f.built.layout.total);
        backend=BACKEND,model_id,binding_id="calibration")
    carry_smc(target,reference;calibration_id="external-summary")
end
function window(f, previous; model_id="selective-v1",binding_id="window",coordinates=:auto)
    prepare_smc_window(f.built,f.plan,zeros(f.built.layout.total);
        backend=BACKEND,model_id,binding_id,previous,coordinates)
end
function signature(state)
    (; state.positions,state.weights,state.log_normalizer,state.temperature,
        state.history,state.ancestors,state.lineage,
        rng_draws=rand(copy(state.rng),UInt64,16))
end
end
