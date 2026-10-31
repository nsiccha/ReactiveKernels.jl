# Run by runtests.jl in two separate Julia processes. Neither endpoint query
# nor any compiled AD cache crosses the process boundary.
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random, Serialization

module CheckpointGeometry
using ReactiveKernelsPPL
struct ExternalExp end
ReactiveKernelsPPL.sampling_logdensity(::ExternalExp,x) = x > 0 ? -x : -Inf
ReactiveKernelsPPL.sampling_geometry(::Type{ExternalExp},args,shape) =
    ParameterGeometry(shape,1; support=:positive,
        constrain=(u,shape)->exp(u[1]), unconstrain=(x,shape)->[log(x)],
        logjac=(u,shape)->u[1])
end

function target(n)
    data = (; y=[1,2,0,3,1,2][1:n])
    source = quote
        rate ~ ExternalExp()
        y .~ Poisson.(rate)
    end
    plan = bind_data(lower_rkppl(source,data; conditioned=(:y,),mod=CheckpointGeometry),data)
    built = build_kernel(plan)
    return prepare_smc(built,plan,[0.0]; backend=AutoEnzyme(mode=Enzyme.Reverse),
        model_id="fresh-process-model",binding_id="prefix-$n")
end
snapshot(s) = (; s.positions,s.weights,s.temperature,s.log_normalizer,
    s.ancestors,s.lineage,s.history,rng=rand(copy(s.rng),UInt64,8))
old, new = target(2), target(6)
if ARGS[1] == "write"
    initial = smc(initialize_smc(new; rng=Xoshiro(71),particles=300); stages=1)
    update = update_smc(smc(old; rng=Xoshiro(72),particles=300),new; stages=1)
    serialize(ARGS[2],(; initial=smc_checkpoint(initial),update=smc_checkpoint(update),
        initial_result=snapshot(smc(initial)),update_result=snapshot(smc(update))))
elseif ARGS[1] == "read"
    saved = deserialize(ARGS[2])
    @assert snapshot(smc(resume_smc(new,saved.initial))) == saved.initial_result
    @assert snapshot(smc(resume_smc(new,saved.update; previous=old))) == saved.update_result
    println("Fresh-process batch/update checkpoint identity passed")
else
    error("expected write or read")
end
