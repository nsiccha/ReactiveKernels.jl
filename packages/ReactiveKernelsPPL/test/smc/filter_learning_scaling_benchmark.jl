# Run separately from correctness tests: timing is evidence, not a CI threshold.
include("filter_learning_fixture.jl")
using .FilterLearningFixture
using ReactiveKernelsPPL, Random, Statistics
const F=FilterLearningFixture
const OPT=SMCOptions(moves=1,stepsize=0.35,max_tree_depth=2)

function measure(state,observation;repeats=21)
    for _ in 1:3
        filter_step(state,observation)
    end
    GC.gc()
    samples=[@timed(filter_step(state,observation)) for _ in 1:repeats]
    output=last(samples).value
    times=getproperty.(samples,:time)
    (;seconds=median(times),quartiles=Tuple(quantile(times,[0.25,0.75])),
        allocated_bytes=median(getproperty.(samples,:bytes)),
        retained_bytes=Base.summarysize(filter_checkpoint(output)),
        evaluations=output.history[end].likelihood_evaluations,
        reference_evaluations=output.history[end].reference_evaluations,
        replenished=output.history[end].replenished)
end

if isempty(ARGS) || only(ARGS)=="standard"
for (members,particles,groups) in ((16,128,1),(32,128,1),(16,256,1),(16,128,2),(16,128,4))
    target=F.target(group_count=groups)
    state=initialize_filter_smc(target;rng=Xoshiro(21),members,particles,
        window_length=4,history_limit=2,ess_fraction=1e-6,options=OPT)
    early=nothing
    for t in 1:240
        state=filter_step(state,(;))
        t==24 && (early=state)
    end
    # Both are the same phase of the bounded window, with identical storage.
    a,b=measure(early,(;)),measure(state,(;))
    @assert a.retained_bytes==b.retained_bytes
    @assert a.allocated_bytes==b.allocated_bytes
    @assert a.evaluations==b.evaluations==members*particles
    @assert !a.replenished && !b.replenished
    @info "Warm filter scaling" members particles groups elapsed=(24,240) early=a late=b
end

for members in (16,32),particles in (128,256)
    target=F.target()
    state=initialize_filter_smc(target;rng=Xoshiro(56),members,particles,
        window_length=3,history_limit=2,ess_fraction=1.0,options=OPT)
    state=filter_stream(state,F.records([0.9,1.1]))
    result=measure(state,(;y=0.8))
    @assert result.replenished
    @assert result.evaluations==members*particles*(1+2*3)
    @info "Warm replenishment scaling" members particles window=3 result
end
end

if isempty(ARGS) || only(ARGS)=="groups"
for groups in (64,256,1024,1700)
    members,particles=8,32
    target=F.target(group_count=groups)
    state=initialize_filter_smc(target;rng=Xoshiro(21),members,particles,
        window_length=4,history_limit=2,ess_fraction=1e-6,options=OPT)
    for _ in 1:4
        state=filter_step(state,(;))
    end
    prediction=measure(state,(;);repeats=9)
    observation=measure(state,(;y=fill(0.8,groups));repeats=9)
    @assert !prediction.replenished && !observation.replenished
    @assert prediction.evaluations==observation.evaluations==members*particles
    @info "Wide group scaling" members particles groups prediction observation
end
end
