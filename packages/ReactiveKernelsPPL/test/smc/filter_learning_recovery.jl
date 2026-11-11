include("filter_learning_fixture.jl")
using .FilterLearningFixture
using ReactiveKernelsPPL, Distributions, Random, Statistics, Test, Serialization
const F = FilterLearningFixture
const E = F.EXT

function caught(f)
    try
        f()
        nothing
    catch e
        e
    end
end

# Build consistent checkpoint-era clouds, including each member's likelihood
# estimate and conditional state. No invalid parameter/state pairing is used.
function cloud(state,positions;mass=state.parameter_mass,kernel=state.kernel)
    members = [first(E._window_filter(state.target,state.boundary,state.records,
        positions[:,j],Xoshiro(100+j),first(state.members).particles)) for j in axes(positions,2)]
    fields = (; (k=>getfield(state,k) for k in fieldnames(typeof(state)))...)
    fields = merge(fields,(;positions,members,parameter_mass=mass,
        log_parameter_mass=log.(mass),kernel))
    E.SMCFilterState(values(fields)...)
end

@testset "Replenishment retains learned support and likelihood estimates" begin
    target=F.direct_target()
    s=initialize_filter_smc(target;rng=Xoshiro(42),members=64,particles=1,
        window_length=24,ess_fraction=1.0,
        options=SMCOptions(stepsize=0.35,moves=1,max_tree_depth=1))
    # A broad window prior, but an already learned narrow population. Moves are
    # local walks scaled by that population, never fresh window-prior draws.
    positions=reshape(collect(range(0.8,1.2;length=64)),1,:)
    s=cloud(s,positions)
    before=deepcopy(filter_checkpoint(s))
    next=filter_step(s,(;y=1.0))
    stage=last(next.history)
    @test stage.replenished
    w=[exp(logpdf(Normal(u,0.1),1.0)) for u in vec(positions)]
    w./=sum(w)
    m=sum(w.*vec(positions))
    @test only(stage.proposal_scale)≈2.38*sqrt(sum(w.*(vec(positions).-m).^2)) rtol=1e-8
    # Unmoved members keep their selected learned value exactly.
    @test count(u -> u in positions,next.positions)==64-stage.moved_members
    @test stage.distinct_members>1
    @test 0<stage.moved_members==stage.accepted<stage.proposals
    @test stage.reference_evaluations==0
    @test stage.likelihood_evaluations==64*(1+1)
    @test E._same(filter_checkpoint(s),before)
    @test all(j -> only(only(next.members[j].carries).h)==next.positions[1,j],1:64)
end

@testset "Narrow Gaussian oracle across repeated incremental resampling" begin
    target=F.direct_target(scale=0.08)
    estimates=Float64[]
    variances=Float64[]
    for seed in 1:3
        s=initialize_filter_smc(target;rng=Xoshiro(seed),members=768,particles=1,
            window_length=12,ess_fraction=0.9,
            options=SMCOptions(stepsize=0.02,moves=2,max_tree_depth=2))
        s=filter_stream(s,F.records(fill(0.5,12)))
        push!(estimates,F.moments(s).mean)
        push!(variances,F.moments(s).variance)
        @test all(stage -> stage.distinct_members>1,s.history)
        @test any(stage -> stage.moved_members>0,s.history)
    end
    variance=inv(1+12/0.08^2)
    meanvalue=variance*12*0.5/0.08^2
    @info "Narrow direct Gaussian" meanvalue variance estimates variances
    @test abs(mean(estimates)-meanvalue)<0.008
    @test abs(mean(variances)-variance)<0.00015
end

@testset "Transactional collapse, explicit recovery and retry" begin
    # Observation scale 0.5: the far pair below still puts relative weight
    # e^-128 on one member, so the population walk's scale underflows and the
    # pair collapses, while recovered prior-scale pairs keep usable spread.
    target=F.direct_target(scale=0.5)
    s=initialize_filter_smc(target;rng=Xoshiro(52),members=2,particles=1,
        window_length=2,ess_fraction=0.9,history_limit=3,
        options=SMCOptions(stepsize=1e-200,moves=1,max_tree_depth=1))
    s=cloud(s,reshape([1.0,9.0],1,:))
    before=deepcopy(filter_checkpoint(s))
    err=caught(() -> filter_step(s,(;y=1.0)))
    @test err isa SMCDegeneracyError
    @test err.state===s
    @test err.phase==:filter_step
    @test err.diagnostics.time==s.time+1
    @test err.diagnostics.distinct_members==1
    @test err.diagnostics.moved_members==0
    @test err.diagnostics.parameter_ess_after≈2
    @test E._same(filter_checkpoint(s),before)
    @test length(sprint(showerror,err,backtrace()))<1000

    # Collapse is tested after mutation: a single selected ancestor whose
    # population still had weighted spread is moved apart by the walk.
    spread=cloud(s,reshape([1.0,2.66],1,:);kernel=merge(s.kernel,(;moves=16)))
    next=filter_step(spread,(;y=1.0))
    @test last(next.history).unique_parents==1
    @test last(next.history).distinct_members==2
    @test only(last(next.history).proposal_scale)>0

    tuning=tune_filter_smc(s;rng=Xoshiro(901),init=[0.0],temperatures=(0.0,1.0),
        pilot_steps=24,options=SMCOptions(moves=8,max_tree_depth=2))
    input=cloud(s,ones(1,2))
    saved=deepcopy(filter_checkpoint(input))
    @test caught(() -> filter_step(input,(;y=1.0))).phase==:input
    bad=deepcopy(tuning)
    for i in eachindex(bad.kernels)
        bad.kernels[i]=merge(bad.kernels[i],(;stepsize=1e-200))
    end
    failed=caught(() -> rejuvenate_smc(input;tuning=bad,moves=3))
    @test failed isa SMCDegeneracyError
    @test failed.state===input
    @test failed.phase==:rejuvenation
    @test failed.diagnostics.proposals==6
    @test E._same(filter_checkpoint(input),saved)
    recovered=rejuvenate_smc(input;tuning,moves=8)
    @test recovered.time==input.time
    @test recovered.window==input.window
    @test recovered.records==input.records
    @test recovered.parameter_mass==input.parameter_mass
    @test recovered.log_normalizer==input.log_normalizer
    @test recovered.predictive_logscore==input.predictive_logscore
    @test recovered.kernel==last(tuning.kernels)
    @test last(recovered.history).kind==:rejuvenate
    @test last(recovered.history).predictive_logscore==0
    @test E._same(filter_checkpoint(input),saved)
    @test_throws ArgumentError rejuvenate_smc(input;tuning,moves=0)

    resumed=resume_filter(target,filter_checkpoint(recovered))
    @test F.equal_signatures(F.signature(filter_step(recovered,(;y=1.0))),
        F.signature(filter_step(resumed,(;y=1.0))))

    # Recover the preserved pre-observation state, then retry exactly once.
    retry=filter_step(rejuvenate_smc(err.state;tuning,moves=8),(;y=1.0))
    @test retry.time==1
    @test length(retry.records)==1
    @test retry.log_normalizer==retry.predictive_logscore

    # At later boundaries the independent schedule must match the new reference.
    s=initialize_filter_smc(target;rng=Xoshiro(90),members=64,particles=1,
        window_length=1,ess_fraction=0.5,options=SMCOptions(stepsize=0.03,moves=2))
    s=filter_stream(s,F.records([0.5,0.6]))
    @test s.window==2
    @test_throws ArgumentError rejuvenate_smc(s;tuning,moves=1)
    current=tune_filter_smc(s;rng=Xoshiro(92),init=[0.0],pilot_steps=16,
        temperatures=(0.0,1.0),options=SMCOptions(max_tree_depth=2))
    next=rejuvenate_smc(s;tuning=current,moves=8)
    @test next.time==s.time
    @test next.log_normalizer==s.log_normalizer
    @test next.kernel==last(current.kernels)
end

@testset "Copies count once in the replenishment trigger" begin
    target=F.direct_target()
    s=initialize_filter_smc(target;rng=Xoshiro(61),members=64,particles=1,
        window_length=24,ess_fraction=0.5,
        options=SMCOptions(stepsize=0.35,moves=2,max_tree_depth=1))
    # Equal weights on 32 copies of each of two values: member ESS reads 64,
    # but only two parameter values carry the weight (a collapsed cloud's signature).
    positions=reshape(repeat([0.9,1.1];inner=32),1,:)
    s=cloud(s,positions)
    next=filter_step(s,(;y=1.0))
    stage=last(next.history)
    @test stage.parameter_ess_before≈64
    @test stage.distinct_ess_before≈2
    @test stage.replenished
    @test stage.distinct_members>2
    @test stage.distinct_ess_after>stage.distinct_ess_before
    # Without new weight differences a diverse population is left alone.
    calm=filter_step(cloud(s,reshape(collect(range(0.9,1.1;length=64)),1,:)),(;))
    @test !last(calm.history).replenished
    @test last(calm.history).distinct_ess_before≈64
end
