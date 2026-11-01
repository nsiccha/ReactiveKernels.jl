include("filter_learning_fixture.jl")
using .FilterLearningFixture
using Test, Random, ReactiveKernelsPPL, Statistics, Serialization
const F = FilterLearningFixture
const OPT = SMCOptions(stepsize=0.35,moves=1,max_tree_depth=3)
roundtrip(x) = (io=IOBuffer();serialize(io,x);seekstart(io);deserialize(io))

@testset "Parameter learning and Kalman oracle" begin
    target=F.target()
    oracle=F.oracle()
    estimates=Float64[]
    variances=Float64[]
    replenishments=Int[]
    for seed in 1:3
        rng=Xoshiro(seed)
        before=copy(rng)
        state=initialize_filter_smc(target;rng,members=256,particles=96,
            window_length=length(F.Y),ess_fraction=0.8,options=OPT,history_limit=16)
        @test rand(rng,UInt64,8)==rand(before,UInt64,8)
        initial=roundtrip(filter_checkpoint(state))
        Random.seed!(591)
        default_before=copy(Random.default_rng())
        state=filter_stream(state,F.records())
        @test rand(Random.default_rng(),UInt64,8)==rand(default_before,UInt64,8)
        @test initial.time==0
        @test state.time==length(F.Y)
        @test length(state.records)==length(F.Y)
        @test all(m -> m.time==state.time && m.log_normalizer<0,state.members)
        @test all(j -> only(state.members[j].ensemble).theta==state.positions[1,j],1:256)
        push!(estimates,F.moments(state).mean)
        push!(variances,F.moments(state).variance)
        push!(replenishments,count(s -> s.replenished,state.history))
        @test sum(s.accepted for s in state.history)>0
        @test sum(s.proposals-s.accepted for s in state.history)>0
        @test all(s -> s.likelihood_evaluations>=256*96,state.history)
        @test all(s -> !s.replenished || s.parameter_ess_after≈256,state.history)
    end
    @info "Kalman parameter oracle" oracle estimates=Tuple(estimates) variances=Tuple(variances) replenishments=Tuple(replenishments)
    @test all(>(0),replenishments)
    @test abs(mean(estimates)-oracle.mean)<0.10
    @test abs(mean(variances)-oracle.variance)<0.045
end

@testset "Zero conditional likelihoods and rejection" begin
    target=F.support_target()
    s=initialize_filter_smc(target;rng=Xoshiro(47),members=128,particles=2,
        window_length=4,ess_fraction=0.01,options=OPT)
    before=roundtrip(filter_checkpoint(s))
    next=filter_step(s,(;y=0.2))
    @test next.history[end].replenished
    @test all(u -> -0.8<=u<=0.2,next.positions)
    @test next.history[end].accepted<next.history[end].proposals
    # Uniform(theta, theta+1) has unit width in real arithmetic. Floating
    # subtraction of its endpoints and the PF normalizer can round by ulps.
    @info "Unit-width conditional likelihood roundoff" extrema=extrema(m.log_normalizer for m in next.members)
    @test all(m -> isapprox(m.log_normalizer,0.0;atol=1e-14),next.members)
    @test all(j -> all(c -> only(c.h)==next.positions[1,j],next.members[j].carries),1:128)
    @test_throws ArgumentError filter_step(s,(;y=1e6))
    @test F.EXT._same(filter_checkpoint(s),before)
end

@testset "Bounded windows, identity, tuning and replay" begin
    target=F.target()
    pilot_rng=Xoshiro(812)
    before=copy(pilot_rng)
    schedule=tune_filter_smc(target;rng=pilot_rng,init=[0.0],temperatures=(0.0,1.0),
        pilot_steps=12,options=OPT)
    @test rand(pilot_rng,UInt64,8)==rand(before,UInt64,8)
    state=initialize_filter_smc(target;rng=Xoshiro(23),members=24,particles=20,
        window_length=3,ess_fraction=0.95,history_limit=3,options=SMCOptions(tuning=schedule))
    state=filter_stream(state,F.records(F.Y[1:3]))
    checkpoint=roundtrip(filter_checkpoint(state))
    saved=deepcopy(checkpoint)
    next=filter_step(state,(;y=F.Y[4]))
    replay=filter_step(resume_filter(F.target(),checkpoint),(;y=F.Y[4]))
    @test F.equal_signatures(F.signature(next),F.signature(replay))
    @test F.EXT._same(checkpoint,saved)
    @test next.window==2
    @test next.boundary.time==3
    @test length(next.boundary.carries)==20
    @test length(next.records)==1
    @test next.kernel==state.kernel
    @test next.history[end].boundary_reset
    @test all(m -> m.time==4,next.members)
    @test_throws ArgumentError resume_filter(F.target(parameter_id="changed"),checkpoint)
    @test_throws ArgumentError resume_filter(F.target(carried=F.calibration(location=0.2)),checkpoint)
    @test_throws ArgumentError F.target(model_id="different")
    @test_throws ArgumentError initialize_filter_smc(F.target(carried=F.calibration(scale=2.0));
        rng=Xoshiro(1),options=SMCOptions(tuning=schedule))
    invalid=merge(checkpoint,(;time=checkpoint.time+1))
    @test_throws ArgumentError resume_filter(target,invalid)
    @test_throws ArgumentError filter_step(state,(;unknown=1.0))
    @test F.EXT._same(filter_checkpoint(state),saved)
    observations=(;y=[1.0,1.2])
    grouped=F.target(group_count=2)
    s=initialize_filter_smc(grouped;rng=Xoshiro(51),members=12,particles=8,
        window_length=2,options=OPT)
    old=deepcopy(observations)
    s=filter_step(s,observations;groups=(;g=[1]))
    s=filter_step(s,(;);groups=(;g=[2]))
    s=filter_step(s,(;y=[missing,0.9]);groups=(;g=[2]))
    @test observations==old
    @test s.boundary.group_times.g==[1,1]
    @test all(m -> m.group_times.g==[1,2],s.members)
    @test length(s.records)==1
end

@testset "Calibrated state-only boundary" begin
    target=F.target()
    trained=initialize_filter(target.filter;rng=Xoshiro(9),particles=40,
        ensemble=[(;theta=0.0),(;theta=1.0)],weights=[0.2,0.8])
    trained=filter_stream(trained,F.records(F.Y[1:4]);condition_state_only=true)
    cp=roundtrip(filter_checkpoint(trained))
    rng=Xoshiro(11);before=copy(rng)
    s=initialize_filter_smc(target;rng,members=12,particles=16,boundary=trained,options=OPT)
    @test rand(rng,UInt64,8)==rand(before,UInt64,8)
    @test s.time==4
    @test s.log_normalizer==0
    @test isempty(s.records)
    @test length(s.boundary.carries)==16
    @test filter_checkpoint(trained)==cp
    next=filter_step(s,(;y=F.Y[5]))
    @test next.time==5
    @test F.equal_signatures(F.signature(next),F.signature(filter_step(resume_filter(target,filter_checkpoint(s)),(;y=F.Y[5]))))
    @test_throws ArgumentError initialize_filter_smc(F.target(group_count=2);
        rng=Xoshiro(1),boundary=trained)
end
