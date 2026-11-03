include("filter_learning_fixture.jl")
using .FilterLearningFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, Test, Serialization
const F=FilterLearningFixture

function correlated_target()
    data=(;y=Float64[])
    plan=bind_data(lower_rkppl(quote
        a ~ Normal()
        b ~ Normal()
        y .~ Normal.(a+b,0.5)
    end,data;conditioned=(:y,)),data)
    static=prepare_smc(build_kernel(plan),plan,[0.0,0.0];backend=F.BACKEND,
        model_id="correlated-static-filter",binding_id="empty")
    portable=(;diagonal=[sqrt(2.0),sqrt(0.02)],
        indices=[[1,2]],reflections=[[cos(pi/8),sin(pi/8)]])
    carried=carry_smc(static,SMCReference([0.2,-0.1],portable);calibration_id="correlated")
    model=@rkppl begin
        a ~ Normal()
        b ~ Normal()
        @scan begin
            h[1]=a+b
            for t in 2:T
                h[t]=a+b
            end
        end
        y .~ Normal.(h,0.5)
    end
    filter=prepare_filter(model;model_id="correlated-static-filter",parameters=(;a=0.0,b=0.0))
    prepare_filter_smc(filter,carried;parameters=u -> (;a=u[1],b=u[2]),
        parameter_id="a-b/v1",backend=F.BACKEND)
end

@testset "Partial coordinate collapse is allowed until Gaussian rollover" begin
    target=correlated_target()
    state=initialize_filter_smc(target;rng=Xoshiro(73),members=32,particles=1,
        window_length=1,ess_fraction=0.01,
        options=SMCOptions(stepsize=1e-200,moves=1,max_tree_depth=1))
    positions=copy(state.positions)
    positions[2,:].=0.3
    members=[first(F.EXT._window_filter(target,state.boundary,state.records,
        positions[:,j],Xoshiro(400+j),1)) for j in 1:32]
    fields=(; (k=>getfield(state,k) for k in fieldnames(typeof(state)))...)
    state=F.EXT.SMCFilterState(values(merge(fields,(;positions,members)))...)
    state=filter_step(state,(;))
    @test last(state.history).distinct_members==32
    @test last(state.history).coordinate_sd[2]==0.0
    saved=deepcopy(filter_checkpoint(state))
    err=try
        filter_step(state,(;y=0.4))
        nothing
    catch e
        e
    end
    @test err isa SMCDegeneracyError
    @test err.phase==:rollover
    @test err.state===state
    @test err.diagnostics.invalid_coordinates==[2]
    @test F.EXT._same(filter_checkpoint(state),saved)
    schedule=tune_filter_smc(state;rng=Xoshiro(801),init=[0.0,0.0],pilot_steps=24,
        temperatures=(0.0,1.0),options=SMCOptions(max_tree_depth=2))
    bad=deepcopy(schedule)
    for i in eachindex(bad.kernels)
        bad.kernels[i]=merge(bad.kernels[i],(;stepsize=1e-200))
    end
    @test_throws SMCDegeneracyError rejuvenate_smc(state;tuning=bad,moves=2)
    recovered=rejuvenate_smc(state;tuning=schedule,moves=8)
    @test all(>(0),last(recovered.history).coordinate_sd)
    @test filter_step(recovered,(;y=0.4)).time==state.time+1
    @test F.EXT._same(filter_checkpoint(state),saved)
end

@testset "Correlated calibration and diagonal boundary transition" begin
    target=correlated_target()
    s=initialize_filter_smc(target;rng=Xoshiro(29),members=512,particles=1,
        window_length=1,ess_fraction=1.0,
        options=SMCOptions(stepsize=0.1,moves=1,max_tree_depth=3))
    @test s.calibration.reference.scale isa NamedTuple
    s=filter_step(s,(;y=1.0))
    expected_mean=[0.2,-0.1].+(2/4.25)*0.9
    expected_covariance=[1.01 0.99;0.99 1.01].-4/4.25
    estimated_mean=s.positions*s.parameter_mass
    centered=s.positions.-estimated_mean
    # Only the two-dimensional test oracle constructs a dense covariance.
    estimated_covariance=(centered.*s.parameter_mass')*centered'
    @info "Correlated filter oracle" expected_mean=Tuple(expected_mean) estimated_mean=Tuple(estimated_mean) expected_variance=expected_covariance[1,1] estimated_variance=estimated_covariance[1,1] expected_covariance=expected_covariance[1,2] estimated_covariance=estimated_covariance[1,2]
    @test maximum(abs.(estimated_mean.-expected_mean))<0.10
    @test maximum(abs.(estimated_covariance.-expected_covariance))<0.035
    @test all(j -> only(only(s.members[j].carries).h)==sum(s.positions[:,j]),1:512)
    next=filter_step(s,(;y=0.9))
    @test next.calibration.reference.scale isa Vector
    @test next.kernel==s.kernel
    @test next.window==2
    io=IOBuffer();serialize(io,filter_checkpoint(next));seekstart(io)
    replay=resume_filter(correlated_target(),deserialize(io))
    @test F.equal_signatures(F.signature(filter_step(next,(;y=0.8))),
        F.signature(filter_step(replay,(;y=0.8))))
end
