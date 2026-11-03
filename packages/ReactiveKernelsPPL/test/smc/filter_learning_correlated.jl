include("filter_learning_fixture.jl")
using .FilterLearningFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, Test, Serialization
using Distributions, LinearAlgebra, Statistics, WarmupHMC
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

@testset "Correlated calibration and correlated boundary transition" begin
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
    # The boundary keeps the members' correlation. In two dimensions one
    # direction plus the remainder reproduces their weighted covariance.
    boundary=next.calibration.reference
    @test boundary.scale.kind===:diagonal_lowrank
    @test last(next.history).boundary_reset && last(next.history).boundary_directions==1
    sd=sqrt.(boundary.scale.diagonal)
    carried=sd.*(I+boundary.scale.vectors*Diagonal(boundary.scale.values)*boundary.scale.vectors').*sd'
    @test boundary.location≈estimated_mean rtol=1e-12
    @test carried≈estimated_covariance rtol=1e-6
    @test next.kernel==s.kernel
    @test next.window==2
    io=IOBuffer();serialize(io,filter_checkpoint(next));seekstart(io)
    replay=resume_filter(correlated_target(),deserialize(io))
    @test F.equal_signatures(F.signature(filter_step(next,(;y=0.8))),
        F.signature(filter_step(replay,(;y=0.8))))
end

# A hierarchical ridge, y[g] ~ Normal(mu + z[g], s): the data pin down only
# b[g] = mu + z[g]. The direction (mu + c, z .- c) is informed by the prior
# alone, so the exact posterior keeps its width however much data arrives.
# A diagonal window boundary (max_reflections=0) treats the marginal widths of
# mu and z as independent information, and every window contracts that
# direction again (snag learning-filter-8c01d6de: about 1 % of the exact SD
# after twelve windows on the ODLPlaus AP3 replay).
function ridge_target(G,s,location,covariance)
    data=(;y=Float64[],G)
    plan=bind_data(lower_rkppl(quote
        mu ~ Normal()
        z[1:G] .~ Normal.(0,1)
        y .~ Normal.(mu,1)
    end,data;conditioned=(:y,)),data)
    static=prepare_smc(build_kernel(plan),plan,zeros(G+1);backend=F.BACKEND,
        model_id="ridge-static-filter",binding_id="empty")
    # Exact calibration: every correlation direction of the oracle covariance.
    sd=sqrt.(diag(covariance))
    eig=eigen(Symmetric(covariance./(sd*sd')))
    scale=(;kind=:diagonal_lowrank,diagonal=sd.^2,vectors=eig.vectors,values=eig.values.-1)
    carried=carry_smc(static,SMCReference(location,scale);calibration_id="ridge")
    model=@rkppl begin
        mu ~ Normal()
        z[1:G] .~ Normal.(0,1)
        @plate for g in eachindex(y)
            T=length(y[g])
            @scan begin
                h[1]=mu+z[g]
                for t in 2:T
                    h[t]=mu+z[g]
                end
            end
            loc[g]=h
            y[g] .~ Normal.(loc[g],s)
        end
    end
    parameters(u)=(;mu=u[1],z=u[2:end])
    filter=prepare_filter(model;model_id="ridge-static-filter",parameters=parameters(zeros(G+1)),
        groups=(;g=1:G),data=(;G,s))
    prepare_filter_smc(filter,carried;parameters,parameter_id="ridge/v1",backend=F.BACKEND)
end

@testset "Correlated boundary keeps the width the data do not inform" begin
    G,s,calibration_records,window,windows=4,0.2,48,12,6
    rng=Xoshiro(2026)
    b=0.3 .+ randn(rng,G)
    A=[ones(G) Matrix(1.0I,G,G)]
    # Closed-form Gaussian oracle; the d=5 test oracle alone forms dense matrices.
    precision=Matrix(1.0I,G+1,G+1)+(calibration_records/s^2)*(A'*A)
    shift=A'*(calibration_records .* b .+ s*sqrt(calibration_records) .* randn(rng,G))./s^2
    covariance=inv(Symmetric(precision))
    target=ridge_target(G,s,covariance*shift,covariance)
    records=[(;y=b .+ s .* randn(rng,G)) for _ in 1:window*windows]
    precision.+=(length(records)/s^2)*(A'*A)
    exact_sd=sqrt.(diag(inv(Symmetric(precision))))
    spread(state)=(m=state.positions*state.parameter_mass;
        sqrt.(vec(sum(state.parameter_mass'.*(state.positions.-m).^2;dims=2))))
    run(max_reflections)=foldl((state,r)->filter_step(state,r),records;
        init=initialize_filter_smc(target;rng=Xoshiro(7),members=128,particles=1,
            window_length=window,ess_fraction=0.5,max_reflections,history_limit=window*windows,
            options=SMCOptions(stepsize=0.35,moves=2,max_tree_depth=2)))
    correlated,diagonal=run(16),run(0)
    ratio_correlated,ratio_diagonal=spread(correlated)./exact_sd,spread(diagonal)./exact_sd
    @info "Ridge learning-filter spread / exact posterior SD after $windows windows" correlated=Tuple(round.(ratio_correlated;digits=3)) diagonal=Tuple(round.(ratio_diagonal;digits=3))
    # Thresholds sit outside the 0.5–99.5 % range over 200 seeds of an
    # independent simulation of this algorithm (smallest coordinate ratio,
    # correlated 0.58–1.32; largest, diagonal 0.056–0.106).
    @test all(>(0.4),ratio_correlated)
    @test all(<(0.15),ratio_diagonal)
    @test all(h -> !h.boundary_reset || h.boundary_directions==0,diagonal.history)
    @test any(h -> h.boundary_reset && h.boundary_directions>=1,correlated.history)
    @test any(h -> h.replenished && h.proposal_directions>=1,correlated.history)
    @test all(h -> h.proposal_directions==0,diagonal.history)
    checkpoint=filter_checkpoint(correlated)
    @test checkpoint.schema_version==3 && checkpoint.max_reflections==16
    @test_throws ArgumentError initialize_filter_smc(target;rng=Xoshiro(1),max_reflections=-1)
end

@testset "basis=:correlation carries correlations across coordinate scales" begin
    # A cloud whose narrow coordinates (scale 1e-4) are as strongly correlated
    # as its wide ones (scale 1): the native-coordinate fit spends its one
    # reflection on the wide pair; the standardized fit keeps both pairs.
    rng=Xoshiro(91)
    n=4000
    common=randn(rng,n); other=randn(rng,n)
    scales=[1.0,1.0,1e-4,1e-4]
    positions=permutedims(hcat(common.+0.1randn(rng,n),common.+0.1randn(rng,n),
        other.+0.1randn(rng,n),other.+0.1randn(rng,n))).*scales
    w=fill(1/n,n)
    m=positions*w
    X=positions.-m
    # The d=4 oracle alone forms the dense weighted covariance.
    C=(X.*w')*X'
    sd=sqrt.(diag(C))
    scale=F.EXT._correlation_scale(positions,w,m,sd,2)
    @test scale.kind===:diagonal_lowrank && size(scale.vectors)==(4,2)
    dense(s)=sqrt.(s.diagonal).*(I+s.vectors*Diagonal(s.values)*s.vectors').*sqrt.(s.diagonal)'
    R(A)=A./(sqrt.(diag(A))*sqrt.(diag(A))')
    @test R(dense(scale))[3,4]≈R(C)[3,4] rtol=0.05
    @test R(dense(scale))[1,2]≈R(C)[1,2] rtol=0.05
    raw=WarmupHMC.fit_gaussian_moments(positions;weights=w,max_reflections=1).scale
    L=WarmupHMC.scale_from_data(raw)
    Craw=reduce(hcat,[L*Matrix(1.0I,4,4)[:,j] for j in 1:4]); Craw=Craw*Craw'
    @test abs(R(Craw)[3,4])<0.1<R(C)[3,4]
    # Rescaling coordinates rescales the summary and keeps its directions.
    D=[3.0,0.5,2e3,7e2]
    rescaled=F.EXT._correlation_scale(D.*positions,w,D.*m,D.*sd,2)
    @test dense(rescaled)≈(D.*dense(scale).*D') rtol=1e-6
    # carry_smc(completed; basis=:correlation) on a completed SMC fit: in two
    # dimensions one direction plus the remainder reproduces the covariance.
    data=(;y=[1.1,0.9,1.0])
    plan=bind_data(lower_rkppl(quote
        a ~ Normal()
        b ~ Normal()
        y .~ Normal.(a+b,0.3)
    end,data;conditioned=(:y,)),data)
    target=prepare_smc(build_kernel(plan),plan,[0.0,0.0];backend=F.BACKEND,
        model_id="correlation-carry",binding_id="y3")
    fit=smc(target;rng=Xoshiro(17),particles=512,
        options=SMCOptions(stepsize=0.3,moves=4,max_tree_depth=3))
    carried=carry_smc(fit;calibration_id="correlation",basis=:correlation)
    @test carried.reference.scale.kind===:diagonal_lowrank
    @test carried.fit_diagnostics.directions==1
    P,wt=fit.positions,fit.weights
    Xc=P.-P*wt
    @test dense(carried.reference.scale)≈(Xc.*wt')*Xc' rtol=1e-6
    @test carried.reference.location≈P*wt rtol=1e-12
    @test_throws ArgumentError carry_smc(fit;calibration_id="x",basis=:unknown)
end
