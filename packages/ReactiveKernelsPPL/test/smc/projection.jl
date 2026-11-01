include("projection_fixture.jl")
using .SMCProjectionFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, Test, Serialization
const P=SMCProjectionFixture
const F=P.F
const R=ReactiveKernelsPPL

@testset "Completed latent fit to static carried filter" begin
    state=P.completed()
    names=filter_parameter_names(P.model())
    @test names==(:theta,:sigma)
    carried=carry_smc(state;parameters=names,calibration_id="static-summary")
    @test length(carried.reference.location)==2
    # The implementation's weighted reduction and BLAS use different orders.
    @test carried.reference.location≈state.positions[1:2,:]*state.weights rtol=8eps(Float64)
    @test all(isapprox.(values(smc_parameters(carried,[0.2,log(0.5)])),(0.2,0.5)))
    @test carried.frame.model_id==P.BATCH_ID
    @test isempty(carried.frame.parameters.layout.bound_values)
    @test all(e -> e.kind!==:scan,carried.frame.parameters.layout.entries)
    @test_throws ArgumentError carry_smc(state;parameters=(:h,),calibration_id="history")
    @test_throws ArgumentError carry_smc(state;parameters=(:absent,),calibration_id="absent")
    @test_throws DimensionMismatch carry_smc(state;parameters=names,
        scale=ones(size(state.positions,1)),calibration_id="wrong-scale")
    @test_throws ArgumentError prepare_filter_smc(P.static_filter(),carried;backend=F.BACKEND)
    target=P.learning(carried)
    @test target.filter.signature.model_id==P.FILTER_ID
    @test target.signature.calibration_model_id==P.BATCH_ID
    @test all(isapprox.(values(target.parameters([0.2,log(0.5)])),(0.2,0.5)))
    trained=initialize_filter(target.filter;rng=Xoshiro(15),particles=16)
    trained=filter_stream(trained,[(;observation=(;y=0.8)) for _ in 1:3];condition_state_only=true)
    s=initialize_filter_smc(target;rng=Xoshiro(1),members=24,particles=16,
        boundary=trained,window_length=2,ess_fraction=0.8,
        options=SMCOptions(moves=1,stepsize=0.15,max_tree_depth=3))
    @test s.time==3 && s.log_normalizer==0
    s=filter_step(s,(;y=0.9))
    cp=filter_checkpoint(s)
    a=filter_step(s,(;y=1.1))
    b=filter_step(resume_filter(P.learning(carried),cp),(;y=1.1))
    @test F.equal_signatures(F.signature(a),F.signature(b))
    @test_throws ArgumentError resume_filter(P.learning(carried;model_id="changed-filter"),cp)
    @test_throws ArgumentError prepare_filter_smc(P.static_filter(),carried;
        backend=F.BACKEND,calibration_model_id="changed-batch")
    @test_throws ArgumentError initialize_filter_smc(P.learning(carried;model_id="changed-filter");
        rng=Xoshiro(1),boundary=trained)
    frozen=deepcopy(carried.reference.location)
    state.positions .= 1000
    @test carried.reference.location==frozen
end

@testset "Projection drops old paths and folds scalar bound data" begin
    projections=map((8,4096)) do n
        f=P.full(n)
        R._smc_parameter_projection(f.built.layout,(:theta,:sigma))
    end
    @test F.EXT._same(projections[1].record,projections[2].record)
    @test Base.summarysize(projections[1].record)==Base.summarysize(projections[2].record)
    records=map((8,4096)) do n
        data=(;x=fill(3.0,n),y=[0.2])
        plan=bind_data(lower_rkppl(quote
            a ~ Normal()
            width=sum(x)/length(x)
            b ~ Uniform(a,a+width)
            y .~ Normal.(a+b,1)
        end,data;conditioned=(:y,)),data)
        layout=build_kernel(plan).layout
        projected=R._smc_parameter_projection(layout,(:a,:b))
        compact=R._smc_projection_layout(projected.record)
        @test constrain(compact,[0.2,-0.7])==constrain(layout,[0.2,-0.7])
        @test isempty(compact.bound_values)
        @test_throws ArgumentError R._smc_parameter_projection(layout,(:b,))
        projected.record
    end
    @test F.EXT._same(records[1],records[2])
    @test Base.summarysize(records[1])==Base.summarysize(records[2])
    @info "Static geometry after history projection" old_lengths=(8,4096) retained_bytes=Base.summarysize(projections[1].record) bound_geometry_bytes=Base.summarysize(records[1])
end

@rkppl projection_ar(T) = begin
    phi_raw ~ Normal()
    @scan begin
        h[1] ~ Normal()
        for t in 2:T
            h[t] ~ Normal(tanh(phi_raw)*h[t-1],1)
        end
    end
    h
end
@testset "Nested static author paths" begin
    model=@rkppl begin
        z ~ projection_ar(T)
        y .~ Normal.(z,0.5)
    end
    plan=model(;T=4)|(;y=zeros(4))
    layout=build_kernel(plan).layout
    names=filter_parameter_names(model)
    projected=R._smc_parameter_projection(layout,names)
    compact=R._smc_projection_layout(projected.record)
    @test names==(Symbol("z.phi_raw"),)
    @test compact.total==1
    @test constrain(compact,[0.3]).z.phi_raw==0.3
end
