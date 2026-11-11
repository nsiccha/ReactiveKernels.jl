isdefined(@__MODULE__, :SMCSelectiveFixture) || include("selective_fixture.jl")
module SMCSelectiveGrowthTests
using ..SMCSelectiveFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, LinearAlgebra, Statistics, Test, Serialization
const F=SMCSelectiveFixture
const HIERARCHICAL=quote
    m ~ Normal(0,4)
    z[1:nb] .~ Normal.(m,2)
    y .~ Normal.(z[block],1)
end

@testset "Fresh independent prior includes its old hyperparent" begin
    oldbuild=F.build(2,Int[],Float64[];source=HIERARCHICAL)
    old=F.calibration(oldbuild,SMCReference([1.0,-0.3,0.4],[0.5,0.7,0.9]))
    f=F.build(3,[3],[3.0];source=HIERARCHICAL)
    t=F.window(f,old)
    c=smc_window_coordinates(t)
    @test c.current_indices==[1,4]
    @test c.active_old==[1]
    @test c.active_slots==[1]
    @test c.fresh_slots==[2]
    u,g=[1.2,2.4],zeros(2)
    expected=F.normal(u[1],1.0,0.5)+F.normal(u[2],u[1],2)+F.normal(3,u[2],1)
    @test F.EXT._density(t,u) ≈ expected atol=1e-12
    @test F.EXT._gradient(t,g,u) ≈ expected atol=1e-12
    @test g ≈ [-(u[1]-1)/0.25+(u[2]-u[1])/4, -(u[2]-u[1])/4+3-u[2]] atol=1e-10
    @test_throws ArgumentError F.window(f,old;coordinates=:all)
    @test_throws ArgumentError F.window(f,old;model_id="changed")
    initial=initialize_smc(t;rng=Xoshiro(82),particles=2200,
        options=SMCOptions(stepsize=0.35,moves=2,max_tree_depth=5))
    @test size(initial.positions)==(2,2200)
    cp=smc_checkpoint(smc_step(initial))
    fit=smc(resume_smc(t,cp))
    rebuilt=F.window(F.build(3,[3],[3.0];source=HIERARCHICAL),old)
    @test F.signature(fit)==F.signature(smc(resume_smc(rebuilt,cp)))
    changed=F.window(F.build(3,[3],[4.0];source=HIERARCHICAL),old)
    @test_throws ArgumentError resume_smc(changed,cp)
    @test vec(mean(fit.positions;dims=2)) ≈ [1+0.25/5.25*2,1+4.25/5.25*2] atol=0.06
    @test fit.log_normalizer ≈ F.normal(3,1,sqrt(5.25)) atol=0.12
    carried=carry_smc(fit;calibration_id="window-1",basis=:fit,max_reflections=4)
    @test length(carried.reference.location)==4
    @test carried.reference.location[2:3]==[-0.3,0.4]
    @test carried.reference.location[[1,4]] ≈ vec(mean(fit.positions;dims=2))
    @test length(carried.reference.scale.values)<=4
    # An old coordinate that was inactive becomes active in the next window.
    next=F.window(F.build(3,[1],[0.1];source=HIERARCHICAL),carried;binding_id="window-2")
    @test smc_window_coordinates(next).current_indices==[2]
    @test F.EXT._density(next,[-0.2]) ≈ F.normal(-0.2,-0.3,0.7)+F.normal(0.1,-0.2,1) atol=1e-10
end

# BRM names a Gaussian vector prior's arguments (`z_prior_scale = s`). With the
# scale bound as data, the fresh prior's plate reads it through that alias, and
# preparing the window failed with `cannot merge structural aliases for value
# :z_prior_scale: Number does not match Float64` (snag selective-fresh-96f7fe96).
@testset "Fresh prior with a data-valued scale, named or read directly" begin
    named=quote
        m ~ Normal(0,4)
        z_prior_mean = m
        z_prior_scale = s
        z[1:nb] .~ Normal.(z_prior_mean,z_prior_scale)
        y .~ Normal.(z[block],1)
    end
    direct=quote
        m ~ Normal(0,4)
        z[1:nb] .~ Normal.(m,s)
        y .~ Normal.(z[block],1)
    end
    function scaled(source,nb,block,y)
        data=(; nb,block=Int.(block),y=Float64.(y),s=2.0)
        plan=bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
        (; built=build_kernel(plan),plan)
    end
    for source in (named,direct)
        old=F.calibration(scaled(source,2,Int[],Float64[]),
            SMCReference([1.0,-0.3,0.4],[0.5,0.7,0.9]))
        t=F.window(scaled(source,3,[3],[3.0]),old)
        c=smc_window_coordinates(t)
        @test c.current_indices==[1,4]
        @test c.active_old==[1]
        @test c.fresh_slots==[2]
        u,g=[1.2,2.4],zeros(2)
        expected=F.normal(u[1],1.0,0.5)+F.normal(u[2],u[1],2)+F.normal(3,u[2],1)
        @test F.EXT._density(t,u) ≈ expected atol=1e-12
        @test F.EXT._gradient(t,g,u) ≈ expected atol=1e-12
        @test g ≈ [-(u[1]-1)/0.25+(u[2]-u[1])/4, -(u[2]-u[1])/4+3-u[2]] atol=1e-10
    end
end

@testset "Fresh transformed plate factors and Jacobians exactly once" begin
    source=quote
        z[1:nb] .~ LogNormal.(0,1)
        y .~ Normal.(z[block],1)
    end
    oldbuild=F.build(1,Int[],Float64[];source)
    old=F.calibration(oldbuild,SMCReference([0.3],[0.4]))
    f=F.build(3,[3],[1.5];source)
    t=F.window(f,old)
    @test smc_window_coordinates(t).current_indices==[2,3]
    @test isempty(smc_window_coordinates(t).active_old)
    u,g=[-0.2,0.4],zeros(2)
    expected=F.normal(u[1],0,1)+F.normal(u[2],0,1)+F.normal(1.5,exp(u[2]),1)
    @test F.EXT._density(t,u) ≈ expected atol=1e-12
    @test F.EXT._gradient(t,g,u) ≈ expected atol=1e-12
    @test g ≈ [-u[1],-u[2]+exp(u[2])*(1.5-exp(u[2]))] atol=1e-10
    # No old active coordinates: recovery still preserves the independent
    # carried coordinate and the fresh particle's unconstrained values.
    recovered=smc_recover(t,u;rng=Xoshiro(18))
    @test recovered[2:3]==u
    @test length(recovered)==3
end

@testset "Fresh-window particle storage is compact" begin
    oldbuild=F.build(10_000,Int[],Float64[];source=HIERARCHICAL)
    old=F.calibration(oldbuild,SMCReference(zeros(10_001),ones(10_001)))
    t=F.window(F.build(10_001,[10_001],[0.5];source=HIERARCHICAL),old)
    @test smc_window_coordinates(t).current_indices==[1,10_002]
    a=initialize_smc(t;rng=Xoshiro(3),particles=32)
    b=initialize_smc(t;rng=Xoshiro(3),particles=256)
    @test size(a.positions)==(2,32)
    @test size(b.positions)==(2,256)
    @test (Base.summarysize(b)-Base.summarysize(a))/(256-32)<4096
    @test (Base.summarysize(smc_checkpoint(b))-Base.summarysize(smc_checkpoint(a)))/(256-32)<4096
end
end
