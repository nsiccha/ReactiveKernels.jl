isdefined(@__MODULE__, :SMCSelectiveFixture) || include("selective_fixture.jl")
module SMCSelectiveExistingTests
using ..SMCSelectiveFixture
using ReactiveKernels, ReactiveKernelsPPL, Random, LinearAlgebra, Statistics, Test, Serialization
const F=SMCSelectiveFixture

@testset "Selective old marginal and lazy correlated recovery" begin
    f=F.build(3,[2],[1.4])
    scale=(; diagonal=[sqrt(2.0),sqrt(0.02),0.8],
        indices=[[1,2]],reflections=[[cos(pi/8),sin(pi/8)]])
    old=F.calibration(f,SMCReference([0.2,-0.1,0.7],scale))
    t=F.window(f,old)
    c=smc_window_coordinates(t)
    @test c.current_indices == [2]
    @test c.active_old == [2]
    @test isempty(c.fresh_slots)
    u,g=[0.4],zeros(1)
    expected=F.normal(u[1],-0.1,sqrt(1.01))+F.normal(1.4,u[1],1.0)
    @test F.EXT._density(t,u) ≈ expected atol=1e-12
    @test F.EXT._gradient(t,g,u) ≈ expected atol=1e-12
    @test g ≈ [-(u[1]+0.1)/1.01+1.4-u[1]] atol=1e-10

    rng=Xoshiro(62)
    recovered=hcat([smc_recover(t,u;rng) for _ in 1:8000]...)
    @test all(==(u[1]),recovered[2,:])
    @test mean(recovered[1,:]) ≈ 0.2+0.99/1.01*0.5 atol=0.01
    @test var(recovered[1,:]) ≈ 1.01-0.99^2/1.01 atol=0.003
    @test mean(recovered[3,:]) ≈ 0.7 atol=0.025
    @test var(recovered[3,:]) ≈ 0.64 atol=0.025

    state=initialize_smc(t;rng=Xoshiro(17),particles=1600,
        options=SMCOptions(stepsize=0.4,moves=2,max_tree_depth=4))
    @test size(state.positions)==(1,1600)
    @test isequal(smc_window_coordinates(state),c)
    @test smc_recover(state,u;rng=Xoshiro(11))==smc_recover(t,u;rng=Xoshiro(11))
    partial=smc_step(state)
    cp=smc_checkpoint(partial)
    io=IOBuffer();serialize(io,cp);seekstart(io)
    fit=smc(partial)
    @test F.signature(fit)==F.signature(smc(resume_smc(t,deserialize(io))))
    variance=1/(1/1.01+1)
    location=variance*(-0.1/1.01+1.4)
    @test mean(fit.positions) ≈ location atol=0.05
    @test var(vec(fit.positions)) ≈ variance atol=0.04
    @test fit.log_normalizer ≈ F.normal(1.4,-0.1,sqrt(2.01)) atol=0.08
    carried=carry_smc(fit;calibration_id="next",max_reflections=4)
    @test length(carried.reference.location)==3
    @test carried.reference.location[1] ≈ 0.2+0.99/1.01*(mean(fit.positions)+0.1)
    @test carried.reference.location[2] ≈ mean(fit.positions)
    @test carried.reference.location[3] == 0.7
    @test length(carried.reference.scale.values)<=4
    @test !hasproperty(carried,:target)
    changed=F.window(F.build(3,[2],[2.0]),old)
    @test_throws ArgumentError resume_smc(changed,cp)
    changed_id=F.window(f,old;binding_id="changed")
    @test_throws ArgumentError resume_smc(changed_id,cp)
    # The public descriptor is an owned snapshot.
    c.current_indices[1]=1
    @test smc_window_coordinates(t).current_indices==[2]
end

@testset "Selective particle and gradient storage" begin
    function prepared(n)
        f=F.build(n,[n],[0.5])
        old=F.calibration(f,SMCReference(zeros(n),ones(n)))
        F.window(f,old)
    end
    t=prepared(10_001)
    a=initialize_smc(t;rng=Xoshiro(4),particles=32)
    b=initialize_smc(t;rng=Xoshiro(4),particles=256)
    @test size(a.positions)==(1,32)
    @test size(b.positions)==(1,256)
    @test (Base.summarysize(b)-Base.summarysize(a))/(256-32)<4096
    @test (Base.summarysize(smc_checkpoint(b))-Base.summarysize(smc_checkpoint(a)))/(256-32)<4096
    # Warm both gradients before comparing allocation. No full authored
    # vector or its AD shadow may be allocated for a one-coordinate target.
    small=prepared(101)
    u,g=[0.2],zeros(1)
    F.EXT._gradient(small,g,u);F.EXT._gradient(t,g,u)
    smallbytes=@allocated F.EXT._gradient(small,g,u)
    largebytes=@allocated F.EXT._gradient(t,g,u)
    @test largebytes-smallbytes<16_384
    @info "Selective old window storage" smallbytes largebytes state_bytes=Base.summarysize(b) checkpoint_bytes=Base.summarysize(smc_checkpoint(b))
end
end
