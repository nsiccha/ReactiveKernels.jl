include("window_fixture.jl")
using .SMCWindowFixture
using ReactiveKernelsPPL, Random, LinearAlgebra, Statistics, Test, Serialization
const W = SMCWindowFixture

@testset "Carried Gaussian plus new likelihood, priors and Jacobians once" begin
    previous = W.calibration()
    before = deepcopy(previous)
    y = [0.9,1.2,0.8]
    target = W.window(y;previous)
    u,g = [0.4,-0.3], zeros(2)
    delta = u-W.LOCATION
    prior = -log(2pi)-logdet(W.COVARIANCE)/2-dot(delta,W.COVARIANCE\delta)/2
    likelihood = -length(y)*log(2pi)/2-sum(abs2,y.-sum(u))/2
    @test W.EXT._density(target,u) ≈ prior+likelihood atol=1e-12
    @test W.EXT._gradient(target,g,u) ≈ prior+likelihood atol=1e-12
    @test g ≈ -(W.COVARIANCE\delta) .+ sum(y.-sum(u)) atol=1e-10
    @test previous.reference.location == before.reference.location
    @test previous.reference.scale == before.reference.scale
    @test W.EXT._same(previous.frame,before.frame)
    @test_throws ArgumentError W.window(y;previous,model_id="different-preprocessing")
    @test_throws ArgumentError carry_smc(target,previous.reference;calibration_id="")
    @test_throws DimensionMismatch carry_smc(target,SMCReference([0.0],[1.0]);calibration_id="bad")

    rng = Xoshiro(402)
    saved_rng = copy(rng)
    Random.seed!(1251)
    parent_rng = copy(Random.default_rng())
    options = SMCOptions(metric=W.SCALE,stepsize=0.3,moves=2,history_limit=3)
    initial = initialize_smc(target;rng,particles=1600,options)
    @test smc_reference(initial).scale == previous.reference.scale
    @test smc_reference(initial).location == previous.reference.location
    @test_throws ArgumentError carry_smc(initial;calibration_id="unfinished")
    partial = smc_step(initial)
    cp = smc_checkpoint(partial)
    fit = smc(partial)
    @test W.signature(fit) == W.signature(smc(resume_smc(target,cp)))
    @test rand(copy(rng),UInt64,16) == rand(saved_rng,UInt64,16)
    @test rand(copy(Random.default_rng()),UInt64,16) == rand(parent_rng,UInt64,16)
    @test initial.temperature == 0
    @test length(fit.history) <= 3
    h = ones(2)
    covariance = inv(inv(W.COVARIANCE)+length(y)*(h*h'))
    location = covariance*(W.COVARIANCE\W.LOCATION+h*sum(y))
    @test vec(mean(fit.positions;dims=2)) ≈ location atol=0.06
    @test cov(fit.positions;dims=2) ≈ covariance atol=0.04
    marginal = Matrix{Float64}(I,length(y),length(y)) .+ dot(h,W.COVARIANCE*h)
    centered = y .- sum(W.LOCATION)
    logz = -(length(y)*log(2pi)+logdet(marginal)+dot(centered,marginal\centered))/2
    @test fit.log_normalizer ≈ logz atol=0.12

    carried = carry_smc(fit;calibration_id="window-1")
    @test carried.reference.location ≈ vec(mean(fit.positions;dims=2))
    @test carried.reference.scale ≈ vec(std(fit.positions;dims=2,corrected=false))
    correlated = carry_smc(fit;calibration_id="window-1",scale=W.SCALE)
    @test correlated.reference.scale == W.SCALE
    @test !hasproperty(carried,:target)
    next = W.window([1.4];previous=correlated,binding_id="next-window")
    @test_throws ArgumentError resume_smc(next,cp)
    changed = W.window(y;previous=W.calibration(calibration_id="different-calibration"))
    @test_throws ArgumentError resume_smc(changed,cp)
    previous.reference.location[1] = 99.0
    @test W.EXT._density(target,u) ≈ prior+likelihood atol=1e-12
end

@testset "Window tuning and checkpoints bind both likelihood and carried prior" begin
    t = W.window()
    rng, init = Xoshiro(123), [0.1,-0.1]
    rng0, init0 = copy(rng), copy(init)
    schedule = tune_smc(t;rng,init,temperatures=(0.0,0.1,1.0),pilot_steps=12)
    @test init == init0
    @test rand(copy(rng),UInt64,16) == rand(rng0,UInt64,16)
    @test smc_reference(schedule).scale == W.SCALE
    initial = initialize_smc(t;rng=Xoshiro(125),particles=32,
        options=SMCOptions(tuning=schedule))
    cp = smc_checkpoint(smc_step(initial))
    @test W.signature(smc(resume_smc(t,cp))) == W.signature(smc(smc_step(initial)))
    other = W.window([2.0];binding_id="changed-window")
    @test_throws ArgumentError initialize_smc(other;rng=Xoshiro(125),particles=32,
        options=SMCOptions(tuning=schedule))
end

@testset "Window likelihood transforms coordinates without adding another Jacobian" begin
    source = quote
        sigma ~ LogNormal(3,0.2)
        y .~ Normal.(sigma,1)
    end
    data = (; y=[1.0,2.0])
    plan = bind_data(lower_rkppl(source,data;conditioned=(:y,)),data)
    built = build_kernel(plan)
    base = prepare_smc(built,plan,[0.0];backend=W.BACKEND,
        model_id="lognormal-frame",binding_id="old")
    previous = carry_smc(base,SMCReference([0.2],[0.7]);calibration_id="old")
    t = prepare_smc_window(built,plan,[0.0];previous,backend=W.BACKEND,
        model_id="lognormal-frame",binding_id="new")
    u,g = [0.4],zeros(1)
    expected = -log(2pi)/2-log(0.7)-(u[1]-0.2)^2/(2*0.7^2) +
        sum(y -> -log(2pi)/2-(y-exp(u[1]))^2/2,data.y)
    @test W.EXT._density(t,u) ≈ expected atol=1e-12
    @test W.EXT._gradient(t,g,u) ≈ expected atol=1e-12
    @test g[1] ≈ -(u[1]-0.2)/0.7^2 + exp(u[1])*sum(data.y .- exp(u[1])) atol=1e-10
end

@testset "Carried summary drops previous observations" begin
    small, large = W.calibration(n=8), W.calibration(n=4096)
    @test W.EXT._same(small.frame,large.frame)
    @test Base.summarysize(small) == Base.summarysize(large)
    tsmall, tlarge = W.window(previous=small), W.window(previous=large)
    @test W.EXT._density(tsmall,[0.1,0.2]) == W.EXT._density(tlarge,[0.1,0.2])
    # Graph Dict bucket capacity varies even between two builds with the
    # same old history. Compare retained data and logical graph size, not
    # spare compiler hash-table slots counted by whole-target summarysize.
    @test W.EXT._same(tsmall.signature,tlarge.signature)
    @test Base.summarysize(tsmall.query.kernel.constants) ==
        Base.summarysize(tlarge.query.kernel.constants)
    @test length(tsmall.query.kernel.plan.graph.values) ==
        length(tlarge.query.kernel.plan.graph.values)
    checkpoints = map((tsmall,tlarge)) do target
        smc_checkpoint(initialize_smc(target;rng=Xoshiro(722),particles=8))
    end
    @test Base.summarysize(checkpoints[1]) == Base.summarysize(checkpoints[2])
    # Report wall time as measurement, not a timing-based correctness threshold.
    g,u = zeros(2),[0.1,0.2]
    for target in (tsmall,tlarge)
        W.EXT._gradient(target,g,u)
    end
    times = map((tsmall,tlarge)) do target
        median([@elapsed(for _ in 1:1000; W.EXT._gradient(target,g,u); end) for _ in 1:5])/1000
    end
    @info "Fixed three-row window at old history lengths 8 and 4096" summary_bytes=Base.summarysize(small) target_bytes=Base.summarysize.((tsmall,tlarge)) checkpoint_bytes=Base.summarysize(checkpoints[1]) gradient_seconds=times
end
