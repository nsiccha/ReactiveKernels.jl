# Optional integration suite: run in an environment containing RKPPL, Enzyme,
# DifferentiationInterface, SequentialMonteCarlo, WarmupHMC and LogDensityProblems.
# Kept separate so ordinary PPL users do not need inference dependencies.
using ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
using SequentialMonteCarlo, WarmupHMC, LogDensityProblems, Random
using Statistics, Test, Serialization
using ReactiveKernelsPPL.SpecialFunctions: loggamma

const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)
const DATA = [0.9, 1.6, 0.2, 1.3, 0.6, 1.1]
const GAUSSIAN = quote
    mu ~ Normal(0, 1)
    y .~ Normal.(mu, 1)
end

function target_for(src, data; model_id="gaussian-v1", binding_id="data", built=nothing)
    plan = bind_data(lower_rkppl(src, data; conditioned=(:y,)), data)
    built === nothing && (built = build_kernel(plan))
    target = prepare_smc(built, plan, zeros(built.layout.total);
        backend=BACKEND, model_id, binding_id)
    return (; target, plan, built)
end

normal(x, mu=0.0, sigma=1.0) = -log(2pi)/2-log(sigma)-(x-mu)^2/(2sigma^2)
gaussian_logz(y, variance=1.0) = -(length(y)*log(2pi) + log(1+variance*length(y)) +
    sum(abs2,y) - variance*sum(y)^2/(1+variance*length(y)))/2

function equal_clouds(a,b)
    @test a.positions == b.positions
    @test a.weights == b.weights
    @test a.temperature == b.temperature
    @test a.log_normalizer == b.log_normalizer
    @test a.ancestors == b.ancestors
    @test a.lineage == b.lineage
    @test a.history == b.history
    @test rand(copy(a.rng), UInt64, 10) == rand(copy(b.rng), UInt64, 10)
end

@testset "SMC batch, cumulative updates and checkpoint replay" begin
    f = target_for(GAUSSIAN, (; y=DATA); binding_id="six")
    input = copy(DATA)
    rng = Xoshiro(121)
    saved_rng = copy(rng)
    Random.seed!(8181)
    parent_rng = copy(Random.default_rng())
    start = initialize_smc(f.target; rng, particles=2500,
        reference=SMCReference([0.3], [1.2]))
    initial = copy(start.positions)
    partial = smc(start; stages=1)
    @test 0 < partial.temperature < 1
    @test start.positions == initial
    @test start.temperature == 0
    @test rand(copy(rng), UInt64,10) == rand(saved_rng, UInt64,10)
    @test rand(copy(Random.default_rng()), UInt64,10) == rand(parent_rng, UInt64,10)
    checkpoint = smc_checkpoint(partial)
    buffer = IOBuffer()
    serialize(buffer, checkpoint)
    seekstart(buffer)
    restored = resume_smc(f.target, deserialize(buffer))
    batch = smc(partial)
    equal_clouds(batch, smc(restored))
    @test batch.temperature == 1
    @test mean(batch.positions) ≈ sum(DATA)/7 atol=0.04
    @test var(batch.positions) ≈ 1/7 atol=0.025
    @test batch.log_normalizer ≈ gaussian_logz(DATA) atol=0.12
    @test all(==(1/2500), batch.weights)
    @test all(s -> 0 < s.ess_before_resampling <= 2500.00001, batch.history)
    @test all(s -> s.unique_parents <= 2500, batch.history)
    @test all(s -> s.moves == 5000, batch.history)
    @test DATA == input
    @test_throws ArgumentError update_smc(partial, f.target)

    # A frozen program is prepared for each cumulative prefix. The target is
    # always prior + entire prefix, never old posterior + a second prior.
    firstfit = target_for(GAUSSIAN, (; y=DATA[1:1]); binding_id="prefix-1")
    online = smc(firstfit.target; rng=Xoshiro(343), particles=2500)
    for n in 2:length(DATA)
        next = target_for(GAUSSIAN, (; y=DATA[1:n]); binding_id="prefix-$n", built=firstfit.built)
        before = copy(online.positions)
        updating = update_smc(online, next.target; stages=0)
        if n == 3
            checkpoint = smc_checkpoint(smc_step(updating))
            resumed = resume_smc(next.target, checkpoint; previous=online.target)
            equal_clouds(smc(resumed), smc(smc_step(updating)))
            @test_throws ArgumentError resume_smc(next.target, checkpoint)
            @test_throws ArgumentError resume_smc(next.target, checkpoint; previous=f.target)
        end
        new = smc(updating)
        @test online.positions == before
        @test mean(new.positions) ≈ sum(DATA[1:n])/(n+1) atol=0.05
        @test var(new.positions) ≈ 1/(n+1) atol=0.03
        @test new.log_normalizer ≈ gaussian_logz(DATA[1:n]) atol=0.15
        @test new.update == n-1
        online = new
    end
    @test mean(online.positions) ≈ mean(batch.positions) atol=0.05

    # Reference normalization, including the scale determinant, and AD.
    source = start.source
    g = zeros(1)
    @test EXT._density(source,[0.7]) ≈ normal(0.7,0.3,1.2)
    @test EXT._gradient(source,g,[0.7]) ≈ normal(0.7,0.3,1.2)
    @test g ≈ [-(0.7-0.3)/1.2^2]

    badframe = prepare_smc(f.built,f.plan,[0.0]; backend=BACKEND,
        model_id="another-frame",binding_id="six")
    @test_throws ArgumentError update_smc(batch,badframe)
    changed = target_for(quote
        mu ~ Normal(0, 2)
        y .~ Normal.(mu, 1)
    end,(; y=DATA); model_id="gaussian-wide-v1", binding_id="six")
    @test_throws ArgumentError update_smc(batch,changed.target)
    wrongdata = target_for(GAUSSIAN,(; y=DATA .+ 1); binding_id="six",built=f.built)
    @test_throws ArgumentError resume_smc(wrongdata.target,smc_checkpoint(batch))
    @test_throws ArgumentError resume_smc(f.target,merge(checkpoint,(; schema_version=2)))
    @test_throws ArgumentError resume_smc(f.target,merge(smc_checkpoint(batch),(; positions=zeros(0,2500))))
end

@testset "Constrained coordinate Jacobian and dependent priors" begin
    counts = [1,2,0,3,1]
    f = target_for(quote
        rate ~ Exponential(1)
        y .~ Poisson.(rate)
    end,(; y=counts); model_id="poisson-v1")
    for u in (-1.0, 0.3, 1.0)
        # Exactly one log Jacobian, including in the posterior endpoint.
        expected = (sum(counts)+1)*u-(length(counts)+1)*exp(u)-sum(loggamma.(counts .+ 1))
        @test f.target.query([u]) ≈ expected
        bridge = EXT.BridgeTarget(EXT._reference(f.target,SMCReference([0.0],[1.0])),f.target,0.4,1)
        value,gradient = LogDensityProblems.logdensity_and_gradient(bridge,[u])
        @test value ≈ 0.6normal(u)+0.4expected
        @test gradient ≈ [-0.6u+0.4*(sum(counts)+1-(length(counts)+1)*exp(u))]
    end
    result = smc(f.target; rng=Xoshiro(901),particles=3000)
    rates = exp.(result.positions)
    a,b = sum(counts)+1,length(counts)+1
    @test mean(rates) ≈ a/b atol=0.06
    @test var(rates) ≈ a/b^2 atol=0.04
    @test result.log_normalizer ≈ loggamma(a)-a*log(b)-sum(loggamma.(counts .+ 1)) atol=0.13

    dependent = target_for(quote
        mu ~ Normal(0,1)
        z ~ Normal(mu,1)
        y .~ Normal.(z,1)
    end,(; y=DATA); model_id="hierarchical-v1")
    names = coordinate_names(dependent.built.layout)
    mu, z = findfirst(==(:mu),names),findfirst(==(:z),names)
    @test mu !== nothing && z !== nothing
    u = [0.2,-0.3]
    @test dependent.target.query(u) ≈ normal(u[mu])+normal(u[z],u[mu])+sum(normal(y,u[z]) for y in DATA)
    result = smc(dependent.target; rng=Xoshiro(455),particles=3500,
        options=SMCOptions(moves=3))
    n = length(DATA)
    @test mean(result.positions[mu,:]) ≈ sum(DATA)/(2n+1) atol=0.055
    @test mean(result.positions[z,:]) ≈ 2sum(DATA)/(2n+1) atol=0.045
    @test var(result.positions[mu,:]) ≈ (n+1)/(2n+1) atol=0.065
    @test var(result.positions[z,:]) ≈ 2/(2n+1) atol=0.025
    @test cov(result.positions[mu,:],result.positions[z,:]) ≈ 1/(2n+1) atol=0.035
    @test result.log_normalizer ≈ gaussian_logz(DATA,2.0) atol=0.13
end

@testset "Deterministic scan replay after parameter mutation" begin
    src = quote
        a ~ Normal(0,1)
        @scan begin
            h[1] = a*x[1]
            for t in 2:T
                h[t] = 0.7*h[t-1]+a*x[t]
            end
        end
        y .~ Normal.(h,1)
    end
    x = [0.3,-0.2,0.7,0.4,-0.1]
    y = [0.4,-0.1,0.9,0.8,0.2]
    coefficients = accumulate((h,v) -> 0.7h+v,x)
    f = target_for(src,(; x,y,T=length(x)); model_id="scan-v1",binding_id="five")
    for a in (-0.3,0.4,1.1)
        @test f.target.query([a]) ≈ normal(a)+sum(normal(y[i],a*coefficients[i]) for i in eachindex(y))
    end
    prefix = target_for(src,(; x=x[1:2],y=y[1:2],T=2); model_id="scan-v1",binding_id="two")
    old = smc(prefix.target; rng=Xoshiro(773),particles=2000)
    new = update_smc(old,f.target)
    variance = 1/(1+sum(abs2,coefficients))
    @test mean(new.positions) ≈ variance*sum(coefficients.*y) atol=0.06
    @test var(new.positions) ≈ variance atol=0.06
    # Evaluate both old and new full recurrences at a moved position.
    a = new.positions[1,1]
    @test new.source.query([a]) ≈ normal(a)+sum(normal(y[i],a*coefficients[i]) for i in 1:2)
    @test new.target.query([a]) ≈ normal(a)+sum(normal(y[i],a*coefficients[i]) for i in eachindex(y))
    @test recipe_inventory(f.built.spec) |> x -> any(e -> e.kind === :scan,x)
end

@testset "Weight failures, bounded history and linear state" begin
    @test_throws ArgumentError SMCOptions(metric=ones(2,2))
    @test_throws ArgumentError SMCOptions(ess_fraction=1)
    @test_throws ArgumentError SMCReference([0.0],[0.0])
    @test_throws ArgumentError EXT._temperature(fill(-Inf,4),0.0,SMCOptions())
    @test_throws SMCNonfiniteError EXT._checked(NaN, [0.0], :target, 0.5)
    @test_throws SMCNonfiniteError EXT._checked(Inf, [0.0], :target, 0.5)
    @test EXT._checked(-Inf, [0.0], :target, 0.5) == -Inf
    @test EXT._ess([0.0,-Inf],1.0) == 1
    @test EXT._ess([-1e200,-1e200],1.0) == 2
    @test EXT._ess([-1e308,1e308],1e-308) ≈ (1+exp(-2))^2/(1+exp(-4))
    @test_throws ArgumentError EXT._temperature([0.0,-Inf,-Inf,-Inf],0.0,SMCOptions())
    f = target_for(GAUSSIAN,(; y=fill(20.0,5)); binding_id="far")
    start = initialize_smc(f.target; rng=Xoshiro(10),particles=20,
        options=SMCOptions(max_stages=1))
    partial = smc_step(start)
    @test partial.temperature < 1
    @test_throws ArgumentError smc_step(partial)
    @test start.stage == 0
    @test partial.stage == 1
    bounded = smc(f.target; rng=Xoshiro(10),particles=100,
        options=SMCOptions(history_limit=2,moves=3))
    @test bounded.stage > 2
    @test length(bounded.history) == 2
    @test bounded.temperature == 1

    # Measure real checkpoints at growing model dimension and particle count;
    # these include RNG, ancestry, metadata and all inference-owned arrays.
    function footprint(d,n)
        t = target_for(quote
            b[1:d] .~ Normal.(0,1)
            y .~ Normal.(sum(b),1)
        end,(; d,y=[0.0]); model_id="dimension-$d")
        state = initialize_smc(t.target; rng=Xoshiro(4),particles=n,
            options=SMCOptions(metric=ones(d)))
        @test size(state.positions) == (d,n)
        @test length(state.options.metric) == d
        return Base.summarysize(smc_checkpoint(state))
    end
    @test footprint(400,200)/footprint(200,200) < 2.01
    @test footprint(200,400)/footprint(200,200) < 2.01
    a = initialize_smc(f.target; rng=Xoshiro(4),particles=10)
    b = initialize_smc(f.target; rng=Xoshiro(4),particles=100)
    @test Base.summarysize(smc_checkpoint(b))/Base.summarysize(smc_checkpoint(a)) < 10
    @test sizeof(EXT.IndexParticle) == 2sizeof(Int)

    emptydata = target_for(GAUSSIAN,(; y=Float64[]); binding_id="empty")
    prior = smc(emptydata.target; rng=Xoshiro(661),particles=1500)
    @test prior.temperature == 1
    @test prior.log_normalizer ≈ 0 atol=1e-12
    @test mean(prior.positions) ≈ 0 atol=0.07
    @test var(prior.positions) ≈ 1 atol=0.11

    fixed = target_for(quote
        y .~ Normal.(0,1)
    end,(; y=DATA); model_id="no-parameters")
    fixedfit = smc(fixed.target; rng=Xoshiro(66),particles=10)
    @test size(fixedfit.positions) == (0,10)
    @test fixedfit.log_normalizer ≈ sum(normal,DATA)
    @test all(s -> s.moves == 0,fixedfit.history)

    invalid = target_for(GAUSSIAN,(; y=[Inf]); binding_id="infinite")
    failed = initialize_smc(invalid.target; rng=Xoshiro(6),particles=8)
    failed_rng = copy(failed.rng)
    @test_throws ArgumentError smc_step(failed)
    @test failed.temperature == 0
    @test failed.rng == failed_rng
    invalid = target_for(GAUSSIAN,(; y=[NaN]); binding_id="nan")
    @test_throws SMCNonfiniteError smc(invalid.target; rng=Xoshiro(6),particles=8)
end

include("nonfinite.jl")
include("prepared_logdensity.jl")
include("rejuvenation.jl")

@testset "Fresh-process checkpoint continuation" begin
    mktempdir() do directory
        checkpoint = joinpath(directory,"checkpoint.bin")
        fixture = joinpath(@__DIR__,"checkpoint_process.jl")
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $fixture`
        @test success(pipeline(`$command write $checkpoint`; stdout=stdout,stderr=stderr))
        @test success(pipeline(`$command read $checkpoint`; stdout=stdout,stderr=stderr))
    end
end
