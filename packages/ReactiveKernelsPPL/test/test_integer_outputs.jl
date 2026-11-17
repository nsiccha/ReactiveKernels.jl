module IntegerOutputChecks
using ReactiveKernels, ReactiveKernelsPPL, Distributions, Enzyme, Test
using DifferentiationInterface: AutoEnzyme
using Random: Xoshiro

@testset "integer matrix declarations preserve their shapes and element types" begin
    x,y,u = zeros(2,3),fill(0.4,2,3),[0.3]
    for (rhs, data, state, law) in (
            (:(Binomial(n,p)), (; n=3,p=0.3), reshape([0,1,2,3,0,1],2,3), Binomial(3,0.3)),
            (:(Bernoulli(p)), (; p=0.3), reshape([0,1,0,1,0,1],2,3), Bernoulli(0.3)),
            (:(Categorical(p)), (; p=[0.2,0.3,0.5]), reshape([1,2,3,1,2,3],2,3), Categorical([0.2,0.3,0.5])),
            (:(Categorical(p .* 1)), (; p=[0.2,0.3,0.5]), reshape([1,2,3,1,2,3],2,3), Categorical([0.2,0.3,0.5])),
            (:(DiscreteUniform(lo,hi)), (; lo=-2,hi=3), reshape(collect(-2:3),2,3), DiscreteUniform(-2,3)))
        model = RKPPLModel(quote
            mu ~ Normal(0,1)
            k[axes(x,1),axes(x,2)] .~ $rhs
            y .~ Normal.(mu .+ k,1)
        end, @__MODULE__)
        plan = model(;x,data...) | (;y)
        built = build_kernel(plan)
        expected = logpdf(Normal(),only(u))+sum(logpdf.(law,state))+
            sum(logpdf.(Normal.(only(u).+state,1),y))
        @test prepare_query(built,plan,:sampler)(u,(;k=state)) ≈ expected
        draw = constrain(built.layout,u,(;k=state))
        @test draw.k == state
        state[1] += 1
        @test draw.k[1] == state[1]-1
        state[1] -= 1
        @test restore_draws(built.layout,reshape(u,1,1);discrete=(;k=state)).k == [state]
        @test eltype(restore_draws(built.layout,zeros(1,0)).k) === Matrix{Int}
    end
end

@testset "finite families share the same marginal density contract" begin
    for (rhs, data, support, law) in (
            (:(Bernoulli(p)), (; p=0.3), 0:1, Bernoulli(0.3)),
            (:(Categorical(p)), (; p=[0.2,0.3,0.5]), 1:3, Categorical([0.2,0.3,0.5])),
            (:(DiscreteUniform(lo,hi)), (; lo=-2,hi=3), -2:3, DiscreteUniform(-2,3)))
        model = RKPPLModel(quote
            mu ~ Normal(0,1)
            k ~ $rhs
            y .~ Normal.(mu+k,1)
        end, @__MODULE__)
        y,u = [0.2,0.7],[0.3]
        plan = marginalize(model,:k)(;data...) | (;y)
        built = build_kernel(plan)
        terms = [logpdf(law,k)+sum(logpdf.(Normal(only(u)+k,1),y)) for k in support]
        mass = ReactiveKernelsPPL.logsumexp(terms)
        weights = exp.(terms.-mass)
        sampler = prepare_sampler(built,plan,u;
            backend=AutoEnzyme(;mode=Enzyme.Reverse),nonallocating=false)
        value,gradient = sampler_value_and_gradient!(sampler,zeros(1),u)
        @test value ≈ logpdf(Normal(),only(u))+mass
        @test only(gradient) ≈ -only(u)+sum(y)-length(y)*(only(u)+sum(support.*weights))
        @test vec(prepare_discrete_latents(built,plan)(u).parameters.probabilities) ≈ weights
    end
end

@testset "marginal predictive joint states" begin
    model = @rkppl begin
        p ~ Beta(2,3)
        k ~ Bernoulli(p)
        y .~ Normal.(k, 0.1)
        z .~ Normal.(2k, 0.1)
    end
    plan = marginalize(model, :k)() | (; y=[0.5,0.5], z=[1.0])
    built = build_kernel(plan)
    u = [0.2]
    q = prepare_predictive(built, plan)
    latent = prepare_discrete_latents(built, plan)(u)
    weights = vec(latent.parameters.probabilities)
    laws = predictive_laws(q, u)
    @test mean.(laws.y) ≈ fill(weights[2], 2)
    @test mean.(laws.z) ≈ [2weights[2]]
    for seed in 1:5
        rng = Xoshiro(seed)
        k = rand(rng) < weights[1] ? 0 : 1
        expected = (; y=rand(rng, Normal(k,0.1),2), z=rand(rng,Normal(2k,0.1),1))
        @test q(u, Xoshiro(seed)) == expected
        @test predictive_draws!(deepcopy(expected), q, u, Xoshiro(seed)) == expected
    end
    ports = ReactiveKernelsPPL._query_columns(plan)
    data = Tuple(plan.columns[k] for k in ports)
    cut = Base.invokelatest(ReactiveKernels.prepare, built.spec;
        have=(:unconstrained, ports...), want=predictive_node(built),
        exemplars=(u,data...), type_stability=:error)
    batch = Base.invokelatest(ReactiveKernels.vectorize, cut; batched=:unconstrained)
    values = Base.invokelatest(batch, hcat(u,u), data...)
    @test predictive_draws(values, q.targets, Xoshiro(9); position=2) == q(u,Xoshiro(9))
end

@testset "marginal ragged pointwise terms" begin
    model = @rkppl begin
        p ~ Beta(2,3)
        k ~ Bernoulli(p)
        @plate for i in eachindex(x)
            y[i] .~ Normal.(x[i] .+ k, 1)
        end
    end
    x = [[0.1,0.2],Float64[],[0.3]]
    y = [[0.3,0.7],Float64[],[0.2]]
    plan = marginalize(model,:k)(; x) | (;y)
    built = build_kernel(plan)
    u = [0.2]
    ports = ReactiveKernelsPPL._query_columns(plan)
    data = Tuple(plan.columns[k] for k in ports)
    pointwise = Base.invokelatest(ReactiveKernels.prepare,built.spec;
        have=(:unconstrained,ports...),want=:pointwise,
        exemplars=(u,data...),type_stability=:error)
    pw = Base.invokelatest(pointwise,u,data...).y
    @test length.(pw) == [2,0,1]
    @test sum(sum,pw) ≈ prepare_query(built,plan,:likelihood)(u)
    q = prepare_predictive(built,plan)
    @test length.(q(u,Xoshiro(4)).y) == [2,0,1]
    sampler=prepare_sampler(built,plan,u;
        backend=AutoEnzyme(;mode=Enzyme.Reverse),nonallocating=false)
    _,gradient=sampler_value_and_gradient!(sampler,zeros(1),u)
    weights=prepare_discrete_latents(built,plan)(u).parameters.probabilities
    @test only(gradient) ≈ 2+weights[2]-6/(1+exp(-only(u)))
    for empty_x in (Vector{Float64}[], [Float64[],Float64[]])
        empty_plan = marginalize(model,:k)(;x=empty_x) | (;y=deepcopy(empty_x))
        empty_data = Tuple(empty_plan.columns[k] for k in ports)
        empty_cut = Base.invokelatest(ReactiveKernels.prepare,built.spec;
            have=(:unconstrained,ports...),want=:pointwise,
            exemplars=(u,empty_data...),type_stability=:error)
        empty_pw = Base.invokelatest(empty_cut,u,empty_data...).y
        @test length.(empty_pw) == length.(empty_x)
        @test sum(sum,empty_pw;init=0.0) ≈
            prepare_query(built,empty_plan,:likelihood)(u) atol=1e-12
    end
    # The output storage follows the pointwise tree even when an inner or
    # outer extent is empty, without asking map to infer recursive closures.
    nested = [[[Float32[0.1,0.2],Float32[]]],[[Float32[0.3,0.4],Float32[]]]]
    logweights = log.([0.3,0.7])
    terms = @inferred ReactiveKernelsPPL._integer_pointwise_value(nested,copy(logweights))
    @test terms isa Vector{Vector{Vector{Float64}}}
    @test sum(sum,only(terms)) ≈ ReactiveKernelsPPL.logsumexp(
        logweights .+ [sum(v -> sum(Float64,v;init=0.0),only(state)) for state in nested])
end

@testset "mixture latents conditional on integer state" begin
    model = @rkppl begin
        k ~ Bernoulli(p)
        y .~ MixtureModel.(vcat.(Normal.(k, 1.0), Normal.(3.0, 1.0)), Ref([0.4,0.6]))
    end
    plan = marginalize(model,:k)(;p=0.4) | (;y=[0.2,1.1])
    built = build_kernel(plan)
    q = prepare_discrete_latents(built,plan)
    value = q(Float64[])
    @test size(value._conditional.y.probabilities) == (2,2,2)
    for state in 1:2
        expected = [w * pdf(Normal(mu,1), yy) for yy in [0.2,1.1],
            (w,mu) in [(0.4,state-1),(0.6,3.0)]]
        @test value._conditional.y.probabilities[:,:,state] ≈ expected ./ sum(expected;dims=2)
    end
    draw = discrete_latent_draws(q,Float64[],Xoshiro(12))
    @test draw.parameters.k in 0:1
    @test length(draw.observations.y)==2
end

@testset "empty restricted integer support" begin
    model = @rkppl begin
        k ~ restricted(Bernoulli(p),lo,hi)
        y .~ Normal.(k,1)
    end
    plan = marginalize(model,:k)(;p=0.4,lo=2,hi=3) | (;y=[0.2])
    built = build_kernel(plan)
    @test prepare_query(built,plan,:sampler)(Float64[]) == -Inf
    @test_throws ArgumentError prepare_query(built,plan,:pointwise)(Float64[])
    @test_throws ArgumentError prepare_predictive(built,plan)(Float64[],Xoshiro(1))
    @test_throws ArgumentError discrete_latent_draws(prepare_discrete_latents(built,plan),Float64[],Xoshiro(1))
end

@testset "integer bounds come from data" begin
    for declaration in (:(Binomial(n,p)), :(DiscreteUniform(0,n)),
            :(restricted(Bernoulli(p),0,n)))
        model = RKPPLModel(quote
            n ~ Normal(3,1)
            k ~ $declaration
        end, @__MODULE__)
        @test_throws ContractValidationError bind_data(model(;p=0.4))
    end
end

@testset "partially eliminated integer state" begin
    model = @rkppl begin
        p ~ Beta(2,3)
        k ~ Bernoulli(p)
        z ~ Bernoulli(p)
        y .~ Normal.(k+z,1)
    end
    plan = marginalize(model,:k)() | (;y=[0.4,0.7])
    built = build_kernel(plan)
    u,state=[0.3],(;z=1)
    p=1/(1+exp(-only(u)))
    terms=[logpdf(Bernoulli(p),k)+sum(logpdf.(Normal(k+1,1),[0.4,0.7])) for k in 0:1]
    mass=ReactiveKernelsPPL.logsumexp(terms)
    reference=logpdf(Beta(2,3),p)+log(p)+mass+log(p)+log1p(-p)
    @test prepare_query(built,plan,:sampler)(u,state) ≈ reference
    weights=exp.(terms.-mass)
    laws=predictive_laws(prepare_predictive(built,plan),u,state)
    @test mean.(laws.y) ≈ fill(1+weights[2],2)
end

@testset "conditional values after integer elimination" begin
    model = @rkppl begin
        mu ~ Normal(0,1)
        k ~ Bernoulli(p)
        location = mu+k
        independent = 2mu
        y .~ Normal.(location,1)
    end
    plan = marginalize(model,:k)(;p=0.4) | (;y=[0.2,0.7])
    built = build_kernel(plan)
    @test prepare_query(built,plan,(:independent,))([0.3]) ≈ 0.6
    for want in ((:location,), (:k,), :observations)
        error = try
            prepare_query(built,plan,want)
            nothing
        catch e
            e
        end
        @test error isa ContractValidationError
        @test occursin("eliminated integer",sprint(showerror,error))
    end
    @test_throws ContractValidationError jax_source(built,plan;values=(:location,))
end
end
