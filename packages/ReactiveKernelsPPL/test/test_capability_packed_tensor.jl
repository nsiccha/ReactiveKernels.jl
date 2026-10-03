using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test

# Well-formed full family declarations, with their observed rows packed by
# the hand-authored mi_jobs contract. Jobs addresses the full predictor axis.
function _cap_packed_tensor(kind, n; selected_range=nothing)
    jobs = collect(1:2:n)
    x1 = collect(range(-0.8, 0.9; length=n))
    x2 = 0.4 .+ 0.2 .* x1
    data = Dict{Symbol,Any}(:Jobs => jobs, :x1 => x1, :x2 => x2)
    ast = if kind === :multinomial
        data[:N] = [5 + mod(i, 3) for i in 1:n]
        data[:c1] = ones(Int, length(jobs))
        data[:c2] = [1 + mod(i, 2) for i in jobs]
        data[:c3] = data[:N][jobs] .- data[:c1] .- data[:c2]
        quote
            p ~ Dirichlet([1.2, 2.0, 1.4])
            eachrow(hcat(c1, c2, c3)) .~ Multinomial.(N, Ref(p))
        end
    elseif kind === :joint
        data[:y1] = 0.2 .+ 0.3 .* x1[jobs]
        data[:y2] = -0.1 .+ 0.1 .* x2[jobs]
        quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu1 = a .+ b .* x1
            mu2 = b .+ a .* x2
            L ~ LKJCovarianceFactor(2, Exponential(1), 2)
            [y1, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end
    elseif kind in (:glm, :bernoulli_glm, :poisson_glm)
        data[:y] = kind === :glm ? 0.2 .+ 0.3 .* x1[jobs] :
            kind === :bernoulli_glm ? Int.(x1[jobs] .> 0) : Int.(x1[jobs] .> 0) .+ 1
        obj = kind === :glm ? :(NormalIDGLM(X, a, b, 0.7)) :
            kind === :bernoulli_glm ? :(BernoulliLogitGLM(X, a, b)) : :(PoissonLogGLM(X, a, b))
        quote
            X = hcat(x1, x2)
            a ~ Normal(0, 1)
            b[axes(X, 2)] .~ Normal.(0, 2)
            y ~ $obj
        end
    else
        error("unknown packed tensor fixture $kind")
    end
    plan = lower_rkppl(ast, keys(data); conditioned=keys(data))
    r = ReactiveKernelsPPL._with(only(plan.responses); mi_jobs=:Jobs, range=selected_range)
    plan = ReactiveKernelsPPL._with(plan; responses=[r])
    saved = deepcopy(data)
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = [0.15sin(i + 0.2) for i in 1:built.layout.total]
    packed_positions = selected_range === nothing ? eachindex(jobs) :
        findall(j -> j in selected_range, jobs)
    observed_jobs = jobs[packed_positions]
    function pointwise(v)
        p = constrain(built.layout, v)
        if kind === :multinomial
            [logpdf(Multinomial(data[:N][j], p.p),
                [data[:c1][k], data[:c2][k], data[:c3][k]])
                for (j, k) in zip(observed_jobs, packed_positions)]
        elseif kind === :joint
            L = Diagonal(p.L_scales) * p.L_L_corr
            covariance = L * L'
            [logpdf(MvNormal([p.a + p.b*x1[j], p.b + p.a*x2[j]], covariance),
                [data[:y1][k], data[:y2][k]])
                for (j, k) in zip(observed_jobs, packed_positions)]
        else
            family(eta) = kind === :glm ? Normal(eta, 0.7) :
                kind === :bernoulli_glm ? Bernoulli(1/(1+exp(-eta))) : Poisson(exp(eta))
            [logpdf(family(p.a + p.b[1]*x1[j] + p.b[2]*x2[j]), data[:y][k])
                for (j, k) in zip(observed_jobs, packed_positions)]
        end
    end
    function oracle(v)
        p = constrain(built.layout, v)
        prior = if kind === :multinomial
            logpdf(Dirichlet([1.2, 2.0, 1.4]), p.p)
        elseif kind === :joint
            logpdf(Normal(), p.a) + logpdf(Normal(), p.b) +
                sum(logpdf.(Exponential(), p.L_scales)) +
                logpdf(LKJCholesky(2, 2), Cholesky(LowerTriangular(p.L_L_corr)))
        else
            logpdf(Normal(), p.a) + sum(logpdf.(Normal(0, 2), p.b))
        end
        sum(pointwise(v)) + prior + logjac(built.layout, v)
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    (; kind, plan, data, saved, bound, built, u, sampler, oracle, pointwise, observed_jobs)
end

# Query the likelihood with constrained probabilities as its active port.
# This isolates the zero-count convention from a Dirichlet prior at the
# boundary and from numerical saturation of the simplex transform.
function _cap_packed_zero_counts(n)
    fx = _cap_packed_tensor(:multinomial, n)
    data = deepcopy(fx.data)
    data[:c1] = data[:N][data[:Jobs]]
    data[:c2] = zeros(Int, length(data[:Jobs]))
    data[:c3] = zeros(Int, length(data[:Jobs]))
    bound = bind_data(fx.plan, data)
    built = build_kernel(bound)
    names = sort!(collect(keys(data)))
    columns = NamedTuple{Tuple(names)}(Tuple(data[name] for name in names))
    kernel = Base.invokelatest(prepare, built.spec; have=(:p, names...), want=:likelihood, bound=columns)
    p = [1.0, 0.0, 0.0]
    ad = Base.invokelatest(prepare_ad, kernel, AutoEnzyme(; mode=Enzyme.Reverse), p; active=:p)
    (; data, saved=deepcopy(data), built, kernel, p, ad, expected=[Float64(sum(data[:c1])), 0.0, 0.0])
end

@testset "zero Multinomial counts skip inactive logarithms and derivatives" begin
    for n in (6, 18)
        fx = _cap_packed_zero_counts(n)
        value, gradient = ReactiveKernels.ad_value_and_gradient!(fx.ad, similar(fx.p), fx.p)
        @test value == 0.0
        @test gradient ≈ fx.expected
        @test fx.data == fx.saved
    end
end

@testset "packed tensor observations preserve outcome columns and full predictor rows" begin
    for kind in (:multinomial, :joint, :glm, :bernoulli_glm, :poisson_glm), n in (6, 18)
        fx = _cap_packed_tensor(kind, n)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _cap_range_fd(fx.oracle, fx.u) rtol=1e-5 atol=1e-7
        pw = only(values(Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u)))
        @test pw ≈ fx.pointwise(fx.u)
        @test length(pw) == length(fx.observed_jobs)
        @test fx.data == fx.saved
        changed = fx.u .+ 0.05
        @test first(sampler_value_and_gradient!(fx.sampler, similar(changed), changed)) ≈ fx.oracle(changed)
    end
end

@testset "packed tensor range and outcome alignment" begin
    for kind in (:multinomial, :joint)
        fx = _cap_packed_tensor(kind, 6; selected_range=2:6)
        @test first(sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)) ≈ fx.oracle(fx.u)
        @test only(values(Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u))) ≈ fx.pointwise(fx.u)
        second = kind === :multinomial ? :c2 : :y2
        bad = merge(fx.data, Dict(second => fx.data[second][1:end-1]))
        @test_throws ContractValidationError bind_data(fx.plan, bad)
    end
end
