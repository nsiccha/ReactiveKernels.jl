using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernelsPPL, Test

function _missing_fixture(y; x=collect(range(-0.4, 0.6; length=size(y, 1))), plate=false, computed=false)
    data = (; y, x)
    ast = computed ? quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        @plate for i in eachindex(y)
            y[i] ~ Normal(mu[i] + x[i], 0.7)
        end
    end : plate ? quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a + b * x[i], 0.7)
        end
    end : quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ Normal.(mu, 0.7)
    end
    plan = lower_rkppl(ast, data; conditioned=keys(data))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a=0.2, b=-0.3))
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    function oracle(v)
        p = constrain(built.layout, v)
        pw = broadcast(y, x) do yy, xx
            ismissing(yy) ? 0.0 : logpdf(Normal(p.a + p.b * xx + (computed ? xx : 0.0), 0.7), yy)
        end
        logpdf(Normal(), p.a) + logpdf(Normal(), p.b) + sum(pw; init=0.0)
    end
    return (; data, plan, bound, built, u, sampler, oracle)
end

function _missing_family_fixture(kind; n=4)
    response = kind in (:binomial, :bernoulli, :stopping) ?
        Union{Missing,Int}[1, missing, kind === :stopping ? 3 : 0, 1] :
        Union{Missing,Float64}[0.2, missing, 0.4, 0.7]
    indices = mod1.(1:n, 4)
    y = response[indices]
    data = (; y, x=[0.1, 0.2, -0.3, 0.4][indices], sigma=[0.7, -1.0, 0.7, 0.7][indices])
    ast = quote
        a ~ Normal(0, 1)
        eta = a .+ 0.2 .* x
    end
    response = if kind === :binomial
        :(y .~ Binomial.(2, logistic.(eta)))
    elseif kind === :bernoulli
        :(y .~ Bernoulli.(logistic.(eta)))
    elseif kind === :beta
        :(y .~ Beta.(exp.(eta), 2.0))
    elseif kind === :stopping
        push!(ast.args, :(c[1:2] .~ Normal.(0, 1)))
        :(y .~ Ordinal.(StoppingRatio(), LogitLink(), eta, Ref(c), 1.0))
    else
        :(y .~ Normal.(eta, sigma))
    end
    push!(ast.args, response)
    bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = kind === :stopping ? unconstrain(built.layout, (; a=0.2, c=[-0.4, 0.5])) : [0.2]
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    function oracle(v)
        p = constrain(built.layout, v)
        eta = p.a .+ 0.2 .* data.x
        out = 0.0
        for i in eachindex(y)
            ismissing(y[i]) && continue
            out += if kind === :binomial
                logpdf(Binomial(2, 1/(1 + exp(-eta[i]))), y[i])
            elseif kind === :bernoulli
                logpdf(Bernoulli(1/(1 + exp(-eta[i]))), y[i])
            elseif kind === :beta
                logpdf(Beta(exp(eta[i]), 2.0), y[i])
            elseif kind === :stopping
                sum((logccdf(Logistic(), p.c[j] - eta[i]) for j in 1:y[i]-1); init=0.0) +
                    (y[i] == 3 ? 0.0 : logcdf(Logistic(), p.c[y[i]] - eta[i]))
            else
                logpdf(Normal(eta[i], data.sigma[i]), y[i])
            end
        end
        out + logpdf(Normal(), p.a) + (kind === :stopping ? sum(logpdf.(Normal(), p.c)) : 0.0)
    end
    return (; kind, data, bound, built, u, sampler, oracle)
end

function _missing_fd(f, u)
    h = cbrt(eps(Float64))
    [(f(u + h*e) - f(u - h*e))/(2h) for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

@testset "missing GLM rows guard density values and ordinary reverse" begin
    for head in (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM)
        y = head === :NormalIDGLM ? Union{Missing,Float64}[0.2, missing, 0.5, missing] :
            Union{Missing,Int}[1, missing, 0, missing]
        # Poisson's inactive rates overflow. A zero selected output must also
        # have zero derivative, rather than multiplying an infinite tape by 0.
        data = (; y, x1=[0.1, 10000.0, -0.3, 10000.0], x2=[0.2, 0.2, 0.4, -0.2])
        saved = deepcopy(data)
        rhs = Expr(:call, head, :X, :alpha, :beta)
        head === :NormalIDGLM && push!(rhs.args, 0.7)
        ast = quote
            X = hcat(x1, x2)
            alpha ~ Normal(0, 1)
            beta[axes(X, 2)] .~ Normal.(0, 1)
            y ~ $rhs
        end
        bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
        built = build_kernel(bound)
        u = unconstrain(built.layout, (; alpha=0.2, beta=[0.3, -0.1]))
        sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        function oracle(v)
            p = constrain(built.layout, v)
            eta = p.alpha .+ hcat(data.x1, data.x2) * p.beta
            out = logpdf(Normal(), p.alpha) + sum(logpdf.(Normal(), p.beta))
            for i in eachindex(y)
                ismissing(y[i]) && continue
                family = head === :NormalIDGLM ? Normal(eta[i], 0.7) :
                    head === :BernoulliLogitGLM ? Bernoulli(1/(1 + exp(-eta[i]))) : Poisson(exp(eta[i]))
                out += logpdf(family, y[i])
            end
            out
        end
        for v in (u, u .+ [0.1, -0.05, 0.03])
            value, gradient = sampler_value_and_gradient!(sampler, similar(v), v)
            @test value ≈ oracle(v)
            @test gradient ≈ _missing_fd(oracle, v) rtol=1e-5 atol=1e-8
            pw = Base.invokelatest(prepare_query(built, bound, :pointwise), v).y
            @test length(pw) == 4
            @test all(iszero, pw[[2, 4]])
        end
        @test bound.n_obs == 4
        @test isequal(data, saved)
    end
end

@testset "condition refreshes observation presence without changing caller data" begin
    fx = _missing_fixture(Union{Missing,Float64}[0.1, missing, 0.4])
    saved = deepcopy(fx.data)
    original_pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y
    for replacement in ((; y=[0.2, 0.3, 0.5]),
            (; y=Union{Missing,Float64}[missing, 0.3, 0.5]),
            (; y=fill(missing, 3)), (; x=[-0.2, 0.5, 0.8]))
        data = merge(fx.data, replacement)
        bound = condition(fx.bound; replacement...)
        built = build_kernel(bound)
        sampler = prepare_sampler(built, bound, fx.u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        pointwise(v) = let p=constrain(built.layout, v)
            [ismissing(y) ? 0.0 : logpdf(Normal(p.a + p.b*x, 0.7), y)
                for (y, x) in zip(data.y, data.x)]
        end
        oracle(v) = let p=constrain(built.layout, v)
            logpdf(Normal(), p.a) + logpdf(Normal(), p.b) + sum(pointwise(v))
        end
        value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
        @test value ≈ oracle(fx.u)
        @test gradient ≈ _missing_fd(oracle, fx.u) rtol=1e-5 atol=1e-8
        @test Base.invokelatest(prepare_query(built, bound, :pointwise), fx.u).y ≈ pointwise(fx.u)
        @test bound.n_obs == 3
        @test isequal(fx.data, saved)
    end
    @test Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y == original_pw
end

@testset "a missing plate cell skips its invalid local arithmetic" begin
    for scale in (false, true)
    data = (; y=Union{Missing,Float64}[0.1, missing, 0.4], x=[0.2, -1.0, 0.5])
    ast = scale ? quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] ~ Normal(a + b*x[i], sqrt(x[i]))
        end
    end : quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        @plate for i in eachindex(y)
            mu = a + b * sqrt(x[i])
            y[i] ~ Normal(mu, 0.7)
        end
    end
    saved = deepcopy(data)
    bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a=0.2, b=-0.3))
    oracle(v) = logpdf(Normal(), v[1]) + logpdf(Normal(), v[2]) +
        sum(logpdf(Normal(v[1] + v[2]*(scale ? data.x[i] : sqrt(data.x[i])),
            scale ? sqrt(data.x[i]) : 0.7), data.y[i]) for i in (1, 3))
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u)
    @test gradient ≈ _missing_fd(oracle, u) rtol=1e-5 atol=1e-8
    @test Base.invokelatest(prepare_query(built, bound, :pointwise), u).y[2] == 0.0
    @test isequal(data, saved)
    end
end

@testset "presence guards use ordinary response family code" begin
    for kind in (:binomial, :bernoulli, :beta, :stopping, :invalid_missing_scale)
        fx = _missing_family_fixture(kind)
        saved = deepcopy(fx.data)
        value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
        @test value ≈ fx.oracle(fx.u)
        @test gradient ≈ _missing_fd(fx.oracle, fx.u) rtol=1e-5 atol=1e-8
        pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), fx.u).y
        @test length(pw) == 4
        @test pw[2] == 0.0
        @test isequal(fx.data, saved)
    end
end

@testset "whole responses skip missing observations automatically" begin
    for y in (Union{Missing,Float64}[0.1, missing, 0.4],
            Union{Missing,Float64}[missing, missing, missing],
            fill(missing, 3), Union{Missing,Float64}[0.1, 0.2, 0.4],
            Union{Missing,Float64}[],
            Union{Missing,Float64}[0.1 missing; missing 0.2; 0.4 missing]),
            plate in (false, true)
        plate && ndims(y) != 1 && continue
        before = copy(y)
        fx = _missing_fixture(y; plate)
        @test all(v -> v isa Number || isconcretetype(eltype(v)), values(fx.bound.columns))
        @test all(v -> v isa Number || !any(ismissing, v), values(fx.bound.columns))
        @test fx.bound.n_obs == length(y)
        @test length(fx.u) == 2
        for u in (fx.u, fx.u .+ [0.15, -0.2])
            value, gradient = sampler_value_and_gradient!(fx.sampler, similar(u), u)
            @test value ≈ fx.oracle(u)
            @test gradient ≈ _missing_fd(fx.oracle, u) rtol=5e-6 atol=1e-8
            pw = Base.invokelatest(prepare_query(fx.built, fx.bound, :pointwise), u).y
            @test size(pw) == size(y)
            @test all(iszero, pw[ismissing.(y)])
        end
        @test isequal(y, before)
    end
end

@testset "missing response skipping retains full computed values and latent declarations" begin
    data = (; y=Union{Missing,Float64}[0.1, missing, 0.4, missing],
        x=[0.1, -0.2, 0.3, 0.4], g=[1, 2, 1, 2], rows=[4, 2, 3, 1])
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        r[levels(g)] .~ Normal.(0, 1)
        mu = a .+ r[g]
        @plate for i in 1:4
            y[i] ~ Normal(mu[i] + b * x[i], 0.7)
        end
    end
    bound = bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; a=0.2, b=-0.1, r=[0.1, -0.2]))
    oracle(v) = let p=constrain(built.layout, v)
        logpdf(Normal(), p.a) + logpdf(Normal(), p.b) + sum(logpdf.(Normal(), p.r)) +
            sum(logpdf(Normal(p.a + p.r[data.g[i]] + p.b*data.x[i], 0.7), data.y[i])
                for i in (1, 3))
    end
    sampler = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
    @test value ≈ oracle(u)
    @test gradient ≈ _missing_fd(oracle, u) rtol=1e-5
    @test length(constrain(built.layout, u).r) == 2
    @test bound.n_obs == 4
    # A full permutation remains an ordinary authored gather; it is not a
    # hand-selected missing-data subset.
    gathered = quote
        a ~ Normal(0, 1)
        y[rows] .~ Normal.(a, 0.7)
    end
    gb = bind_data(lower_rkppl(gathered, data; conditioned=keys(data)), data)
    gk = build_kernel(gb)
    pw = Base.invokelatest(prepare_query(gk, gb, :pointwise), [0.2])
    @test only(values(pw)) ≈ [0.0, 0.0, logpdf(Normal(0.2, 0.7), 0.4), logpdf(Normal(0.2, 0.7), 0.1)]
    @test all(v -> v isa Number || isconcretetype(eltype(v)), values(gb.columns))
end

@testset "missing entries do not authorize partial observation" begin
    for y in ([0.1, 0.2, 0.3], Union{Missing,Float64}[missing, 0.2, 0.3], fill(missing, 3))
        for range in (:(2:3), :(1:0))
            ast = quote
                a ~ Normal(0, 1)
                @plate for i in $range
                    y[i] ~ Normal(a, 0.7)
                end
            end
            data = (; y)
            plan = lower_rkppl(ast, data; conditioned=keys(data))
            @test_throws ContractValidationError bind_data(plan, data)
        end
        data = (; y, rows=[2, 3])
        ast = quote
            a ~ Normal(0, 1)
            y[rows] .~ Normal.(a, 0.7)
        end
        @test_throws ContractValidationError bind_data(lower_rkppl(ast, data; conditioned=keys(data)), data)
    end
end
