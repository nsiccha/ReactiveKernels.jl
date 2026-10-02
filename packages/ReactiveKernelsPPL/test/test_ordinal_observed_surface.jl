using DifferentiationInterface, Distributions, Enzyme
using ReactiveKernels, ReactiveKernelsPPL, Test

# These models use only authored declarations and ordinary value operations.
# The independent oracle uses Distributions CDFs, not the generated graph.
function _oos_ordinal(; structure = :stopping, disc = :modeled,
        effects = true, bound_effects = false, K = 3, n = 7)
    data = Dict{Symbol,Any}(:y => [mod1(i, K) for i in 1:n],
        :x => [0.2 * sin(i) for i in 1:n],
        :d => [1.1 + 0.1 * cos(i) for i in 1:n],
        :X => [0.1 * sin(i + j) for i in 1:n, j in 1:2])
    ast = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        eta = a .+ b .* x
    end
    if structure === :cumulative
        push!(ast.args, :(c ~ Ordered(Normal(0, 1), length(levels(y)) - 1)))
    else
        push!(ast.args, :(c[1:length(levels(y)) - 1] .~ Normal.(0, 1)))
    end
    d = if disc in (:modeled, :direct)
        push!(ast.args, :(gamma ~ Normal(0, 1)))
        disc === :modeled ? :(exp.(gamma .* x)) : :gamma
    elseif disc === :scalar
        push!(ast.args, :(dpar ~ Exponential(1)))
        :dpar
    elseif disc === :data
        :d
    else
        1.7
    end
    tag = Expr(:call, structure === :cumulative ? :Cumulative : :StoppingRatio)
    if effects
        if bound_effects
            data[:E] = [0.1 * cos(i + j) for i in 1:n, j in 1:K-1]
        else
            push!(ast.args,
                :(delta[axes(X, 2), 1:length(levels(y)) - 1] .~ Normal.(0, 1)),
                :(E = X * delta))
        end
        push!(ast.args, :(y .~ Ordinal.($tag, LogitLink(), eta, Ref(c), $d, eachrow(E))))
    else
        push!(ast.args, :(y .~ Ordinal.($tag, LogitLink(), eta, Ref(c), $d)))
    end
    bound = bind_data(lower_rkppl(ast, keys(data)), data)
    built = build_kernel(bound)
    u = disc === :direct ? fill(0.1, built.layout.total) :
        [0.2 * cos(i) for i in 1:built.layout.total]
    return (; bound, built, data, u, structure, disc, effects, bound_effects)
end

function _oos_ordinal_reference(fx, u)
    th = constrain(fx.built.layout, u)
    eta = th.a .+ th.b .* fx.data[:x]
    d = fx.disc === :modeled ? exp.(th.gamma .* fx.data[:x]) :
        fx.disc === :direct ? fill(th.gamma, length(eta)) :
        fx.disc === :scalar ? fill(th.dpar, length(eta)) :
        fx.disc === :data ? fx.data[:d] : fill(1.7, length(eta))
    E = !fx.effects ? zeros(length(eta), length(th.c)) :
        fx.bound_effects ? fx.data[:E] : fx.data[:X] * th.delta
    likelihood = 0.0
    for i in eachindex(eta)
        y, K = fx.data[:y][i], length(th.c) + 1
        F(j) = cdf(Logistic(), d[i] * (th.c[j] - eta[i] - E[i, j]))
        if fx.structure === :cumulative
            likelihood += log((y == K ? 1.0 : F(y)) - (y == 1 ? 0.0 : F(y-1)))
        else
            likelihood += sum((log1p(-F(j)) for j in 1:y-1); init = 0.0)
            y == K || (likelihood += log(F(y)))
        end
    end
    prior = logpdf(Normal(), th.a) + logpdf(Normal(), th.b) +
        sum(logpdf.(Normal(), th.c))
    fx.disc in (:modeled, :direct) && (prior += logpdf(Normal(), th.gamma))
    fx.disc === :scalar && (prior += logpdf(Exponential(), th.dpar))
    fx.effects && !fx.bound_effects && (prior += sum(logpdf.(Normal(), th.delta)))
    return likelihood + prior + logjac(fx.built.layout, u)
end

function _oos_observed(; n = 4, p = 2)
    m = 2n
    data = Dict{Symbol,Any}(
        :y => Union{Missing,Float64}[isodd(i) ? 0.1 * i : missing for i in 1:m],
        :rows => [isodd(i) ? m - 1 : 1 for i in 1:n],
        :B => [0.1 * cos(i + j) for i in 1:m, j in 1:p])
    ast = quote
        a ~ Normal(0, 1)
        sigma ~ Exponential(1)
        w[axes(B, 2)] .~ Normal.(0, 1)
        mu = a .+ B * w
        y[rows] .~ Normal.(mu[rows], sigma)
    end
    bound = bind_data(lower_rkppl(ast, keys(data)), data)
    built = build_kernel(bound)
    u = [0.2 * cos(i) for i in 1:built.layout.total]
    return (; bound, built, data, u)
end

function _oos_observed_reference(fx, u)
    th = constrain(fx.built.layout, u)
    mu = th.a .+ fx.data[:B] * th.w
    rows = fx.data[:rows]
    return sum(logpdf.(Normal.(mu[rows], th.sigma), fx.data[:y][rows])) +
        logpdf(Normal(), th.a) + logpdf(Exponential(), th.sigma) +
        sum(logpdf.(Normal(), th.w)) + logjac(fx.built.layout, u)
end

function _oos_findiff(f, u; h = cbrt(eps(Float64)))
    map(eachindex(u)) do i
        up, dn = copy(u), copy(u)
        up[i] += h
        dn[i] -= h
        (f(up) - f(dn)) / (2h)
    end
end

function _oos_native(fx, reference)
    before = deepcopy(fx.data)
    kern = prepare_query(fx.built, fx.bound, :sampler)
    @test Base.invokelatest(kern, fx.u) ≈ reference(fx, fx.u) rtol = 1e-12
    q = prepare_sampler(fx.built, fx.bound, fx.u;
        backend = AutoEnzyme(; mode = Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(q, similar(fx.u), fx.u)
    @test value ≈ reference(fx, fx.u) rtol = 1e-12
    @test grad ≈ _oos_findiff(u -> reference(fx, u), fx.u) rtol = 1e-5 atol = 1e-7
    @test isequal(fx.data, before)
end

@testset "ordinal broadcast values: density and native gradients" begin
    for structure in (:cumulative, :stopping), disc in (:literal, :data, :scalar, :modeled)
        _oos_native(_oos_ordinal(; structure, disc, effects = false), _oos_ordinal_reference)
    end
    for bound_effects in (false, true)
        _oos_native(_oos_ordinal(; bound_effects), _oos_ordinal_reference)
    end
    for structure in (:cumulative, :stopping)
        fx = _oos_ordinal(; structure, disc = :direct, effects = false)
        _oos_native(fx, _oos_ordinal_reference)
        q = prepare_sampler(fx.built, fx.bound, fx.u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        bad = -fx.u
        value, grad = sampler_value_and_gradient!(q, similar(bad), bad)
        @test value == -Inf
        @test all(isfinite, grad)
    end
end

@testset "indexed observations and sized innovations" begin
    fx = _oos_observed()
    @test fx.bound.n_obs == length(fx.data[:rows])
    @test only(fx.bound.responses).mi_jobs === nothing
    @test only(fx.bound.array_parameters).name === :w
    _oos_native(fx, _oos_observed_reference)
    # The rows are an ordinary Julia gather: neither sorted nor deduplicated.
    @test fx.data[:rows] == [7, 1, 7, 1]
end
