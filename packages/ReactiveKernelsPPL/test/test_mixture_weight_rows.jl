module MixtureWeightRowsTests
using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
import Distributions as D
include(joinpath(pkgdir(ReactiveKernels), "test", "type_stability_audit.jl"))

const BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

function differences(f, u)
    h = cbrt(eps(Float64))
    return [begin
        up, down = copy(u), copy(u)
        up[i] += h
        down[i] -= h
        (f(up) - f(down)) / (2h)
    end for i in eachindex(u)]
end

function check(source, data, position, oracle; replay = true, audit = false)
    saved = deepcopy(data)
    plan = lower_rkppl(source, keys(data); mod = @__MODULE__, conditioned = (:y,))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, position)
    query = prepare_sampler(built, bound, u; backend = BACKEND)
    target(v) = oracle(constrain(built.layout, v), data) + logjac(built.layout, v)
    value, gradient = sampler_value_and_gradient!(query, similar(u), u)
    @test value ≈ target(u) rtol = 1e-11 atol = 1e-11
    @test gradient ≈ differences(target, u) rtol = 1e-5 atol = 1e-7
    @test all(isfinite, gradient)
    if audit
        @test isempty(TypeStabilityAudit.findings(query.kernel, (Vector{Float64},)))
        @test isempty(TypeStabilityAudit.ad_call_findings(query.ad, u))
    end
    pw = prepare_query(built, bound, :pointwise)(u)
    @test sum(pw.y) ≈ prepare_query(built, bound, :likelihood)(u)
    if replay
        # Parse the ordinary printer's complete program, with only line
        # numbers removed, so hidden captured state cannot make this pass.
        program = kernel_expr(bound, built.layout)
        Base.remove_linenums!(program)
        printed = sprint(show, program)
        parsed = Meta.parse(printed)
        parsed = parsed.head === :quote ? parsed.args[1] : parsed
        spec = ReactiveKernelsPPL._eval_kernel_def(parsed)
        restored = prepare_query((; spec, layout = built.layout), bound, :sampler)
        @test restored(u) ≈ target(u)
    end
    @test data == saved
    return (; bound, built, u, value, gradient, pointwise = pw.y)
end

sigmoid(x) = 1 / (1 + exp(-x))
normal_prior(x) = D.logpdf(D.Normal(), x)

@testset "predictor-dependent mixture weights" begin
    function source(named)
        ex = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            sigma ~ Exponential(1)
            tau ~ Exponential(1)
            z[1:2] .~ Normal.(0, 1)
            h[1:2] .~ Normal.(0, 1)
            group_scale ~ Exponential(1)
            mu = a .+ b .* x .+ group_scale .* z[g]
            w = logistic.(b .+ h[g])
        end
        weights = :(vcat.(1 .- w, w))
        if named
            push!(ex.args, :(weights = $weights))
            weights = :weights
        end
        push!(ex.args, :(y .~ MixtureModel.(
            vcat.(Normal.(mu, sigma), Normal.(mu, tau)), $weights)))
        ex
    end
    function oracle(p, data)
        prior = sum(normal_prior, [p.a, p.b, p.z..., p.h...]) +
            sum(s -> D.logpdf(D.Exponential(), s), [p.sigma, p.tau, p.group_scale])
        return prior + sum(eachindex(data.y); init = 0.0) do i
            mu = p.a + p.b * data.x[i] + p.group_scale * p.z[data.g[i]]
            w = sigmoid(p.b + p.h[data.g[i]])
            D.logpdf(D.MixtureModel([D.Normal(mu, p.sigma), D.Normal(mu, p.tau)],
                [1 - w, w]), data.y[i])
        end
    end
    p = (; a = 0.2, b = -0.4, sigma = 0.7, tau = 1.8,
        z = [0.3, -0.2], h = [-0.6, 0.5], group_scale = 0.8)
    for n in (0, 1, 7)
        data = (; x = sin.(1:n), y = cos.(1:n), g = [1 + mod(i, 2) for i in 1:n])
        inline = check(source(false), data, p, oracle; audit = n == 7)
        named = check(source(true), data, p, oracle)
        @test inline.value ≈ named.value
        @test inline.gradient ≈ named.gradient
        expected = [begin
            mu = p.a + p.b * data.x[i] + p.group_scale * p.z[data.g[i]]
            w = sigmoid(p.b + p.h[data.g[i]])
            D.logpdf(D.MixtureModel([D.Normal(mu, p.sigma), D.Normal(mu, p.tau)],
                [1 - w, w]), data.y[i])
        end for i in eachindex(data.y)]
        @test inline.pointwise ≈ expected
        if n == 7
            # Equal component densities still differentiate both locations,
            # scales and the predictor-dependent probabilities normally.
            check(source(false), data, merge(p, (; sigma = 1.0, tau = 1.0)),
                oracle; replay = false)
        end
    end
    expr_count(x) = x isa Expr ? 1 + sum(expr_count, x.args; init = 0) : 0
    counts = map((3, 30)) do n
        data = (; x = sin.(1:n), y = cos.(1:n), g = [mod1(i, 2) for i in 1:n])
        bound = bind_data(lower_rkppl(source(false), keys(data); conditioned = (:y,)), data)
        expr_count(kernel_expr(bound, assign_layout(bound)))
    end
    @test counts[1] == counts[2]
end

@testset "arbitrary three-component probability rows and shared vectors" begin
    source = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        p = logistic.(a .+ x)
        q = logistic.(b .- x)
        y .~ MixtureModel.(vcat.(Normal.(a, 0.7), Normal.(b, 1.2), Normal.(0, 2)),
            vcat.(p, (1 .- p) .* q, (1 .- p) .* (1 .- q)))
    end
    data = (; x = [-0.5, 0.3, 0.8], y = [0.1, -0.7, 1.3])
    oracle(p, d) = normal_prior(p.a) + normal_prior(p.b) +
        sum(eachindex(d.y)) do i
            w, v = sigmoid(p.a + d.x[i]), sigmoid(p.b - d.x[i])
            D.logpdf(D.MixtureModel([D.Normal(p.a, 0.7), D.Normal(p.b, 1.2),
                D.Normal(0, 2)], [w, (1 - w) * v, (1 - w) * (1 - v)]), d.y[i])
        end
    check(source, data, (; a = 0.2, b = -0.4), oracle)

    for rowwise in (true, false), probabilities in ([0.3, 0.7], [0.0, 1.0])
        weights = rowwise ? :weights : :(Ref(weights))
        ex = quote
            a ~ Normal(0, 1)
            y .~ MixtureModel.(vcat.(Normal.(a, 1), Normal.(0, 2)), $weights)
        end
        d = (; y = data.y, weights = rowwise ? [copy(probabilities) for _ in data.y] : probabilities)
        f(p, d) = normal_prior(p.a) + sum(y -> D.logpdf(
            D.MixtureModel([D.Normal(p.a, 1), D.Normal(0, 2)], probabilities), y), d.y)
        check(ex, d, (; a = 0.2), f)
    end
end

@testset "saturated logistic mixture probabilities" begin
    ex = quote
        a ~ Normal(0, 1)
        w = logistic.(a .+ x)
        y .~ MixtureModel.(vcat.(Normal.(0, 1), Normal.(1, 2)), vcat.(1 .- w, w))
    end
    data = (; x = [0.1, 0.2], y = [0.3, -0.5])
    oracle(p, d) = normal_prior(p.a) + sum(eachindex(d.y)) do i
        w = sigmoid(p.a + d.x[i])
        D.logpdf(D.MixtureModel([D.Normal(0, 1), D.Normal(1, 2)], [1 - w, w]), d.y[i])
    end
    for a in (-50.0, 50.0)
        check(ex, data, (; a), oracle)
    end
end

@testset "live mixture weight support" begin
    ex = quote
        a ~ Normal(0, 1)
        y .~ MixtureModel.(vcat.(Normal.(0, 1), Normal.(1, 2)), vcat.(a .+ x, 1 .- a .- x))
    end
    d = (; x = [0.1, 0.2], y = [0.4, -0.5])
    bound = bind_data(lower_rkppl(ex, keys(d); conditioned = (:y,)), d)
    built = build_kernel(bound)
    likelihood = prepare_query(built, bound, :likelihood)
    @test isfinite(likelihood([0.3]))
    @test likelihood([-0.5]) == -Inf
    @test likelihood([1.5]) == -Inf
    @test likelihood([Inf]) == -Inf
    @test likelihood([NaN]) == -Inf

    # Bound rows obey the same simplex contract, including shape. A bare
    # flat vector broadcasts numbers, so it is not a column of simplexes.
    bound_weights = quote
        a ~ Normal(0, 1)
        y .~ MixtureModel.(vcat.(Normal.(a, 1), Normal.(0, 2)), weights)
    end
    plan = lower_rkppl(bound_weights, (:y, :weights); conditioned = (:y,))
    for weights in ([0.3, 0.7], [[1.0], [1.0]], [[0.3, 0.8], [0.3, 0.7]],
            [[-0.1, 1.1], [0.3, 0.7]], [[NaN, 0.7], [0.3, 0.7]],
            [[Inf, 0.7], [0.3, 0.7]])
        @test_throws ContractValidationError bind_data(plan, (; d.y, weights))
    end
    unnormalized = quote
        a ~ Normal(0, 1)
        y .~ MixtureModel.(vcat.(Normal.(0, 1), Normal.(1, 2)), vcat.(a .+ x, 1 .- x))
    end
    bound = bind_data(lower_rkppl(unnormalized, keys(d); conditioned = (:y,)), d)
    query = prepare_query(build_kernel(bound), bound, :likelihood)
    @test query([0.3]) == -Inf
end
end
