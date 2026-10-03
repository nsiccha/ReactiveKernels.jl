# Values compose with Julia semantics (standing principles 3/10a;
# capability-values todo 15lq8iu). Keep independent numeric oracles alongside
# lowering checks: admitting a spelling must also preserve its mathematics.
using Distributions
using ReactiveKernels
using ReactiveKernelsPPL
using Test

_vc_shifted(x; by) = x .+ by
@rkppl _vc_function_return(f) = begin
    v = f(1.0)
    return v
end

function _vc_model(ast, data)
    bound = bind_data(lower_rkppl(ast, data; conditioned = (:y,)),
        Dict{Symbol,ColumnData}(pairs(data)))
    return bound, build_kernel(bound)
end

function _vc_cases(n)
    x = [0.4 + i / n for i in 1:n]
    y = [0.3 * cos(i) for i in 1:n]
    data = (; x, y)
    cases = [
        (label = "literal offset", ast = quote
            b ~ Normal(0, 1)
            mu = 1.5 .+ b .* x
            y .~ Normal.(mu, 1.0)
        end, data, mean = q -> 1.5 .+ q.b .* x),
        (label = "bound scalar offset", ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ off .+ b .* x
            y .~ Normal.(mu, 1.0)
        end, data = merge(data, (; off = 0.3)),
        mean = q -> q.a .+ 0.3 .+ q.b .* x),
        (label = "undotted vector difference", ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            d ~ Normal(0, 1)
            be ~ Normal(0, 1)
            th = a .+ b .* x
            al = d .* x
            mu = be * th - al
            y .~ Normal.(mu, 1.0)
        end, data, mean = q -> q.be * (q.a .+ q.b .* x) - q.d .* x),
        (label = "nary scalar product", ast = quote
            a ~ Normal(0, 1)
            mu = a .+ 2 * 3 * x
            y .~ Normal.(mu, 1.0)
        end, data, mean = q -> q.a .+ 2 * 3 * x),
        (label = "nested reduction", ast = quote
            a ~ Normal(0, 1)
            z = mean(log.(x))
            mu = a .+ z .+ x
            y .~ Normal.(mu, 1.0)
        end, data, mean = q -> q.a .+ sum(log.(x)) / length(x) .+ x),
        (label = "derived weights", ast = quote
            a ~ Normal(0, 1)
            w = x .+ 1
            mu = a .+ x
            y .~ weighted.(Normal.(mu, 1.0), w)
        end, data, mean = q -> q.a .+ x, weights = x .+ 1),
        (label = "derived scale", ast = quote
            a ~ Normal(0, 1)
            w = x .+ 1
            mu = a .+ x
            y .~ Normal.(mu, w)
        end, data, mean = q -> q.a .+ x, scale = q -> x .+ 1),
        (label = "logistic value", ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            v = a .+ b .* x
            p = logistic.(v)
            y .~ Normal.(p, 1.0)
        end, data, mean = q -> 1 ./ (1 .+ exp.(-(q.a .+ q.b .* x)))),
        (label = "function result location", ast = quote
            s ~ Normal(0, 1)
            t = _vc_shifted(x; by = s)
            y .~ Normal.(t, 1.0)
        end, data, mean = q -> x .+ q.s),
        (label = "indexed function result scale", ast = quote
            s ~ Normal(0, 1)
            t = _vc_shifted(x; by = s)
            y .~ Normal.(x, t[1])
        end, data, mean = q -> x, scale = q -> first(x) + q.s),
        (label = "array and observation data", ast = quote
            z[1:$n] .~ Normal.(0, 1)
            a ~ Normal(0, 1)
            w = z .+ x
            mu = a .+ w
            y .~ Normal.(mu, 1.0)
        end, data, mean = q -> q.a .+ q.z .+ x),
        (label = "shared scalar distribution", ast = quote
            z[levels(g)] .~ Normal(0, 1)
            y .~ Normal.(z[g], 1.0)
        end, data = (; y, g = [mod1(i, 2) for i in 1:n]),
        mean = q -> q.z[[mod1(i, 2) for i in 1:n]]),
        (label = "whole array beside a gather", ast = quote
            a[levels(g)] .~ Normal.(0, 1)
            c[levels(g)] .~ Normal.(0, 2)
            mu = a .+ c[g]
            y .~ Normal.(mu, 1.0)
        end, data = (; y, g = collect(1:n)),
        mean = q -> q.a .+ q.c,
        prior = (q, u) -> sum(logpdf.(Normal(), q.a)) +
            sum(logpdf.(Normal(0, 2), q.c))),
        (label = "distribution value", ast = quote
            a ~ Normal(0, 1)
            m = Normal(0, 1)
            mu = a .+ x
            y .~ Normal.(mu, 1.0)
        end, data, mean = q -> q.a .+ x),
        (label = "library contrast composition", ast = quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            s ~ Dirichlet([1.0, 1.0])
            m ~ monotonic(c, s)
            w = 2 .* b .* m .* x
            unused = sum(m)
            mu = a .- w .+ sum(m)
            y .~ Normal.(mu, 1.0)
        end, data = merge(data, (; c = [mod1(i, 3) for i in 1:n])),
        mean = q -> begin
            m = [0.0, q.s[1], 1.0][[mod1(i, 3) for i in 1:n]]
            q.a .- 2 .* q.b .* m .* x .+ sum(m)
        end,
        prior = (q, u) -> logpdf(Normal(), q.a) + logpdf(Normal(), q.b) +
            logpdf(Dirichlet([1.0, 1.0]), q.s) + sum(log.(q.s))),
        (label = "derived grouping and indicator", ast = quote
            h = g .+ 1
            z[levels(h)] .~ Normal.(0, 1)
            a ~ Normal(0, 1)
            w = (h .== 2) .* z[h]
            mu = a .- 2 .* w .* x
            y .~ Normal.(mu, 1.0)
        end, data = merge(data, (; g = [mod1(i, 2) for i in 1:n])),
        mean = q -> q.a .- 2 .* ([mod1(i, 2) == 1 for i in 1:n] .*
            q.z[[mod1(i, 2) for i in 1:n]]) .* x),
        (label = "function argument", ast = quote
            a ~ Normal(0, 1)
            v ~ _vc_function_return(exp)
            mu = a .+ v .+ x
            y .~ Normal.(mu, 1.0)
        end, data, mean = q -> q.a .+ exp(1.0) .+ x),
        (label = "plate gather through a data alias", ast = quote
            h = g
            z[levels(g)] .~ Normal.(0, 1)
            @plate for i in eachindex(y)
                y[i] ~ Normal(z[h[i]], 1.0)
            end
        end, data = (; y, g = [mod1(i, 2) for i in 1:n]),
        mean = q -> q.z[[mod1(i, 2) for i in 1:n]]),
    ]
    return cases
end

function _vc_oracle(case, layout, u)
    q = constrain(layout, u)
    prior = hasproperty(case, :prior) ? case.prior(q, u) : sum(logpdf.(Normal(), u))
    scale = hasproperty(case, :scale) ? case.scale(q) : 1.0
    weights = hasproperty(case, :weights) ? case.weights : 1.0
    mu = case.mean(q)
    return prior + sum(weights .* logpdf.(Normal.(mu, scale), case.data.y))
end

_vc_empty_cases() = filter(c -> c.label in
    ("literal offset", "bound scalar offset", "function result location",
        "array and observation data"), _vc_cases(0))

@testset "ordinary composed values: density and native gradient" begin
    for case in vcat(_vc_cases(4), _vc_empty_cases())
        @testset "$(case.label) / n=$(length(case.data.y))" begin
            before = deepcopy(case.data)
            plan, built = _vc_model(case.ast, case.data)
            @test all(t -> t.kind === OffsetTerm ||
                haskey(t.options, :parameter) || t.kind === ComposedTerm,
                Iterators.flatten(p.terms for p in plan.predictors))
            u = [0.2 * sin(i) for i in 1:built.layout.total]
            oracle = w -> _vc_oracle(case, built.layout, w)
            @test _query(built.spec, plan, :posterior, u) ≈ oracle(u) rtol=1e-12
            gradient = _check_gradient(built.spec, plan, u)
            @test gradient ≈ _findiff_grad(oracle, u) rtol=1e-5 atol=1e-7
            @test case.data == before
        end
    end
end
