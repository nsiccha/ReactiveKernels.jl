using Test, ReactiveKernels, ReactiveKernelsPPL
using DifferentiationInterface, Enzyme
import Distributions as PD

_positive_half(d) = PD.truncated(d, 0, Inf)
function _positive_prior_query(expr, data, oracle)
    plan = bind_data(lower_rkppl(expr, data; conditioned=keys(data)), data)
    built = build_kernel(plan)
    u = [0.12sin(i) for i in 1:built.layout.total]
    q = constrain(built.layout, u)
    query = prepare_query(built, plan, :prior)
    @test Base.invokelatest(query, u) ≈ oracle(q, built.layout) atol=1e-12 rtol=1e-12
    @test unconstrain(built.layout, q) ≈ u atol=1e-12
    return plan, built, u
end

@testset "scalar half priors keep a named density call" begin
    data = Dict(:y => [0.1, 0.2])
    scale = 10.0
    sigma = 1.5
    u = [log(sigma)]
    for (family, helper) in ((:HalfNormal, "_half_normal_logpdf_on_support"),
                             (:HalfCauchy, "_half_cauchy_logpdf_on_support"))
        prior = Expr(:call, family, scale)
        plan = bind_data(lower_rkppl(quote
            sigma ~ $prior
            y .~ Normal.(0, sigma)
        end, data; conditioned=(:y,)), data)
        built = build_kernel(plan)
        source = string(model_view(built; bound=plan).source)
        @test occursin(helper, source)
        @test !occursin("_ppl_prior_sigma.logpdf.", source)
        # Generated code calls operations by slot; the prior's recipe source
        # names the helper it calls.
        recipe_sources(k) = join((string(e.recipe.source) for e in recipe_inventory(k)), "\n")
        @test occursin(helper, recipe_sources(built.spec))
        code = string(code_expr(built.spec))
        @test !occursin("_ppl_prior_sigma.logpdf.", code)

        q = prepare_sampler(built, plan, u;
            backend=AutoEnzyme(mode=Enzyme.Reverse))
        @test occursin(helper, recipe_sources(q.kernel))
        prepared_code = string(code_expr(q.kernel))
        @test !occursin("_ppl_prior_sigma.logpdf.", prepared_code)
        grad = zeros(length(u))
        value, _ = sampler_value_and_gradient!(q, grad, u)
        @test isfinite(value)
        prior_grad = family === :HalfNormal ? 1 - (sigma / scale)^2 :
            1 - 2sigma^2 / (scale^2 + sigma^2)
        expected = prior_grad - length(data[:y]) + sum(abs2, data[:y]) / sigma^2
        @test only(grad) ≈ expected atol=1e-10 rtol=1e-10
    end
end

@testset "positive priors have one normalized meaning in generic slots" begin
    data = Dict(:y => [0.1, -0.2, 0.3, 0.4])
    for (rhs, d) in ((:(HalfNormal(2)), PD.Normal(0, 2)),
                    (:(HalfCauchy(2)), PD.Cauchy(0, 2)),
                    (:(truncated(Normal(0, 2), 0, Inf)), PD.Normal(0, 2)),
                    (:(truncated(Cauchy(0, 2), 0, Inf)), PD.Cauchy(0, 2)))
        expected = _positive_half(d)
        _positive_prior_query(quote s ~ $rhs; y .~ Normal.(s, 1) end,
            data, (q, l) -> PD.logpdf(expected, q.s))
        _positive_prior_query(quote
            s[1:2] .~ $rhs
            mu = sum(s)
            y .~ Normal.(mu, 1)
        end, data, (q, l) -> sum(PD.logpdf.(expected, q.s)))
        _positive_prior_query(quote
            s[1:2, 1:2] .~ $rhs
            mu = sum(s)
            y .~ Normal.(mu, 1)
        end, data, (q, l) -> sum(PD.logpdf.(expected, q.s)))
    end
    # A nonzero location and finite upper bound exercise the full normalizer.
    rhs = :(truncated(Normal(0.7, 2), 0.3, 4))
    expected = PD.truncated(PD.Normal(0.7, 2), 0.3, 4)
    _positive_prior_query(quote
        s[1:2] .~ $rhs
        mu = sum(s)
        y .~ Normal.(mu, 1)
    end, data, (q, l) -> sum(PD.logpdf.(expected, q.s)))
end

@testset "legacy positive spellings name normalized replacements" begin
    # Refused: non-Julia constructor keywords and silently constrained bare
    # distributions violate P3 and the user-approved stan-halves decision 0m1j3iz.
    for rhs in (:(Normal(0,2;lower=0)), :(Cauchy(0,2;lower=0)), :(Flat(;lower=0)), :(Flat(;lower=0,upper=2)))
        err = try
            lower_rkppl(quote s ~ $rhs; y .~ Normal.(s,1) end,(:y,);conditioned=(:y,))
            nothing
        catch e; e end
        @test err isa SurfaceLoweringError
        @test occursin(rhs.args[1]===:Flat ? "Exponential" : "truncated",sprint(showerror,err))
    end
end
