using DifferentiationInterface, Distributions, Enzyme, LinearAlgebra, ReactiveKernels, ReactiveKernelsPPL, Test
using LogExpFunctions: logsumexp

# A categorical response whose model states its class count K observes any
# codes in 1:K, including a subset of them. CategoricalLogit states K with its
# non-reference locations (K = 3 here), and Categorical states it with its
# probability vector (a three-entry simplex). Each row's density is the
# ordinary one, as Stan's `categorical_logit` and `categorical` define it.
# Replaying a fit on a design whose rows all hold one class k gives that
# class's held-out probabilities through the pointwise query (snag
# rkppl-leveled-re-328038d0).

const _CAP_CAT_LOGIT = quote
    a2 ~ Normal(0, 1)
    b2 ~ Normal(0, 1)
    a3 ~ Normal(0, 1)
    b3 ~ Normal(0, 1)
    eta2 = a2 .+ b2 .* x
    eta3 = a3 .+ b3 .* x
    y .~ CategoricalLogit.(eta2, eta3)
end

const _CAP_CAT_SIMPLEX = quote
    s ~ Dirichlet(3, 1.0)
    y .~ Categorical(s)
end

_cap_cat_data(kind, y) = kind === :logit ?
    Dict(:x => collect(range(-0.5, 0.6; length = length(y))), :y => y) : Dict(:y => y)

_cap_cat_ast(kind) = kind === :logit ? _CAP_CAT_LOGIT : _CAP_CAT_SIMPLEX

function _cap_cat_pointwise(kind, q, data)
    kind === :logit || return log.(q.s[data[:y]])
    map(data[:x], data[:y]) do x, y
        logits = [0.0, q.a2 + q.b2 * x, q.a3 + q.b3 * x]
        logits[y] - logsumexp(logits)
    end
end

function _cap_cat_fixture(kind, observed; n = 6, plan = nothing)
    data = _cap_cat_data(kind, [observed[mod1(i, length(observed))] for i in 1:n])
    plan = plan === nothing ? lower_rkppl(_cap_cat_ast(kind), data; conditioned = keys(data)) : plan
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    q0 = kind === :logit ? (; a2 = 0.3, b2 = -0.4, a3 = -0.2, b3 = 0.7) : (; s = [0.2, 0.5, 0.3])
    u = unconstrain(built.layout, q0)
    sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    return (; kind, data, plan, bound, built, u, sampler)
end

_cap_cat_query(fx, name, u) =
    Base.invokelatest(prepare_query(fx.built, fx.bound, name), u)

function _cap_cat_fd(f, u)
    h = cbrt(eps(Float64))
    [(f(u + h * e) - f(u - h * e)) / (2h) for e in eachcol(Matrix{Float64}(I, length(u), length(u)))]
end

@testset "categorical responses observe a subset of their stated classes" begin
    for kind in (:logit, :simplex)
        full = _cap_cat_fixture(kind, (1, 2, 3))
        for observed in ((1, 2), (2,), (1, 3), (3, 1, 3))
            @testset "$kind $observed" begin
                fx = _cap_cat_fixture(kind, observed)
                @test only(fx.bound.responses).n_levels == 3
                kind === :simplex && @test only(fx.bound.vector_parameters).size == 3
                # The fit's coordinates are unchanged by the subset.
                @test fx.built.layout.total == full.built.layout.total
                q = constrain(fx.built.layout, fx.u)
                pointwise = _cap_cat_pointwise(kind, q, fx.data)
                @test _cap_cat_query(fx, :pointwise, fx.u).y ≈ pointwise
                @test _cap_cat_query(fx, :likelihood, fx.u) ≈ sum(pointwise)
                # Prior and Jacobian terms do not depend on y, so the total
                # moves by exactly the likelihood difference from the binding
                # that holds every class.
                value, gradient = sampler_value_and_gradient!(fx.sampler, similar(fx.u), fx.u)
                fullq = constrain(full.built.layout, full.u)
                fullvalue = first(sampler_value_and_gradient!(full.sampler, similar(full.u), full.u))
                @test value - fullvalue ≈
                    sum(pointwise) - sum(_cap_cat_pointwise(kind, fullq, full.data))
                if kind === :logit
                    @test value ≈ sum(pointwise) + sum(logpdf.(Normal(), (q.a2, q.b2, q.a3, q.b3)))
                end
                primal(v) = first(sampler_value_and_gradient!(fx.sampler, similar(v), v))
                @test gradient ≈ _cap_cat_fd(primal, fx.u) rtol = 5e-6
            end
        end
        @testset "$kind replay of a fit's plan, one class per design" begin
            m = 5
            probs = zeros(m, 3)
            for k in 1:3
                fx = _cap_cat_fixture(kind, (k,); n = m, plan = full.plan)
                @test only(fx.bound.responses).n_levels == 3
                @test fx.built.layout.total == full.built.layout.total
                pw = _cap_cat_query(fx, :pointwise, fx.u).y
                @test pw ≈ _cap_cat_pointwise(kind, constrain(fx.built.layout, fx.u), fx.data)
                probs[:, k] = exp.(pw)
            end
            @test vec(sum(probs; dims = 2)) ≈ ones(m)
        end
        @testset "$kind codes outside 1:3 stay refused" begin
            for bad in ([1, 2, 4, 1, 2, 3], [0, 1, 2, 1, 2, 3])
                data = _cap_cat_data(kind, bad)
                # refused: an observed code outside the stated support 1:K has no category
                @test_throws ContractValidationError bind_data(full.plan, data)
            end
            data = _cap_cat_data(kind, Float64[1, 2, 1, 2, 1, 2])
            # refused: a non-integer category code (leveled responses hold integers)
            @test_throws ContractValidationError bind_data(full.plan, data)
        end
    end
end
