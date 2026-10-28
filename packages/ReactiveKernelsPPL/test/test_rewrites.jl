using ReactiveKernelsPPL, Distributions, Test, LinearAlgebra

_rewrite_node(plan, node, u) = Base.invokelatest(
    prepare_query(build_kernel(plan), plan, node), u)

function _rewrite_glm_ast(head)
    rhs = Expr(:call, head, :X, :alpha, :beta)
    head === :NormalIDGLM && push!(rhs.args, 2.0)
    return quote
        X = hcat(x1, x2)
        alpha ~ Normal(0, 5)
        beta[axes(X, 2)] .~ Normal.(0, 2)
        y ~ $rhs
    end
end

function _rewrite_glm_data(head, n)
    x1 = collect(range(-0.8, 0.9; length = n))
    x2 = fill(0.4, n)
    y = head === :NormalIDGLM ? 0.2 .+ x1 :
        head === :BernoulliLogitGLM ? Int.(x1 .> 0) : Int.(x1 .> 0) .+ 1
    return (; x1, x2, y)
end

function _rewrite_glm_oracle(head, data, alpha, beta)
    eta = alpha .+ hcat(data.x1, data.x2) * beta
    family = head === :NormalIDGLM ? Normal.(eta, 2.0) :
        head === :BernoulliLogitGLM ? Bernoulli.(1 ./ (1 .+ exp.(-eta))) :
        Poisson.(exp.(eta))
    return logpdf(Normal(0, 5), alpha) + sum(logpdf.(Normal(0, 2), beta)) +
        sum(logpdf.(family, data.y))
end

@testset "whole-data GLM observations keep response roles" begin
    for head in (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM), n in (2, 4, 9)
        ast = _rewrite_glm_ast(head)
        data = _rewrite_glm_data(head, n)
        ast_before, data_before = deepcopy(ast), deepcopy(data)
        model = ReactiveKernelsPPL.RKPPLModel(ast, @__MODULE__)
        u = [0.1, -0.2, 0.3]
        oracle = _rewrite_glm_oracle(head, data, u[1], u[2:3])
        columns = Dict(pairs(data))
        # Names-only, values, and model entry points share observation roles.
        plans = (
            bind_data(lower_rkppl(ast, keys(data); conditioned = (:y,)), columns),
            bind_data(lower_rkppl(ast, data; conditioned = keys(data)), columns),
            bind_data(lower_rkppl(ast, data; conditioned = (; y = data.y)), columns),
            bind_data(lower_rkppl(model, data; conditioned = (:y,)), columns),
            model(; data.x1, data.x2) | (; data.y),
            condition(model(; data.x1, data.x2); data.y),
        )
        for plan in plans
            @test isempty(plan.conditioned)
            @test only(plan.responses).response === :y
            @test plan.n_obs == n
            @test build_kernel(plan).layout.total == 3
            @test _rewrite_node(plan, :sampler, u) ≈ oracle
        end
        # Conditioning an ordinary parameter still retains its prior density.
        fixed = model(; data.x1, data.x2) | (; data.y, alpha = u[1])
        @test fixed.conditioned == Set([:alpha])
        @test build_kernel(fixed).layout.total == 2
        @test _rewrite_node(fixed, :sampler, u[2:3]) ≈ oracle
        rebound = condition(first(plans); y = reverse(data.y))
        @test _rewrite_node(rebound, :sampler, u) ≈
            _rewrite_glm_oracle(head, merge(data, (; y = reverse(data.y))), u[1], u[2:3])
        # Call pins remove the response statement (conditioning principle 9).
        # Pin X too: only its shape is needed by the remaining beta prior.
        pinned = bind_data(model(; X = hcat(data.x1, data.x2), y = data.y))
        @test isempty(pinned.responses)
        @test _rewrite_node(pinned, :sampler, u) ≈
            logpdf(Normal(0, 5), u[1]) + sum(logpdf.(Normal(0, 2), u[2:3]))
        # Refused: supplied sampling names require explicit observation or a
        # merge pin, per standing conditioning principle 9 (decision 18h1h54).
        @test_throws SurfaceLoweringError lower_rkppl(ast, keys(data))
        @test ast == ast_before
        @test data == data_before
    end
end

@rkppl _rewrite_scaled_columns(X) = begin
    scale ~ Exponential(1)
    center[axes(X, 2)] .~ Normal.(0, 1)
    z[axes(X, 2)] .~ Normal.(0, 1)
    return scale .* center .* z
end

@testset "caller-owned scalar and array priors accept scoped operations" begin
    body = deepcopy(_rewrite_scaled_columns.body)
    model = @rkppl begin
        b ~ _rewrite_scaled_columns(X)
        y .~ Normal.(X * b, 1)
    end
    changed = merge(model, quote
        b.scale ~ Exponential(0.25)
        b.center[axes(X, 2)] .~ Normal.([0.2, -0.1], [1.0, 0.5])
    end)
    X, y = ones(3, 2), [0.1, -0.2, 0.3]
    observed = changed(; X) |
        (; y, var"b.scale" = 0.3, var"b.center" = [0.2, 0.4])
    @test build_kernel(observed).layout.total == 2
    @test _rewrite_node(observed, :likelihood, zeros(2)) ≈
        logpdf(Exponential(0.25), 0.3) + logpdf(Normal(0.2, 1), 0.2) +
        logpdf(Normal(-0.1, 0.5), 0.4) + sum(logpdf.(Normal(), y))
    @test _rewrite_node(observed, :prior, zeros(2)) ≈ 2logpdf(Normal(), 0)
    pinned = merge(changed, (; var"b.scale" = 0.3,
        var"b.center" = [0.2, 0.4]))(; X) | (; y)
    @test _rewrite_node(pinned, :likelihood, zeros(2)) ≈ sum(logpdf.(Normal(), y))
    @test _rewrite_scaled_columns.body == body
end

@testset "conditioning inputs cannot capture authored names" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        y .~ Normal.(a, _rkppl_conditioned_a)
    end
    @test_throws SurfaceLoweringError model(; _rkppl_conditioned_a = 0.8) |
        (; a = 0.3, y = [0.1, 0.2])
    @test_throws SurfaceLoweringError lower_rkppl(model.ast,
        (:y, :a, :_rkppl_conditioned_a); conditioned = (:y, :a))
    plan = model(; _rkppl_conditioned_a = 0.8) | (; y = [0.1, 0.2])
    @test_throws SurfaceLoweringError condition(plan; a = 0.3)
end

@rkppl _rewrite_scale() = begin
    tau ~ HalfNormal(1)
    tau
end

@rkppl _rewrite_coefficients() = begin
    b[1:2] .~ Normal.(0, 1)
    b
end
const _rewrite_pinned_coefficients = merge(_rewrite_coefficients, (; b = [0.2, -0.1]))

@testset "prior-only models and complete joint declarations" begin
    prior = @rkppl begin
        a ~ Normal(0, 1)
    end
    plan = bind_data(prior())
    @test _rewrite_node(plan, :prior, [0.4]) ≈ logpdf(Normal(), 0.4)
    pinned = bind_data(prior(; a = 0.2))
    @test build_kernel(pinned).layout.total == 0
    @test _rewrite_node(pinned, :sampler, Float64[]) == 0

    joint = @rkppl begin
        a ~ Normal(0, 1)
        L_scales[1:2] .~ Exponential.(1)
        L_L_corr ~ LKJCholesky(2, 2)
        L = L_scales .* L_L_corr
        [y1, y2] ~ MvNormalCholesky([a, a], L)
    end
    changed = merge(joint, :([y1, y2] ~ MvNormalCholesky([a, 2a], L)))
    @test changed.ast != joint.ast
    @test length((changed() | (; y1 = [0.2], y2 = [0.1])).responses) == 1
    @test_throws SurfaceLoweringError merge(joint, (; y1 = [0.2]))
    fixed = merge(joint, (; y1 = [0.2], y2 = [0.1]))
    @test isempty(bind_data(fixed()).responses)
    @test_throws SurfaceLoweringError merge(joint, :(y1 .~ Normal.(0, 1)))
end

@testset "literal and scoped array observations, pins and variants" begin
    X = [1.0 0.2; -0.5 1.0; 0.3 -0.2]
    b = [0.2, -0.1]
    y = [0.1, -0.2, 0.3]
    model = @rkppl begin
        a ~ Normal(0, 1)
        b[1:2] .~ Normal.(0, 1)
        y .~ Normal.(a .+ X * b, 1)
    end
    observed = model(; X) | (; b, y)
    pinned = model(; X, b) | (; y)
    @test build_kernel(observed).layout.total == 1
    @test _rewrite_node(observed, :likelihood, [0.4]) -
        _rewrite_node(pinned, :likelihood, [0.4]) ≈ sum(logpdf.(Normal(), b))
    @test _rewrite_node(pinned, :likelihood, [0.4]) ≈
        sum(logpdf.(Normal.(0.4 .+ X * b, 1), y))
    nested = @rkppl begin
        z ~ _rewrite_coefficients()
        y .~ Normal.(X * z, 1)
    end
    scoped = merge(nested, :(z.b[1:2] .~ Normal.(0, 2)))
    @test only((scoped(; X) | (; y)).array_parameters).args.arg2 == 2
    @test _rewrite_node(merge(nested, (; var"z.b" = b))(; X) | (; y),
        :likelihood, Float64[]) ≈ sum(logpdf.(Normal.(X * b, 1), y))
    variant = @rkppl begin
        z ~ _rewrite_pinned_coefficients()
        y .~ Normal.(X * z, 1)
    end
    @test _rewrite_node(variant(; X) | (; y), :likelihood, Float64[]) ≈
        sum(logpdf.(Normal.(X * b, 1), y))
end

_rewrite_data_transform(x) = sum(abs2, x)
@testset "conditioning recomputes bound data definitions" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        tau ~ HalfNormal(1)
        x2 = _rewrite_data_transform(x)
        sd = tau * x2
        y .~ Normal.(a, sd)
    end
    x = [0.2, 0.4]
    y = [0.1, -0.2]
    original = model(; x) | (; y, tau = 0.3)
    rebound = condition(original; tau = 0.6)
    @test _rewrite_node(rebound, :likelihood, [0.4]) ≈
        sum(logpdf.(Normal(0.4, 0.6 * sum(abs2, x)), y)) +
        logpdf(truncated(Normal(), 0, Inf), 0.6)
end

@testset "conditioned structured and bounded values retain constrained density" begin
    matrix = @rkppl begin
        L ~ LKJCholesky(2, 2)
        y .~ Normal.(L[2, 1], 1)
    end
    L = [1.0 0.0; 0.2 sqrt(1 - 0.2^2)]
    y = [0.1, 0.3]
    plan = matrix() | (; L, y)
    @test build_kernel(plan).layout.total == 0
    @test _rewrite_node(plan, :likelihood, Float64[]) ≈
        logpdf(LKJCholesky(2, 2), Cholesky(LowerTriangular(L))) +
        sum(logpdf.(Normal(0.2, 1), y))
    @test _rewrite_node(plan, :log_jacobian, Float64[]) == 0
    bounded = @rkppl begin
        a ~ Normal(0, 1)
        p ~ Uniform(0, 1)
        y .~ Normal.(a, 1)
    end
    outside = bounded() | (; p = -0.2, y)
    @test _rewrite_node(outside, :likelihood, [0.4]) == -Inf
    @test _rewrite_node(outside, :prior, [0.4]) ≈ logpdf(Normal(), 0.4)
    truncated_array = @rkppl begin
        b[axes(X, 2)] .~ truncated.(Normal.(0, 1), -1, 1)
        y .~ Normal.(0, 1)
    end
    X = zeros(length(y), 2)
    good = truncated_array(; X) | (; b = [0.2, -0.1], y)
    @test _rewrite_node(good, :likelihood, Float64[]) ≈
        sum(logpdf.(truncated(Normal(), -1, 1), [0.2, -0.1])) +
        sum(logpdf.(Normal(), y))
    bad = condition(good; b = [0.2, -2.0])
    @test _rewrite_node(bad, :likelihood, Float64[]) == -Inf
end

@testset "conditioned truncated priors keep their normalizer and gradients" begin
    y = [0.1, 0.3]
    for (prior, D, inside, outside) in (
            (:(truncated(Normal(0, 1), -1, 1)), truncated(Normal(), -1, 1), 0.2, -2.0),
            (:(truncated(Normal(0, 1), -Inf, 1)), truncated(Normal(), -Inf, 1), 0.2, 2.0),
            (:(truncated(LogNormal(0, 1), 0.5, Inf)), truncated(LogNormal(0, 1), 0.5, Inf), 0.8, 0.2),
            (:(truncated(Gamma(2, 1), 0.5, 3)), truncated(Gamma(2, 1), 0.5, 3), 0.8, 4.0))
        model = ReactiveKernelsPPL.RKPPLModel(quote
            a ~ $prior
            y .~ Normal.(a, 1)
        end, @__MODULE__)
        plan = model() | (; a = inside, y)
        @test _rewrite_node(plan, :likelihood, Float64[]) ≈
            logpdf(D, inside) + sum(logpdf.(Normal(inside, 1), y))
        @test _rewrite_node(condition(plan; a = outside), :likelihood, Float64[]) == -Inf
    end
    # The truncation normalizer depends on a sampled location; mu = -1.7 takes
    # the upper-tail difference, mu = 0.3 the lower one.
    hierarchical = @rkppl begin
        mu ~ Normal(0, 1)
        a ~ truncated(Normal(mu, 1), -1, 1)
        b[axes(X, 2)] .~ truncated.(Normal.(mu, 1), -1, 1)
        y .~ Normal.(mu, 1)
    end
    X = zeros(length(y), 2)
    b = [0.2, -0.1]
    plan = hierarchical(; X) | (; a = 0.4, b, y)
    built = build_kernel(plan)
    oracle(u) = logpdf(Normal(), u[1]) +
        logpdf(truncated(Normal(u[1], 1), -1, 1), 0.4) +
        sum(logpdf.(truncated(Normal(u[1], 1), -1, 1), b)) +
        sum(logpdf.(Normal(u[1], 1), y))
    for u in ([0.3], [-1.7])
        _check_model_math(built, plan, u, oracle)
    end
end
@rkppl _rewrite_nested() = begin
    s ~ _rewrite_scale()
    s
end

@testset "conditioned scopes, whole declarations, and prior-only density" begin
    model = @rkppl begin
        scale ~ _rewrite_nested()
        y .~ Normal.(scale, 1)
    end
    data = (; y = [0.2, -0.1], var"scale.s.tau" = 0.3)
    observed = model() | data
    @test build_kernel(observed).layout.total == 0
    @test _rewrite_node(observed, :likelihood, Float64[]) ≈
        sum(logpdf.(Normal(0.3, 1), data.y)) + logpdf(truncated(Normal(), 0, Inf), 0.3)
    @test _rewrite_node(condition(observed; var"scale.s.tau" = 0.5),
        :likelihood, Float64[]) ≈ sum(logpdf.(Normal(0.5, 1), data.y)) +
        logpdf(truncated(Normal(), 0, Inf), 0.5)

    scalar = @rkppl begin
        tau ~ HalfNormal(1)
    end
    capture = scalar | (; tau = 0.3)
    plan = capture()
    @test build_kernel(capture).layout.total == 0
    @test _rewrite_node(plan, :likelihood, Float64[]) ≈ logpdf(truncated(Normal(), 0, Inf), 0.3)
    @test _rewrite_node(plan, :prior, Float64[]) == 0
    @test _rewrite_node(plan, :log_jacobian, Float64[]) == 0

    @test_throws SurfaceLoweringError lower_rkppl(model.ast, (; y = data.y))
    @test_throws SurfaceLoweringError (merge(model, (; var"scale.s.absent" = 1))() | (; y = data.y))
end
const _rewrite_variant = merge(_rewrite_nested, :(s.tau ~ HalfCauchy(2)))
const _rewrite_fixed_variant = merge(_rewrite_nested, (; var"s.tau" = 0.3))
const _rewrite_scalar_variant = merge(_rewrite_scale, (; tau = 0.4))
@rkppl _rewrite_alt() = begin
    sd ~ HalfCauchy(0.7)
    sd
end
@rkppl _rewrite_grand() = begin
    child ~ _rewrite_nested()
    child
end
const _rewrite_call_variant = merge(_rewrite_grand, :(child.s ~ _rewrite_alt()))

@testset "replacing and pinning a child call removes its former densities" begin
    model = @rkppl begin
        scale ~ _rewrite_nested()
        y .~ Normal.(scale, 1)
    end
    y = [0.2, -0.1]
    changed = merge(model, :(scale.s ~ _rewrite_alt()))() | (; y)
    @test length(changed.parameters) == 1
    @test only(changed.parameters).family === :cauchy
    @test only(changed.parameters).args.arg2 == 0.7
    fixed = merge(model, (; var"scale.s" = 0.3))() | (; y)
    @test isempty(fixed.parameters)
    @test _rewrite_node(fixed, :prior, Float64[]) == 0
    @test _rewrite_node(fixed, :likelihood, Float64[]) ≈ sum(logpdf.(Normal(0.3, 1), y))
    variant = @rkppl begin
        scale ~ _rewrite_call_variant()
        y .~ Normal.(scale, 1)
    end
    vp = variant() | (; y)
    @test length(vp.parameters) == 1
    @test only(vp.parameters).args.arg2 == 0.7
end

@testset "merged submodel variants execute in their own scopes" begin
    original_body = deepcopy(_rewrite_nested.body)
    variant = @rkppl begin
        scale ~ _rewrite_variant()
        y .~ Normal.(scale, 1)
    end
    plan = variant() | (; y = [0.2, -0.1])
    @test only(plan.parameters).family === :cauchy
    @test only(plan.parameters).args.arg2 == 2
    @test _rewrite_nested.body == original_body
    for sm in (:_rewrite_fixed_variant, :_rewrite_scalar_variant)
        ast = Expr(:block, :(scale ~ $sm()), :(y .~ Normal.(scale, 1)))
        bound = RKPPLModel(ast, @__MODULE__)() | (; y = [0.2, -0.1])
        value = sm === :_rewrite_fixed_variant ? 0.3 : 0.4
        @test build_kernel(bound).layout.total == 0
        @test _rewrite_node(bound, :likelihood, Float64[]) ≈
            sum(logpdf.(Normal(value, 1), [0.2, -0.1]))
    end
end

@testset "statement rewrites preserve declarations and scopes" begin
    base = @rkppl begin
        b[1:2] .~ Normal.(0, 1)
        scale ~ _rewrite_nested()
        y .~ Normal.(scale, 1)
    end
    data = (; y = [0.2, -0.1])
    before = deepcopy(base.ast)
    changed = merge(base, :(b[1:2] .~ Normal.(0, 2)),
        :(scale.s.tau ~ HalfCauchy(0.5)))
    plan = lower_rkppl(changed, data; conditioned = (:y,))
    @test only(plan.array_parameters).args.arg2 == 2
    @test only(plan.parameters).family == :cauchy
    @test only(plan.parameters).args.arg2 == 0.5
    original = lower_rkppl(base, data; conditioned = (:y,))
    @test only(original.parameters).family == :normal
    @test isempty(base.fixed) && isempty(base.rewrites)

    pinned = merge(base, (; b = [0.4, 0.6], var"scale.s.tau" = 0.3))
    bound = pinned() | data
    @test isempty(bound.array_parameters) && isempty(bound.parameters)
    @test any(==(0.3), values(bound.columns))
    @test base.ast == before
    # refused: a partial array write is not replacement of a complete statement (P9).
    @test_throws SurfaceLoweringError merge(base, :(b[1] = 2))

    variant = merge(_rewrite_scale, :(tau ~ HalfCauchy(2)))
    @test variant !== _rewrite_scale
    @test first(filter(x -> x isa Expr, variant.body.args)).args[3] == :(HalfCauchy(2))
    pinned_variant = merge(_rewrite_scale, (; tau = 0.3))
    @test _rewrite_scale.body != pinned_variant.body
end

@testset "conditioning retains density without coordinates or Jacobian" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        tau ~ HalfNormal(1)
        y .~ Normal.(a, tau)
    end
    y = [0.2, -0.1]
    pinned = model(; tau = 0.3) | (; y)
    observed = model() | (; y, tau = 0.3)
    @test build_kernel(pinned).layout.total == 1
    @test build_kernel(observed).layout.total == 1
    @test _rewrite_node(observed, :likelihood, [0.4]) -
        _rewrite_node(pinned, :likelihood, [0.4]) ≈
        logpdf(truncated(Normal(), 0, Inf), 0.3)
    @test _rewrite_node(observed, :prior, [0.4]) ≈ logpdf(Normal(), 0.4)
    @test _rewrite_node(observed, :log_jacobian, [0.4]) == 0
    rebound = condition(observed; tau = 0.6)
    @test _rewrite_node(rebound, :likelihood, [0.4]) ≈
        sum(logpdf.(Normal(0.4, 0.6), y)) + logpdf(truncated(Normal(), 0, Inf), 0.6)
    @test observed.columns[ReactiveKernelsPPL._conditioned_input(:tau)] == 0.3
    # Direct lowering and binding expose observation roles explicitly too.
    values = (; y, tau = 0.3)
    direct = lower_rkppl(model.ast, values; conditioned = (:y, :tau))
    direct = bind_data(direct, Dict{Symbol,Any}(pairs(values)))
    @test _rewrite_node(direct, :likelihood, [0.4]) ≈
        _rewrite_node(observed, :likelihood, [0.4])

    for n in (2, 7)
        array = @rkppl begin
            b[axes(X, 2)] .~ Normal.(0, 1)
            a ~ Normal(0, 1)
            y .~ Normal.(a, 1)
        end
        b = fill(0.2, n)
        X = zeros(length(y), n)
        plan = array(; X) | (; b, y)
        @test build_kernel(plan).layout.total == 1
        @test _rewrite_node(plan, :likelihood, [0.4]) ≈
            sum(logpdf.(Normal(0.4, 1), y)) + sum(logpdf.(Normal(), b))
        @test b == fill(0.2, n)
    end
end
