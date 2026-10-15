using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL, Test

# Whole-data GLM objects read their arguments as values, like the ordinary
# `X * b` predictor path: the design matrix may be bound data, a definition
# or an inline matrix (an intercept column or a parameter column included);
# the intercept a literal, scalar data, a sampled value or an expression; the
# coefficients any vector value; the Normal scale any positive scalar value.

const _GV_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

_gv_rows(n) = collect(range(-1.0, 1.0; length = n))
_gv_X(n) = hcat(_gv_rows(n), cos.(1:n), 0.3 .* sin.(2 .* (1:n)))
_gv_y(n) = _gv_X(n) * [0.4, -0.2, 0.1] .+ 0.3 .+ 0.05 .* sin.(3 .* (1:n))
_gv_counts(n) = [mod(3i, 4) for i in 1:n]
_gv_flags(n) = [isodd(i ÷ 2) ? 1 : 0 for i in 1:n]
_gv_u(built) = [0.15sin(i + 0.3) for i in 1:built.layout.total]

function _gv_fd(f, u; h = 1e-6)
    return [(f(u .+ h .* (eachindex(u) .== i)) - f(u .- h .* (eachindex(u) .== i))) / 2h
        for i in eachindex(u)]
end

# Likelihood against the independent oracle `lik(constrained values)`, the
# native Enzyme gradient against central differences, and the caller's data
# left unchanged.
function _gv_check(plan, data, lik)
    saved = deepcopy(data)
    built = build_kernel(plan)
    u = _gv_u(built)
    likelihood = prepare_query(built, plan, :likelihood)
    @test Base.invokelatest(likelihood, u) ≈ lik(constrain(built.layout, u)) rtol = 1e-10
    q = prepare_sampler(built, plan, u; backend = _GV_BACKEND)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ q(u)
    @test g ≈ _gv_fd(v -> q(v), u) rtol = 1e-5 atol = 1e-6
    @test data == saved
    return built
end

@testset "zero-centered Normal vector prior scales before squaring" begin
    m = @rkppl begin
        beta[1:2] .~ Normal.(0, 1e150)
    end
    plan = bind_data(m())
    built = build_kernel(plan)
    u = [1e160, 0.0]
    expected = sum(logpdf.(Normal(0, 1e150), u))
    prior = prepare_query(built, plan, :prior)
    @test isfinite(Base.invokelatest(prior, u))
    @test Base.invokelatest(prior, u) ≈ expected rtol = 1e-12
    q = prepare_sampler(built, plan, u; backend = _GV_BACKEND)
    g = similar(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test value ≈ expected rtol = 1e-12
    @test g ≈ [-1e-140, 0.0] rtol = 1e-10
end

@testset "GLM objects over bound data matrices and literal intercepts" begin
    n = 7
    X, y, counts, flags = _gv_X(n), _gv_y(n), _gv_counts(n), _gv_flags(n)
    m = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(X, 0.0, beta, sigma)
    end
    built = _gv_check(m(; X) | (; y), (; X, y),
        nt -> sum(logpdf.(Normal.(X * nt.beta, nt.sigma), y)))
    @test coordinate_names(built.layout) == [:sigma, Symbol("beta.1"),
        Symbol("beta.2"), Symbol("beta.3")]
    # A literal zero intercept adds no ones column.
    code = sprint(print, kernel_expr(m(; X) | (; y), built.layout))
    @test !occursin("ones(", code)
    @test occursin("normal_id_glm", code)

    m = @rkppl begin
        alpha ~ Normal(0, 10)
        beta[axes(lags, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(lags, alpha, beta, sigma)
    end
    _gv_check(m(; lags = X) | (; y), (; X, y),
        nt -> sum(logpdf.(Normal.(nt.alpha .+ X * nt.beta, nt.sigma), y)))

    m = @rkppl begin
        alpha ~ Normal(0, 10)
        beta1 ~ Normal(0, 1)
        beta2 ~ Normal(0, 1)
        beta3 ~ Normal(0, 1)
        b = [beta1, beta2, beta3]
        counts ~ PoissonLogGLM(X, alpha, b)
    end
    built = _gv_check(m(; X) | (; counts), (; X, counts),
        nt -> sum(logpdf.(Poisson.(exp.(nt.alpha .+ X * [nt.beta1, nt.beta2, nt.beta3])),
            counts)))
    @test coordinate_names(built.layout) == [:alpha, :beta1, :beta2, :beta3]

    m = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        flags ~ BernoulliLogitGLM(X, 0, beta)
    end
    _gv_check(m(; X) | (; flags), (; X, flags),
        nt -> sum(logpdf.(Bernoulli.(1 ./ (1 .+ exp.(.-(X * nt.beta)))), flags)))
end

# The Poisson-log object sums its `logkernel` endpoint and adds its
# `lognormalizer`, `-log(y!)`, over the bound counts: preparation folds the
# normalizer, so the prepared program evaluates no `loggamma`. The pointwise
# densities add the two parts, and a sweep that retains them returns them
# beside the same value and gradient.
@testset "PoissonLogGLM folds its log-factorial normalizer" begin
    n = 9
    X, counts = _gv_X(n), _gv_counts(n)
    m = @rkppl begin
        alpha ~ Normal(0, 10)
        beta[axes(X, 2)] .~ Normal.(0, 10)
        counts ~ PoissonLogGLM(X, alpha, beta)
    end
    plan = m(; X) | (; counts)
    oracle = nt -> logpdf.(Poisson.(exp.(nt.alpha .+ X * nt.beta)), counts)
    built = _gv_check(plan, (; X, counts), nt -> sum(oracle(nt)))
    source = sprint(print, kernel_expr(plan, built.layout))
    @test occursin(".logkernel(", source) && occursin(".lognormalizer(", source)
    prepared = prepare_query(built, plan, :sampler)
    code = sprint(print, Base.remove_linenums!(deepcopy(code_expr(prepared))))
    @test !occursin("loggamma", code)
    normalizer = match(r"\b_ppl_lnorm_\w+ = \(__ops__\[(\d+)\]\)\(\)", code)
    @test normalizer !== nothing
    @test prepared.ops[parse(Int, normalizer.captures[1])] isa ReactiveKernels._BoundConstant
    u = _gv_u(built)
    cells = oracle(constrain(built.layout, u))
    @test Base.invokelatest(prepare_query(built, plan, :pointwise), u).counts ≈ cells
    q = prepare_sampler(built, plan, u; backend = _GV_BACKEND)
    retaining = prepare_sampler(built, plan, u; backend = _GV_BACKEND, retain = (:pointwise,))
    value, gradient = sampler_value_and_gradient!(q, similar(u), u)
    rvalue, rgradient, retained = sampler_value_gradient_and_retained!(retaining, similar(u), u)
    @test rvalue ≈ value
    @test rgradient ≈ gradient
    @test retained.pointwise.counts ≈ cells
end

@testset "GLM object arguments are values: scalar data, expressions, inline matrices" begin
    n = 6
    X, y = _gv_X(n), _gv_y(n)
    x1, x2, x3 = X[:, 1], X[:, 2], X[:, 3]
    m = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(X, off, beta, sigma)
    end
    _gv_check(m(; X, off = 0.7) | (; y), (; X, y),
        nt -> sum(logpdf.(Normal.(0.7 .+ X * nt.beta, nt.sigma), y)))

    inline = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Normal(0, 1)
        beta[axes(X, 2)] .~ Normal.(0, 10)
        y ~ NormalIDGLM(X, 2 * a + 1, beta, exp(s))
    end
    named = @rkppl begin
        a ~ Normal(0, 1)
        s ~ Normal(0, 1)
        beta[axes(X, 2)] .~ Normal.(0, 10)
        intercept = 2 * a + 1
        scale = exp(s)
        y ~ NormalIDGLM(X, intercept, beta, scale)
    end
    oracle = nt -> sum(logpdf.(Normal.(2nt.a + 1 .+ X * nt.beta, exp(nt.s)), y))
    bi = _gv_check(inline(; X) | (; y), (; X, y), oracle)
    bn = _gv_check(named(; X) | (; y), (; X, y), oracle)
    # Naming a subexpression changes neither the density nor the coordinates.
    @test coordinate_names(bi.layout) == coordinate_names(bn.layout)

    m = @rkppl begin
        alpha ~ Normal(0, 10)
        beta[1:4] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(hcat(ones(length(x1)), x1, x2, x3), alpha, beta, sigma)
    end
    _gv_check(m(; x1, x2, x3) | (; y), (; x1, x2, x3, y),
        nt -> sum(logpdf.(Normal.(nt.alpha .+ hcat(ones(n), X) * nt.beta, nt.sigma), y)))

    m = @rkppl begin
        a ~ Normal(0, 1)
        alpha ~ Normal(0, 10)
        Z = hcat(x1, exp.(a .* x2))
        beta[axes(Z, 2)] .~ Normal.(0, 10)
        y ~ NormalIDGLM(Z, alpha, beta, 0.8)
    end
    _gv_check(m(; x1, x2) | (; y), (; x1, x2, y),
        nt -> sum(logpdf.(Normal.(nt.alpha .+ hcat(x1, exp.(nt.a .* x2)) * nt.beta, 0.8), y)))

    # Coefficients bound as data are read whole: their length is the
    # matrix's column count, not the observation count.
    m = @rkppl begin
        alpha ~ Normal(0, 10)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(X, alpha, b, sigma)
    end
    b = [0.4, -0.2, 0.1]
    _gv_check(m(; X, b) | (; y), (; X, b, y),
        nt -> sum(logpdf.(Normal.(nt.alpha .+ X * b, nt.sigma), y)))
end

@testset "GLM objects over matrix values keep their row contract" begin
    m = @rkppl begin
        alpha ~ Normal(0, 1)
        beta[axes(X, 2)] .~ Normal.(0, 1)
        y ~ NormalIDGLM(X, alpha, beta, 0.7)
    end
    # A build evaluates another binding with other rows.
    plan5 = m(; X = _gv_X(5)) | (; y = _gv_y(5))
    plan9 = m(; X = _gv_X(9)) | (; y = _gv_y(9))
    built = build_kernel(plan5)
    u = _gv_u(built)
    nt = constrain(built.layout, u)
    q = prepare_query(built, plan9, :likelihood)
    @test Base.invokelatest(q, u) ≈
        sum(logpdf.(Normal.(nt.alpha .+ _gv_X(9) * nt.beta, 0.7), _gv_y(9)))
    pointwise = Base.invokelatest(prepare_query(built, plan9, :pointwise), u)
    @test length(pointwise.y) == 9
    # The design matrix's rows are the observation axis.
    @test_throws ContractValidationError m(; X = _gv_X(5)) | (; y = _gv_y(6))
end

@testset "GLM objects over matrix values compose with evidence" begin
    n = 6
    X, y = _gv_X(n), _gv_y(n)
    lo, hi = 0.0, 0.6
    yc = clamp.(y, lo, hi)
    m = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y ~ censored(NormalIDGLM(X, 0.0, beta, sigma), lo, hi)
    end
    _gv_check(m(; X, lo, hi) | (; y = yc), (; X, y = yc),
        nt -> sum(logpdf.(censored.(Normal.(X * nt.beta, nt.sigma), lo, hi), yc)))
end

@testset "GLM object per-observation intercept" begin
    n = 5
    X, y = _gv_X(n), _gv_y(n)
    m = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        y ~ NormalIDGLM(X, offset, beta, 0.7)
    end
    # Not built yet: the object takes one scalar intercept. A per-observation
    # offset belongs in the location of the ordinary predictor path.
    @test_broken try
        built = build_kernel(m(; X, offset = 0.1 .* (1:n)) | (; y))
        Base.invokelatest(prepare_query(built, m(; X, offset = 0.1 .* (1:n)) | (; y),
            :likelihood), _gv_u(built)) isa Float64
    catch
        false
    end
end

_gv_family(head, eta) = head === :NormalIDGLM ? Normal(eta, 0.7) :
    head === :BernoulliLogitGLM ? Bernoulli(1 / (1 + exp(-eta))) : Poisson(exp(eta))

_gv_glm_ast(head, alpha) = (rhs = Expr(:call, head, :X, alpha, :beta);
    head === :NormalIDGLM && push!(rhs.args, 0.7);
    quote
        a ~ Normal(0, 1)
        beta[axes(X, 2)] .~ Normal.(0, 1)
        y ~ $rhs
    end)

@testset "GLM objects over bound matrices skip missing rows" begin
    X = [0.1 0.2; 10000.0 0.2; -0.3 0.4; 10000.0 -0.2]
    for head in (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM), alpha in (0.0, :a)
        y = head === :NormalIDGLM ? Union{Missing,Float64}[0.2, missing, 0.5, missing] :
            Union{Missing,Int}[1, missing, 0, missing]
        data = (; y, X)
        saved = deepcopy(data)
        bound = bind_data(lower_rkppl(_gv_glm_ast(head, alpha), data;
            conditioned = (:y,)), data)
        built = build_kernel(bound)
        u = unconstrain(built.layout, (; a = 0.2, beta = [0.3, -0.1]))
        sampler = prepare_sampler(built, bound, u; backend = _GV_BACKEND)
        function oracle(v)
            p = constrain(built.layout, v)
            eta = (alpha === :a ? p.a : 0.0) .+ X * p.beta
            logpdf(Normal(), p.a) + sum(logpdf.(Normal(), p.beta)) +
                sum(logpdf(_gv_family(head, eta[i]), y[i]) for i in eachindex(y)
                    if !ismissing(y[i]))
        end
        value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ oracle(u)
        @test gradient ≈ _gv_fd(oracle, u) rtol = 1e-5 atol = 1e-8
        pw = Base.invokelatest(prepare_query(built, bound, :pointwise), u).y
        @test length(pw) == 4
        @test all(iszero, pw[[2, 4]])
        @test isequal(data, saved)
    end
end

@testset "GLM objects over bound matrices read packed rows" begin
    X = hcat(collect(range(-1.0, 1.0; length = 8)), cos.(1:8))
    jobs = [1, 3, 4, 7]
    for head in (:NormalIDGLM, :BernoulliLogitGLM, :PoissonLogGLM), alpha in (0.0, :a)
        y = head === :NormalIDGLM ? [0.2, -0.1, 0.4, 0.3] : [1, 0, 1, 1]
        data = Dict{Symbol,Any}(:y => y, :X => X, :Jobs => jobs)
        plan = lower_rkppl(_gv_glm_ast(head, alpha), collect(keys(data));
            conditioned = (:y,))
        r = ReactiveKernelsPPL._with(only(plan.responses); mi_jobs = :Jobs)
        plan = ReactiveKernelsPPL._with(plan; responses = [r])
        bound = bind_data(plan, data)
        built = build_kernel(bound)
        u = _gv_u(built)
        p = constrain(built.layout, u)
        eta = (alpha === :a ? p.a : 0.0) .+ X[jobs, :] * p.beta
        pw = Base.invokelatest(prepare_query(built, bound, :pointwise), u).y
        @test pw ≈ [logpdf(_gv_family(head, eta[k]), y[k]) for k in eachindex(jobs)]
    end
end

# The Normal-identity object's density calls the owned rule directly,
# whose reverse branch is one X'r product (DK `normal_id_glm_lpdf`): the
# sampler neither builds nor differentiates the per-observation vector, which
# `:pointwise` still returns. Forward and second-order queries go through the
# rule's authored forward branch.
@testset "NormalIDGLM density calls the owned rule directly" begin
    n = 9
    X, y = _gv_X(n), _gv_y(n)
    x1 = X[:, 1]
    zero_intercept = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(X, 0.0, beta, sigma)
    end
    live_design = @rkppl begin
        a ~ Normal(0, 1)
        alpha ~ Normal(0, 10)
        Z = hcat(x1, exp.(a .* x1))
        beta[axes(Z, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y ~ NormalIDGLM(Z, alpha, beta, sigma)
    end
    second_order = SecondOrder(AutoEnzyme(; mode = Enzyme.Reverse),
        AutoEnzyme(; mode = Enzyme.Forward))
    cases = (
        (zero_intercept(; X) | (; y), (; X, y), nt -> X * nt.beta),
        (live_design(; x1) | (; y), (; x1, y),
            nt -> nt.alpha .+ hcat(x1, exp.(nt.a .* x1)) * nt.beta),
    )
    for (plan, data, mean) in cases
        oracle = nt -> logpdf.(Normal.(mean(nt), nt.sigma), y)
        built = _gv_check(plan, data, nt -> sum(oracle(nt)))
        code = sprint(print, Base.remove_linenums!(deepcopy(
            code_expr(prepare_query(built, plan, :sampler)))))
        kernel_code = sprint(print, kernel_expr(plan, built.layout))
        @test occursin("normal_id_glm(", kernel_code)
        @test occursin("normal_id_glm_lpdf(", kernel_code)
        @test !occursin("_ppl_pw_", code)
        u = _gv_u(built)
        cells = oracle(constrain(built.layout, u))
        @test Base.invokelatest(prepare_query(built, plan, :pointwise), u).y ≈ cells
        q = prepare_sampler(built, plan, u; backend = _GV_BACKEND)
        retaining = prepare_sampler(built, plan, u; backend = _GV_BACKEND, retain = (:pointwise,))
        value, gradient = sampler_value_and_gradient!(q, similar(u), u)
        rvalue, rgradient, retained = sampler_value_gradient_and_retained!(retaining, similar(u), u)
        @test rvalue ≈ value
        @test rgradient ≈ gradient
        @test retained.pointwise.y ≈ cells
        # Reverse over forward, against a central difference of the gradient.
        grad(v) = sampler_value_and_gradient!(q, similar(v), v)[2]
        v = [cos(i) for i in eachindex(u)]
        h = 1e-5
        hv_fd = (grad(u .+ h .* v) .- grad(u .- h .* v)) ./ 2h
        qh = prepare_query_ad(prepare_ad_hvp, built, plan, :sampler, second_order, (v,), u)
        hvp_gradient, (hv,) = ad_gradient_and_hvp(qh, (v,), u)
        @test hvp_gradient ≈ gradient
        @test hv ≈ hv_fd rtol = 1e-5 atol = 1e-7
    end
end

# The per-observation spelling of the same regression uses ordinary AD:
# a Gaussian response located at `M * v` (inline or named) or
# `alpha .+ M * v` (bound data matrix, declared coefficient vector,
# sampled intercept; one shared scale)
# has a residual-and-dot total. The explicit NormalIDGLM object retains the
# authored rule. Other locations keep the per-cell plate, and a
# non-positive scale keeps the plate cell's `-Inf`.
@testset "per-observation Normal over a design product uses ordinary AD" begin
    n = 9
    X, y = _gv_X(n), _gv_y(n)
    g = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    product = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        y .~ Normal.(X * beta, sigma)
    end
    named_product = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        mu = X * beta
        y .~ Normal.(mu, sigma)
    end
    intercept = @rkppl begin
        alpha ~ Normal(0, 10)
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Exponential(1)
        mu = alpha .+ X * beta
        y .~ Normal.(mu, sigma)
    end
    fixed_scale = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        y .~ Normal.(X * beta, s)
    end
    flat = @rkppl begin
        beta[axes(X, 2)] .~ Flat()
        sigma ~ restricted(Flat(), 0, Inf)
        mu = X * beta
        y .~ Normal.(mu, sigma)
    end
    cases = (
        (product(; X) | (; y), (; X, y), nt -> X * nt.beta, nt -> nt.sigma),
        (named_product(; X) | (; y), (; X, y), nt -> X * nt.beta, nt -> nt.sigma),
        (intercept(; X) | (; y), (; X, y), nt -> nt.alpha .+ X * nt.beta, nt -> nt.sigma),
        (fixed_scale(; X, s = 2) | (; y), (; X, y), nt -> X * nt.beta, nt -> 2.0),
        (flat(; X) | (; y), (; X, y), nt -> X * nt.beta, nt -> nt.sigma),
    )
    for (plan, data, mean, scale) in cases
        oracle = nt -> logpdf.(Normal.(mean(nt), scale(nt)), y)
        built = _gv_check(plan, data, nt -> sum(oracle(nt)))
        code = sprint(print, kernel_expr(plan, built.layout))
        @test !occursin("normal_id_glm_lpdf(", code)
        @test occursin("_ppl_normal_residual_", code)
        u = _gv_u(built)
        nt = constrain(built.layout, u)
        @test Base.invokelatest(prepare_query(built, plan, :pointwise), u).y ≈ oracle(nt)
        # The per-observation queries still read the plate.
        obs = Base.invokelatest(prepare_query(built, plan, :observations), u).y
        @test obs.location ≈ mean(nt) && all(obs.scale .≈ scale(nt))
        cell = prepare_cell_query(built, plan, :y)
        @test Base.invokelatest(cell, u, 4) ≈ oracle(nt)[4]
    end

    # A real-support scale keeps the plate cell's guard: -Inf at sigma <= 0.
    signed = @rkppl begin
        beta[axes(X, 2)] .~ Normal.(0, 10)
        sigma ~ Normal(0, 1)
        y .~ Normal.(X * beta, sigma)
    end
    plan = signed(; X) | (; y)
    built = build_kernel(plan)
    likelihood = prepare_query(built, plan, :likelihood)
    u = _gv_u(built)
    nt = constrain(built.layout, u)
    k = findfirst(==(:sigma), coordinate_names(built.layout))
    u[k] = 0.7
    @test Base.invokelatest(likelihood, u) ≈
        sum(logpdf.(Normal.(X * constrain(built.layout, u).beta, 0.7), y))
    u[k] = -0.7
    @test Base.invokelatest(likelihood, u) == -Inf

    # A location that is not a data-matrix product keeps the plate.
    grouped = @rkppl begin
        z[levels(g)] .~ Normal.(0, 1)
        sigma ~ Exponential(1)
        y .~ Normal.(z[g], sigma)
    end
    plan = grouped(; g) | (; y)
    built = _gv_check(plan, (; g, y), nt -> sum(logpdf.(Normal.(nt.z[g], nt.sigma), y)))
    @test !occursin("normal_id_glm_lpdf", sprint(print, kernel_expr(plan, built.layout)))
end
