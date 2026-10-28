# Prior-vocabulary expansion (design decision
# `ReactiveKernels:brm:parity-prior-vocab/decisions/2026-09-26T18-50-15-961-1x52bl2`,
# resolved maximal population + recommended sampled + per-addressee mixing;
# scope confirmed by reconciliation `1ljidem`): surface admission for the new
# population/sampled families, Distributions.jl hand-oracle value parity,
# Enzyme-vs-findiff gradients (Reactant/XLA legs: test_prior_vocab_reactant.jl).
# Ordinary declarations
# retain their priors through affine and GLM optimizations.
# SB-parity probes follow the BRM peer's BridgeStan brief
# (phase 2 of their todo). (`_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, included first.)
using Distributions: Normal, Cauchy, TDist, Laplace, Logistic, Uniform,
    Exponential, Bernoulli, logpdf
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Lower + bind + build + query a prior-vocab program; return
# `(bound, built, kern, layout)`.
function _pv_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_pv_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

const _PV_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
const _PV_Y = [1.0, 2.0, 1.5, 2.5, 3.0, 2.0]
const _PV_G = [1, 2, 1, 3, 2, 3]
const _PV_SAMPLED_UNIFORM = quote
    lo ~ Normal(0, 1)
    u ~ Uniform(lo, 2)
    y .~ Normal.(u, 1)
end
_pv_cols() = Dict{Symbol,AbstractVector}(:y => copy(_PV_Y), :x => copy(_PV_X))
_pv_gcols() = Dict{Symbol,AbstractVector}(:y => copy(_PV_Y), :x => copy(_PV_X),
    :g => copy(_PV_G))

# Location-scale Student-t log-density from the standard TDist
# (Distributions.jl has no 3-arg StudentT; the thin layer follows the
# Stan `(nu, mu, sigma)` order).
_pv_student(nu::Real, mu::Real, s::Real, x::Real) =
    logpdf(TDist(nu), (x - mu) / s) - log(s)

# Affine-logistic interval log-Jacobian in the CONSTRAINED value (mirrors
# `interval_bijector`'s math: `log(x-lo) + log(hi-x) - log(hi-lo)`).
_pv_interval_logjac(lo::Real, hi::Real, x::Real) =
    log(x - lo) + log(hi - x) - log(hi - lo)

# Gaussian linear-model log-likelihood over the shared (x, y) fixture.
function _pv_gauss_ll(a::Real, b::Real, s::Real)
    return sum(logpdf(Normal(a + b * x, s), y) for (x, y) in zip(_PV_X, _PV_Y))
end

_pv_param(plan, nm::Symbol) = only(p for p in plan.parameters if p.name === nm)

# Sampling RHS calls resolve in the lowering module (open RHS protocol,
# `cb0b3228`). Spelling refusals lower in a module without kernel-source
# bindings, independent of what earlier test files import into `Main`.
const _PV_UNBOUND = Module(:PriorVocabUnbound)

@testset "prior vocab sampled admission" begin
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            t ~ StudentT(3, 1, 2)
            l ~ Laplace(0, 1.5)
            g ~ Logistic(2, 0.5)
            u ~ Uniform(-1, 2)
            s ~ Exponential(1)
        end, (:y, :x); conditioned = (:y, :x))
    t = _pv_param(plan, :t)
    @test t.family === :student_t
    @test t.args == (arg1 = 3, arg2 = 1, arg3 = 2)
    @test t.support_override === nothing
    l = _pv_param(plan, :l)
    @test l.family === :laplace
    @test l.args == (arg1 = 0, arg2 = 1.5)
    g = _pv_param(plan, :g)
    @test g.family === :logistic
    @test g.args == (arg1 = 2, arg2 = 0.5)
    u = _pv_param(plan, :u)
    @test u.family === :uniform
    @test u.args == (arg1 = -1, arg2 = 2)
    @testset "sampled hyperparameters" begin
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                nu ~ Exponential(1)
                t ~ StudentT(nu, 0, 1)
                s ~ Exponential(1)
            end, (:y, :x); conditioned = (:y, :x))
        t = _pv_param(plan, :t)
        @test t.family === :student_t
        @test t.args == (arg1 = :nu, arg2 = 0, arg3 = 1)
    end
    @testset "constructor spellings" begin
        for (rhs, msg) in ((:(student_t(3, 0, 1)), "`student_t`, which is not defined"),
                (:(laplace(0, 1)), "`laplace`, which is not defined"),
                (:(TDist(3)), nothing))
            err = try
                lower_rkppl(quote
                        a ~ Normal(0, 1)
                        b ~ Normal(0, 1)
                        mu = a .+ b .* x
                        y .~ Normal.(mu, s)
                        t ~ $rhs
                        s ~ Exponential(1)
                    end, (:y, :x); conditioned = (:y, :x), mod = _PV_UNBOUND)
                nothing
            catch e
                e
            end
            if rhs == :(TDist(3))
                # admitted: standard TDist prior constructor (P3; todo `139j2uo`).
                @test (err === nothing || throw(err))
            else
                # refused: a lowercase Stan spelling is no Distributions.jl
                # constructor; unbound, it names an undefined function in the
                # model module (P3, open RHS `cb0b3228`).
                @test err isa SurfaceLoweringError
                @test occursin(msg, sprint(showerror, err))
            end
        end
    end
    @testset "Uniform bounds retain sampled dependencies" begin
        sampled = lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                lo ~ Normal(0, 1)
                u ~ Uniform(lo, 2)
                s ~ Exponential(1)
            end, (:y, :x); conditioned = (:y, :x))
        uniform = _pv_param(sampled, :u)
        @test uniform.family === :uniform
        @test uniform.args == (arg1 = :lo, arg2 = 2)
        @test uniform.support_override === nothing
        # refused: Uniform bounds out of order (malformed distribution)
        for rhs in (:(Uniform(2, -1)), :(Uniform(0, Inf)))
            # refused: battery of malformed Uniform literals (all entries P)
            @test_throws ContractValidationError lower_rkppl(quote
                    a ~ Normal(0, 1)
                    b ~ Normal(0, 1)
                    mu = a .+ b .* x
                    y .~ Normal.(mu, s)
                    u ~ $rhs
                    s ~ Exponential(1)
                end, (:y, :x); conditioned = (:y, :x))
        end
    end
end

@testset "prior vocab plate admission" begin
    # Plate parameters share the sampled distribution grammar.
    _plate_theta(rhs) = Expr(:block,
        :(s ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ $rhs),
                    :(y[i] ~ Normal.(theta[i], 1.0))))))
    plate = lower_rkppl(_plate_theta(:(StudentT(3, 0, 2))), (:y,); conditioned = (:y,))
    pp = only(plate.plate_parameters)
    @test pp.family === :student_t
    @test pp.args == (arg1 = 3, arg2 = 0, arg3 = 2)
    plate = lower_rkppl(_plate_theta(:(Uniform(-1, 2))), (:y,); conditioned = (:y,))
    pp = only(plate.plate_parameters)
    @test pp.family === :uniform
    @test pp.support_override === nothing
    # Per-cell data bounds retain their column dependencies.
    indexed = lower_rkppl(Expr(:block,
        :(s ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ Uniform(lo[i], hi[i])),
                    :(y[i] ~ Normal.(theta[i], 1.0)))))), (:y, :lo, :hi); conditioned = (:y, :lo, :hi))
    theta = only(p for p in vcat(indexed.plate_parameters, indexed.array_parameters)
        if p.name === :theta)
    @test theta.family === :uniform
    # Each bound is the column selected at the authored loop cells (`4333b93b`).
    selections = Dict(d.name => d.expr for d in indexed.derived)
    bounds = [selections[a] for a in values(theta.args)]
    @test all(b -> Meta.isexpr(b, :ref, 2), bounds)
    @test [b.args[1] for b in bounds] == [:lo, :hi]
    @test allequal(b.args[2] for b in bounds)
    @test haskey(selections, first(bounds).args[2])
    @test theta.support_override === nothing
end

@testset "prior vocab truncated halves" begin
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            h ~ truncated(StudentT(3, 0, 2), 0, Inf)
            l ~ truncated(Laplace(0, 1), 0, Inf)
            g ~ truncated(Logistic(0, 1), 0, Inf)
            n ~ truncated(Normal(0, 2), 0, Inf)
            s ~ Exponential(1)
        end, (:y, :x); conditioned = (:y, :x))
    h = _pv_param(plan, :h)
    @test h.family === :student_t
    @test h.args == (arg1 = 3, arg2 = 0, arg3 = 2)
    @test h.support_override === (:truncated, 0.0, Inf)
    @test _pv_param(plan, :l).support_override === (:truncated, 0.0, Inf)
    @test _pv_param(plan, :g).support_override === (:truncated, 0.0, Inf)
    @test _pv_param(plan, :n).support_override === (:truncated, 0.0, Inf)
    @testset "non-zero-location truncation" begin
        @test (lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                h ~ truncated(StudentT(3, 1, 2), 0, Inf)
                s ~ Exponential(1)
            end, (:y, :x); conditioned = (:y, :x)); true)
    end
    @testset "upper-only and finite intervals across families" begin
        @test (lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                h ~ truncated(StudentT(3, 0, 2), -Inf, 1)
                s ~ Exponential(1)
            end, (:y, :x); conditioned = (:y, :x)); true)
        @test (lower_rkppl(quote
                a ~ Normal(0, 1)
                b ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                h ~ truncated(Laplace(0, 1), -1, 1)
                s ~ Exponential(1)
            end, (:y, :x); conditioned = (:y, :x)); true)
    end
end

function _pv_prior(plan, pred::Symbol, addr::Symbol)
    legacy = filter(p -> p.predictor === pred && p.addressee === addr,
        plan.population_priors)
    isempty(legacy) || return only(legacy)
    predictor = only(p for p in plan.predictors if p.name === pred)
    term = only(t for t in predictor.terms if t.addressee === addr)
    name = term.options.parameter
    p = only(p for p in Iterators.flatten((plan.parameters, plan.array_parameters))
        if p.name === name)
    args = collect(values(p.args))
    p.family === :flat && return (family = :flat, location = nothing,
        scale = nothing, nu = nothing)
    li, si, ni = p.family === :student_t ? (2, 3, 1) : (1, 2, 0)
    return (family = p.family, location = args[li], scale = args[si],
        nu = ni == 0 ? nothing : args[ni])
end

@testset "prior vocab coefficient admission" begin
    plan = lower_rkppl(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            a ~ StudentT(4, 0, 2)
            b ~ Laplace(0, 1)
            s ~ Exponential(1)
        end, (:y, :x); conditioned = (:y, :x))
    pa = _pv_prior(plan, :mu, :Intercept)
    @test pa.family === :student_t
    @test (pa.location, pa.scale, pa.nu) == (0.0, 2.0, 4.0)
    pb = _pv_prior(plan, :mu, :x)
    @test pb.family === :laplace
    @test (pb.location, pb.scale) == (0.0, 1.0)
    @testset "cauchy, logistic, and flat coefficients" begin
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                a ~ Cauchy(0, 1)
                b ~ Flat()
                s ~ Exponential(1)
            end, (:y, :x); conditioned = (:y, :x))
        @test _pv_prior(plan, :mu, :Intercept).family === :cauchy
        @test _pv_prior(plan, :mu, :x).family === :flat
        plan = lower_rkppl(quote
                a ~ Normal(0, 1)
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                b ~ Logistic(1, 2)
                s ~ Exponential(1)
            end, (:y, :x); conditioned = (:y, :x))
        @test _pv_prior(plan, :mu, :x).family === :logistic
        @test _pv_prior(plan, :mu, :x).location == 1.0
        # Unstated coefficients keep the Normal(0, 1) default.
        pa = _pv_prior(plan, :mu, :Intercept)
        @test pa.family === :normal
        @test (pa.location, pa.scale) == (0.0, 1.0)
    end
    @testset "factor broadcast families" begin
        plan = lower_rkppl(quote
                c[levels(g)] .~ StudentT.(3, 0, 2)
                mu = c[g] .+ b .* x
                b ~ Normal(0, 1)
                y .~ Normal.(mu, 1.5)
            end, (:y, :g, :x); conditioned = (:y, :g, :x))
        pc = _pv_prior(plan, :mu, :g)
        @test pc.family === :student_t
        @test (pc.location, pc.scale, pc.nu) == (0.0, 2.0, 3.0)
        plan = lower_rkppl(quote
                c[levels(g)] .~ Laplace.(0, 1)
                mu = c[g] .+ o
                y .~ Normal.(mu, 1.5)
            end, (:y, :g, :o); conditioned = (:y, :g, :o))
        @test _pv_prior(plan, :mu, :g).family === :laplace
    end
    @testset "matrix broadcast families" begin
        plan = lower_rkppl(quote
                b[axes(X, 2)] .~ Laplace.(0, 1)
                X = hcat(ones(length(x1)), x1, x2)
                mu = X * b
                y .~ Normal.(mu, 1.0)
            end, (:y, :x1, :x2); conditioned = (:y, :x1, :x2))
        @test isempty(plan.population_priors)
        @test only(plan.array_parameters).name === :b
        @test only(plan.array_parameters).family === :laplace
    end
    @testset "lowercase coefficient spelling rejected" begin
        err = try
            lower_rkppl(quote
                    a ~ Normal(0, 1)
                    mu = a .+ b .* x
                    y .~ Normal.(mu, s)
                    b ~ student_t(3, 0, 1)
                    s ~ Exponential(1)
                end, (:y, :x); conditioned = (:y, :x), mod = _PV_UNBOUND)
            nothing
        catch e
            e
        end
        # refused: lowercase student_t is no Distributions.jl constructor;
        # unbound, it names an undefined function in the model module (P3,
        # open RHS `cb0b3228`).
        @test err isa SurfaceLoweringError
        @test occursin("`student_t`, which is not defined", sprint(showerror, err))
    end
end

@testset "prior vocab block homogeneity" begin
    # One family per wide (data-width) block: hand-mutated matrix-element
    # and GLM-vector rows fail validation (the surface can only state one
    # broadcast family, so only hand plans reach this).
    plan = StructuralPlan(
        [LikelihoodSpec(GaussianFam, IdentityLink, :y, :mu, 1.0, nothing,
            _none_evidence(), :y_resp)],
        [PredictorSpec(:mu, IdentityLink,
            [TermSpec(MatrixTerm, [:x1, :x2], (matrix=:X,), :X, :X_term)], :mu)],
        [PopulationPrior(:mu, name, 0.0, 1.0) for name in (:Intercept, :x1, :x2)],
        SampledParameter[], AssignmentSpec[], Dict{Symbol,AbstractVector}(), 0;
        matrices = [DesignMatrix(:X, Union{Nothing,Symbol}[nothing, :x1, :x2], :X)])
    i = findfirst(p -> p.predictor === :mu && p.addressee === :x1,
        plan.population_priors)
    plan.population_priors[i] = PopulationPrior(:mu, :x1, :student_t,
        0.0, 1.0, 3.0)
    # refused: one prior family per data-width coefficient block (IR contract)
    @test_throws ContractValidationError validate_structure(plan)
    @testset "glm-object beta vectors retain ordinary priors" begin
        plan = lower_rkppl(quote
                    X = hcat(x1, x2)
                    y ~ NormalIDGLM(X, alpha, beta, 1.0)
                    alpha ~ Normal(0, 10)
                    beta[axes(X, 2)] .~ Laplace.(0, 1)
                end, (:y, :x1, :x2); conditioned = (:y, :x1, :x2))
        @test only(plan.array_parameters).name === :beta
        @test only(plan.array_parameters).family === :laplace
        @test isempty(plan.population_priors)
    end
end

# M1: mixed population priors (StudentT intercept, Laplace slope).
const _PV_M1 = quote
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    a ~ StudentT(4, 0, 2)
    b ~ Laplace(0, 1)
    s ~ Exponential(1)
end
_pv_m1_oracle(a::Real, b::Real, s::Real) =
    _pv_student(4, 0, 2, a) + logpdf(Laplace(0, 1), b) +
    logpdf(Exponential(1), s) + _pv_gauss_ll(a, b, s) + log(s)

# M2: new sampled families as pure prior contributors alongside ordinary
# Normal declarations for the intercept and slope.
const _PV_M2 = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    t ~ StudentT(3, 1, 2)
    l ~ Laplace(0, 1.5)
    g ~ Logistic(2, 0.5)
    s ~ Exponential(1)
end
_pv_m2_oracle(a::Real, b::Real, t::Real, l::Real, g::Real, s::Real) =
    logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
    _pv_student(3, 1, 2, t) + logpdf(Laplace(0, 1.5), l) +
    logpdf(Logistic(2, 0.5), g) + logpdf(Exponential(1), s) +
    _pv_gauss_ll(a, b, s) + log(s)

# M3: interval-constrained Uniform plus a half-StudentT (exact +log(2)).
const _PV_M3 = quote
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
    u ~ Uniform(-1, 2)
    h ~ truncated(StudentT(3, 0, 2), 0, Inf)
    s ~ Exponential(1)
end
_pv_m3_oracle(a::Real, b::Real, u::Real, h::Real, s::Real) =
    logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
    logpdf(Uniform(-1, 2), u) + _pv_student(3, 0, 2, h) + log(2) +
    logpdf(Exponential(1), s) + _pv_gauss_ll(a, b, s) +
    _pv_interval_logjac(-1, 2, u) + log(h) + log(s)

# M4: mixed predictor with a wide homogeneous factor plate (StudentT
# levels) plus a Cauchy width-1 block.
const _PV_M4 = quote
    c[levels(g)] .~ StudentT.(3, 0, 2)
    mu = c[g] .+ b .* x
    b ~ Cauchy(0, 1)
    y .~ Normal.(mu, 1.5)
end
function _pv_m4_oracle(c::AbstractVector, b::Real)
    pr = sum(_pv_student(3, 0, 2, ci) for ci in c) +
        logpdf(Cauchy(0, 1), b)
    ll = sum(logpdf(Normal(c[gi] + b * x, 1.5), y)
        for (gi, x, y) in zip(_PV_G, _PV_X, _PV_Y))
    return pr + ll
end

@testset "prior vocab values vs Distributions oracles" begin
    _, _, kern, lay = _pv_query(_PV_M1, _pv_cols())
    @test _pv_posterior(kern, lay, (a = 0.5, b = -0.25, s = 1.3)) ≈
        _pv_m1_oracle(0.5, -0.25, 1.3) rtol = 1e-12
    _, _, kern, lay = _pv_query(_PV_M2, _pv_cols())
    q = (a = 0.25, b = 0.5, t = 1.5, l = -0.5, g = 2.25, s = 0.8)
    @test _pv_posterior(kern, lay, q) ≈
        _pv_m2_oracle(0.25, 0.5, 1.5, -0.5, 2.25, 0.8) rtol = 1e-12
    _, _, kern, lay = _pv_query(_PV_M3, _pv_cols())
    q = (a = -0.5, b = 1.25, u = 0.5, h = 1.1, s = 2.0)
    @test _pv_posterior(kern, lay, q) ≈
        _pv_m3_oracle(-0.5, 1.25, 0.5, 1.1, 2.0) rtol = 1e-12
    _, _, kern, lay = _pv_query(_PV_M4, _pv_gcols())
    q = (c = [0.3, -0.4, 0.1], b = 0.75,)
    @test _pv_posterior(kern, lay, q) ≈
        _pv_m4_oracle([0.3, -0.4, 0.1], 0.75) rtol = 1e-12
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _pv_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _pv_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "prior vocab Enzyme gradients" begin
    _pv_enzyme_check(_PV_M1, _pv_cols(), (a = 0.5, b = -0.25, s = 1.3))
    _pv_enzyme_check(_PV_M2, _pv_cols(),
        (a = 0.25, b = 0.5, t = 1.5, l = -0.5, g = 2.25, s = 0.8))
    _pv_enzyme_check(_PV_M3, _pv_cols(),
        (a = -0.5, b = 1.25, u = 0.5, h = 1.1, s = 2.0))
    _pv_enzyme_check(_PV_M4, _pv_gcols(), (c = [0.3, -0.4, 0.1], b = 0.75,))
end

@testset "sampled Uniform bounds density and gradients" begin
    cols = Dict{Symbol,AbstractVector}(:y => [0.2, -0.4, 0.7])
    for q in ((lo = -0.4, u = 0.3), (lo = 0.6, u = 1.2))
        _, _, kernel, layout = _pv_query(_PV_SAMPLED_UNIFORM, cols)
        expected = logpdf(Normal(), q.lo) + logpdf(Uniform(q.lo, 2), q.u) +
            sum(logpdf.(Normal(q.u, 1), cols[:y])) +
            _pv_interval_logjac(q.lo, 2, q.u)
        @test _pv_posterior(kernel, layout, q) ≈ expected rtol = 1e-12
        _pv_enzyme_check(_PV_SAMPLED_UNIFORM, cols, q)
    end
end

@testset "prior vocab SB parity" begin
    # Peer literals: `BayesianRegressionModels:rk:parity-prior-vocab`
    # brief `2026-09-26T22-18-21-872-8yw5i7` (BRM `ff5e589`, SB `24578c3`,
    # BridgeStan 2.9.0; `propto=false`, `jacobian=true`; SB-vs-oracle
    # ~1e-15, central-diff ≤7e-10). RK lane `fade3b9`. Conventions:
    # full posterior with constants, log-Jacobian included.
    _pv_sb_check(prog, cols, q, sbv, sbg; names = [:a, :b, :s]) = begin
        _, _, kern, lay = _pv_query(prog, cols)
        @test _pv_posterior(kern, lay, q) ≈ sbv rtol = 1e-12
        g = _pv_enzyme_check(prog, cols, q)
        coords = coordinate_names(lay)
        @test [g[only(findall(isequal(name), coords))] for name in names] ≈
            sbg rtol = 1e-9
    end
    # P1: StudentT intercept + Laplace slope.
    _pv_sb_check(_PV_M1, _pv_cols(), (a = 0.5, b = -0.25, s = 1.3),
        -15.676861749966013,
        [5.393491124260353, 1.998520710059171, 3.491050295857985])
    # P2: Cauchy intercept + Flat slope (Flat is exactly 0.0 both sides).
    _pv_sb_check(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            a ~ Cauchy(0, 1)
            b ~ Flat()
            s ~ Exponential(1)
        end, _pv_cols(), (a = 0.5, b = -0.25, s = 1.3),
        -14.388851106658095,
        [4.7473372781065075, 0.9985207100591711, 3.491050295857985])
    # P3: factor StudentT broadcast + Cauchy slope, fixed s = 1.5 (SB
    # native order is [slope, cats]; the literal below is already in RK
    # [c1, c2, c3, b] order).
    _pv_sb_check(_PV_M4, _pv_gcols(), (c = [0.3, -0.4, 0.1], b = 0.75,),
        -21.633064049357102,
        [0.07852219465122685, 3.2093567251461983, 1.5444721990933479,
            -2.565555555555555]; names = [Symbol("c.1"), Symbol("c.2"), Symbol("c.3"), :b])
    # P4: Uniform(0.5, 1.5) response scale, default Normal population
    # priors (SB `real<0.5,1.5>` is the same affine-logit leg).
    _pv_sb_check(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            s ~ Uniform(0.5, 1.5)
        end, _pv_cols(), (a = 0.5, b = -0.25, s = 1.3),
        -15.810050464119632,
        [5.047337278106507, 1.248520710059171, -0.1334091943559405])
    # P5: half-StudentT(4, 0, 1) scale — SB's `truncated(...; lower=0)`
    # is the renormalized truncated distribution, matching RK
    # `:positive` (exact +log(2)).
    _pv_sb_check(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            s ~ truncated(StudentT(4, 0, 1), 0, Inf)
        end, _pv_cols(), (a = 0.5, b = -0.25, s = 1.3),
        -14.883826525901483,
        [5.047337278106507, 1.248520710059171, 3.305988784434435])
    # Refused: legacy IR overrides violate the normalized support contract
    # (user decision stan-halves, 0m1j3iz).
    plan = lower_rkppl(quote s ~ HalfNormal(1); y .~ Normal.(s,1) end,
        (:y,); conditioned=(:y,))
    p = only(plan.parameters)
    plan.parameters[1] = SampledParameter(p.name,p.family,p.args,:positive_stan,p.label)
    @test_throws ContractValidationError validate_structure(plan)

end

# Normalized sampled halves, migrated from the retired support keywords.
const _PV_M5 = quote
    a ~ Normal(0,1)
    b ~ Normal(0,1)
    s ~ HalfNormal(2)
    t ~ HalfCauchy(5)
    mu = a .+ b .* x
    y .~ Normal.(mu,s)
end
@testset "normalized halves values and gradients" begin
    q = (a=0.5,b=-0.25,s=1.3,t=2.1)
    _,_,kern,lay = _pv_query(_PV_M5,_pv_cols())
    expected = logpdf(Normal(),q.a) + logpdf(Normal(),q.b) +
        logpdf(truncated(Normal(0,2),0,Inf),q.s) +
        logpdf(truncated(Cauchy(0,5),0,Inf),q.t) +
        _pv_gauss_ll(q.a,q.b,q.s) + log(q.s) + log(q.t)
    @test _pv_posterior(kern,lay,q) ≈ expected rtol=1e-12
    _pv_enzyme_check(_PV_M5,_pv_cols(),q)
end

# M8: centered hierarchical factor prior — `c[levels(g)] .~
# Normal.(mu_alpha, sigma_alpha)` with sampled hypers, plus a literal
# continuous term (the mixed-predictor shape: hyper plate over the
# factor block, scalar node over the slope).
const _PV_M8 = quote
    mu_alpha ~ Normal(0, 10)
    sigma_alpha ~ Exponential(1)
    s ~ Exponential(1)
    c[levels(g)] .~ Normal.(mu_alpha, sigma_alpha)
    b ~ Normal(0, 10)
    mu = c[g] .+ b .* x
    y .~ Normal.(mu, s)
end
function _pv_m8_oracle(c::AbstractVector, b::Real, mua::Real, saa::Real,
        s::Real)
    pr = logpdf(Normal(0, 10), mua) + logpdf(Exponential(1), saa) +
        logpdf(Exponential(1), s) + logpdf(Normal(0, 10), b) +
        sum(logpdf(Normal(mua, saa), cj) for cj in c)
    ll = sum(logpdf(Normal(c[gi] + b * x, s), y)
        for (gi, x, y) in zip(_PV_G, _PV_X, _PV_Y))
    return pr + ll + log(saa) + log(s)
end
_pv_m8_q() = (c = [0.3, -0.4, 0.1], b = 0.75, mu_alpha = 0.5,
    sigma_alpha = 1.3, s = 1.1)

@testset "centered factor priors admission" begin
    plan = lower_rkppl(_PV_M8, (:y, :x, :g); conditioned = (:y, :x, :g))
    row = _pv_prior(plan, :mu, :g)
    @test row.family === :normal
    @test row.location === :mu_alpha
    @test row.scale === :sigma_alpha
    @testset "ordinary array arguments and preparation validation" begin
        for rhs in (:(StudentT.(ndf, 0, 2)), :(Normal.([0, 0, 0], 2)))
            admitted = lower_rkppl(quote
                    ndf ~ Exponential(1)
                    c[levels(g)] .~ $rhs
                    b ~ Normal(0, 10)
                    mu = c[g] .+ b .* x
                    y .~ Normal.(mu, 1.5)
                end, (:y, :x, :g); conditioned = (:y, :x, :g))
            @test only(admitted.array_parameters).name === :c
            @test bind_data(admitted, _pv_gcols()) isa StructuralPlan
        end
        for rhs in (:(Normal.(nope, 2)), :(Normal.(x, 2)))
            prepared = lower_rkppl(quote
                    c[levels(g)] .~ $rhs
                    b ~ Normal(0, 10)
                    mu = c[g] .+ b .* x
                    y .~ Normal.(mu, 1.5)
                end, (:y, :x, :g); conditioned = (:y, :x, :g))
            # refused: unknown nope or 6 location values for 3 levels (name/shape contract, P3/P6).
            @test_throws ContractValidationError bind_data(prepared, _pv_gcols())
        end
        prepared = lower_rkppl(quote
                ly = log.(earn)
                c[levels(g)] .~ Normal.(ly, 2)
                b ~ Normal(0, 10)
                mu = c[g] .+ b .* x
                ly .~ Normal.(mu, 1.5)
            end, (:earn, :x, :g); conditioned = (:earn, :x, :g))
        # refused: 6 derived location values cannot broadcast over 3 levels (P3).
        @test_throws ContractValidationError bind_data(prepared,
            Dict(:earn => exp.(_PV_Y), :x => copy(_PV_X), :g => copy(_PV_G)))
    end
    # Expression arguments, scalar-coefficient hyper names, sign-flipped
    # hierarchical locations, a coefficient another prior reads, and
    # assignment scales all lower (coef_grammar A; oracles in
    # test_fallback.jl).
    @testset "widened coefficient-prior grammar admits" begin
        mk(rhs, use = :(c[g])) = quote
            mu_alpha ~ Normal(0, 10)
            sigma_alpha ~ Exponential(1)
            s ~ Exponential(1)
            sc = 1.0
            c[levels(g)] .~ $rhs
            mu = $use .+ b .* x
            b ~ Normal(0, 10)
            y .~ Normal.(mu, s)
        end
        computed = lower_rkppl(mk(:(Normal.(mu_alpha + 0, 2))), (:y, :x, :g); conditioned = (:y, :x, :g))
        row = _pv_prior(computed, :mu, :g)
        @test row.location === :c_location
        @test any(a -> a.name === :c_location, computed.assignments)
        flipped = lower_rkppl(mk(:(Normal.(mu_alpha, sigma_alpha)),
            :(-c[g])), (:y, :x, :g); conditioned = (:y, :x, :g))
        row = _pv_prior(flipped, :mu, :g)
        @test row.location === :mu_alpha
        @test isempty(filter(a -> a.name === :_rkppl_neg_mu_alpha, flipped.assignments))
        assigned = lower_rkppl(mk(:(Normal.(mu_alpha, sc))), (:y, :x, :g); conditioned = (:y, :x, :g))
        row = _pv_prior(assigned, :mu, :g)
        @test row.scale === :sc
        coefread = lower_rkppl(mk(:(Normal.(b, sigma_alpha))), (:y, :x, :g); conditioned = (:y, :x, :g))
        @test any(p -> p.name === :b, coefread.parameters)
        scalar = lower_rkppl(quote
                mu_alpha ~ Normal(0, 10)
                s ~ Exponential(1)
                a ~ Normal(0, 1)
                mu = a .+ b .* x
                b ~ Normal(mu_alpha, 2)
                y .~ Normal.(mu, s)
            end, (:y, :x); conditioned = (:y, :x))
        row = _pv_prior(scalar, :mu, :x)
        @test row.location === :mu_alpha
    end
    @testset "ordinary prior arguments retain scalar declarations" begin
        plan = lower_rkppl(quote
                mu_alpha ~ Normal(0, 10)
                mu_beta ~ Normal(0, 10)
                c[levels(g)] .~ Normal.(mu_alpha, mu_beta)
                b ~ Normal(0, 10)
                mu = c[g] .+ b .* x
                y .~ Normal.(mu, 1.5)
            end, (:y, :x, :g); conditioned = (:y, :x, :g))
        @test only(plan.array_parameters).args.arg2 === :mu_beta
        @test _pv_param(plan, :mu_beta).family === :normal
    end
end

@testset "centered factor priors interval scale hypers" begin
    # The array prior reads the scalar value; its declaration's support
    # remains independent of affine recognition.
    for hyper in (:(Uniform(0, 100)), :(Uniform(-1, 100)))
        plan = lower_rkppl(quote
                mu_alpha ~ Normal(0, 10)
                sigma_alpha ~ $hyper
                s ~ Exponential(1)
                c[levels(g)] .~ Normal.(mu_alpha, sigma_alpha)
                mu = c[g] .+ b .* x
                b ~ Normal(0, 10)
                y .~ Normal.(mu, s)
            end, (:y, :x, :g); conditioned = (:y, :x, :g))
        row = _pv_prior(plan, :mu, :g)
        @test row.scale === :sigma_alpha
        @test bind_data(plan, _pv_gcols()) isa StructuralPlan
    end
end

@testset "centered factor priors values vs oracles" begin
    _, _, kern, lay = _pv_query(_PV_M8, _pv_gcols())
    @test _pv_posterior(kern, lay, _pv_m8_q()) ≈
        _pv_m8_oracle([0.3, -0.4, 0.1], 0.75, 0.5, 1.3, 1.1) rtol = 1e-12
end

@testset "centered factor priors Enzyme gradients" begin
    _pv_enzyme_check(_PV_M8, _pv_gcols(), _pv_m8_q())
end

# M9: mixed flat intercept + Normal slope — regression for the mixed
# scalar offset bug (a flat `continue` skipped the design-position
# accumulation, so every later prior read the wrong coefficient).
const _PV_M9 = quote
    a ~ Flat()
    b ~ Normal(0, 2)
    s ~ Exponential(1)
    mu = a .+ b .* x
    y .~ Normal.(mu, s)
end
_pv_m9_oracle(a::Real, b::Real, s::Real) =
    logpdf(Normal(0, 2), b) + logpdf(Exponential(1), s) +
    _pv_gauss_ll(a, b, s) + log(s)

@testset "mixed flat scalar offset values vs oracles" begin
    _, _, kern, lay = _pv_query(_PV_M9, _pv_cols())
    # a ≠ b, so a misaligned prior read cannot hide.
    @test _pv_posterior(kern, lay, (a = 0.5, b = -0.25, s = 1.3)) ≈
        _pv_m9_oracle(0.5, -0.25, 1.3) rtol = 1e-12
end

@testset "mixed flat scalar offset Enzyme gradients" begin
    _pv_enzyme_check(_PV_M9, _pv_cols(), (a = 0.5, b = -0.25, s = 1.3))
end

# M10: uniform coefficients (the dogs_log shape) — flat intercept +
# two interval slopes. The layout splits the block into transform runs
# and the generator reassembles it; priors are -log(width) per coef.
const _PV_M10 = quote
    a ~ Flat()
    b1 ~ Uniform(-100, 0)
    b2 ~ Uniform(0, 100)
    mu = a .+ b1 .* x .+ b2 .* z
    y .~ BernoulliLogit.(mu)
end
const _PV_Z = [3.0, 2.0, 1.0, 0.5, 1.5, 2.5]
const _PV_YB = [true, false, true, true, false, true]
_pv_m10_cols() = Dict{Symbol,AbstractVector}(
    :y => copy(_PV_YB), :x => copy(_PV_X), :z => copy(_PV_Z))
function _pv_m10_oracle(a::Real, b1::Real, b2::Real)
    pr = -log(100.0) - log(100.0)
    eta = a .+ b1 .* _PV_X .+ b2 .* _PV_Z
    ll = sum(logpdf.(Bernoulli.(1 ./ (1 .+ exp.(-eta))), _PV_YB))
    jac = _pv_interval_logjac(-100, 0, b1) + _pv_interval_logjac(0, 100, b2)
    return pr + ll + jac
end
_pv_m10_q() = (a = 0.5, b1 = -1.0, b2 = 2.0,)

@testset "uniform coefficients admission" begin
    plan = lower_rkppl(_PV_M10, (:y, :x, :z); conditioned = (:y, :x, :z))
    rows = Dict(addr => _pv_prior(plan, :mu, addr) for addr in (:Intercept, :x, :z))
    @test rows[:x].family === :uniform
    @test (rows[:x].location, rows[:x].scale) == (-100.0, 0.0)
    @test rows[:z].family === :uniform
    @test (rows[:z].location, rows[:z].scale) == (0.0, 100.0)
    bound = bind_data(plan, _pv_m10_cols())
    lay = assign_layout(bound)
    coefs = [e for e in lay.entries if e.kind === :sampled]
    @test length(coefs) == 3
    @test [e.transform for e in coefs] == [:identity, :interval, :interval]
    @test [(e.lo, e.hi) for e in coefs[2:3]] ==
        [(-100.0, 0.0), (0.0, 100.0)]
    @test coordinate_names(lay) ==
        [:a, :b1, :b2]
    # Uniform factor broadcast lowers with literal bounds.
    fplan = lower_rkppl(quote
            c[levels(g)] .~ Uniform.(0, 10)
            mu = c[g]
            y .~ Normal.(mu, 1.5)
        end, (:y, :g); conditioned = (:y, :g))
    frow = _pv_prior(fplan, :mu, :g)
    @test frow.family === :uniform
    @test (frow.location, frow.scale) == (0.0, 10.0)
    # A sampled bound keeps the coefficient as an ordinary parameter.
    sampled = lower_rkppl(quote
            a ~ Flat()
            lo ~ Normal(0, 1)
            b ~ Uniform(lo, 3)
            mu = a .+ b .* x
            y .~ Normal.(mu, 1.5)
        end, (:y, :x); conditioned = (:y, :x))
    b = _pv_param(sampled, :b)
    @test b.family === :uniform
    @test b.args == (arg1 = :lo, arg2 = 3)
    @test b.support_override === nothing
    @testset "bounds gate fails closed" begin
        scalar_cases = (
            ("scalar inverted",
                :(Uniform(10, 5)),
                ContractValidationError,
                "lower < upper"),
            ("scalar infinite",
                :(Uniform(0, Inf)),
                ContractValidationError,
                "bounds must be finite values or declared names"),
        )
        for (label, rhs, ex, msg) in scalar_cases
            err = try
                lower_rkppl(quote
                        a ~ Flat()
                        lo ~ Normal(0, 1)
                        b ~ $rhs
                        mu = a .+ b .* x
                        y .~ Normal.(mu, 1.5)
                    end, (:y, :x); conditioned = (:y, :x))
                nothing
            catch e
                e
            end
            # refused: malformed distribution bounds (P3/P6, 05oe96l).
            @test err isa ex
            @test occursin(msg, sprint(showerror, err))
        end
        broad_cases = (
            ("broadcast hyper bound",
                :(Uniform.(lo, 3)),
                ContractValidationError,
                "bounds are literals"),
            ("broadcast inverted",
                :(Uniform.(10, 5)),
                ContractValidationError,
                "lo < hi"),
        )
        for (label, rhs, ex, msg) in broad_cases
            err = try
                lower_rkppl(quote
                        lo ~ Normal(0, 1)
                        c[levels(g)] .~ $rhs
                        mu = c[g]
                        y .~ Normal.(mu, 1.5)
                    end, (:y, :g); conditioned = (:y, :g))
                nothing
            catch e
                e
            end
            if occursin("hyper bound", label)
                # capability: sampled Uniform bounds (P8 1cmodra; todo `0fkd9yk`).
                @test err === nothing
            else
                # refused: remaining entries violate constructor signature,
                # strict declarations or distribution domains (P3/P6, 05oe96l).
                @test err isa ex
                @test occursin(msg, sprint(showerror, err))
            end
        end
    end
end

@testset "uniform coefficients values vs oracles" begin
    _, _, kern, lay = _pv_query(_PV_M10, _pv_m10_cols())
    @test _pv_posterior(kern, lay, _pv_m10_q()) ≈
        _pv_m10_oracle(0.5, -1.0, 2.0) rtol = 1e-12
end

@testset "uniform coefficients Enzyme gradients" begin
    _pv_enzyme_check(_PV_M10, _pv_m10_cols(), _pv_m10_q())
end
