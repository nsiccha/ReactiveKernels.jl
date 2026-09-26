# Prior-vocabulary expansion (design decision
# `ReactiveKernels:brm:parity-prior-vocab/decisions/2026-09-26T18-50-15-961-1x52bl2`,
# resolved maximal population + recommended sampled + per-addressee mixing;
# scope confirmed by reconciliation `1ljidem`): surface admission for the new
# population/sampled families, Distributions.jl hand-oracle value parity,
# Enzyme-vs-findiff gradients, Reactant/XLA legs. Pair contract: GLM-object
# beta vectors stay Normal-only (non-Normal betas use the decomposed
# predictor path). SB-parity probes follow the BRM peer's BridgeStan brief
# (phase 2 of their todo). (`_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, included first.)
using Distributions: Normal, Cauchy, TDist, Laplace, Logistic, Uniform,
    Exponential, logpdf
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a prior-vocab program; return
# `(bound, built, kern, layout)`.
function _pv_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
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

@testset "prior vocab sampled admission" begin
    plan = lower_rkppl(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            t ~ StudentT(3, 1, 2)
            l ~ Laplace(0, 1.5)
            g ~ Logistic(2, 0.5)
            u ~ Uniform(-1, 2)
            s ~ Exponential(1)
        end, (:y, :x))
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
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                nu ~ Exponential(1)
                t ~ StudentT(nu, 0, 1)
            end, (:y, :x))
        t = _pv_param(plan, :t)
        @test t.family === :student_t
        @test t.args == (arg1 = :nu, arg2 = 0, arg3 = 1)
    end
    @testset "lowercase and TDist spellings rejected" begin
        for (rhs, msg) in ((:(student_t(3, 0, 1)), "Normal"),
                (:(laplace(0, 1)), "Normal"),
                (:(TDist(3)), "unknown distribution"))
            err = try
                lower_rkppl(quote
                        mu = a .+ b .* x
                        y .~ Normal.(mu, s)
                        t ~ $rhs
                        s ~ Exponential(1)
                    end, (:y, :x))
                nothing
            catch e
                e
            end
            @test err isa SurfaceLoweringError
            @test occursin(msg, sprint(showerror, err))
        end
    end
    @testset "uniform bounds must be finite literals in order" begin
        # Sampled bound (a known parameter name — the literal-only rule).
        @test_throws ContractValidationError lower_rkppl(quote
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                lo ~ Normal(0, 1)
                u ~ Uniform(lo, 2)
                s ~ Exponential(1)
            end, (:y, :x))
        for rhs in (:(Uniform(2, -1)), :(Uniform(0, Inf)))
            @test_throws ContractValidationError lower_rkppl(quote
                    mu = a .+ b .* x
                    y .~ Normal.(mu, s)
                    u ~ $rhs
                    s ~ Exponential(1)
                end, (:y, :x))
        end
    end
end

@testset "prior vocab plate admission" begin
    # Plate parameters share the sampled grammar: new families (and the
    # uniform literal-bounds rule) ride the same tables.
    _plate_theta(rhs) = Expr(:block,
        :(s ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ $rhs),
                    :(y[i] ~ Normal.(theta[i], 1.0))))))
    plate = lower_rkppl(_plate_theta(:(StudentT(3, 0, 2))), (:y,))
    pp = only(plate.plate_parameters)
    @test pp.family === :student_t
    @test pp.args == (arg1 = 3, arg2 = 0, arg3 = 2)
    plate = lower_rkppl(_plate_theta(:(Uniform(-1, 2))), (:y,))
    pp = only(plate.plate_parameters)
    @test pp.family === :uniform
    @test pp.support_override === nothing
    # Per-cell (column) uniform bounds are rejected — one plate entry
    # shares one interval transform, so bounds stay finite literals.
    @test_throws ContractValidationError lower_rkppl(Expr(:block,
        :(s ~ Exponential(1)),
        Expr(:macrocall, Symbol("@plate"), LineNumberNode(2),
            Expr(:for, Expr(:(=), :i, :(eachindex(y))),
                Expr(:block,
                    :(theta[i] ~ Uniform(lo[i], hi[i])),
                    :(y[i] ~ Normal.(theta[i], 1.0)))))), (:y, :lo, :hi))
end

@testset "prior vocab truncated halves" begin
    plan = lower_rkppl(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            h ~ truncated(StudentT(3, 0, 2), 0, Inf)
            l ~ truncated(Laplace(0, 1), 0, Inf)
            g ~ truncated(Logistic(0, 1), 0, Inf)
            n ~ truncated(Normal(0, 2), 0, Inf)
            s ~ Exponential(1)
        end, (:y, :x))
    h = _pv_param(plan, :h)
    @test h.family === :student_t
    @test h.args == (arg1 = 3, arg2 = 0, arg3 = 2)
    @test h.support_override === :positive
    @test _pv_param(plan, :l).support_override === :positive
    @test _pv_param(plan, :g).support_override === :positive
    @test _pv_param(plan, :n).support_override === :positive
    @testset "non-zero-location half rejected" begin
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                h ~ truncated(StudentT(3, 1, 2), 0, Inf)
                s ~ Exponential(1)
            end, (:y, :x))
    end
    @testset "upper-only and finite intervals stay Normal-only" begin
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                h ~ truncated(StudentT(3, 0, 2), -Inf, 1)
                s ~ Exponential(1)
            end, (:y, :x))
        @test_throws SurfaceLoweringError lower_rkppl(quote
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                h ~ truncated(Laplace(0, 1), -1, 1)
                s ~ Exponential(1)
            end, (:y, :x))
    end
end

_pv_prior(plan, pred::Symbol, addr::Symbol) =
    only(p for p in plan.population_priors if p.predictor === pred && p.addressee === addr)

@testset "prior vocab coefficient admission" begin
    plan = lower_rkppl(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            a ~ StudentT(4, 0, 2)
            b ~ Laplace(0, 1)
            s ~ Exponential(1)
        end, (:y, :x))
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
            end, (:y, :x))
        @test _pv_prior(plan, :mu, :Intercept).family === :cauchy
        @test _pv_prior(plan, :mu, :x).family === :flat
        plan = lower_rkppl(quote
                mu = a .+ b .* x
                y .~ Normal.(mu, s)
                b ~ Logistic(1, 2)
                s ~ Exponential(1)
            end, (:y, :x))
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
            end, (:y, :g, :x))
        pc = _pv_prior(plan, :mu, :g)
        @test pc.family === :student_t
        @test (pc.location, pc.scale, pc.nu) == (0.0, 2.0, 3.0)
        plan = lower_rkppl(quote
                c[levels(g)] .~ Laplace.(0, 1)
                mu = c[g] .+ o
                y .~ Normal.(mu, 1.5)
            end, (:y, :g, :o))
        @test _pv_prior(plan, :mu, :g).family === :laplace
    end
    @testset "matrix broadcast families" begin
        plan = lower_rkppl(quote
                b[axes(X, 2)] .~ Laplace.(0, 1)
                X = hcat(1, x1, x2)
                mu = X * b
                y .~ Normal.(mu, 1.0)
            end, (:y, :x1, :x2))
        rows = [p for p in plan.population_priors if p.predictor === :mu]
        @test length(rows) == 3
        @test all(p -> p.family === :laplace, rows)
    end
    @testset "lowercase coefficient spelling rejected" begin
        err = try
            lower_rkppl(quote
                    mu = a .+ b .* x
                    y .~ Normal.(mu, s)
                    b ~ student_t(3, 0, 1)
                    s ~ Exponential(1)
                end, (:y, :x))
            nothing
        catch e
            e
        end
        @test err isa SurfaceLoweringError
    end
end

@testset "prior vocab block homogeneity" begin
    # One family per wide (data-width) block: hand-mutated matrix-element
    # and GLM-vector rows fail validation (the surface can only state one
    # broadcast family, so only hand plans reach this).
    plan = lower_rkppl(quote
            b[axes(X, 2)] .~ Normal.(0, 1)
            X = hcat(1, x1, x2)
            mu = X * b
            y .~ Normal.(mu, 1.0)
        end, (:y, :x1, :x2))
    i = findfirst(p -> p.predictor === :mu && p.addressee === :x1,
        plan.population_priors)
    plan.population_priors[i] = PopulationPrior(:mu, :x1, :student_t,
        0.0, 1.0, 3.0)
    @test_throws ContractValidationError validate_structure(plan)
    @testset "glm-object beta vectors stay Normal-only" begin
        # Pair contract (peer: else decomposed): a stated non-Normal GLM
        # beta prior fails closed at the surface naming the decomposed path.
        err = try
            lower_rkppl(quote
                    X = hcat(x1, x2)
                    y ~ NormalIDGLM(X, alpha, beta, 1.0)
                    alpha ~ Normal(0, 10)
                    beta[axes(X, 2)] .~ Laplace.(0, 1)
                end, (:y, :x1, :x2))
            nothing
        catch e
            e
        end
        @test err isa SurfaceLoweringError
        @test occursin("Normal-only", sprint(showerror, err))
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

# M2: new sampled families as pure prior contributors (unstated a/b keep
# their Normal(0, 1) defaults).
const _PV_M2 = quote
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
# levels) plus a Cauchy width-1 block (no intercept: intercept +
# full-cover factor is unidentified).
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
    @test _pv_posterior(kern, lay, (mu = [0.5, -0.25], s = 1.3)) ≈
        _pv_m1_oracle(0.5, -0.25, 1.3) rtol = 1e-12
    _, _, kern, lay = _pv_query(_PV_M2, _pv_cols())
    q = (mu = [0.25, 0.5], t = 1.5, l = -0.5, g = 2.25, s = 0.8)
    @test _pv_posterior(kern, lay, q) ≈
        _pv_m2_oracle(0.25, 0.5, 1.5, -0.5, 2.25, 0.8) rtol = 1e-12
    _, _, kern, lay = _pv_query(_PV_M3, _pv_cols())
    q = (mu = [-0.5, 1.25], u = 0.5, h = 1.1, s = 2.0)
    @test _pv_posterior(kern, lay, q) ≈
        _pv_m3_oracle(-0.5, 1.25, 0.5, 1.1, 2.0) rtol = 1e-12
    _, _, kern, lay = _pv_query(_PV_M4, _pv_gcols())
    q = (mu = [0.3, -0.4, 0.1, 0.75],)
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
    _pv_enzyme_check(_PV_M1, _pv_cols(), (mu = [0.5, -0.25], s = 1.3))
    _pv_enzyme_check(_PV_M2, _pv_cols(),
        (mu = [0.25, 0.5], t = 1.5, l = -0.5, g = 2.25, s = 0.8))
    _pv_enzyme_check(_PV_M3, _pv_cols(),
        (mu = [-0.5, 1.25], u = 0.5, h = 1.1, s = 2.0))
    _pv_enzyme_check(_PV_M4, _pv_gcols(), (mu = [0.3, -0.4, 0.1, 0.75],))
end

@testset "prior vocab SB parity" begin
    # Peer literals: `BayesianRegressionModels:rk:parity-prior-vocab`
    # brief `2026-09-26T22-18-21-872-8yw5i7` (BRM `ff5e589`, SB `24578c3`,
    # BridgeStan 2.9.0; `propto=false`, `jacobian=true`; SB-vs-oracle
    # ~1e-15, central-diff ≤7e-10). RK lane `fade3b9`. Conventions:
    # full posterior with constants, log-Jacobian included.
    _pv_sb_check(prog, cols, q, sbv, sbg) = begin
        _, _, kern, lay = _pv_query(prog, cols)
        @test _pv_posterior(kern, lay, q) ≈ sbv rtol = 1e-12
        g = _pv_enzyme_check(prog, cols, q)
        @test g ≈ sbg rtol = 1e-9
    end
    # P1: StudentT intercept + Laplace slope.
    _pv_sb_check(_PV_M1, _pv_cols(), (mu = [0.5, -0.25], s = 1.3),
        -15.676861749966013,
        [5.393491124260353, 1.998520710059171, 3.491050295857985])
    # P2: Cauchy intercept + Flat slope (Flat is exactly 0.0 both sides).
    _pv_sb_check(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            a ~ Cauchy(0, 1)
            b ~ Flat()
            s ~ Exponential(1)
        end, _pv_cols(), (mu = [0.5, -0.25], s = 1.3),
        -14.388851106658095,
        [4.7473372781065075, 0.9985207100591711, 3.491050295857985])
    # P3: factor StudentT broadcast + Cauchy slope, fixed s = 1.5 (SB
    # native order is [slope, cats]; the literal below is already in RK
    # [c1, c2, c3, b] order).
    _pv_sb_check(_PV_M4, _pv_gcols(), (mu = [0.3, -0.4, 0.1, 0.75],),
        -21.633064049357102,
        [0.07852219465122685, 3.2093567251461983, 1.5444721990933479,
            -2.565555555555555])
    # P4: Uniform(0.5, 1.5) response scale, default Normal population
    # priors (SB `real<0.5,1.5>` is the same affine-logit leg).
    _pv_sb_check(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            s ~ Uniform(0.5, 1.5)
        end, _pv_cols(), (mu = [0.5, -0.25], s = 1.3),
        -15.810050464119632,
        [5.047337278106507, 1.248520710059171, -0.1334091943559405])
    # P5: half-StudentT(4, 0, 1) scale — SB's `truncated(...; lower=0)`
    # is the renormalized truncated distribution, matching RK
    # `:positive` (exact +log(2)).
    _pv_sb_check(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            s ~ truncated(StudentT(4, 0, 1), 0, Inf)
        end, _pv_cols(), (mu = [0.5, -0.25], s = 1.3),
        -14.883826525901483,
        [5.047337278106507, 1.248520710059171, 3.305988784434435])
    # P5 Stan-kernel twin: `:positive_stan` (emitter/hand path) is the
    # unrenormalized declaration kernel — exactly SB minus log(2), same
    # grads.
    plan = lower_rkppl(quote
            mu = a .+ b .* x
            y .~ Normal.(mu, s)
            s ~ truncated(StudentT(4, 0, 1), 0, Inf)
        end, (:y, :x))
    i = findfirst(p -> p.name === :s, plan.parameters)
    p = plan.parameters[i]
    plan.parameters[i] =
        SampledParameter(p.name, p.family, p.args, :positive_stan, p.label)
    validate_structure(plan)
    bound = bind_data(plan, _pv_cols())
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    q = (mu = [0.5, -0.25], s = 1.3)
    @test _pv_posterior(kern, built.layout, q) ≈
        -14.883826525901483 - log(2) rtol = 1e-12
    u = unconstrain(built.layout, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test g ≈ [5.047337278106507, 1.248520710059171, 3.305988784434435] rtol = 1e-9
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _pv_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_pv_reactant_measure, built, bound, post_q, u)
end

function _pv_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "prior vocab under Reactant" begin
    @testset "mixed population" begin
        fx = _pv_reactant(_PV_M1, _pv_cols())
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.val ≈ fx.native rtol = 1e-12
        @test fx.rval ≈ fx.native rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-8
    end
    @testset "uniform plus half" begin
        fx = _pv_reactant(_PV_M3, _pv_cols())
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.val ≈ fx.native rtol = 1e-12
        @test fx.rval ≈ fx.native rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-8
    end
    @testset "traced program is O(1) in n_obs and n_levels" begin
        small = _pv_reactant(_PV_M4, _pv_gcols())
        bigcols = Dict{Symbol,AbstractVector}(
            :y => vcat(_PV_Y, _PV_Y), :x => vcat(_PV_X, _PV_X),
            :g => vcat(_PV_G, _PV_G))
        @test _pv_reactant(_PV_M4, bigcols).lines == small.lines
        morelevels = Dict{Symbol,AbstractVector}(
            :y => vcat(_PV_Y, _PV_Y), :x => vcat(_PV_X, _PV_X),
            :g => vcat(_PV_G, _PV_G .+ 3))
        @test _pv_reactant(_PV_M4, morelevels).lines == small.lines
    end
end

