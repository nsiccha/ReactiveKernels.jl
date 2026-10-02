# Sweep-owned R items (v1 inventory pair 4, RK half): the rate family
# (Rate_1..4 shapes — Beta prior + Binomial likelihood over 1- and 2-element
# columns; the thin layer is column-oriented, so the scalar-data spelling
# fails closed, pinned in test_sweep_failclosed.jl), ark (AR(K) as Gaussian
# regression on a lag design matrix) and dugongs (nonlinear
# von-Bertalanffy mean via computed-column extraction). Surface admission,
# value parity vs Distributions.jl hand oracles,
# Enzyme-vs-findiff gradients, O(1) emission, Reactant/XLA value+grad.
# (`_findiff_grad` / `_GEN_BACKEND` come from test_generator.jl, included
# first.)
#
# SB parity: partner numbers LANDED and pinned under `_SR_SB` (briefs
# 2026-09-28T12-50-19-611-xlkh75 (R1/R3/ARK) and
# 2026-09-28T15-12-36-933-u490ai (R2/R4/DUG) on
# BayesianRegressionModels:rk:kernel:everything; records
# test/sb_sweep_probe_records.jsonl, promoted 7be1eeb, repromote 58a838df,
# merge 92cd40cd on ns/devibe; driver ran at brm_tip 77c97a3a).
# Stan propto=false with Jacobians, BridgeStan AD grads. Pinned probes:
#   R1: cols k=[6], n=[10]; probe theta=0.6 (u=logit(0.6)); model
#       theta ~ Beta(1,1), k ~ Binomial(n,theta).
#   R3: cols k=[6,8], n=[10,12]; probe theta=0.6; same model, two trials.
#   R2: cols k1=[6], n1=[10], k2=[8], n2=[12]; probe theta1=0.6, theta2=0.7;
#       two independent Beta-Binomial responses.
#   R4: cols k=[6], n=[10]; probe theta=0.6, thetaprior=0.4; R1 plus a
#       prior-only thetaprior ~ Beta(1,1) (SB keeps it sampled via a
#       density-exact `0 *` link; BRM would otherwise demote it to GQ).
#   ARK: cols yt=[1.0,2.0,1.5], ylag1=[0.5,1.0,2.0], ylag2=[0.2,0.5,1.0];
#       probe alpha=1.0, b=[0.5,-0.25], sigma=1.5; model alpha/b ~ Normal(0,10),
#       sigma ~ truncated-Cauchy(0,2.5) NORMALIZED (+log 2), yt ~ Normal.
#       (The inventory `ark` case record is unnormalized decl-bound on a
#       different dataset — the pin uses the ARK probe, not the case.)
#   DUG: cols y=[1.0,2.0,1.5], age=[1.0,5.0,9.0]; DUG1 probe Linf=1.0,
#       kk=1.0, t0=1.0, sigma=e (u=ones(4)) plus the DUG0 zeros probe;
#       von-Bertalanffy mean, Normal response. DUGh is the same model on
#       the STALE header dataset (y=[0.8,1.6,2.1], age=[0.5,1.0,2.0]) —
#       documented here, no RK leg (RK cols follow the code data).
using DifferentiationInterface
using Distributions: Beta, Binomial, Cauchy, Exponential, Normal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a sweep program; return
# `(bound, built, kern, layout)`.
function _sr_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_sr_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

const _SR_R1_K = [6]
const _SR_R1_N = [10]
_sr_r1_cols() = Dict{Symbol,AbstractVector}(:k => copy(_SR_R1_K),
    :n => copy(_SR_R1_N))
const _SR_R3_K = [6, 8]
const _SR_R3_N = [10, 12]
_sr_r3_cols() = Dict{Symbol,AbstractVector}(:k => copy(_SR_R3_K),
    :n => copy(_SR_R3_N))
const _SR_ARK_YT = [1.0, 2.0, 1.5]
const _SR_ARK_L1 = [0.5, 1.0, 2.0]
const _SR_ARK_L2 = [0.2, 0.5, 1.0]
_sr_ark_cols() = Dict{Symbol,AbstractVector}(:yt => copy(_SR_ARK_YT),
    :ylag1 => copy(_SR_ARK_L1), :ylag2 => copy(_SR_ARK_L2))

const _SR_RATE_PROG = quote
    theta ~ Beta(1.0, 1.0)
    k .~ Binomial.(n, theta)
end
const _SR_R2_PROG = quote
    theta1 ~ Beta(1.0, 1.0)
    theta2 ~ Beta(1.0, 1.0)
    k1 .~ Binomial.(n1, theta1)
    k2 .~ Binomial.(n2, theta2)
end
const _SR_R4_PROG = quote
    theta ~ Beta(1.0, 1.0)
    thetaprior ~ Beta(1.0, 1.0)
    k .~ Binomial.(n, theta)
end
const _SR_DUG_PROG = quote
    Linf ~ Normal(2.0, 1.0)
    kk ~ Normal(0.0, 1.0)
    t0 ~ Normal(0.0, 1.0)
    sigma ~ Exponential(1.0)
    mu = Linf .* (1 .- exp.(-kk .* (age .- t0)))
    y .~ Normal.(mu, sigma)
end
_sr_r2_cols() = Dict{Symbol,AbstractVector}(:k1 => [6], :n1 => [10],
    :k2 => [8], :n2 => [12])
const _SR_DUG_Y = [1.0, 2.0, 1.5]
const _SR_DUG_AGE = [1.0, 5.0, 9.0]
_sr_dug_cols() = Dict{Symbol,AbstractVector}(:y => copy(_SR_DUG_Y),
    :age => copy(_SR_DUG_AGE))

# Layout-agnostic value probe (all-zero / all-one u need no coordinate
# order): identity unscaling maps 0/1 to themselves for real-support
# names, exp maps 0/1 to 1/e for positive scales.
function _sr_value_u(prog::Expr, cols::Dict{Symbol,AbstractVector}, u::Vector{Float64})
    _, _, kern, lay = _sr_query(prog, cols)
    @test length(u) == lay.total
    return Base.invokelatest(kern, u)
end

const _SR_ARK_PROG = quote
    alpha ~ Normal(0.0, 10.0)
    b1 ~ Normal(0.0, 10.0)
    b2 ~ Normal(0.0, 10.0)
    sigma ~ truncated(Cauchy(0.0, 2.5), 0.0, Inf)
    mu = alpha .+ b1 .* ylag1 .+ b2 .* ylag2
    yt .~ Normal.(mu, sigma)
end

# BridgeStan reference pins (lp + AD grad, propto=false, Jacobian
# included) from the partner briefs cited in the header. `grad` rides the
# SB declaration order (`stan_names` in the records); legs map it onto RK
# coordinates by name. Stan shas: R1/R3 92591dcff428ab7f, R2
# a55e3effae366dca, R4 b37f7c8c3c098b8e, ARK ccd97a8d2768f599, DUG
# 7f7ebc2298cb633d.
const _SR_SB = (
    R1 = (lp = -2.8101254950152406, grad = [-0.19999999999999996]),
    R3 = (lp = -4.357335650071095, grad = [0.6000000000000006]),
    R2 = (lp = -5.83550624952482,
        grad = [-0.19999999999999996, -0.7999999999999996]),
    R4 = (lp = -4.237241850655386,
        grad = [-0.19999999999999996, 0.19999999999999996]),
    ARK = (lp = -15.023820664671407,
        grad = [-2.3102450980392155, 0.06777777777777777,
            0.006111111111111098, 0.012499999999999982]),
    DUG0 = (lp = -12.138631199228037, grad = [2.0, 0.0, 0.0, 4.25]),
    DUG1 = (lp = -11.88668938104969,
        grad = [1.202980209660058, -0.9897216700153967,
            -1.1378821505379024, -4.4087291217338285]),
)
_sr_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "sweep replicate admission" begin
    @testset "rate: Beta-sampled prob into Binomial" begin
        # Bare-location form (matrix-b): the bare Beta parameter lowers
        # as a BinomialLogitFam location with no link inversion and no
        # built predictor.
        plan = lower_rkppl(_SR_RATE_PROG, (:k, :n); conditioned = (:k, :n))
        r = only(plan.responses)
        @test r.family === BinomialLogitFam
        @test r.link === LogitLink
        @test r.predictor === :theta
        @test r.trials === :n
        @test only(plan.parameters).family === :beta
        @test isempty(plan.predictors)
    end
    @testset "ark: lag design as data columns" begin
        plan = lower_rkppl(_SR_ARK_PROG, (:yt, :ylag1, :ylag2); conditioned = (:yt, :ylag1, :ylag2))
        r = only(plan.responses)
        @test r.family === GaussianFam
        @test r.predictor === :mu
    end
    @testset "rate_2: two independent prob responses" begin
        plan = lower_rkppl(_SR_R2_PROG, (:k1, :n1, :k2, :n2); conditioned = (:k1, :n1, :k2, :n2))
        @test length(plan.responses) == 2
        @test all(r -> r.family === BinomialLogitFam, plan.responses)
        @test [r.predictor for r in plan.responses] == [:theta1, :theta2]
    end
    @testset "rate_4: prior-only parameter alongside" begin
        plan = lower_rkppl(_SR_R4_PROG, (:k, :n); conditioned = (:k, :n))
        @test only(plan.responses).family === BinomialLogitFam
        @test Set(p.name for p in plan.parameters) == Set([:theta, :thetaprior])
    end
    @testset "dugongs: nonlinear mean via extracted column" begin
        plan = lower_rkppl(_SR_DUG_PROG, (:y, :age); conditioned = (:y, :age))
        r = only(plan.responses)
        @test r.family === GaussianFam
        @test length(plan.predictors) == 1
    end
end

@testset "sweep replicate value parity" begin
    @testset "rate_1 (1-element)" begin
        _, _, kern, lay = _sr_query(_SR_RATE_PROG, _sr_r1_cols())
        q = (theta = 0.6,)
        got = _sr_posterior(kern, lay, q)
        # Hand oracle: Beta prior + Binomial likelihood + logit Jacobian
        # (log(theta*(1-theta))), fully independent of the layout.
        want = logpdf(Beta(1.0, 1.0), 0.6) +
               logpdf(Binomial(10, 0.6), 6) + log(0.6 * 0.4)
        @test got ≈ want rtol = 1e-12
        @test logjac(lay, unconstrain(lay, q)) ≈ log(0.6 * 0.4) rtol = 1e-12
    end
    @testset "rate_3 (shared theta, 2-element)" begin
        _, _, kern, lay = _sr_query(_SR_RATE_PROG, _sr_r3_cols())
        q = (theta = 0.6,)
        got = _sr_posterior(kern, lay, q)
        want = logpdf(Beta(1.0, 1.0), 0.6) +
               logpdf(Binomial(10, 0.6), 6) +
               logpdf(Binomial(12, 0.6), 8) + log(0.6 * 0.4)
        @test got ≈ want rtol = 1e-12
    end
    @testset "rate_2 (two thetas)" begin
        _, _, kern, lay = _sr_query(_SR_R2_PROG, _sr_r2_cols())
        q = (theta1 = 0.6, theta2 = 0.7)
        got = _sr_posterior(kern, lay, q)
        want = logpdf(Beta(1.0, 1.0), 0.6) + logpdf(Binomial(10, 0.6), 6) +
               log(0.6 * 0.4) + logpdf(Beta(1.0, 1.0), 0.7) +
               logpdf(Binomial(12, 0.7), 8) + log(0.7 * 0.3)
        @test got ≈ want rtol = 1e-12
    end
    @testset "rate_4 (prior-only thetaprior)" begin
        _, _, kern, lay = _sr_query(_SR_R4_PROG, _sr_r1_cols())
        q = (theta = 0.6, thetaprior = 0.4)
        got = _sr_posterior(kern, lay, q)
        want = logpdf(Beta(1.0, 1.0), 0.6) + logpdf(Binomial(10, 0.6), 6) +
               log(0.6 * 0.4) + logpdf(Beta(1.0, 1.0), 0.4) + log(0.4 * 0.6)
        @test got ≈ want rtol = 1e-12
    end
    @testset "dugongs (nonlinear mean, u0 point)" begin
        # Layout-agnostic: u=0 → Linf=kk=t0=0 (identity), sigma=1 (exp).
        _, _, _, dlay = _sr_query(_SR_DUG_PROG, _sr_dug_cols())
        got = _sr_value_u(_SR_DUG_PROG, _sr_dug_cols(), zeros(dlay.total))
        want = sum(logpdf(Normal(0.0, 1.0), v) for v in _SR_DUG_Y) +
            logpdf(Normal(2.0, 1.0), 0.0) + logpdf(Normal(0.0, 1.0), 0.0) +
            logpdf(Normal(0.0, 1.0), 0.0) + logpdf(Exponential(1.0), 1.0)
        @test got ≈ want rtol = 1e-10
    end
    @testset "dugongs (nonlinear mean, u1 point)" begin
        # u=1 → Linf=kk=t0=1 (identity), sigma=e (exp).
        _, _, _, dlay = _sr_query(_SR_DUG_PROG, _sr_dug_cols())
        got = _sr_value_u(_SR_DUG_PROG, _sr_dug_cols(), ones(dlay.total))
        mu = 1.0 .* (1 .- exp.(-1.0 .* (_SR_DUG_AGE .- 1.0)))
        want = sum(logpdf(Normal(m, exp(1.0)), v)
            for (m, v) in zip(mu, _SR_DUG_Y)) +
            logpdf(Normal(2.0, 1.0), 1.0) + logpdf(Normal(0.0, 1.0), 1.0) +
            logpdf(Normal(0.0, 1.0), 1.0) +
            logpdf(Exponential(1.0), exp(1.0)) + 1.0
        @test got ≈ want rtol = 1e-10
    end
            @testset "ark (AR(2) lag regression)" begin
        _, _, kern, lay = _sr_query(_SR_ARK_PROG, _sr_ark_cols())
        q = (alpha = 1.0, b1 = 0.5, b2 = -0.25, sigma = 1.5)
        got = _sr_posterior(kern, lay, q)
        mu = q.alpha .+ q.b1 .* _SR_ARK_L1 .+ q.b2 .* _SR_ARK_L2
        # Truncated-Cauchy(0, 2.5) on (0, Inf): half-mass normalization
        # (+log 2, Stan-faithful); sigma rides exp (log Jacobian).
        want = sum(logpdf(Normal(m, 1.5), y)
            for (m, y) in zip(mu, _SR_ARK_YT)) +
            sum(logpdf(Normal(0.0, 10.0), c) for c in (q.alpha, q.b1, q.b2)) +
            logpdf(Cauchy(0.0, 2.5), 1.5) + log(2) + log(1.5)
        @test got ≈ want rtol = 1e-12
    end
end

# One SB value+grad pin at a constrained probe (the _SR_SB leg pattern):
# posterior vs the banked lp, Enzyme grad vs the banked BridgeStan grad
# mapped onto RK coordinates by name.
function _sr_sb_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple, pin::NamedTuple, pairs::Vector{<:Pair})
    bound, built, kern, lay = _sr_query(prog, cols)
    @test abs(_sr_posterior(kern, lay, q) - pin.lp) < 1e-12
    names = coordinate_names(lay)
    @test Set(names) == Set(first.(pairs))
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    want = _sr_sb_vec(names, pairs)
    @test maximum(abs.(g .- want)) < 1e-10
    return nothing
end

@testset "sweep replicate SB parity" begin
    # Direct u-parity throughout: RK interval/exp maps plus the
    # Beta/Cauchy/Exponential densities match Stan's propto=false forms
    # (map audit in test_sb_parity.jl), and both sides evaluate the same
    # model, data, and probe. RK layout order differs from SB declaration
    # order, so grad pins compare by coordinate name.
    @testset "rate_1 (R1)" begin
        # SB: theta ~ Beta(1,1); k ~ Binomial(n,theta); k=[6], n=[10];
        # probe theta=0.6.
        _sr_sb_check(_SR_RATE_PROG, _sr_r1_cols(), (theta = 0.6,), _SR_SB.R1,
            [:theta => only(_SR_SB.R1.grad)])
    end
    @testset "rate_3 (R3)" begin
        # SB: same model; k=[6,8], n=[10,12]; probe theta=0.6.
        _sr_sb_check(_SR_RATE_PROG, _sr_r3_cols(), (theta = 0.6,), _SR_SB.R3,
            [:theta => only(_SR_SB.R3.grad)])
    end
    @testset "rate_2 (R2)" begin
        # SB: two independent Beta-Binomials; probe (0.6, 0.7).
        _sr_sb_check(_SR_R2_PROG, _sr_r2_cols(), (theta1 = 0.6, theta2 = 0.7),
            _SR_SB.R2, [:theta1 => _SR_SB.R2.grad[1],
                :theta2 => _SR_SB.R2.grad[2]])
    end
    @testset "rate_4 (R4)" begin
        # SB: R1 plus prior-only thetaprior ~ Beta(1,1), kept sampled via
        # a density-exact `0 *` link; probe (0.6, 0.4).
        _sr_sb_check(_SR_R4_PROG, _sr_r1_cols(),
            (theta = 0.6, thetaprior = 0.4), _SR_SB.R4,
            [:theta => _SR_SB.R4.grad[1],
                :thetaprior => _SR_SB.R4.grad[2]])
    end
    @testset "ark (ARK)" begin
        # SB: alpha/b ~ Normal(0,10), sigma ~ normalized
        # truncated-Cauchy(0,2.5); probe (1.0, 0.5, -0.25, 1.5) in
        # [pop.1, pop.2, pop.3, sigma] order.
        _sr_sb_check(_SR_ARK_PROG, _sr_ark_cols(),
            (alpha = 1.0, b1 = 0.5, b2 = -0.25, sigma = 1.5), _SR_SB.ARK,
            [:alpha => _SR_SB.ARK.grad[2],
                :b1 => _SR_SB.ARK.grad[3],
                :b2 => _SR_SB.ARK.grad[4],
                :sigma => _SR_SB.ARK.grad[1]])
    end
    @testset "dugongs (DUG0 zeros)" begin
        # SB: Linf ~ Normal(2,1), kk/t0 ~ Normal(0,1), sigma ~
        # Exponential(1); von-Bertalanffy mean; u=zeros(4). Linf rides
        # the synth coordinate; order-agnostic u, name-mapped grad.
        bound, built, kern, lay = _sr_query(_SR_DUG_PROG, _sr_dug_cols())
        names = coordinate_names(lay)
        u = zeros(lay.total)
        @test abs(Base.invokelatest(kern, u) - _SR_SB.DUG0.lp) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _sr_sb_vec(names,
            [:Linf => _SR_SB.DUG0.grad[1],
                :kk => _SR_SB.DUG0.grad[2], :t0 => _SR_SB.DUG0.grad[3],
                :sigma => _SR_SB.DUG0.grad[4]])
        @test maximum(abs.(g .- want)) < 1e-10
    end
    @testset "dugongs (DUG1 ones)" begin
        # SB: same model; u=ones(4) → Linf=kk=t0=1, sigma=e.
        bound, built, kern, lay = _sr_query(_SR_DUG_PROG, _sr_dug_cols())
        names = coordinate_names(lay)
        u = ones(lay.total)
        @test abs(Base.invokelatest(kern, u) - _SR_SB.DUG1.lp) < 1e-12
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _sr_sb_vec(names,
            [:Linf => _SR_SB.DUG1.grad[1],
                :kk => _SR_SB.DUG1.grad[2], :t0 => _SR_SB.DUG1.grad[3],
                :sigma => _SR_SB.DUG1.grad[4]])
        @test maximum(abs.(g .- want)) < 1e-10
    end
end

# One Enzyme-vs-findiff gradient check at an unconstrained probe (no
# oracle needed; layout-agnostic).
function _sr_enzyme_check_u(prog::Expr, cols::Dict{Symbol,AbstractVector},
        u::Vector{Float64})
    bound, built, kern, lay = _sr_query(prog, cols)
    @test length(u) == lay.total
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

# One Enzyme-vs-findiff gradient check at a constrained probe (no oracle
# needed).
function _sr_enzyme_check(prog::Expr, cols::Dict{Symbol,AbstractVector},
        q::NamedTuple)
    bound, built, kern, lay = _sr_query(prog, cols)
    u = unconstrain(lay, q)
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    _, _ = sampler_value_and_gradient!(prep, g, u)
    @test all(isfinite, g)
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
    return g
end

@testset "sweep replicate Enzyme gradients" begin
    @testset "rate_1" begin
        _sr_enzyme_check(_SR_RATE_PROG, _sr_r1_cols(), (theta = 0.6,))
    end
    @testset "rate_3" begin
        _sr_enzyme_check(_SR_RATE_PROG, _sr_r3_cols(), (theta = 0.6,))
    end
    @testset "rate_2" begin
        _sr_enzyme_check(_SR_R2_PROG, _sr_r2_cols(), (theta1 = 0.6, theta2 = 0.7))
    end
    @testset "rate_4" begin
        _sr_enzyme_check(_SR_R4_PROG, _sr_r1_cols(), (theta = 0.6, thetaprior = 0.4))
    end
    @testset "dugongs" begin
        _, _, _, dlay = _sr_query(_SR_DUG_PROG, _sr_dug_cols())
        _sr_enzyme_check_u(_SR_DUG_PROG, _sr_dug_cols(), ones(dlay.total))
    end
        @testset "ark" begin
        _sr_enzyme_check(_SR_ARK_PROG, _sr_ark_cols(),
            (alpha = 1.0, b1 = 0.5, b2 = -0.25, sigma = 1.5))
    end
end

# Statement-head histogram of the generated kernel (the joint-parity
# O(1) pattern): sweep plates must not unroll over observations.
function _sr_statement_heads(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    def = ReactiveKernelsPPL.kernel_expr(bound, assign_layout(bound))
    heads = Dict{String,Int}()
    for st in def.args[2].args
        st isa Expr || continue
        heads[string(st.head)] = get(heads, string(st.head), 0) + 1
    end
    return heads
end

@testset "sweep replicate emission is O(1) in n_obs" begin
    h2 = _sr_statement_heads(_SR_RATE_PROG, _sr_r3_cols())
    h4 = _sr_statement_heads(_SR_RATE_PROG, Dict{Symbol,AbstractVector}(
        :k => vcat(_SR_R3_K, _SR_R3_K), :n => vcat(_SR_R3_N, _SR_R3_N)))
    @test h2 == h4
    a3 = _sr_statement_heads(_SR_ARK_PROG, _sr_ark_cols())
    a6 = _sr_statement_heads(_SR_ARK_PROG, Dict{Symbol,AbstractVector}(
        :yt => vcat(_SR_ARK_YT, _SR_ARK_YT),
        :ylag1 => vcat(_SR_ARK_L1, _SR_ARK_L1),
        :ylag2 => vcat(_SR_ARK_L2, _SR_ARK_L2)))
    @test a3 == a6
    d3 = _sr_statement_heads(_SR_DUG_PROG, _sr_dug_cols())
    d6 = _sr_statement_heads(_SR_DUG_PROG, Dict{Symbol,AbstractVector}(
        :y => vcat(_SR_DUG_Y, _SR_DUG_Y),
        :age => vcat(_SR_DUG_AGE, _SR_DUG_AGE)))
    @test d3 == d6
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size. Ladder-1 support
# (robust verdict 2026-09-27T13-41-02-194-15s8owb, §7n): scalar-mu
# (intercept-only broadcast) + sampled Normal-id scale silently miscompiles
# under the default pipeline (grad off by exactly n-1) — such legs pin
# `optimize = :only_enzyme` and carry default-pipeline `@test_broken`.
# NARROWED rule (coordinator 2026-09-27 17:23 CEST, provisional pending
# robust re-audit; evidence: matrix-a RK V1–V6): vector-mu legs assert the
# default pipeline (exact, delta 0.0) — the pin misfires there
# (`@test_broken` Unexpected Pass = red). All six sweep legs below are
# vector-mu or non-Normal: default assertions throughout. Drop pin + marker
# together the day `@test_broken` goes red (upstream fixed).
function _sr_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector};
        ladder1::Bool = false)
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_sr_reactant_measure, built, bound, post_q, u;
        ladder1)
end

function _sr_reactant_measure(built, bound, post_q, u; ladder1::Bool = false)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    if ladder1
        cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u);
            optimize = :only_enzyme)
        rval, rgrad = cad(Reactant.to_rarray(u))
        cad_default = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
        _, rgrad_default = cad_default(Reactant.to_rarray(u))
        return (; lines = count(==('\n'), hlo), native, primal, val, g,
            rval = Float64(rval), rgrad = Array(rgrad),
            rgrad_default = Array(rgrad_default))
    end
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad), rgrad_default = nothing)
end

@testset "sweep replicate under Reactant" begin
    progs = [
        ("rate_1", _SR_RATE_PROG, _sr_r1_cols(), false),
        ("rate_3", _SR_RATE_PROG, _sr_r3_cols(), false),
        ("rate_2", _SR_R2_PROG, _sr_r2_cols(), false),
        ("rate_4", _SR_R4_PROG, _sr_r1_cols(), false),
        ("dugongs", _SR_DUG_PROG, _sr_dug_cols(), false),
        ("ark", _SR_ARK_PROG, _sr_ark_cols(), false),
    ]
    for (name, prog, cols, ladder1) in progs
        @testset "$name" begin
            fx = _sr_reactant(prog, cols; ladder1)
            @test fx.primal ≈ fx.native rtol = 1e-9
            @test fx.val ≈ fx.native rtol = 1e-12
            @test fx.rval ≈ fx.native rtol = 1e-9
            @test fx.rgrad ≈ fx.g rtol = 1e-8
            ladder1 && @test_broken fx.rgrad_default ≈ fx.g rtol = 1e-9
        end
    end
    @testset "traced program is O(1) in n_obs" begin
        _, prog, _, _ = progs[6]
        small = _sr_reactant(prog, _sr_ark_cols())
        large = _sr_reactant(prog, Dict{Symbol,AbstractVector}(
            :yt => vcat(_SR_ARK_YT, _SR_ARK_YT),
            :ylag1 => vcat(_SR_ARK_L1, _SR_ARK_L1),
            :ylag2 => vcat(_SR_ARK_L2, _SR_ARK_L2)))
        @test small.lines == large.lines
    end
end
