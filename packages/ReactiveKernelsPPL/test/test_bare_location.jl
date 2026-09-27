# Bare sampled-parameter locations (single-family Bernoulli/Binomial/Poisson
# over a constrained-scale param, no link inversion — the mixture bare-mean
# slots, single-family form): surface admission, fail-closed battery, value
# parity vs Distributions.jl oracles (rate_1..4 shapes), Enzyme-vs-findiff
# gradients, and Reactant/XLA value+grad. (`_check_gradient` / `_findiff_grad`
# / `_GEN_BACKEND` come from test_generator.jl, included first.)
using Distributions: Beta, Binomial, Bernoulli, Poisson, Gamma, Exponential,
    logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a bare-location program; return
# `(bound, built, kern, layout)`.
function _bare_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_bare_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

@testset "bare lowering" begin
    prog = Meta.parse("""begin
        theta ~ Beta(1.0, 1.0)
        k .~ Binomial.(n, theta)
    end""")
    plan = lower_rkppl(prog, (:k, :n))
    r = only(plan.responses)
    @test r.family === BinomialLogitFam && r.link === LogitLink
    @test r.predictor === :theta
    @test isempty(plan.predictors)

    # Poisson bare rate.
    pplan = lower_rkppl(Meta.parse("""begin
        lambda ~ Gamma(2.0, 1.0)
        y .~ Poisson.(lambda)
    end"""), (:y,))
    pr = only(pplan.responses)
    @test pr.family === PoissonLogFam && pr.link === LogLink
    @test pr.predictor === :lambda

    # Bernoulli bare probability.
    bplan = lower_rkppl(Meta.parse("""begin
        theta ~ Beta(2.0, 2.0)
        y .~ Bernoulli.(theta)
    end"""), (:y,))
    br = only(bplan.responses)
    @test br.family === BernoulliLogitFam && br.link === LogitLink
    @test br.predictor === :theta

    # Prior-only params lower (no likelihood term reads thetaprior).
    prior = lower_rkppl(Meta.parse("""begin
        theta ~ Beta(1.0, 1.0)
        thetaprior ~ Beta(1.0, 1.0)
        k .~ Binomial.(n, theta)
    end"""), (:k, :n))
    @test Set(p.name for p in prior.parameters) == Set([:theta, :thetaprior])

    # Literals stay fail-closed (intercept-only predictor message).
    @test_throws SurfaceLoweringError lower_rkppl(Meta.parse("""begin
        k .~ Binomial.(n, 0.3)
    end"""), (:k, :n))

    # Other families keep the strict broadcast-link message.
    @test_throws SurfaceLoweringError lower_rkppl(Meta.parse("""begin
        mu ~ Normal(0.0, 5.0)
        y .~ NegativeBinomial2.(mu, phi)
    end"""), (:y,))

    # Unknown symbols fail as locations, not as predictors.
    @test_throws SurfaceLoweringError lower_rkppl(Meta.parse("""begin
        theta ~ Beta(1.0, 1.0)
        k .~ Binomial.(n, nosuch)
    end"""), (:k, :n))

    # Evidence on a bare location fails closed (cdf arms are link-space).
    @test_throws ContractValidationError bind_data(
        lower_rkppl(Meta.parse("""begin
            mu = a .+ b .* x
            sigma ~ Exponential(1.0)
            theta ~ Beta(1.0, 1.0)
            k .~ censored.(Binomial.(n, theta), lo, hi)
        end"""), (:k, :n, :x, :lo, :hi)),
        Dict{Symbol,AbstractVector}(:k => [3], :n => [10], :x => [0.5],
            :lo => [0], :hi => [10]))
end

@testset "bare rate values" begin
    @testset "rate_1 posterior" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3], :n => [10]))
        q = (; theta = 0.3)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Beta(1.0, 1.0), 0.3) +
               logpdf(Binomial(10, 0.3), 3) +
               log(0.3) + log(1 - 0.3) # logistic Jacobian at u=logit(0.3)
        @test got ≈ want atol = 1e-12
    end

    @testset "rate_2 two thetas" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta1 ~ Beta(1.0, 1.0)
            theta2 ~ Beta(1.0, 1.0)
            k1 .~ Binomial.(n1, theta1)
            k2 .~ Binomial.(n2, theta2)
        end"""), Dict{Symbol,AbstractVector}(:k1 => [4], :n1 => [12],
            :k2 => [7], :n2 => [15]))
        q = (; theta1 = 0.25, theta2 = 0.6)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Binomial(12, 0.25), 4) + logpdf(Binomial(15, 0.6), 7) +
               log(0.25) + log(0.75) + log(0.6) + log(0.4)
        @test got ≈ want atol = 1e-12
    end

    @testset "rate_4 prior-only param" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            thetaprior ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3], :n => [10]))
        q = (; theta = 0.3, thetaprior = 0.7)
        got = _bare_posterior(kern, lay, q)
        # thetaprior contributes its prior + Jacobian, no likelihood term.
        want = logpdf(Binomial(10, 0.3), 3) +
               log(0.3) + log(0.7) + log(0.7) + log(0.3)
        @test got ≈ want atol = 1e-12
    end

    @testset "poisson bare rate" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            lambda ~ Gamma(2.0, 1.0)
            y .~ Poisson.(lambda)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2]))
        q = (; lambda = 1.5)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Gamma(2.0, 1.0), 1.5) +
               sum(logpdf(Poisson(1.5), v) for v in (0, 1, 3, 5, 2)) +
               log(1.5) # exp Jacobian at u=log(1.5)
        @test got ≈ want atol = 1e-12
    end

    @testset "bernoulli bare probability" begin
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(2.0, 2.0)
            y .~ Bernoulli.(theta)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0, 1]))
        q = (; theta = 0.6)
        got = _bare_posterior(kern, lay, q)
        want = logpdf(Beta(2.0, 2.0), 0.6) +
               sum(logpdf(Bernoulli(0.6), v) for v in (0, 1, 1, 0, 1)) +
               log(0.6) + log(0.4)
        @test got ≈ want atol = 1e-12
    end
end

@testset "bare Enzyme-vs-findiff" begin
    progs = [
        (Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3], :n => [10])),
        (Meta.parse("""begin
            lambda ~ Gamma(2.0, 1.0)
            y .~ Poisson.(lambda)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2])),
    ]
    for (prog, cols) in progs
        plan = lower_rkppl(prog, keys(cols))
        bound = bind_data(plan, cols)
        built = build_kernel(bound)
        u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
        _check_gradient(built.spec, bound, u)
    end
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _bare_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_bare_reactant_measure, built, bound, post_q, u)
end

function _bare_reactant_measure(built, bound, post_q, u)
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

@testset "bare locations under Reactant" begin
    progs = [
        (Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [3, 5], :n => [10, 12])),
        (Meta.parse("""begin
            lambda ~ Gamma(2.0, 1.0)
            y .~ Poisson.(lambda)
        end"""), Dict{Symbol,AbstractVector}(:y => [0, 1, 3, 5, 2])),
    ]
    for (prog, cols) in progs
        fx = _bare_reactant(prog, cols)
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
    end
    # Data-length invariance (constraints.md): more rows must not
    # replicate the loop body.
    prog = Meta.parse("""begin
        theta ~ Beta(2.0, 2.0)
        y .~ Bernoulli.(theta)
    end""")
    small = _bare_reactant(prog, Dict{Symbol,AbstractVector}(:y => [0, 1]))
    large = _bare_reactant(prog,
        Dict{Symbol,AbstractVector}(:y => [0, 1, 1, 0, 1, 0]))
    @test small.lines == large.lines
end

# SB parity vs the pair partner's BridgeStan numbers (brief
# 2026-09-27T12-42-35-032-1ad3lwv on
# BayesianRegressionModels:rk:kernel:matrix-b, BRM 97bb538, StanBlocks
# 24578c3, BridgeStan 2.9.0): full posterior at u, propto=false,
# Jacobian included, BridgeStan AD grads. SB is logit-parameterized
# (eta + Logistic(0,1) prior, BinomialLogit likelihood); RK is
# theta-parameterized (Beta prior + logistic Jacobian). The densities
# are algebraically identical in u (Beta(1,1) on theta + Jacobian ≡
# Logistic(0,1) on eta), and u is the logit in both, so values and
# grads compare directly. Synthetic (n,k) (PosteriorDB is not in the
# BRM env); probe u == slice q. RK layout order differs from SB
# declaration order, so pins compare by coordinate name. rate_4 has no
# SB counterpart (verified wall: BRM lowers unused sampled statements
# to generated-quantities RNG, never to parameters — partner verdict
# brief 2026-09-27T12-43-03-912-1rtm6ac); its Distributions-oracle
# value test above stands in place of SB parity.
_bare_sb_vec(names, pairs) = [Dict(pairs)[n] for n in names]

@testset "bare SB parity" begin
    @testset "R1 rate_1" begin
        # SB: eta ~ 1, Logistic(0,1); k ~ BinomialLogit(n, eta);
        # k=[7], n=[10], u=[0.1].
        _, _, kern, lay = _bare_query(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), Dict{Symbol,AbstractVector}(:k => [7], :n => [10]))
        names = coordinate_names(lay)
        u = _bare_sb_vec(names, [:theta => 0.1])
        @test abs(Base.invokelatest(kern, u) - (-3.345268178100804)) < 1e-12
        bound = bind_data(lower_rkppl(Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k .~ Binomial.(n, theta)
        end"""), (:k, :n)), Dict{Symbol,AbstractVector}(:k => [7], :n => [10]))
        built = build_kernel(bound)
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _bare_sb_vec(names, [:theta => 1.7002497502527198])
        @test maximum(abs.(g .- want)) < 1e-10
    end

    @testset "R2 rate_2" begin
        # SB: eta ~ 0 + grp, per-level Logistic(0,1);
        # k=[9,3], n=[12,8], u=[0.1,-0.1] (grp1, grp2 order).
        prog = Meta.parse("""begin
            theta1 ~ Beta(1.0, 1.0)
            theta2 ~ Beta(1.0, 1.0)
            k1 .~ Binomial.(n1, theta1)
            k2 .~ Binomial.(n2, theta2)
        end""")
        cols = Dict{Symbol,AbstractVector}(:k1 => [9], :n1 => [12],
            :k2 => [3], :n2 => [8])
        _, _, kern, lay = _bare_query(prog, cols)
        names = coordinate_names(lay)
        @test Set(names) == Set([:theta1, :theta2])
        u = _bare_sb_vec(names, [:theta1 => 0.1, :theta2 => -0.1])
        @test abs(Base.invokelatest(kern, u) - (-6.8465406046781885)) < 1e-12
        bound = bind_data(lower_rkppl(prog, keys(cols)), cols)
        built = build_kernel(bound)
        prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
        g = similar(u)
        sampler_value_and_gradient!(prep, g, u)
        @test all(isfinite, g)
        want = _bare_sb_vec(names,
            [:theta1 => 2.65029137529484, :theta2 => -0.7502081252106003])
        @test maximum(abs.(g .- want)) < 1e-10
    end

    @testset "R3/R5 rate_3/rate_5" begin
        # SB: eta ~ 1 over two (n,k) rows; k=[9,3], n=[12,8].
        # R3 u=[0.2]; R5 same density at u=[0.0].
        prog = Meta.parse("""begin
            theta ~ Beta(1.0, 1.0)
            k1 .~ Binomial.(n1, theta)
            k2 .~ Binomial.(n2, theta)
        end""")
        cols = Dict{Symbol,AbstractVector}(:k1 => [9], :n1 => [12],
            :k2 => [3], :n2 => [8])
        _, _, kern, lay = _bare_query(prog, cols)
        names = coordinate_names(lay)
        bound = bind_data(lower_rkppl(prog, keys(cols)), cols)
        built = build_kernel(bound)
        for (uu, val, grad) in (([0.2], -5.5400758893075075,
            [0.9036520591254853]),
                ([0.0], -5.830258735231283, [2.0]))
            u = _bare_sb_vec(names, [:theta => only(uu)])
            @test abs(Base.invokelatest(kern, u) - val) < 1e-12
            prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
            g = similar(u)
            sampler_value_and_gradient!(prep, g, u)
            @test all(isfinite, g)
            want = _bare_sb_vec(names, [:theta => only(grad)])
            @test maximum(abs.(g .- want)) < 1e-10
        end
    end
end
