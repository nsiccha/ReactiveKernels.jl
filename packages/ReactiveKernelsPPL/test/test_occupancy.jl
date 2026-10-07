# Occupancy building blocks (detection masking on the logit scale +
# `logaddexp.` marginalization in derived columns): surface admission,
# value parity vs hand oracles and Enzyme-vs-findiff gradients;
# test_occupancy_reactant.jl adds Reactant/XLA value+grad. Full per-model occupancy programs
# (`multi_occupancy`, `mt`, `mtbh_model`, `mth_model`) extend this file.
# (`_check_gradient` / `_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, included first.)
using Distributions: Bernoulli, Exponential, Normal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Lower + bind + build + query an occupancy program; return
# `(bound, built, kern, layout)`.
function _occ_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_occ_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

@testset "occupancy lowering" begin
    # Detection masking on the logit scale (the admitted shape —
    # `logistic.` lowers only as a `.~` link).
    plan = lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        pick = ifelse.(det .== 1, mu, -30.0)
        y .~ Bernoulli.(logistic.(pick))
    end"""), (:y, :x, :det); conditioned = (:y, :x, :det))
    r = only(plan.responses)
    @test r.family === BernoulliLogitFam && r.link === LogitLink

    # `logaddexp.` in derived columns (marginalization).
    lplan = lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        marg = logaddexp.(mu, lo)
        y .~ Normal.(marg, sigma)
    end"""), (:y, :x, :lo); conditioned = (:y, :x, :lo))
    @test only(lplan.responses).family === GaussianFam

    # `logaddexp.` takes exactly two arguments.
    # refused: one-argument logaddexp is a Julia MethodError (P3)
    @test_throws ContractValidationError bind_data(
        lower_rkppl(Meta.parse("""begin
            a ~ Normal(0, 1)
            mu = a .+ b .* x
            marg = logaddexp.(mu)
            y .~ Normal.(marg, 1.0)
        end"""), (:y, :x); conditioned = (:y, :x)),
        Dict{Symbol,AbstractVector}(:y => [1.0], :x => [0.0]))
end

@testset "occupancy values" begin
    # Masked detection: det==0 rows see logit -30 (≈ 0 probability).
    cols = Dict{Symbol,AbstractVector}(:y => [1, 0, 1, 0],
        :x => [0.0, 1.0, 2.0, 3.0], :det => [1, 0, 1, 1])
    _, _, kern, lay = _occ_query(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        pick = ifelse.(det .== 1, mu, -30.0)
        y .~ Bernoulli.(logistic.(pick))
    end"""), cols)
    names = coordinate_names(lay)
    # `a`/`b` stay scalar sampled params; the location definition
    # carries a zero-size coefficient entry (landed behavior), so the
    # probe pairs an empty predictor vector with the scalars.
    @test Set(names) == Set([:a, :b])
    got = _occ_posterior(kern, lay, (; pick = Float64[], a = 0.5, b = 0.25))
    mus = [0.5 + 0.25x for x in cols[:x]]
    want = logpdf(Normal(0, 1), 0.5) + logpdf(Normal(0, 1), 0.25) +
           sum(zip(cols[:y], mus, cols[:det])) do (yv, mu, d)
        p = d == 1 ? 1 / (1 + exp(-mu)) : 1 / (1 + exp(30.0))
        logpdf(Bernoulli(p), yv)
    end
    @test got ≈ want atol = 1e-10

    # logaddexp marginalization vs the stable two-term oracle.
    lcols = Dict{Symbol,AbstractVector}(:y => [1.0, 2.0], :x => [0.0, 1.0],
        :lo => [-1.0, 0.5])
    _, _, lkern, llay = _occ_query(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        sigma ~ Exponential(1)
        mu = a .+ b .* x
        marg = logaddexp.(mu, lo)
        y .~ Normal.(marg, sigma)
    end"""), lcols)
    lgot = _occ_posterior(lkern, llay,
        (; marg = Float64[], a = 0.5, b = 1.0, sigma = 1.5))
    lse(a, b) = max(a, b) + log1p(exp(-abs(a - b)))
    lmus = [0.5 + 1.0x for x in lcols[:x]]
    lwant = logpdf(Normal(0, 1), 0.5) + logpdf(Normal(0, 1), 1.0) +
            logpdf(Exponential(1), 1.5) + log(1.5) +
            sum(zip(lcols[:y], lmus, lcols[:lo])) do (yv, mu, lo)
        logpdf(Normal(lse(mu, lo), 1.5), yv)
    end
    @test lgot ≈ lwant atol = 1e-10
end

@testset "occupancy Enzyme-vs-findiff" begin
    cols = Dict{Symbol,AbstractVector}(:y => [1, 0, 1, 0],
        :x => [0.0, 1.0, 2.0, 3.0], :det => [1, 0, 1, 1])
    prog = Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        pick = ifelse.(det .== 1, mu, -30.0)
        y .~ Bernoulli.(logistic.(pick))
    end""")
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    _check_gradient(built.spec, bound, u)
end

# An in-model `ifelse` on the detection column and its host-masked exact
# equivalent: the mask column carries the branch outcomes, so the math is
# identical.
const _OCC_IFELSE_PROG = Meta.parse("""begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    pick = ifelse.(det .== 1, mu, -30.0)
    y .~ Bernoulli.(logistic.(pick))
end""")
const _OCC_MASK_PROG = Meta.parse("""begin
    a ~ Normal(0, 1)
    b ~ Normal(0, 1)
    mu = a .+ b .* x
    pick = mu .+ mask
    y .~ Bernoulli.(logistic.(pick))
end""")
_occ_mask(det) = [d == 1 ? 0.0 : -30.0 for d in det]
_occ_masked(cols) = merge(cols,
    Dict{Symbol,AbstractVector}(:mask => _occ_mask(cols[:det])))
_occ_ifelse_cols() = Dict{Symbol,AbstractVector}(:y => [1, 0, 1],
    :x => [0.0, 1.0, 2.0], :det => [1, 0, 1])

@testset "occupancy host mask equals the in-model ifelse" begin
    _, _, ikern, _ = _occ_query(_OCC_IFELSE_PROG, _occ_ifelse_cols())
    _, _, bkern, _ = _occ_query(_OCC_MASK_PROG, _occ_masked(_occ_ifelse_cols()))
    u = [0.2, -0.1]
    @test Base.invokelatest(bkern, u) ≈ Base.invokelatest(ikern, u) atol = 1e-12
end
