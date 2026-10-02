# Complement-pair mixture weights (`[s, 1.0 - s]` / `[1.0 - s, s]` over
# one `:unit`-support sampled parameter — the collapsed Beta-weight
# shape): surface admission, fail-closed battery, value parity vs
# Distributions.jl oracles, Enzyme-vs-findiff gradients, and
# Reactant/XLA value+grad. (`_check_gradient` / `_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using Distributions: Beta, Exponential, Normal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query a complement program; return
# `(bound, built, kern, layout)`.
function _mixc_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_mixc_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

@testset "complement lowering" begin
    plan = lower_rkppl(Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        sigma ~ Exponential(1.0)
        theta ~ Beta(5.0, 5.0)
        y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)], [theta, 1.0 - theta])
    end"""), (:y,))
    r = only(plan.responses)
    @test r.family === MixtureFam
    @test r.mixture_weights == MixtureComplementWeights(:theta, true)
    @test r.predictor === :theta

    # Reversed order.
    rplan = lower_rkppl(Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        sigma ~ Exponential(1.0)
        theta ~ Beta(5.0, 5.0)
        y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)], [1.0 - theta, theta])
    end"""), (:y,))
    rr = only(rplan.responses)
    @test rr.mixture_weights == MixtureComplementWeights(:theta, false)

    # Non-complement sampled vectors stay fail-closed.
    # refused: [a, b] from independent Betas does not sum to 1 (probabilities not summing to 1)
    @test_throws SurfaceLoweringError lower_rkppl(Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        sigma ~ Exponential(1.0)
        a ~ Beta(2.0, 2.0)
        b ~ Beta(2.0, 2.0)
        y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)], [a, b])
    end"""), (:y,))

    # Non-unit params fail at contract.
    # refused: [s, 1 - s] with s ~ Exponential gives a negative weight (support mismatch)
    @test_throws ContractValidationError bind_data(
        lower_rkppl(Meta.parse("""begin
            mu1 ~ Normal(0.0, 2.0)
            mu2 ~ Normal(0.0, 2.0)
            sigma ~ Exponential(1.0)
            s ~ Exponential(1.0)
            y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma)], [s, 1.0 - s])
        end"""), (:y,)),
        Dict{Symbol,AbstractVector}(:y => [0.5, -1.0]))

    # K != 2 fails at contract.
    # refused: 2 weights for 3 components (length mismatch)
    @test_throws ContractValidationError bind_data(
        lower_rkppl(Meta.parse("""begin
            mu1 ~ Normal(0.0, 2.0)
            mu2 ~ Normal(0.0, 2.0)
            mu3 ~ Normal(0.0, 2.0)
            sigma ~ Exponential(1.0)
            theta ~ Beta(5.0, 5.0)
            y .~ MixtureModel.([Normal.(mu1, sigma), Normal.(mu2, sigma), Normal.(mu3, sigma)], [theta, 1.0 - theta])
        end"""), (:y,)),
        Dict{Symbol,AbstractVector}(:y => [0.5, -1.0]))
end

# Two-term log-sum-exp oracle (stable; no new test dep).
_mixc_lse(a, b) = max(a, b) + log1p(exp(-abs(a - b)))

@testset "complement values" begin
    prog = Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        sigma1 ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        theta ~ Beta(5.0, 5.0)
        y .~ MixtureModel.([Normal.(mu1, sigma1), Normal.(mu2, sigma2)], [theta, 1.0 - theta])
    end""")
    cols = Dict{Symbol,AbstractVector}(:y => [0.5, -1.0, 2.0])
    _, _, kern, lay = _mixc_query(prog, cols)
    q = (; mu1 = 0.0, mu2 = 1.0, sigma1 = 1.0, sigma2 = 2.0, theta = 0.4)
    got = _mixc_posterior(kern, lay, q)
    want = logpdf(Normal(0, 2), 0.0) + logpdf(Normal(0, 2), 1.0) +
           logpdf(Exponential(1), 1.0) + logpdf(Exponential(1), 2.0) +
           logpdf(Beta(5, 5), 0.4) + log(2.0) + log(0.4) + log(0.6) +
           sum(_mixc_lse(log(0.4) + logpdf(Normal(0, 1), y),
               log(0.6) + logpdf(Normal(1, 2), y)) for y in cols[:y])
    @test got ≈ want atol = 1e-12
end

@testset "complement Enzyme-vs-findiff" begin
    prog = Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        sigma1 ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        theta ~ Beta(5.0, 5.0)
        y .~ MixtureModel.([Normal.(mu1, sigma1), Normal.(mu2, sigma2)], [theta, 1.0 - theta])
    end""")
    cols = Dict{Symbol,AbstractVector}(:y => [0.5, -1.0, 2.0])
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    _check_gradient(built.spec, bound, u)
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _mixc_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_mixc_reactant_measure, built, bound, post_q, u)
end

function _mixc_reactant_measure(built, bound, post_q, u)
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

@testset "complement under Reactant" begin
    prog = Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        sigma1 ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        theta ~ Beta(5.0, 5.0)
        y .~ MixtureModel.([Normal.(mu1, sigma1), Normal.(mu2, sigma2)], [theta, 1.0 - theta])
    end""")
    fx = _mixc_reactant(prog,
        Dict{Symbol,AbstractVector}(:y => [0.5, -1.0, 2.0]))
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.rval ≈ fx.val rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
    # Data-length invariance (constraints.md): more rows must not
    # replicate the loop body.
    small = _mixc_reactant(prog, Dict{Symbol,AbstractVector}(:y => [0.5, -1.0]))
    large = _mixc_reactant(prog,
        Dict{Symbol,AbstractVector}(:y => [0.5, -1.0, 2.0, 1.5, -0.5, 0.0]))
    @test small.lines == large.lines
end
