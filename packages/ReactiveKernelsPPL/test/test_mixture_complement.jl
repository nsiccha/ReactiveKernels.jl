# Complement-pair mixture weights (`[s, 1.0 - s]` / `[1.0 - s, s]` over
# one `:unit`-support sampled parameter — the collapsed Beta-weight
# shape): surface admission, fail-closed battery, value parity vs
# Distributions.jl oracles, Enzyme-vs-findiff gradients, and
# Reactant/XLA value+grad (in test_mixture_complement_reactant.jl).
# (`_check_gradient` / `_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using Distributions: Beta, Exponential, Normal, Uniform, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Lower + bind + build + query a complement program; return
# `(bound, built, kern, layout)`.
function _mixc_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
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
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([theta, 1.0 - theta]))
    end"""), (:y,); conditioned = (:y,))
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
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([1.0 - theta, theta]))
    end"""), (:y,); conditioned = (:y,))
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
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([a, b]))
    end"""), (:y,); conditioned = (:y,))

    # Non-unit params fail at contract.
    # refused: [s, 1 - s] with s ~ Exponential gives a negative weight (support mismatch)
    @test_throws ContractValidationError bind_data(
        lower_rkppl(Meta.parse("""begin
            mu1 ~ Normal(0.0, 2.0)
            mu2 ~ Normal(0.0, 2.0)
            sigma ~ Exponential(1.0)
            s ~ Exponential(1.0)
            y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma)), Ref([s, 1.0 - s]))
        end"""), (:y,); conditioned = (:y,)),
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
            y .~ MixtureModel.(vcat.(Normal.(mu1, sigma), Normal.(mu2, sigma), Normal.(mu3, sigma)), Ref([theta, 1.0 - theta]))
        end"""), (:y,); conditioned = (:y,)),
        Dict{Symbol,AbstractVector}(:y => [0.5, -1.0]))
end

# Two-term log-sum-exp oracle (stable; no new test dep).
_mixc_lse(a, b) = max(a, b) + log1p(exp(-abs(a - b)))

@testset "complement weights from bounded priors" begin
    model(prior) = Meta.parse("""begin
        m1 ~ Normal(0, 2)
        m2 ~ Normal(0, 2)
        theta ~ $prior
        y .~ MixtureModel.(vcat.(Normal.(m1, 1), Normal.(m2, 1)), Ref([theta, 1-theta]))
    end""")
    cols = Dict{Symbol,AbstractVector}(:y => [0.5, -1.0])

    for prior in ("Uniform(0, 1)", "Uniform(0.2, 0.8)",
            "truncated(Normal(0, 1), 0, 1)")
        plan = lower_rkppl(model(prior), keys(cols); conditioned = keys(cols))
        @test bind_data(plan, cols) isa StructuralPlan
    end
    for prior in ("Uniform(-0.1, 1)", "Uniform(0, 1.1)", "Normal(0, 1)")
        @test_throws ContractValidationError lower_rkppl(
            model(prior), keys(cols); conditioned = keys(cols))
    end
    data = Dict{Symbol,Any}(:y => cols[:y], :lo => 0.0, :hi => 1.0)
    data_plan = lower_rkppl(model("Uniform(lo, hi)"), data; conditioned = (:y,))
    @test bind_data(data_plan, data) isa StructuralPlan
    bad_data = merge(data, Dict{Symbol,Any}(:lo => -0.5))
    @test_throws ContractValidationError bind_data(data_plan, bad_data)

    bound = bind_data(lower_rkppl(model("Uniform(0, 1)"), keys(cols);
        conditioned = keys(cols)), cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    q = (; m1 = 0.0, m2 = 1.0, theta = 0.4)
    got = _mixc_posterior(kern, built.layout, q)
    want = logpdf(Normal(0, 2), q.m1) + logpdf(Normal(0, 2), q.m2) +
        logpdf(Uniform(0, 1), q.theta) + log(q.theta) + log1p(-q.theta) +
        sum(_mixc_lse(log(q.theta) + logpdf(Normal(q.m1, 1), y),
            log1p(-q.theta) + logpdf(Normal(q.m2, 1), y)) for y in cols[:y])
    @test got ≈ want atol = 1e-12
    u = [0.1, -0.2, 0.3]
    uniform_gradient = _check_gradient(built.spec, bound, u)

    # Beta(1, 1) is the same prior measure as Uniform(0, 1), so the
    # alternate spelling must emit the same posterior and gradient.
    beta_bound = bind_data(lower_rkppl(model("Beta(1, 1)"), keys(cols);
        conditioned = keys(cols)), cols)
    beta_built = build_kernel(beta_bound)
    beta_kern = prepare_query(beta_built, beta_bound, :sampler)
    @test Base.invokelatest(kern, u) ≈ Base.invokelatest(beta_kern, u) atol = 1e-12
    @test uniform_gradient ≈ _check_gradient(beta_built.spec, beta_bound, u)
end

@testset "complement values" begin
    prog = Meta.parse("""begin
        mu1 ~ Normal(0.0, 2.0)
        mu2 ~ Normal(0.0, 2.0)
        sigma1 ~ Exponential(1.0)
        sigma2 ~ Exponential(1.0)
        theta ~ Beta(5.0, 5.0)
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma1), Normal.(mu2, sigma2)), Ref([theta, 1.0 - theta]))
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
        y .~ MixtureModel.(vcat.(Normal.(mu1, sigma1), Normal.(mu2, sigma2)), Ref([theta, 1.0 - theta]))
    end""")
    cols = Dict{Symbol,AbstractVector}(:y => [0.5, -1.0, 2.0])
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    _check_gradient(built.spec, bound, u)
end
