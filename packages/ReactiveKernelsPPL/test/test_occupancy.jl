# Occupancy building blocks (detection masking on the logit scale +
# `logaddexp.` marginalization in derived columns): surface admission,
# value parity vs hand oracles, Enzyme-vs-findiff gradients, and
# Reactant/XLA value+grad. Full per-model occupancy programs
# (`multi_occupancy`, `mt`, `mtbh_model`, `mth_model`) extend this file.
# (`_check_gradient` / `_findiff_grad` / `_GEN_BACKEND` come from
# test_generator.jl, included first.)
using Distributions: Bernoulli, Exponential, Normal, logpdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Test

# Lower + bind + build + query an occupancy program; return
# `(bound, built, kern, layout)`.
function _occ_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
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
    end"""), (:y, :x, :det))
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
    end"""), (:y, :x, :lo))
    @test only(lplan.responses).family === GaussianFam

    # `logaddexp.` takes exactly two arguments.
    @test_throws ContractValidationError bind_data(
        lower_rkppl(Meta.parse("""begin
            a ~ Normal(0, 1)
            mu = a .+ b .* x
            marg = logaddexp.(mu)
            y .~ Normal.(marg, 1.0)
        end"""), (:y, :x)),
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
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    _check_gradient(built.spec, bound, u)
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _occ_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_occ_reactant_measure, built, bound, post_q, u)
end

function _occ_reactant_measure(built, bound, post_q, u)
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

# Upstream XLA gap (the besselix/gamma_inc/beta_inc pin precedent):
# broadcast `ifelse.` with an untraced data condition has no Reactant
# materialization rule (`similar` over `Broadcasted{typeof(ifelse)}`
# with a plain `Vector{Bool}` condition), so the masked program fails
# at trace time (measured on Reactant 0.2.288). The signature below is
# exactly that gap; anything else rethrows loudly.
_occ_is_upstream_gap(e) =
    e isa MethodError && e.f === Base.similar && !isempty(e.args) &&
    e.args[1] isa Base.Broadcast.Broadcasted && e.args[1].f === ifelse

@testset "occupancy under Reactant" begin
    iprog = Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        pick = ifelse.(det .== 1, mu, -30.0)
        y .~ Bernoulli.(logistic.(pick))
    end""")
    icols = Dict{Symbol,AbstractVector}(:y => [1, 0, 1],
        :x => [0.0, 1.0, 2.0], :det => [1, 0, 1])
    try
        fx = _occ_reactant(iprog, icols)
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
        # Self-firing pin: errors (Unexpected Pass) once upstream
        # wires the mixed-ifelse broadcast rule, forcing removal of
        # the try/catch.
        @test_broken true
    catch e
        _occ_is_upstream_gap(e) || rethrow()
        # Known upstream mixed-ifelse gap (above): pinned, not passing.
    end
    # Host-masked exact equivalent (the mask column carries the branch
    # outcomes, so the traced math is identical): full XLA parity on
    # identical math. Native equality first, so the XLA leg provably
    # covers the same density.
    bprog = Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        pick = mu .+ mask
        y .~ Bernoulli.(logistic.(pick))
    end""")
    _occ_mask(det) = [d == 1 ? 0.0 : -30.0 for d in det]
    bcols = merge(icols,
        Dict{Symbol,AbstractVector}(:mask => _occ_mask(icols[:det])))
    _, _, ikern, ilay = _occ_query(iprog, icols)
    _, _, bkern, blay = _occ_query(bprog, bcols)
    u = [0.2, -0.1]
    @test Base.invokelatest(bkern, u) ≈ Base.invokelatest(ikern, u) atol = 1e-12
    fx = _occ_reactant(bprog, bcols)
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.rval ≈ fx.val rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
    # Data-length invariance (constraints.md): more rows must not
    # replicate the loop body.
    sdet, ldet = [1, 0], [1, 0, 1, 1, 0, 1]
    small = _occ_reactant(bprog, Dict{Symbol,AbstractVector}(:y => [1, 0],
        :x => [0.0, 1.0], :det => sdet, :mask => _occ_mask(sdet)))
    large = _occ_reactant(bprog, Dict{Symbol,AbstractVector}(
        :y => [1, 0, 1, 0, 1, 0], :x => collect(0.0:5.0), :det => ldet,
        :mask => _occ_mask(ldet)))
    @test small.lines == large.lines
end
