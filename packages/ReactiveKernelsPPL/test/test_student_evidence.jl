# Student-t evidence wrappers (`censored.`/`truncated.`/`interval_censored.`
# over `StudentT`, the Gaussian clamp law over the `student_t` cdf):
# surface admission, fail-closed battery, value parity vs
# Distributions.jl oracles, Enzyme-vs-findiff gradients, and
# Reactant/XLA value+grad. (`_check_gradient` / `_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using Distributions: TDist, Normal, Exponential, logpdf, logcdf, logccdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using SpecialFunctions
using Test

# Lower + bind + build + query an evidence program; return
# `(bound, built, kern, layout)`.
function _stev_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    return bound, built, kern, built.layout
end

# Posterior at a constrained probe (world-age-safe call).
_stev_posterior(kern, lay, q::NamedTuple) =
    Base.invokelatest(kern, unconstrain(lay, q))

# Constrained probe from (intercept, slope, sigma): `unconstrain`
# takes the grouped form (one coefficient vector per predictor), with
# coefficients in `coordinate_names` order.
function _stev_q(lay, a, b, s)
    coefs = Dict(Symbol("mu.Intercept") => a, Symbol("mu.x") => b)
    muv = [coefs[n] for n in coordinate_names(lay) if startswith(string(n), "mu.")]
    return (; mu = muv, sigma = s)
end

@testset "student evidence lowering" begin
    for wrap in ("censored", "truncated")
        prog = Meta.parse("""begin
            mu = a .+ b .* x
            sigma ~ Exponential(1.0)
            y .~ $wrap.(StudentT.(3.0, mu, sigma), lo, hi)
        end""")
        plan = lower_rkppl(prog, (:y, :x, :lo, :hi))
        r = only(plan.responses)
        @test r.family === StudentTFam
        @test r.evidence.kind === Symbol(wrap)
    end
    # interval_censored takes the object + upper only (the response
    # itself is the lower endpoint).
    iplan = lower_rkppl(Meta.parse("""begin
        mu = a .+ b .* x
        sigma ~ Exponential(1.0)
        y .~ interval_censored.(StudentT.(3.0, mu, sigma), hi)
    end"""), (:y, :x, :hi))
    ir = only(iplan.responses)
    @test ir.family === StudentTFam
    @test ir.evidence.kind === :interval_censored

    # Other families keep the fail-closed gate (new message).
    @test_throws ContractValidationError bind_data(
        lower_rkppl(Meta.parse("""begin
            mu = a .+ b .* x
            y .~ censored.(HurdlePoisson.(exp.(mu), 0.1), lo, hi)
        end"""), (:y, :x, :lo, :hi)),
        Dict{Symbol,AbstractVector}(:y => [1, 2], :x => [0.5, 1.5],
            :lo => [0, 0], :hi => [5, 5]))
end

# Clamp-law oracle over a location-scale TDist: at-bound rows take CDF
# mass (non-strict arms, the Gaussian precedent).
function _stev_censored_oracle(ys, mus, sigmas, nu, lo, hi)
    total = 0.0
    for (y, mu, s) in zip(ys, mus, sigmas)
        d = TDist(nu) * s + mu
        total += y <= lo ? logcdf(d, lo) :
                 y >= hi ? logccdf(d, hi) : logpdf(d, y)
    end
    return total
end

function _stev_truncated_oracle(ys, mus, sigmas, nu, lo, hi)
    total = 0.0
    for (y, mu, s) in zip(ys, mus, sigmas)
        d = TDist(nu) * s + mu
        # log(F(hi) - F(lo)): F(hi) > F(lo) strictly here.
        total += logpdf(d, y) - log(exp(logcdf(d, hi)) - exp(logcdf(d, lo)))
    end
    return total
end

@testset "student evidence values" begin
    @testset "censored incl. at-bound rows" begin
        # y rows: below lo, at lo, interior, at hi, above hi.
        cols = Dict{Symbol,AbstractVector}(:y => [-1.0, 0.0, 4.0, 10.0, 12.0],
            :x => [0.0, 1.0, 2.0, 3.0, 4.0], :lo => fill(0.0, 5),
            :hi => fill(10.0, 5))
        _, _, kern, lay = _stev_query(Meta.parse("""begin
            a ~ Normal(0.0, 1.0)
            b ~ Normal(0.0, 1.0)
            sigma ~ Exponential(1.0)
            mu = a .+ b .* x
            y .~ censored.(StudentT.(3.0, mu, sigma), lo, hi)
        end"""), cols)
        a, b, s = 1.0, 0.5, 2.0
        got = _stev_posterior(kern, lay, _stev_q(lay, a, b, s))
        mus = [a + b * x for x in cols[:x]]
        want = _stev_censored_oracle(cols[:y], mus, fill(s, 5), 3.0, 0.0, 10.0) +
               logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
               logpdf(Exponential(1.0), s) + log(s)
        @test got ≈ want atol = 1e-10
    end

    @testset "truncated two-sided" begin
        cols = Dict{Symbol,AbstractVector}(:y => [0.5, 5.0, 9.5],
            :x => [0.0, 1.0, 2.0])
        _, _, kern, lay = _stev_query(Meta.parse("""begin
            a ~ Normal(0.0, 1.0)
            b ~ Normal(0.0, 1.0)
            sigma ~ Exponential(1.0)
            mu = a .+ b .* x
            y .~ truncated.(StudentT.(3.0, mu, sigma), 0.0, 10.0)
        end"""), cols)
        a, b, s = 0.5, 1.0, 1.5
        got = _stev_posterior(kern, lay, _stev_q(lay, a, b, s))
        mus = [a + b * x for x in cols[:x]]
        want = _stev_truncated_oracle(cols[:y], mus, fill(s, 3), 3.0, 0.0, 10.0) +
               logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
               logpdf(Exponential(1.0), s) + log(s)
        @test got ≈ want atol = 1e-10
    end

    @testset "interval censored" begin
        cols = Dict{Symbol,AbstractVector}(:y => [1.0, 4.0, 7.0],
            :x => [0.0, 1.0, 2.0], :hi => [3.0, 6.0, 9.0])
        _, _, kern, lay = _stev_query(Meta.parse("""begin
            a ~ Normal(0.0, 1.0)
            b ~ Normal(0.0, 1.0)
            sigma ~ Exponential(1.0)
            mu = a .+ b .* x
            y .~ interval_censored.(StudentT.(3.0, mu, sigma), hi)
        end"""), cols)
        a, b, s = 0.0, 0.5, 2.0
        got = _stev_posterior(kern, lay, _stev_q(lay, a, b, s))
        total = 0.0
        for (yv, xv, hiv) in zip(cols[:y], cols[:x], cols[:hi])
            d = TDist(3.0) * s + (a + b * xv)
            total += log(exp(logcdf(d, hiv)) - exp(logcdf(d, yv)))
        end
        want = total + logpdf(Normal(0, 1), a) + logpdf(Normal(0, 1), b) +
               logpdf(Exponential(1.0), s) + log(s)
        @test got ≈ want atol = 1e-10
    end
end

@testset "student evidence Enzyme-vs-findiff" begin
    cols = Dict{Symbol,AbstractVector}(:y => [-1.0, 0.0, 4.0, 10.0, 12.0],
        :x => [0.0, 1.0, 2.0, 3.0, 4.0], :lo => fill(0.0, 5),
        :hi => fill(10.0, 5))
    prog = Meta.parse("""begin
        mu = a .+ b .* x
        sigma ~ Exponential(1.0)
        y .~ censored.(StudentT.(3.0, mu, sigma), lo, hi)
    end""")
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    _check_gradient(built.spec, bound, u)
end

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size. StudentT is
# verified green on the default pipeline (robust baseline §7n scope
# note), so no ladder-1 pin here.
function _stev_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_stev_reactant_measure, built, bound, post_q, u)
end

function _stev_reactant_measure(built, bound, post_q, u)
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

# Upstream XLA gap (the besselix/gamma_inc pin precedent): `student_t`
# evidence arms route through `SpecialFunctions.beta_inc`, which has no
# method for a traced scalar, so every evidence program fails at trace
# time (measured on Reactant 0.2.288: the primal trace throws before any
# gradient is staged). The signature below is exactly that gap; anything
# else rethrows loudly.
_stev_is_upstream_gap(e) =
    e isa MethodError && e.f === SpecialFunctions.beta_inc &&
    length(e.args) == 3 && e.args[3] isa Reactant.TracedRNumber

@testset "student evidence under Reactant" begin
    prog = Meta.parse("""begin
        mu = a .+ b .* x
        sigma ~ Exponential(1.0)
        y .~ censored.(StudentT.(3.0, mu, sigma), lo, hi)
    end""")
    try
        fx = _stev_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => [-1.0, 4.0, 12.0], :x => [0.0, 1.0, 2.0],
            :lo => fill(0.0, 3), :hi => fill(10.0, 3)))
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.rval ≈ fx.val rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-7 atol = 1e-9
        # Data-length invariance (constraints.md): more rows must not
        # replicate the loop body.
        small = _stev_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => [1.0, 12.0], :x => [0.0, 1.0], :lo => fill(0.0, 2),
            :hi => fill(10.0, 2)))
        large = _stev_reactant(prog, Dict{Symbol,AbstractVector}(
            :y => [-1.0, 1.0, 4.0, 10.0, 12.0, 5.0], :x => collect(0.0:5.0),
            :lo => fill(0.0, 6), :hi => fill(10.0, 6)))
        @test small.lines == large.lines
        # Self-firing pin: errors (Unexpected Pass) once upstream wires
        # beta_inc, forcing removal of the try/catch.
        @test_broken true
    catch e
        _stev_is_upstream_gap(e) || rethrow()
        # Known upstream beta_inc gap (above): pinned, not passing.
    end
end
