# Student-t evidence wrappers (`censored.`/`truncated.`/`interval_censored.`
# over `StudentT`, the Gaussian clamp law over the `student_t` cdf):
# surface admission, fail-closed battery, value parity vs
# Distributions.jl oracles, Enzyme-vs-findiff gradients, and
# Reactant/XLA value+grad (in test_student_evidence_reactant.jl).
# (`_check_gradient` / `_findiff_grad` /
# `_GEN_BACKEND` come from test_generator.jl, included first.)
using Distributions: TDist, Normal, Exponential, logpdf, logcdf, logccdf
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Test

# Lower + bind + build + query an evidence program; return
# `(bound, built, kern, layout)`.
function _stev_query(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
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
    return (; a, b, sigma = s)
end

@testset "student evidence lowering" begin
    for wrap in ("censored", "truncated")
        prog = Meta.parse("""begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            sigma ~ Exponential(1.0)
            y .~ $wrap.(StudentT.(3.0, mu, sigma), lo, hi)
        end""")
        plan = lower_rkppl(prog, (:y, :x, :lo, :hi); conditioned = (:y, :x, :lo, :hi))
        r = only(plan.responses)
        @test r.family === StudentTFam
        @test r.evidence.kind === Symbol(wrap)
    end
    # interval_censored takes the object + upper only (the response
    # itself is the lower endpoint).
    iplan = lower_rkppl(Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        sigma ~ Exponential(1.0)
        y .~ interval_censored.(StudentT.(3.0, mu, sigma), hi)
    end"""), (:y, :x, :hi); conditioned = (:y, :x, :hi))
    ir = only(iplan.responses)
    @test ir.family === StudentTFam
    @test ir.evidence.kind === :interval_censored

    # Other families keep the fail-closed gate (new message).
    # admitted: censored evidence over HurdlePoisson (families beyond Gaussian/Student-t) (todo `0ze68k8`)
    @test (bind_data(
        lower_rkppl(Meta.parse("""begin
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            mu = a .+ b .* x
            y .~ censored.(HurdlePoisson.(exp.(mu), 0.1), lo, hi)
        end"""), (:y, :x, :lo, :hi); conditioned = (:y, :x, :lo, :hi)),
        Dict{Symbol,AbstractVector}(:y => [1, 2], :x => [0.5, 1.5],
            :lo => [0, 0], :hi => [5, 5])); true)
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
        # Observed values at both endpoints and in the interior.
        cols = Dict{Symbol,AbstractVector}(:y => [0.0, 0.0, 4.0, 10.0, 10.0],
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
    cols = Dict{Symbol,AbstractVector}(:y => [0.0, 0.0, 4.0, 10.0, 10.0],
        :x => [0.0, 1.0, 2.0, 3.0, 4.0], :lo => fill(0.0, 5),
        :hi => fill(10.0, 5))
    prog = Meta.parse("""begin
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        sigma ~ Exponential(1.0)
        y .~ censored.(StudentT.(3.0, mu, sigma), lo, hi)
    end""")
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    _check_gradient(built.spec, bound, u)
end
