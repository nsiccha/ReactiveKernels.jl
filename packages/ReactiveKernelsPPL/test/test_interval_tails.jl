using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme
import SpecialFunctions

# MPFR evaluates the probability before taking its logarithm, independently
# of the Float64 log-tail implementation. Boundary densities give analytic
# location and log-scale derivatives with no numerical differentiation.
function _normal_interval_reference(lo, hi, m, s)
    setprecision(BigFloat, 256) do
        a, b = (BigFloat(lo) - m) / s, (BigFloat(hi) - m) / s
        q(z) = SpecialFunctions.erfc(z / sqrt(big(2))) / 2
        p = a >= 0 ? q(a) - q(b) : q(-b) - q(-a)
        phi(z) = exp(-z^2 / 2) / sqrt(2big(pi))
        pa, pb = phi(a), phi(b)
        za = isfinite(a) ? a * pa : zero(pa)
        zb = isfinite(b) ? b * pb : zero(pb)
        (Float64(log(p)), Float64((pa - pb) / (s * p)),
            Float64((za - zb) / p))
    end
end

function _tail_query(expr, data, q; want=:sampler, conditioned=(:y,))
    bound = bind_data(lower_rkppl(expr, data; conditioned), data)
    built = build_kernel(bound)
    k = prepare_query(built, bound, want)
    u = unconstrain(built.layout, q)
    return k, u
end

@testset "LogNormal and mixture intervals preserve Normal tail masses" begin
    m, s = 0.03, 0.5
    for mixture in (false, true)
        a, b = m .+ [38.5, 40.] .* s, m .+ [40.5, 42.] .* s
        lo, hi = mixture ? (a, b) : (exp.(a), exp.(b))
        law = mixture ? :(MixtureModel.(vcat.(Normal.(m, s),
            Normal.(m .+ 0.1, s)), Ref([0.4, 0.6]))) : :(LogNormal.(m, s))
        expr = quote
            m ~ Normal(0., 10.)
            y .~ interval_censored.($law, hi)
        end
        k, u = _tail_query(expr, (; y=lo, hi, s), (; m))
        value, grad = -log(2pi)/2-log(10)-m^2/200, -m/100
        for i in eachindex(lo)
            # LogNormal changes the integration bounds, not the probability.
            l, h = mixture ? (lo[i], hi[i]) : (log(BigFloat(lo[i])), log(BigFloat(hi[i])))
            v, g, _ = _normal_interval_reference(l, h, m, s)
            if mixture
                v2, g2, _ = _normal_interval_reference(l, h, m+0.1, s)
                p1, p2 = big(0.4)*exp(BigFloat(v)), big(0.6)*exp(BigFloat(v2))
                v, g = Float64(log(p1+p2)), Float64((p1*g+p2*g2)/(p1+p2))
            end
            value += v; grad += g
        end
        Base.invokelatest() do
            @test k(u) ≈ value atol=2e-11
            g = only(Enzyme.gradient(Enzyme.Reverse, k, u))
            @test all(isfinite, g)
            @test only(g) ≈ grad atol=2e-10 rtol=2e-12
        end
    end
end

@testset "Exponential and Weibull evidence select their log-tail endpoints" begin
    for (law, lo, hi, shape) in ((:(Exponential.(s)), 800., 802., 1.),
            (:(Weibull.(2., s)), 40., 42., 2.))
        expr = quote
            t ~ Normal(0., 10.)
            s = exp(t)
            y .~ interval_censored.($law, hi)
        end
        k, u = _tail_query(expr, (; y=[lo], hi=[hi]), (; t=0.))
        value, gradient = setprecision(BigFloat, 256) do
            a, b = BigFloat(lo)^shape, BigFloat(hi)^shape
            qa, qb = exp(-a), exp(-b)
            (Float64(log(qa-qb))-log(2pi)/2-log(10),
                Float64(shape*(a*qa-b*qb)/(qa-qb)))
        end
        Base.invokelatest() do
            @test k(u) ≈ value atol=2e-12
            g = only(Enzyme.gradient(Enzyme.Reverse, k, u))
            @test all(isfinite, g)
            @test only(g) ≈ gradient rtol=2e-12
        end
    end
end

@testset "Normal evidence: active scale, one-sided bounds and truncation" begin
    m, s = 0.03, 0.05
    t = log(s)
    lo = m .+ [-42., 38.5, 40., 332.6] .* s
    hi = lo .+ 2s
    mids = (lo .+ hi) ./ 2
    for kind in (:interval_censored, :truncated, :censored)
        observations = kind === :interval_censored ? lo :
            kind === :truncated ? mids : [lo[1], hi[2], lo[3], hi[4]]
        wrapper = kind === :interval_censored ? :(interval_censored.(Normal.(m, s), hi)) :
            Expr(:., kind, Expr(:tuple, :(Normal.(m, s)), :lo, :hi))
        expr = quote
            m ~ Normal(0., 10.)
            t ~ Normal(0., 10.)
            s = exp(t)
            y .~ $wrapper
        end
        k, u = _tail_query(expr, (; y=observations, lo, hi), (; m, t))
        prior = -log(2pi) - 2log(10) - (m^2+t^2)/200
        value, dm, dt = prior, -m/100, -t/100
        for i in eachindex(lo)
            if kind === :censored
                l, h = observations[i] == lo[i] ? (-Inf, lo[i]) : (hi[i], Inf)
                v, gm, gt = _normal_interval_reference(l, h, m, exp(t))
            else
                v, gm, gt = _normal_interval_reference(lo[i], hi[i], m, exp(t))
                if kind === :truncated
                    z = (observations[i]-m)/exp(t)
                    v = -log(2pi)/2-t-z^2/2-v
                    gm = z/exp(t)-gm
                    gt = z^2-1-gt
                end
            end
            value += v; dm += gm; dt += gt
        end
        Base.invokelatest() do
            @test k(u) ≈ value atol=3e-10 rtol=2e-13
            g = only(Enzyme.gradient(Enzyme.Reverse, k, u))
            @test all(isfinite, g)
            @test g ≈ [dm, dt] rtol=2e-10 atol=3e-9
        end
    end
end

@testset "Normal truncated priors use the same finite log normalizer" begin
    for (lo, hi) in ((38.5, 40.5), (-42., -40.), (40., Inf), (-Inf, -40.))
        x = isfinite(lo) && isfinite(hi) ? (lo+hi)/2 : isfinite(lo) ? lo+1 : hi-1
        expr = quote
            m ~ Normal(0., 10.)
            x ~ truncated(Normal(m, 1.), $lo, $hi)
        end
        k, u = _tail_query(expr, NamedTuple(), (; m=0., x); conditioned=())
        mass, gm, _ = _normal_interval_reference(lo, hi, 0., 1.)
        # The truncation transform is the only additional Jacobian.
        jac = isfinite(lo) && isfinite(hi) ? log(hi-lo)-log(4) : 0.0
        expected = -log(2pi)-log(10)-x^2/2-mass+jac
        Base.invokelatest() do
            @test k(u) ≈ expected atol=3e-12
            g = only(Enzyme.gradient(Enzyme.Reverse, k, u))
            @test all(isfinite, g)
            @test first(g) ≈ x-gm atol=3e-11
        end
    end
end

@testset "Normal interval tails remain finite under ordinary Reverse" begin
    m, s = 0.03, 0.05
    # Include the original subnormal/underflow boundary, both tails, the
    # downstream distance, and narrow intervals near and far from the centre.
    z = [3., 7., 12., 30., 36., 37., 37.5, 38., 38.5, 40., 332.6,
        -36., -39., -42., -334.6, -1., 0., 1., 40.]
    widths = vcat(fill(2., 15), [1e-5, 1e-5, 1e-5, 1e-5])
    lo, hi = m .+ z .* s, m .+ (z .+ widths) .* s
    refs = [_normal_interval_reference(l, h, m, s) for (l, h) in zip(lo, hi)]
    data = (; y=lo, hi, s)
    expr = quote
        m ~ Normal(0., 10.)
        y .~ interval_censored.(Normal.(m, s), hi)
    end
    pw, u = _tail_query(expr, data, (; m); want=:pointwise)
    k, _ = _tail_query(expr, data, (; m))
    Base.invokelatest() do
        got = pw(u).y
        @test all(isfinite, got)
        for (actual, reference) in zip(got, first.(refs))
            @test actual ≈ reference rtol=2e-11 atol=2e-9
        end
        gradient = only(Enzyme.gradient(Enzyme.Reverse, k, u))
        @test all(isfinite, gradient)
        @test only(gradient) ≈ sum(r[2] for r in refs) - m / 100 rtol=2e-9
    end
end
