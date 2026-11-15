using Test, ReactiveKernels, ReactiveKernelsPPL, DifferentiationInterface, Enzyme
import SpecialFunctions
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))
@isdefined(jax_raw_data) || include("jax_ppl_data.jl")

# Interval evidence on an ordinal law reads the link's log tails at the
# cutpoints. Far in a tail the log mass is finite but its probability is
# below floatmin, so a probability-space endpoint would read -Inf there (snag
# rkppl-ordinal-in-553ce5b3). The reference is 512-bit arithmetic with MPFR's
# special functions. Each tail takes the logarithm of the smaller probability
# and `log1p` of its complement, and an event subtracts the smaller tails, so
# no probability within the working precision of one rounds to one. Its
# derivative in eta is a central difference of the same reference, which the
# precision makes exact to Float64.

# log(1 - exp(x)) for x <= 0, by the form that keeps its relative precision.
_ordinal_big_log1mexp(x::BigFloat) = x < -log(big(2)) ? log1p(-exp(x)) : log(-expm1(x))

function _ordinal_big_logtails(link, z::BigFloat)
    if link === :ProbitLink
        small = SpecialFunctions.erfc(abs(z) / sqrt(big(2))) / 2
        tail, rest = log(small), log1p(-small)
        return z < 0 ? (tail, rest) : (rest, tail)
    end
    link === :LogitLink && return (-log1p(exp(-z)), -log1p(exp(z)))
    return (_ordinal_big_log1mexp(-exp(z)), -exp(z))
end

# The law's (log P(Y <= k), log P(Y > k)) for a level 1 <= k < K. A
# stopping-ratio observation exceeds k when it survives the first k stages.
function _ordinal_big_logcdfs(structure, link, eta, cuts, k)
    tails(j) = _ordinal_big_logtails(link, BigFloat(cuts[j]) - eta)
    structure === :Cumulative && return tails(k)
    survival = sum(last(tails(j)) for j in 1:k)
    return (_ordinal_big_log1mexp(survival), survival)
end

# log P(lo < Y <= hi).
function _ordinal_big_event(structure, link, eta, cuts, lo, hi)
    K = length(cuts) + 1
    lo == 0 && hi == K && return big(0.0)
    lo == 0 && return first(_ordinal_big_logcdfs(structure, link, eta, cuts, hi))
    flo, clo = _ordinal_big_logcdfs(structure, link, eta, cuts, lo)
    hi == K && return clo
    fhi, chi = _ordinal_big_logcdfs(structure, link, eta, cuts, hi)
    return flo > -log(big(2)) ? clo + _ordinal_big_log1mexp(chi - clo) :
        fhi + _ordinal_big_log1mexp(flo - fhi)
end

function _ordinal_interval_reference(structure, link, eta, cuts, lo, hi)
    setprecision(BigFloat, 512) do
        f(e) = _ordinal_big_event(structure, link, e, cuts, lo, hi)
        e, h = BigFloat(eta), big"1e-40"
        (Float64(f(e)), Float64((f(e + h) - f(e - h)) / 2h))
    end
end

# Every row has its own coefficient, so each gradient entry is one row's
# derivative. Locations reach both tails (the snag's 30, 60 and 400 and their
# negatives); the events take each category on its own, the lowest and
# highest two, and interior unions.
const _ORDINAL_TAIL_CUTS = [-1.0, 0.0, 1.0]
const _ORDINAL_TAIL_ETAS = [0.5, 30.0, 60.0, -60.0, 400.0, -400.0]
const _ORDINAL_TAIL_EVENTS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4)]

function _ordinal_tail_relerr(actual, expected)
    actual == expected && return 0.0
    return abs(actual - expected) / max(abs(expected), floatmin())
end

@testset "ordinal interval evidence keeps finite tail log masses ($structure, $link)" for
        structure in (:Cumulative, :StoppingRatio),
        link in (:ProbitLink, :LogitLink, :CloglogLink)
    rows = [(eta, lo, hi) for eta in _ORDINAL_TAIL_ETAS for (lo, hi) in _ORDINAL_TAIL_EVENTS]
    etas = first.(rows)
    S, L = Expr(:call, structure), Expr(:call, link)
    expr = quote
        b[axes(x, 1)] .~ Normal.(0.0, 10.0)
        eta = b .* x
        ylo .~ interval_censored.(Ordinal.($S, $L, eta, Ref(cuts)), yhi)
    end
    data = (; x = ones(length(rows)), cuts = _ORDINAL_TAIL_CUTS,
        ylo = [r[2] for r in rows], yhi = [r[3] for r in rows])
    bound = bind_data(lower_rkppl(expr, data; conditioned = (:ylo,)), data)
    built = build_kernel(bound)
    u = unconstrain(built.layout, (; b = etas))
    refs = [_ordinal_interval_reference(structure, link, eta, _ORDINAL_TAIL_CUTS, lo, hi)
        for (eta, lo, hi) in rows]
    # The coefficients' Normal(0, 10) prior, value and derivative.
    prior = sum(-log(10.0) - log(2pi) / 2 - b^2 / 200 for b in etas)
    expected_value = sum(first, refs) + prior
    expected_grad = [last(r) - b / 100 for (r, b) in zip(refs, etas)]
    pointwise = prepare_query(built, bound, :pointwise)
    density = prepare_query(built, bound, :sampler)
    Base.invokelatest() do
        got = pointwise(u).ylo
        @test all(isfinite, got)
        value_errors = map(_ordinal_tail_relerr, got, first.(refs))
        println("ORDINAL_TAIL ", structure, " ", link, " max value relerr ", maximum(value_errors))
        @test maximum(value_errors) < 1e-12
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value == density(u)
        @test value ≈ expected_value rtol = 1e-13
        @test all(isfinite, grad)
        grad_errors = map(_ordinal_tail_relerr, grad, expected_grad)
        println("ORDINAL_TAIL ", structure, " ", link, " max grad relerr ", maximum(grad_errors))
        @test maximum(grad_errors) < 1e-12
        if rk_jax_available()
            src = jax_source(built, bound)
            r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(bound, src)...);
                grad = true, prepare = true)
            @test r.value ≈ expected_value rtol = 1e-13
            jax_errors = map(_ordinal_tail_relerr, r.grad, expected_grad)
            println("ORDINAL_TAIL ", structure, " ", link, " max JAX grad relerr ", maximum(jax_errors))
            @test maximum(jax_errors) < 1e-12
        else
            @info "RK_JAX_PYTHON is not set: JAX parity for ordinal interval tails skipped"
            @test_skip false
        end
    end
end

# Censoring and truncation read the same endpoints: the clamp atoms
# P(Y <= lo) and P(Y >= hi), and the truncation normalizer P(lo <= Y <= hi).
@testset "ordinal censored and truncated evidence keep finite tail log masses ($structure)" for
        structure in (:Cumulative, :StoppingRatio)
    etas = [60.0, -60.0, 400.0, -400.0]
    censored_rows = [(eta, y) for eta in etas for y in (1, 2, 3)]
    truncated_rows = [(eta, y) for eta in etas for y in (2, 3)]
    S = Expr(:call, structure)
    expr = quote
        bc[axes(xc, 1)] .~ Normal.(0.0, 10.0)
        bt[axes(xt, 1)] .~ Normal.(0.0, 10.0)
        yc .~ censored.(Ordinal.($S, ProbitLink(), bc .* xc, Ref(cuts)), 1, 3)
        yt .~ truncated.(Ordinal.($S, ProbitLink(), bt .* xt, Ref(cuts)), 2, 3)
    end
    data = (; xc = ones(length(censored_rows)), xt = ones(length(truncated_rows)),
        cuts = _ORDINAL_TAIL_CUTS, yc = last.(censored_rows), yt = last.(truncated_rows))
    bound = bind_data(lower_rkppl(expr, data; conditioned = (:yc, :yt)), data)
    built = build_kernel(bound)
    q = (; bc = first.(censored_rows), bt = first.(truncated_rows))
    u = unconstrain(built.layout, q)
    event(eta, lo, hi) = _ordinal_interval_reference(structure, :ProbitLink, eta, _ORDINAL_TAIL_CUTS, lo, hi)
    # A clamp atom below 1 is P(Y <= 1), above 3 is P(Y >= 3), between them
    # the level's own mass; a truncated level is renormalized on {2, 3}.
    censored_refs = [y == 1 ? event(eta, 0, 1) : y == 3 ? event(eta, 2, 4) : event(eta, 1, 2)
        for (eta, y) in censored_rows]
    truncated_refs = [event(eta, y - 1, y) .- event(eta, 1, 3) for (eta, y) in truncated_rows]
    refs = vcat(censored_refs, truncated_refs)
    prior = sum(-log(10.0) - log(2pi) / 2 - b^2 / 200 for b in vcat(q.bc, q.bt))
    # The coefficients' transforms are the identity, so `unconstrain` of the
    # per-row derivatives orders them as the gradient.
    expected_grad = unconstrain(built.layout, (;
        bc = [last(r) - b / 100 for (r, b) in zip(censored_refs, q.bc)],
        bt = [last(r) - b / 100 for (r, b) in zip(truncated_refs, q.bt)]))
    pointwise = prepare_query(built, bound, :pointwise)
    Base.invokelatest() do
        p = pointwise(u)
        got = vcat(p.yc, p.yt)
        @test all(isfinite, got)
        value_errors = map(_ordinal_tail_relerr, got, first.(refs))
        println("ORDINAL_TAIL censored/truncated ", structure, " max value relerr ", maximum(value_errors))
        @test maximum(value_errors) < 1e-12
        sampler = prepare_sampler(built, bound, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
        value, grad = sampler_value_and_gradient!(sampler, similar(u), u)
        @test value ≈ sum(first, refs) + prior rtol = 1e-13
        @test all(isfinite, grad)
        grad_errors = map(_ordinal_tail_relerr, grad, expected_grad)
        println("ORDINAL_TAIL censored/truncated ", structure, " max grad relerr ", maximum(grad_errors))
        @test maximum(grad_errors) < 1e-12
        if rk_jax_available()
            src = jax_source(built, bound)
            r = rk_jax_evaluate(src, "log_density", (u, jax_raw_data(bound, src)...);
                grad = true, prepare = true)
            @test r.value ≈ value rtol = 1e-13
            @test maximum(map(_ordinal_tail_relerr, r.grad, expected_grad)) < 1e-12
        else
            @test_skip false
        end
    end
end
