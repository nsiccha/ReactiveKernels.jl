# One evidence algebra for continuous and integer-valued univariate laws.
# Bounds remain real values: F(floor(b)) includes b, while F(ceil(b)-1)
# excludes it. The difference matters for fractional bounds and clamp atoms.
_evidence_at(b, discrete::Bool) = discrete ? :(ReactiveKernels._tensorized_trunc(Int, floor($b))) : b
_evidence_below(b, discrete::Bool) = discrete ? :(ReactiveKernels._tensorized_trunc(Int, ceil($b)) - 1) : b

function _evidence_logmass(cdf, ccdf, lo, hi)
    lo === nothing && hi === nothing && return :(0.0)
    lo === nothing && return :(log($(cdf(hi))))
    hi === nothing && return :(log($(ccdf(lo))))
    # Both tails are defined; the inactive log remains inactive, including AD.
    return :($(cdf(lo)) > 0.5 ?
        log($(ccdf(lo)) - $(ccdf(hi))) : log($(cdf(hi)) - $(cdf(lo))))
end

_evidence_cell(::Val{:none}, base, y, lo, hi, cdf, ccdf, discrete) = base

function _evidence_cell(::Val{:truncated}, base, y, lo, hi, cdf, ccdf, discrete)
    below = lo === nothing ? nothing : _evidence_below(lo, discrete)
    at = hi === nothing ? nothing : _evidence_at(hi, discrete)
    result = :($base - $(_evidence_logmass(cdf, ccdf, below, at)))
    hi === nothing || (result = :($y > $hi ? -Inf : $result))
    lo === nothing || (result = :($y < $lo ? -Inf : $result))
    if lo !== nothing && hi !== nothing
        result = :($lo <= $hi ? $result : -Inf)
    end
    return result
end

function _evidence_cell(::Val{:censored}, base, y, lo, hi, cdf, ccdf, discrete)
    result = base
    hi === nothing || (result = :($y >= $hi ?
        log($(ccdf(_evidence_below(hi, discrete)))) : $result))
    lo === nothing || (result = :($y <= $lo ?
        log($(cdf(_evidence_at(lo, discrete)))) : $result))
    if lo !== nothing && hi !== nothing
        result = :($lo == $hi ? 0.0 : $result)
        result = :($lo <= $hi ? $result : -Inf)
    end
    hi === nothing || (result = :($y > $hi ? -Inf : $result))
    lo === nothing || (result = :($y < $lo ? -Inf : $result))
    return result
end

function _evidence_cell(::Val{:interval_censored}, base, y, lo, hi, cdf, ccdf, discrete)
    mass = _evidence_logmass(cdf, ccdf,
        _evidence_at(y, discrete), _evidence_at(hi, discrete))
    return :($y < $hi ? $mass : -Inf)
end

_evidence_discrete(f::LikelihoodFamily) = _is_bernoulli_family(f) ||
    _is_binomial_family(f) || f in (PoissonLogFam, NegativeBinomial2Fam,
    NegativeBinomialFam, HurdlePoissonFam, ZeroInflatedPoissonFam,
    CategoricalFam, CategoricalLogitFam, OrderedLogisticFam, OrdinalFam)

# Recover the already-authored scalar distribution from its logpdf endpoint.
# This keeps prior argument preparation and parameterization in the existing
# emitter, while every family uses the same wrapper algebra.
function _evidence_endpoint(cell::Expr)
    cell.head === :call && cell.args[1] isa Expr &&
        cell.args[1].head === :. && cell.args[1].args[2] == QuoteNode(:logpdf) ||
        throw(ContractValidationError("[generator] evidence requires a distribution endpoint"))
    return cell.args[1].args[1]
end

_evidence_integer_index(ex, y) = ex == y ? :(ReactiveKernels._tensorized_trunc(Int, $y)) : ex isa Expr ?
    Expr(ex.head, (_evidence_integer_index(a, y) for a in ex.args)...) : ex

function _evidence_integer_observation(ex, y)
    ex isa Expr || return ex
    if ex.head === :ref
        return Expr(:ref, ex.args[1], (_evidence_integer_index(a, y) for a in ex.args[2:end])...)
    end
    if ex.head === :call && ex.args[1] isa Expr &&
            ex.args[1].head === :. && ex.args[1].args[2] == QuoteNode(:logpdf)
        return Expr(:call, ex.args[1], :(ReactiveKernels._tensorized_trunc(Int, $y)))
    end
    return Expr(ex.head, (_evidence_integer_observation(a, y) for a in ex.args)...)
end

function _wrap_evidence!(r, plan, pre, inputs, base; cdf = nothing, ccdf = nothing)
    r.evidence.kind === :none && return base
    family = r.family === MixtureFam ? r.mixture_family :
        r.family === NormalIDGLMFam ? GaussianFam :
        r.family === BernoulliLogitGLMFam ? BernoulliLogitFam :
        r.family === PoissonLogGLMFam ? PoissonLogFam : r.family
    discrete = _evidence_discrete(family)
    if cdf === nothing
        dist = _evidence_endpoint(base)
        cdf = b -> :($dist.cdf($b))
        ccdf = b -> :($dist.ccdf($b))
        if _is_bernoulli_family(family)
            cdf = b -> :($b < 0 ? 0.0 : ($b >= 1 ? 1.0 : $dist.cdf(false)))
            ccdf = b -> :($b < 0 ? 1.0 : ($b >= 1 ? 0.0 : $dist.cdf(true) - $dist.cdf(false)))
        end
    end
    y = _dovar(1)
    if _is_bernoulli_family(family)
        # The logpdf's folded Bool input is independent of the real clamp
        # endpoints. Preserve the authored observation for comparisons/CDFs.
        y = _thread_ref!(inputs, r.response)
        base = :(($y == 0 || $y == 1) ? $base : -Inf)
    elseif discrete
        # Clamp observations may be real even when the base law is discrete.
        # Convert only in the interior density arm, and only at integers.
        base = _evidence_integer_observation(base, y)
        base = :((isfinite($y) && floor($y) == $y) ? $base : -Inf)
        if family in (CategoricalFam, CategoricalLogitFam, OrderedLogisticFam, OrdinalFam)
            K = r.n_levels
            base = :(1 <= $y <= $K ? $base : -Inf)
        end
    end
    bounds = map((r.evidence.lower, r.evidence.upper)) do b
        b === nothing && return nothing
        # Response selection gathers every row-shaped plate input, including
        # bounds, after this shared cell has been emitted.
        _thread_ref!(inputs, b)
    end
    return _evidence_cell(Val(r.evidence.kind), base, y, bounds...,
        cdf, ccdf, discrete)
end
