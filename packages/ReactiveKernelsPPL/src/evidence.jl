# One evidence algebra for continuous and integer-valued univariate laws.
# Bounds remain real values: F(floor(b)) includes b, while F(ceil(b)-1)
# excludes it. The difference matters for fractional bounds and clamp atoms.
_evidence_at(b, discrete::Bool) = discrete ? :(Base.trunc(Int, floor($b))) : b
_evidence_below(b, discrete::Bool) = discrete ? :(Base.trunc(Int, ceil($b)) - 1) : b

# Select an owned log-tail endpoint at generation time when the distribution
# supplies one. Other laws keep their existing CDF calculation; taking its
# log here cannot repair underflow inside that family's CDF implementation.
function _evidence_logtail(tail)
    if tail isa Expr && tail.head === :call && length(tail.args) == 2
        endpoint = tail.args[1]
        if endpoint isa Expr && endpoint.head === :. &&
                endpoint.args[2] in (QuoteNode(:cdf), QuoteNode(:ccdf))
            dist = endpoint.args[1]
            if dist isa Expr && dist.head === :call
                head = dist.args[1]
                object = head isa GlobalRef ? getfield(head.mod, head.name) :
                    head isa Symbol && isdefined(PPLGeneratedModels, head) ?
                        getfield(PPLGeneratedModels, head) : nothing
                method = endpoint.args[2] == QuoteNode(:cdf) ? :logcdf : :logccdf
                if applicable(ReactiveKernels.kernel_endpoint_names, object) &&
                        method in ReactiveKernels.kernel_endpoint_names(object)
                    return Expr(:call, Expr(:., dist, QuoteNode(method)), tail.args[2])
                end
            end
        end
    end
    return :(log($tail))
end

function _evidence_logdifference(a, b)
    # Equality includes two empty tails (-Inf). Avoid -Inf - -Inf before
    # log1mexp; a zero interval mass has log mass -Inf.
    return :($a == $b ? -Inf : $a + $(GlobalRef(LogExpFunctions, :log1mexp))($b - $a))
end

function _evidence_logmass(logcdf, logccdf, lo, hi)
    lo === nothing && hi === nothing && return :(0.0)
    lo === nothing && return logcdf(hi)
    hi === nothing && return logccdf(lo)
    # Subtract on the smaller tail, entirely in log space. The selected
    # branch is lazy, including its derivative work.
    return :($(logcdf(lo)) > -log(2) ?
        $(_evidence_logdifference(logccdf(lo), logccdf(hi))) :
        $(_evidence_logdifference(logcdf(hi), logcdf(lo))))
end

# A literal infinite bound is no bound (`nothing`) from parsing on. A bound
# read at run time (data, a sampled value or a definition) can still be its
# side's infinity: data `-Inf` below or `Inf` above removes that bound on its
# rows. Such a side selects the one-sided mass lazily, so neither the value
# nor the derivative reads an infinite endpoint, and a discrete endpoint
# never converts one to an integer. `lo`/`hi` are the bounds as read; `below`
# and `at` the endpoints the mass is taken at (`nothing`: that side is open).
_evidence_runtime_bound(b) = b !== nothing && !(b isa Real)

function _evidence_open_logmass(logcdf, logccdf, lo, below, hi, at)
    _evidence_runtime_bound(lo) && return :($lo == -Inf ?
        $(_evidence_open_logmass(logcdf, logccdf, nothing, nothing, hi, at)) :
        $(_evidence_open_logmass(logcdf, logccdf, nothing, below, hi, at)))
    _evidence_runtime_bound(hi) && return :($hi == Inf ?
        $(_evidence_open_logmass(logcdf, logccdf, lo, below, nothing, nothing)) :
        $(_evidence_open_logmass(logcdf, logccdf, lo, below, nothing, at)))
    return _evidence_logmass(logcdf, logccdf, below, at)
end

_evidence_cell(::Val{:none}, base, y, lo, hi, cdf, ccdf, discrete) = base

function _evidence_cell(::Val{:truncated}, base, y, lo, hi, cdf, ccdf, discrete;
        logcdf=b -> _evidence_logtail(cdf(b)), logccdf=b -> _evidence_logtail(ccdf(b)))
    below = lo === nothing ? nothing : _evidence_below(lo, discrete)
    at = hi === nothing ? nothing : _evidence_at(hi, discrete)
    result = :($base - $(_evidence_open_logmass(logcdf, logccdf, lo, below, hi, at)))
    hi === nothing || (result = :($y > $hi ? -Inf : $result))
    lo === nothing || (result = :($y < $lo ? -Inf : $result))
    if lo !== nothing && hi !== nothing
        result = :($lo <= $hi ? $result : -Inf)
    end
    return result
end

function _evidence_cell(::Val{:censored}, base, y, lo, hi, cdf, ccdf, discrete;
        logcdf=b -> _evidence_logtail(cdf(b)), logccdf=b -> _evidence_logtail(ccdf(b)))
    result = base
    hi === nothing || (result = :($y >= $hi ?
        $(logccdf(_evidence_below(hi, discrete))) : $result))
    lo === nothing || (result = :($y <= $lo ?
        $(logcdf(_evidence_at(lo, discrete))) : $result))
    if lo !== nothing && hi !== nothing
        result = :($lo == $hi ? 0.0 : $result)
        result = :($lo <= $hi ? $result : -Inf)
    end
    hi === nothing || (result = :($y > $hi ? -Inf : $result))
    lo === nothing || (result = :($y < $lo ? -Inf : $result))
    return result
end

function _evidence_cell(::Val{:interval_censored}, base, y, lo, hi, cdf, ccdf, discrete;
        logcdf=b -> _evidence_logtail(cdf(b)), logccdf=b -> _evidence_logtail(ccdf(b)))
    # An `Inf` upper end is the right-censored event `(y, Inf)`.
    mass = _evidence_open_logmass(logcdf, logccdf, nothing,
        _evidence_at(y, discrete), hi, _evidence_at(hi, discrete))
    return :($y < $hi ? $mass : -Inf)
end

_evidence_discrete(f::LikelihoodFamily) = _is_bernoulli_family(f) ||
    _is_binomial_family(f) || f in (PoissonLogFam, NegativeBinomial2Fam,
    NegativeBinomialFam, HurdlePoissonFam, ZeroInflatedPoissonFam,
    CategoricalFam, CategoricalLogitFam, OrderedLogisticFam, OrdinalFam)

# Recover the already-authored scalar distribution from its logpdf endpoint.
# This keeps prior argument preparation and parameterization in the existing
# emitter, while every family uses the same wrapper algebra.
_evidence_density_guard(cell) = cell isa Expr && cell.head === :if &&
    length(cell.args) == 3 && (cell.args[3] === -Inf || cell.args[3] == :(-Inf))

function _evidence_endpoint(cell::Expr)
    _evidence_density_guard(cell) && return _evidence_endpoint(cell.args[2])
    cell.head === :call && cell.args[1] isa Expr &&
        cell.args[1].head === :. && cell.args[1].args[2] == QuoteNode(:logpdf) ||
        throw(ContractValidationError("[generator] evidence requires a distribution endpoint"))
    return cell.args[1].args[1]
end

_evidence_integer_index(ex, y) = ex == y ? :(Base.trunc(Int, $y)) : ex isa Expr ?
    Expr(ex.head, (_evidence_integer_index(a, y) for a in ex.args)...) : ex

function _evidence_integer_observation(ex, y)
    ex isa Expr || return ex
    if ex.head === :ref
        return Expr(:ref, ex.args[1], (_evidence_integer_index(a, y) for a in ex.args[2:end])...)
    end
    if ex.head === :call && ex.args[1] isa Expr &&
            ex.args[1].head === :. && ex.args[1].args[2] == QuoteNode(:logpdf)
        return Expr(:call, ex.args[1], :(Base.trunc(Int, $y)))
    end
    return Expr(ex.head, (_evidence_integer_observation(a, y) for a in ex.args)...)
end

# `levels` is a leveled family's class count: the bound `n_levels`, or an
# expression the cell evaluates when the count is a run-time length
# (categorical logit slices).
function _wrap_evidence!(r, plan, pre, inputs, base;
        cdf=nothing, ccdf=nothing, logcdf=nothing, logccdf=nothing,
        levels=r.n_levels)
    r.evidence.kind === :none && return base
    # Parameter support remains outside the evidence computation. Its inactive
    # density, normalizer, tails and derivative work must stay unevaluated.
    if _evidence_density_guard(base) &&
            !(_dovar(1) in _expr_value_symbols(base.args[1]))
        guarded = _wrap_evidence!(r, plan, pre, inputs, base.args[2]; cdf, ccdf, logcdf, logccdf, levels)
        return Expr(:if, base.args[1], guarded, base.args[3])
    end
    family = r.family === MixtureFam ? something(r.mixture_family, GaussianFam) :
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
    logcdf === nothing && (logcdf = b -> _evidence_logtail(cdf(b)))
    logccdf === nothing && (logccdf = b -> _evidence_logtail(ccdf(b)))
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
            base = :(1 <= $y <= $levels ? $base : -Inf)
        end
    end
    bounds = map((r.evidence.lower, r.evidence.upper)) do b
        b === nothing && return nothing
        # Response selection gathers every row-shaped plate input, including
        # bounds, after this shared cell has been emitted.
        _thread_ref!(inputs, b)
    end
    return _evidence_cell(Val(r.evidence.kind), base, y, bounds...,
        cdf, ccdf, discrete; logcdf, logccdf)
end
