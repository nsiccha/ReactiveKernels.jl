# Packed layout + transforms (D5b thin-layer-owned).
#
# SCALAR constrained-parameter transforms are provided by the reusable bijector
# library (`bijectors.jl`, decision 0l3dsru): the in-graph generator splices
# each bijector's `constrain`/`logjac` endpoints and the host path runs the
# same prepared endpoints, so in-graph and host evaluation agree by
# construction (structural), not by hand-kept duplication. `:identity` (real
# support) is a genuine no-op and stays a direct read in both paths. Per-cell
# (plate) blocks route their HOST transform through the same library; their
# in-graph edges are still a hand-rolled broadcast form (extending the bijector
# splice to the plate graph is a follow-up).

"""
    support_of(family, override) -> Symbol

Inferred unconstrained support (`:real`/`:positive`/`:unit`) for a sampled
family plus optional `:positive` override (half-Normal/half-Cauchy style).
Loud on unknown families and inapplicable overrides.
"""
function support_of(family::Symbol, override::SupportOverride)
    haskey(SAMPLED_SUPPORT, family) ||
        throw(ContractValidationError("[layout] sampled family $family unknown"))
    inferred = SAMPLED_SUPPORT[family]
    override === nothing && return inferred
    if override isa Tuple
        override[1] === :interval || throw(ContractValidationError(
            "[layout] tuple support override must be (:interval, lo, hi), got $override"))
        inferred === :real || throw(ContractValidationError(
            "[layout] :interval override needs a real-support family"))
        return :interval
    end
    override === :positive || throw(
        ContractValidationError("[layout] support override must be :positive, got $override"),
    )
    inferred === :real || throw(
        ContractValidationError("[layout] :positive override needs a real-support family"),
    )
    return :positive
end

# The transform kind and (for :interval) the constrained bounds a layout entry
# needs, from a parameter's family + support override.
function _entry_transform(family::Symbol, override::SupportOverride)
    support = support_of(family, override)
    if support === :interval
        return (:interval, Float64(override[2]), Float64(override[3]))
    end
    transform =
        support === :real ? :identity :
        support === :positive ? :exp : :logistic
    return (transform, NaN, NaN)
end

"""One packed slice: a coefficient block, a scalar latent, or a scan vector latent.
`lo`/`hi` are the constrained bounds of an `:interval` transform (`NaN` otherwise)."""
struct LayoutEntry
    kind::Symbol # :coefficient | :sampled | :scan | :plate
    predictor::Union{Nothing,Symbol}
    name::Symbol # block name (`mu_coef`), parameter name, or scan-state name
    labels::Vector{Symbol} # per-coordinate labels (length == size)
    offset::Int # 1-based packed offset
    size::Int
    transform::Symbol # :identity | :exp | :logistic | :interval
    lo::Float64 # :interval lower bound (else NaN)
    hi::Float64 # :interval upper bound (else NaN)
end
# Non-interval entries omit the bounds.
LayoutEntry(kind::Symbol, predictor::Union{Nothing,Symbol}, name::Symbol,
    labels::Vector{Symbol}, offset::Int, size::Int, transform::Symbol) =
    LayoutEntry(kind, predictor, name, labels, offset, size, transform, NaN, NaN)

"""Packed unconstrained layout: ordered entries + total dimension."""
struct LayoutTable
    entries::Vector{LayoutEntry}
    total::Int
end

"""
    assign_layout(plan) -> LayoutTable

Assign packed coordinates: coefficient blocks in `plan.predictors` order,
then sampled parameters in `plan.parameters` order (pinned, deterministic).
Assumes `validate_plan` passed.
"""
function assign_layout(plan::StructuralPlan)
    isbound(plan) || throw(ContractValidationError(
        "[layout] assign_layout requires a bound plan (bind_data first)"))
    entries = LayoutEntry[]
    offset = 1
    for pred in plan.predictors
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps)
        labels = Symbol[]
        for b in shape.blocks
            append!(labels, b.labels)
        end
        push!(entries,
            LayoutEntry(:coefficient, pred.name, block_name(pred.name), labels,
                offset, shape.width, :identity))
        offset += shape.width
    end
    for p in plan.parameters
        transform, lo, hi = _entry_transform(p.family, p.support_override)
        push!(entries,
            LayoutEntry(:sampled, nothing, p.name, [p.name], offset, 1, transform,
                lo, hi))
        offset += 1
    end
    for p in plan.plate_parameters
        transform, lo, hi = _entry_transform(p.family, p.support_override)
        size = p.range === nothing ? plan.n_obs : length(p.range)
        push!(entries,
            LayoutEntry(:plate, nothing, p.name, [p.name], offset, size, transform,
                lo, hi))
        offset += size
    end
    # Sequential-recurrence latents: one identity array slice per scan. The
    # length T is the loop bound — a literal Int, or `n_obs` when the bound is a
    # data length name (the canonical observation-indexed state-space case).
    # v1 latents have real support (identity transform ⇒ no Jacobian); the
    # emitter reconstructs the carried state from this slice.
    for s in plan.scans
        T = s.hi isa Int ? s.hi : plan.n_obs
        T >= s.lo || throw(ContractValidationError(
            "[layout] scan $(s.state) length $(T) < loop start $(s.lo) — " *
            "the recurrence must run at least once"))
        labels = Symbol[Symbol(i) for i in 1:T]
        push!(entries,
            LayoutEntry(:scan, nothing, s.state, labels, offset, T, :identity))
        offset += T
    end
    return LayoutTable(entries, offset - 1)
end

"""
    coordinate_names(layout) -> Vector{Symbol}

Flat per-coordinate names (R10 read API): coefficients qualified
`Symbol("predictor.coef")`, sampled parameters bare. Length == total.
"""
function coordinate_names(layout::LayoutTable)
    names = Symbol[]
    for e in layout.entries
        if e.kind === :coefficient
            for label in e.labels
                push!(names, Symbol(string(e.predictor) * "." * string(label)))
            end
        elseif e.kind === :plate
            for i in 1:e.size
                push!(names, Symbol(string(e.name) * "." * string(i)))
            end
        elseif e.kind === :scan
            for label in e.labels
                push!(names, Symbol(string(e.name) * "." * string(label)))
            end
        else
            push!(names, e.name)
        end
    end
    return names
end

"""
    constrain(layout, unconstrained) -> NamedTuple

Host-side constrain: packed vector → `(predictor => Vector, param => scalar,
…)`. For testing, output mapping, and future prediction; the generator emits
the in-graph equivalent.
"""
function constrain(layout::LayoutTable, u::AbstractVector{<:Real})
    length(u) == layout.total ||
        throw(ContractValidationError("[layout] unconstrained length $(length(u)) ≠ $(layout.total)"))
    pairs = Pair{Symbol,Any}[]
    for e in layout.entries
        seg = u[e.offset:(e.offset + e.size - 1)]
        if e.kind === :coefficient
            push!(pairs, e.predictor => Vector{Float64}(seg))
        elseif e.kind === :plate
            v = [_constrain_elt(e, Float64(x)) for x in seg]
            push!(pairs, e.name => v)
        elseif e.kind === :scan
            push!(pairs, e.name => Vector{Float64}(seg))
        else
            v = _constrain_elt(e, Float64(only(seg)))
            push!(pairs, e.name => v)
        end
    end
    return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
end

"""
    unconstrain(layout, constrained) -> Vector{Float64}

Inverse of [`constrain`](@ref): named values → packed unconstrained vector.
"""
function unconstrain(layout::LayoutTable, nt::NamedTuple)
    u = Vector{Float64}(undef, layout.total)
    for e in layout.entries
        if e.kind === :coefficient
            haskey(nt, e.predictor) || throw(
                ContractValidationError("[layout] missing predictor $(e.predictor)"),
            )
            v = nt[e.predictor]
            length(v) == e.size || throw(
                ContractValidationError("[layout] predictor $(e.predictor) length mismatch"),
            )
            u[e.offset:(e.offset + e.size - 1)] .= Float64.(v)
        elseif e.kind === :plate
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing plate parameter $(e.name)"),
            )
            v = nt[e.name]
            length(v) == e.size || throw(
                ContractValidationError("[layout] plate parameter $(e.name) length mismatch"),
            )
            for (k, x) in enumerate(v)
                u[e.offset + k - 1] = _unconstrain_elt(e, Float64(x))
            end
        elseif e.kind === :scan
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing scan state $(e.name)"),
            )
            v = nt[e.name]
            length(v) == e.size || throw(
                ContractValidationError("[layout] scan state $(e.name) length mismatch"),
            )
            u[e.offset:(e.offset + e.size - 1)] .= Float64.(v)
        else
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing parameter $(e.name)"),
            )
            u[e.offset] = _unconstrain_elt(e, Float64(nt[e.name]))
        end
    end
    return u
end

"""
    logjac(layout, unconstrained) -> Float64

Host-side total log-Jacobian, using the identical operations as the
in-graph terms (positive: the unconstrained value; unit:
`log(x) + log1p(-x)` from the constrained value).
"""
function logjac(layout::LayoutTable, u::AbstractVector{<:Real})
    length(u) == layout.total ||
        throw(ContractValidationError("[layout] unconstrained length $(length(u)) ≠ $(layout.total)"))
    total = 0.0
    for e in layout.entries
        e.transform === :identity && continue
        seg = u[e.offset:(e.offset + e.size - 1)]
        for v in seg
            total += _logjac_elt(e, Float64(v))
        end
    end
    return total
end

# Host transform values route through the bijector library (the single source
# of truth shared with the in-graph splices); `:identity` is a genuine no-op.
_constrain_value(transform::Symbol, u) =
    transform === :identity ? u : _prepared_endpoint(transform, :constrain)(u)
_unconstrain_value(transform::Symbol, x) =
    transform === :identity ? x : _prepared_endpoint(transform, :unconstrain)(x)
_logjac_value(transform::Symbol, u) =
    transform === :identity ? 0.0 : _prepared_endpoint(transform, :logjac)(u)

# Entry-level constrain/unconstrain/log-Jacobian. `:exp`/`:logistic` route
# through the bijector library (`_constrain_value(transform::Symbol, …)`, the
# single source of truth shared with the in-graph splices). An `:interval`
# transform is PARAMETERIZED by per-entry bounds, so it does not fit the
# Symbol-keyed parameterless bijector registry (like `:identity`, it lives
# outside it): it maps ℝ → (lo, hi) via an affine-logistic (`lo + (hi-lo)·σ(u)`)
# with log-Jacobian `log(x-lo) + log(hi-x) - log(hi-lo)` in the CONSTRAINED
# value. The in-graph interval edges (below) use the IDENTICAL operations, so
# host and graph agree bit-for-bit.
_constrain_elt(e::LayoutEntry, u) = e.transform === :interval ?
    (e.lo + (e.hi - e.lo) / (1 + exp(-u))) : _constrain_value(e.transform, u)
_unconstrain_elt(e::LayoutEntry, x) = e.transform === :interval ?
    (log(x - e.lo) - log(e.hi - x)) : _unconstrain_value(e.transform, x)
function _logjac_elt(e::LayoutEntry, u)
    e.transform === :interval || return _logjac_value(e.transform, u)
    x = e.lo + (e.hi - e.lo) / (1 + exp(-u))
    return log(x - e.lo) + log(e.hi - x) - log(e.hi - e.lo)
end

"""
    coordinate_read(offset) -> Expr

Scalar packed-coordinate read (`sum(view(unconstrained, o:o))`,
allocation-free, `@ppl` shape).
"""
coordinate_read(offset::Int) = :(sum(view(unconstrained, $offset:$offset)))

"""
    block_read(offset, width) -> Expr

Packed-slice read (`view(unconstrained, lo:hi)`, `@ppl` vector shape).
"""
block_read(offset::Int, width::Int) =
    :(view(unconstrained, $offset:$(offset + width - 1)))

"""
    transform_statements(entry) -> Vector{Expr}

In-graph constrain edge(s) for one layout entry. Coefficient/scan slices read a
`view`; a `:identity` sampled entry reads its scalar coordinate; a constrained
sampled entry (`:exp`/`:logistic`) splices the bijector's `constrain` endpoint,
which the planner inlines.
"""
function transform_statements(e::LayoutEntry)
    if e.kind === :coefficient || e.kind === :scan
        # both are identity array slices read into `e.name` (a coefficient
        # block name, or a scan-state name the emitter reconstructs from)
        lo = e.offset
        hi = e.offset + e.size - 1
        return Expr[:($(e.name)::AbstractVector{Float64} =
            view(unconstrained, $lo:$hi))]
    end
    if e.kind === :plate
        return _plate_transform_statements(e)
    end
    o = e.offset
    coord = coordinate_read(o)
    e.transform === :identity && return Expr[:($(e.name)::Float64 = $coord)]
    if e.transform === :interval
        # An :interval transform is parameterized by per-entry bounds, so it is
        # not in the Symbol-keyed (parameterless) bijector registry — its
        # forward + inverse edges are hand-rolled here, with the IDENTICAL math
        # to the host `_*_elt` path so in-graph and host agree bit-for-bit.
        u = Symbol(:_ppl_int_, e.name)
        blo, bhi = e.lo, e.hi
        return Expr[
            :($u::Float64 = $coord),
            :($(e.name)::Float64 = $blo + ($bhi - $blo) / (1 + exp(-$u))),
            :($u::Float64 = log($(e.name) - $blo) - log($bhi - $(e.name))),
        ]
    end
    # Constrained supports splice the bijector's `constrain` endpoint; the
    # planner inlines it (no runtime call survives) and shares `coord` with the
    # Jacobian term via structural CSE.
    bij = _bijector_name(e.transform)
    return Expr[:($(e.name)::Float64 = $(bij)().constrain($coord))]
end

# Per-cell latent (plate) block: the scalar transform edges, broadcast over
# the block view (identical operations to the host constrain/logjac path, so
# in-graph and host agree bit-for-bit). Bidirectional forward+inverse edges
# match the coefficient/scalar `@ppl` shape so the planner resolves either
# direction.
function _plate_transform_statements(e::LayoutEntry)
    lo = e.offset
    hi = e.offset + e.size - 1
    view_read = :(view(unconstrained, $lo:$hi))
    if e.transform === :identity
        return Expr[:($(e.name)::AbstractVector{Float64} = $view_read)]
    elseif e.transform === :exp
        u = Symbol(:_ppl_log_, e.name)
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} = exp.($u)),
            :($u::AbstractVector{Float64} = log.($(e.name))),
        ]
    elseif e.transform === :logistic
        u = Symbol(:_ppl_logit_, e.name)
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} = 1 ./ (1 .+ exp.(-$u))),
            :($u::AbstractVector{Float64} = log.($(e.name)) .- log1p.(-$(e.name))),
        ]
    elseif e.transform === :interval
        u = Symbol(:_ppl_int_, e.name)
        blo, bhi = e.lo, e.hi
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} =
                $blo .+ ($bhi - $blo) ./ (1 .+ exp.(-$u))),
            :($u::AbstractVector{Float64} =
                log.($(e.name) .- $blo) .- log.($bhi .- $(e.name))),
        ]
    else
        throw(ContractValidationError("[layout] unknown transform $(e.transform)"))
    end
end

"""
    jacobian_term(entry) -> Union{Nothing,Expr}

This entry's log-Jacobian contribution (`nothing` for identity). A scalar
constrained support splices the bijector's `logjac` endpoint over the same
coordinate the `constrain` edge reads (shared via structural CSE); a per-cell
latent (plate) block contributes the SUM over its cells from its hand-rolled
broadcast edges.
"""
function jacobian_term(e::LayoutEntry)
    e.transform === :identity && return nothing
    if e.kind === :coefficient
        throw(ContractValidationError("[layout] non-identity coefficient block"))
    end
    if e.kind === :plate
        if e.transform === :exp
            u = Symbol(:_ppl_log_, e.name)
            return :(sum($u))
        elseif e.transform === :logistic
            return :(sum(log.($(e.name)) .+ log1p.(-$(e.name))))
        elseif e.transform === :interval
            blo, bhi = e.lo, e.hi
            return :(sum(log.($(e.name) .- $blo) .+ log.($bhi .- $(e.name)) .-
                         log($bhi - $blo)))
        else
            throw(ContractValidationError("[layout] unknown transform $(e.transform)"))
        end
    end
    if e.transform === :interval
        # Parameterized bounds ⇒ hand-rolled (not in the bijector registry);
        # uses the constrained `e.name` from the interval constrain edge.
        blo, bhi = e.lo, e.hi
        return :(log($(e.name) - $blo) + log($bhi - $(e.name)) - log($bhi - $blo))
    end
    bij = _bijector_name(e.transform)
    return :($(bij)().logjac($(coordinate_read(e.offset))))
end
