# Packed layout + transforms (D5b thin-layer-owned).
#
# Constrained-parameter transforms are provided by the reusable bijector
# library (`bijectors.jl`, decision 0l3dsru): the host path runs the bijector's
# prepared endpoints and the in-graph generator splices the same endpoints, so
# in-graph and host evaluation agree by construction (structural), not by
# hand-kept duplication. A SCALAR sampled entry splices the endpoints directly;
# a per-cell (plate) block maps the same scalar endpoints over its cell view via
# the `plate` primitive (host + graph both). `:identity` (real support) is a
# genuine no-op and stays a direct read in both paths.

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

"""One packed slice: a coefficient block, a scalar latent, a scan vector
latent, a per-cell latent block, a leveled vector latent (cutpoints,
thresholds, simplex), a spline coefficient-vector block, or a K=1 ranef
standardized group vector. `lo`/`hi` are the constrained bounds of an
`:interval` transform (`NaN` otherwise)."""
struct LayoutEntry
    kind::Symbol # :coefficient | :sampled | :scan | :plate | :vector | :spline | :ranef
    predictor::Union{Nothing,Symbol}
    name::Symbol # block name (`mu_coef`), parameter name, or scan-state name
    labels::Vector{Symbol} # per-coordinate labels (length == size)
    offset::Int # 1-based packed offset
    size::Int
    transform::Symbol # :identity | :exp | :logistic | :interval | :ordered | :simplex
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
then sampled parameters, plate parameters, and vector parameters in plan
order (pinned, deterministic). Assumes `validate_plan` passed.
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
    for p in plan.vector_parameters
        p.size === nothing && throw(ContractValidationError(
            "[layout] vector parameter $(p.name) has unresolved size " *
            "(bind_data infers it from the linked response)"))
        transform = p.family === :ordered_normal ? :ordered :
            p.family === :simplex_dirichlet ? :simplex : :identity
        # `size` is the PACKED (unconstrained) length: a simplex packs
        # K−1 stick-breaking logits (a 1-simplex packs zero — Stan's
        # deterministic `[1.0]`); threshold vectors pack their size.
        # Constrained length recovers as `_vector_constrained_size`.
        packed = transform === :simplex ? p.size - 1 : p.size
        labels = [Symbol(string(p.name) * "." * string(i)) for i in 1:packed]
        push!(entries,
            LayoutEntry(:vector, nothing, p.name, labels, offset, packed,
                transform))
        offset += packed
    end
    # Spline coefficient vectors: one contiguous block per SplineVector
    # (plate-shaped; priors broadcast over cells). Width is static from k.
    for v in plan.spline_vectors
        transform, lo, hi = _entry_transform(v.family, v.support_override)
        push!(entries,
            LayoutEntry(:spline, nothing, v.name, [v.name], offset, v.width,
                transform, lo, hi))
        offset += v.width
    end
    # K=1 ranef buckets (Stage B): the scalar scale (`:sampled`, real for
    # `log_scale`, exp for `tau` — the exp Jacobian is Stan's lower-bound
    # kernel term, no renormalizer) plus the standardized G-vector `xi`
    # (`:ranef`, plate-shaped identity block; G from bind levels).
    # Correlated buckets own no Stage-B entries (Stage C).
    for b in plan.ranef_buckets
        b.kind === :correlated && continue
        scale, xi = _ranef_k1_names(b)
        transform = b.kind === :intercept1 ? :identity : :exp
        push!(entries,
            LayoutEntry(:sampled, nothing, scale, [scale], offset, 1,
                transform))
        offset += 1
        G = length(_grouping_levels(plan.columns[b.group]))
        push!(entries,
            LayoutEntry(:ranef, nothing, xi, [xi], offset, G, :identity))
        offset += G
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

"""Constrained length of a `:vector` entry: simplex packs K−1 logits
for K probabilities; threshold vectors pack their size."""
_vector_constrained_size(e::LayoutEntry) =
    e.transform === :simplex ? e.size + 1 : e.size

# Scalar logistic (Stan `inv_logit`) and logit, shared by the host
# simplex edges and their in-graph twins below.
_ordered_sigma(x::Float64) = 1.0 / (1.0 + exp(-x))
_simplex_logit(z::Float64) = log(z) - log1p(-z)

"""
    ordered_constrain(u) -> Vector{Float64}

Host-side ordered transform (Stan `ordered`): `y[1] = u[1]`,
`y[i] = y[i-1] + exp(u[i])`. The in-graph twin unrolls the identical
scalar chain.
"""
function ordered_constrain(u::AbstractVector{<:Real})
    y = Vector{Float64}(undef, length(u))
    for (i, x) in enumerate(u)
        y[i] = i == 1 ? Float64(x) : y[i-1] + exp(Float64(x))
    end
    return y
end

"""Inverse of [`ordered_constrain`](@ref)."""
function ordered_unconstrain(y::AbstractVector{<:Real})
    u = Vector{Float64}(undef, length(y))
    prev = 0.0
    for (i, v) in enumerate(y)
        u[i] = i == 1 ? Float64(v) : log(Float64(v) - prev)
        prev = Float64(v)
    end
    return u
end

"""Log-Jacobian of [`ordered_constrain`](@ref): `Σ u[2:end]`."""
function ordered_logjac(u::AbstractVector{<:Real})
    total = 0.0
    for (i, x) in enumerate(u)
        i > 1 && (total += Float64(x))
    end
    return total
end

"""
    simplex_constrain(u) -> Vector{Float64}

Host-side stick-breaking simplex transform (Stan `simplex_constrain`):
`z[j] = σ(u[j] + log(K-j))`, `s[j] = remaining[j]*z[j]`,
`s[K] = remaining[K]`. Empty input constrains to `[1.0]` (Stan's
deterministic 1-simplex). The in-graph twin unrolls the identical
scalar chain.
"""
function simplex_constrain(u::AbstractVector{<:Real})
    K = length(u) + 1
    s = Vector{Float64}(undef, K)
    K == 1 && (s[1] = 1.0; return s)
    remaining = 1.0
    for (j, x) in enumerate(u)
        z = _ordered_sigma(Float64(x) + log(K - j))
        s[j] = remaining * z
        remaining -= s[j]
    end
    s[K] = remaining
    return s
end

"""Inverse of [`simplex_constrain`](@ref) (Stan `simplex_free`)."""
function simplex_unconstrain(s::AbstractVector{<:Real})
    K = length(s)
    u = Vector{Float64}(undef, max(K - 1, 0))
    K <= 1 && return u
    remaining = 1.0
    for (j, v) in enumerate(s)
        j >= K && break
        z = Float64(v) / remaining
        u[j] = _simplex_logit(z) - log(K - j)
        remaining -= Float64(v)
    end
    return u
end

"""Log-Jacobian of [`simplex_constrain`](@ref): Stan's
`Σ [log(remaining) + log(z) + log1p(-z)]` over the K−1 breaks."""
function simplex_logjac(u::AbstractVector{<:Real})
    K = length(u) + 1
    K == 1 && return 0.0
    total = 0.0
    remaining = 1.0
    for (j, x) in enumerate(u)
        z = _ordered_sigma(Float64(x) + log(K - j))
        total += log(remaining) + log(z) + log1p(-z)
        remaining -= remaining * z
    end
    return total
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
        elseif e.kind === :plate || e.kind === :spline || e.kind === :ranef
            for i in 1:e.size
                push!(names, Symbol(string(e.name) * "." * string(i)))
            end
        elseif e.kind === :vector
            append!(names, e.labels)
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
        elseif e.kind === :plate || e.kind === :spline || e.kind === :ranef
            v = [_constrain_elt(e, Float64(x)) for x in seg]
            push!(pairs, e.name => v)
        elseif e.kind === :scan
            push!(pairs, e.name => Vector{Float64}(seg))
        elseif e.kind === :vector
            push!(pairs, e.name => _vector_constrain(e, seg))
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
        elseif e.kind === :plate || e.kind === :spline || e.kind === :ranef
            what = e.kind === :plate ? "plate parameter" :
                e.kind === :spline ? "spline vector" : "ranef xi vector"
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing $what $(e.name)"),
            )
            v = nt[e.name]
            length(v) == e.size || throw(
                ContractValidationError("[layout] $what $(e.name) length mismatch"),
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
        elseif e.kind === :vector
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing vector parameter $(e.name)"),
            )
            v = nt[e.name]
            want = _vector_constrained_size(e)
            length(v) == want || throw(
                ContractValidationError("[layout] vector parameter $(e.name) length mismatch"),
            )
            u[e.offset:(e.offset + e.size - 1)] .= _vector_unconstrain(e, v)
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
        if e.kind === :vector
            # Vector Jacobians couple coordinates (ordered sums, simplex
            # stick-breaking) — entry-level, never per-coordinate.
            total += _vector_logjac(e, seg)
            continue
        end
        for v in seg
            total += _logjac_elt(e, Float64(v))
        end
    end
    return total
end

# Entry-level vector edges (host side). The in-graph twins in
# `_vector_transform_statements`/`jacobian_term` unroll the IDENTICAL
# scalar chains, so host and graph agree bit-for-bit.
_vector_constrain(e::LayoutEntry, seg) =
    e.transform === :identity ? Vector{Float64}(seg) :
    e.transform === :ordered ? ordered_constrain(seg) :
    simplex_constrain(seg)
_vector_unconstrain(e::LayoutEntry, v) =
    e.transform === :identity ? Vector{Float64}(v) :
    e.transform === :ordered ? ordered_unconstrain(v) :
    simplex_unconstrain(v)
_vector_logjac(e::LayoutEntry, seg) =
    e.transform === :ordered ? ordered_logjac(seg) : simplex_logjac(seg)

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
    if e.kind === :plate || e.kind === :spline || e.kind === :ranef
        # Spline vectors ride the plate transform path (block + scalar
        # endpoints); the contract pins their supports to real/positive,
        # so the :interval arm below is unreachable for them. Ranef `xi`
        # vectors ride it too (always `:identity` in Stage B).
        return _plate_transform_statements(e)
    end
    if e.kind === :vector
        return _vector_transform_statements(e)
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

# The per-cell log-Jacobian vector name a plate block's `jacobian_term` sums.
_plate_logjac_name(name::Symbol) = Symbol(:_ppl_ljcells_, name)

# `plate(view) do cell; body; end` as an Expr (an allocation-free per-cell loop).
function _plate_map(view_read, cell::Symbol, body)
    lambda = Expr(:(->), Expr(:tuple, cell),
        Expr(:block, LineNumberNode(0, :layout), body))
    return Expr(:do, Expr(:call, :plate, view_read), lambda)
end

# Per-cell latent (plate) block: map the SCALAR bijector endpoints over the
# block view via the `plate` primitive — the same library the scalar and host
# paths use, so in-graph and host agree by construction rather than by a
# hand-kept broadcast. `constrain` yields the constrained cell vector; a
# companion `logjac` plate supplies the per-cell Jacobian this block's
# `jacobian_term` sums (pruned by have→want when the Jacobian is not wanted).
# `:identity` (real support) is a direct view read — no transform, no Jacobian.
function _plate_transform_statements(e::LayoutEntry)
    lo = e.offset
    hi = e.offset + e.size - 1
    view_read = :(view(unconstrained, $lo:$hi))
    e.transform === :identity &&
        return Expr[:($(e.name)::AbstractVector{Float64} = $view_read)]
    if e.transform === :interval
        # Parameterized bounds ⇒ not in the (parameterless) bijector registry;
        # hand-rolled broadcast edges over the block view, identical math to the
        # host `_*_elt` path so in-graph and host agree bit-for-bit. Its
        # `jacobian_term` sums the same expression (below), so no companion
        # `logjac` plate is emitted.
        u = Symbol(:_ppl_int_, e.name)
        blo, bhi = e.lo, e.hi
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} =
                $blo .+ ($bhi - $blo) ./ (1 .+ exp.(-$u))),
            :($u::AbstractVector{Float64} =
                log.($(e.name) .- $blo) .- log.($bhi .- $(e.name))),
        ]
    end
    bij = _bijector_name(e.transform)
    cell = Symbol(:_ppl_cell_, e.name)
    ljcells = _plate_logjac_name(e.name)
    return Expr[
        :($(e.name)::AbstractVector{Float64} =
            $(_plate_map(view_read, cell, :($(bij)().constrain($cell))))),
        :($(ljcells)::AbstractVector{Float64} =
            $(_plate_map(view_read, cell, :($(bij)().logjac($cell))))),
    ]
end

# Constrained-element / stick-breaking-temporary names for a `:vector`
# entry: `_ppl_v_<name>_<i>` (elements, shared with the generator's prior
# and likelihood cells), `_ppl_vz_<name>_<j>` (break fractions),
# `_ppl_vr_<name>_<j>` (remainders). All `_ppl_`-hygienic.
_vector_elt_name(name::Symbol, i::Int) = Symbol(:_ppl_v_, name, :_, i)
_vector_elt(e::LayoutEntry, i::Int) = _vector_elt_name(e.name, i)
_vector_z(e::LayoutEntry, j::Int) = Symbol(:_ppl_vz_, e.name, :_, j)
_vector_r(e::LayoutEntry, j::Int) = Symbol(:_ppl_vr_, e.name, :_, j)

# Leveled vector edges: scalar-unrolled twins of the host
# `ordered_constrain`/`simplex_constrain` (IDENTICAL scalar ops in the
# IDENTICAL order, so in-graph and host agree bit-for-bit). No vector
# ever materializes in-graph: downstream prior/likelihood cells read the
# `_ppl_v_` scalars directly (fully transparent to the planner and the
# reverse pass — no new Enzyme surface). Empty threshold vectors emit no
# statements; a 1-simplex emits its constant `[1.0]`.
function _vector_transform_statements(e::LayoutEntry)
    if e.transform === :identity
        return Expr[:($(_vector_elt(e, i))::Float64 =
            $(coordinate_read(e.offset + i - 1))) for i in 1:e.size]
    end
    if e.transform === :ordered
        stmts = Expr[]
        for i in 1:e.size
            t = _vector_elt(e, i)
            if i == 1
                push!(stmts, :($t::Float64 = $(coordinate_read(e.offset))))
            else
                prev = _vector_elt(e, i - 1)
                push!(stmts, :($t::Float64 =
                    $prev + exp($(coordinate_read(e.offset + i - 1)))))
            end
        end
        return stmts
    end
    # :simplex — Stan stick-breaking, unrolled.
    K = e.size + 1
    stmts = Expr[]
    K == 1 && return Expr[:($(_vector_elt(e, 1))::Float64 = 1.0)]
    push!(stmts, :($(_vector_r(e, 1))::Float64 = 1.0))
    for j in 1:K-1
        z = _vector_z(e, j)
        s = _vector_elt(e, j)
        r = _vector_r(e, j)
        logK = log(K - j)
        push!(stmts, :($z::Float64 =
            1.0 / (1.0 + exp(-($(coordinate_read(e.offset + j - 1)) + $logK)))))
        push!(stmts, :($s::Float64 = $r * $z))
        push!(stmts, :($(_vector_r(e, j + 1))::Float64 = $r - $s))
    end
    push!(stmts, :($(_vector_elt(e, K))::Float64 = $(_vector_r(e, K))))
    return stmts
end

"""
    jacobian_term(entry) -> Union{Nothing,Expr}

This entry's log-Jacobian contribution (`nothing` for identity). A scalar
constrained support splices the bijector's `logjac` endpoint over the same
coordinate the `constrain` edge reads (shared via structural CSE); a per-cell
latent (plate) block sums its companion `logjac` plate (`_plate_logjac_name`);
a leveled vector entry sums its unrolled twin of the host
`ordered_logjac`/`simplex_logjac` (shared coordinates via CSE).
"""
function jacobian_term(e::LayoutEntry)
    e.transform === :identity && return nothing
    if e.kind === :coefficient
        throw(ContractValidationError("[layout] non-identity coefficient block"))
    end
    if e.kind === :vector
        if e.transform === :ordered
            e.size < 2 && return nothing
            terms = Any[coordinate_read(e.offset + i - 1) for i in 2:e.size]
            return foldl((a, b) -> :($a + $b), terms)
        end
        # :simplex — Stan's Σ [log(r) + log(z) + log1p(-z)] over the
        # breaks, reading the `_vector_transform_statements` temps.
        e.size < 1 && return nothing
        terms = Any[:(log($(_vector_r(e, j))) + log($(_vector_z(e, j))) +
            log1p(-$(_vector_z(e, j)))) for j in 1:e.size]
        return foldl((a, b) -> :($a + $b), terms)
    end
    if e.kind === :plate || e.kind === :spline || e.kind === :ranef
        # Interval plates hand-roll the per-cell Jacobian sum (parameterized
        # bounds, no companion `logjac` plate); every registry transform sums
        # its companion `logjac` plate from `_plate_transform_statements`.
        # Spline vectors share the shape (their supports never reach
        # :interval, but the arm stays correct if that ever changes), as do
        # ranef `xi` vectors (always `:identity` in Stage B).
        if e.transform === :interval
            blo, bhi = e.lo, e.hi
            return :(sum(log.($(e.name) .- $blo) .+ log.($bhi .- $(e.name)) .-
                         log($bhi - $blo)))
        end
        return :(sum($(_plate_logjac_name(e.name))))
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
