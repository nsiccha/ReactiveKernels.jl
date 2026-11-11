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

Inferred unconstrained support (`:real`/`:positive`/`:unit`/`:interval`)
for a sampled family plus an optional support override (`:positive`
half-Normal/half-Cauchy style, `:truncated`, `:interval`, or `:upper`).
`:uniform` infers `:interval` from its own
literal args and takes no override. Loud on unknown families and
inapplicable overrides.
"""
function support_of(family::Symbol, override::SupportOverride)
    haskey(SAMPLED_SUPPORT, family) ||
        throw(ContractValidationError("[layout] sampled family $family unknown"))
    inferred = SAMPLED_SUPPORT[family]
    override === nothing && return inferred
    _has_interval_bounds(override) && return :truncated
    family === :uniform && throw(ContractValidationError(
        "[layout] a uniform prior carries its own interval support — no " *
        "support override applies, got $override"))
    if override isa Tuple
        if override[1] === :lower
            inferred === :positive || throw(ContractValidationError(
                "[layout] :lower override needs a positive-support family"))
            return :floored
        end
        if override[1] === :upper
            length(override) == 2 || throw(ContractValidationError(
                "[layout] tuple support override must be (:upper, hi), got $override"))
            inferred === :real || throw(ContractValidationError(
                "[layout] :upper override needs a real-support family"))
            return :upper
        end
        head = override[1]
        (head === :interval &&
            length(override) == 3) ||
            throw(ContractValidationError(
                "[layout] tuple support override must be (:interval, lo, hi), " *
                "(:interval, lo, hi), or (:upper, hi), got $override"))
        inferred === :real || throw(ContractValidationError(
            "[layout] $head override needs a real-support family"))
        return :interval
    end
    override === :positive || throw(
        ContractValidationError("[layout] support override must be :positive, got $override"),
    )
    inferred === :real || throw(
        ContractValidationError("[layout] $override override needs a real-support family"),
    )
    return :positive
end

# Fold bound data and data-only definitions, retaining sampled dependencies.
_bound_override(::StructuralPlan, ov) = ov
function _bound_override(plan::StructuralPlan, ov::Tuple)
    return (ov[1], (_layout_bound(plan, x) for x in ov[2:end])...)
end
function _layout_bound(plan::StructuralPlan, x)
    x isa Real && return Float64(x)
    exprs = Dict{Symbol,Any}(a.name => a.expr for a in plan.assignments)
    merge!(exprs, Dict(d.name => d.expr for d in plan.derived))
    function lookup(nm)
        _holds_bound_value(plan, plan.columns, nm) && return _bound_value(plan, plan.columns, nm)
        haskey(exprs, nm) && return _memoized_data(plan, nm) do
            _eval_value_expr(exprs[nm], lookup, nm)
        end
        return nm
    end
    # A definition reading sampled values remains a graph expression.
    refs = _value_symbols(x)
    function isdata(nm, seen = Set{Symbol}())
        _holds_bound_value(plan, plan.columns, nm) && return true
        nm in seen && return false
        haskey(exprs, nm) || return false
        all(r -> isdata(r, union(seen, Set([nm]))), _value_symbols(exprs[nm]))
    end
    all(isdata, refs) || return x
    v = _eval_value_expr(x, lookup, :truncation_bound)
    v isa Real && !isnan(v) || throw(ContractValidationError(
        "[layout] truncation bound must be a number, got $(summary(v))"))
    return Float64(v)
end

# The transform kind and (for :interval/:upper) the constrained bounds a
# layout entry needs, from a parameter's family + support override (+ args:
# a uniform's bounds come from its literal args, not an override).
function _entry_transform(family::Symbol, override::SupportOverride,
        args::NamedTuple)
    support = support_of(family, override)
    if support === :truncated
        lo, hi = override[2], override[3]
        natural = SAMPLED_SUPPORT[family]
        clip(fn, a, b) = a isa Real && b isa Real ? fn(a, b) :
            Expr(:call, nameof(fn), a, b)
        if natural === :positive || natural === :unit
            lo = clip(max, lo, 0.0)
        end
        natural === :unit && (hi = clip(min, hi, 1.0))
        if natural === :interval
            lo = clip(max, lo, args.arg1)
            hi = clip(min, hi, args.arg2)
        end
        if lo isa Real && hi isa Real
            lo < hi || throw(ContractValidationError(
                "[layout] truncation has no mass in the base support: ($lo, $hi)"))
        end
        lo == -Inf && hi == Inf && return (:identity, NaN, NaN)
        hi == Inf && return (:floored, lo, NaN)
        lo == -Inf && return (:upper, NaN, hi)
        return (:interval, lo, hi)
    end
    if support === :interval
        family === :uniform &&
            return (:interval, args.arg1, args.arg2)
        return (:interval, Float64(override[2]), Float64(override[3]))
    end
    if support === :upper
        return (:upper, NaN, Float64(override[2]))
    end
    if support === :floored
        override[2] isa Real || throw(ContractValidationError(
            "[layout] the lower bound `$(override[2])` is a data name — " *
            "bind the plan first (`bind_data`)"))
        return (:floored, Float64(override[2]), NaN)
    end
    transform =
        support === :real ? :identity :
        support === :positive ? :exp : :logistic
    return (transform, NaN, NaN)
end

"""One packed slice: a coefficient block, a scalar latent, a scan vector
latent, a per-cell latent block, a leveled vector latent (cutpoints,
thresholds, simplex), a joint-outcomes LKJ Cholesky factor, or an
elementwise array parameter
(`:array`, column-major over `dims`). `lo` is the constrained lower
bound of an `:interval`/`:floored` transform (`:interval` also sets
`hi`; an `:upper` transform sets `hi` only; `NaN` otherwise). A real
bound is stored as `Float64`, or in the number type of the sampled values it
was resolved from (an AD dual while differentiating `constrain`). `dims` is
the axes of a declared array, including an LKJ factor (empty for legacy
scalar/block entries)."""
struct LayoutEntry
    kind::Symbol # :coefficient | :sampled | :scan | :plate | :vector | :cholesky_corr | :glm | :array
    predictor::Union{Nothing,Symbol}
    name::Symbol # block name (`mu_coef`), parameter name, or scan-state name
    labels::Vector{Symbol} # per-coordinate labels (length == size)
    offset::Int # 1-based packed offset
    size::Int
    transform::Symbol # :identity | :exp | :logistic | :interval | :floored | :upper | :ordered | :simplex | :lkj
    lo::Union{Real,Symbol,Expr} # constrained lower bound (else NaN)
    hi::Union{Real,Symbol,Expr} # constrained upper bound (else NaN)
    dims::Vector{Int} # declared-array axes, including :cholesky_corr (else empty)
    scan_blocks::Vector{LayoutEntry} # structural latent-family blocks (else empty)
    sampling::Union{Nothing,NamedTuple} # external RHS and its geometry
    # How the generated program reads what the data determine, so that it does
    # not depend on this binding's sizes or numbers (`_layout_reads`). `positions`
    # is the packed range as `(first, last)`, each a literal or a name; `nothing`
    # reads the literal `offset:offset + size - 1`. `axes` is the packed block's
    # shape when an axis reads data. `bound_reads` is the authored data expression
    # of `lo`/`hi`, which keep their numbers for host transforms.
    positions::Union{Nothing,NTuple{2,Union{Int,Symbol}}}
    axes::Union{Nothing,Vector{Any}}
    bound_reads::NTuple{2,Any}
    LayoutEntry(kind, predictor, name, labels, offset, size, transform, lo, hi,
            dims, scan_blocks, sampling, positions = nothing, axes = nothing,
            bound_reads = (nothing, nothing)) =
        new(kind, predictor, name, labels, offset, size, transform,
            _layout_bound(lo), _layout_bound(hi), dims, scan_blocks, sampling,
            positions, axes, bound_reads)
end

# Host transforms (`constrain`, `unconstrain`, `logjac` and the exported
# vector/LKJ helpers) follow the number type of their input, so dual numbers,
# `BigFloat` and other reals pass through. Plain reals keep at least `Float64`
# precision, so `Float64`, `Float32` and integer inputs give the same
# `Float64` results as before.
_host_type(::Type{T}) where {T} = promote_type(Float64, T)
_host_eltype(x) = _host_type(eltype(x))
_host_real(x::Real) = convert(_host_type(typeof(x)), x)
_layout_bound(x::Real) = _host_real(x)
_layout_bound(x) = x
_bound_type(x::Real) = typeof(x)
_bound_type(x) = Float64
# Element type of an entry's constrained values for input type `T`: a bound
# resolved from sampled values can carry a dual of its own.
_entry_eltype(e::LayoutEntry, ::Type{T}) where {T} =
    promote_type(T, _bound_type(e.lo), _bound_type(e.hi))
LayoutEntry(kind::Symbol, predictor::Union{Nothing,Symbol}, name::Symbol,
    labels::Vector{Symbol}, offset::Int, size::Int, transform::Symbol,
    lo, hi, dims::Vector{Int}, blocks::Vector{LayoutEntry}) =
    LayoutEntry(kind, predictor, name, labels, offset, size, transform, lo, hi,
        dims, blocks, nothing)
LayoutEntry(kind::Symbol, predictor::Union{Nothing,Symbol}, name::Symbol,
    labels::Vector{Symbol}, offset::Int, size::Int, transform::Symbol,
    lo, hi, dims::Vector{Int}) =
    LayoutEntry(kind, predictor, name, labels, offset, size, transform, lo, hi,
        dims, LayoutEntry[])
# Legacy scalar/block entries carry no axes.
LayoutEntry(kind::Symbol, predictor::Union{Nothing,Symbol}, name::Symbol,
    labels::Vector{Symbol}, offset::Int, size::Int, transform::Symbol,
    lo, hi) =
    LayoutEntry(kind, predictor, name, labels, offset, size, transform, lo,
        hi, Int[])
# Non-interval entries omit the bounds.
LayoutEntry(kind::Symbol, predictor::Union{Nothing,Symbol}, name::Symbol,
    labels::Vector{Symbol}, offset::Int, size::Int, transform::Symbol) =
    LayoutEntry(kind, predictor, name, labels, offset, size, transform, NaN, NaN)

# One block per sampled body statement, independent of the number of
# iterations, and one per run of adjacent sampled seeds that share a transform
# and its bounds: a seed is one coordinate, so such a run is one transformed
# vector rather than one vector per seed. Native and generated transforms use
# the existing bijectors, with the support a `truncated` / `restricted` / half
# prior states. A seed block's size is its number of seeds (the authored setup
# statements, never a data length); `_entry_reads!` relies on it.
function _scan_layout_entry(plan, s, name, offset, T)
    blocks = LayoutEntry[]
    seed_run = false # the last block holds seeds and may take the next one
    pos = offset
    for st in (s.setup..., s.step...)
        st.kind === :sample || continue
        width = st isa ScanSetup ? 1 : T - (s.lo - 1)
        width == 0 && continue
        args = NamedTuple{Tuple(Symbol(:arg, i) for i in eachindex(st.args))}(Tuple(st.args))
        transform, lo, hi = _entry_transform(st.family, st.support, args)
        bound_reads = (nothing, nothing)
        if transform in (:interval, :floored, :upper)
            # One block transforms every step with the same bounds.
            locals = Set(step.target for step in s.step if !step.indexed)
            bounds = union(_expr_value_symbols(lo), _expr_value_symbols(hi))
            (_scan_reads_carried(lo, s) || _scan_reads_carried(hi, s) ||
                s.loopvar in bounds || !isdisjoint(locals, bounds)) &&
                _fail(s.label, "scan prior bounds that vary by step (reading " *
                    "a carried array, the loop index or a step local) are not " *
                    "supported yet")
            authored = (lo, hi)
            lo, hi = _layout_bound(plan, lo), _layout_bound(plan, hi)
            bound_reads = _data_bound_reads(authored, lo, hi)
        end
        if st isa ScanSetup && seed_run && (last_block = blocks[end];
                last_block.transform === transform && isequal(last_block.lo, lo) &&
                isequal(last_block.hi, hi) && isequal(last_block.bound_reads, bound_reads))
            blocks[end] = LayoutEntry(:plate, nothing, last_block.name, Symbol[],
                last_block.offset, last_block.size + 1, transform, lo, hi, Int[],
                LayoutEntry[], nothing, nothing, nothing, bound_reads)
            pos += 1
            continue
        end
        seed_run = st isa ScanSetup
        blockname = Symbol(name, :_block_, length(blocks) + 1)
        push!(blocks, LayoutEntry(:plate, nothing, blockname, Symbol[], pos,
            width, transform, lo, hi, Int[], LayoutEntry[], nothing, nothing, nothing,
            bound_reads))
        pos += width
    end
    n = pos - offset
    all(e -> e.transform === :identity, blocks) &&
        return LayoutEntry(:scan, nothing, name, Symbol[Symbol(i) for i in 1:n], offset, n, :identity)
    return LayoutEntry(:scan, nothing, name, Symbol[Symbol(i) for i in 1:n],
        offset, n, :scan_blocks, NaN, NaN, Int[], blocks)
end

"""Packed unconstrained layout: ordered entries + total dimension."""
struct LayoutTable
    entries::Vector{LayoutEntry}
    total::Int
    name_paths::Dict{Symbol,Tuple{Vararg{Symbol}}}
    bound_values::Dict{Symbol,Any}
    bound_exprs::Dict{Symbol,Any}
    # The packed array of a dotted cell latent => (authored name, dimensions).
    cell_latents::Dict{Symbol,Tuple{Symbol,Symbol}}
    # The joint cell latent's packed array => its caller-owned geometry.
    cell_geometries::Dict{Symbol,CellGeometry}
    # The program's named data-derived positions and axis lengths (`_layout_reads`).
    reads::Vector{Expr}
    # A declared array's bound axis labels (`_array_axis_labels`): one entry per
    # constrained axis, the labels of a `levels(g)`, `unique(g)` or data-vector
    # axis, `nothing` for a positional one. Arrays with no label axis are absent.
    axis_labels::Dict{Symbol,Vector{Any}}
    # `total` as the program reads it (`_layout_reads`): the number when no
    # position reads data, else the name of the read that computes it. The
    # program checks its packed input's length against it (`_ppl_check_packed`).
    packed_total::Union{Int,Symbol}
end
LayoutTable(entries::Vector{LayoutEntry}, total::Int, paths::Dict{Symbol,Tuple{Vararg{Symbol}}},
        bound_values::Dict{Symbol,Any}, bound_exprs::Dict{Symbol,Any},
        cell_latents::Dict{Symbol,Tuple{Symbol,Symbol}},
        cell_geometries::Dict{Symbol,CellGeometry}, reads::Vector{Expr},
        axis_labels::Dict{Symbol,Vector{Any}}) =
    LayoutTable(entries, total, paths, bound_values, bound_exprs, cell_latents,
        cell_geometries, reads, axis_labels, total)
LayoutTable(entries::Vector{LayoutEntry}, total::Int, paths::Dict{Symbol,Tuple{Vararg{Symbol}}},
        bound_values::Dict{Symbol,Any}, bound_exprs::Dict{Symbol,Any},
        cell_latents::Dict{Symbol,Tuple{Symbol,Symbol}},
        cell_geometries::Dict{Symbol,CellGeometry}, reads::Vector{Expr}) =
    LayoutTable(entries, total, paths, bound_values, bound_exprs, cell_latents,
        cell_geometries, reads, Dict{Symbol,Vector{Any}}())
LayoutTable(entries::Vector{LayoutEntry}, total::Int, paths::Dict{Symbol,Tuple{Vararg{Symbol}}},
        bound_values::Dict{Symbol,Any}, bound_exprs::Dict{Symbol,Any},
        cell_latents::Dict{Symbol,Tuple{Symbol,Symbol}},
        cell_geometries::Dict{Symbol,CellGeometry}) =
    LayoutTable(entries, total, paths, bound_values, bound_exprs, cell_latents,
        cell_geometries, Expr[])
LayoutTable(entries::Vector{LayoutEntry}, total::Int, paths::Dict{Symbol,Tuple{Vararg{Symbol}}},
        bound_values::Dict{Symbol,Any}, bound_exprs::Dict{Symbol,Any},
        cell_latents::Dict{Symbol,Tuple{Symbol,Symbol}}) =
    LayoutTable(entries, total, paths, bound_values, bound_exprs, cell_latents,
        Dict{Symbol,CellGeometry}())
LayoutTable(entries::Vector{LayoutEntry}, total::Int, paths::Dict{Symbol,Tuple{Vararg{Symbol}}},
        bound_values::Dict{Symbol,Any}, bound_exprs::Dict{Symbol,Any}) =
    LayoutTable(entries, total, paths, bound_values, bound_exprs,
        Dict{Symbol,Tuple{Symbol,Symbol}}())
LayoutTable(entries::Vector{LayoutEntry}, total::Int, paths::Dict{Symbol,Tuple{Vararg{Symbol}}}) =
    LayoutTable(entries, total, paths, Dict{Symbol,Any}(), Dict{Symbol,Any}())
LayoutTable(entries::Vector{LayoutEntry}, total::Int) =
    LayoutTable(entries, total, Dict{Symbol,Tuple{Vararg{Symbol}}}())

"""
    _coefficient_runs(plan, pred, shape) -> Vector{NamedTuple}

Maximal transform runs over a predictor's design positions: uniform
prior rows mark interval elements (bounds from the row — validated
finite lo < hi by the time layout runs); everything else, including
blocks with no prior row, is identity. Each run is `(labels,
transform, lo, hi)` with per-position labels in design order.
"""
function _coefficient_runs(plan::StructuralPlan, pred, shape)
    rows = Dict{Symbol,PopulationPrior}()
    for pr in plan.population_priors
        pr.predictor === pred.name || continue
        rows[pr.addressee] = pr
    end
    runs = NamedTuple{(:labels, :transform, :lo, :hi),
        Tuple{Vector{Symbol},Symbol,Float64,Float64}}[]
    pushkey(lab, tr, lo, hi) = begin
        if !isempty(runs) && runs[end].transform === tr &&
                (tr === :identity ||
                    (runs[end].lo == lo && runs[end].hi == hi))
            push!(runs[end].labels, lab)
        else
            push!(runs, (labels = [lab], transform = tr, lo = lo, hi = hi))
        end
    end
    for b in shape.blocks
        b.width == 0 && continue
        if b.kind === MatrixTerm
            for (e, lab) in zip(b.elements, b.labels)
                addr = e === nothing ? :Intercept : e
                pushkey(lab, _coef_position_key(rows, addr)...)
            end
        else
            for lab in b.labels
                pushkey(lab, _coef_position_key(rows, b.addressee)...)
            end
        end
    end
    return runs
end

function _coef_position_key(rows::Dict{Symbol,PopulationPrior}, addr::Symbol)
    pr = get(rows, addr, nothing)
    pr === nothing && return (:identity, NaN, NaN)
    pr.family === :uniform &&
        return (:interval, Float64(pr.location), Float64(pr.scale))
    return (:identity, NaN, NaN)
end

# Legacy constructs own generated parameter blocks. Derive their displayed
# names from the same naming helpers, using the authored local id, so private
# identifiers cannot escape through scan innovations.
function _scope_layout_paths(plan::StructuralPlan)
    paths = _scope_name_paths(plan.submodel_scopes)
    isempty(paths) && return paths
    occupied = Set(values(paths))
    union!(occupied, (scope.path for scope in plan.submodel_scopes))
    function generated!(owner, internal, authored)
        path = get(paths, owner, nothing)
        path === nothing && return nothing
        for (name, shown) in zip(internal, authored)
            candidate = (Base.front(path)..., shown)
            get(paths, name, nothing) == candidate && continue
            suffix = 0
            while candidate in occupied
                suffix += 1
                candidate = (Base.front(path)..., Symbol(shown, :_, suffix))
            end
            paths[name] = candidate
            push!(occupied, candidate)
        end
        return nothing
    end
    for scan in plan.scans
        _is_noncentered_scan(scan) || continue
        state = first(scan.states)
        path = get(paths, state, nothing)
        path === nothing && continue
        generated!(state, [_scan_innovation_name(scan)],
            [Symbol(:_ppl_scan_z_, last(path))])
    end
    return paths
end

"""
    assign_layout(plan) -> LayoutTable

Assign packed coordinates: legacy coefficient blocks in `plan.predictors`
order, then ordinary sampled, plate, vector and array parameters in plan
order (pinned, deterministic). Affine readers add no coordinates.
Assumes `validate_plan` passed.
"""
function assign_layout(plan::StructuralPlan)
    isbound(plan) || throw(ContractValidationError(
        "[layout] assign_layout requires a bound plan (bind_data first)"))
    entries = LayoutEntry[]
    offset = 1
    for pred in plan.predictors
        pred = _legacy_predictor(pred)
        shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps,
            matrices = plan.matrices)
        shape.width == 0 && continue
        runs = _coefficient_runs(plan, pred, shape)
        if isempty(runs) ||
                (length(runs) == 1 && runs[1].transform === :identity)
            labels = Symbol[]
            for b in shape.blocks
                append!(labels, b.labels)
            end
            push!(entries,
                LayoutEntry(:coefficient, pred.name, block_name(pred.name),
                    labels, offset, shape.width, :identity))
            offset += shape.width
            continue
        end
        # Split block: one entry per maximal transform run (uniform
        # coefficients carry per-run interval bounds); the generator
        # reassembles the block name by offset order
        # (`_coef_reassembly_statements`).
        for (i, run) in enumerate(runs)
            seg = Symbol(string(block_name(pred.name)) * "__s" * string(i))
            push!(entries,
                LayoutEntry(:coefficient, pred.name, seg, run.labels,
                    offset, length(run.labels), run.transform, run.lo,
                    run.hi))
            offset += length(run.labels)
        end
    end
    # GLM-object coefficient vectors: one contiguous identity block per
    # response matrix (Normal priors on the real line). Width is static
    # from the intercept-free matrix; a beta shared across responses must
    # agree on width.
    glm_widths = Dict{Symbol,Int}()
    for r in plan.responses
        _is_glm_family(r.family) && _glm_coefficient_block(plan, r) || continue
        m = _find_matrix(plan, r.predictor)
        K = count(c -> c !== nothing, m.columns)
        if haskey(glm_widths, r.glm_beta)
            glm_widths[r.glm_beta] == K || throw(ContractValidationError(
                "[layout] GLM coefficient vector $(r.glm_beta) sizes $K " *
                "under $(r.label) but $(glm_widths[r.glm_beta]) elsewhere"))
            continue
        end
        glm_widths[r.glm_beta] = K
        push!(entries,
            LayoutEntry(:glm, nothing, r.glm_beta, [r.glm_beta], offset, K,
                :identity, NaN, NaN))
        offset += K
    end
    for p in plan.parameters
        p.name in plan.conditioned && continue
        if p.family === :external
            entry = _external_layout_entry(plan, p, offset, Int[])
            push!(entries, entry)
            offset += entry.size
            continue
        end
        transform, lo, hi = _entry_transform(p.family,
            _bound_override(plan, p.support_override),
            p.family === :uniform ? map(x -> _layout_bound(plan, x), p.args) : p.args)
        push!(entries,
            LayoutEntry(:sampled, nothing, p.name, [p.name], offset, 1, transform,
                lo, hi))
        offset += 1
    end
    for p in plan.plate_parameters
        p.name in plan.conditioned && continue
        if p.family === :external
            size = _plate_rows(plan, p)
            labels = [Symbol(p.name, ".", i) for i in 1:size]
            push!(entries, LayoutEntry(:external_plate, nothing, p.name, labels,
                offset, size, :external, NaN, NaN, Int[], LayoutEntry[], p.args))
            offset += size
            continue
        end
        transform, lo, hi =
            _entry_transform(p.family, _bound_override(plan, p.support_override),
                p.family === :uniform ? map(x -> _layout_bound(plan, x), p.args) : p.args)
        # An `eachindex(v)` plate has the rows of its authored range.
        size = _plate_rows(plan, p)
        push!(entries,
            LayoutEntry(:plate, nothing, p.name, [p.name], offset, size, transform,
                lo, hi))
        offset += size
    end
    for p in plan.vector_parameters
        p.name in plan.conditioned && continue
        p.size === nothing && throw(ContractValidationError(
            "[layout] vector parameter $(p.name) has unresolved size " *
            "(bind_data infers it from a linked response or explicit declaration)"))
        # A joint-outcomes LKJ Cholesky factor packs K(K−1)/2 thetas under
        # its own kind (K=1 packs zero, constraining to `[1.0]`).
        if p.family === :cholesky_corr_lkj
            K = p.size
            packed = K * (K - 1) ÷ 2
            labels = [Symbol(string(p.name) * "." * string(i)) for i in 1:packed]
            push!(entries,
                LayoutEntry(:cholesky_corr, nothing, p.name, labels, offset,
                    packed, :lkj))
            offset += packed
            continue
        end
        transform = _is_ordered_parameter(p.family) ? :ordered :
            p.family === :simplex_dirichlet ? :simplex :
            p.family === :positive_exponential ? :exp : :identity
        # `size` is the PACKED (unconstrained) length: a simplex packs
        # K−1 stick-breaking logits (a 1-simplex packs zero — Stan's
        # deterministic `[1.0]`); threshold and positive vectors pack
        # their size. Constrained length recovers as
        # `_vector_constrained_size`.
        packed = transform === :simplex ? p.size - 1 : p.size
        labels = [Symbol(string(p.name) * "." * string(i)) for i in 1:packed]
        push!(entries,
            LayoutEntry(:vector, nothing, p.name, labels, offset, packed,
                transform))
        offset += packed
    end
    # Sequential-recurrence latents: one identity array slice per scan. The
    # length T is the loop bound — a literal Int, or `n_obs` when the bound is a
    # data length name (the canonical observation-indexed state-space case).
    # Scan latents have real support (identity transform ⇒ no Jacobian). A
    # centered slice holds the carried state itself; a non-centered slice
    # (`_ppl_scan_z_<first state>`) holds the sampled seeds in setup order,
    # then each innovation local's `T - m` per-step values in body order,
    # while the state names bind the emitter's `scan(...)` reconstruction. A
    # non-centered scan with no sampled seed and no innovation (a fully
    # deterministic recurrence) has no slice.
    for raw in plan.scans
        s = _resolve_scan(plan, raw)
        T = _scan_length(plan, s)
        T >= s.lo - 1 || throw(ContractValidationError(
            "[layout] scan $(join(s.states, ", ")) length $(T) cannot hold " *
            "the $(s.lo - 1) seed values"))
        if _is_noncentered_scan(s)
            n = _scan_latent_size(s, T)
            n == 0 && continue
            push!(entries, _scan_layout_entry(plan, s, _scan_innovation_name(s), offset, T))
            offset += n
        else
            gap = _scan_shape_gap(s)
            gap === nothing || throw(ContractValidationError("[layout] " * gap))
            push!(entries, _scan_layout_entry(plan, s, only(s.states), offset, T))
            offset += T
        end
    end
    # Declared array parameters last (`arrays.jl`): LKJ factors and
    # simplexes reuse the `:cholesky_corr` / `:vector` edges, elementwise
    # arrays pack one `:array` block each.
    offset = _array_layout_entries!(entries, plan, offset)
    entries, reads, packed_total = _layout_reads(plan, entries)
    # A dotted cell latent's packed array draws under its authored name.
    paths = _scope_layout_paths(plan)
    for (site, (name, _)) in plan.cell_latents
        paths[site] = get(paths, name, (name,))
    end
    # So does a joint cell latent's.
    for (site, g) in plan.cell_geometries
        paths[site] = get(paths, g.name, (g.name,))
    end
    # Host-side reads (external arguments, dynamic bounds, cell geometries)
    # take a frozen whole value the graph keeps as bound, like a column
    # (`_holds_bound_value`), never re-evaluating it over absent inputs.
    return LayoutTable(entries, offset - 1, paths,
        merge!(Dict{Symbol,Any}(plan.columns), _definition_memo(plan),
            pairs(_frozen_graph_values(plan))),
        Dict{Symbol,Any}(a.name => a.expr for a in _definitions(plan)),
        plan.cell_latents, plan.cell_geometries, reads, _array_axis_labels(plan),
        packed_total)
end

"""Constrained length of a `:vector` entry: simplex packs K−1 logits
for K probabilities; threshold vectors pack their size."""
_vector_constrained_size(e::LayoutEntry) =
    e.transform === :simplex ? e.size + 1 : e.size

# Scalar logit, the host simplex inverse's break-fraction map.
_simplex_logit(z::Real) = log(z) - log1p(-z)

"""
    ordered_constrain(u) -> Vector

Host-side ordered transform (Stan `ordered`): `y[1] = u[1]`,
`y[i] = y[i-1] + exp(u[i])`. The in-graph transform uses the same
recurrence.
Like every host transform, it follows the input's number type and returns
`Float64` values for plain reals.
"""
function ordered_constrain(u::AbstractVector{<:Real})
    T = _host_eltype(u)
    y = Vector{T}(undef, length(u))
    for (i, x) in enumerate(u)
        y[i] = i == 1 ? convert(T, x) : y[i-1] + exp(convert(T, x))
    end
    return y
end

# The packed sampler input stays read-only. Native execution builds the
# constrained vector in one buffer of its own, then reuses it for the prefix
# sum. A non-allocating kernel passes its previous result as that buffer
# (`src/mutating.jl`).
@inline _ordered_constrain_native(u::AbstractVector{Float64}) =
    _ordered_constrain_native!(Vector{Float64}(undef, length(u)), u)

@inline function _ordered_constrain_native!(y::Vector{Float64},
                                            u::AbstractVector{Float64})
    isempty(u) && return y
    y[1] = u[1]
    @inbounds @simd for i in 2:length(u)
        y[i] = exp(u[i])
    end
    @inbounds for i in 2:length(u)
        y[i] = y[i-1] + y[i]
    end
    return y
end

@inline _ordered_constrain_kernel(u::AbstractVector{Float64}, lo::Int, hi::Int) =
    _ordered_constrain_native(view(u, lo:hi))

# Build the native simplex and its log-Jacobian in one pass. Only the result
# buffer is written; the packed coordinates remain read-only. The loop count
# is the runtime block length, including when that length came from bound data.
@inline _simplex_constrain_logjac_native(u::AbstractVector{Float64},
                                         lo::Int, hi::Int) =
    _simplex_constrain_logjac_native!(Vector{Float64}(undef, hi - lo + 2),
                                      u, lo, hi)

# The same pass into `p`, of length `hi - lo + 2`, which a non-allocating
# kernel passes as its previous result.
@inline function _simplex_constrain_logjac_native!(p::Vector{Float64},
        u::AbstractVector{Float64}, lo::Int, hi::Int)
    n = hi - lo + 1
    K = n + 1
    lr = 0.0
    jac = 0.0
    @inbounds for j in 1:n
        z = 1.0 / (1.0 + exp(-(u[lo + j - 1] + log(K - j))))
        l = log1p(-z)
        p[j] = exp(lr) * z
        jac += (lr + log(z)) + l
        lr += l
    end
    p[K] = exp(lr)
    return p, jac
end

@inline _simplex_constrain_logjac_kernel(u::AbstractVector{Float64}, lo::Int, hi::Int) =
    _simplex_constrain_logjac_native(u, lo, hi)

"""Inverse of [`ordered_constrain`](@ref)."""
function ordered_unconstrain(y::AbstractVector{<:Real})
    T = _host_eltype(y)
    u = Vector{T}(undef, length(y))
    prev = zero(T)
    for (i, v) in enumerate(y)
        u[i] = i == 1 ? convert(T, v) : log(convert(T, v) - prev)
        prev = convert(T, v)
    end
    return u
end

"""Log-Jacobian of [`ordered_constrain`](@ref): `Σ u[2:end]`."""
function ordered_logjac(u::AbstractVector{<:Real})
    T = _host_eltype(u)
    total = zero(T)
    for (i, x) in enumerate(u)
        i > 1 && (total += convert(T, x))
    end
    return total
end

"""
    simplex_constrain(u) -> Vector

Host-side stick-breaking simplex transform (thin-layer-owned
parameterization — NOT Stan's isometric-log-ratio `simplex_constrain`;
the middle layer owns layout+transforms, the LKJ-factor precedent):
break fractions `z[j] = σ(u[j] + log(K-j))`, stick remainders
`r[j] = Π_{i<j} (1 - z[i])` (in closed form, `exp` of the running sum of
`log1p(-z)`), `s[j] = r[j]*z[j]`, `s[K] = r[K]`. Empty input constrains
to `[1.0]` (the deterministic 1-simplex, as in Stan). The in-graph
transform fuses these vector operations into one result buffer and a scalar
log-Jacobian.
"""
function simplex_constrain(u::AbstractVector{<:Real})
    length(u) == 0 && return [one(_host_eltype(u))]
    z, _, lr = _simplex_breaks(u)
    return exp.(lr) .* vcat(z, ones(1))
end

# Break fractions `z` (K−1), `l = log1p(-z)` (K−1), and log stick remainders
# `lr` (K, `lr[1] = 0`) of the stick-breaking simplex.
function _simplex_breaks(u::AbstractVector{<:Real})
    K = length(u) + 1
    z = 1.0 ./ (1.0 .+ exp.(-(_host_real.(u) .+ log.(K .- (1:(K - 1))))))
    l = log1p.(-z)
    lr = cumsum(vcat(0.0, l))
    return z, l, lr
end

"""Inverse of [`simplex_constrain`](@ref) (stick-breaking, not Stan's ILR
`simplex_free`)."""
function simplex_unconstrain(s::AbstractVector{<:Real})
    K = length(s)
    T = _host_eltype(s)
    u = Vector{T}(undef, max(K - 1, 0))
    K <= 1 && return u
    remaining = one(T)
    for (j, v) in enumerate(s)
        j >= K && break
        z = convert(T, v) / remaining
        u[j] = _simplex_logit(z) - log(K - j)
        remaining -= convert(T, v)
    end
    return u
end

"""Log-Jacobian of [`simplex_constrain`](@ref):
`Σ [log(r) + log(z) + log1p(-z)]` over the K−1 breaks."""
function simplex_logjac(u::AbstractVector{<:Real})
    length(u) == 0 && return zero(_host_eltype(u))
    z, l, lr = _simplex_breaks(u)
    return sum(view(lr, 1:length(u)) .+ log.(z) .+ l)
end

"""
    _lkj_dim(packed) -> Int

Margin count K from a packed LKJ-theta length (`packed == K*(K-1)/2`;
0 ⟺ K=1). Loud on non-triangular lengths.
"""
function _lkj_dim(packed::Int)
    packed == 0 && return 1
    packed > 0 || throw(ContractValidationError(
        "[layout] LKJ packed length $packed is negative"))
    K = (1 + isqrt(1 + 8 * packed)) ÷ 2
    K * (K - 1) ÷ 2 == packed ||
        throw(ContractValidationError("[layout] LKJ packed length $packed " *
              "is not triangular (want K*(K-1)/2)"))
    return K
end

"""
    lkj_chol_constrain(u, K) -> Matrix

Host-side LKJ Cholesky-factor transform: Stan's partial-correlation
C-vine (user direction — follow Stan where possible; the
hyperspherical form survives as `lkj_chol_constrain_hyperspherical`).
Packed column-block order `for j in 2:K, i in 1:(j-1)` (Stan's
unconstrained order): `z[i,j] = tanh(u)`, upper factor
`w[i,j] = z[i,j]*d[i-1,j]` and `w[j,j] = d[j-1,j]`, where the running
product `d[i,j] = d[i-1,j]*√(1-z[i,j]²)` starts at `d[0,j] = 1`, and
`L = w'`. Like Stan's running sum of squares, each column costs one
square root per partial. The in-graph twins carry the identical
left-associated running product, so host and graph agree bit-for-bit.
K=1 constrains `[]` to `[1.0]`.
"""
function lkj_chol_constrain(u::AbstractVector{<:Real}, K::Int)
    K >= 1 || throw(ContractValidationError(
        "[layout] LKJ margin count K=$K < 1"))
    length(u) == K * (K - 1) ÷ 2 || throw(ContractValidationError(
        "[layout] LKJ packed length $(length(u)) ≠ $K*$(K-1)/2"))
    T = _host_eltype(u)
    z = zeros(T, K, K)
    p = 0
    for j in 2:K, i in 1:(j - 1)
        p += 1
        z[i, j] = tanh(convert(T, u[p]))
    end
    w = zeros(T, K, K)
    w[1, 1] = one(T)
    for j in 2:K
        d = one(T)
        for i in 1:(j - 1)
            w[i, j] = z[i, j] * d
            d *= sqrt(1 - z[i, j]^2)
        end
        w[j, j] = d
    end
    return Matrix(w')
end

"""Inverse of [`lkj_chol_constrain`](@ref): sequential vine inversion
(`w = L'`; `z[1,j] = w[1,j]`; `z[i,j] = w[i,j]/Π_{ip<i}√(1-z[ip,j]²)`),
then `u = atanh(z)`. Loud on non-square input, non-positive
diagonals, and out-of-`(-1,1)` partials (not a Cholesky factor)."""
function lkj_chol_unconstrain(L::AbstractMatrix{<:Real}, K::Int)
    size(L) == (K, K) || throw(ContractValidationError(
        "[layout] LKJ factor size $(size(L)) ≠ ($K, $K)"))
    T = _host_eltype(L)
    for i in 1:K
        convert(T, L[i, i]) > 0 || throw(ContractValidationError(
            "[layout] LKJ factor has a non-positive diagonal " *
            "(not a Cholesky factor)"))
    end
    z = zeros(T, K, K)
    for j in 2:K
        z[1, j] = convert(T, L[j, 1])
    end
    for i in 2:K, j in (i + 1):K
        d = one(T)
        for ip in 1:(i - 1)
            d *= sqrt(1 - z[ip, j]^2)
        end
        z[i, j] = convert(T, L[j, i]) / d
    end
    u = Vector{T}(undef, K * (K - 1) ÷ 2)
    p = 0
    for j in 2:K, i in 1:(j - 1)
        p += 1
        abs(z[i, j]) < 1 || throw(ContractValidationError(
            "[layout] LKJ partial z[$i,$j] leaves (-1,1) " *
            "(not a Cholesky factor)"))
        u[p] = atanh(z[i, j])
    end
    return u
end

"""Log-Jacobian of [`lkj_chol_constrain`](@ref): Stan's vine
`Σ_{i<j}((j-i+1)/2)·log(1-z[i,j]²)` with `z = tanh(u)` (the L→L'
Jacobian plus the tanh terms — NOT the manual's corr_matrix formula,
which carries the L→LL' step too). FD-verified. The in-graph twin
unrolls the identical sum over the shared z temps, so host and graph
agree bit-for-bit. K=2: `log(1-tanh(u)²)`."""
function lkj_chol_logjac(u::AbstractVector{<:Real}, K::Int)
    length(u) == K * (K - 1) ÷ 2 || throw(ContractValidationError(
        "[layout] LKJ packed length $(length(u)) ≠ $K*$(K-1)/2"))
    T = _host_eltype(u)
    total = zero(T)
    p = 0
    for j in 2:K, i in 1:(j - 1)
        p += 1
        z = tanh(convert(T, u[p]))
        total += ((j - i + 1) / 2) * log(1 - z^2)
    end
    return total
end

"""
    lkj_chol_constrain_hyperspherical(u, K) -> Matrix

RETAINED alternative to [`lkj_chol_constrain`](@ref) (the pre-vine
thin-layer-owned parameterization): row `i >= 2` is a unit vector
from `i-1` logistic angles `theta = pi*sigma(u)` packed ROW-major:
`L[i,j] = cos(theta_j) * prod(sin(theta[1:j-1]))`,
`L[i,i] = prod(sin(theta))`. Valid density, same constrained space —
kept for comparison, not wired into any layout.
"""
function lkj_chol_constrain_hyperspherical(u::AbstractVector{<:Real}, K::Int)
    K >= 1 || throw(ContractValidationError(
        "[layout] LKJ margin count K=$K < 1"))
    length(u) == K * (K - 1) ÷ 2 || throw(ContractValidationError(
        "[layout] LKJ packed length $(length(u)) ≠ $K*$(K-1)/2"))
    T = _host_eltype(u)
    L = zeros(T, K, K)
    L[1, 1] = one(T)
    p = 0
    for i in 2:K
        th = Vector{T}(undef, i - 1)
        for j in 1:i-1
            p += 1
            s = 1.0 / (1.0 + exp(-convert(T, u[p])))
            th[j] = pi * s
        end
        for j in 1:i-1
            v = cos(th[j])
            for m in 1:j-1
                v *= sin(th[m])
            end
            L[i, j] = v
        end
        d = one(T)
        for m in 1:i-1
            d *= sin(th[m])
        end
        L[i, i] = d
    end
    return L
end

"""Inverse of [`lkj_chol_constrain_hyperspherical`](@ref):
hyperspherical inversion per row
(`theta_j = atan(hypot(row[j+1:i]), row[j])`), then
`u = log(theta) - log(pi - theta)`."""
function lkj_chol_unconstrain_hyperspherical(L::AbstractMatrix{<:Real}, K::Int)
    size(L) == (K, K) || throw(ContractValidationError(
        "[layout] LKJ factor size $(size(L)) ≠ ($K, $K)"))
    T = _host_eltype(L)
    u = Vector{T}(undef, K * (K - 1) ÷ 2)
    p = 0
    for i in 2:K
        convert(T, L[i, i]) > 0 || throw(ContractValidationError(
            "[layout] LKJ factor row $i leaves the hemisphere " *
            "(non-positive diagonal — not a Cholesky factor)"))
        for j in 1:i-1
            p += 1
            rest = sqrt(sum(convert(T, L[i, m])^2 for m in j+1:i))
            th = atan(rest, convert(T, L[i, j]))
            u[p] = log(th) - log(pi - th)
        end
    end
    return u
end

"""Log-Jacobian of [`lkj_chol_constrain_hyperspherical`](@ref):
per-angle `(i-j)*log(sin theta)` + logistic
`log(pi) + log(s) + log1p(-s)`, summed row-major."""
function lkj_chol_logjac_hyperspherical(u::AbstractVector{<:Real}, K::Int)
    length(u) == K * (K - 1) ÷ 2 || throw(ContractValidationError(
        "[layout] LKJ packed length $(length(u)) ≠ $K*$(K-1)/2"))
    T = _host_eltype(u)
    total = zero(T)
    p = 0
    for i in 2:K, j in 1:i-1
        p += 1
        x = convert(T, u[p])
        s = 1.0 / (1.0 + exp(-x))
        th = pi * s
        total += (i - j) * log(sin(th)) + log(pi) + log(s) + log1p(-s)
    end
    return total
end

"""
    lkj_logconst(K, eta) -> Float64

LKJ normalizing constant: verbatim port of Stan's `do_lkj_constant`
(Lewandowski–Kurowicka–Joe 2009, theorem 5), INCLUDING the
`eta == 1.0` fast branch — the joint parity case compares against
Stan's `lkj_corr_cholesky_lpdf` with the same branch taken, so the
branch dispatch is load-bearing, not cosmetic. K=1 is ±0.0 (== 0.0
either way — the eta==1.0 branch yields -0.0, exactly as Stan does).
"""
function lkj_logconst(K::Int, eta::Real)
    e = Float64(eta)
    Km1 = K - 1
    if e == 1.0
        denom = 0.0
        for k in 1:Km1÷2
            denom += loggamma(2.0 * k)
        end
        constant = -denom
        if K % 2 == 1
            constant -= 0.25 * (K * K - 1) * log(pi) -
                0.25 * (Km1 * Km1) * log(2.0) -
                Km1 * loggamma(0.5 * (K + 1))
        else
            constant -= 0.25 * K * (K - 2) * log(pi) +
                0.25 * (3 * K * K - 4 * K) * log(2.0) +
                K * loggamma(0.5 * K) - Km1 * loggamma(Float64(K))
        end
        return constant
    end
    constant = Km1 * loggamma(e + 0.5 * Km1)
    for k in 1:Km1
        constant -= 0.5 * k * log(pi) + loggamma(e + 0.5 * (Km1 - k))
    end
    return constant
end

"""
    lkj_corr_cholesky_logpdf(L, eta) -> Float64

Host-side LKJ Cholesky log-density (Stan `lkj_corr_cholesky_lpdf`,
propto=false): diagonal sum over rows 2..K plus
[`lkj_logconst`](@ref). Stan's op order is preserved verbatim
(per-diagonal `(Km1-k-1)*ld + (2*eta-2)*ld`; the `eta == 1.0`
single-term branch) so joint parity against Stan holds to 1ulp.
K=1 is exactly 0.0.
"""
function lkj_corr_cholesky_logpdf(L::AbstractMatrix{<:Real}, eta::Real)
    K = size(L, 1)
    size(L, 2) == K || throw(ContractValidationError(
        "[layout] LKJ factor is not square: $(size(L))"))
    e = Float64(eta)
    lp = lkj_logconst(K, e)
    if e == 1.0
        for k in 0:K-2
            lp += (K - 1 - k - 1) * log(Float64(L[k+2, k+2]))
        end
        return lp
    end
    for k in 0:K-2
        ld = log(Float64(L[k+2, k+2]))
        lp += (K - 1 - k - 1) * ld + (2 * e - 2) * ld
    end
    return lp
end

_scope_segment(name::Symbol) = Base.isidentifier(string(name)) ?
    string(name) : "var" * repr(string(name))
_scope_path_string(path::Tuple) = join(_scope_segment.(path), ".")

function _draw_name(layout::LayoutTable, name::Symbol)
    path = get(layout.name_paths, name, (name,))
    return _scope_path_string(path)
end

function _authored_coordinate(layout::LayoutTable, name::Symbol, label::Symbol)
    raw = string(name)
    shown = _draw_name(layout, name)
    string(label) == raw && return Symbol(shown)
    prefix = raw * "."
    startswith(string(label), prefix) || return label
    return Symbol(shown, ".", string(label)[(lastindex(prefix) + 1):end])
end

"""
    coordinate_names(layout) -> Vector{Symbol}

Flat per-coordinate author paths, such as `b`, `z.b`, and `z.w.1`.
Scoped locals use dotted paths; literal identifier segments that contain
punctuation use Julia's `var"..."` spelling. Length equals `layout.total`.
"""
function coordinate_names(layout::LayoutTable)
    names = Symbol[]
    for e in layout.entries
        if e.kind === :coefficient
            for label in e.labels
                push!(names, Symbol(_draw_name(layout, e.predictor) * "." * string(label)))
            end
        elseif e.kind === :plate ||
               e.kind === :glm
            for i in 1:e.size
                push!(names, Symbol(_draw_name(layout, e.name) * "." * string(i)))
            end
        elseif e.kind === :array && haskey(layout.cell_latents, e.name)
            append!(names, _cell_latent_coordinates(layout, e.name))
        elseif e.kind === :array && haskey(layout.cell_geometries, e.name)
            append!(names, _cell_geometry_coordinates(layout, e.name))
        elseif e.kind === :vector || e.kind === :array || e.kind === :external || e.kind === :external_plate
            append!(names, [_authored_coordinate(layout, e.name, label)
                for label in e.labels])
        elseif e.kind === :cholesky_corr
            append!(names, [_authored_coordinate(layout, e.name, label)
                for label in e.labels])
        elseif e.kind === :scan
            for label in e.labels
                push!(names, Symbol(_draw_name(layout, e.name) * "." * string(label)))
            end
        else
            push!(names, Symbol(_draw_name(layout, e.name)))
        end
    end
    return names
end

"""
    coordinate_identities(layout) -> Vector{@NamedTuple{path::Tuple{Vararg{Symbol}}, index::Tuple}}

One identity per packed coordinate, in packed order, that names what the
coordinate means rather than where the binding put it, so the same coordinate
of two bindings of one program (a slice and the whole, one window and the next)
compares equal:

- `path` is the authored path, as in [`parameter_metadata`](@ref): `(:b,)`, or
  `(:z, :b)` inside a submodel.
- `index` has one entry per packed axis. An axis declared `levels(g)`,
  `unique(g)` or as a data vector holds that axis's bound label (the level of
  `g`, with the declaration's selection applied). So does the positional twin
  `1:length(levels(g))`, whose positions are the levels of `g` in `levels(g)`
  order, as in BRM's group effects (`z[1:length(levels(g)), 1:K]`, read at each
  row's level position). Any other axis holds its one-based position, including
  `1:length(levels(g)) - k`, which does not say which levels it leaves out. A
  scalar's index is `()`. A legacy coefficient holds its
  design label. The packed axis of a slice transform (a simplex or ordered row
  or column) is positional; its other axis keeps its labels. Coordinates of a
  joint cell latent or cell geometry are positional within their array.

So `z[levels(station)] .~ Normal.(0, s)` packs `(path = (:z,), index =
("st-05",))` wherever `st-05` falls in that binding's `levels(station)`, while
[`coordinate_names`](@ref) names its position (`z.2` in one binding, `z.4` in
another). Labels are data: comparing identities across bindings assumes the
same level values mean the same thing in both.
"""
function coordinate_identities(layout::LayoutTable)
    T = @NamedTuple{path::Tuple{Vararg{Symbol}}, index::Tuple}
    ids = Vector{T}(undef, layout.total)
    for e in layout.entries
        path = _smc_output_path(layout, e)
        indices = _entry_identity_indices(layout, e)
        length(indices) == e.size || throw(DimensionMismatch(
            "coordinate identities of $(e.name): $(length(indices)) for $(e.size) coordinates"))
        for (j, index) in enumerate(indices)
            ids[e.offset + j - 1] = T((path, index))
        end
    end
    ids
end

function _entry_identity_indices(layout::LayoutTable, e::LayoutEntry)
    e.kind === :coefficient && return Tuple[(label,) for label in e.labels]
    e.kind === :sampled && e.size == 1 && return Tuple[()]
    if e.kind === :array && !haskey(layout.cell_latents, e.name) &&
            !haskey(layout.cell_geometries, e.name)
        labels = get(layout.axis_labels, e.name, nothing)
        if _is_lkj_stack(e.transform)
            # `L.i.k`: partial i of level k, levels along the third axis.
            K, S = e.dims[1], e.dims[3]
            third = labels === nothing ? nothing : labels[3]
            return Tuple[(i, third === nothing ? k : third[k])
                for k in 1:S for i in 1:(K * (K - 1) ÷ 2)]
        end
        packed = _is_slice_transform(e.transform) ?
            _slice_packed_dims(e.transform, e.dims) : collect(e.dims)
        prod(packed; init = 1) == e.size || return Tuple[(j,) for j in 1:e.size]
        event = !_is_slice_transform(e.transform) ? 0 :
            _slice_transform_orientation(e.transform) isa _SliceRows ? 2 : 1
        axis(a, i) = labels === nothing || a == event || labels[a] === nothing ? i :
            labels[a][i]
        return Tuple[ntuple(a -> axis(a, I[a]), length(packed))
            for I in CartesianIndices(Tuple(packed))]
    end
    Tuple[(j,) for j in 1:e.size]
end

# Coefficient entries grouped by predictor in offset order (split
# blocks reassemble to one predictor vector in design order).
function _coefficient_groups(layout::LayoutTable)
    groups = Dict{Symbol,Vector{LayoutEntry}}()
    for e in layout.entries
        e.kind === :coefficient || continue
        p = e.predictor::Symbol
        push!(get!(groups, p, LayoutEntry[]), e)
    end
    for g in values(groups)
        sort!(g; by = e -> e.offset)
    end
    return groups
end

# One predictor's constrained coefficient vector from its entries:
# identity runs copy through, interval runs constrain per element.
function _constrain_coefficient(entries::Vector{LayoutEntry}, u)
    T = _host_eltype(u)
    if length(entries) == 1 && entries[1].transform === :identity
        e = entries[1]
        return Vector{T}(u[e.offset:(e.offset + e.size - 1)])
    end
    out = Vector{mapreduce(e -> _entry_eltype(e, T), promote_type, entries; init = T)}()
    for e in entries
        seg = u[e.offset:(e.offset + e.size - 1)]
        if e.transform === :identity
            append!(out, convert.(T, seg))
        else
            append!(out, [_constrain_elt(e, convert(T, x)) for x in seg])
        end
    end
    return out
end

# Preserve layout encounter order while forming nested author namespaces.
function _scope_namedtuple(items)
    names = Symbol[]
    groups = Dict{Symbol,Vector{Any}}()
    for (path, value) in items
        name = first(path)
        if !haskey(groups, name)
            push!(names, name)
            groups[name] = Any[]
        end
        push!(groups[name], (Base.tail(path), value))
    end
    values = map(names) do name
        group = groups[name]
        leaves = filter(item -> isempty(first(item)), group)
        if !isempty(leaves)
            length(group) == 1 || throw(ContractValidationError(
                "[layout] author name $name is both a value and a namespace"))
            return last(only(leaves))
        end
        return _scope_namedtuple(group)
    end
    return NamedTuple{Tuple(names)}(Tuple(values))
end

function _scoped_draw_values(layout::LayoutTable, pairs)
    isempty(layout.name_paths) &&
        return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
    return _scope_namedtuple(Any[(get(layout.name_paths, name, (name,)), value)
        for (name, value) in pairs])
end

struct _MissingScopedValue end
const _MISSING_SCOPED_VALUE = _MissingScopedValue()
function _scope_lookup(nt::NamedTuple, path::Tuple)
    value = nt
    for name in path
        value isa NamedTuple && haskey(value, name) ||
            return _MISSING_SCOPED_VALUE
        value = value[name]
    end
    return value
end

function _flat_draw_values(layout::LayoutTable, nt::NamedTuple)
    isempty(layout.name_paths) && return nt
    pairs = Pair{Symbol,Any}[]
    seen = Set{Symbol}()
    for entry in layout.entries
        name = entry.kind === :coefficient ? entry.predictor::Symbol : entry.name
        name in seen && continue
        push!(seen, name)
        value = _scope_lookup(nt, get(layout.name_paths, name, (name,)))
        value === _MISSING_SCOPED_VALUE && continue
        haskey(layout.cell_latents, name) &&
            (value = _cell_latent_flat(layout, name, value))
        push!(pairs, name => value)
    end
    return NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
end

# Bounds read the same constrained values and ordinary definitions as the
# graph. Packing order stays fixed; evaluation follows bound dependencies.
_dynamic_bounds(e::LayoutEntry) = e.sampling !== nothing || !(e.lo isa Real && e.hi isa Real) ||
    any(_dynamic_bounds, e.scan_blocks)
function _layout_lookup(layout::LayoutTable, values)
    active = Set{Symbol}()
    function lookup(name)
        haskey(values, name) && return values[name]
        haskey(layout.bound_values, name) && return layout.bound_values[name]
        haskey(layout.bound_exprs, name) || throw(ContractValidationError(
            "[layout] bound references unavailable value $name"))
        name in active && throw(ContractValidationError("[layout] cyclic bound at $name"))
        push!(active, name)
        result = _eval_value_expr(layout.bound_exprs[name], lookup, name)
        delete!(active, name)
        return result
    end
    return lookup
end
function _resolved_entry(layout::LayoutTable, e::LayoutEntry, values)
    (_dynamic_bounds(e) && e.sampling === nothing) || return e
    lookup = _layout_lookup(layout, values)
    resolve(x) = x isa Real ? x : _eval_value_expr(x, lookup, e.name)
    lo, hi = resolve(e.lo), resolve(e.hi)
    lo isa Real && hi isa Real ||
        throw(ContractValidationError("[layout] bounds for $(e.name) must be real scalars"))
    if e.transform === :interval
        isfinite(lo) && isfinite(hi) && lo < hi || throw(ContractValidationError(
            "[layout] bounds for $(e.name) require finite lo < hi, got ($lo, $hi)"))
    elseif e.transform === :floored
        isfinite(lo) || throw(ContractValidationError("[layout] lower bound for $(e.name) must be finite"))
    elseif e.transform === :upper
        isfinite(hi) || throw(ContractValidationError("[layout] upper bound for $(e.name) must be finite"))
    end
    return LayoutEntry(e.kind, e.predictor, e.name, e.labels, e.offset,
        e.size, e.transform, lo, hi, e.dims,
        [_resolved_entry(layout, b, values) for b in e.scan_blocks])
end
function _bound_entry_order(layout::LayoutTable)
    any(_dynamic_bounds, layout.entries) || return layout.entries
    byname = Dict(e.name => e for e in layout.entries)
    done, active = Set{Symbol}(), Set{Symbol}()
    out = LayoutEntry[]
    function visit(name)
        name in done && return
        name in active && throw(ContractValidationError("[layout] cyclic bound at $name"))
        push!(active, name)
        if haskey(byname, name)
            e = byname[name]
            reads = Any[e.lo, e.hi]
            e.sampling === nothing || push!(reads, e.sampling.rhs)
            for block in e.scan_blocks
                append!(reads, (block.lo, block.hi))
            end
            for x in reads, dep in _value_symbols(x)
                visit(dep)
            end
            push!(out, e)
        elseif haskey(layout.bound_exprs, name)
            for dep in _value_symbols(layout.bound_exprs[name])
                visit(dep)
            end
        end
        delete!(active, name)
        push!(done, name)
    end
    for e in layout.entries
        visit(e.name)
    end
    return out
end

"""
    constrain(layout, unconstrained) -> NamedTuple

Host-side constrain of the packed vector. Parameters keep their author
names; submodel locals are nested (`nt.z.b`, `nt.z.w.b`). Scalar, vector,
and array leaves retain their constrained shapes. Values follow the number
type of `unconstrained` (`Float64` for plain reals). Deterministic submodel
locals and return values are read in the model rather than included in this
draw container. The generator emits the equivalent transforms inside the
mathematical graph.

Concurrent calls may share a completed layout when its inputs and bound values
remain read-only. Built-in transforms synchronize their lazy endpoint caches;
caller-owned geometry and functions evaluated in bounds must themselves allow
concurrent calls. Each call constructs its own result containers.
"""
function constrain(layout::LayoutTable, u::AbstractVector{<:Real})
    length(u) == layout.total ||
        throw(ContractValidationError("[layout] unconstrained length $(length(u)) ≠ $(layout.total)"))
    pairs = Pair{Symbol,Any}[]
    coef_groups = _coefficient_groups(layout)
    seen_coef = Set{Symbol}()
    values = Dict{Symbol,Any}()
    T = _host_eltype(u)
    for entry in _bound_entry_order(layout)
        # A joint cell latent's arguments may read any other value.
        haskey(layout.cell_geometries, entry.name) && continue
        e = _resolved_entry(layout, entry, values)
        seg = u[e.offset:(e.offset + e.size - 1)]
        if e.kind === :external_plate
            args = _external_host_args(layout, e, values)
            v = _external_host_cells(e, :constrain, seg, args; _external_host_kwargs(layout, e, values)...)
            push!(pairs, e.name => v)
        elseif e.kind === :external
            args = _external_host_args(layout, e, values)
            v = _sampling_constrain(e.sampling.geometry, seg, Tuple(e.dims), args...; _external_host_kwargs(layout, e, values)...)
            (isempty(e.dims) ? v isa Number : v isa AbstractArray && size(v) == Tuple(e.dims)) ||
                throw(ContractValidationError("[layout] external constrain for $(e.name) returned the wrong shape"))
            push!(pairs, e.name => v)
        elseif e.kind === :coefficient
            p = e.predictor::Symbol
            p in seen_coef && continue
            push!(seen_coef, p)
            push!(pairs, p => _constrain_coefficient(coef_groups[p], u))
        elseif e.kind === :plate ||
               e.kind === :glm
            v = _entry_eltype(e, T)[_constrain_elt(e, convert(T, x)) for x in seg]
            push!(pairs, e.name => v)
        elseif e.kind === :scan
            push!(pairs, e.name => (isempty(e.scan_blocks) ? Vector{T}(seg) :
                vcat([_entry_eltype(b, T)[_constrain_elt(b, convert(T, u[i]))
                    for i in b.offset:(b.offset + b.size - 1)]
                    for b in e.scan_blocks]...)))
        elseif e.kind === :vector
            push!(pairs, e.name => _vector_constrain(e, seg))
        elseif e.kind === :array && _is_lkj_stack(e.transform)
            push!(pairs, e.name => _lkj_stack_constrain(e, seg))
        elseif e.kind === :array && _is_slice_transform(e.transform)
            push!(pairs, e.name => _array_slices_constrain(e, seg))
        elseif e.kind === :array && haskey(layout.cell_latents, e.name)
            v = _entry_eltype(e, T)[_constrain_elt(e, convert(T, x)) for x in seg]
            push!(pairs, e.name => _cell_latent_nested(layout, e.name, v))
        elseif e.kind === :array
            v = _entry_eltype(e, T)[_constrain_elt(e, convert(T, x)) for x in seg]
            push!(pairs, e.name =>
                (length(e.dims) == 1 ? v : reshape(v, e.dims...)))
        elseif e.kind === :cholesky_corr
            L = lkj_chol_constrain(seg, _lkj_dim(e.size))
            push!(pairs, e.name => (_is_upper_lkj(e.transform) ? permutedims(L) : L))
        else
            v = _constrain_elt(e, convert(T, only(seg)))
            push!(pairs, e.name => v)
        end
        lastpair = last(pairs)
        values[first(lastpair)] = last(lastpair)
    end
    for e in layout.entries
        haskey(layout.cell_geometries, e.name) || continue
        seg = u[e.offset:(e.offset + e.size - 1)]
        push!(pairs, e.name => _cell_geometry_constrain(layout, e.name, seg, values))
    end
    order = Dict(e.name => i for (i, e) in enumerate(layout.entries))
    sort!(pairs; by = p -> get(order, first(p), 0))
    return _scoped_draw_values(layout, pairs)
end

"""
    unconstrain(layout, constrained) -> Vector

Inverse of [`constrain`](@ref): named values → packed unconstrained vector.
Its element type promotes the values' number types with `Float64`.
"""
function unconstrain(layout::LayoutTable, nt::NamedTuple)
    nt = _flat_draw_values(layout, nt)
    T = _host_value_type(nt)
    u = Vector{T}(undef, layout.total)
    coef_groups = _coefficient_groups(layout)
    seen_coef = Set{Symbol}()
    for entry in layout.entries
        e = _resolved_entry(layout, entry, nt)
        if haskey(layout.cell_geometries, e.name)
            haskey(nt, e.name) || throw(ContractValidationError("[layout] missing parameter $(e.name)"))
            u[e.offset:(e.offset + e.size - 1)] .= _cell_geometry_unconstrain(layout,
                e.name, nt[e.name], nt)
        elseif e.kind === :external_plate
            haskey(nt, e.name) || throw(ContractValidationError("[layout] missing parameter $(e.name)"))
            v = nt[e.name]
            v isa AbstractVector && length(v) == e.size ||
                throw(ContractValidationError("[layout] external plate $(e.name) has the wrong shape"))
            args = _external_host_args(layout, e, nt)
            u[e.offset:(e.offset + e.size - 1)] .= _external_host_cells(e, :unconstrain, v, args; _external_host_kwargs(layout, e, nt)...)
        elseif e.kind === :external
            haskey(nt, e.name) || throw(ContractValidationError("[layout] missing parameter $(e.name)"))
            args = _external_host_args(layout, e, nt)
            v = e.sampling.geometry.unconstrain(nt[e.name], Tuple(e.dims), args...; _external_host_kwargs(layout, e, nt)...)
            v isa AbstractVector && length(v) == e.size ||
                throw(ContractValidationError("[layout] external inverse for $(e.name) returned the wrong packed dimension"))
            u[e.offset:(e.offset + e.size - 1)] .= v
        elseif e.kind === :coefficient
            p = e.predictor::Symbol
            p in seen_coef && continue
            push!(seen_coef, p)
            haskey(nt, p) || throw(
                ContractValidationError("[layout] missing predictor $p"),
            )
            v = nt[p]
            entries = coef_groups[p]
            total = sum(e2.size for e2 in entries)
            length(v) == total || throw(
                ContractValidationError("[layout] predictor $p length mismatch"),
            )
            pos = 1
            for e2 in entries
                seg = v[pos:(pos + e2.size - 1)]
                if e2.transform === :identity
                    u[e2.offset:(e2.offset + e2.size - 1)] .= convert.(T, seg)
                else
                    for (k, x) in enumerate(seg)
                        u[e2.offset + k - 1] =
                            _unconstrain_elt(e2, convert(T, x))
                    end
                end
                pos += e2.size
            end
        elseif e.kind === :plate ||
               e.kind === :glm
            what = e.kind === :plate ? "plate parameter" :
                "GLM coefficient vector"
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing $what $(e.name)"),
            )
            v = nt[e.name]
            length(v) == e.size || throw(
                ContractValidationError("[layout] $what $(e.name) length mismatch"),
            )
            for (k, x) in enumerate(v)
                u[e.offset + k - 1] = _unconstrain_elt(e, convert(T, x))
            end
        elseif e.kind === :scan
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing scan state $(e.name)"),
            )
            v = nt[e.name]
            length(v) == e.size || throw(
                ContractValidationError("[layout] scan state $(e.name) length mismatch"),
            )
            if isempty(e.scan_blocks)
                u[e.offset:(e.offset + e.size - 1)] .= convert.(T, v)
            else
                for b in e.scan_blocks, i in b.offset:(b.offset + b.size - 1)
                    u[i] = _unconstrain_elt(b, convert(T, v[i - e.offset + 1]))
                end
            end
        elseif e.kind === :array
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing array parameter $(e.name)"),
            )
            v = nt[e.name]
            size(v) == Tuple(e.dims) || throw(ContractValidationError(
                "[layout] array parameter $(e.name) has size $(size(v)), " *
                "want $(Tuple(e.dims))"))
            if _is_lkj_stack(e.transform)
                u[e.offset:(e.offset + e.size - 1)] .=
                    _lkj_stack_unconstrain(e, v)
                continue
            end
            if _is_slice_transform(e.transform)
                u[e.offset:(e.offset + e.size - 1)] .=
                    _array_slices_unconstrain(e, v)
                continue
            end
            for (k, x) in enumerate(vec(v))
                u[e.offset + k - 1] = _unconstrain_elt(e, convert(T, x))
            end
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
        elseif e.kind === :cholesky_corr
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing LKJ factor $(e.name)"),
            )
            v = nt[e.name]
            v isa AbstractMatrix || throw(
                ContractValidationError("[layout] LKJ factor $(e.name) " *
                      "must be a K×K matrix, got $(typeof(v))"),
            )
            u[e.offset:(e.offset + e.size - 1)] .=
                lkj_chol_unconstrain(_is_upper_lkj(e.transform) ?
                    permutedims(v) : v, _lkj_dim(e.size))
        else
            haskey(nt, e.name) || throw(
                ContractValidationError("[layout] missing parameter $(e.name)"),
            )
            u[e.offset] = _unconstrain_elt(e, convert(T, nt[e.name]))
        end
    end
    return u
end

# Packed element type of constrained values: their number types promoted with
# `Float64`, so plain reals pack as `Float64` and AD numbers pass through.
_host_value_type(x::Number) = _host_type(typeof(x))
_host_value_type(x::AbstractArray) =
    eltype(x) <: Number && isconcretetype(eltype(x)) ? _host_type(eltype(x)) :
    mapreduce(_host_value_type, promote_type, x; init = Float64)
_host_value_type(x::Union{Tuple,NamedTuple}) =
    mapreduce(_host_value_type, promote_type, values(x); init = Float64)
_host_value_type(x) = Float64

"""
    logjac(layout, unconstrained) -> Real

Host-side total log-Jacobian, using the same source-coordinate formulas as
the in-graph terms (positive: `u`; unit:
`-log1pexp(-u) - log1pexp(u)`).
"""
function logjac(layout::LayoutTable, u::AbstractVector{<:Real})
    length(u) == layout.total ||
        throw(ContractValidationError("[layout] unconstrained length $(length(u)) ≠ $(layout.total)"))
    T = _host_eltype(u)
    total = zero(T)
    values = any(_dynamic_bounds, layout.entries) || !isempty(layout.cell_geometries) ?
        _flat_draw_values(layout, constrain(layout, u)) : NamedTuple()
    for entry in layout.entries
        e = _resolved_entry(layout, entry, values)
        if haskey(layout.cell_geometries, e.name)
            total += _cell_geometry_logjac(layout, e.name,
                u[e.offset:(e.offset + e.size - 1)], values)
            continue
        end
        e.transform === :identity && continue
        seg = u[e.offset:(e.offset + e.size - 1)]
        if e.kind === :external_plate
            args = _external_host_args(layout, e, values)
            total += sum(_external_host_cells(e, :logjac, seg, args; _external_host_kwargs(layout, e, values)...))
            continue
        elseif e.kind === :external
            args = _external_host_args(layout, e, values)
            total += _sampling_logjac(e.sampling.geometry, seg, Tuple(e.dims), args...; _external_host_kwargs(layout, e, values)...)
            continue
        end
        if e.kind === :scan && !isempty(e.scan_blocks)
            for b in e.scan_blocks, i in b.offset:(b.offset + b.size - 1)
                total += _logjac_elt(b, convert(T, u[i]))
            end
            continue
        end
        if e.kind === :vector
            # Vector Jacobians couple coordinates (ordered sums, simplex
            # stick-breaking) — entry-level, never per-coordinate.
            total += _vector_logjac(e, seg)
            continue
        end
        if e.kind === :array && _is_lkj_stack(e.transform)
            # Level by level, the vine's coupled thetas.
            total += _lkj_stack_logjac(e, seg)
            continue
        end
        if e.kind === :array && _is_slice_transform(e.transform)
            # Per slice, the same coupling (simplex / ordered slices).
            total += _array_slices_logjac(e, seg)
            continue
        end
        if e.kind === :cholesky_corr
            # LKJ thetas couple through the hyperspherical rows —
            # entry-level, never per-coordinate.
            total += lkj_chol_logjac(seg, _lkj_dim(e.size))
            continue
        end
        for v in seg
            total += _logjac_elt(e, convert(T, v))
        end
    end
    return total
end

# Entry-level vector edges (host side). `_vector_transform_statements` and
# `jacobian_term` use the same vector operations in the graph; native simplex
# execution uses the equivalent fused loop.
_vector_constrain(e::LayoutEntry, seg) =
    e.transform === :identity ? Vector{_host_eltype(seg)}(seg) :
    e.transform === :ordered ? ordered_constrain(seg) :
    e.transform === :exp ? _host_eltype(seg)[_constrain_value(:exp, _host_real(x)) for x in seg] :
    simplex_constrain(seg)
_vector_unconstrain(e::LayoutEntry, v) =
    e.transform === :identity ? Vector{_host_eltype(v)}(v) :
    e.transform === :ordered ? ordered_unconstrain(v) :
    e.transform === :exp ? _host_eltype(v)[_unconstrain_value(:exp, _host_real(x)) for x in v] :
    simplex_unconstrain(v)
_vector_logjac(e::LayoutEntry, seg) =
    e.transform === :ordered ? ordered_logjac(seg) :
    e.transform === :exp ? sum(_logjac_value(:exp, _host_real(x)) for x in seg) :
    simplex_logjac(seg)

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
# outside it): it maps ℝ → (lo, hi) through RK's affine-logistic rule
# (`lo + (hi-lo)·σ(u)`). Its log-Jacobian reads the source coordinate:
# `log(hi-lo) - log1pexp(-u) - log1pexp(u)`. `:floored` is likewise
# parameterized (ℝ → (lo, ∞), `lo + exp(u)`, log-Jacobian `u`), as is
# `:upper` (ℝ → (-∞, hi), `hi - exp(u)`, log-Jacobian `u`). Prior density
# normalizers are emitted separately.
# The in-graph interval constrain edge calls that same rule. The other edges
# fold literal bounds (`_bound_add`), so host and graph agree bit-for-bit up
# to the sign of an exact zero.
function _constrain_elt(e::LayoutEntry, u)
    e.transform === :interval &&
        return rk_affine_logistic(e.lo, e.hi, u)
    e.transform === :floored && return e.lo + exp(u)
    e.transform === :upper && return e.hi - exp(u)
    return _constrain_value(e.transform, u)
end
function _unconstrain_elt(e::LayoutEntry, x)
    e.transform === :interval && return log(x - e.lo) - log(e.hi - x)
    e.transform === :floored && return log(x - e.lo)
    e.transform === :upper && return log(e.hi - x)
    return _unconstrain_value(e.transform, x)
end
function _logjac_elt(e::LayoutEntry, u)
    e.transform === :interval || e.transform === :floored ||
        e.transform === :upper || return _logjac_value(e.transform, u)
    e.transform === :floored && return u
    e.transform === :upper && return u
    # The width takes the coordinate's number type, so a wide input keeps
    # its precision; no rounded constrained value enters the logarithms.
    return log(oftype(u, e.hi - e.lo)) - log1pexp(-u) - log1pexp(u)
end

# The program's check of its packed point (snag `rk-prepared-cut-0e0b98ed`):
# `unconstrained` must hold the binding's packed total. A point of another
# length would otherwise be read with this binding's positions, so a longer one
# (a draw of a binding with more levels) silently gives values from the wrong
# coordinates, and a shorter one fails only at whichever read passes its end.
# The total is data the program reads like every position (`_layout_reads`), so
# one program still serves every extent. ReactiveKernels runs it at the entry of
# every cut that reads `unconstrained` (`ReactiveKernels.input_check!`, in
# `_register_packed_check!`); it reads only the point's length.
function _ppl_check_packed(unconstrained, total::Integer)
    length(unconstrained) == total || _packed_length_error(length(unconstrained), total)
    return nothing
end
@noinline _packed_length_error(n, total) = throw(ContractValidationError(
    "[layout] packed length $n ≠ layout total $total: a packed point holds the " *
    "coordinates of the binding whose layout packed it; match another binding's " *
    "coordinates by `coordinate_identities(layout)`"))

# Register the packed point's check on a program's graph.
_register_packed_check!(spec, layout::LayoutTable) =
    ReactiveKernels.input_check!(spec, :unconstrained, _ppl_check_packed,
        :unconstrained, layout.packed_total)

# A compact overlay computes the packed point from a base the caller passes
# (`_compact_cell_point(base, active, slots)`, as long as `base`), so the cut
# takes `base` instead of `unconstrained`: `base` gets the packed point's checks.
function _check_overlay_base!(graph::ReactiveKernels.Graph, packed::ReactiveKernels.Value,
        base::ReactiveKernels.Value)
    id = ReactiveKernels.canon_id(graph, packed.id)
    swap(a) = a isa ReactiveKernels.Value && ReactiveKernels.canon_id(graph, a.id) == id ? base : a
    for c in collect(graph.checks)
        ReactiveKernels.canon_id(graph, c.input) == id || continue
        push!(graph.checks, ReactiveKernels.InputCheck(base.id, c.f, map(swap, c.args)))
    end
    graph
end

"""
    coordinate_read(offset) -> Expr

Scalar packed-coordinate read (`unconstrained[o]`).
"""
coordinate_read(offset::Int) = :(unconstrained[$offset])

"""
    block_read(offset, width) -> Expr

Packed-slice read (`view(unconstrained, lo:hi)`, `@ppl` vector shape).
"""
block_read(offset::Int, width::Int) =
    :(view(unconstrained, $offset:$(offset + width - 1)))

# The packed range an entry's program reads.
_entry_range(e::LayoutEntry) = e.positions === nothing ?
    :($(e.offset):$(e.offset + e.size - 1)) :
    :($(e.positions[1]):$(e.positions[2]))
_entry_view(e::LayoutEntry) = :(view(unconstrained, $(_entry_range(e))))

# Sizes and data-derived numbers stay data in the generated program, never this
# binding's literals (user principle
# `runtime-values requirement`). Layout reads
# them from the values the program already has: a block sized `1:n` is
# `view(unconstrained, 1:n)`, and the next block starts after it. One program
# then serves every extent and value of one declaration; a raw cut supplies the
# same data it always did. A literal extent stays a literal, so an entry whose
# position and size depend on no data reads exactly as before.
#
# A read is an `Int`, a name, or an expression over data. Literal arithmetic
# folds, so a read stays as plain as the authored extent.
_read_add(a::Int, b::Int) = a + b
_read_add(a, b::Int) = b == 0 ? a : b < 0 ? :($a - $(-b)) : :($a + $b)
_read_add(a::Int, b) = a == 0 ? b : :($a + $b)
_read_add(a, b) = :($a + $b)
_read_mul(a::Int, b::Int) = a * b
_read_mul(a::Int, b) = a == 0 ? 0 : a == 1 ? b : :($a * $b)
_read_mul(a, b::Int) = _read_mul(b, a)
_read_mul(a, b) = :($a * $b)
_read_length(x) = Expr(:call, GlobalRef(Base, :length), x)
_read_size(x, d) = Expr(:call, GlobalRef(Base, :size), x, d)

# A range endpoint or `reshape` dimension is a literal or a name, the forms RK
# reads per row; any other read gets a statement of its own.
function _named_read!(reads::Vector{Expr}, name::Symbol, read)
    (read isa Int || read isa Symbol) && return read
    push!(reads, :($name::Int = $read))
    return name
end

# Which reads a binding's program may use. A read must name data that every
# binding of this model has: an observation can be absent, so nothing reads one
# (a design binding of a fit's build generates the same program). And it must
# evaluate to the size this binding packs: a read that disagrees would silently
# read other coordinates, so that block keeps its literal size instead.
function _read_tests(plan::StructuralPlan)
    exprs = Dict{Symbol,Any}(a.name => a.expr for a in plan.assignments)
    merge!(exprs, Dict{Symbol,Any}(d.name => d.expr for d in plan.derived))
    observed = union(Set{Symbol}(plan.conditioned),
        Set{Symbol}(r.response for r in plan.responses if r.response isa Symbol))
    known = Dict{Symbol,Bool}()
    function isdata(name)
        name in observed && return false
        _holds_bound_value(plan, plan.columns, name) &&
            return _supplied(get(plan.columns, name, nothing))
        haskey(exprs, name) || return false
        return get!(known, name) do
            known[name] = false # a cyclic definition is not data
            all(isdata, _value_symbols(exprs[name]))
        end
    end
    # A read of no name at all, `length([1.0, 1.0])`, is a literal.
    reads_data(ex) = (names = _value_symbols(ex); !isempty(names) && all(isdata, names))
    function lookup(name)
        _holds_bound_value(plan, plan.columns, name) && return _bound_value(plan, plan.columns, name)
        return _memoized_data(plan, name) do
            _eval_value_expr(exprs[name], lookup, name)
        end
    end
    # Sizes evaluate at binding anyway; one that cannot evaluate here keeps the
    # literal the binding already computed.
    agrees(read, value::Int) = try
        x = _eval_value_expr(read, lookup, :layout_read)
        x isa Integer && !(x isa Bool) && x == value
    catch
        false
    end
    return (; reads_data, agrees)
end

# `read` when it reads data and agrees with this binding's `value`, else `value`.
_checked_read(tests, read, value::Int) =
    tests.reads_data(read) && tests.agrees(read, value) ? read : value

# A property read in an extent, `support.geometry.n`, as the model reads any
# value's property: `getproperty(getproperty(support, :geometry), :n)`. An
# authored extent and a geometry hook's AST keep Julia's dotted syntax, which
# `_value_symbols` takes for a module path, so the read would name no data. A
# callee keeps its path (`Base.length(x)`).
_property_read(ex) = ex
function _property_read(ex::Expr)
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode &&
            ex.args[2].value isa Symbol
        return Expr(:call, GlobalRef(Base, :getproperty), _property_read(ex.args[1]), ex.args[2])
    end
    callee = ex.head === :call || (ex.head === :. && length(ex.args) == 2)
    return Expr(ex.head, Any[callee && i == 1 ? a : _property_read(a)
        for (i, a) in enumerate(ex.args)]...)
end

# An authored dimension as a runtime length expression.
function _array_axis_read(d)
    d isa Int && return d
    if _is_levels_dim(d)
        return _read_length(_axis_labels_expr(d))
    elseif (cnt = _levels_count(d)) !== nothing
        return _read_add(_read_length(_axis_labels_expr(:levels, cnt[1], Colon())), -cnt[2])
    elseif _is_bound_extent_dim(d)
        return _property_read(d)
    else
        return _read_size(d.args[2], d.args[3])
    end
end

# An array axis as the program reads it; `value` is this binding's length.
function _array_dim_read(plan::StructuralPlan, tests, d, value::Int)
    d isa Int && return d
    d isa Symbol && haskey(plan.columns, d) && !(plan.columns[d] isa Integer) && return value
    return _checked_read(tests, _array_axis_read(d), value)
end

function _array_packed_reads(plan::StructuralPlan, tests, e::LayoutEntry)
    p = _array_param(plan, e.name)
    dims = Any[_array_dim_read(plan, tests, d, n) for (d, n) in zip(p.dims, e.dims)]
    e.transform === :simplex_rows && return Any[dims[1], _read_add(dims[2], -1)]
    e.transform === :simplex_cols && return Any[_read_add(dims[1], -1), dims[2]]
    return dims
end

# A plate latent's rows as the program reads them.
function _plate_rows_read(plan::StructuralPlan, tests, p::PlateParameter, value::Int)
    r = p.range
    r isa UnitRange && return value
    iterator = r isa Symbol ? Expr(:call, :eachindex, r) : r
    if iterator isa Expr
        Meta.isexpr(iterator, :call) && !isempty(iterator.args) &&
            iterator.args[1] in (:eachindex, :axes) || return value
        return _checked_read(tests, _read_length(iterator), value)
    end
    # No authored range: the rows of the data the latent reads (`_value_rows`).
    columns = sort!(collect(_response_reads(plan, p.name, _row_columns(plan))))
    isempty(columns) && return value
    return _checked_read(tests, _read_size(first(columns), 1), value)
end

# An external law's axes and packed size as the program reads them: its
# declared axes, which are its geometry's shape, and its geometry's count of
# packed coordinates, each when it reads data. The constrain and Jacobian
# endpoints then receive the shape the data give, not this binding's numbers.
function _external_reads(plan::StructuralPlan, tests, e::LayoutEntry)
    i = findfirst(p -> p.name === e.name, plan.array_parameters)
    dims = i === nothing ? Any[] : Any[_array_dim_read(plan, tests, d, n)
        for (d, n) in zip(plan.array_parameters[i].dims, e.dims)]
    packed = e.sampling.geometry.unconstrained
    size = packed isa Int ? e.size : _checked_read(tests, _property_read(packed), e.size)
    return dims, size
end

# A threshold or simplex vector's length as the program reads it: its authored
# extent, when that reads data, such as `length(alpha)` for a simplex whose
# concentration `alpha` is bound. A count inferred from a response stays
# literal: the response can be absent, and it can declare unobserved levels.
function _vector_length_read(tests, p::VectorParameter)
    p.extent_expr === nothing && return p.size
    return _checked_read(tests, p.extent_expr, p.size)
end

# The same vector's packed length: a simplex packs one coordinate fewer.
function _vector_size_read(plan::StructuralPlan, tests, e::LayoutEntry)
    i = findfirst(p -> p.name === e.name, plan.vector_parameters)
    i === nothing && return e.size
    read = _vector_length_read(tests, plan.vector_parameters[i])
    return e.transform === :simplex ? _read_add(read, -1) : read
end

# A declared LKJ factor's dimension as the program reads it: its authored `K`,
# when that reads data. The factor's statements, its prior and the blocks packed
# after it then read the dimension and its `K(K - 1)/2` partials from the data,
# so one program serves every dimension (snag `rkppl-lkjcholesk-e8d4d0d6`).
_lkj_dim_read(plan::StructuralPlan, tests, p::ArrayParameter, K::Int) =
    _array_dim_read(plan, tests, p.dims[1], K)

# The names of a data-read LKJ dimension and of its partials count, `[K, P]`.
_lkj_axis_name(L::Symbol) = Symbol(:_ppl_axis1_, L)
_lkj_count_name(L::Symbol) = Symbol(:_ppl_lkj_count_, L)
function _lkj_axes_reads!(reads::Vector{Expr}, L::Symbol, K)
    K = _named_read!(reads, _lkj_axis_name(L), K)
    P = _named_read!(reads, _lkj_count_name(L),
        Expr(:call, GlobalRef(Base, :div), :($K * ($K - 1)), 2))
    return Any[K, P]
end

# Whether an entry's program reads `positions`. Scalars, per-coordinate positive
# vectors and LKJ scalar edges and stacks read one literal coordinate per
# statement.
_reads_positions(e::LayoutEntry) = e.kind === :cholesky_corr ?
    !isempty(e.dims) && !_is_lkj_stack(e.transform) :
    e.kind === :vector ? e.transform !== :exp :
    e.kind === :array ? !_is_lkj_stack(e.transform) :
    e.kind in (:plate, :glm, :coefficient, :scan, :external, :external_plate)

_scan_read_length(plan::StructuralPlan, s::ScanSpec, T::Int) =
    s.hi isa Symbol && get(plan.columns, s.hi, nothing) isa Integer ? s.hi : T

# A scan's innovation ranges in its latent slice, as the program reads them.
function _scan_read_ranges(plan::StructuralPlan, s::ScanSpec, ranges, T::Int)
    read = _scan_read_length(plan, s, T)
    read isa Int && return ranges
    seeds = count(f -> f.kind === :sample, s.setup)
    n = _read_add(read, -(s.lo - 1))
    return Any[(_read_add(_read_add(seeds, _read_mul(j - 1, n)), 1),
        _read_add(seeds, _read_mul(j, n))) for j in eachindex(ranges)]
end

# A data-derived bound is read as authored; a literal one stays a literal. A
# bound that evaluated to a finite number read data alone (`_layout_bound`).
_data_bound_reads(authored, lo, hi) =
    (!(authored[1] isa Real) && lo isa Real && isfinite(lo) ? authored[1] : nothing,
     !(authored[2] isa Real) && hi isa Real && isfinite(hi) ? authored[2] : nothing)

function _authored_bounds(p)
    ov = p.support_override
    if ov isa Tuple && length(ov) == 3 && support_of(p.family, ov) === :truncated
        _, lo, hi = _entry_transform(p.family, ov, p.args)
        return lo, hi
    elseif ov isa Tuple && length(ov) == 2
        return ov[1] === :upper ? (NaN, ov[2]) : (ov[2], NaN)
    elseif ov === nothing && p.family === :uniform
        return p.args.arg1, p.args.arg2
    end
    return NaN, NaN
end

# The program's reads for every entry, in packing order, with the statements
# that name them.
function _layout_reads(plan::StructuralPlan, entries::Vector{LayoutEntry})
    reads = Expr[]
    tests = _read_tests(plan)
    parameters = Dict{Symbol,Any}()
    for ps in (plan.parameters, plan.plate_parameters, plan.array_parameters), p in ps
        parameters[p.name] = p
    end
    scans = Dict{Symbol,ScanSpec}()
    for raw in plan.scans
        s = _resolve_scan(plan, raw)
        scans[_is_noncentered_scan(s) ? _scan_innovation_name(s) : only(s.states)] = s
    end
    at = 1
    out = LayoutEntry[]
    for e in entries
        bound_reads = e.transform in (:interval, :floored, :upper) &&
            haskey(parameters, e.name) && parameters[e.name].family !== :external ?
            _data_bound_reads(_authored_bounds(parameters[e.name]), e.lo, e.hi) : e.bound_reads
        bound_reads = map(x -> x !== nothing && tests.reads_data(x) ? x : nothing, bound_reads)
        entry, at = _entry_reads!(reads, plan, tests, e, at, get(scans, e.name, nothing), bound_reads)
        push!(out, entry)
    end
    # The packed total, the position before `at`: the last block's own last
    # position when it ends the layout, a literal when no position reads data.
    last = Meta.isexpr(at, :call, 3) && at.args[1] === :+ && at.args[3] === 1 ?
        at.args[2] : _read_add(at, -1)
    return out, reads, _named_read!(reads, :_ppl_packed_total, last)
end

# The first read at `at`, and the read after a block of `size` coordinates.
function _block_reads!(reads, name::Symbol, at, size)
    start = _named_read!(reads, Symbol(:_ppl_first_, name), at)
    # Literal arithmetic folds: a block from position 1 ends at its own size.
    stop = start isa Int ? _read_add(start - 1, size) : _read_add(start, _read_add(size, -1))
    stop = _named_read!(reads, Symbol(:_ppl_last_, name), stop)
    return start, stop, _read_add(stop, 1)
end

function _entry_reads!(reads, plan, tests, e::LayoutEntry, at, scan, bound_reads)
    with(positions, axes = nothing, blocks = e.scan_blocks) =
        LayoutEntry(e.kind, e.predictor, e.name, e.labels, e.offset, e.size, e.transform,
            e.lo, e.hi, e.dims, blocks, e.sampling, positions, axes, bound_reads)
    if !_reads_positions(e)
        return with(nothing), _read_add(at, e.size)
    end
    axes = nothing
    if !isempty(e.scan_blocks)
        # A seed block holds `b.size` seeds of one coordinate each
        # (`_scan_layout_entry`); a step block holds one statement, one
        # coordinate per iteration.
        T = _scan_read_length(plan, scan, _scan_length(plan, scan))
        steps = _read_add(T, -(scan.lo - 1))
        sampled = [st for st in (scan.setup..., scan.step...) if st.kind === :sample &&
            (st isa ScanSetup || _scan_length(plan, scan) > scan.lo - 1)]
        blocks = LayoutEntry[]
        k = 1
        for b in e.scan_blocks
            st = sampled[k]
            k += st isa ScanSetup ? b.size : 1
            start, stop, next = at isa Int && T isa Int ? (nothing, nothing, _read_add(at, b.size)) :
                _block_reads!(reads, b.name, at, st isa ScanSetup ? b.size : steps)
            push!(blocks, LayoutEntry(b.kind, b.predictor, b.name, b.labels, b.offset, b.size,
                b.transform, b.lo, b.hi, b.dims, b.scan_blocks, b.sampling,
                start === nothing ? nothing : (start, stop), nothing,
                map(x -> x !== nothing && tests.reads_data(x) ? x : nothing, b.bound_reads)))
            at = next
        end
        return with(nothing, nothing, blocks), at
    elseif e.kind === :array
        packed = _array_packed_reads(plan, tests, e)
        if !all(a -> a isa Int, packed)
            axes = Any[_named_read!(reads, Symbol(:_ppl_axis, d, :_, e.name), a)
                for (d, a) in enumerate(packed)]
        end
        size = foldl(_read_mul, something(axes, packed); init = 1)
    elseif e.kind === :external
        dims, size = _external_reads(plan, tests, e)
        if !all(a -> a isa Int, dims)
            axes = Any[_named_read!(reads, Symbol(:_ppl_axis, d, :_, e.name), a)
                for (d, a) in enumerate(dims)]
        end
    elseif e.kind === :plate || e.kind === :external_plate
        i = findfirst(p -> p.name === e.name, plan.plate_parameters)
        size = i === nothing ? e.size : _plate_rows_read(plan, tests, plan.plate_parameters[i], e.size)
    elseif e.kind === :vector
        size = _vector_size_read(plan, tests, e)
    elseif e.kind === :cholesky_corr
        K = _lkj_dim_read(plan, tests, _array_param(plan, e.name), e.dims[1])
        axes = K isa Int ? nothing : _lkj_axes_reads!(reads, e.name, K)
        size = axes === nothing ? e.size : axes[2]
    elseif e.kind === :scan
        T = _scan_read_length(plan, scan, _scan_length(plan, scan))
        size = T isa Int ? e.size : !_is_noncentered_scan(scan) ? T :
            _read_add(count(f -> f.kind === :sample, scan.setup),
                _read_mul(length(_scan_innovations(scan)), _read_add(T, -(scan.lo - 1))))
    else
        size = e.size
    end
    # Literal position and size: the literal range, as ever.
    at isa Int && size isa Int && return with(nothing), at + size
    start, stop, next = _block_reads!(reads, e.name, at, size)
    # The Jacobian of an ordered vector reads its tail on its own.
    if e.kind === :vector && e.transform === :ordered
        _named_read!(reads, _vector_ordered_tail_first(e), _read_add(start, 1))
    end
    return with((start, stop), axes), next
end

# The entry as its generated program reads it: a data-derived bound is its
# authored expression, never this binding's number.
function _program_entry(e::LayoutEntry)
    e.bound_reads === (nothing, nothing) && return e
    return LayoutEntry(e.kind, e.predictor, e.name, e.labels, e.offset, e.size,
        e.transform, something(e.bound_reads[1], e.lo),
        something(e.bound_reads[2], e.hi), e.dims, e.scan_blocks, e.sampling,
        e.positions, e.axes)
end

# Bound arithmetic of the parameterized (`:interval`/`:floored`/`:upper`)
# inverse, Jacobian and one-sided edges. Literal operands fold
# (`(0.5 - 0.0)` is `0.5`, `x - -1.0` is `x + 1.0`) and a literal zero
# drops out (`0.0 + exp(u)` is `exp(u)`, `log(x - 0.0)` is `log(x)`,
# `0.0 - exp(u)` is `-exp(u)`, `- log(1.0)` vanishes). No fold changes an
# IEEE result (a literal difference is computed once at emission; `x - -1.0`
# and `x + 1.0` round identically), so values and derivatives are the host
# `_*_elt` path's. Only an exact zero's sign can differ (`0.0 - 0.0` is
# `+0.0`, `-0.0` is not): a zero ceiling's value shows it when `exp(u)`
# underflows, and a logarithm absorbs it everywhere else. Sampled and data
# bounds stay as authored. `dot` spells the elementwise form.
_literal_bound(b) = b isa Real
_zero_bound(b) = _literal_bound(b) && iszero(b)
function _bound_add(a, b; dot::Bool = false)
    _literal_bound(a) && _literal_bound(b) && return a + b
    _zero_bound(a) && return b
    _zero_bound(b) && return a
    return dot ? :($a .+ $b) : :($a + $b)
end
function _bound_sub(a, b; dot::Bool = false)
    _literal_bound(a) && _literal_bound(b) && return a - b
    _zero_bound(b) && return a
    _literal_bound(b) && b < 0 && return _bound_add(a, -b; dot)
    _zero_bound(a) && return dot ? :(.-$b) : :(-$b)
    return dot ? :($a .- $b) : :($a - $b)
end

# Interval edges ℝ → (lo, hi): the RK-owned affine-logistic constrain rule,
# whose primal is `lo + (hi - lo) / (1 + exp(-u))`, its
# inverse `log(x - lo) - log(hi - x)`, and the log-Jacobian
# `log(hi - lo) - log1pexp(-u) - log1pexp(u)` (summed over the block when
# `dot`).
function _interval_constrain(lo, hi, u; dot::Bool = false)
    return dot ? :(rk_affine_logistic.($lo, $hi, $u)) :
        :(rk_affine_logistic($lo, $hi, $u))
end
_interval_unconstrain(lo, hi, x; dot::Bool = false) = dot ?
    :(log.($(_bound_sub(x, lo; dot))) .- log.($(_bound_sub(hi, x; dot)))) :
    :(log($(_bound_sub(x, lo))) - log($(_bound_sub(hi, x))))
function _interval_logjac(lo, hi, u; dot::Bool = false)
    w = _bound_sub(hi, lo)
    logw = _literal_bound(w) && isone(w) ? 0.0 : :(log($w))
    dot || return :($logw - log1pexp(-$u) - log1pexp($u))
    return :(sum($logw .- log1pexp.(-$u) .- log1pexp.($u)))
end

"""
    transform_statements(entry) -> Vector{Expr}

In-graph constrain edge(s) for one layout entry. Coefficient/scan slices read a
`view`; a `:identity` sampled entry reads its scalar coordinate; a constrained
sampled entry (`:exp`/`:logistic`) splices the bijector's `constrain` endpoint,
which the planner inlines.
"""
function transform_statements(e::LayoutEntry)
    e = _program_entry(e)
    e.kind === :external_plate && return Expr[
        :($(e.name) = $(_external_cells_expr(e, :constrain))),
        :($(_plate_logjac_name(e.name)) = $(_external_cells_expr(e, :logjac)))]
    e.kind === :external && return Expr[
        :($(_external_geometry_name(e)) = $(_external_geometry_expr(e))),
        :($(e.name) = $(_external_edge(e, :_sampling_constrain)))]
    if e.kind === :scan && !isempty(e.scan_blocks)
        stmts = Expr[]
        for b in e.scan_blocks
            if b.transform === :identity
                push!(stmts, :($(b.name)::Vector{Float64} = copy($(_entry_view(b)))))
            else
                append!(stmts, _plate_transform_statements(_program_entry(b)))
            end
        end
        push!(stmts, :($(e.name)::AbstractVector{Float64} = vcat($([b.name for b in e.scan_blocks]...))))
        return stmts
    end
    if e.kind === :scan
        # scan-state slices read a view into `e.name` (a scan-state slice,
        # a non-centered scan's `_ppl_scan_z_<state>` innovation slice, or
        # a dar trajectory's `_ppl_dar_z_<state>` slice — the emitter
        # reconstructs from the innovation slices)
        return Expr[:($(e.name)::AbstractVector{Float64} = $(_entry_view(e)))]
    end
    if e.kind === :coefficient
        # Coefficient runs: identity runs read a view; interval runs
        # (uniform coefficients) hand-roll the broadcast constrain edge
        # with the host `_*_elt` path's math (mirroring the plate
        # `:interval` arm below).
        view_read = _entry_view(e)
        e.transform === :identity &&
            return Expr[:($(e.name)::AbstractVector{Float64} = $view_read)]
        e.transform === :interval || throw(ContractValidationError(
            "[layout] coefficient block $(e.name) has no transform rule " *
            "for $(e.transform) (admitted: :identity, :interval)"))
        u = Symbol(:_ppl_int_, e.name)
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} =
                $(_interval_constrain(e.lo, e.hi, u; dot = true))),
            :($u::AbstractVector{Float64} =
                $(_interval_unconstrain(e.lo, e.hi, e.name; dot = true))),
        ]
    end
    if e.kind === :plate ||
       e.kind === :glm
        # Plate and GLM vectors share this transform.
        return _plate_transform_statements(e)
    end
    if e.kind === :vector
        return _vector_transform_statements(e)
    end
    e.kind === :array && return _array_transform_statements(e)
    if e.kind === :cholesky_corr
        isempty(e.dims) || return _lkj_array_transform_statements(e)
        # Scalar-edge factors read named edges; declared arrays above retain
        # their triangular iteration for every K.
        return _lkj_corr_transform_statements(e)
    end
    o = e.offset
    coord = coordinate_read(o)
    e.transform === :identity && return Expr[:($(e.name)::Float64 = $coord)]
    if e.transform === :interval
        # An :interval transform is parameterized by per-entry bounds, so it is
        # not in the Symbol-keyed (parameterless) bijector registry — its
        # forward edge calls the host `_*_elt` path's RK-owned rule; the
        # inverse edge retains the authored log-ratio expression.
        u = Symbol(:_ppl_int_, e.name)
        return Expr[
            :($u::Float64 = $coord),
            :($(e.name)::Float64 = $(_interval_constrain(e.lo, e.hi, u))),
            :($u::Float64 = $(_interval_unconstrain(e.lo, e.hi, e.name))),
        ]
    end
    if e.transform === :floored
        # Parameterized like :interval (the per-entry floor is not in the
        # registry): forward + inverse edges hand-rolled with the host
        # `_*_elt` path's math (`lo + exp(u)` / `log(x - lo)`).
        u = Symbol(:_ppl_fl_, e.name)
        return Expr[
            :($u::Float64 = $coord),
            :($(e.name)::Float64 = $(_bound_add(e.lo, :(exp($u))))),
            :($u::Float64 = log($(_bound_sub(e.name, e.lo)))),
        ]
    end
    if e.transform === :upper
        # Parameterized like :interval (the per-entry ceiling is not in the
        # registry): forward + inverse edges hand-rolled with the host
        # `_*_elt` path's math (`hi - exp(u)` / `log(hi - x)`).
        u = Symbol(:_ppl_up_, e.name)
        return Expr[
            :($u::Float64 = $coord),
            :($(e.name)::Float64 = $(_bound_sub(e.hi, :(exp($u))))),
            :($u::Float64 = log($(_bound_sub(e.hi, e.name)))),
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
# A real-support block (per-cell latents, elementwise arrays, GLM vectors) is
# its packed coordinates, read in place with no transform or Jacobian.
function _plate_transform_statements(e::LayoutEntry)
    range = _entry_range(e)
    view_read = :(view(unconstrained, $range))
    e.transform === :identity && return Expr[:($(e.name)::AbstractVector{Float64} =
        Base.view(unconstrained, $range))]
    if e.transform === :interval
        # Parameterized bounds ⇒ not in the (parameterless) bijector registry;
        # hand-rolled broadcast edges over the block view, with the host
        # `_*_elt` path's math (`_bound_add` notes the folded literal bounds).
        # Its `jacobian_term` sums the same expression (below), so no
        # companion `logjac` plate is emitted.
        u = Symbol(:_ppl_int_, e.name)
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} =
                $(_interval_constrain(e.lo, e.hi, u; dot = true))),
            :($u::AbstractVector{Float64} =
                $(_interval_unconstrain(e.lo, e.hi, e.name; dot = true))),
        ]
    end
    if e.transform === :floored
        u = Symbol(:_ppl_floor_, e.name)
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} =
                $(_bound_add(e.lo, :(exp.($u)); dot = true))),
        ]
    end
    if e.transform === :upper
        # Parameterized bound ⇒ not in the (parameterless) bijector registry;
        # hand-rolled broadcast edges over the block view, with the host
        # `_*_elt` path's math. Its `jacobian_term` sums the unconstrained
        # view (below), so no companion `logjac` plate is emitted.
        u = Symbol(:_ppl_up_, e.name)
        return Expr[
            :($u::AbstractVector{Float64} = $view_read),
            :($(e.name)::AbstractVector{Float64} =
                $(_bound_sub(e.hi, :(exp.($u)); dot = true))),
            :($u::AbstractVector{Float64} =
                log.($(_bound_sub(e.hi, e.name; dot = true)))),
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

# Per-element scalar names of a positive (`:exp`) `:vector` entry
# (`_ppl_v_<name>_<i>`, read by the joint-factor prior and the structural
# L-entry algebra), and the constrained value/Jacobian pair of a `:simplex`
# entry. All `_ppl_`-hygienic.
_vector_elt_name(name::Symbol, i::Int) = Symbol(:_ppl_v_, name, :_, i)
_vector_elt(e::LayoutEntry, i::Int) = _vector_elt_name(e.name, i)
_vector_simplex_pair(e::LayoutEntry) = Symbol(:_ppl_simplex_pair_, e.name)
_vector_simplex_logjac(e::LayoutEntry) = Symbol(:_ppl_simplex_logjac_, e.name)
# The packed block of an ordered or simplex entry, and the coordinates an
# ordered entry's Jacobian reads (all but the first). Each is read as a
# statement of its own, as every other block is: a read nested inside a larger
# expression has no row form, so RK's footprint would count the whole packed
# vector as read (`factor_support`, `partition_footprint`).
_vector_block(e::LayoutEntry) = Symbol(:_ppl_vector_block_, e.name)
_vector_ordered_tail(e::LayoutEntry) = Symbol(:_ppl_ordered_tail_, e.name)
_vector_ordered_tail_first(e::LayoutEntry) = Symbol(:_ppl_first_ordered_tail_, e.name)

# Leveled vector edges (thresholds, threshold-coefficient packs,
# simplexes): ONE vector-valued statement chain per entry, bound to the
# entry's own name, whatever its (data-inferred) level count — the
# statement count does not grow with K (core constraint 1). Ordered and
# simplex execution use runtime loops.
# Consumers gather from the vector (`Ref(v)` plate inputs, `v[idx]`).
# Native ordered execution uses one buffer. The native simplex path reads
# packed coordinates directly.
# Size specializations
# (empty packs emit nothing — their consumers never read them; a one-element
# ordered vector is its coordinate; a 1-simplex is the constant `[1.0]`)
# are finite shape cases, never per-level expansion.
#
# `:exp` entries are the joint-factor scales, whose size is the joint
# outcome count — program structure, not data — and whose readers are the
# structural per-entry L algebra; they keep per-element scalar edges (the
# plate `:exp` bijector shape).
function _vector_transform_statements(e::LayoutEntry)
    range = _entry_range(e)
    # A data-sized vector takes its length from its own block.
    size_read = e.positions === nothing ? e.size : _read_length(_vector_block(e))
    if e.transform === :exp
        bij = _bijector_name(:exp)
        return Expr[:($(_vector_elt(e, i))::Float64 =
            $(bij)().constrain($(coordinate_read(e.offset + i - 1))))
            for i in 1:e.size]
    end
    if e.transform === :identity
        e.size == 0 && return Expr[]
        # Materialize the ordinary range slice for the cumulative-cutpoint
        # validity reduction.
        return Expr[:($(e.name)::AbstractVector{Float64} =
            unconstrained[$range])]
    end
    if e.transform === :ordered
        e.size == 0 && return Expr[]
        e.size == 1 && e.positions === nothing &&
            return Expr[:($(e.name)::AbstractVector{Float64} = view(unconstrained, $range))]
        f = GlobalRef(@__MODULE__, :_ordered_constrain_kernel)
        block, tail = _vector_block(e), _vector_ordered_tail(e)
        tail_range = e.positions === nothing ? :($(e.offset + 1):$(e.offset + e.size - 1)) :
            :($(let f = e.positions[1]; f isa Int ? f + 1 : _vector_ordered_tail_first(e) end):$(e.positions[2]))
        return Expr[
            :($block::AbstractVector{Float64} = view(unconstrained, $range)),
            :($(e.name)::AbstractVector{Float64} = $f($block, 1, $size_read)),
            # Read only by the Jacobian (`jacobian_term`); pruned otherwise.
            :($tail::AbstractVector{Float64} = view(unconstrained, $tail_range)),
        ]
    end
    # :simplex — stick-breaking (thin-layer-owned, not Stan's ILR).
    K = e.size + 1
    K == 1 && return Expr[:($(e.name)::AbstractVector{Float64} = ones(1))]
    pair, jac = _vector_simplex_pair(e), _vector_simplex_logjac(e)
    f = GlobalRef(@__MODULE__, :_simplex_constrain_logjac_kernel)
    block = _vector_block(e)
    return Expr[
        :($block::AbstractVector{Float64} = view(unconstrained, $range)),
        :($pair = $f($block, 1, $size_read)),
        :($(e.name)::AbstractVector{Float64} = $pair[1]),
        :($jac::Float64 = $pair[2]),
    ]
end

# Constrained Cholesky entries / logistic-sigma + theta temps for an
# LKJ-factor entry: `_ppl_rl_<L>_<i>_<j>` (lower triangle incl.
# diagonal; downstream effect/prior cells read these scalars directly),
# `_ppl_rsg_<L>_<i>_<j>` (sigma(u)), `_ppl_rth_<L>_<i>_<j>` (theta).
# All `_ppl_`-hygienic.
_rl_name(L::Symbol, i::Int, j::Int) = Symbol(:_ppl_rl_, L, :_, i, :_, j)
_rzb_name(L::Symbol, i::Int, j::Int) = Symbol(:_ppl_rzb_, L, :_, i, :_, j)
_rd_name(L::Symbol, i::Int, j::Int) = Symbol(:_ppl_rd_, L, :_, i, :_, j)

# LKJ Cholesky edges: scalar-unrolled twin of the host vine
# `lkj_chol_constrain` (IDENTICAL scalar ops in the IDENTICAL order —
# `tanh`/`sqrt` calls, `^2`, the left-associated running product — so
# in-graph and host agree bit-for-bit). No matrix ever materializes: the effect
# reads the `_ppl_rl_` scalars directly (fully transparent to the
# planner and the reverse pass — no new Enzyme surface). The `_ppl_rzb_`
# partial temps are shared with the log-Jacobian twin. K=1 emits its
# constant `[1.0]` edge only.
_lkj_corr_transform_statements(e::LayoutEntry) =
    _lkj_vine_statements(e.name, _lkj_dim(e.size),
        p -> coordinate_read(e.offset + p - 1))

# The vine edges for one K×K factor named `L`, reading the p-th packed
# partial through `coord(p)`. `stacked=true` is the stratified-draws
# form: `coord(p)` reads that partial for every stratum at once and every
# edge is the same op broadcast over the S-vector, so S never multiplies
# statements (core constraint 1). `L[1,1]` stays the scalar `1.0` in both.
function _lkj_vine_statements(L::Symbol, K::Int, coord; stacked::Bool = false)
    T = stacked ? :(AbstractVector{Float64}) : :Float64
    call(f, x) = stacked ? :($f.($x)) : :($f($x))
    mul(a, b) = stacked ? :($a .* $b) : :($a * $b)
    comp(zt) = stacked ? :(sqrt.(1 .- $zt .^ 2)) : :(sqrt(1 - $zt^2))
    stmts = Expr[:($(_rl_name(L, 1, 1))::Float64 = 1.0)]
    K == 1 && return stmts
    # Partials in column-block packing order (Stan's unconstrained
    # order — the p-th coordinate is z[i,j] for j in 2:K, i in 1:j-1).
    p = 0
    for j in 2:K, i in 1:(j - 1)
        p += 1
        z = _rzb_name(L, i, j)
        push!(stmts, :($z::$T = $(call(:tanh, coord(p)))))
    end
    # Row j: L[j,1] = z[1,j], L[j,i] = z[i,j]*d[i-1,j], L[j,j] = d[j-1,j],
    # with the running product d[i,j] = d[i-1,j]*√(1-z[i,j]²) from 1.0.
    for j in 2:K
        d = 1.0
        for i in 1:(j - 1)
            z = _rzb_name(L, i, j)
            push!(stmts, :($(_rl_name(L, j, i))::$T = $(i == 1 ? z : mul(z, d))))
            next = i == j - 1 ? _rl_name(L, j, j) : _rd_name(L, i, j)
            push!(stmts, :($next::$T = $(mul(d, comp(z)))))
            d = next
        end
    end
    return stmts
end

# The vine's log-Jacobian over the `_ppl_rzb_` partials of factor `L`
# (`nothing` at K=1); `stacked=true` sums each partial's S-vector.
function _lkj_vine_logjac(L::Symbol, K::Int; stacked::Bool = false,
        pointwise::Bool = false)
    K == 1 && return nothing
    terms = Any[]
    for j in 2:K, i in 1:(j - 1)
        z = _rzb_name(L, i, j)
        w = (j - i + 1) / 2
        push!(terms, stacked && pointwise ? :($w .* log.(1 .- $z .^ 2)) :
            stacked ? :($w * sum(log.(1 .- $z .^ 2))) :
            :($w * log(1 - $z^2)))
    end
    return foldl((a, b) -> stacked && pointwise ? :($a .+ $b) : :($a + $b), terms)
end

"""
    jacobian_term(entry) -> Union{Nothing,Expr,Symbol}

This entry's log-Jacobian contribution (`nothing` for identity). A scalar
constrained support splices the bijector's `logjac` endpoint over the same
coordinate the `constrain` edge reads (shared via structural CSE); a per-cell
latent (plate) block sums its companion `logjac` plate (`_plate_logjac_name`);
a leveled vector entry reads the ordered coordinates or the simplex
transform's scalar Jacobian; an LKJ
declared LKJ entry reads the retained sum over the partials shared with its
transform. Legacy structural-margin LKJ blocks sum their named scalar edges.
"""
function jacobian_term(e::LayoutEntry)
    e = _program_entry(e)
    e.kind === :external_plate && return :(sum($(_plate_logjac_name(e.name))))
    e.kind === :external && return _external_edge(e, :_sampling_logjac)
    e.transform === :identity && return nothing
    if e.kind === :scan && !isempty(e.scan_blocks)
        terms = Any[jacobian_term(b) for b in e.scan_blocks if b.transform !== :identity]
        return _sum_terms(terms)
    end
    if e.kind === :coefficient
        # Interval runs (uniform coefficients) hand-roll the per-cell
        # Jacobian sum — the same expression as the host `_logjac_elt`
        # path, so no companion `logjac` plate is emitted.
        e.transform === :interval || throw(ContractValidationError(
            "[layout] coefficient block $(e.name) has no Jacobian rule " *
            "for $(e.transform) (admitted: :identity, :interval)"))
        return _interval_logjac(e.lo, e.hi, _entry_view(e); dot = true)
    end
    if e.kind === :cholesky_corr
        isempty(e.dims) || return _lkj_array_logjac(e.name)
        return _lkj_vine_logjac(e.name, _lkj_dim(e.size))
    end
    e.kind === :array && return _array_jacobian_term(e)
    if e.kind === :vector
        if e.transform === :ordered
            # Σ u[2:end] over one packed slice (the host `ordered_logjac`),
            # read by its own transform statement. A vector read at its
            # binding's positions has one program for every length.
            e.size < (e.positions === nothing ? 2 : 1) && return nothing
            return :(sum($(_vector_ordered_tail(e))))
        end
        if e.transform === :exp
            # :exp — Σ u (the exp log-Jacobian is the unconstrained value).
            e.size < 1 && return nothing
            terms = Any[coordinate_read(e.offset + i - 1) for i in 1:e.size]
            return foldl((a, b) -> :($a + $b), terms)
        end
        # :simplex — Σ [log(r) + log(z) + log1p(-z)] over the K−1 breaks,
        # computed alongside the constrained vector.
        e.size < 1 && return nothing
        return _vector_simplex_logjac(e)
    end
    if e.kind === :plate ||
       e.kind === :glm
        # Interval/upper plates hand-roll the per-cell Jacobian sum
        # (parameterized bounds, no companion `logjac` plate); every registry
        # transform sums its companion `logjac` plate from
        # `_plate_transform_statements`. GLM vectors use the same shape.
        # Missing companion plates must fail explicitly.
        e.transform === :interval &&
            return _interval_logjac(e.lo, e.hi, _entry_view(e); dot = true)
        if e.transform === :upper || e.transform === :floored
            # Both one-sided exp transforms have the bare-coordinate Jacobian.
            return :(sum($(_entry_view(e))))
        end
        e.transform === :exp || e.transform === :logistic ||
            throw(ContractValidationError("[layout] $(e.kind) block " *
                  "$(e.name) has no Jacobian rule for transform " *
                  "$(e.transform)"))
        return :(sum($(_plate_logjac_name(e.name))))
    end
    if e.transform === :interval
        # Parameterized bounds ⇒ hand-rolled (not in the bijector registry);
        # reads the original coordinate, even if `e.name` has rounded.
        return _interval_logjac(e.lo, e.hi, coordinate_read(e.offset))
    end
    if e.transform === :floored
        # Stan's lower-bound kernel: the bare unconstrained coordinate
        # (shared with the constrain edge via structural CSE) — no
        # truncation normalizer.
        return coordinate_read(e.offset)
    end
    if e.transform === :upper
        # Stan's upper-bound kernel: the bare unconstrained coordinate
        # (shared with the constrain edge via structural CSE) — no
        # truncation normalizer.
        return coordinate_read(e.offset)
    end
    bij = _bijector_name(e.transform)
    return :($(bij)().logjac($(coordinate_read(e.offset))))
end
