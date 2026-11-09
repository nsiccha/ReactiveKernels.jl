# Provenance is recorded where a density is emitted. Generated graph names
# stay an implementation detail; a public identity is (kind, site, entry).
struct _DensitySite
    kind::Symbol
    site::Union{Symbol,Tuple{Symbol,Symbol}}
    total::Symbol
    plate::Union{Nothing,Symbol}
    count::Int
    offset::Int
    owner::Union{Nothing,Symbol,Tuple{Symbol,Symbol},LayoutEntry}
end

_record_density!(::Nothing, args...; kwargs...) = nothing
function _record_density!(sites::Vector{_DensitySite}, stmts, kind, site, total;
        pointwise = nothing, count = 1, owner = site)
    plate = if pointwise === nothing || count == 0
        nothing
    else
        name = Symbol(:_ppl_density_cells_, length(sites) + 1)
        value = _dovar(1)
        # An identity plate exposes each atomic density to RK's cell/partition
        # lowering without duplicating any distribution or transform formula.
        append!(stmts, _plate_sum_stmts(name, Symbol(name, :_sum),
            Any[pointwise], Expr(:block, value)))
        name
    end
    push!(sites, _DensitySite(kind, site, total, plate, count, 0, owner))
    nothing
end

function _record_jacobian!(sites, stmts, e::LayoutEntry, total;
        site = e.kind === :coefficient ? e.predictor : e.name)
    if e.kind === :scan && !isempty(e.scan_blocks)
        for b in e.scan_blocks
            t = jacobian_term(b)
            t === nothing && continue
            node = Symbol(:_ppl_density_jacobian_, length(sites) + 1)
            push!(stmts, :($node::Float64 = $t))
            _record_jacobian!(sites, stmts, b, node; site)
        end
    elseif e.kind === :array && _is_lkj_stack(e.transform)
        pw = _lkj_vine_logjac(e.name, e.dims[1]; stacked = true, pointwise = true)
        _record_density!(sites, stmts, :log_jacobian, site, total;
            pointwise = pw, count = e.dims[3], owner = e)
    elseif e.kind === :array && _is_slice_transform(e.transform)
        o = _slice_transform_orientation(e.transform)
        rows = o isa _SliceRows
        n = e.dims[rows ? 1 : 2]
        index, packed = _dovar(1), _dovar(2)
        _, jfn = _slice_function_names(e.transform)
        shape = _slice_packed_dims(e.transform, e.dims)
        linear = reshape(collect(1:prod(shape)), shape...)
        windows = rows ? collect.(eachrow(linear)) : collect.(eachcol(linear))
        localdims = rows ? (1, shape[2]) : (shape[1], 1)
        slice = :(reshape($packed[$index], $(localdims...)))
        pw = Symbol(:_ppl_density_jacobian_cells_, length(sites) + 1)
        append!(stmts, _plate_sum_stmts(pw, Symbol(pw, :_sum),
            Any[QuoteNode(windows), :(Ref($(_array_flat_name(_array_slices_packed_name(e.name), 2))))],
            :($jfn($(_orientation_expr(o)), $slice))))
        _record_density!(sites, stmts, :log_jacobian, site, total;
            pointwise = pw, count = n, owner = e)
    elseif e.kind === :external_plate
        _record_density!(sites, stmts, :log_jacobian, site, total;
            pointwise = _plate_logjac_name(e.name), count = e.size, owner = e)
    elseif e.kind !== :external && e.transform in (:exp, :logistic, :interval, :floored, :upper)
        # The same scalar Jacobian emitter used by the sampler. Parameters
        # of bounds remain graph inputs, so their support is retained.
        program = _program_entry(e)
        scalar = LayoutEntry(:sampled, nothing, e.name, Symbol[], 1, 1,
            e.transform, program.lo, program.hi)
        value = _dovar(1)
        expression = _replace_density_coordinate(jacobian_term(scalar), value)
        pw = Symbol(:_ppl_density_jacobian_cells_, length(sites) + 1)
        append!(stmts, _plate_sum_stmts(pw, Symbol(pw, :_sum),
            Any[_entry_view(e)], Expr(:block, expression)))
        _record_density!(sites, stmts, :log_jacobian, site, total;
            pointwise = pw, count = e.size, owner = e)
    else
        _record_density!(sites, stmts, :log_jacobian, site, total; owner = e)
    end
    nothing
end

# One undotted multivariate draw per row/column. Slice before invoking the
# existing whole-draw density, so no unrelated independent draw is evaluated.
function _density_slice_plate!(stmts, plan, p, density, dims)
    kind = _slice_kind(p)
    count = dims[kind === :rows ? 1 : 2]
    index = _dovar(1)
    inputs = Any[:(1:$count)]
    function slice(kind, value, shape; flat = nothing)
        if flat === nothing
            push!(inputs, :(Ref($value)))
            v = _dovar(length(inputs))
            return kind === :rows ? :($v[$index, :]) : :($v[:, $index])
        end
        # Bind the draw's linear positions once. Reading the flat graph value
        # exposes exactly those coordinates, including for column draws.
        indices = reshape(collect(1:prod(shape)), shape...)
        windows = kind === :rows ? collect.(eachrow(indices)) : collect.(eachcol(indices))
        push!(inputs, QuoteNode(windows))
        at = _dovar(length(inputs))
        push!(inputs, :(Ref($flat)))
        :($(_dovar(length(inputs)))[$at])
    end
    transform = _slice_transform(Val(_slice_stem(p)), kind)
    variate = if _is_slice_transform(transform)
        shape = _slice_packed_dims(transform, Tuple(dims))
        flat = _array_flat_name(_array_slices_packed_name(p.name), 2)
        packed = slice(kind, p.name, shape; flat)
        cfn, _ = _slice_function_names(transform)
        o = _slice_transform_orientation(transform)
        localdims = kind === :rows ? (1, shape[2]) : (shape[1], 1)
        :(vec($cfn($(_orientation_expr(o)), reshape($packed, $(localdims...)))))
    else
        slice(kind, p.name, dims; flat = _array_flat_name(p.name, 2))
    end
    # An ICAR field's law takes the field, then its node vectors, with no
    # orientation: each cell evaluates one field's `icar_logdensity`.
    icar = _slice_stem(p) === :icar
    args = Any[]
    for a in density.args[(icar ? 3 : 4):end]
        if Meta.isexpr(a, :call) && a.args[1] === :_PerSlice
            rows = a.args[2] == :(_SliceRows())
            value = a.args[3]
            parameter = findfirst(q -> q.name === value && !_is_structured_array(q),
                plan.array_parameters)
            shape = parameter === nothing ? nothing :
                _array_dims(plan, plan.array_parameters[parameter])
            flat = shape === nothing ? nothing : _array_flat_name(value, length(shape))
            push!(args, slice(rows ? :rows : :cols, value, shape; flat))
        elseif a isa Number
            push!(args, a)
        else
            push!(inputs, :(Ref($a)))
            push!(args, _dovar(length(inputs)))
        end
    end
    cell = icar ? Expr(:call, GlobalRef(DistributionKernelSources,
        :icar_logdensity), variate, args...) :
        Expr(:call, density.args[1], :(_SliceWhole()), variate, args...)
    pw = Symbol(:_ppl_density_slices_, p.name)
    append!(stmts, _plate_sum_stmts(pw, Symbol(pw, :_sum), inputs, cell))
    pw
end
function _replace_density_coordinate(ex, value)
    ex == :(unconstrained[1]) && return value
    ex isa Expr || return ex
    Expr(ex.head, (_replace_density_coordinate(a, value) for a in ex.args)...)
end

"An authored atomic density event, identified by kind, site and one-based entry."
struct DensityFactor
    kind::Symbol
    site::Symbol
    entry::Int
end

# Compact ownership metadata, independent of a density's transitive reads.
# Regular events use a strided block; ragged geometry retains bound offsets.
struct _DensityCoordinates
    start::Int
    step::Int
    width::Int
    stride::Int
    offsets::Union{Nothing,Vector{Int}}
end
_DensityCoordinates(start, step, width, stride = 1) =
    _DensityCoordinates(start, step, width, stride, nothing)

function _density_coordinates(e::LayoutEntry, count, plan)
    count <= 1 && return _DensityCoordinates(e.offset, 0, e.size)
    if e.kind === :array
        p = only(p for p in plan.array_parameters if p.name === e.name)
        rows = _is_slice_array(p) && _slice_kind(p) === :rows
        width, remainder = divrem(e.size, count)
        iszero(remainder) || error("density ownership does not match packed layout for $(e.name)")
        return rows ? _DensityCoordinates(e.offset, 1, width, count) :
            _DensityCoordinates(e.offset, width, width)
    end
    e.size == count || error("density ownership does not match packed layout for $(e.name)")
    _DensityCoordinates(e.offset, 1, 1)
end

function _density_coordinates(s::_DensitySite, plan, layout)
    owner = s.owner
    owner === nothing && return _DensityCoordinates(1, 0, 0)
    owner isa LayoutEntry && return _density_coordinates(owner, s.count, plan)
    # Joint cell geometry owns its packed window, not the scale/bound parents.
    for (name, g) in plan.cell_geometries
        g.name === owner || continue
        e = only(e for e in layout.entries if e.name === name)
        offsets = collect(Int, _layout_lookup(layout, Dict{Symbol,Any}())(g.offsets))
        length(offsets) == s.count + 1 || error("density geometry event count mismatch")
        return _DensityCoordinates(e.offset, 0, 0, 1, offsets)
    end
    predictor = owner isa Tuple ? first(owner) : owner
    coefficients = [e for e in layout.entries if e.kind === :coefficient && e.predictor === predictor]
    if !isempty(coefficients)
        start = minimum(e.offset for e in coefficients)
        if owner isa Tuple
            pred = _legacy_predictor(only(p for p in plan.predictors if p.name === predictor))
            shape = design_shape(pred, plan.columns; levelmaps = plan.levelmaps, matrices = plan.matrices)
            for b in shape.blocks
                tag = b.kind === MatrixTerm ? b.column : b.addressee
                tag === last(owner) && return _DensityCoordinates(start, 1, 1)
                start += b.width
            end
            error("density coefficient owner missing from design")
        end
        return _DensityCoordinates(start, 1, 1)
    end
    e = only(e for e in layout.entries if e.name === owner)
    _density_coordinates(e, s.count, plan)
end

# A hash of every element of a binding's data, recursively: equal for data that
# is `isequal`, and different, but for a 64-bit collision, after any in-place
# change of an entry. Julia's own `hash` of a large array reads only some of its
# elements, so it cannot detect every change.
_binding_fingerprint(x) = _fingerprint(x, UInt(0x9d46c6de), IdDict{Any,Nothing}())
_fingerprint(x::Union{Number,AbstractString,Symbol,Nothing,Missing,Type}, h::UInt, _) = hash(x, h)
function _fingerprint(x::AbstractArray, h::UInt, seen)
    h = hash(typeof(x), hash(size(x), h))
    for v in x
        h = _fingerprint(v, h, seen)
    end
    h
end
function _fingerprint(x::AbstractDict, h::UInt, seen)
    h = hash(typeof(x), hash(length(x), h))
    for (k, v) in x
        h = _fingerprint(v, _fingerprint(k, h, seen), seen)
    end
    h
end
function _fingerprint(x::Union{Tuple,NamedTuple}, h::UInt, seen)
    h = hash(keys(x), h)
    for v in x
        h = _fingerprint(v, h, seen)
    end
    h
end
function _fingerprint(x, h::UInt, seen)
    T = typeof(x)
    h = hash(T, h)
    isstructtype(T) && nfields(x) > 0 || return hash(x, h)
    # A mutable record reached again (a cycle, or shared twice) adds its
    # identity, not its fields again.
    if ismutable(x)
        haskey(seen, x) && return hash(objectid(x), h)
        seen[x] = nothing
    end
    for i in 1:nfields(x)
        h = isdefined(x, i) ? _fingerprint(getfield(x, i), h, seen) : hash(i, h)
    end
    h
end

"A binding-scoped catalogue of authored prior and coordinate-Jacobian factors."
struct DensityFactors{B,P,S,H,D}
    built::B
    plan::P
    spec::S
    have::H
    binding::D
    # The content of `binding` when the catalogue was made (`_binding_fingerprint`).
    fingerprint::UInt
    sites::Vector{_DensitySite}
    coordinates::Vector{_DensityCoordinates}
    support_exemplar::Vector{Float64}
    support_kernels::Dict{Tuple{Symbol,Symbol,Int},Any}
    support_lock::ReentrantLock
    # One query graph per selection structure (`_density_structure_spec`).
    structures::Dict{Any,Any}
    structure_lock::ReentrantLock
    # One prepared program per structure and boundary, which every derivative
    # query of that structure calls with its own selection (`_density_kernel`).
    kernels::Dict{Any,Any}
end

Base.length(c::DensityFactors) = sum(s.count for s in c.sites; init = 0)
Base.eltype(::Type{<:DensityFactors}) = DensityFactor
function Base.iterate(c::DensityFactors, state = (1, 1))
    i, entry = state
    while i <= length(c.sites)
        s = c.sites[i]
        entry <= s.count && return DensityFactor(s.kind, s.site, s.offset + entry), (i, entry + 1)
        i += 1
        entry = 1
    end
    nothing
end

"""
    density_factors(built, plan) -> DensityFactors

Inventory authored prior and coordinate-Jacobian events of a bound model.
Each `DensityFactor` has `kind` (`:prior` or `:log_jacobian`), authored `site`,
and one-based `entry`. Elementwise declarations have one event per element
(linear, column-major order); a joint law or coupled transform stays atomic.
Built-in identity transforms contribute no Jacobian events; a caller's
geometry retains its authored Jacobian even when it returns zero. Conditioned declarations
are observations and are excluded.

The catalogue belongs to this program and binding. Select events with
[`select_factors`](@ref), inspect their structural packed-coordinate support
with [`factor_support`](@ref), and evaluate them with [`prepare_query`](@ref)
or [`prepare_query_ad`](@ref). Catalogue and query preparation may inspect the
whole model; repeated selected evaluation uses precomputed row plans.
"""
function density_factors(built, plan::StructuralPlan)
    built = _binding_build(built, plan)
    _check_built_program(built, plan)
    sites = _DensitySite[]
    def = Base.invokelatest(_kernel_expr_latest, plan, built.layout; density_sites = sites)
    coordinates = [_density_coordinates(s, plan, built.layout) for s in sites]
    counts = Dict{Tuple{Symbol,Symbol},Int}()
    for (i, s) in enumerate(sites)
        owner = s.site isa Tuple ? first(s.site) : s.site
        suffix = s.site isa Tuple ? Base.tail(s.site) : ()
        name = haskey(plan.cell_latents, owner) ? first(plan.cell_latents[owner]) : owner
        path = (get(built.layout.name_paths, name, (name,))..., suffix...)
        authored = Symbol(_scope_path_string(path))
        key = (s.kind, authored)
        offset = get(counts, key, 0)
        sites[i] = _DensitySite(s.kind, authored, s.total, s.plate, s.count, offset, s.owner)
        counts[key] = offset + s.count
    end
    def = _elide_named_location_aliases(plan, def)
    def = _pointwise_arithmetic_plates(plan, def)
    def = _name_plate_cells(def, _location_value_aliases(plan, def))
    spec = _eval_kernel_def(def)
    _register_packed_check!(spec, built.layout)
    # The catalogue reads the binding's own data. A copy of it, kept to detect a
    # later in-place change, held a second copy of every bound array for the
    # catalogue's lifetime (snag per-unit-partiti-9d46c6de); its fingerprint
    # detects the same changes.
    binding = _query_bound(plan)
    fingerprint = _binding_fingerprint(binding)
    # Bind the catalogue before cutting cells from it. This lets ordinary
    # partial evaluation hoist data-only cell indices once, even when a later
    # selection fuses a plate into a single cell and drops its other outputs.
    wanted = unique(vcat([s.total for s in sites],
        [s.plate for s in sites if s.plate !== nothing]))
    if !isempty(wanted)
        prepared = Base.invokelatest(_prepare_cached, plan, spec; have = _query_have(plan),
            want = Tuple(wanted), bound = binding)
        p = prepared.plan
        old = p.graph
        g = ReactiveKernels.Graph(copy(old.values), ReactiveKernels.Recipe[],
            Dict{ReactiveKernels.ValueId,Vector{Int}}(), copy(old.aliases), old.version, nothing,
            copy(old.checks))
        for r in p.recipes
            ReactiveKernels._add_recipe!(g, r.inputs, r.outputs, r.op, r.cost,
                r.cse_key, r.effectful, r.source)
        end
        spec = ReactiveKernels.KernelSpec(g, copy(spec.ports), copy(spec.port_order),
            copy(spec.have_names), copy(spec.want_names), nothing)
    end
    DensityFactors(built, plan, spec, _query_have(plan), binding, fingerprint, sites, coordinates,
        zeros(built.layout.total),
        Dict{Tuple{Symbol,Symbol,Int},Any}(), ReentrantLock(),
        Dict{Any,Any}(), ReentrantLock(), Dict{Any,Any}())
end

"A set of authored atomic events from one density catalogue."
struct FactorSelection{C}
    catalogue::C
    factors::Vector{DensityFactor}
end
Base.length(s::FactorSelection) = length(s.factors)
Base.iterate(s::FactorSelection, state...) = iterate(s.factors, state...)
Base.eltype(::Type{<:FactorSelection}) = DensityFactor

"""
    select_factors(catalogue; kind = nothing, site = nothing, entries = :)
    select_factors(catalogue, factors)

Select a set of whole atomic density events. `kind` and `site` optionally
filter the catalogue; `entries` selects event indices within each matching
site, never coordinates within a joint event. Explicit `factors` must belong
to the catalogue. Duplicate events are counted once. Empty selections sum
to zero. Iterate the catalogue to discover the valid authored identities.
"""
function select_factors(c::DensityFactors; kind = nothing, site = nothing, entries = Colon())
    kind === nothing || kind in (:prior, :log_jacobian) ||
        throw(ArgumentError("factor kind must be :prior or :log_jacobian"))
    matches = [s for s in c.sites if (kind === nothing || s.kind === kind) &&
        (site === nothing || s.site === site)]
    site === nothing || any(s -> s.site === site, c.sites) ||
        throw(ArgumentError("unknown density site $site"))
    result = DensityFactor[]
    for (k, name) in unique([(s.kind, s.site) for s in matches])
        count = sum(s.count for s in matches if s.kind === k && s.site === name)
        indices = entries isa Colon ? (1:count) : entries isa Integer ? (entries,) : entries
        for i in indices
            i isa Integer && 1 <= i <= count || throw(ArgumentError(
                "density $k at $name has $count atomic events; invalid entry $i"))
            push!(result, DensityFactor(k, name, i))
        end
    end
    FactorSelection(c, unique(result))
end
function select_factors(c::DensityFactors, factors)
    result = unique(DensityFactor[f for f in factors])
    for f in result
        any(s -> s.kind === f.kind && s.site === f.site && s.offset < f.entry <= s.offset + s.count,
            c.sites) || throw(ArgumentError("factor $f does not belong to this catalogue"))
    end
    FactorSelection(c, result)
end

"""
    factor_coordinates(catalogue, event) -> Vector{Int}
    factor_coordinates(selection) -> Vector{Int}

Packed coordinates owned by an authored draw or transform, excluding its
hyperparameters and other read dependencies. Joint laws and coupled transforms
own their whole packed block. A density on a derived expression owns no new
coordinates; a flat prior still owns its declared coordinates despite having
empty read support. Ownership comes from lowering/layout metadata at catalogue
creation, not from `factor_support` or numerical derivatives.

For a set of fresh coordinates, select an event by these owned coordinates and
retain it whole. If only part of an atomic block is fresh, that event cannot be
split. Use `factor_support` separately to determine all inputs it reads.
"""
function factor_coordinates(c::DensityFactors, event::DensityFactor)
    select_factors(c, (event,))
    index = only(i for (i, s) in enumerate(c.sites) if s.kind === event.kind &&
        s.site === event.site && s.offset < event.entry <= s.offset + s.count)
    entry = event.entry - c.sites[index].offset
    owned = c.coordinates[index]
    if owned.offsets !== nothing
        lo, hi = owned.offsets[entry], owned.offsets[entry + 1]
        return collect((owned.start + lo):(owned.start + hi - 1))
    end
    collect(range(owned.start + (entry - 1) * owned.step; step = owned.stride, length = owned.width))
end
factor_coordinates(s::FactorSelection) =
    sort!(unique!(reduce(vcat, (factor_coordinates(s.catalogue, f) for f in s); init = Int[])))

function _check_factor_selection(built, plan, selection)
    c = selection.catalogue
    select_factors(c, selection.factors)
    _check_factor_binding(built, plan, c)
end

function _check_factor_binding(built, plan, c)
    _check_built_program(built, plan)
    _same_program(built.program, c.built.program) &&
        _binding_fingerprint(_query_bound(plan)) == c.fingerprint || throw(ContractValidationError(
        "[query] density selection belongs to another program or data binding"))
    nothing
end

const _DENSITY_TOTAL = :_ppl_density_selected
const _DENSITY_LABEL = :_ppl_density_label
const _DENSITY_PLAN = :_ppl_density_plan
const _DENSITY_PARTITION = :_ppl_density_partition
const _DENSITY_CONSTANT = :_ppl_density_constant
_density_membership(j::Int) = Symbol(:_ppl_density_membership_, j)

# Catalogue preparation can fold a whole data-only density plate. Its bound
# cells still identify independent events, but no authored plate remains to
# partition. Recognize the compiler's proof, not numerical zeros or a law name.
function _density_bound_cells(c::DensityFactors, s::_DensitySite)
    s.plate === nothing && return nothing
    g = c.spec.graph
    producers = ReactiveKernels.producers_of(g, c.spec[s.plate].id)
    length(producers) == 1 || return nothing
    op = g.recipes[only(producers)].op
    op isa ReactiveKernels._BoundConstant ? op.value : nothing
end

# Selections of one structure share one query graph: the sites they read
# whole (`totals`), the sites they read as partitioned cells (`plates`),
# whether folded data-only cells add a constant, the row plan's type and the
# compact overlay. A graph copied per selection kept a whole catalogue graph
# alive in every query and in the binding's data store.
#
# A value query (`prepare_query`) binds its selection's entries, row plan,
# constant and coordinate slots, so RK rebinds the structure's compiled
# template for every selection (`prepare(…; bound)` keeps that work per graph).
# A compact derivative query (`prepare_query_ad(...; coordinates)`, `runtime`)
# passes them on each call instead, so every query of the structure calls one
# prepared program: a rebound program is a kernel, plan and constants table of
# its own, 9.9 KB of heap per query for one scalar prior against 4.0 KB (snag
# per-unit-partiti-9d46c6de). Its memberships are constant per selection
# (`plan_constant`), so the program's own plan kernel plans each selection's
# rows once.
#
# With `slots`, the query's coordinates cover every coordinate its factors read
# (`coordinates = :support` always does), so its slot map is constant per
# selection as well: the plan then also computes the active vector's slots of
# the rows a partition reads the packed point at (`_cell_lazy_rows`), and the
# body reads those rows as a plain view of the active vector. Without it, each
# element read goes through the overlay, a sparse slot lookup and a branch, as
# coordinates that leave a read coordinate to the base need.
function _density_structure_spec(c::DensityFactors, plan, totals::Tuple, plates::Tuple,
        constant::Bool, locality, rowplan, compact::Bool; runtime::Bool = false,
        slots::Bool = false)
    # Only a compact program that reads cells takes the slot map into its plan.
    slots &= compact && runtime && !isempty(plates)
    key = (totals, plates, constant, locality, rowplan, compact, runtime, slots)
    lock(c.structure_lock) do
        get!(c.structures, key) do
            _build_density_structure_spec(c, plan, totals, plates, constant, locality,
                rowplan, compact, runtime, slots)
        end
    end
end

function _build_density_structure_spec(c, plan, totals, plates, constant, locality,
        rowplan, compact, runtime = false, slots = false)
    spec = ReactiveKernels._copy_cell_factor_spec(c.spec)
    terms = Symbol[c.sites[j].total for j in totals]
    if !isempty(plates)
        by = Dict{Symbol,Symbol}()
        for j in plates
            key = _density_membership(j)
            spec.ports[key] = value!(spec.graph, key, ReactiveKernels._PartitionIndex{Bool})
            push!(spec.port_order, key)
            by[c.sites[j].plate] = key
        end
        spec = plate_partition(spec, Symbol[c.sites[j].plate for j in plates]; by,
            index = _DENSITY_LABEL, name = _DENSITY_PARTITION, locality,
            plan = rowplan === nothing ? _DENSITY_PLAN : _DENSITY_PLAN => rowplan,
            plan_constant = runtime ? (Tuple(_density_membership(j) for j in plates)...,
                (slots ? (:_ppl_cell_slots,) : ())...) : ())
        push!(terms, _DENSITY_PARTITION)
    end
    if constant
        spec.ports[_DENSITY_CONSTANT] = value!(spec.graph, _DENSITY_CONSTANT, Float64)
        push!(spec.port_order, _DENSITY_CONSTANT)
        push!(terms, _DENSITY_CONSTANT)
    end
    output = value!(spec.graph, _DENSITY_TOTAL, Float64)
    spec.ports[_DENSITY_TOTAL] = output
    push!(spec.port_order, _DENSITY_TOTAL)
    if isempty(terms)
        add!(spec.graph, () => output, () -> 0.0)
    elseif length(terms) == 1
        add!(spec.graph, (spec[only(terms)],) => output, +)
    else
        # Two terms per `+`, left to right as Base folds `+(a, b, c...)`: a
        # selection can hold dozens of sites (a slice's prior), and Julia does
        # not infer Base's varargs fold past 32 operands, which would leave a
        # call dispatched at run time in the prepared program.
        partial = spec[terms[1]]
        for k in 2:length(terms)
            sum_k = k == length(terms) ? output :
                value!(spec.graph, Symbol(_DENSITY_TOTAL, :_, k), Float64)
            add!(spec.graph, (partial, spec[terms[k]]) => sum_k, +)
            partial = sum_k
        end
    end
    compact && _compact_overlay!(spec, plan)
    spec
end

# The query of a selection: its structure's graph (`spec`), boundary (`have`,
# `want`) and `bound` data. With `runtime`, `bound` holds the binding's data
# only, and `context` holds the selection's own values, the call arguments
# after the packed point, in `have` order. `slots` says the compact slot map
# gives every coordinate the selection reads a slot (`_density_structure_spec`).
function _density_selection_spec(built, plan, selection; locality, on_error, u0,
        checked = false, compact = false, runtime = false, slots = false)
    checked || _check_factor_selection(built, plan, selection)
    c = selection.catalogue
    totals, plates = Int[], Int[]
    constant, folded = 0.0, false
    bindings = Pair{Symbol,Any}[pairs(_query_bound(plan))...]
    context = Any[]
    select!(pair) = runtime ? push!(context, last(pair)) : push!(bindings, pair)
    for (j, s) in enumerate(c.sites)
        entries = [f.entry - s.offset for f in selection if f.kind === s.kind &&
            f.site === s.site && s.offset < f.entry <= s.offset + s.count]
        isempty(entries) && continue
        if s.plate === nothing
            push!(totals, j)
        elseif (cells = _density_bound_cells(c, s)) !== nothing
            constant += sum(i -> cells[i], entries; init = 0.0)
            folded = true
        else
            membership = ReactiveKernels._PartitionIndex(Dict(true => sort!(unique!(entries))), s.count)
            select!(_density_membership(j) => membership)
            push!(plates, j)
        end
    end
    totals, plates = Tuple(totals), Tuple(plates)
    have = (_query_have(plan)..., (_density_membership(j) for j in plates)...)
    rowplan = nothing
    plan_index = nothing
    if !isempty(plates) && runtime
        # The program's own plan kernel plans each selection
        # (`_density_kernel`); its argument stays `nothing` until then.
        select!(_DENSITY_LABEL => true)
        select!(_DENSITY_PLAN => nothing)
        plan_index = length(context)
        have = (have..., _DENSITY_LABEL, _DENSITY_PLAN)
    elseif !isempty(plates)
        planner_spec = _density_structure_spec(c, plan, (), plates, false, locality,
            nothing, false)
        planner = Base.invokelatest(_prepare_cached_partition, plan, planner_spec;
            have = (have..., _DENSITY_LABEL, _DENSITY_PLAN), want = _DENSITY_PARTITION,
            bound = (; bindings...), on_error)
        plan_value = Base.invokelatest(planner, u0, true, nothing)
        # The final graph types its plan port: ordinary native AD sees the
        # exact prepared structure, and evaluation performs no membership scan.
        rowplan = typeof(plan_value)
        push!(bindings, _DENSITY_LABEL => true, _DENSITY_PLAN => plan_value)
        have = (have..., _DENSITY_LABEL, _DENSITY_PLAN)
    end
    if folded
        select!(_DENSITY_CONSTANT => constant)
        have = (have..., _DENSITY_CONSTANT)
    end
    spec = _density_structure_spec(c, plan, totals, plates, folded, locality, rowplan, compact;
        runtime, slots)
    (; spec, have, bound = (; bindings...), want = _DENSITY_TOTAL, binding = plan,
        context = Tuple(context), plan_index)
end

# The program a structure's derivative queries share, prepared once for this
# boundary and the binding's data from the arguments of the first selection
# (`args`, `nothing` at the plan port), as `prepare_partition` prepares its
# kernel; and that program's plan kernel. Another selection of the structure
# has arguments of the same types.
function _density_kernel(c::DensityFactors, plan, spec, have, want, bound, on_error, args)
    key = (objectid(spec), have, want, on_error, typeof(args))
    lock(c.structure_lock) do
        get!(c.kernels, key) do
            kernel = Base.invokelatest(_prepare_cached, plan, spec; have, want, bound,
                on_error, exemplars = args)
            (; kernel, planner = Ref{Any}(nothing))
        end
    end
end

# The selection's call arguments with its plan, from the shared program's own
# plan kernel (`partition_plan`), when it reads cells.
function _density_context(c::DensityFactors, shared, q, args::Tuple)
    q.plan_index === nothing && return q.context
    planner = lock(c.structure_lock) do
        shared.planner[] === nothing && (shared.planner[] =
            Base.invokelatest(ReactiveKernels.partition_plan_kernel, shared.kernel))
        shared.planner[]
    end
    plan_value = Base.invokelatest(planner, args...)
    Base.setindex(q.context, plan_value, q.plan_index)
end

"Prepare the sum of selected authored density events, callable with a packed point."
function prepare_query(built, plan::StructuralPlan, selection::FactorSelection;
        on_error = nothing, locality = :error, u0 = nothing)
    exemplar = _factor_exemplar(selection.catalogue, u0)
    q = _density_selection_spec(built, plan, selection; locality, on_error, u0 = exemplar)
    Base.invokelatest(_prepare_cached, plan, q.spec; have = q.have, want = q.want, bound = q.bound, on_error)
end

# Support and row planning read exemplars, never retain or mutate a caller's
# packed point. The catalogue supplies one default point for all its events.
function _factor_exemplar(c::DensityFactors, u0)
    u0 === nothing && return c.support_exemplar
    length(u0) == c.built.layout.total || throw(ContractValidationError(
        "[query] exemplar length $(length(u0)) ≠ layout total $(c.built.layout.total)"))
    u0
end

"""
    factor_support(selection; u0 = nothing) -> Vector{Int}

Sorted structural packed-coordinate support of the selected factors, including
hyperparameters and transform dependencies. This is a conservative compiler
read footprint: numerical zeros do not remove coordinates. `u0` supplies only
an exemplar; the support follows the bound indices and both branch arms.
"""
function factor_support(s::FactorSelection; u0 = nothing)
    c = s.catalogue
    _check_factor_selection(c.built, c.plan, s)
    _factor_support(s, _factor_exemplar(c, u0))
end
function _factor_support(s::FactorSelection, exemplar)
    support = Int[]
    for factor in s
        append!(support, factor_support(s.catalogue, factor; u0 = exemplar))
    end
    sort!(unique!(support))
end
function factor_support(c::DensityFactors, factor::DensityFactor; u0 = nothing)
    # Reuse a dynamic-cell footprint across the entries of each graph segment.
    # Catalogue consumers can classify every event without compiling per entry.
    select_factors(c, (factor,))
    exemplar = _factor_exemplar(c, u0)
    s = only(s for s in c.sites if s.kind === factor.kind && s.site === factor.site &&
        s.offset < factor.entry <= s.offset + s.count)
    retained_plate = s.plate !== nothing && _density_bound_cells(c, s) === nothing
    kernel = lock(c.support_lock) do
        get!(c.support_kernels, (factor.kind, factor.site, s.offset)) do
            spec, have, want = if !retained_plate
                (c.spec, c.have, s.plate === nothing ? s.total : s.plate)
            else
                index, name = :_ppl_density_entry, :_ppl_density_event
                (plate_cell(c.spec, s.plate; index, name, locality = :error),
                    (c.have..., index), name)
            end
            q = Base.invokelatest(_prepare_cached, c.plan, spec; have, want, bound = c.binding)
            Base.invokelatest(ReactiveKernels.partition_footprint_kernel, q)
        end
    end
    args = retained_plate ? (exemplar, factor.entry - s.offset) : (exemplar,)
    footprint = Base.invokelatest(kernel, args...).unconstrained
    footprint === :whole ? collect(1:c.built.layout.total) : footprint
end

function prepare_query_ad(prepare_operator, built, plan::StructuralPlan,
        selection::FactorSelection, backend, exemplars...; on_error = nothing,
        locality = :error, coordinates = nothing)
    built = _binding_build(built, plan)
    isempty(exemplars) && throw(ArgumentError("prepare_query_ad needs a packed exemplar"))
    _check_factor_selection(built, plan, selection)
    u0 = _query_exemplar(built.layout, last(exemplars))
    _prepare_factor_query_ad(prepare_operator, built, plan, selection, backend, exemplars,
        u0; on_error, locality, coordinates)
end

"""
    prepare_query_ad(prepare_operator, built, plan, selections::AbstractVector{<:FactorSelection},
                     backend, exemplars...; coordinates = nothing, kwargs...)

Prepare a family of authored density queries in selection order. All selections
must come from one catalogue. The batch checks its bound program once, then
shares one packed exemplar across preparation. With `coordinates = :support`,
each returned `FactorQueryAD` owns a compact derivative workspace; fixed row
and coordinate metadata use storage proportional to the selected supports.
Selections with one structure (the same kinds and sites, each read whole or
by cells) share one prepared program, so a query keeps only its own entries,
row plan, slots and workspace, never a copy of the model graph.
Empty selections return zero-valued queries; an empty batch returns no queries.
Single-selection calls retain their bound-program validation.
"""
function prepare_query_ad(prepare_operator, built, plan::StructuralPlan,
        selections::AbstractVector{<:FactorSelection}, backend, exemplars...;
        on_error = nothing, locality = :error, coordinates = nothing)
    built = _binding_build(built, plan)
    isempty(exemplars) && throw(ArgumentError("prepare_query_ad needs a packed exemplar"))
    u0 = _query_exemplar(built.layout, last(exemplars))
    if isempty(selections)
        _check_built_program(built, plan)
        return coordinates === nothing ? QueryAD[] : FactorQueryAD[]
    end
    c = first(selections).catalogue
    _check_factor_binding(built, plan, c)
    for selection in selections
        selection.catalogue === c || throw(ArgumentError(
            "prepare_query_ad selections must come from one density catalogue"))
        select_factors(c, selection.factors)
    end
    [_prepare_factor_query_ad(prepare_operator, built, plan, selection, backend, exemplars,
        u0; on_error, locality, coordinates) for selection in selections]
end

function _prepare_factor_query_ad(prepare_operator, built, plan, selection, backend,
        exemplars, u0; on_error, locality, coordinates)
    compact = coordinates !== nothing
    covered = false
    if compact
        selected = coordinates === :support ? _factor_support(selection, u0) : collect(Int, coordinates)
        _check_compact_cell_indices(u0, selected, selected, built.layout.total)
        # Every coordinate the factors read has a slot, so a partition reads its
        # rows from the active vector at slots its plan computes.
        covered = coordinates === :support || issubset(_factor_support(selection, u0), selected)
    end
    # A compact query passes its selection on each call to its structure's
    # shared program. A dense query keeps a rebound program per selection, as
    # its derivative is over the whole packed vector anyway.
    q = _density_selection_spec(built, plan, selection; locality, on_error, u0,
        checked = true, compact, runtime = compact, slots = covered)
    c = selection.catalogue
    if compact
        have = (:_ppl_cell_base, :_ppl_cell_active, :_ppl_cell_slots,
            (n for n in q.have if n !== :unconstrained)...)
        point = (u0, u0[selected], _fixed_cell_slots(built.layout.total, selected))
        shared = _density_kernel(c, plan, q.spec, have, q.want, q.bound, on_error,
            (point..., q.context...))
        context = (last(point), _density_context(c, shared, q, (point..., q.context...))...)
        # A program that reads the compact overlay at a partition's rows (its
        # coordinates leave a coordinate the factors read to the base) takes a
        # `view` of it there, which native Enzyme on Julia 1.10 can need dogfood
        # repair `const_slot_reloads` to differentiate (rkppl-use §14, "Enzyme
        # floor"). The loaded Enzyme is checked before the derivative is prepared.
        shared.kernel isa ReactiveKernels.PreparedKernel &&
            ReactiveKernels._reads_lazy_view_rows(shared.kernel) &&
            ReactiveKernels._require_enzyme_repairs(backend, (:const_slot_reloads,),
                "a compact factor query that reads a parameter block at a partition's rows",
                "dogfood 0147c6f")
        prepared = Base.invokelatest(prepare_operator, shared.kernel, backend,
            Base.front(exemplars)..., Base.front(point)..., context...;
            active = :_ppl_cell_active)
        # The selection keeps its plan behind one reference, which an operator
        # that boxes its plan takes unchanged (`ReactiveKernels._ad_plan_argument`,
        # as `_LabelPlans` does): boxing it on every call allocated the reference
        # each time, 64 B for one plate and 368 B for three. Its position is one
        # past the selection's own, after the slot map `context` starts with.
        q.plan_index === nothing || (context = Base.setindex(context,
            ReactiveKernels._ad_plan_argument(prepared,
                ReactiveKernels._PartitionPlanRef(context[q.plan_index + 1])),
            q.plan_index + 1))
        return FactorQueryAD(prepared, built.layout, selected, context)
    end
    kernel = Base.invokelatest(_prepare_cached, plan, q.spec; have = q.have, want = q.want,
        bound = q.bound, on_error)
    prepared = Base.invokelatest(prepare_operator, kernel, backend,
        Base.front(exemplars)..., u0; active = :unconstrained)
    QueryAD(prepared, built.layout)
end

"""
    FactorQueryAD

A factor derivative prepared with `prepare_query_ad(...; coordinates = indices)`
or `coordinates = :support`. Its `coordinates` are fixed sorted packed indices.
ReactiveKernels derivative verbs take `(base, active)` last, where `base` is the
read-only full point and `active` supplies these coordinates. Directions and
gradient outputs have the compact length. Prepare one operator per caller.
"""
# Mutable, as the other query types are (`SamplerQuery`): `Base.invokelatest`
# passes each argument as a boxed reference, so an immutable query was copied to
# the heap with its prepared operator on every call (9 KB and 12× the direct
# call's time for a three-coordinate factor). A mutable query crosses uncopied.
mutable struct FactorQueryAD{P,L,C}
    const prepared::P
    const layout::L
    const coordinates::Vector{Int}
    # The selection's own arguments of the structure's shared program (its
    # coordinate slots, cell memberships, row plan and folded constant), which
    # every call appends after `(base, active)`.
    const context::C
end
FactorQueryAD(prepared, layout, coordinates) = FactorQueryAD(prepared, layout, coordinates, ())
for verb in (:ad_gradient, :ad_value_and_gradient, :ad_value_and_gradient!,
        :ad_pullback, :ad_value_and_pullback, :ad_value_and_pullback!,
        :ad_pushforward, :ad_value_and_pushforward,
        :ad_hvp, :ad_hvp!, :ad_gradient_and_hvp, :ad_gradient_and_hvp!,
        :ad_third, :ad_third!, :ad_gradient_hvp_and_third,
        :ad_gradient_hvp_and_third!)
    @eval ReactiveKernels.$verb(q::FactorQueryAD, args...) =
        Base.invokelatest(_factor_ad_call, ReactiveKernels.$verb, q, args...)
end
function _factor_ad_call(verb::F, q::FactorQueryAD, args::Vararg{Any,N}) where {F,N}
    N >= 2 || throw(ArgumentError("a compact factor derivative takes base, active last"))
    base, active = args[N - 1], args[N]
    length(base) == q.layout.total && length(active) == length(q.coordinates) ||
        throw(DimensionMismatch("compact factor derivative point has the wrong length"))
    verb(q.prepared, args..., q.context...)
end
