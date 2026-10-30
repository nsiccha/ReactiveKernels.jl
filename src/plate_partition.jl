# Locality of a plate cell's evaluation (decision
# `ReactiveKernels:snag.partition-graph-6306f756/decisions/2026-10-30T11-52-01-648-08uci2h`,
# option 1, increment 1).
#
# The cell lowering computes a value the cell reads only at some rows at those
# rows, when its producer has a row form (`_cell_pushdowns`, cell_windows.jl).
# Before this file the alternative was silent: a value without a form, or one
# some reader reads whole although that reader's own value is needed only at
# rows, was computed whole in every cell, and only a trace of recorded reads
# found it. Here the analysis keeps what it learns, and the rule is one line:
#
#     a value that every reader of the evaluation needs only at some rows is
#     computed only at those rows; otherwise it is a violation, named.
#
# Classification of each value in a cell's cone (`_cell_locality`):
#
# - `:rows` — computed at the rows the cell reads (pushed);
# - `:input` — a HAVE port, or a lazy view of inputs (`_cell_lazy_view`):
#   read at rows as a view, or whole, never computed;
# - `:data` — a bound constant (data): the same;
# - `:scalar` — a number; it has no rows;
# - `:shared` — computed whole because the evaluation genuinely needs it
#   whole: a wanted value, or a value read whole by a reader that is itself
#   at rows or legitimately whole (`sum(x)`, a simplex, the right operand of
#   `A * w`, an index array);
# - `:whole` — a VIOLATION: a value a reader asked for at rows, computed
#   whole. Its `reason` says why: `:no_form` (its producer has no row form),
#   `:untyped` (no row form, and neither a declaration, bound data nor
#   inference says whether it is a number or an array), `:unshaped` (its full
#   shape is not known at rows), `:operands` (an operand of its producer is
#   not provably an integer index or an array), or `:cascade` (it is read
#   whole by a reader that is itself such a violation; only the root is
#   reported by the error).
#
# Preparation applies the cell's `locality` (`plate_cell(…; locality)`, else
# the project preference `cell_locality`): `:error` throws
# `PartitionLocalityError` listing every root, `:warn` names each root once
# per process, `:ignore` says nothing (`_cell_locality_check!`).

"""
    PartitionLocalityError(cell, violations)

Preparation of a plate cell (or partition) with `locality = :error` found
values the cell reads at its rows that would be computed whole. Each entry of
`violations` is a `partition_locality` entry with status `:whole` whose
`reason` is not `:cascade`: the value, its producer's authored `source`, the
`reason`, and the `reader` that asked for it at rows.
"""
struct PartitionLocalityError <: Exception
    cell::Symbol
    violations::Vector{Any}
end

function Base.showerror(io::IO, err::PartitionLocalityError)
    n = length(err.violations)
    print(io, "PartitionLocalityError: evaluating `", err.cell, "` at its rows would ",
          "compute ", n, " value", n == 1 ? "" : "s", " whole:")
    for entry in err.violations
        print(io, "\n  - ", _locality_describe(entry))
    end
    print(io, "\n  A value is computed at a cell's rows only when its producer has a row ",
          "form (a dotted expression, a gather, a column read, hcat/vcat, a product ",
          "with an array left factor, a literal range view or reshape, an authored ",
          "plate); see `partition_locality`. Pass `locality = :warn` to evaluate ",
          "anyway.")
end

function _locality_describe(entry)
    reason = entry.reason === :no_form ?
        "its producer `$(entry.source)` has no row form" :
        entry.reason === :untyped ?
        "its producer `$(entry.source)` has no row form, and its type is not known to " *
        "be a number (declare it, `$(entry.value)::Float64 = …`, if it is one)" :
        entry.reason === :unshaped ?
        "its producer `$(entry.source)` has a row form, but the value's full shape is not known at rows" :
        entry.reason === :operands ?
        "its producer `$(entry.source)` has a row form, but an operand is not provably an integer index or an array" :
        entry.reason === :untyped_index ?
        "the cell's body reads it as `v[q]`, but `q` (a batched argument) is not provably an integer position or a vector of them (declare its type)" :
        entry.reason === :cascade ?
        "it is read whole by `$(entry.reader)`, which is itself computed whole" :
        "its producer `$(entry.source)`: $(entry.reason)"
    reader = entry.reason === :cascade ? "" : " (asked for at rows by `$(entry.reader)`)"
    "`$(entry.value)` is computed whole$reader: $reason"
end

# The authored source of a recipe for a message: its source expression without
# line numbers, else its operation's name.
function _locality_source(r::Recipe)
    r.source isa Expr && return string(_lane_strip_lines(r.source))
    r.source isa Symbol && return string(r.source)
    string(_opname(r.op))
end

_locality_entry(; value, status, reason, reader, source, cell) =
    (; value, status, reason, reader, source, cell)

"""
    _cell_locality(p::Plan, d::_CellPushdowns) -> Vector{NamedTuple}

The locality classification of every value in the cone of each plate cell of
`p` (see the top of this file): one entry per `(cell, value)` with `value`
(its name), `status`, `reason`, `reader` (the recipe that asked for it at rows,
or reads it whole), `source` (its producer's authored source) and `cell` (the
cell value's name). Entries are in plan order per cell, cells in plan order.
"""
function _cell_locality(p::Plan, d::_CellPushdowns)
    g = p.graph
    cidof(v) = canon_id(g, v.id)
    producers = Dict{ValueId,Vector{Recipe}}()
    readers = Dict{ValueId,Vector{Tuple{Recipe,Vector{Int}}}}()
    for r in p.recipes
        foreach(o -> push!(get!(producers, cidof(o), Recipe[]), r), r.outputs)
        for cid in unique(map(cidof, r.inputs))
            slots = findall(v -> cidof(v) == cid, collect(r.inputs))
            push!(get!(readers, cid, Tuple{Recipe,Vector{Int}}[]), (r, slots))
        end
    end
    have = Set(cidof(v) for v in p.have)
    wants = Set(cidof(v) for v in p.want)
    producer(cid) = let candidates = get(producers, cid, Recipe[])
        length(candidates) == 1 ? only(candidates) : nothing
    end
    types = _locality_value_types(p)
    static_type(v) = get(types, cidof(v), Any)
    forms = Dict{Int,Union{Nothing,_CellForm}}()
    form(r) = get!(() -> get(d.forms, r.id, nothing) === nothing ? _cell_form(r) :
                   d.forms[r.id], forms, r.id)
    name(cid) = g.values[cid].name
    # Whether the evaluation legitimately needs `cid` whole (memoized).
    legit = Dict{ValueId,Bool}()
    function whole_ok(cid)
        haskey(legit, cid) && return legit[cid]
        legit[cid] = false              # a cycle cannot occur; guard anyway
        ok = cid in wants || any(get(readers, cid, Tuple{Recipe,Vector{Int}}[])) do (r, slots)
            if r.op isa _AuthoredPlateCellOp
                atomic = typeof(r.op.plate).parameters[2]
                positions = get(d.captures, r.id, Dict{Int,Vector{Tuple{Int,Int}}}())
                unknown = get(d.untyped, r.id, Int[])
                # Read whole by the cell body: an atomic argument that is not
                # a capture read at the cell's own rows. One read that way but
                # through an index of no known type is read whole, not
                # legitimately (`:untyped_index`).
                return any(s -> s > 1 && (s - 1) in atomic && !haskey(positions, s - 1) &&
                           !((s - 1) in unknown), slots)
            end
            f = form(r)
            outputs = map(cidof, r.outputs)
            # A reader computed at rows needs the value whole only in a
            # non-row role (the right operand of a product, an index array);
            # a reader computed whole needs every input whole, legitimately
            # when it is itself legitimately whole.
            if f !== nothing && any(o -> o in d.pushed, outputs)
                return any(s -> !_cell_row_role(f.roles[s]), slots)
            end
            all(whole_ok, outputs)
        end
        legit[cid] = ok
    end
    entries = Any[]
    for cell in p.recipes
        cell.op isa _AuthoredPlateCellOp || continue
        cellname = only(cell.outputs).name
        # The cone: every value the cell's arguments depend on, in plan order.
        cone = Set{ValueId}()
        stack = ValueId[cidof(v) for v in cell.inputs[2:end]]
        while !isempty(stack)
            cid = pop!(stack)
            cid in cone && continue
            push!(cone, cid)
            r = producer(cid)
            r === nothing || append!(stack, map(cidof, r.inputs))
        end
        ordered = ValueId[]
        for r in p.recipes, o in r.outputs
            cidof(o) in cone && push!(ordered, cidof(o))
        end
        for v in p.have
            cidof(v) in cone && pushfirst!(ordered, cidof(v))
        end
        for cid in unique!(ordered)
            r = producer(cid)
            requested = get(d.requests, cid, Tuple{Int,Symbol}[])
            asker(id) = begin
                q = findfirst(q -> q.id == id, p.recipes)
                q === nothing ? nothing : only(p.recipes[q].outputs).name
            end
            # The reader a cascade follows: the one the fixpoint dropped the
            # value for, else the first reader that is not legitimately whole.
            cascade_reader() = haskey(d.dropped, cid) ? asker(d.dropped[cid]) : begin
                found = findfirst(get(readers, cid, Tuple{Recipe,Vector{Int}}[])) do (q, _)
                    !all(o -> whole_ok(cidof(o)), q.outputs)
                end
                found === nothing ? (isempty(requested) ? nothing : asker(first(requested)[1])) :
                    only(readers[cid][found][1].outputs).name
            end
            status, reason, reader = if cid in have ||
                                        (r !== nothing && _cell_lazy_view(r.op))
                (:input, isempty(requested) ? :read_whole : :read_at_rows, nothing)
            elseif r !== nothing && r.op isa _BoundConstant
                (:data, isempty(requested) ? :read_whole : :read_at_rows, nothing)
            elseif cid in d.pushed
                (:rows, :row_form, nothing)
            elseif static_type(g.values[cid]) <: Number
                (:scalar, :number, nothing)
            elseif whole_ok(cid)
                (:shared, isempty(requested) ? :never_at_rows : :needed_whole, nothing)
            elseif haskey(d.refused, cid)
                # A value whose type neither a declaration, bound data nor
                # inference knows may be a number (no rows, nothing to refuse)
                # or an array computed whole: the refusal says so, and a
                # declaration (`s::Float64 = …`) settles it.
                untyped = d.refused[cid] === :no_form &&
                    !(static_type(g.values[cid]) <: AbstractArray) &&
                    !(r !== nothing && r.op isa Union{_AuthoredPlateOp,_AuthoredScanOp})
                (:whole, untyped ? :untyped : d.refused[cid],
                 isempty(requested) ? cellname : asker(first(requested)[1]))
            elseif haskey(d.dropped, cid)
                # Pushed, then dropped: a reader computed at rows reads it in a
                # row role but cannot take it there (its full shape is not
                # known in that role).
                (:whole, :unshaped, asker(d.dropped[cid]))
            else
                (:whole, :cascade, cascade_reader())
            end
            push!(entries, _locality_entry(; value = name(cid), status, reason, reader,
                source = r === nothing ? nothing : _locality_source(r), cell = cellname))
        end
    end
    entries
end

# The types of a plan's values as far as declarations, bound data and
# inference over the recipes' operations say (`_promote_op`, as the
# type-stability check infers them): `Any` where nothing is known. Only
# whether a value is a number or an array matters here, so abstract types
# such as `AbstractVector{Float64}` are kept.
function _locality_value_types(p::Plan)
    _plan_value_types(p, map(valtype, p.have)) do recipe, input_types
        recipe.op isa _BoundConstant && return (typeof(recipe.op.value),)
        map(recipe.outputs) do value
            valtype(value) === Any || return valtype(value)
            recipe.op isa Union{_AuthoredPlateOp,_AuthoredScanOp,_AuthoredPlateCellOp} &&
                return Any
            length(recipe.outputs) == 1 || return Any
            # A literal read `a[k...]` of an array of numbers is one of them;
            # a value with a row form over arrays of numbers is an array of
            # numbers (Base computes its elements from its operands').
            literal = _footprint_literal_read(recipe)
            if literal !== nothing
                S = input_types[first(literal)]
                S isa Type && S <: AbstractArray && eltype(S) <: Number && return eltype(S)
            end
            f = _cell_form(recipe)
            if f !== nothing && f.kind in (:pointwise, :identity, :columns, :hcat, :vcat,
                                           :range_view, :reshape, :product)
                elements = [eltype(S) for S in input_types
                            if S isa Type && S <: AbstractArray && eltype(S) <: Number]
                isempty(elements) || return AbstractArray{promote_type(elements...)}
            end
            T = try
                _promote_op(recipe.op, input_types...)
            catch
                Any
            end
            T === Union{} ? Any : T
        end
    end
end

# The root violations of a cell's entries: `:whole` entries that are not
# cascades of another.
_locality_roots(entries, cell) =
    Any[e for e in entries if e.cell === cell && e.status === :whole && e.reason !== :cascade]

const _LOCALITY_WARNED = Set{String}()
const _LOCALITY_WARNED_LOCK = ReentrantLock()

# Applies each cell's `locality` to the plan's classification (see the top of
# this file). Called by `_lower_with_ops_collecting` once the pushdowns of the
# lowered plan are known.
function _cell_locality_check!(p::Plan, d::_CellPushdowns)
    entries = nothing
    for r in p.recipes
        r.op isa _AuthoredPlateCellOp || continue
        mode = r.op.locality === nothing ? _cell_locality_preference() : r.op.locality
        mode === :ignore && continue
        entries === nothing && (entries = _cell_locality(p, d))
        cell = only(r.outputs).name
        roots = _locality_roots(entries, cell)
        isempty(roots) && continue
        mode === :error && throw(PartitionLocalityError(cell, roots))
        for root in roots
            key = string(cell, '\0', root.value, '\0', something(root.source, ""))
            fresh = lock(_LOCALITY_WARNED_LOCK) do
                key in _LOCALITY_WARNED ? false : (push!(_LOCALITY_WARNED, key); true)
            end
            fresh || continue
            @warn "ReactiveKernels: evaluating `$cell` at its rows computes a value whole: " *
                  _locality_describe(root) *
                  ". Pass `locality = :error` to refuse it, or set the `cell_locality` " *
                  "preference; `partition_locality` lists every value."
        end
    end
    nothing
end

"""
    partition_locality(kernel::PreparedKernel) -> Vector{NamedTuple}

The locality classification of every value each plate cell (or partition) of
a prepared kernel reads: one entry per cell and value, with fields `value`,
`status` (`:rows`, `:input`, `:data`, `:scalar`, `:shared` or `:whole`),
`reason`, `reader`, `source` and `cell`. A `:whole` entry is a value the cell
reads at its rows but computes whole; its `reason` is `:no_form`, `:unshaped`,
`:operands`, or `:cascade` when it is read whole by another `:whole` value.
Empty for a kernel without cells. The classification is the one the kernel's
preparation applied (`plate_cell`'s `locality`); for a kernel prepared with
`bound=`, it is that of the residual program.
"""
function partition_locality(kernel::PreparedKernel)
    p = _lowering_plan(kernel.plan)
    q, cells = _cell_outlined_pushdowns(p)
    cells === nothing && return Any[]
    _cell_locality(q, cells)
end

# ------------------------------------------------------------- partitions

# The linear positions a partition runs: a `Vector{Int}`, as given, or any
# integer vector converted.
@inline _partition_rows(rows::Vector{Int}) = rows
@inline _partition_rows(rows::AbstractVector{<:Integer}) = Vector{Int}(rows)
_partition_rows(rows) = throw(ArgumentError(
    "plate_partition: the partition's rows must be an integer vector of linear " *
    "positions in the plate's domain; got $(typeof(rows))"))

# The rows must lie in the plate's one-dimensional domain.
@inline function _partition_check_rows(shape::Tuple, rows::Vector{Int})
    length(shape) == 1 || throw(ArgumentError(
        "plate_partition: a partition runs over a one-dimensional plate domain; " *
        "this plate's domain has $(length(shape)) dimensions"))
    _cell_check_row(shape, rows)
end

"""
    _PartitionTable()

The positions of each key: `table(key)` maps every distinct entry of `key`
(the key of each cell of a plate, in the plate's linear order) to the
positions holding it, in order. Reads only the key, so under `bound=` it is
computed once per binding.
"""
struct _PartitionTable end
function (::_PartitionTable)(key)
    table = Dict{eltype(key),Vector{Int}}()
    for (position, label) in enumerate(vec(key))
        push!(get!(() -> Int[], table, label), position)
    end
    table
end
_readable_callee(::_PartitionTable) = :partition_table
_opname(::_PartitionTable) = "partition_table"

"""
    _PartitionLanes()

The positions of one label in a partition table: `lanes(table, label)`, empty
when no cell of that plate has the label (the plate contributes nothing to
the partition; `_PartitionLabel` has checked that some named plate has it).
"""
struct _PartitionLanes end
const _PARTITION_NO_ROWS = Int[]
@inline (::_PartitionLanes)(table::Dict{K,Vector{Int}}, label) where {K} =
    get(table, label, _PARTITION_NO_ROWS)
_readable_callee(::_PartitionLanes) = :partition_lanes
_opname(::_PartitionLanes) = "partition_lanes"

"""
    _PartitionLabel()

The label a partition runs, checked against the named plates' tables:
`check(label, tables...)` is `label` when some plate has a cell with it, and
throws `ArgumentError` naming the keys otherwise.
"""
struct _PartitionLabel end
@inline function (::_PartitionLabel)(label, tables::Vararg{Any,N}) where {N}
    any(table -> haskey(table, label), tables) || _partition_missing_label(label, tables)
    label
end
@noinline function _partition_missing_label(label, tables)
    labels = unique!(sort!(String[string(k) for table in tables for k in keys(table)]))
    throw(ArgumentError("plate_partition: no cell has the key $(repr(label)); the keys " *
                        "are $(join(labels, ", "))"))
end
_readable_callee(::_PartitionLabel) = :partition_label
_opname(::_PartitionLabel) = "partition_label"

"""
    plate_partition(spec::KernelSpec, plates; by, index = :partition, name = :partition_total,
                    locality = :error, partitioned = ()) -> KernelSpec

A new kernel in which the value `name` is the sum of the cells of the authored
plates named in `plates` (one name or several) whose key equals the label held
by the HAVE port `index`: a *partition* of the plates, such as one subject's
observations. `by` gives each plate's key, a value of `spec` (usually a bound
data port) with one entry per cell of the plate in the plate's linear order:
one name for every plate, or a named tuple or pairs `plate => key`.

A partition runs as ONE compiled pass over its cells: each named plate loops
once over the positions holding the label, through the plate's own scalar body
and the same native plate lowering, and every value the partition's cells read
at their rows is computed once per partition at the union of those rows
(`plate_cell`'s row forms). Values the partition reads whole are computed
once per call. The positions of each label are computed from the key once per
binding under `bound=` (a label's lookup is then one dictionary read per
call). A label that no cell of any named plate has throws `ArgumentError`; a
named plate without a cell of the label contributes nothing (an empty
partition of it). A plate whose domain is not one-dimensional is refused at
evaluation.

`locality` is as for [`plate_cell`](@ref), with the default `:error`: a
partition is prepared only when every value its cells read at rows is
computed at those rows; otherwise preparation throws a
[`PartitionLocalityError`](@ref) naming each such value.
[`partition_locality`](@ref) reports every value.

`partitioned` is the separability rule: pairs `port => labels`, each a
per-partition HAVE port of `spec` with the label of each of its linear
positions (`missing` for a position every partition may read, such as a
shared parameter). [`partition_separability`](@ref) then checks, for one
label, that the partition reads no other label's positions of those ports;
the check needs the kernel's arguments, so it runs where an exemplar exists
(`ReactiveKernelsPPL.prepare_partition_sampler` runs it for every label).

The default boundary of the result is the original HAVE ports plus `index`,
and the WANT `name`. `spec` is unchanged. `index` may name an existing HAVE
port of `spec` to partition several specs at the same label.
"""
function plate_partition(spec::KernelSpec, plates; by,
                         index::Symbol = :partition, name::Symbol = :partition_total,
                         locality::Union{Nothing,Symbol} = :error, partitioned = ())
    locality in (nothing, :error, :warn, :ignore) || throw(ArgumentError(
        "plate_partition: locality must be :error, :warn, :ignore or nothing; got $(repr(locality))"))
    names = plates isa Symbol ? (plates,) : Tuple(Symbol.(collect(plates)))
    isempty(names) && throw(ArgumentError("plate_partition: name at least one plate"))
    keys_by_plate = by isa Symbol ? Dict{Symbol,Symbol}(plate => by for plate in names) :
        Dict{Symbol,Symbol}(Symbol(k) => Symbol(v) for (k, v) in pairs(by))
    for plate in names
        haskey(keys_by_plate, plate) || throw(ArgumentError(
            "plate_partition: `by` gives no key for plate :$plate"))
    end
    rule = _partition_rule(partitioned)
    for (port, labels) in rule
        haskey(spec.ports, port) || throw(ArgumentError(
            "plate_partition: `partitioned` names :$port, which is not a port of the kernel"))
        labels isa AbstractVector || throw(ArgumentError(
            "plate_partition: `partitioned` gives :$port a $(typeof(labels)); give the " *
            "label of each of its positions (`missing` for a shared position)"))
    end
    graph = spec.graph
    haskey(spec.ports, name) && throw(ArgumentError(
        "plate_partition: the kernel already has a port :$name"))
    copied = Graph(copy(graph.values), copy(graph.recipes),
        Dict{ValueId,Vector{Int}}(id => copy(ids) for (id, ids) in graph.producers),
        copy(graph.aliases), graph.version, nothing)
    ports = copy(spec.ports)
    order = copy(spec.port_order)
    have_names = copy(spec.have_names)
    label = if haskey(ports, index)
        existing = ports[index]
        isempty(producers_of(graph, existing.id)) || throw(ArgumentError(
            "plate_partition: the index port :$index is computed by the kernel"))
        existing
    else
        _kernel_declare!(copied, ports, order, index, Any)
    end
    _kernel_push_unique!(have_names, index)
    recipes = Recipe[]
    tables = Value[]
    for plate in names
        target = spec[plate]
        producers = Recipe[graph.recipes[id] for id in producers_of(graph, target.id)
                           if graph.recipes[id].op isa _AuthoredPlateOp]
        isempty(producers) && throw(ArgumentError(
            "plate_partition: :$plate is not produced by an authored plate"))
        length(producers) == 1 || throw(ArgumentError(
            "plate_partition: :$plate has $(length(producers)) authored plate producers"))
        push!(recipes, only(producers))
        key = spec[keys_by_plate[plate]]
        table = value!(copied, Symbol(:_partition_table_, plate), Any)
        _add_recipe!(copied, (key,), (table,), _PartitionTable(), 0.0, nothing, false, nothing)
        push!(tables, table)
    end
    checked = value!(copied, Symbol(:_partition_label), Any)
    _add_recipe!(copied, (label, tables...), (checked,), _PartitionLabel(), 0.0, nothing,
                 false, nothing)
    totals = Value[]
    for (plate, recipe, table) in zip(names, recipes, tables)
        lanes = value!(copied, Symbol(:_partition_rows_, plate), Vector{Int})
        _add_recipe!(copied, (table, checked), (lanes,), _PartitionLanes(), 0.0, nothing,
                     false, nothing)
        T = valtype(only(recipe.op.kernel.plan.want))
        total = length(names) == 1 ? _kernel_declare!(copied, ports, order, name, T) :
            value!(copied, Symbol(:_partition_, plate), T)
        _add_recipe!(copied, (lanes, recipe.inputs...), (total,),
                     _AuthoredPlateCellOp(recipe.op, locality, true, rule), recipe.cost,
                     nothing, false, recipe.source)
        push!(totals, total)
    end
    if length(names) > 1
        T = mapreduce(valtype, (a, b) -> a === b ? a : Any, totals)
        total = _kernel_declare!(copied, ports, order, name, T)
        _add_recipe!(copied, Tuple(totals), (total,), +, 0.0, nothing, false, nothing)
    end
    KernelSpec(copied, ports, order, have_names, Symbol[name], nothing)
end

# ------------------------------------------------------------- footprints

# The footprint of an evaluation: the positions of each HAVE port it reads.
# A lowering in footprint mode (`_lower_with_ops_collecting(…; footprint =
# true)`) records every read the window emitter and the cell lowering make
# of a value (`_footprint_record!`), by the kind of read and the expression
# holding its rows or positions, and returns, instead of the wanted values, a
# named tuple of each HAVE port's positions (`_footprint_return`,
# `_footprint_collect`): the rows it is sliced to, the positions gathers and
# captures read, the constant positions of a bound-indexed read, or `:whole`.
# Reads of values computed whole are whole; a port read by nothing is empty.
const _FOOTPRINT = :ReactiveKernels_footprint
mutable struct _FootprintRecords
    const records::Dict{ValueId,Vector{Tuple{Symbol,Any}}}
    # Whether the lowering being footprinted has begun: a lowering nested in
    # it (a plate body, an embedded kernel) runs blind to the recording
    # (`_lower_with_ops_collecting`).
    active::Bool
end
_FootprintRecords() = _FootprintRecords(Dict{ValueId,Vector{Tuple{Symbol,Any}}}(), false)

_footprint_recording() = get(task_local_storage(), _FOOTPRINT, nothing) !== nothing
function _footprint_record!(cid::ValueId, kind::Symbol, expr)
    recording = get(task_local_storage(), _FOOTPRINT, nothing)
    recording === nothing && return
    push!(get!(() -> Tuple{Symbol,Any}[], recording.records, cid), (kind, expr))
    nothing
end

# The return expression of a footprint body: each HAVE port's name, value and
# recorded reads, plus the reads of the recipes the window emitter never
# saw (computed whole: a bound-indexed gather or literal range view reads
# constant positions, anything else the whole port).
function _footprint_return(p::Plan, cells, names)
    recording = task_local_storage(_FOOTPRINT)
    g = p.graph
    cidof(v) = canon_id(g, v.id)
    pushed = cells === nothing ? Set{ValueId}() : cells.pushed
    producers = Dict{ValueId,Recipe}()
    for r in p.recipes, o in r.outputs
        producers[cidof(o)] = r
    end
    wants = Set(cidof(v) for v in p.want)
    entries = Any[]
    for port in p.have
        cid = cidof(port)
        records = Any[Expr(:tuple, QuoteNode(kind), expr === nothing ? nothing : expr)
                      for (kind, expr) in get(recording.records, cid, Tuple{Symbol,Any}[])]
        cid in wants && push!(records, Expr(:tuple, QuoteNode(:whole), nothing))
        for r in p.recipes
            r.op isa _AuthoredPlateCellOp && continue
            any(o -> cidof(o) in pushed, r.outputs) && continue
            slots = findall(v -> cidof(v) == cid, collect(r.inputs))
            isempty(slots) && continue
            f = _cell_form(r)
            literal = f === nothing ? _footprint_literal_read(r) : nothing
            for s in slots
                if f !== nothing
                    constant = _footprint_constant(r, f, s, producers, cidof)
                    push!(records, constant === nothing ?
                        Expr(:tuple, QuoteNode(:whole), nothing) :
                        Expr(:tuple, QuoteNode(:positions), constant))
                elseif literal !== nothing && first(literal) == s
                    push!(records, Expr(:tuple, QuoteNode(:literal), last(literal)))
                else
                    push!(records, Expr(:tuple, QuoteNode(:whole), nothing))
                end
            end
        end
        push!(entries, Expr(:tuple, QuoteNode(port.name), names[cid], Expr(:tuple, records...)))
    end
    Expr(:call, GlobalRef(@__MODULE__, :_footprint_collect), Expr(:tuple, entries...))
end

# The constant positions a recipe computed whole reads of its input `s`, when
# its form gathers that input by a bound index or views a literal range of it;
# `nothing` otherwise (the input is read whole).
function _footprint_constant(r::Recipe, f::_CellForm, s::Int, producers, cidof)
    role = f.roles[s]
    if role in (:gather, :linear_gather)
        positions = Int[]
        for (k, via) in zip(f.indices[s], f.vias[s])
            producer = get(producers, cidof(r.inputs[k]), nothing)
            producer !== nothing && producer.op isa _BoundConstant || return nothing
            index = producer.op.value
            index isa AbstractArray{<:Integer} || return nothing
            role === :linear_gather || return nothing
            via == 0 || return nothing
            append!(positions, vec(index))
        end
        return positions
    elseif role === :range_source && f.kind === :range_view
        all(part -> part isa Int, f.literal) || return nothing
        return collect(_cell_range(f.literal))
    end
    nothing
end

# A recipe that reads one literal position of an input, `a = u[12]` (RKPPL's
# scalar coordinate reads): `(the input's position, the indices)`.
function _footprint_literal_read(r::Recipe)
    analyzed = _cell_native(r)
    analyzed === nothing && return nothing
    native, params, _ = analyzed
    native isa Expr && native.head === :ref && length(native.args) >= 2 || return nothing
    source = native.args[1]
    source isa Symbol && all(i -> i isa Int, native.args[2:end]) || return nothing
    j = findfirst(==(source), params)
    j === nothing && return nothing
    (j, Tuple(Int.(native.args[2:end])))
end

# The footprint named tuple from the body's recorded reads.
function _footprint_collect(entries::Tuple)
    names = Tuple(first(entry) for entry in entries)
    positions = map(entries) do (_, port, records)
        _footprint_positions(port, records)
    end
    NamedTuple{names}(positions)
end
function _footprint_positions(port, records::Tuple)
    port isa AbstractArray || return isempty(records) ? Int[] : :whole
    positions = Int[]
    for (kind, read) in records
        kind === :whole && return :whole
        if kind in (:rows, :cell, :gathered_rows)
            rows = kind === :cell ? Int[first(Tuple(read))] : _cell_partition_positions(read)
            # A port of one row read as a lane is extruded: every row reads it.
            kind === :gathered_rows || size(port, 1) != 1 || (rows = Int[firstindex(port, 1)])
            for row in rows, J in CartesianIndices(Base.tail(axes(port)))
                push!(positions, LinearIndices(port)[row, Tuple(J)...])
            end
        elseif kind in (:gathered, :positions)
            append!(positions, _cell_partition_positions(read))
        elseif kind === :literal
            push!(positions, LinearIndices(port)[read...])
        else
            throw(ArgumentError("unknown footprint read kind $kind"))
        end
    end
    unique!(sort!(positions))
end
@inline _cell_partition_positions(q::Integer) = Int[q]

const _FOOTPRINT_KERNELS = IdDict{Any,Any}()
const _FOOTPRINT_LOCK = ReentrantLock()

"""
    partition_footprint(kernel::PreparedKernel, args...) -> NamedTuple

The positions of each HAVE port of `kernel` that evaluating it with `args`
(its HAVE arguments, in order: for a partition, the label last) reads:
`port => Vector{Int}` of linear positions (sorted, unique), or `:whole` when
the evaluation reads the port whole (a scalar port, a port read by a value
computed whole). The positions follow from the bound index data and the
label by the same row analysis that computes the partition: a lazy branch
contributes both arms, a bound-indexed read its constant positions. For a
partition of a packed parameter vector this is its exact coordinate set, the
support of its gradient (`ReactiveKernelsPPL.partition_footprint`).
"""
function partition_footprint(kernel::PreparedKernel, args...)
    footprint = lock(_FOOTPRINT_LOCK) do
        get!(() -> partition_footprint_kernel(kernel), _FOOTPRINT_KERNELS, kernel)
    end
    Base.invokelatest(footprint, args...)
end

"""
    partition_footprint_kernel(kernel::PreparedKernel) -> PreparedKernel

The kernel [`partition_footprint`](@ref) evaluates: called with `kernel`'s HAVE
arguments, it returns their footprint. `partition_footprint` prepares and keeps
one per kernel for the process; a caller evaluating many labels prepares it
once here and holds it itself.
"""
function partition_footprint_kernel(kernel::PreparedKernel)
    p = _lowering_plan(kernel.plan)
    ast, ops, recipes = _sharing_cell_plans() do
        _lower_with_ops(p; footprint = true)
    end
    _prepare(p, ast, ops, recipes)
end

# ----------------------------------------------------------- separability

# `partitioned` as pairs `port => labels`: given as pairs, a tuple or vector
# of pairs, a named tuple or a dictionary.
function _partition_rule(partitioned)
    entries = partitioned isa Pair ? (partitioned,) :
        partitioned isa Union{Tuple,AbstractVector} && all(x -> x isa Pair, partitioned) ?
            partitioned : partitioned isa Union{NamedTuple,AbstractDict} ? pairs(partitioned) :
            throw(ArgumentError("plate_partition: `partitioned` must be pairs `port => labels`, " *
                "each a port with the label of each of its positions; got $(repr(partitioned))"))
    rule = Pair{Symbol,AbstractVector}[]
    for (k, v) in entries
        k isa Union{Symbol,AbstractString} || throw(ArgumentError(
            "plate_partition: `partitioned` names ports by symbol; got $(repr(k))"))
        v isa AbstractVector || throw(ArgumentError(
            "plate_partition: `partitioned` gives :$k a $(typeof(v)); give the label of each " *
            "of its positions (`missing` for a shared position)"))
        push!(rule, Symbol(k) => v)
    end
    Tuple(rule)
end

"""
    PartitionSeparabilityError(cell, label, port, positions, labels)

A partition declared separable in a port (`plate_partition`'s `partitioned`)
reads positions of it that belong to other labels: `positions` are those
positions and `labels` their labels (`:whole` when the partition reads the
port whole).
"""
struct PartitionSeparabilityError <: Exception
    cell::Symbol
    label::Any
    port::Symbol
    positions::Any
    labels::Any
end

function Base.showerror(io::IO, err::PartitionSeparabilityError)
    print(io, "PartitionSeparabilityError: partition ", repr(err.label), " of `", err.cell,
          "` reads ")
    if err.positions === :whole
        print(io, "`", err.port, "` whole, so every other partition's positions of it")
    else
        n = length(err.positions)
        print(io, n, " position", n == 1 ? "" : "s", " of `", err.port,
              "` that belong to other partitions: ")
        shown = min(n, 8)
        print(io, join(("$(err.positions[k]) (label $(repr(err.labels[k])))" for k in 1:shown),
                       ", "))
        n > shown && print(io, ", …")
    end
    print(io, ". A separable partition reads only its own and the shared positions of ",
          "a partitioned port; lift a coupling across partitions into a shared value.")
end

"""
    partition_separability(kernel::PreparedKernel, args...)

Check the separability rule of the partition in `kernel`
(`plate_partition`'s `partitioned`) for the evaluation with `args` (the HAVE
arguments, the label last): every position the partition reads of a
partitioned port carries its own label or `missing`. Returns the footprint
([`partition_footprint`](@ref)); throws [`PartitionSeparabilityError`](@ref)
naming the port and the other partitions' positions otherwise. A kernel with
no partition, or one with no rule, passes.
"""
function partition_separability(kernel::PreparedKernel, args...; footprint = nothing)
    footprint === nothing && (footprint = partition_footprint(kernel, args...))
    p = kernel.plan
    for r in p.recipes
        r.op isa _AuthoredPlateCellOp && r.op.rows && !isempty(r.op.partitioned) || continue
        cell = only(r.outputs).name
        label = _partition_label_of(p, r, args)
        for (port, labels) in r.op.partitioned
            haskey(footprint, port) || continue
            read = footprint[port]
            read === :whole && throw(PartitionSeparabilityError(cell, label, port, :whole, :whole))
            foreign = Int[q for q in read if !ismissing(labels[q]) && !isequal(labels[q], label)]
            isempty(foreign) || throw(PartitionSeparabilityError(cell, label, port, foreign,
                                                                 labels[foreign]))
        end
    end
    footprint
end

# The label an evaluation runs: the argument of the HAVE port that feeds the
# partition's label check (`_PartitionLabel`), through its lanes.
function _partition_label_of(p::Plan, partition::Recipe, args)
    g = p.graph
    lanes = canon_id(g, first(partition.inputs).id)
    producer(cid) = (q = findfirst(r -> length(r.outputs) == 1 &&
                                        canon_id(g, only(r.outputs).id) == cid, p.recipes);
                     q === nothing ? nothing : p.recipes[q])
    r = producer(lanes)
    r !== nothing && r.op isa _PartitionLanes || return nothing
    checked = producer(canon_id(g, r.inputs[2].id))
    checked !== nothing && checked.op isa _PartitionLabel || return nothing
    port = canon_id(g, first(checked.inputs).id)
    index = findfirst(v -> canon_id(g, v.id) == port, p.have)
    index === nothing ? nothing : args[index]
end
