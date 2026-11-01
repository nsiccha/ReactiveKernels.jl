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
          "with an array left factor, a literal range view or reshape, a reduction over ",
          "the axes other than the first such as `vec(sum(x; dims = 2))`, a transposed ",
          "vector, an authored plate); see `partition_locality`. Pass `locality = :warn` ",
          "to evaluate anyway.")
end

function _locality_describe(entry)
    reason = entry.reason === :no_form ?
        "its producer `$(entry.source)` has no row form" :
        entry.reason === :untyped ?
        "its producer `$(entry.source)` has no row form, and its type is not known to " *
        "be a number" :
        entry.reason === :unshaped ?
        "its producer `$(entry.source)` has a row form, but the value's full shape is not known at rows" :
        entry.reason === :operands ?
        "its producer `$(entry.source)` has a row form, but an operand does not fit it (an index " *
        "not provably an integer vector, a factor not an array, a transposed value not a vector, " *
        "or a `vec` of a reduction that keeps another axis)" :
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
    # A plan without a cell has no locality to report; nothing below may
    # compile per port count for an ordinary kernel.
    any(r -> r.op isa _AuthoredPlateCellOp, p.recipes) || return NamedTuple[]
    producers = Dict{ValueId,Vector{Recipe}}()
    readers = Dict{ValueId,Vector{Tuple{Recipe,Vector{Int}}}}()
    for r in p.recipes
        foreach(o -> push!(get!(producers, cidof(o), Recipe[]), r), r.outputs)
        for cid in unique!(ValueId[_input_list(cidof, r.inputs)...])
            slots = findall(_input_list(v -> cidof(v) == cid, r.inputs))
            push!(get!(readers, cid, Tuple{Recipe,Vector{Int}}[]), (r, slots))
        end
    end
    have = Set(cidof(v) for v in p.have)
    wants = Set(cidof(v) for v in p.want)
    producer(cid) = let candidates = get(producers, cid, Recipe[])
        length(candidates) == 1 ? only(candidates) : nothing
    end
    types = d.types
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
            status, reason, reader = if cid in have || cid in d.views
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
                # or an array computed whole: the refusal says so.
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
@inline _partition_rows(rows::Vector{Int}) = _cell_canonical_rows(rows)
@inline _partition_rows(rows::AbstractVector{<:Integer}) = _cell_canonical_rows(Vector{Int}(rows))
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
                    locality = :error, partitioned = (), plan = nothing,
                    plan_constant = ()) -> KernelSpec

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

`plan = port` adds a HAVE port for [`partition_plan`](@ref), which caches
label-constant bookkeeping and data. Declare its type with `port => T` or
supply `exemplars` to `prepare`. Compute a plan for each label; cached rows
are checked against that label, and planned shapes against the arguments
at evaluation. `plan_constant` names
additional HAVE ports whose values the caller guarantees are constant per
label while reusing a plan.

The default boundary of the result is the original HAVE ports plus `index`,
and the WANT `name`. `spec` is unchanged. `index` may name an existing HAVE
port of `spec` to partition several specs at the same label.
"""
function plate_partition(spec::KernelSpec, plates; by,
                         index::Symbol = :partition, name::Symbol = :partition_total,
                         locality::Union{Nothing,Symbol} = :error, partitioned = (),
                         plan::Union{Nothing,Symbol,Pair{Symbol,<:Type}} = nothing,
                         plan_constant::Tuple{Vararg{Symbol}} = ())
    locality in (nothing, :error, :warn, :ignore) || throw(ArgumentError(
        "plate_partition: locality must be :error, :warn, :ignore or nothing; got $(repr(locality))"))
    plan_port, plan_type = plan isa Pair ? (first(plan), last(plan)) : (plan, Any)
    plan_port === nothing || plan_port !== index || throw(ArgumentError(
        "plate_partition: the plan port must differ from the index port :$index"))
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
    if plan_port !== nothing
        haskey(ports, plan_port) && throw(ArgumentError(
            "plate_partition: the kernel already has a port :$plan_port"))
        _kernel_declare!(copied, ports, order, plan_port, plan_type)
        _kernel_push_unique!(have_names, plan_port)
    end
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
                     _AuthoredPlateCellOp(recipe.op, locality, true, rule, plan_port,
                                          plan_constant), recipe.cost,
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
            slots = findall(_input_list(v -> cidof(v) == cid, r.inputs))
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

# ----------------------------------------------------------- shared prefix

# Whether a recipe's read of one mixed HAVE port is confined to shared
# positions. These are the same structural reads reported by the footprint;
# an unknown read stays in the residual unless every position is shared.
function _partition_shared_read(r::Recipe, slot::Int, labels, producers, cidof)
    form = _cell_form(r)
    positions = if form !== nothing
        _footprint_constant(r, form, slot, producers, cidof)
    else
        literal = _footprint_literal_read(r)
        literal !== nothing && first(literal) == slot && length(last(literal)) == 1 ?
            last(literal) : nothing
    end
    positions === nothing && return false
    all(i -> checkbounds(Bool, labels, i) && ismissing(labels[i]), positions)
end

# Split a selected, bound-data plan without replanning its producers. The
# original HAVE boundary stays on the residual; shared intermediates enter
# as additional ordinary HAVE values, so their cotangents can be propagated
# through a separately prepared prefix pullback. They are NOT BoundConstants.
# `varying` names controls such as the partition label and its row plan.
# This internal compiler surface is also used by the PPL sampler; it carries
# no mutable cache, backend choice, or packed-parameter naming convention.
function _partition_shared_parts(p::Plan, partitioned; varying = (), preserve_rows::Bool = false)
    # Keep computations with row forms in the residual when preparing a
    # strict shared query. Only their whole shared dependencies need caching;
    # lifting a gather or observation-sized broadcast would materialize work
    # that the residual can instead evaluate at the requested rows.
    row_values = Set{ValueId}()
    row_types = Dict{ValueId,Any}()
    if preserve_rows
        p, cells = _cell_outlined_pushdowns(_lowering_plan(p))
        if cells !== nothing
            union!(row_values, cells.pushed)
            row_types = cells.types
        end
    end
    cidof(v) = canon_id(p.graph, v.id)
    rules = Dict(_partition_rule(partitioned))
    # Scan each declaration once, rather than once per recipe that reads it.
    mixed = Dict(cidof(v) => rules[v.name] for v in p.have
                 if haskey(rules, v.name) && !all(ismissing, rules[v.name]))
    shared = Set(cidof(v) for v in p.have
                 if !haskey(mixed, cidof(v)) && !(v.name in varying))
    # Only data already bound into the plan may be duplicated as constant
    # recipes on both sides. Other shared HAVE values still vary with theta.
    static, _, static_owned = _partial_split(p, Set{ValueId}())
    prefix, residual, owned = _partial_split(p, shared, Set{Int}()) do r, available
        if any(v -> cidof(v) in row_values, r.outputs)
            # Retain literal shared slices of a mixed input as owned cache
            # values. They extract parameter blocks, not observation rows.
            # The shared-read proof below still checks every extracted index.
            form = _cell_form(r)
            (form !== nothing && form.kind === :range_view &&
             any(v -> haskey(mixed, cidof(v)), r.inputs)) || return false
        end
        all(enumerate(r.inputs)) do (slot, input)
            cid = cidof(input)
            cid in available ||
                (haskey(mixed, cid) &&
                 _partition_shared_read(r, slot, mixed[cid], p.producer, cidof))
        end
    end
    have_ids = Set(cidof(v) for v in p.have)
    frontier = Value[v for v in _partial_constants(p, owned, residual)
                     if !(cidof(v) in static_owned) && !(cidof(v) in have_ids)]
    isempty(frontier) && return nothing
    # The residual no longer contains the producers that proved a frontier
    # operand's type. Retain their numeric element/rank facts so, for example,
    # transpose(weights) still proves a single row. Do not retain a concrete
    # array container: a caller may own a copied cache rather than a view.
    map!(frontier, frontier) do v
        T = typeintersect(valtype(v), _cell_element_type(get(row_types, cidof(v), Any)))
        T === Union{} || T === valtype(v) ? v : Value{T}(v.id, v.name)
    end

    # Prune against authoritative outputs in forward execution order before
    # walking backwards, so collateral duplicates cannot replace a producer.
    assigned = copy(have_ids)
    authoritative = Dict{Int,Vector{ValueId}}()
    for r in p.recipes
        outputs = ValueId[]
        for output in r.outputs
            cid = cidof(output)
            cid in assigned && continue
            push!(assigned, cid)
            push!(outputs, cid)
        end
        authoritative[r.id] = outputs
    end
    needed = Set(cidof(v) for v in frontier)
    selected = Recipe[]
    for r in Iterators.reverse(prefix)
        any(cid -> cid in needed, authoritative[r.id]) || continue
        push!(selected, r)
        union!(needed, (cidof(v) for v in r.inputs))
    end
    reverse!(selected)
    prefix_have = Value[v for v in p.have if cidof(v) in needed]
    prefix_plan = _partial_subplan(p, prefix_have, frontier, selected)
    residual_ids = Set(r.id for r in residual)
    union!(residual_ids, (r.id for r in static))
    residual_recipes = Recipe[r for r in p.recipes if r.id in residual_ids]
    residual_plan = _partial_subplan(p, [p.have; frontier], p.want, residual_recipes)
    (; prefix = prefix_plan, residual = residual_plan, frontier)
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
(its HAVE arguments, in order, including a plan port when present) reads:
`port => Vector{Int}` of linear positions (sorted, unique), or `:whole` when
the evaluation reads the port whole (a scalar port, a port read by a value
computed whole). The positions follow from the bound index data and the
label or cell index by the same row analysis that computes the evaluation:
a lazy branch contributes both arms, a bound-indexed read its constant
positions. Retained branches use capture row windows when bound indices
prove both arms' requests in range; unproven requests may conservatively
contribute the whole capture. Arm arithmetic remains lazy.
For a partition or cell of a packed parameter vector this is its structural read
set, a conservative support for its gradient; numerical zeros and
cancellations do not remove positions. This also works on a `plate_cell`
kernel, with its cell index among `args`. For a planned partition, pass
`nothing` in the plan port: footprint lowering computes its own bookkeeping.
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
one per kernel for the process; a caller evaluating many labels or cells
prepares it once here and holds it itself. Call a freshly prepared kernel
through `Base.invokelatest` when crossing Julia's world-age boundary.
"""
function partition_footprint_kernel(kernel::PreparedKernel)
    p = _lowering_plan(kernel.plan)
    ast, constants, callables, recipes = _sharing_cell_plans() do
        _lower_with_ops(p; footprint = true)
    end
    _prepare(p, ast, constants, _native_table(ast, callables), recipes)
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
arguments in order, including a plan port when present): every position read
of a partitioned port carries its own label or `missing`. Returns the footprint
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

# ------------------------------------------------------------------ plans

# The constant work of a partition (its rows from the table, every pushed
# value's rows and unions, the positions a read bisects, the bounds checks,
# the slices of bound lanes) depends on the label and the bound data only.
# With a plan port (`plate_partition(…; plan)`), the generated body reads that
# bookkeeping from the port's value, a named tuple `partition_plan` computes
# once per label, and the differentiated body carries none of it. The split
# is by dependency over the emitted statements: a statement reads only the
# label, bound constants (`_BoundConstant` getters, the partition table ops)
# and other such statements, or it stays in the body.
const _PLAN_ARG = :__plan__
# `_PLAN_MODE` / `_plan_mode()` (graphops.jl): set while `partition_plan_kernel(spec; …)`
# prepares; the lowering then emits the plan body instead of the partition's.

# The operations a plan may apply: a value computed from bound data and the
# label alone is the same on every call, so a pure operation over such
# arguments (a source, a plain function, a partition op, a bound value of any
# kind: an index column, a data column, a design matrix) is computed once per
# label. Plates, scans and cells stay in the body.
_plan_eligible_op(op) = op isa Union{_PartitionTable,_PartitionLabel,_PartitionLanes,
                                     _BoundConstant,_KernelSourceOp,_TypeOperation} ||
                        op isa Function
# Whether a bound value is index bookkeeping (as against data); informational.
_plan_bookkeeping(::Integer) = true
_plan_bookkeeping(v::AbstractArray) = _plan_bookkeeping_type(eltype(v))
_plan_bookkeeping(v::AbstractDict) = _plan_bookkeeping_type(Base.valtype(v))
_plan_bookkeeping(v::Tuple) = all(_plan_bookkeeping, v)
_plan_bookkeeping(@nospecialize(_)) = false
# An element type of index bookkeeping: integers, or arrays, ranges and
# tuples of them, nested (a ragged row structure is a vector of row vectors).
function _plan_bookkeeping_type(@nospecialize(T))
    T <: Integer && return true
    T <: AbstractArray && return _plan_bookkeeping_type(eltype(T))
    T <: Tuple && T isa DataType && return all(_plan_bookkeeping_type, T.parameters)
    false
end

# Whether `ex` reads nothing that varies per call: no data symbol (an unbound
# HAVE port other than the label, or a value computed from one) and no
# operation other than a plan-eligible pure operation. Bound data and
# explicitly label-constant ports may be read by the plan.
function _plan_eligible(ex, data::Set{Symbol}, ops)
    ex isa Symbol && return !(ex in data)
    ex isa Expr || return true
    if ex.head === :call
        _plan_mutating_call(ex, ops) && return false
        # A type the body computes (`typeof`, `eltype`, `_promote_op`, a
        # plate's element type) is a constant inference folds; read from the
        # port it would be a run-time `DataType`, every use of it a dynamic
        # dispatch, and native Reverse would meet the operation table's untyped
        # metadata in a run-time `return_type` call.
        _plan_call_name(ex.args[1]) in _PLAN_TYPE_CALLS && return false
        slot = _callable_slot(ex.args[1])
        if slot !== nothing
            _plan_eligible_op(ops[slot]) || return false
            return all(a -> _plan_eligible(a, data, ops), ex.args[2:end])
        end
        return all(a -> _plan_eligible(a, data, ops), ex.args)
    end
    slot = _constant_slot(ex)
    slot === nothing || return _plan_eligible_op(ops[slot])
    ex.head in (:tuple, :ref, :., :kw, :parameters, :vect, :quote, :&&, :||, :comparison,
                :(::), :curly, :if) || return false
    all(a -> _plan_eligible(a, data, ops), ex.args)
end

const _PLAN_TYPE_CALLS = Set{Symbol}([:typeof, :eltype, :promote_type, :_promote_op,
                                      :_plate_result_eltype])
# The locals the lowering binds to types (`_bind_native_type_exprs!`, a
# plate's element type): never planned, for the reason above.
_plan_type_local(name::Symbol) = (text = String(name);
    startswith(text, "##native_type") || startswith(text, "##plate_eltype"))

# The helpers that read only the shapes of their arguments: a shape of a value
# computed from the data is constant for arguments of one shape, so the plan
# holds it, and the body checks the plan's shape against the value's on every
# call (`_cell_plan_check`).
const _PLAN_SHAPE_HELPERS = Set{Symbol}([:_cell_leaf_shape, :_cell_product_shape,
    :_cell_factor_shape, :_cell_column_shape, :_cell_hcat_shape, :_cell_vcat_shape,
    :_cell_vcat_starts, :_plate_broadcast_shape, :_plate_require_axes, :_cell_dims_shape,
    :_cell_range_shape, :axes])
_plan_call_name(f) = f isa GlobalRef ? f.name :
    Meta.isexpr(f, :., 2) && f.args[2] isa QuoteNode ? f.args[2].value :
    f isa Symbol ? f : nothing

# Whether `ex` is a shape statement's value: a shape helper's call whose
# arguments are shape expressions or data symbols read for their shape, or a
# tuple of such calls and plan expressions. A bare data symbol is a value, not
# a shape: an alias of one stays in the body.
function _plan_shape_statement(ex, data::Set{Symbol}, ops)
    ex isa Expr || return false
    ex.head === :call && return _plan_shape_eligible(ex, data, ops)
    ex.head === :tuple && return all(ex.args) do a
        _plan_eligible(a, data, ops) || _plan_shape_statement(a, data, ops)
    end
    false
end
# Whether `ex` reads only shapes: plan-eligible, a data symbol an enclosing
# shape helper reads for its shape, or a shape helper's call over such.
function _plan_shape_eligible(ex, data::Set{Symbol}, ops)
    _plan_eligible(ex, data, ops) && return true
    ex isa Symbol && return true                 # a data value, read for its shape
    ex isa Expr || return false
    if ex.head === :call
        name = _plan_call_name(ex.args[1])
        name in _PLAN_SHAPE_HELPERS || return false
        return all(a -> _plan_shape_eligible(a, data, ops), ex.args[2:end])
    end
    ex.head === :tuple && return all(a -> _plan_shape_eligible(a, data, ops), ex.args)
    false
end

# The plan's shape against the value's: arguments of another shape need a plan
# of their own.
@inline function _cell_plan_check(planned, current)
    planned == current || _plan_mismatch(planned, current)
    current
end
@noinline _plan_mismatch(planned, current) = throw(ArgumentError(
    "partition plan: the arguments' shapes differ from the plan's ($(current) against " *
    "$(planned)); compute the plan for arguments of these shapes (`partition_plan`)"))

# The names a statement assigns anywhere inside it (plain and tuple
# assignments, loop variables, locals).
function _assigned_names!(names::Set{Symbol}, ex)
    ex isa Expr || return names
    if ex.head === :(=) || ex.head === :local
        lhs = ex.args[1]
        lhs isa Symbol && push!(names, lhs)
        Meta.isexpr(lhs, :tuple) && foreach(x -> x isa Symbol && push!(names, x), lhs.args)
        Meta.isexpr(lhs, :(::), 2) && lhs.args[1] isa Symbol && push!(names, lhs.args[1])
    end
    foreach(a -> _assigned_names!(names, a), ex.args)
    names
end
function _symbol_refs!(refs::Set{Symbol}, ex)
    ex isa Symbol && (push!(refs, ex); return refs)
    ex isa Expr && foreach(a -> _symbol_refs!(refs, a), ex.args)
    refs
end

# The emitted body split into the plan's statements (each assigning or
# checking something that varies with the label only) and the body's rest, with
# the plan symbols the rest reads and their field names (`plan_k`, positional:
# the plan-mode lowering of the same plan names them the same way).
function _partition_plan_split(body::Expr, data_names, ops)
    # Find writes before moving any statements. A write through an alias
    # must keep its backing allocation in the body too, even when only the
    # alias (or a reduction of the backing array) would be a plan field.
    bindings = Dict{Symbol,Vector{Any}}()
    function bind(ex)
        ex isa Expr || return
        if ex.head === :(=) || (_plan_update_head(ex.head) &&
                !startswith(String(ex.head), ".") && ex.args[1] isa Symbol)
            lhs = Expr(:(=), ex.args[1], nothing)
            for name in _assigned_names!(Set{Symbol}(), lhs)
                push!(get!(() -> Any[], bindings, name), ex.args[2])
            end
        end
        foreach(bind, ex.args)
    end
    bind(body)
    names = _assigned_names!(Set{Symbol}(), body)
    mutated = _written_plan_symbols(body.args, names, ops; reassign = false)
    forced = copy(mutated)
    union!(forced, (name for (name, values) in bindings if length(values) > 1))
    # A function of a written value may alias it. Conservatively follow all
    # bindings upstream; no alias promise is required of a pure helper.
    # Rebinding a scalar accumulator does not write its input values. Only
    # actual object mutations require following possible aliases upstream.
    pending = collect(mutated)
    followed = copy(mutated)
    while !isempty(pending)
        name = pop!(pending)
        for value in get(bindings, name, ())
            # This allocator returns fresh storage. Writing its result does
            # not write the axis marker, element-type inputs or shape used
            # to allocate it. Following those as aliases would pull the
            # marker's entire row-bookkeeping ancestry back into the body.
            # Keep unknown calls conservative; match the owned allocator's
            # identity, never an authored function with the same name.
            Meta.isexpr(value, :call) &&
                value.args[1] === GlobalRef(@__MODULE__, :_plate_similar_output) &&
                continue
            for ref in intersect(_symbol_refs!(Set{Symbol}(), value), names)
                ref in followed && continue
                push!(forced, ref)
                push!(followed, ref)
                push!(pending, ref)
            end
        end
    end
    _partition_plan_classify(body, data_names, ops, forced)
end
function _plan_mutating_call(ex, ops)
    slot = _callable_slot(ex.args[1])
    f = slot === nothing ? ex.args[1] : ops[slot]
    name = f isa Function ? nameof(f) : _plan_call_name(f)
    name isa Symbol && name !== :! && endswith(String(name), "!")
end
_plan_update_head(head) = endswith(String(head), "=") &&
    !(head in (:(=), :(==), :(===), :(!=), :(!==), :(<=), :(>=), :(=>)))
# Mutating functions need not write their first argument (`broadcast!`), and
# indexed/property updates write their base. Read-only arguments may be kept
# in the body conservatively; preserving ownership takes priority here.
function _written_plan_symbols(main, names::Set{Symbol}, ops; reassign = true)
    written = Set{Symbol}()
    record(ex) = union!(written, intersect(_symbol_refs!(Set{Symbol}(), ex), names))
    target(ex) = if ex isa Expr && ex.head in (:ref, :.)
        target(ex.args[1])  # indices and property names are read, not written
    else
        record(ex)
    end
    visit(ex) = begin
        ex isa Expr || return
        if ex.head === :call && _plan_mutating_call(ex, ops)
            foreach(record, ex.args[2:end])
        elseif ex.head === :(=)
            lhs = ex.args[1]
            (reassign || (lhs isa Expr && lhs.head in (:ref, :.))) && target(lhs)
        elseif _plan_update_head(ex.head)
            (reassign || startswith(String(ex.head), ".") || !(ex.args[1] isa Symbol)) &&
                target(ex.args[1])
        end
        foreach(visit, ex.args)
    end
    foreach(visit, main)
    written
end
function _partition_plan_classify(body::Expr, data_names, ops, forced::Set{Symbol})
    # Every symbol the body assigns anywhere is data until a plan statement
    # assigns it, so a reference ahead of a later or nested assignment can
    # never leave the body; the hidden cache arguments are data too.
    data = union!(_assigned_names!(Set{Symbol}(), body), data_names,
                  (_CACHES_ARG, _CACHE_APPLY_ARG))
    kinds = Symbol[]
    for st in body.args
        kind = if st isa Expr && st.head === :(=) && st.args[1] isa Symbol &&
                  (_plan_type_local(st.args[1]) || st.args[1] in forced)
            :main
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol &&
                  _plan_eligible(st.args[2], data, ops)
            delete!(data, st.args[1])
            :plan
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol &&
               _plan_shape_statement(st.args[2], data, ops)
            delete!(data, st.args[1])
            :shape
        elseif st isa Expr && st.head === :call && _plan_eligible(st, data, ops)
            :plan
        else
            :main
        end
        push!(kinds, kind)
    end
    # Cache a shape read from active values only when another plan statement
    # needs it. In particular, a final plate's loop axes belong in the value
    # kernel: planning them would execute the entire active predictor just to
    # discover those axes, although no row bookkeeping consumes them. Keep the
    # original shape expression in the body, preserving its axes and checks.
    shapes = Dict(st.args[1] => st.args[2] for (k, st) in enumerate(body.args)
                  if kinds[k] === :shape)
    refs = Set{Symbol}()
    for (k, st) in enumerate(body.args)
        kinds[k] === :plan && _symbol_refs!(refs, st)
    end
    needed_shapes = intersect(refs, keys(shapes))
    pending_shapes = collect(needed_shapes)
    while !isempty(pending_shapes)
        name = pop!(pending_shapes)
        for ref in _symbol_refs!(Set{Symbol}(), shapes[name])
            haskey(shapes, ref) && !(ref in needed_shapes) || continue
            push!(needed_shapes, ref)
            push!(pending_shapes, ref)
        end
    end
    for (k, st) in enumerate(body.args)
        kinds[k] === :shape && !(st.args[1] in needed_shapes) && (kinds[k] = :main)
    end
    any(k -> k !== :main, kinds) || return nothing
    # The plan's body: its statements with every statement they depend on (a
    # shape of a value computed from the data needs that value, computed from
    # the exemplar), in the body's order.
    wanted = Set{Symbol}()
    for (k, st) in enumerate(body.args)
        kinds[k] === :main || _symbol_refs!(wanted, st)
    end
    keep = falses(length(body.args))
    for k in reverse(eachindex(body.args))
        st = body.args[k]
        if kinds[k] !== :main
            keep[k] = true
        elseif any(n -> n in wanted, union!(_assigned_names!(Set{Symbol}(), st),
                    _written_plan_symbols((st,), wanted, ops)))
            keep[k] = true
            _symbol_refs!(wanted, st)
        end
    end
    plan = Any[st for (k, st) in enumerate(body.args) if keep[k] && !(st isa LineNumberNode)]
    # The kernel's body: everything else, a shape statement checked against the
    # plan's value where it stood.
    main = Any[]
    assigned = Symbol[]
    for (k, st) in enumerate(body.args)
        if kinds[k] === :plan
            st isa Expr && st.head === :(=) && push!(assigned, st.args[1])
        elseif kinds[k] === :shape
            push!(assigned, st.args[1])
            push!(main, Expr(:call, GlobalRef(@__MODULE__, :_cell_plan_check), st.args[1],
                             st.args[2]))
        else
            push!(main, st)
        end
    end
    refs = Set{Symbol}()
    foreach(st -> _symbol_refs!(refs, st), main)
    needed = Symbol[name for name in assigned if name in refs]
    isempty(needed) && return nothing
    fields = Symbol[Symbol(:plan_, k) for k in eachindex(needed)]
    # A typed declaration of a plan symbol (`local x::T`) precedes its read
    # from the port, where the body's own assignment stood after it.
    declared = Set{Symbol}(needed)
    declarations = Any[st for st in main if _declares_any(st, declared)]
    filter!(st -> !_declares_any(st, declared), main)
    (plan, Any[declarations..., main...], needed, fields)
end
_declares_any(st, names::Set{Symbol}) =
    st isa Expr && st.head === :local && !isempty(intersect(_assigned_names!(Set{Symbol}(), st), names))

# Planning and footprint queries reuse the existing row-bookkeeping helpers.
# Inlining their vector loops and bounds checks at every read makes a wide
# query's native body much larger than its source. Keep those calls separate
# here; shape helpers still fold normally, and value/AD bodies are unchanged.
const _PARTITION_METADATA_HELPERS = Set{Symbol}([
    :_cell_slice_positions, :_cell_rows_view, :_cell_canonical_rows,
    :_cell_leaf_row, :_cell_range_rows, :_cell_reshape_rows,
    :_cell_vcat_part_rows, :_cell_window_positions, :_cell_take_positions,
    :_cell_union_rows, :_cell_union_rows_sorted, :_cell_gather_row,
    :_cell_factor_rows, :_cell_check_row, :_partition_check_rows, :_partition_rows,
])
function _partition_metadata_calls(body::Expr)
    function annotate(ex)
        ex isa Expr || return ex
        ex.head === :quote && return ex
        call = Expr(ex.head, (annotate(a) for a in ex.args)...)
        if ex.head === :call && ex.args[1] isa GlobalRef &&
                ex.args[1].mod === (@__MODULE__) &&
                ex.args[1].name in _PARTITION_METADATA_HELPERS
            return Expr(:macrocall, GlobalRef(Base, Symbol("@noinline")), LineNumberNode(0), call)
        end
        call
    end
    macroexpand(@__MODULE__, annotate(body))
end

# The body of a kernel whose partition has a plan port: in plan mode the plan's
# statements returning the named tuple of plan fields; otherwise the rest of
# the body reading those fields from the port, typed by the plan function's
# inferred return type. `nothing` leaves the body as emitted (no plan port,
# nothing to plan, or a plan whose type inference cannot name concretely).
function _partition_plan_lowering(p::Plan, body::Expr, argexprs, runtime_ops, nm,
                                  plan_mode::Bool)
    cell = findfirst(r -> r.op isa _AuthoredPlateCellOp && r.op.plan !== nothing, p.recipes)
    cell === nothing && return nothing
    port = p.recipes[cell].op.plan
    position = findfirst(v -> v.name === port, p.have)
    position === nothing && return nothing
    # The label port is the input of the label check (`_PartitionLabel`); it
    # varies per call but names the plan, so it is not data.
    labels = unique(Symbol[nm(r.inputs[1]) for r in p.recipes if r.op isa _PartitionLabel])
    # A port whose value is constant per label (`plan_constant`) is not data.
    constant = p.recipes[cell].op.plan_constant
    data = Symbol[nm(v) for v in p.have
                  if !(nm(v) in labels) && v.name !== port && !(v.name in constant)]
    split = _partition_plan_split(body, data, runtime_ops)
    split === nothing && return nothing
    plan, main, needed, fields = split
    # A plan belongs to its label. The label's table lookup is planned too,
    # so the body checks the caller's label before reading cached rows. Keep
    # the selected row identities, not the label's concrete type: mixed key
    # types in one column must still produce one concrete plan type.
    planned = setdiff(_assigned_names!(Set{Symbol}(), Expr(:block, plan...)),
                      _assigned_names!(Set{Symbol}(), Expr(:block, main...)))
    checks = [r for r in p.recipes if r.op isa _PartitionLabel &&
              all(v -> nm(v) in planned, r.outputs)]
    guards = [Expr(:call, GlobalRef(@__MODULE__, :_partition_plan_label), nm(r.inputs[1]),
                   Expr(:tuple, (nm(v) for v in r.inputs[2:end])...)) for r in checks]
    push!(fields, :partition_labels)
    plan_body = Expr(:block, plan..., Expr(:return, Expr(:call,
        Expr(:curly, GlobalRef(Core, :NamedTuple), Expr(:tuple, (QuoteNode(f) for f in fields)...)),
        Expr(:tuple, needed..., Expr(:tuple, guards...)))))
    plan_body = _partition_metadata_calls(plan_body)
    plan_mode && return plan_body
    # The port's declared type names the plan's type (`plate_partition(…;
    # plan = port => T)`, T the type of a plan the plan kernel computed); an
    # untyped port is typed by one evaluation on empty data, which serves a
    # plan that reads no data.
    T = valtype(p.have[position])
    if T === Any
        T = _partition_plan_type(Expr(:function, Expr(:tuple, argexprs...), plan_body), p,
                                 runtime_ops, length(argexprs) - 2 - length(p.have),
                                 port, position, labels, nm)
    elseif !(T <: NamedTuple && isconcretetype(T))
        throw(ArgumentError("partition plan: the plan port :$port is declared $T, not the " *
            "concrete named-tuple type of its plan; `partition_plan_kernel(spec; …)` " *
            "computes a plan of that type, declare it with `plan = :$port => typeof(plan)`"))
    end
    reads = Any[Expr(:(=), _PLAN_ARG, Expr(:(::), nm(p.have[position]), T))]
    push!(reads, Expr(:call, GlobalRef(@__MODULE__, :_partition_plan_labels_check),
        Expr(:call, GlobalRef(Core, :getfield), _PLAN_ARG, QuoteNode(:partition_labels)),
        Expr(:tuple, (nm(r.inputs[1]) for r in checks)...)))
    for (name, field) in zip(needed, fields)
        push!(reads, Expr(:(=), name, Expr(:call, GlobalRef(Core, :getfield), _PLAN_ARG,
                                             QuoteNode(field))))
    end
    Expr(:block, reads..., main...)
end

_partition_plan_label(label, tables::Tuple) =
    (tables, map(table -> get(table, label, _PARTITION_NO_ROWS), tables))
@inline function _partition_plan_label_matches(plan, label)
    tables, rows = plan
    all(map((table, expected) -> get(table, label, _PARTITION_NO_ROWS) === expected,
            tables, rows))
end
@inline function _partition_plan_labels_check(planned, current)
    all(map(_partition_plan_label_matches, planned, current)) || throw(ArgumentError(
        "partition plan: labels $(repr(current)) differ from the plan's labels; " *
        "compute the plan for these labels (`partition_plan`)"))
    nothing
end

# The concrete named-tuple type the plan function returns, found by one
# evaluation at preparation: on the exemplars when `prepare` received them
# (`exemplars`, one value per HAVE port, `nothing` for the plan port), else on
# the first bound label with empty stand-ins for the data ports, which serves
# a plan that reads no unbound data. Inference cannot name the type: a
# generated function's call crosses a world-age lookup it does not see
# through. A plan port the caller asked for is never silently ignored: a plan
# the evaluation cannot type is an error naming the remedy.
function _partition_plan_type(plan_ast::Expr, p::Plan, runtime_ops, extra::Int, port::Symbol,
                              position::Int, labels, nm)
    f = compile(plan_ast)
    record = _plan_exemplars()
    exemplars = record === nothing ? nothing : record.exemplars
    remedy = "prepare with `exemplars` (one value per HAVE port, `nothing` for the plan " *
             "port) or declare the port's type (`partition_plan_kernel(spec; …)`, " *
             "`plate_partition(…; plan = :$port => T)`)"
    arguments = if exemplars === nothing
        table = findfirst(op -> op isa _BoundConstant && op.value isa AbstractDict &&
                                Base.valtype(op.value) <: Vector{Int}, runtime_ops)
        keys_ = table === nothing ? () : keys(runtime_ops[table].value)
        isempty(keys_) && throw(ArgumentError(
            "partition plan: the plan port :$port is untyped and no bound label names a " *
            "plan to type it from; $remedy"))
        label = first(keys_)
        Any[nm(v) in labels ? label : _plan_dummy(valtype(v)) for v in p.have]
    else
        length(exemplars) == length(p.have) || throw(ArgumentError(
            "internal: $(length(exemplars)) exemplars for $(length(p.have)) HAVE ports"))
        _input_list(identity, exemplars)
    end
    arguments[position] = nothing
    append!(arguments, fill(nothing, extra))
    constants, callables = _lowering_tables(runtime_ops)
    plan = try
        Base.invokelatest(f, constants, _native_table(plan_ast, callables), arguments...)
    catch err
        throw(ArgumentError(
            "partition plan: the plan port :$port is untyped and evaluating its plan on " *
            (exemplars === nothing ? "empty stand-ins for the data ports" : "the exemplars") *
            " failed: $(sprint(showerror, err)); $remedy"))
    end
    T = typeof(plan)
    T <: NamedTuple && isconcretetype(T) || throw(ArgumentError(
        "partition plan: the plan of :$port has type $T, which is not a concrete named tuple"))
    record === nothing || (record.typed[] = (position, T))
    T
end
# A stand-in of the declared type of a data port the plan never reads.
function _plan_dummy(::Type{T}) where {T}
    T <: AbstractArray || return nothing
    E = eltype(T)
    E = E === Any || !isconcretetype(E) ? Float64 : E
    N = try ndims(T) catch; 1 end
    Array{E,N}(undef, ntuple(_ -> 0, N))
end

const _PLAN_KERNELS = IdDict{Any,Any}()
const _PLAN_LOCK = ReentrantLock()

"""
    partition_plan_kernel(kernel::PreparedKernel) -> PreparedKernel

The kernel [`partition_plan`](@ref) evaluates: called with `kernel`'s HAVE
arguments (the plan port's own value is ignored; pass `nothing`), it returns
the plan: the named tuple of label-constant bookkeeping and data the body reads
from its plan port. `partition_plan` prepares and keeps one per kernel for the
process; a caller evaluating many labels prepares it once here and holds it.
"""
function partition_plan_kernel(kernel::PreparedKernel)
    p = _lowering_plan(kernel.plan)
    ast, constants, callables, recipes = _sharing_cell_plans() do
        _lower_with_ops(p; partition_plan = true)
    end
    _prepare(p, ast, constants, _native_table(ast, callables), recipes)
end
"""
    partition_plan_kernel(spec::KernelSpec; have, want, bound, on_error) -> PreparedKernel

The plan kernel of a partition prepared from `spec` with these arguments
(`prepare`'s), without preparing the partition itself: it computes a label's
plan from the exemplar's data, and the type of that plan declares the plan
port of the partition to prepare (`plate_partition(…; plan = port => T)`),
which then reads every plan field with that type.
"""
partition_plan_kernel(spec::KernelSpec; kwargs...) =
    task_local_storage(() -> prepare(spec; kwargs...), _PLAN_MODE, true)

"""
    partition_plan(kernel::PreparedKernel, args...) -> NamedTuple

The plan of a partition kernel prepared with a plan port
(`plate_partition(…; plan)`): the constant bookkeeping and data of the label among
`args` (the kernel's HAVE arguments in order; pass `nothing` for the plan
port), computed from the bound data. The kernel's body reads it from the port
instead of recomputing it per call, and a derivative carries none of it:
`kernel(args..., label, partition_plan(kernel, args..., label, nothing))`.
Compute it once per label and keep it; a label no plate has throws
`ArgumentError` naming the keys.
"""
function partition_plan(kernel::PreparedKernel, args...)
    planner = lock(_PLAN_LOCK) do
        get!(() -> partition_plan_kernel(kernel), _PLAN_KERNELS, kernel)
    end
    Base.invokelatest(planner, args...)
end
