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
# - `:input` — a HAVE port, a lazy view of inputs (`_cell_lazy_view`), or a
#   `view(A, lo:hi)` of those that one reader also reads whole: read at rows
#   as a view, or whole, never computed;
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
          "vector, an authored plate, a proved per-index helper comprehension or ",
          "per-position loop over vector or matrix rows, `vec` of a value with rows, a ",
          "field of a named tuple of data, a shape check that returns its argument, ",
          "a function whose package gives it a row method); ",
          "see `partition_locality`. Pass `locality = :warn` to evaluate anyway, which ",
          "computes each such value again for every cell or partition. ", _LOCALITY_SHARED_ROUTE)
end

# The route a value that reads only shared parameters has instead of being
# computed for every partition, which `locality = :warn` would do.
const _LOCALITY_SHARED_ROUTE =
    "If it reads only shared parameters, ReactiveKernelsPPL's queries can compute it " *
    "once and reuse it: declare what a partition owns, `partitioned = (:z,)` or " *
    "`partitioned = (:z => labels,)`, or `partitioned = :none` when partitions own no " *
    "parameter."

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
                # A sparse product's operand computed whole is needed whole.
                return any(s -> !_cell_row_role(f.roles[s]) ||
                               f.roles[s] === :sparse_operand && !(cid in d.pushed), slots)
            end
            all(whole_ok, outputs)
        end
        legit[cid] = ok
    end
    # Whether `cid` is existing storage or a `view` of it, which builds no
    # elements however a reader reads it. `A[lo:hi]` shares its row form but
    # copies, so it is not one.
    function storage_view(cid)
        (cid in have || cid in d.views) && return true
        r = producer(cid)
        r === nothing && return false
        r.op isa _BoundConstant && return true
        f = form(r)
        f !== nothing && f.kind === :range_view && _cell_is_view_call(r) || return false
        storage_view(cidof(r.inputs[findfirst(==(:range_source), f.roles)]))
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
            elseif storage_view(cid)
                (:input, isempty(requested) ? :read_whole : :read_at_rows, nothing)
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
                  "preference; `partition_locality` lists every value. " *
                  _LOCALITY_SHARED_ROUTE
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
    signature = _exact_signature(kernel)
    argument_types = signature === Nothing ? nothing : Any[fieldtypes(signature)...]
    q, cells = _cell_outlined_pushdowns(p; argument_types)
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
# A key holds one entry per cell of its plate: a shorter key would leave cells in
# no partition, and the labels' sum would silently miss them.
@noinline _partition_cells_error(cells, n, key) = throw(ArgumentError(
    "plate_partition: the key :$key has $cells entries, but its plate has $n cells; " *
    "a key holds one entry per cell, in the plate's linear order"))
@inline function _partition_check_rows(shape::Tuple, rows::Vector{Int}, cells::Int, key::Symbol)
    length(shape) == 1 && length(first(shape)) != cells &&
        _partition_cells_error(cells, length(first(shape)), key)
    _partition_check_rows(shape, rows)
end

"""
    _PartitionTable()

The positions of each key: `table(key)` maps every distinct entry of `key`
(the key of each cell of a plate, in the plate's linear order) to the
positions holding it, in order. Reads only the key, so under `bound=` it is
computed once per binding.
"""
struct _PartitionTable end

# A frontend that already owns a partition's row inventory need not expand it
# to one key per cell and scan it again. Omitted labels have no selected rows;
# `cells` still describes the complete domain for the usual shape checks.
struct _PartitionIndex{K}
    table::Dict{K,Vector{Int}}
    cells::Int
end
(::_PartitionTable)(key::_PartitionIndex) = key.table

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
    _PartitionCells()

The entries of a partition key: `cells(key)`, the number of cells the key says
its plate has. Reads only the key, so under `bound=` it is computed once per
binding.
"""
struct _PartitionCells end
(::_PartitionCells)(key) = length(key)
(::_PartitionCells)(key::_PartitionIndex) = key.cells
_readable_callee(::_PartitionCells) = :partition_cells
_opname(::_PartitionCells) = "partition_cells"

"""
    _PartitionLanes(key)

The positions of one label in a partition table: `lanes(table, label, cells)`,
empty when no cell of that plate has the label (the plate contributes nothing
to the partition; `_PartitionLabel` has checked that some named plate has it).
`cells` is the key's entry count: the value does not depend on it, but the
plate's lowering reads it through this recipe to check it against the plate's
domain (`_partition_cells_arguments`). `key` names the key in that refusal.
"""
struct _PartitionLanes
    key::Symbol
end
const _PARTITION_NO_ROWS = Int[]
@inline (::_PartitionLanes)(table::Dict{K,Vector{Int}}, label, ::Int) where {K} =
    get(table, label, _PARTITION_NO_ROWS)
_readable_callee(::_PartitionLanes) = :partition_lanes
_opname(::_PartitionLanes) = "partition_lanes"

"""
    _PartitionLane(key)

The lanes of a plate whose key holds every label of the partition exactly
once (`plate_partition(…; single = true)`): `lane(table, label, cells)` is the
one position of `label`, as a one-element vector like `_PartitionLanes`', and
throws `ArgumentError` for a label with none or several. The plate's lowering
reads this recipe to run the partition as that one cell, at an `Int` row
(`_partition_single`), so its values are computed at one row of known count.
"""
struct _PartitionLane
    key::Symbol
end
@inline function (lane::_PartitionLane)(table::Dict{K,Vector{Int}}, label, ::Int) where {K}
    rows = get(table, label, _PARTITION_NO_ROWS)
    length(rows) == 1 || _partition_lane_error(lane.key, label, length(rows))
    rows
end
@noinline _partition_lane_error(key, label, n) = throw(ArgumentError(
    "plate_partition: the key :$key was declared to hold every label once (`single = true`), " *
    "but it holds $(repr(label)) $n times"))
_readable_callee(::_PartitionLane) = :partition_lane
_opname(::_PartitionLane) = "partition_lane"

# The operations giving a partition's lanes.
_partition_lanes_op(@nospecialize(op)) = op isa Union{_PartitionLanes,_PartitionLane}

# What a partition's rows check reads beside the domain and the rows: the name
# of the key's entry count in the emitted body and the key's name. `nothing`
# when the lanes are no recipe of this plan: a label bound at preparation folds
# them, which only RK's and PPL's own internal callers do, with keys they sized.
function _partition_cells_arguments(windows, recipe::Recipe)
    g = windows.pushdowns.graph
    lanes = canon_id(g, first(recipe.inputs).id)
    for id in producers_of(g, lanes)
        r = g.recipes[id]
        _partition_lanes_op(r.op) || continue
        cells = get(windows.names, canon_id(g, r.inputs[3].id), nothing)
        cells === nothing && return nothing
        return (cells, QuoteNode(r.op.key))
    end
    nothing
end

# Whether partition cell `recipe` runs one cell per label: its lanes are a
# `_PartitionLane` recipe of this graph. Lanes folded at preparation keep the
# rows form.
function _partition_single(windows, recipe::Recipe)
    windows === nothing && return false
    g = windows.pushdowns.graph
    lanes = canon_id(g, first(recipe.inputs).id)
    any(id -> g.recipes[id].op isa _PartitionLane, producers_of(g, lanes))
end

# The row of a partition of one cell per label, from its lanes
# (`_PartitionLane` checked their count), and its check against the plate's
# one-dimensional domain and the key's entry count.
@inline _partition_row(rows::Vector{Int}) = (length(rows) == 1 || _partition_row_error(rows);
                                             @inbounds rows[1])
@noinline _partition_row_error(rows) = throw(ArgumentError(
    "plate_partition: a partition of one cell per label has $(length(rows)) cells"))
@inline function _partition_check_rows(shape::Tuple, row::Int)
    length(shape) == 1 || throw(ArgumentError(
        "plate_partition: a partition runs over a one-dimensional plate domain; " *
        "this plate's domain has $(length(shape)) dimensions"))
    _cell_check_row(shape, row)
end
@inline function _partition_check_rows(shape::Tuple, row::Int, cells::Int, key::Symbol)
    length(shape) == 1 && length(first(shape)) != cells &&
        _partition_cells_error(cells, length(first(shape)), key)
    _partition_check_rows(shape, row)
end

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

_partition_by_entries(by::Union{NamedTuple,AbstractDict}) = pairs(by)
_partition_by_entries(by::Pair) = (by,)
_partition_by_entries(by) = applicable(iterate, by) ? by : (by,)

# The key of each name in `names`: `by` is one key for all of them or, per name,
# a named tuple, a dictionary, one `name => key` pair or a collection of such
# pairs. PPL's observation queries read their `by` here too: `what` is the word
# for a name in a refusal, and `refuse` builds its exception.
function _partition_by_map(by, names; what = "plate",
                           refuse = message -> ArgumentError("plate_partition: " * message))
    forms = "one key for every $what or, per $what, a named tuple, a dictionary, one " *
        "`$what => key` pair or a collection of such pairs"
    keys_by_name = Dict{Symbol,Symbol}()
    if by isa Symbol
        for name in names
            keys_by_name[name] = by
        end
    else
        for entry in _partition_by_entries(by)
            # By value: a `Dict{Symbol,Any}` or `Any[...]` holds names as well.
            entry isa Pair && first(entry) isa Union{Symbol,AbstractString} &&
                last(entry) isa Union{Symbol,AbstractString} ||
                throw(refuse("`by` must be $forms; got $(repr(by; context = :limit => true))"))
            name, key = Symbol(first(entry)), Symbol(last(entry))
            # Two keys are two different partitions of one name's cells, so
            # neither entry can silently win.
            get(keys_by_name, name, key) === key || throw(refuse(
                "`by` gives $what :$name two keys, :$(keys_by_name[name]) and :$key"))
            keys_by_name[name] = key
        end
    end
    for name in names
        haskey(keys_by_name, name) || throw(refuse(
            "`by` gives no key for $what :$name; it names " *
            "$(isempty(keys_by_name) ? "nothing" : join(repr.(sort!(collect(keys(keys_by_name)))), ", ")). " *
            "Give $forms"))
    end
    keys_by_name
end

"""
    plate_partition(spec::KernelSpec, plates; by, index = :partition, name = :partition_total,
                    locality = :error, partitioned = (), plan = nothing,
                    plan_constant = (), single = false) -> KernelSpec

A new kernel in which the value `name` is the sum of the cells of the authored
plates named in `plates` (one name or several) whose key equals the label held
by the HAVE port `index`: a *partition* of the plates, such as one subject's
observations. `by` gives each plate's key, a value of `spec` (usually a bound
data port) with one entry per cell of the plate in the plate's linear order
(another length is refused at evaluation, so no cell can fall outside every
partition): one name for every plate or, per plate, a named tuple, a dictionary,
one `plate => key` pair or a collection of such pairs.

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

`single = true` declares that every key holds each label of the partition
exactly once, so a partition is one cell of each named plate (one subject's
cell of a plate over subjects). The partition then runs that cell at one row,
as [`plate_cell`](@ref) does: the values it reads at rows are computed at that
row only, and values of one element per row can stay numbers instead of
one-element arrays. A label that a key holds once is unchanged; a key that
holds a label zero or several times throws `ArgumentError` at evaluation.

The default boundary of the result is the original HAVE ports plus `index`,
and the WANT `name`. `spec` is unchanged. `index` may name an existing HAVE
port of `spec` to partition several specs at the same label.
"""
function plate_partition(spec::KernelSpec, plates; by,
                         index::Symbol = :partition, name::Symbol = :partition_total,
                         locality::Union{Nothing,Symbol} = :error, partitioned = (),
                         plan::Union{Nothing,Symbol,Pair{Symbol,<:Type}} = nothing,
                         plan_constant::Tuple{Vararg{Symbol}} = (), single::Bool = false)
    locality in (nothing, :error, :warn, :ignore) || throw(ArgumentError(
        "plate_partition: locality must be :error, :warn, :ignore or nothing; got $(repr(locality))"))
    plan_port, plan_type = plan isa Pair ? (first(plan), last(plan)) : (plan, Any)
    plan_port === nothing || plan_port !== index || throw(ArgumentError(
        "plate_partition: the plan port must differ from the index port :$index"))
    names = plates isa Symbol ? (plates,) : Tuple(Symbol.(collect(plates)))
    isempty(names) && throw(ArgumentError("plate_partition: name at least one plate"))
    keys_by_plate = _partition_by_map(by, names)
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
        copy(graph.aliases), graph.version, nothing, copy(graph.checks))
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
    cell_counts = Value[]
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
        counted = value!(copied, Symbol(:_partition_cells_, plate), Int)
        _add_recipe!(copied, (key,), (counted,), _PartitionCells(), 0.0, nothing, false, nothing)
        push!(cell_counts, counted)
    end
    checked = value!(copied, Symbol(:_partition_label), Any)
    _add_recipe!(copied, (label, tables...), (checked,), _PartitionLabel(), 0.0, nothing,
                 false, nothing)
    totals = Value[]
    for (plate, recipe, table, counted) in zip(names, recipes, tables, cell_counts)
        lanes = value!(copied, Symbol(:_partition_rows_, plate), Vector{Int})
        _add_recipe!(copied, (table, checked, counted), (lanes,),
                     (single ? _PartitionLane : _PartitionLanes)(keys_by_plate[plate]),
                     0.0, nothing, false, nothing)
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
    const selection::Any
    # Whether the lowering being footprinted has begun: a lowering nested in
    # it (a plate body, an embedded kernel) runs blind to the recording
    # (`_lower_with_ops_collecting`).
    active::Bool
    # Nonempty-row guard of the exclusive region currently being emitted.
    guard::Any
end
_FootprintRecords(selection = nothing) =
    _FootprintRecords(Dict{ValueId,Vector{Tuple{Symbol,Any}}}(), selection, false, nothing)

# A packed-vector projection, owned by the prepared footprint. The sorted
# indices let ranges skip excluded blocks without visiting their elements.
struct _FootprintSelection
    mask::BitVector
    indices::Vector{Int}
end
_FootprintSelection(mask::AbstractVector{Bool}) =
    _FootprintSelection(BitVector(mask), findall(mask))

_footprint_recording() = get(task_local_storage(), _FOOTPRINT, nothing) !== nothing
function _footprint_record!(cid::ValueId, kind::Symbol, expr)
    recording = get(task_local_storage(), _FOOTPRINT, nothing)
    recording === nothing && return
    push!(get!(() -> Tuple{Symbol,Any}[], recording.records, cid),
        _footprint_guarded(recording.guard, kind, expr))
    nothing
end

_footprint_guarded(guard, kind, expr) = guard === nothing ? (kind, expr) :
    (:conditional, (guard, kind, expr))
# Keep the read expression itself lazy: its view or row metadata may only be
# bound inside the same nonempty region in the value lowering.
function _footprint_entry(kind, expr)
    kind === :conditional || return Expr(:tuple, QuoteNode(kind), expr)
    guard, inner_kind, inner_expr = expr
    Expr(:if, guard, _footprint_entry(inner_kind, inner_expr),
        Expr(:tuple, QuoteNode(:positions), Expr(:ref, Int)))
end

# A capture reads linear positions `v[q]` of a vector. Of any other array it may
# read `M[q, :]` or `M[q, 2]` instead, and the capture records no later indices:
# report the positions and those whole rows, which covers either read. Without
# the rows, `X[q, 2]` of a matrix port reported positions of its first column.
function _footprint_record_capture!(d, cid::ValueId, expr)
    _footprint_record!(cid, :gathered, expr)
    get(d.types, cid, Any) <: AbstractVector || _footprint_record!(cid, :gathered_rows, expr)
    nothing
end

# The return expression of a footprint body: each HAVE port's name, value and
# recorded reads, plus the reads of the recipes the window emitter never
# saw (computed whole: a bound-indexed gather or literal range view reads
# constant positions, anything else the whole port). `recipes` are the
# lowering's table recipes (`runtime_recipes`): a position read from bound
# data is then read from the constants table when the footprint runs, so
# the body of one program is the same for every binding.
function _footprint_return(p::Plan, cells, names, declared = nothing, body = nothing;
                           guards = Dict{Int,Any}(), recipes = ())
    recording = task_local_storage(_FOOTPRINT)
    g = p.graph
    cidof(v) = canon_id(g, v.id)
    pushed = cells === nothing ? Set{ValueId}() : cells.pushed
    producers = Dict{ValueId,Recipe}()
    for r in p.recipes, o in r.outputs
        producers[cidof(o)] = r
    end
    wants = Set(cidof(v) for v in p.want)
    selection = recording.selection
    selected_port = selection === nothing ? nothing :
        findfirst(port -> port.name == first(selection), p.have)
    selected_cid = selected_port === nothing ? nothing : cidof(p.have[selected_port])
    # A selection is itself one query's data in this body (`QuoteNode(selected)`
    # below), and its constant reads are projected onto it here, once, so a
    # local cell footprint never enumerates a large shared block per call.
    # Without one, bound positions stay runtime reads of the constants table.
    slots = IdDict{Recipe,Int}()
    if selection === nothing
        for (k, r) in enumerate(recipes)
            r.op isa _BoundConstant && (slots[r] = k)
        end
    end
    slot(r) = get(slots, r, nothing)
    reads = Dict(cid => copy(entries) for (cid, entries) in recording.records)
    guard = nothing
    record(cid, kind, expr) = push!(get!(() -> Tuple{Symbol,Any}[], reads, cid),
        _footprint_guarded(guard, kind, expr))
    # A view's reads are mapped at its binding: only a view whose value and
    # operands the body binds at its top level (not one a scan, a fused plate
    # or a branch arm absorbed) forwards its readers' positions.
    bound = body === nothing ? Set{Symbol}() :
        _footprint_bound_names(body, p, names; guards = values(guards))
    for cid in wants
        record(cid, :whole, nothing)
    end
    # Walk consumers before producers so an actual view forwards its readers'
    # positions to the parent. Constructing a view only reads shape/index
    # metadata; a copying gather retains the ordinary eager read below.
    for r in reverse(p.recipes)
        guard = get(guards, r.id, nothing)
        r.op isa _AuthoredPlateCellOp && continue
        any(o -> cidof(o) in pushed, r.outputs) && continue
        typed = declared === nothing ||
            any(o -> get(names, cidof(o), nothing) in declared, r.outputs)
        access = _footprint_access(r; typed, checked = true)
        view = access !== nothing && access.kind === :view ? access : nothing
        f = _cell_form(r)
        literal = f === nothing ? _footprint_literal_read(r) : nothing
        for (s, input) in enumerate(r.inputs)
            cid = cidof(input)
            if view !== nothing && s == view.slot && _footprint_indices_free(view) &&
                    all(v -> get(names, cidof(v), nothing) in bound,
                        (r.inputs..., only(r.outputs)))
                mapping = Dict{Symbol,Any}(view.params[j] => names[cidof(v)]
                                          for (j, v) in enumerate(r.inputs))
                # The view itself, before its typed local converts it.
                raw = _footprint_replace(view.native, mapping)
                mapping[view.params[s]] = Expr(:call, GlobalRef(Base, :LinearIndices), names[cid])
                indexmap = _footprint_replace(view.native, mapping)
                out = cidof(only(r.outputs))
                entries = Expr(:tuple, (_footprint_entry(k, e)
                    for (k, e) in get(reads, out, Tuple{Symbol,Any}[]))...)
                positions = view.declared_check === nothing ?
                    Expr(:call, GlobalRef(@__MODULE__, :_footprint_view_positions),
                        indexmap, names[out], entries) :
                    Expr(:call, GlobalRef(@__MODULE__, :_footprint_typed_view_positions),
                        raw, view.declared_check, indexmap, names[out], entries)
                record(cid, :gathered, positions)
            elseif literal !== nothing && first(literal) == s
                record(cid, :literal, last(literal))
            else
                constant = f === nothing ? nothing :
                    _footprint_constant(r, f, s, producers, cidof, slot)
                constant === nothing && (constant = _footprint_inline_positions(r, s))
                if constant !== nothing && cid == selected_cid
                    positions = Int[]
                    _footprint_selected_positions!(positions, last(selection), constant)
                    constant = positions
                end
                record(cid, constant === nothing ? :whole : :positions, constant)
            end
        end
    end
    entries = Any[]
    for port in p.have
        selection === nothing || port.name == first(selection) || continue
        selected = selection === nothing ? nothing : last(selection)
        cid = cidof(port)
        records = Any[_footprint_entry(kind, expr)
                      for (kind, expr) in get(reads, cid, Tuple{Symbol,Any}[])]
        push!(entries, Expr(:tuple, QuoteNode(port.name), names[cid], Expr(:tuple, records...),
                            QuoteNode(selected)))
    end
    Expr(:call, GlobalRef(@__MODULE__, :_footprint_collect), Expr(:tuple, entries...))
end

# The constant positions a recipe computed whole reads of its input `s`, when
# its form gathers that input by a bound index or views a literal range of it;
# `nothing` otherwise (the input is read whole). A bound index or endpoint
# (RKPPL's binding-supplied packed positions) is as fixed for one binding as a
# literal, but it is the binding's data: where `slot` gives its position in the
# constants table, the positions are an expression reading it there, so
# another binding of the program reuses the footprint's compiled body. Only
# the value's type decides the form. Without a slot (a selected footprint,
# whose body holds its selection anyway), they are this binding's positions.
function _footprint_constant(r::Recipe, f::_CellForm, s::Int, producers, cidof,
                             slot = _ -> nothing)
    bound(producer) = (k = slot(producer); k === nothing ? producer.op.value :
                                                           Expr(:ref, _CONSTANTS_ARG, k))
    role = f.roles[s]
    if role in (:gather, :linear_gather)
        indices = Any[]
        for (k, via) in zip(f.indices[s], f.vias[s])
            producer = get(producers, cidof(r.inputs[k]), nothing)
            producer !== nothing && producer.op isa _BoundConstant || return nothing
            producer.op.value isa AbstractArray{<:Integer} || return nothing
            role === :linear_gather || return nothing
            via == 0 || return nothing
            push!(indices, bound(producer))
        end
        all(index -> index isa AbstractArray, indices) &&
            return _footprint_bound_positions(indices...)
        return Expr(:call, GlobalRef(@__MODULE__, :_footprint_bound_positions), indices...)
    elseif role === :range_source && f.kind === :range_view
        parts = map(f.literal) do part
            part isa Int && return part
            producer = get(producers, cidof(r.inputs[part.input]), nothing)
            producer !== nothing && producer.op isa _BoundConstant &&
                producer.op.value isa Integer || return nothing
            value = bound(producer)
            value isa Integer ? Int(value) : Expr(:call, GlobalRef(Base, :Int), value)
        end
        any(isnothing, parts) && return nothing
        all(part -> part isa Int, parts) && return _cell_range(parts)
        return Expr(:call, GlobalRef(@__MODULE__, :_cell_range), Expr(:tuple, parts...))
    end
    nothing
end
# The positions gathers by bound integer indices read: each index's entries in
# linear order, one after another.
function _footprint_bound_positions(indices::AbstractArray{<:Integer}...)
    positions = Int[]
    for index in indices
        append!(positions, vec(index))
    end
    positions
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

# The positions a literal index selects (`3`, `2:4`, `8:3:14`), or `nothing`.
_footprint_literal_index(ex::Integer) = Int[ex]
_footprint_literal_index(ex::AbstractRange{<:Integer}) = collect(Int, ex)
function _footprint_literal_index(ex::Expr)
    ex.head === :call && ex.args[1] === :(:) && length(ex.args) in (3, 4) &&
        all(a -> a isa Integer, ex.args[2:end]) || return nothing
    collect(Int, (:)(ex.args[2:end]...))
end
_footprint_literal_index(ex) = nothing

# The positions a recipe computed whole reads of its input `s` when every read
# of that input, however deep in the expression, selects literal linear
# positions: `tanh.(view(u, 8:3:14))` (RKPPL's stacked LKJ edges), `f(u[3])`.
# The reader of such a view may read all of it, never more, so this is a
# structural read set; `nothing` when any read is bare, has another index, or
# sits under a branch, whose inactive arm may hold an invalid literal.
function _footprint_inline_positions(r::Recipe, s::Int)
    analyzed = _cell_native(r)
    analyzed === nothing && return nothing
    native, params, mod = analyzed
    source = params[s]
    positions = Int[]
    function walk(ex)
        ex isa Symbol && return ex !== source
        ex isa Expr || return true
        ex.head in _CELL_PLAIN_HEADS && !(ex.head in (:if, :&&, :||)) || return false
        target, index = if ex.head === :ref && length(ex.args) == 2
            ex.args[1], ex.args[2]
        elseif ex.head === :call && length(ex.args) == 3 &&
                !_native_callee_shadowed(ex.args[1], params) &&
                _cell_binding(ex.args[1], mod) in (Base.view, Base.getindex)
            ex.args[2], ex.args[3]
        else
            nothing, nothing
        end
        if target === source
            selected = _footprint_literal_index(index)
            selected === nothing && return false
            append!(positions, selected)
            return true
        end
        all(walk, ex.args)
    end
    walk(native) && !isempty(positions) ? positions : nothing
end

# ----------------------------------------------------------- shared prefix

# Whether a recipe's read of one mixed HAVE port is confined to shared
# positions. These are the same structural reads reported by the footprint;
# an unknown read stays in the residual unless every position is shared.
function _partition_shared_read_positions(r::Recipe, slot::Int, producers, cidof)
    form = _cell_form(r)
    positions = if form !== nothing
        _footprint_constant(r, form, slot, producers, cidof)
    else
        literal = _footprint_literal_read(r)
        literal !== nothing && first(literal) == slot && length(last(literal)) == 1 ?
            last(literal) : nothing
    end
    positions === nothing && (positions = _footprint_inline_positions(r, slot))
    positions
end
function _partition_shared_read(r::Recipe, slot::Int, labels, producers, cidof)
    positions = _partition_shared_read_positions(r, slot, producers, cidof)
    positions !== nothing &&
        all(i -> checkbounds(Bool, labels, i) && ismissing(labels[i]), positions)
end

# A prefix's fixed read set of one packed input. Unknown reads keep the whole
# input: a footprint evaluated at the exemplar could omit another arm's reads
# and would not be a valid active boundary or cache-invalidation key.
function _partition_prefix_positions(p::Plan, name::Symbol)
    cidof(v) = canon_id(p.graph, v.id)
    port = only(v for v in p.have if v.name === name)
    cid = cidof(port)
    any(v -> cidof(v) == cid, p.want) && return :whole
    positions = Int[]
    for r in p.recipes, (slot, input) in enumerate(r.inputs)
        cidof(input) == cid || continue
        read = _partition_shared_read_positions(r, slot, p.producer, cidof)
        read === nothing && return :whole
        append!(positions, read)
    end
    sort!(unique!(positions))
end

# The values a dense packed point keeps at rows, added to `row_values` (ids of
# `declared`). The declared packed input `AbstractVector{Float64}` cannot decide
# a type-selected branch (`_type_selected_branch`) whose condition reads a value
# computed from parameter views: an opaque helper reading an abstract view
# infers `Any`. A guard such as `eltype(cells) <: Real ? cells : throw(...)` then
# keeps no arm and no row form, and the split would lift its whole cone into the
# prefix, whose pullback every partition gradient repeats. Decide which values
# stay at rows with the dense stand-in each query is prepared with, but keep the
# declared plan, branch included: every argument type's preparation of the
# prefix or residual proves its own arm. Value ids are allocated globally, so a
# value either analysis creates is simply absent from the other.
function _partition_dense_rows!(row_values, lowered::Plan, declared::Plan)
    any(r -> r.op isa _KernelSourceOp && r.op.f isa _KernelBranch, declared.recipes) ||
        return row_values
    types = Any[valtype(v) for v in lowered.have]
    dense = Any[_cell_standin_type(T) for T in types]
    isequal(dense, types) && return row_values
    _, cells = _cell_outlined_pushdowns(lowered; argument_types = dense)
    cells === nothing && return row_values
    for cid in cells.pushed
        haskey(declared.graph.values, cid) && push!(row_values, canon_id(declared.graph, cid))
    end
    row_values
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
        lowered = _lowering_plan(p)
        p, cells = _cell_outlined_pushdowns(lowered)
        if cells !== nothing
            union!(row_values, cells.pushed)
            row_types = cells.types
            _partition_dense_rows!(row_values, lowered, p)
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
        # This includes views of shared parameter blocks: when every reader
        # needs selected rows, caching the whole view would turn a compact
        # read into a whole-block active frontier at the derivative boundary.
        any(v -> cidof(v) in row_values, r.outputs) && return false
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
    positions = map(entries) do (_, port, records, selection)
        selection === nothing ? _footprint_positions(port, records) :
            _footprint_selected(port, records, selection)
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

function _footprint_selected_positions!(out, selection, i::Integer)
    checkbounds(Bool, selection.mask, i) && selection.mask[i] && push!(out, i)
    out
end
function _footprint_selected_positions!(out, selection, read::AbstractRange{<:Integer})
    isempty(read) && return out
    lo, hi = minmax(first(read), last(read))
    indices = selection.indices
    for j in searchsortedfirst(indices, lo):searchsortedlast(indices, hi)
        i = indices[j]
        i in read && push!(out, i)
    end
    out
end
function _footprint_selected_positions!(out, selection, reads)
    for read in reads
        _footprint_selected_positions!(out, selection, read)
    end
    out
end
function _footprint_selected(port::AbstractVector, records::Tuple, selection::_FootprintSelection)
    length(port) == length(selection.mask) || throw(DimensionMismatch(
        "footprint selection must match the packed vector"))
    positions = Int[]
    for (kind, read) in records
        kind === :whole && return copy(selection.indices)
        if kind === :literal
            _footprint_selected_positions!(positions, selection, LinearIndices(port)[read...])
        elseif kind === :cell
            _footprint_selected_positions!(positions, selection,
                length(port) == 1 ? firstindex(port) : first(Tuple(read)))
        elseif kind in (:rows, :gathered_rows, :gathered, :positions)
            rows = kind === :rows && length(port) == 1 ? firstindex(port) : read
            _footprint_selected_positions!(positions, selection, rows)
        else
            throw(ArgumentError("unknown footprint read kind $kind"))
        end
    end
    unique!(sort!(positions))
end

const _FOOTPRINT_KERNELS = IdDict{Any,Any}()
# Per kernel prepared from exemplars, the footprint kernels of other argument
# types (`partition_footprint`), by the arguments' types.
const _FOOTPRINT_RETYPED = IdDict{Any,Dict{Any,Any}}()
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
positions, and a view the parent positions its readers read (constructing a
view reads none; a typed local that converts a view into a copy reads every
viewed position). A read nested in a larger expression, `tanh.(view(u, 2:3:8))`,
contributes its literal positions. A product over a bound sparse factor reads its
operand at the columns its rows store. A scalar generator sum with `init`
contributes its visible gathers, including through views, when its domains,
filters and indices do not read the captured array's values. Nested sums keep
runtime loops; empty domains contribute no positions. Opaque reads, lazy term
branches and other generator consumers retain conservative footprints.
Direct captures in one scalar expression also request only the needed rows of
eligible producers, including copying packed-parameter slices.
Retained branches use capture row windows when bound indices
prove both arms' requests in range; unproven requests may conservatively
contribute the whole capture. Arm arithmetic remains lazy.
For a partition or cell of a packed parameter vector this is its structural read
set, a conservative support for its gradient; numerical zeros and
cancellations do not remove positions. This also works on a `plate_cell`
kernel, with its cell index among `args`. For a planned partition, pass
`nothing` in the plan port: footprint lowering computes its own bookkeeping.

A kernel prepared from `exemplars` is typed for exactly their types, and so is
its [`partition_footprint_kernel`](@ref). Called with arguments of other types
(another label type, a view of the point), `partition_footprint` returns the
footprint of the same plan lowered from those arguments' types, which is what
a kernel prepared from them would read; it prepares that once per type.
"""
function partition_footprint(kernel::PreparedKernel, args...)
    footprint = lock(_FOOTPRINT_LOCK) do
        typed = get!(() -> partition_footprint_kernel(kernel), _FOOTPRINT_KERNELS, kernel)
        signature = _exact_signature(typed)
        signature === Nothing || args isa signature ||
            length(args) != fieldcount(signature) || return get!(
            () -> _partition_footprint_kernel(kernel, nothing; argument_types = map(typeof, args)),
            get!(() -> Dict{Any,Any}(), _FOOTPRINT_RETYPED, kernel), typeof(args))
        typed
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
partition_footprint_kernel(kernel::PreparedKernel) = _partition_footprint_kernel(kernel, nothing)
function _partition_footprint_kernel(kernel::PreparedKernel, selection; argument_types = nothing)
    # A plate several cells read is composed into them first: each composed
    # cell's reads are recorded at that cell's positions
    # (`_footprint_capture_program`), and their union is what computing the
    # plate once at the union of the rows reads. Composing here, before
    # lowering, also keeps a view the plate read from becoming a typed local
    # (`_owns_array_regions`), which a footprint must take as a whole read.
    p = _cell_compose_shared(_lowering_plan(kernel.plan))
    # Row demand uses the value kernel's type evidence and retains its call
    # guard, except for a plan port: footprints recompute that bookkeeping
    # and ignore the supplied plan (the documented placeholder is nothing).
    # `argument_types` replaces that evidence with other arguments' types.
    exemplar_types = argument_types === nothing ? _plan_port_exemplar_types(p, kernel, Any) :
        _plan_port_types(p, Any[argument_types...], Any)
    exact_signature = exemplar_types === nothing ? Nothing : Tuple{exemplar_types...}
    ast, constants, callables, recipes = _sharing_cell_plans() do
        _lower_with_ops(p; footprint = true, footprint_selection = selection, exemplar_types)
    end
    _prepare(p, ast, constants, _native_table(ast, callables), recipes; exact_signature)
end

# The argument types `kernel` was prepared from (its exact signature), with
# the type at every plan port replaced by `T`; `nothing` for a kernel
# prepared without exemplars. A prepared kernel's signature carries its plan
# type at the plan port, while its lowering saw the exemplar there
# (`nothing`, `_plan_typed_exemplars`).
function _plan_port_exemplar_types(p::Plan, kernel::PreparedKernel, T)
    signature = _exact_signature(kernel)
    signature === Nothing && return nothing
    _plan_port_types(p, Any[fieldtypes(signature)...], T)
end
# `types`, one per HAVE port of `p`, with the type at every plan port replaced by `T`.
function _plan_port_types(p::Plan, types::Vector{Any}, T)
    for r in p.recipes
        r.op isa _AuthoredPlateCellOp && r.op.plan !== nothing || continue
        position = findfirst(v -> v.name === r.op.plan, p.have)
        position === nothing || (types[position] = T)
    end
    types
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
    r !== nothing && _partition_lanes_op(r.op) || return nothing
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

# A label's plan is one named tuple of every row-bookkeeping value of the
# partition: thousands of fields for a large model (4,561 for the frozen
# Atlas's dense partition). Passed by value into a differentiated call it is an
# LLVM aggregate, and LLVM's sparse conditional constant propagation visits every
# field on each insertion: that pass's share of the first gradient's compile time
# grew about cubically with the plan's width (snag `frozen-atlas-fir-2d4269a6`). A
# differentiated call therefore receives the plan behind this reference
# (`_ad_box_plan`, ad.jl), and the body reads either form through
# `_partition_plan_value`. This is a compile-time measure: a warm gradient takes
# about as long either way. Callers still pass, and `partition_plan` still
# returns, the named tuple.
mutable struct _PartitionPlanRef{T}
    const plan::T
end
@inline _partition_plan_value(plan) = plan
@inline _partition_plan_value(plan::_PartitionPlanRef) = getfield(plan, :plan)

# A planned body is one function to LLVM and Enzyme unless something keeps its
# statements apart, and their slowest passes are superlinear per function: the
# frozen Atlas's first gradient spent all of about 5 h in Enzyme on one
# `CreatePrimalAndGradient` (snag `frozen-atlas-fir-2d4269a6`). So a statement of the
# body with an array operand calls through this `@noinline` wrapper, which compiles
# once per signature and which LLVM cannot inline back, as a call-site annotation
# allows it to. Identical operations then share one compiled function. First-gradient
# CPU on a synthetic partition went from 216 to 67 s at 64 blocks and from 899 to 199 s
# at 128. Warm gradients were about 15% slower at 8 blocks and no slower at 32 or more.
# Scalar statements stay inline (`_partition_statement_call`), and so do plate loops:
# moving them into separate functions in one module made compile time worse.
#
# One positional method per operand count, each operand's type a parameter of its
# own (`_PlateCall` has the same methods). A `Vararg{Any}` method would lose two
# things. It takes its operands as one tuple, and Enzyme cannot prove a structure
# stored there constant, even one passed to it as `Const`: a derivative rule received
# its bound data record `Duplicated`, with a fresh zero shadow of each floating-point
# field on every call, and refused it or could not accumulate its cotangent (snag
# `derivativerule-r-0d7e245c`). And Julia compiles a `Vararg{Any}` position that holds
# a type as `::Type`, and one that holds a function the method does not call as
# `::Function`, so the wrapper read the type's parameter and dispatched the statement
# at run time: `_plate_similar_output(marker, T, axes)` called `similar` dynamically
# and allocated on every evaluation (snag `partition-array-bb24ae62`).
for count in 0:_PLATE_CALL_MAX_ARGUMENTS
    local arguments = Any[Symbol("argument", index) for index in 1:count]
    local types = Any[Symbol("Argument", index) for index in 1:count]
    local typed = Any[:($a::$T) for (a, T) in zip(arguments, types)]
    @eval @noinline _partition_array_call(f::F, $(typed...)) where {F,$(types...)} =
        f($(arguments...))
end
@inline @generated function _partition_statement_call(f, args::Vararg{Any,N}) where {N}
    array = any(T -> T <: Union{AbstractArray,Base.Broadcast.Broadcasted}, args)
    # Past the positional methods a call would need one argument tuple again.
    (array && N <= _PLATE_CALL_MAX_ARGUMENTS) || return :(f(args...))
    :(_partition_array_call(f, args...))
end
_partition_has_loop(@nospecialize(x)) =
    x isa Expr && (x.head in (:for, :while) || any(_partition_has_loop, x.args))
# A lazy view only packages existing storage. Keep that construction visible
# to its readers: an outlined mixed base/active overlay can require a shadow
# of its whole inactive base at the return boundary, before any row is read.
_partition_keeps_inline(@nospecialize(f)) = f === :tuple || f === :getfield ||
    f === :(===) || f === :typeof || (f isa GlobalRef && f.mod === Core) ||
    (f isa GlobalRef && isdefined(f.mod, f.name) && _cell_lazy_view(getglobal(f.mod, f.name))) ||
    (f isa Expr && f.head === :. && f.args[1] === :Core)
function _partition_separate_call(@nospecialize(x))
    x isa Expr || return x
    x.head === :let && return Expr(:let, x.args[1], _partition_separate_call(x.args[2]))
    x.head === :block && return Expr(:block, map(_partition_separate_call, x.args)...)
    statement = GlobalRef(@__MODULE__, :_partition_statement_call)
    if x.head === :ref && !any(a -> a === :end, x.args)
        return Expr(:call, statement, GlobalRef(Base, :getindex), x.args...)
    elseif x.head === :call && !_partition_keeps_inline(x.args[1]) &&
           !any(a -> a isa Expr && a.head in (:parameters, :kw, :..., :(=)), x.args[2:end])
        return Expr(:call, statement, x.args...)
    end
    x
end
function _partition_separate_statement(@nospecialize(s))
    s isa Expr && !_partition_has_loop(s) || return s
    s.head === :(=) && return Expr(:(=), s.args[1], _partition_separate_call(s.args[2]))
    s.head in (:call, :let, :block) ? _partition_separate_call(s) : s
end

# The wrapper above buys compile time on wide bodies and costs warm time on
# small ones, where each wrapped statement is its own differentiated call with
# its own tape: first-gradient CPU was unchanged at 8 synthetic blocks while the
# warm gradient got 15% slower. Leaving bodies below this size whole, together
# with one plan reference per label (`_ad_plan_argument`), took a one-subject
# gradient from 1.64-1.87 to 1.24-1.47 µs on a 10-row PK cell (45 wrapped
# statements) and from 7.6-8.3 to 6.0-6.5 µs on a joint cell (83), first
# gradients unchanged (snag `partition-gradie-38d830a4`). So a body is separated
# only when it has at least this many wrapped statements: a count of the
# program's statements, never of its data.
const _PARTITION_SEPARATE_MIN = 128
function _partition_separate_body(main::Vector; minimum::Int = _PARTITION_SEPARATE_MIN)
    separated = map(_partition_separate_statement, main)
    sum(_partition_wrapped_count, separated; init = 0) >= minimum ? separated : main
end
function _partition_wrapped_count(@nospecialize(x))
    x isa Expr || return 0
    own = x.head === :call && x.args[1] == GlobalRef(@__MODULE__, :_partition_statement_call)
    Int(own) + sum(_partition_wrapped_count, x.args; init = 0)
end
# `_PLAN_MODE` / `_plan_mode()` (graphops.jl): set while `partition_plan_kernel(spec; …)`
# prepares; the lowering then emits the plan body instead of the partition's.

# The operations a plan may apply: a value computed from bound data and the
# label alone is the same on every call, so a pure operation over such
# arguments (a source, a plain function, a partition op, a bound value of any
# kind: an index column, a data column, a design matrix) is computed once per
# label. Plates, scans and cells stay in the body.
_plan_eligible_op(op) = op isa Union{_PartitionTable,_PartitionCells,_PartitionLabel,
                                     _PartitionLanes,_PartitionLane,_BoundConstant,_KernelSourceOp,
                                     _TypeOperation} ||
                        op isa Function
# Whether a bound value is index bookkeeping (as against data); informational.
_plan_bookkeeping(::Integer) = true
_plan_bookkeeping(v::AbstractArray) = _plan_bookkeeping_type(eltype(v))
_plan_bookkeeping(v::AbstractDict) = _plan_bookkeeping_type(Base.valtype(v))
_plan_bookkeeping(v::Tuple) = all(_plan_bookkeeping, v)
_plan_bookkeeping(@nospecialize(_)) = false

# A planned row operand is fixed for the label. Store just those rows of a
# dense input, retaining the view's concrete type and axes. Keeping a view of
# the whole panel here makes every local call carry the whole parent across
# its outlined array operations; ordinary reverse may then zero that parent
# even though the call reads only this label's rows. This is plan storage,
# shared by primal and derivative execution, and costs nothing per evaluation.
_cell_plan_rows(arg, positions) = _cell_rows_view(arg, positions)
function _cell_plan_rows(arg::Array, positions::Vector{Int})
    selected = copy(_cell_rows_view(arg, positions))
    _cell_rows_view(selected, collect(axes(selected, 1)))
end
function _cell_plan_rows(arg::Array, position::Integer)
    selected = copy(_cell_rows_view(arg, position))
    _cell_rows_view(selected, 1)
end

function _plan_row_storage(st)
    Meta.isexpr(st, :(=), 2) || return st
    rhs = st.args[2]
    Meta.isexpr(rhs, :call) &&
        rhs.args[1] === GlobalRef(@__MODULE__, :_cell_rows_view) || return st
    Expr(:(=), st.args[1], Expr(:call, GlobalRef(@__MODULE__, :_cell_plan_rows),
                              rhs.args[2:end]...))
end
# An element type of index bookkeeping: integers, or arrays, ranges and
# tuples of them, nested (a ragged row structure is a vector of row vectors).
function _plan_bookkeeping_type(@nospecialize(T))
    T <: Integer && return true
    T <: AbstractArray && return _plan_bookkeeping_type(eltype(T))
    T <: Tuple && T isa DataType && return all(_plan_bookkeeping_type, T.parameters)
    false
end

# Source recipes bind their arguments lexically before evaluating the call.
# A let can join the plan when its bindings and body read only plan inputs.
function _plan_let_eligible(ex, data, ops)
    length(ex.args) == 2 || return false
    bindings = Meta.isexpr(ex.args[1], :block) ? ex.args[1].args : Any[ex.args[1]]
    scoped = copy(data)
    for binding in bindings
        binding isa LineNumberNode && continue
        Meta.isexpr(binding, :(=), 2) && binding.args[1] isa Symbol || return false
        _plan_eligible(binding.args[2], scoped, ops) || return false
        delete!(scoped, binding.args[1])
    end
    body = ex.args[2]
    statements = Meta.isexpr(body, :block) ? body.args : Any[body]
    all(st -> st isa LineNumberNode || _plan_eligible(st, scoped, ops), statements)
end

# Whether `ex` reads nothing that varies per call: no data symbol (an unbound
# HAVE port other than the label, or a value computed from one) and no
# operation other than a plan-eligible pure operation. Bound data and
# explicitly label-constant ports may be read by the plan.
function _plan_eligible(ex, data::Set{Symbol}, ops)
    ex isa Symbol && return !(ex in data)
    ex isa Expr || return true
    ex.head === :let && return _plan_let_eligible(ex, data, ops)
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
    :_cell_sparse_shape,
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

# References that may supply the storage returned by an expression. Fresh
# owned allocations cannot alias their markers or dimensions. A conditional
# can return either arm, but not its condition; writes made while evaluating
# that condition are found independently by `_written_plan_symbols`.
# Unknown calls remain conservative, even ones named like an owned allocator.
function _plan_storage_alias_refs!(refs::Set{Symbol}, ex)
    ex isa Symbol && (push!(refs, ex); return refs)
    ex isa Expr || return refs
    if ex.head === :call && ex.args[1] in (
            GlobalRef(@__MODULE__, :_plate_similar_output),
            GlobalRef(@__MODULE__, :_scan_similar_output))
        return refs
    elseif ex.head === :if
        foreach(a -> _plan_storage_alias_refs!(refs, a), ex.args[2:end])
    else
        foreach(a -> _plan_storage_alias_refs!(refs, a), ex.args)
    end
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
            for ref in intersect(_plan_storage_alias_refs!(Set{Symbol}(), value), names)
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
# Rename plan bindings without changing a let's lexical scope or field labels.
function _plan_rename(ex, names)
    ex isa Symbol && return get(names, ex, ex)
    ex isa Expr || return ex
    ex.head in (:quote, :inert) && return ex
    if ex.head === :let
        bindings = Meta.isexpr(ex.args[1], :block) ? ex.args[1].args : Any[ex.args[1]]
        scoped = copy(names)
        renamed = map(bindings) do binding
            binding isa LineNumberNode && return binding
            lhs, rhs = binding.args
            value = _plan_rename(rhs, scoped)
            delete!(scoped, lhs)
            Expr(:(=), lhs, value)
        end
        return Expr(:let, Expr(:block, renamed...), _plan_rename(ex.args[2], scoped))
    elseif ex.head === :kw
        return Expr(:kw, ex.args[1], _plan_rename(ex.args[2], names))
    elseif ex.head === :. && ex.args[2] isa Union{Symbol,QuoteNode}
        return Expr(:., _plan_rename(ex.args[1], names), ex.args[2])
    end
    Expr(ex.head, map(arg -> _plan_rename(arg, names), ex.args)...)
end

# Keep branch-local plan values behind the same label-only guard. A 0/1-element
# vector retains the inferred tuple element type across branch occupancy.
# The main guard stays in place.
function _partition_guarded_plan(st, data, ops, forced)
    Meta.isexpr(st, :if) && _plan_eligible(st.args[1], data, ops) || return nothing
    cond = st.args[1]
    arms = Any[st.args[2:end]...]
    planned = Any[]
    for k in eachindex(arms)
        arm = arms[k]
        arm isa Expr && arm.head === :block || continue
        # Nested cached comprehensions lose concrete element inference on an
        # empty outer branch. Leave nested branches in the original main body.
        split = _partition_plan_classify(arm, data, ops, forced;
                                         allow_shapes = false, allow_guards = false)
        split === nothing && continue
        plan, main, needed, _ = split
        cache = gensym(:guarded_plan)
        count = Expr(:if, cond, k == 1 ? 1 : 0, k == 1 ? 0 : 1)
        domain = Expr(:call, GlobalRef(Base, :OneTo), count)
        # Give the comprehension's bindings their own identities. Otherwise
        # outer dependency scanning treats them as reads of the original
        # branch locals, and Julia captures those locals in boxes.
        locals = Set{Symbol}()
        for statement in plan
            Meta.isexpr(statement, :(=), 2) &&
                _assigned_names!(locals, Expr(:(=), statement.args[1], nothing))
        end
        renamed = Dict(name => gensym(name) for name in locals)
        element = _plan_rename(Expr(:block, plan..., Expr(:tuple, needed...)), renamed)
        collect_plan = Expr(:comprehension, Expr(:generator, element,
            Expr(:(=), gensym(:guarded_plan_index), domain)))
        push!(planned, Expr(:(=), cache, collect_plan))
        held = gensym(:guarded_plan_values)
        declared = Set(needed)
        declarations = Any[x for x in main if _declares_any(x, declared)]
        filter!(x -> !_declares_any(x, declared), main)
        reads = Any[Expr(:(=), held, Expr(:call, GlobalRef(Base, :only), cache))]
        append!(reads, [Expr(:(=), name, Expr(:call, GlobalRef(Core, :getfield), held, j))
                       for (j, name) in enumerate(needed)])
        arms[k] = Expr(:block, declarations..., reads..., main...)
    end
    isempty(planned) ? nothing : (planned, Expr(:if, cond, arms...))
end
function _partition_plan_classify(body::Expr, data_names, ops, forced::Set{Symbol};
                                  allow_shapes::Bool = true, allow_guards::Bool = true)
    # Every symbol the body assigns anywhere is data until a plan statement
    # assigns it, so a reference ahead of a later or nested assignment can
    # never leave the body; the hidden cache arguments are data too.
    data = union!(_assigned_names!(Set{Symbol}(), body), data_names,
                  (_CACHES_ARG, _CACHE_APPLY_ARG))
    kinds = Symbol[]
    statements = Any[]
    for original in body.args
        st = original
        guarded = allow_guards ? _partition_guarded_plan(st, data, ops, forced) : nothing
        if guarded !== nothing
            planned, st = guarded
            append!(statements, planned)
            append!(kinds, fill(:plan, length(planned)))
            for assignment in planned
                delete!(data, assignment.args[1])
            end
        end
        kind = if st isa Expr && st.head === :(=) && st.args[1] isa Symbol &&
                  (_plan_type_local(st.args[1]) || st.args[1] in forced)
            :main
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol &&
                  _plan_eligible(st.args[2], data, ops)
            delete!(data, st.args[1])
            :plan
        elseif st isa Expr && st.head === :(=) && st.args[1] isa Symbol &&
               allow_shapes && _plan_shape_statement(st.args[2], data, ops)
            delete!(data, st.args[1])
            :shape
        elseif st isa Expr && st.head === :call && _plan_eligible(st, data, ops)
            :plan
        else
            :main
        end
        push!(kinds, kind)
        push!(statements, st)
    end
    body = Expr(:block, statements...)
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
    plan = Any[kinds[k] === :plan ? _plan_row_storage(st) : st
               for (k, st) in enumerate(body.args) if keep[k] && !(st isa LineNumberNode)]
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

# The statements of a footprint body that its result depends on. The body is
# the value lowering with every HAVE read recorded, and it returns those
# reads instead of the wanted values (`_footprint_return`). The wanted values'
# own work (a density plate's loop, numeric cell rows) therefore reaches the
# result only if a recorded read names it, yet Julia keeps and compiles it,
# because allocation and loops are not removable: for the frozen Atlas the
# footprint's native compile was half of preparing its partition sampler
# (snag `frozen-atlas-fir-2d4269a6`). Keep, in the body's order: the return;
# every statement assigning or writing a value a kept statement reads,
# including writes through an alias (`_written_plan_symbols`, as the plan
# split keeps them); and every statement that assigns no value of the body,
# such as a row check, which may throw. A plain assignment nothing kept reads
# is dropped, together with any check it held only for its own value.
function _footprint_live_statements(body::Expr, ops)
    statements = body.args
    # The body's own values: every top-level assignment target (plain, tuple
    # or typed) and declared local. Names assigned only inside a statement
    # (a `let`'s or a loop's) are its own.
    outer = Set{Symbol}()
    for st in statements
        Meta.isexpr(st, :(=)) &&
            _assigned_names!(outer, Expr(:(=), st.args[1], nothing))
        Meta.isexpr(st, :local) && _assigned_names!(outer, st)
    end
    wanted = Set{Symbol}()
    keep = falses(length(statements))
    for k in reverse(eachindex(statements))
        st = statements[k]
        st isa LineNumberNode && continue
        defines = _assigned_names!(Set{Symbol}(), st)
        writes = _written_plan_symbols((st,), outer, ops)
        if Meta.isexpr(st, :return) ||
           any(in(wanted), defines) || any(in(wanted), writes) ||
           (isempty(intersect(defines, outer)) && isempty(writes))
            keep[k] = true
            _symbol_refs!(wanted, st)
        end
    end
    Expr(body.head, (st for (k, st) in enumerate(statements)
                     if keep[k] || st isa LineNumberNode)...)
end

# Planning and footprint queries reuse the existing row-bookkeeping helpers.
# Inlining their vector loops and bounds checks at every read makes a wide
# query's native body much larger than its source. Keep those calls separate
# here; shape helpers still fold normally, and value/AD bodies are unchanged.
const _PARTITION_METADATA_HELPERS = Set{Symbol}([
    :_cell_slice_positions, :_cell_rows_view, :_cell_plan_rows, :_cell_canonical_rows,
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
    reads = Any[Expr(:(=), _PLAN_ARG, Expr(:(::), Expr(:call,
        GlobalRef(@__MODULE__, :_partition_plan_value), nm(p.have[position])), T))]
    push!(reads, Expr(:call, GlobalRef(@__MODULE__, :_partition_plan_labels_check),
        Expr(:call, GlobalRef(Core, :getfield), _PLAN_ARG, QuoteNode(:partition_labels)),
        Expr(:tuple, (nm(r.inputs[1]) for r in checks)...)))
    for (name, field) in zip(needed, fields)
        push!(reads, Expr(:(=), name, Expr(:call, GlobalRef(Core, :getfield), _PLAN_ARG,
                                             QuoteNode(field))))
    end
    Expr(:block, reads..., _partition_separate_body(main)...)
end

_partition_plan_label(label, tables::Tuple) =
    (tables, map(table -> get(table, label, _PARTITION_NO_ROWS), tables))
# Both comparisons are unrolled over their tuples, one `getfield` per table:
# a plan holds one table per plate or factor, of several table types, and
# neither `map` nor tail recursion over a long tuple of mixed types is
# inferred, so the check dispatched and allocated on every call (5 to 19 µs
# for 40 tables).
@generated function _partition_rows_match(tables::Tuple, rows::Tuple, label)
    n = fieldcount(tables)
    fieldcount(rows) == n || return :(false)
    n == 0 && return :(true)
    foldr((a, b) -> :($a && $b), [:(get(getfield(tables, $i), label,
        _PARTITION_NO_ROWS) === getfield(rows, $i)) for i in 1:n])
end
@inline _partition_plan_label_matches(plan, label) =
    _partition_rows_match(plan[1], plan[2], label)
@generated function _partition_plans_match(planned::Tuple, current::Tuple)
    n = fieldcount(planned)
    fieldcount(current) == n || return :(false)
    n == 0 && return :(true)
    foldr((a, b) -> :($a && $b), [:(_partition_plan_label_matches(getfield(planned, $i),
        getfield(current, $i))) for i in 1:n])
end
@inline function _partition_plan_labels_check(planned, current)
    _partition_plans_match(planned, current) || throw(ArgumentError(
        "partition plan: labels $(repr(current)) differ from the plan's labels; " *
        "compute the plan for these labels (`partition_plan`)"))
    nothing
end
# The check returns `nothing` and reads only integer tables: nothing about it
# is differentiable, so Enzyme treats the call as constant. Differentiated, the
# check over a long plan of mixed table types ran through Enzyme's
# runtime-dispatch rules on every gradient (2% of a partitioned SVI update) or
# failed type analysis outright. The check itself still runs and throws.
EnzymeCore.EnzymeRules.inactive(::typeof(_partition_plan_labels_check), args...) = nothing

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
A kernel prepared from exemplars has its plan lowered from the same argument
types, with `nothing` at the plan port as `prepare` documents, so the plan
has the type `prepare` gave `kernel`'s plan port.
"""
function partition_plan_kernel(kernel::PreparedKernel)
    p = _lowering_plan(kernel.plan)
    # Lower the plan from the type evidence the kernel's own lowering had:
    # its exemplars, with `nothing` at the plan port. Row demand, scalar row
    # plates and partition-sum regions depend on those types, and a plan
    # lowered without them can carry other fields than the plan type the
    # kernel reads (snag `brm-joint-cell-s-2d923270`). The plan kernel keeps
    # no call guard, as before: one plan kernel serves labels of mixed types.
    exemplar_types = _plan_port_exemplar_types(p, kernel, Nothing)
    ast, constants, callables, recipes = _sharing_cell_plans() do
        _lower_with_ops(p; partition_plan = true, exemplar_types)
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

# A caller that keeps one plan per label (RKPPL's partition queries and
# samplers, which keep up to two plan sets per label) keeps many equal small
# arrays: the rows of a label read by several statements, the same rows and
# gathered data in the plan of a residual and of its conditional kernel. On a
# one-unit-per-label toy, the 4 plans of one label held 82 such arrays and 11
# distinct ones (snag `per-unit-partiti-9d46c6de`). A plan is read-only label
# bookkeeping (`_partition_plan_split` keeps every value the body writes, and its
# aliases, in the body), so equal arrays can be one object. A pool is used for
# the plans of one preparation and then dropped: it keeps nothing alive itself.
struct _PartitionPlanPool
    # An array already seen => its canonical array, so an array every plan
    # shares (a whole bound column) is compared by content once.
    seen::IdDict{Any,Any}
    # (type, size, array) => the canonical array of that content.
    contents::Dict{Any,Any}
end
_PartitionPlanPool() = _PartitionPlanPool(IdDict{Any,Any}(), Dict{Any,Any}())

"""
    _intern_partition_plan(pool::_PartitionPlanPool, plan) -> plan

`plan` (a `partition_plan` result) with each array of plain bits that holds the
same content as one already in `pool` replaced by that array, descending
through tuples, named tuples, views and broadcast extrusions. The result has
the type of `plan`, equal values and the same reads.
"""
function _intern_partition_plan(pool::_PartitionPlanPool, plan::T) where {T}
    # `partition_labels` holds the label's rows of each partition table, which
    # the body compares with the table's own rows by identity
    # (`_partition_plan_label_matches`). Keep them, and make them the arrays
    # the other fields share.
    labels = plan isa NamedTuple && haskey(plan, :partition_labels) ? plan.partition_labels : nothing
    labels === nothing || _plan_register!(pool, labels)
    interned = labels === nothing ? _plan_intern(pool, plan) :
        merge(_plan_intern(pool, Base.structdiff(plan, NamedTuple{(:partition_labels,)})),
            (; partition_labels = labels))
    interned isa T || throw(ArgumentError(
        "internal: interning a partition plan changed its type to $(typeof(interned))"))
    interned
end
# Record the arrays of `x` as canonical where no array of their content is yet,
# without replacing any of them.
_plan_register!(::_PartitionPlanPool, x) = nothing
_plan_register!(pool::_PartitionPlanPool, x::Union{Tuple,NamedTuple}) =
    foreach(v -> _plan_register!(pool, v), x)
function _plan_register!(pool::_PartitionPlanPool, x::Array{T}) where {T}
    isbitstype(T) || return nothing
    haskey(pool.seen, x) && return nothing
    pool.seen[x] = x
    get!(pool.contents, (typeof(x), size(x), x), x)
    nothing
end
_plan_intern(::_PartitionPlanPool, x) = x
_plan_intern(pool::_PartitionPlanPool, x::Tuple) = map(v -> _plan_intern(pool, v), x)
_plan_intern(pool::_PartitionPlanPool, x::NamedTuple{K}) where {K} =
    NamedTuple{K}(_plan_intern(pool, Tuple(x)))
function _plan_intern(pool::_PartitionPlanPool, x::Array{T}) where {T}
    isbitstype(T) || return x
    get!(pool.seen, x) do
        get!(pool.contents, (typeof(x), size(x), x), x)
    end::typeof(x)
end
function _plan_intern(pool::_PartitionPlanPool, x::SubArray{T,N,P,I,L}) where {T,N,P,I,L}
    parent = _plan_intern(pool, Base.parent(x))
    indices = _plan_intern(pool, x.indices)
    SubArray{T,N,P,I,L}(parent, indices, x.offset1, x.stride1)
end
_plan_intern(pool::_PartitionPlanPool, x::Base.Broadcast.Extruded) =
    typeof(x)(_plan_intern(pool, x.x), x.keeps, x.defaults)
