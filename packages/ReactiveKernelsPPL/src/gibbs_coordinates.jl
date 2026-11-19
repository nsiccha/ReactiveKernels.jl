# Conditionals of one packed coordinate. Changing one coefficient `w[m]` moves
# a product `A * w` by `(w[m] - previous) .* A[:, m]`, so the conditional reads
# each such product from a cached value instead of recomputing it. The planner
# applies no algebraic identities; this file states that one for the emitted
# program. Everything downstream of the product is the emitted program.
#
# A scalar conditional follows reads of its packed slot to the density terms.
# A coordinate of a vector parameter also needs only the density terms that
# read that vector. Of a sum over a plate that reads the vector it needs only
# the cells that read the moved element: the element's own cell when the plate
# runs over the vector's cells, and the rows holding the element's index when
# the plate reads the vector through a gather by a bound index vector.

"Prepared coordinate conditionals returned by [`prepare_coordinate_conditional`](@ref)."
struct CoordinateConditional{L,J,C,U,R,B<:Tuple,S<:Tuple}
    layout::L
    joint::J
    conditional::C
    update::U
    refresh::R
    products::Vector{Symbol}
    blocks::B
    scalars::S
end

# The conditional of the coordinates in `range`: the terms that read them,
# with each plate in `cells` read at the moved coordinate's cell only, and each
# plate in `parts` at the rows whose gathered index is the moved coordinate's.
struct CoordinateBlock{K,C<:Tuple{Vararg{Symbol}},P<:Tuple{Vararg{Symbol}}}
    range::UnitRange{Int}
    kernel::K
    cells::C
    parts::P
end

_is_packed_view(f) = f === Base.view || f === :view || f == :(Base.view) ||
    (f isa GlobalRef && f.mod === Base && f.name === :view)

# `(first, last)` of `view(unconstrained, first:last)`, as emitted: runtime
# values of the binding, never spliced numbers.
function _packed_range(ex)
    Meta.isexpr(ex, :call, 3) && _is_packed_view(ex.args[1]) &&
        ex.args[2] === :unconstrained || return nothing
    range = ex.args[3]
    Meta.isexpr(range, :call, 3) && range.args[1] === :(:) || return nothing
    return (range.args[2], range.args[3])
end

_is_plate(ex) = Meta.isexpr(ex, :do, 2) && Meta.isexpr(ex.args[1], :call) &&
    ex.args[1].args[1] === :plate
_reads_packed(ex) = :unconstrained in _statement_value_reads!(Set{Symbol}(), ex)

# Literal positions `unconstrained[i]` read by `ex`; false when it reads the
# packed coordinates in any other way.
function _packed_scalar_reads!(positions, ex)
    ex === :unconstrained && return false
    ex isa Expr || return true
    if Meta.isexpr(ex, :ref, 2) && ex.args[1] === :unconstrained && ex.args[2] isa Integer
        push!(positions, Int(ex.args[2]))
        return true
    end
    return all(arg -> _packed_scalar_reads!(positions, arg), ex.args)
end

# How the statements read the packed coordinates directly: through a whole
# range (a view, or a plate over a view) or at literal positions. Each position
# records the names assigned from it. Other reads set `recognized` to false.
function _packed_readers(statements)
    ranges = Pair{Tuple{Any,Any},Vector{Symbol}}[]
    views, plates = Dict{Symbol,Tuple{Any,Any}}(), Set{Symbol}()
    positions = Dict{Int,Set{Symbol}}()
    recognized = true
    for st in statements
        Meta.isexpr(st, :(=), 2) && _reads_packed(st.args[2]) || continue
        name, rhs = _assigned_name(st), st.args[2]
        range = _packed_range(rhs)
        if name !== nothing && range !== nothing
            views[name] = range
        elseif name !== nothing && _is_plate(rhs)
            arguments = rhs.args[1].args[2:end]
            range = isempty(arguments) ? nothing : _packed_range(first(arguments))
            if range !== nothing && !any(_reads_packed, arguments[2:end]) &&
                    !_reads_packed(rhs.args[2])
                push!(plates, name)
            else
                range = nothing
            end
        end
        if range === nothing
            indices = Int[]
            if _packed_scalar_reads!(indices, rhs)
                names = _assigned_names(st)
                for k in indices
                    union!(get!(Set{Symbol}, positions, k), names)
                end
            else
                recognized = false
            end
            continue
        end
        at = findfirst(entry -> entry.first == range, ranges)
        at === nothing ? push!(ranges, range => Symbol[name]) : push!(ranges[at].second, name)
    end
    return (; ranges, views, plates, positions, recognized)
end

# Follow a scalar through every statement reading its packed slot. A term can
# also read the slot directly (for example the Jacobian of a positive scale).
function _coordinate_scalars(statements, selected, readers, spans, layout)
    scalars = NamedTuple[]
    readers.recognized && !isempty(readers.positions) || return scalars
    direct = map(selected) do term
        indices = Int[]
        _packed_scalar_reads!(indices, term)
        indices
    end
    for (i, entry) in enumerate(layout.entries)
        entry.kind === :sampled || continue
        k = entry.offset
        roots = get(readers.positions, k, nothing)
        roots === nothing && continue
        # Include every view or plate over a range containing this slot, even
        # when overlapping reads made that range ineligible for a vector cut.
        reached = copy(roots)
        for (span, range) in zip(spans, readers.ranges)
            k in span && union!(reached, range.second)
        end
        reached = _gibbs_dependents(statements, reached)
        terms = [t for (t, indices) in zip(selected, direct) if
            k in indices || _gibbs_reads(t, reached)]
        length(terms) == length(selected) && continue
        push!(scalars, (; k, name=Symbol(:_ppl_cg_scalar_, i), terms))
    end
    return scalars
end

# Statements `name = A * w` where `w` is a packed view and `A` is constant in
# the coordinates. Any other product stays as emitted and is recomputed.
function _coordinate_products(statements, views)
    dependent = _gibbs_dependents(statements, (:unconstrained,))
    nodes = NamedTuple[]
    for (position, st) in enumerate(statements)
        name = _assigned_name(st)
        name === nothing && continue
        rhs = st.args[2]
        Meta.isexpr(rhs, :call, 3) && rhs.args[1] === :* || continue
        A, w = rhs.args[2], rhs.args[3]
        A isa Symbol && w isa Symbol && haskey(views, w) && A ∉ dependent || continue
        lo, hi = views[w]
        push!(nodes, (; position, name, matrix=A, vector=w, first=lo, last=hi))
    end
    return nodes
end

function _named_statement(statements, name)
    at = findfirst(st -> _assigned_name(st) === name, statements)
    return at === nothing ? nothing : statements[at]
end

# How a plate's operand reads a vector whose cells it runs over (see
# `_coordinate_operand`); a gather reads it by the name of its key instead.
struct _CoordinateCell end

_is_named_callee(f, name::Symbol) = f === name ||
    (f isa GlobalRef && f.name === name) ||
    (Meta.isexpr(f, :., 2) && f.args[2] isa QuoteNode && f.args[2].value === name)

# How the plate operand `x` reads the block whose vector values are `roots`:
# `_CoordinateCell()` when its element `i` is the vector's element `i` (the
# vector itself, a name for it, or `_ppl_rows` of it, which is the vector when
# the plate has one row per element), `key` when its element `i` is the
# vector's element `key[i]` (a gather by `key`, a port or statement in `named`
# that no coordinate moves, outside `moving`), and `nothing` for any other
# read. `seen` stops a cycle of names.
function _coordinate_operand(statements, x, roots, moving, named, seen = Set{Symbol}())
    if x isa Symbol
        x in roots && return _CoordinateCell()
        x in seen && return nothing
        push!(seen, x)
        st = _named_statement(statements, x)
        return st === nothing ? nothing :
            _coordinate_operand(statements, st.args[2], roots, moving, named, seen)
    end
    if Meta.isexpr(x, :call, 3) && _is_named_callee(x.args[1], :_ppl_rows) &&
            !_gibbs_reads(x.args[3], moving)
        # `_ppl_rows(v, n)` is `v` when `v` holds one Float64 per row, and
        # broadcasts `v` otherwise. Preparation checks the plate's length, so
        # a broadcast single element is no aligned operand.
        return _coordinate_operand(statements, x.args[2], roots, moving, named, seen)
    end
    gather = Meta.isexpr(x, :ref, 2) ? (x.args[1], x.args[2]) :
        Meta.isexpr(x, :call, 3) && _is_named_callee(x.args[1], :_readonly_gather) ?
        (x.args[2], x.args[3]) : nothing
    if gather !== nothing
        source, key = gather
        key isa Symbol && key in named && key ∉ moving || return nothing
        return _coordinate_operand(statements, source, roots, moving, named, seen) isa
            _CoordinateCell ? key : nothing
    end
    return nothing
end

# The plate summed by `term` and how a coordinate of the block reads it, when
# that coordinate moves only some of its cells: the plate's operands that read
# the block all read it the same way (see `_coordinate_operand`), and its cell
# body reads nothing that depends on the block. Returns `(plate,
# _CoordinateCell(), direct)` when the plate runs over the block's cells, so a
# coordinate moves its own cell, `(plate, key, false)` when the plate reads the
# block through a gather by `key`, so a coordinate moves the cells whose key
# holds its index, and `nothing` otherwise. `direct` says every operand reading
# the block is one of its vector values itself, which a cell reads as a number
# at its row; any other read keeps row bookkeeping per call.
function _coordinate_term_plate(statements, term, roots, plates, reached, moving, named)
    term isa Symbol || return nothing
    total = _named_statement(statements, term)
    total === nothing && return nothing
    rhs = total.args[2]
    Meta.isexpr(rhs, :call, 2) && rhs.args[1] === :sum && rhs.args[2] isa Symbol || return nothing
    summed = rhs.args[2]
    summed in roots && return summed in plates ? (summed, _CoordinateCell(), true) : nothing
    producer = _named_statement(statements, summed)
    producer === nothing && return nothing
    source = producer.args[2]
    _is_plate(source) || return nothing
    read, direct = nothing, true
    for argument in source.args[1].args[2:end]
        _gibbs_reads(argument, reached) || continue
        how = _coordinate_operand(statements, argument, roots, moving, named)
        how === nothing && return nothing
        read === nothing || read === how || return nothing
        read = how
        direct &= argument isa Symbol && argument in roots
    end
    read !== nothing && !_gibbs_reads(source.args[2], reached) || return nothing
    return (summed, read, direct)
end

"""
    _coordinate_rows(key, n)

The rows of each element of a vector of `n` elements that a gather by `key`
reads, as a partition index (`plate_partition`'s key): row `i` belongs to
element `key[i]`. Every element `1:n` has an entry, so an element that no row
reads has an empty partition, which sums to zero. Computed from bound data,
once per binding.
"""
function _coordinate_rows(key::AbstractVector{<:Integer}, n::Integer)
    table = Dict{Int,Vector{Int}}(j => Int[] for j in 1:n)
    for (row, j) in enumerate(key)
        rows = get(table, Int(j), nothing)
        rows === nothing && throw(BoundsError(1:n, j))
        push!(rows, row)
    end
    return ReactiveKernels._PartitionIndex(table, length(key))
end

# The same partition index for a plate over the cells of a vector of `n`
# elements: each element's rows are its own cell.
_coordinate_cells(n::Integer) =
    ReactiveKernels._PartitionIndex(Dict{Int,Vector{Int}}(j => [j] for j in 1:n), Int(n))

# The length of a gather key, or -1 when it is no vector of integer indices
# (a `Bool` vector is a mask, not indices). A preparation measures it to
# check the key has one entry per cell.
_coordinate_key_length(key::AbstractVector{<:Integer}) = length(key)
_coordinate_key_length(key::AbstractVector{Bool}) = -1
_coordinate_key_length(key) = -1

# A block kernel with a partition plan port, called without it: the plan of
# the element index (the last argument) is read from `plans`. Each plan holds
# that element's rows of the partitioned plates and their bookkeeping
# (`ReactiveKernels.partition_plan`), computed once at preparation from the
# bound data, so a call does not rebuild them.
struct _PlannedBlockKernel{K,T}
    kernel::K
    plans::Vector{T}
end
@inline (p::_PlannedBlockKernel)(args::Vararg{Any,N}) where {N} =
    p.kernel(args..., p.plans[args[N]])
ReactiveKernels.code_expr(p::_PlannedBlockKernel) = ReactiveKernels.code_expr(p.kernel)

# The plan of each element of `span` for `kernel`, whose arguments are the
# block's (`exemplars`: the point, the products, the coordinate and its previous
# value), the element's index and the plan. A plan holds no value of the
# coordinates, so the exemplar point serves every element.
function _planned_block_kernel(kernel, span, exemplars)
    prepared = ReactiveKernels._kernel_latest_prepared(kernel)
    planner = Base.invokelatest(ReactiveKernels.partition_plan_kernel, prepared)
    pool = ReactiveKernels._PartitionPlanPool()
    leading = exemplars[1:(end - 2)]
    plans = [ReactiveKernels._intern_partition_plan(pool, Base.invokelatest(planner,
        leading..., first(span) + j - 1, last(exemplars), j, nothing)) for j in eachindex(span)]
    types = unique!(Any[typeof(plan) for plan in plans])
    length(types) == 1 || throw(ArgumentError(
        "coordinate conditional: the partition plans have several types: $(join(types, ", "))"))
    return _PlannedBlockKernel(kernel, Vector{only(types)}(plans))
end

# The length of a candidate plate, or -1 when it is no vector of cells.
_coordinate_plate_length(cells::AbstractVector) = length(cells)
_coordinate_plate_length(cells) = -1

"""
    prepare_coordinate_conditional(built, plan, u0)

Prepare the conditional log density of one packed coordinate given all others,
for coordinate-wise transitions such as slice sampling within Gibbs.

A conditional keeps the emitted density terms that read the moved coordinate
and omits the rest, so it equals the joint log density up to a constant. That
constant differs between coordinates: compare values of one coordinate only.

- Each emitted matrix–vector product `A * w` whose vector is a contiguous run
  of packed coordinates and whose matrix does not depend on them is read from
  a cached value. A trial then costs one column of `A` instead of the whole
  product. Products over a transformed or reshaped vector are recomputed.
- A coordinate of a vector parameter evaluates only the terms that read that
  vector. A term that sums a plate over the vector's cells, such as its prior,
  its Jacobian or an `@plate for j in eachindex(y)` likelihood reading
  `theta[j]`, evaluates the moved coordinate's cell alone. A term that sums a
  plate reading the vector through a bound index vector, as
  `y .~ Normal.(theta[g], 1)` does, evaluates only the rows whose index is the
  moved coordinate's (none when no row has it).
- A scalar parameter read at a known packed position evaluates only the
  terms that read it, including through transforms and derived values.
  Unrecognized packed reads use the general conditional.

Use [`coordinate_state`](@ref) to hold a point with its cached products,
[`coordinate_logdensity`](@ref) for a trial value and [`set_coordinate!`](@ref)
to move. Prepare once per binding; states are independent of each other.
Models with free integer parameters are not supported yet.
"""
function prepare_coordinate_conditional(built, plan::StructuralPlan, u0::AbstractVector;
        on_error=nothing)
    built = _binding_build(built, plan)
    _check_built_program(built, plan)
    _has_integer_parameters(plan) && throw(ArgumentError(
        "prepare_coordinate_conditional does not support free integer parameters yet"))
    layout = built.layout
    u = Vector{Float64}(u0)
    length(u) == layout.total || throw(DimensionMismatch(
        "coordinate conditional position has the wrong dimension"))

    sites = _DensitySite[]
    def = Base.invokelatest(_kernel_expr_latest, plan, layout; density_sites=sites)
    def = _elide_named_location_aliases(plan, def)
    def = _pointwise_arithmetic_plates(plan, def)
    def = _name_plate_cells(def, _location_value_aliases(plan, def))
    statements = Expr[s for s in def.args[2].args if s isa Expr && s.head !== :return]
    terms = Any[]
    for name in (:prior, :likelihood, :log_jacobian)
        st = only(s for s in statements if _assigned_name(s) === name)
        append!(terms, _gibbs_sum_terms(st.args[2]))
    end
    dependent = _gibbs_dependents(statements, (:unconstrained,))
    selected = filter(t -> _gibbs_reads(t, dependent), terms)
    readers = _packed_readers(statements)
    nodes = _coordinate_products(statements, readers.views)
    products = Expr(:tuple, (node.name for node in nodes)...)
    bound = _query_bound(plan)

    # The program's ports and values: the names a gather key may take.
    named = Set{Symbol}(name for name in map(_assigned_name, statements) if name !== nothing)
    for argument in def.args[1].args[2:end]
        Meta.isexpr(argument, :(::)) && (argument = argument.args[1])
        argument isa Symbol && push!(named, argument)
    end
    # Vector blocks: packed ranges no other read can reach. Each term that
    # reads a block's vector is classified from the program; one preparation
    # of the emitted program then measures, on this binding, the ranges, the
    # length of each candidate plate and of each gather key. Anything
    # unrecognized keeps every coordinate on the general conditional below.
    found = [begin
        roots = Set{Symbol}(entry.second)
        reached = _gibbs_dependents(statements, roots)
        [(term, _coordinate_term_plate(statements, term, roots, readers.plates, reached,
            dependent, named)) for term in terms if _gibbs_reads(term, reached)]
    end for entry in readers.ranges]
    candidates = unique!([how for entries in found for (_, how) in entries if how !== nothing])
    gather_keys = unique!(Symbol[how[2] for how in candidates if how[2] isa Symbol])
    measured = Expr(:tuple, (x for entry in readers.ranges for x in entry.first)...,
        (:($(GlobalRef(@__MODULE__, :_coordinate_plate_length))($(first(how))))
            for how in candidates)...,
        (:($(GlobalRef(@__MODULE__, :_coordinate_key_length))($key)) for key in gather_keys)...)

    # The emitted program with the values this preparation reads once: the
    # cached products at a point, and the measurements above.
    whole_spec = _eval_kernel_def(Expr(:(=), def.args[1], Expr(:block, statements...,
        (isempty(nodes) ? () : (:(_ppl_cg_products = $products),))...,
        (isempty(readers.ranges) ? () : (:(_ppl_cg_measured = $measured),))...,
        :(return posterior))))
    _register_packed_check!(whole_spec, layout)
    whole(want) = Base.invokelatest(_prepare_cached, plan, whole_spec;
        have=_query_have(plan), want, bound, exemplars=(u,), on_error)

    blocks = NamedTuple[]
    spans = UnitRange{Int}[]
    if readers.recognized && !isempty(readers.ranges)
        values = Base.invokelatest(whole(:_ppl_cg_measured), u)
        r = 2length(readers.ranges)
        spans = [Int(values[2i - 1]):Int(values[2i]) for i in eachindex(readers.ranges)]
        plate_length = Dict(first(how) => Int(values[r + c]) for (c, how) in enumerate(candidates))
        key_length = Dict(key => Int(values[r + length(candidates) + c])
            for (c, key) in enumerate(gather_keys))
        for (i, entry) in enumerate(readers.ranges)
            span = spans[i]
            isempty(span) && continue
            any(in(span), keys(readers.positions)) && continue
            all(j -> j == i || isempty(intersect(span, spans[j])), eachindex(spans)) || continue
            cells, direct, parts, rest = Symbol[], Symbol[], NamedTuple[], Any[]
            for (term, how) in found[i]
                summed, read, at_row = how === nothing ? (nothing, nothing, false) : how
                if read isa _CoordinateCell && plate_length[summed] == length(span)
                    push!(cells, summed)
                    at_row && push!(direct, summed)
                elseif read isa Symbol && plate_length[summed] >= 0 &&
                        key_length[read] == plate_length[summed]
                    push!(parts, (; term, plate=summed, key=read))
                else
                    push!(rest, term)
                end
            end
            push!(blocks, (; span, bounds=entry.first, cells, direct, parts, rest,
                partial=Symbol(:_ppl_cg_partial_, i)))
        end
    end
    scalars = _coordinate_scalars(statements, selected, readers, spans, layout)

    ports = Symbol[Symbol(:_ppl_cg_product_, i) for i in eachindex(nodes)]
    derived = copy(statements)
    for (node, port) in zip(nodes, ports)
        m = :(_ppl_cg_index - ($(node.first) - 1))
        # A coordinate outside this product's vector leaves the product as cached.
        derived[node.position] = :($(node.name)::Vector{Float64} =
            ($(node.first) <= _ppl_cg_index) & (_ppl_cg_index <= $(node.last)) ?
            $port .+ ($(node.vector)[$m] - _ppl_cg_previous) .*
                Base.view($(node.matrix), Base.Colon(), $m) :
            $port)
    end
    signature = copy(def.args[1])
    append!(signature.args, [:($port::Vector{Float64}) for port in ports])
    push!(signature.args, :(_ppl_cg_index::Int), :(_ppl_cg_previous::Float64))
    body = Expr(:block, derived...,
        :(_ppl_cg_density::Float64 = $(_sum_terms(selected))),
        (:($(s.name)::Float64 = $(_sum_terms(s.terms))) for s in scalars)...,
        (isempty(nodes) ? () : (:(_ppl_cg_products = $products),))...,
        :(return _ppl_cg_density))
    spec = _eval_kernel_def(Expr(:(=), signature, body))
    _register_packed_check!(spec, layout)
    have = (_query_have(plan)..., ports..., :_ppl_cg_index, :_ppl_cg_previous)
    exemplars = (u, (Float64[] for _ in nodes)..., 1, 0.0)
    cut(want) = Base.invokelatest(_prepare_cached, plan, spec;
        have, want, bound, exemplars, on_error)
    conditional = cut(:_ppl_cg_density)
    # A program without such a product has nothing to cache or to bring along.
    update, refresh = isempty(nodes) ? (Returns(()), Returns(())) :
        (cut(:_ppl_cg_products), whole(:_ppl_cg_products))

    prepared = map(blocks) do block
        # Inside a block the moved coordinate always lies in the block's own
        # vector: its products take the column update without a branch, so
        # the update composes with its reader, and every other product is
        # the cached value itself.
        own = copy(statements)
        for (node, port) in zip(nodes, ports)
            m = :(_ppl_cg_index - ($(node.first) - 1))
            own[node.position] = (node.first, node.last) == block.bounds ?
                :($(node.name)::Vector{Float64} = $port .+
                    ($(node.vector)[$m] - _ppl_cg_previous) .*
                    Base.view($(node.matrix), Base.Colon(), $m)) :
                :($(node.name)::Vector{Float64} = $port)
        end
        lo, hi = block.bounds
        n = :($hi - $lo + 1)
        # The plates a coordinate reads at its own element's rows: a plate over
        # the vector's cells at its own cell (each element owns its cell), a
        # gathered plate at the rows its key gives the element.
        aligned = [(; plate, rows=:($(GlobalRef(@__MODULE__, :_coordinate_cells))($n)))
            for plate in block.cells if plate ∉ block.direct]
        gathered = [(; part.plate, rows=:($(GlobalRef(@__MODULE__, :_coordinate_rows))(
            $(part.key), $n))) for part in block.parts]
        # `celled` plates are read as `plate_cell`s at the element's index (the
        # plates reading the vector directly, which a cell reads as numbers at
        # one row), the `members` as one partition labelled by that index,
        # whose row bookkeeping per label is a plan computed once per label.
        function build(celled, members, rest)
            rows = [Symbol(block.partial, :_rows_, c) for c in eachindex(members)]
            blockspec = _eval_kernel_def(Expr(:(=), signature, Expr(:block, own...,
                (:($name = $(member.rows)) for (name, member) in zip(rows, members))...,
                :($(block.partial)::Float64 = $(_sum_terms(rest))),
                :(return $(block.partial)))))
            _register_packed_check!(blockspec, layout)
            # Every cell and the partition read the moved element's index in
            # its vector, so a plate they share runs at that index once.
            index = Symbol(block.partial, :_index)
            cellspec, names = blockspec, Symbol[]
            for (c, summed) in enumerate(celled)
                push!(names, Symbol(block.partial, :_cell_, c))
                cellspec = ReactiveKernels.plate_cell(cellspec, summed; index, name=last(names))
            end
            planned = ()
            if !isempty(members)
                push!(names, Symbol(block.partial, :_parts))
                planned = (Symbol(block.partial, :_plan),)
                cellspec = ReactiveKernels.plate_partition(cellspec,
                    Tuple(member.plate for member in members);
                    by=Dict(member.plate => rows[c] for (c, member) in enumerate(members)),
                    index, name=last(names), locality=:error, plan=only(planned))
            end
            indexed = isempty(names) ? () : (index,)
            kernel = Base.invokelatest(_prepare_cached, plan, cellspec;
                have=(have..., indexed..., planned...), want=(block.partial, names...),
                bound, exemplars=(exemplars..., (1 for _ in indexed)...,
                    (nothing for _ in planned)...), on_error)
            isempty(members) || (kernel = _planned_block_kernel(kernel, block.span, exemplars))
            # The block names the plates read at the element's rows: its
            # aligned plates, and the gathered ones read through the partition.
            CoordinateBlock(block.span, kernel, Tuple(block.cells),
                Tuple(member.plate for member in members if member.plate ∉ block.cells))
        end
        # A partition that would compute a value it reads at rows whole costs
        # as much as reading the term whole: read the aligned plates as cells
        # instead, and failing that the gathered terms whole.
        attempts = Any[(block.direct, [aligned; gathered], block.rest)]
        isempty(aligned) || push!(attempts, (block.cells, gathered, block.rest))
        isempty(gathered) || push!(attempts,
            (block.cells, NamedTuple[], Any[block.rest; [part.term for part in block.parts]]))
        for (a, attempt) in enumerate(attempts)
            try
                return build(attempt...)
            catch err
                err isa ReactiveKernels.PartitionLocalityError && a < length(attempts) || rethrow()
            end
        end
    end
    joint = prepare_query(built, plan, :sampler; exemplars=(u,), on_error)
    scalar_blocks = Tuple(CoordinateBlock(s.k:s.k, cut(s.name), (), ()) for s in scalars)
    return CoordinateConditional(layout, joint, conditional, update, refresh,
        Symbol[node.name for node in nodes], Tuple(prepared), scalar_blocks)
end

"One point with its cached products, returned by [`coordinate_state`](@ref)."
mutable struct CoordinateState{P,T<:Tuple}
    const problem::P
    const position::Vector{Float64}
    products::T
end

"""
    coordinate_state(problem, u0)

Start a state at the packed point `u0`. The state owns a copy of the point and
its cached products; the joint log density must be finite there.
"""
function coordinate_state(problem::CoordinateConditional, u0::AbstractVector)
    position = Vector{Float64}(u0)
    length(position) == problem.layout.total || throw(DimensionMismatch(
        "coordinate state position has the wrong dimension"))
    isfinite(problem.joint(position)) || throw(ArgumentError(
        "coordinate state needs a finite joint density at its start"))
    return CoordinateState(problem, position, map(copy, problem.refresh(position)))
end

@inline _coordinate_density(::Tuple{}, problem::CoordinateConditional, u, products, k, previous) =
    _coordinate_density(problem.scalars, problem.conditional, u, products, k, previous)
@inline _coordinate_density(::Tuple{}, kernel, u, products, k, previous) =
    kernel(u, products..., k, previous)
@inline function _coordinate_density(blocks::Tuple, fallback, u, products, k, previous)
    block = first(blocks)
    k in block.range && return _block_density(block, u, products, k, previous)
    return _coordinate_density(Base.tail(blocks), fallback, u, products, k, previous)
end
# The kernel reads the moved element's index in the block's vector, unless the
# block has neither cells nor partitions.
_block_density(block::CoordinateBlock{K,Tuple{},Tuple{}}, u, products, k, previous) where {K} =
    sum(block.kernel(u, products..., k, previous))
_block_density(block::CoordinateBlock, u, products, k, previous) =
    sum(block.kernel(u, products..., k, previous, k - first(block.range) + 1))

"""
    coordinate_logdensity(state, k, z)

Conditional log density of coordinate `k` at the value `z`, with every other
coordinate at the state's point. The state is left unchanged. Values are
comparable for one `k` only.
"""
function coordinate_logdensity(state::CoordinateState, k::Integer, z::Real)
    u = state.position
    checkbounds(u, k)
    previous = u[k]
    u[k] = z
    try
        problem = state.problem
        return _coordinate_density(problem.blocks, problem, u, state.products, Int(k), previous)
    finally
        u[k] = previous
    end
end

"""
    set_coordinate!(state, k, z)

Move coordinate `k` to `z` and update each cached product by one column. If
the update throws, the state keeps its point.
"""
function set_coordinate!(state::CoordinateState, k::Integer, z::Real)
    u = state.position
    checkbounds(u, k)
    previous = u[k]
    u[k] = z
    state.products = try
        state.problem.update(u, state.products..., Int(k), previous)
    catch
        u[k] = previous
        rethrow()
    end
    return state
end

"""
    refresh_products!(state)

Recompute every cached product from the state's point and return the largest
absolute change, the rounding drift accumulated by the column updates.
"""
function refresh_products!(state::CoordinateState)
    fresh = map(copy, state.problem.refresh(state.position))
    drift = 0.0
    for (old, new) in zip(state.products, fresh)
        isempty(new) || (drift = max(drift, maximum(abs, old .- new)))
    end
    state.products = fresh
    return drift
end

# Slice updates of one coordinate, after Neal (2003), "Slice sampling", Annals
# of Statistics 31(3): stepping out by doubling (Fig. 4), then shrinkage
# (Fig. 5) with the acceptance test that keeps doubling reversible (Fig. 6).
# `f` is a log density of the coordinate up to a constant; a point whose value
# is not above the slice level, NaN included, lies outside the slice.

# An Exponential(1) draw from `rand` alone: `1 - rand(rng)` lies in (0, 1].
_slice_exponential(rng) = -log1p(-rand(rng))

"""
    _slice_doubling(rng, f, x0, fx0, w, p) -> (x1, evaluations)

One slice update from `x0`, where `fx0 == f(x0)` is finite: an interval of
width `w` placed at random around `x0`, doubled at most `p` times while either
end is in the slice, then shrunk towards `x0` until an acceptable point is
drawn. `evaluations` counts the calls of `f` this update made.
"""
function _slice_doubling(rng, f, x0::Float64, fx0::Float64, w::Float64, p::Int)
    level = fx0 - _slice_exponential(rng)
    left = x0 - w * rand(rng)
    right = left + w
    fleft, fright = Float64(f(left)), Float64(f(right))
    evaluations = 2
    doublings = p
    while doublings > 0 && (level < fleft || level < fright)
        if rand(rng) < 0.5
            left -= right - left
            fleft = Float64(f(left))
        else
            right += right - left
            fright = Float64(f(right))
        end
        evaluations += 1
        doublings -= 1
    end
    lo, hi = left, right
    while true
        x1 = lo + rand(rng) * (hi - lo)
        fx1 = Float64(f(x1))
        evaluations += 1
        if level < fx1
            accepted, extra = _slice_acceptable(f, x0, x1, level, w, left, right, fleft, fright)
            evaluations += extra
            accepted && return x1, evaluations
        end
        x1 < x0 ? (lo = x1) : (hi = x1)
    end
end

# Fig. 6: `x1` is acceptable unless some halving of the doubled interval that
# separates `x0` from `x1` has both ends outside the slice, in which case the
# doubling could not have reached the interval from `x1`. Midpoints are
# evaluated only once such a separation has occurred.
function _slice_acceptable(f, x0, x1, level, w, left, right, fleft, fright)
    evaluations = 0
    separated = false
    known_left = known_right = true
    while right - left > 1.1 * w
        middle = (left + right) / 2
        separated |= (x0 < middle) != (x1 < middle)
        if x1 < middle
            right, known_right = middle, false
        else
            left, known_left = middle, false
        end
        if separated
            known_left || (fleft = Float64(f(left)); known_left = true; evaluations += 1)
            known_right || (fright = Float64(f(right)); known_right = true; evaluations += 1)
            !(level < fleft) && !(level < fright) && return false, evaluations
        end
    end
    return true, evaluations
end

_slice_width(w::Real, k) = Float64(w)
_slice_width(w::AbstractVector, k) = Float64(w[k])

function _check_slice_settings(width, max_doublings)
    widths = width isa AbstractVector ? width : (width,)
    all(w -> w isa Real && isfinite(w) && w > 0, widths) || throw(ArgumentError(
        "slice widths must be positive and finite"))
    max_doublings >= 0 || throw(ArgumentError("max_doublings must be nonnegative"))
    return nothing
end

"""
    coordinate_slice_step!(rng, state, k; width=10.0, max_doublings=20)

Update coordinate `k` of a [`coordinate_state`](@ref) by one slice-sampling
transition of its conditional (Neal 2003: stepping out by doubling, shrinkage
and the acceptance test for doubled intervals), then move it with
[`set_coordinate!`](@ref). The transition leaves the conditional of `k`, and
so the joint law, invariant. `width` is the initial interval width and
`max_doublings` bounds the stepping out; the defaults are those of the slice
sampler in Pigeons.jl.

`rng` is any random number generator accepted by `rand`. Returns the number of
conditional density evaluations, including the one at the current value.
"""
function coordinate_slice_step!(rng, state::CoordinateState, k::Integer;
        width::Real=10.0, max_doublings::Integer=20)
    _check_slice_settings(width, max_doublings)
    return _coordinate_slice_step!(rng, state, Int(k), Float64(width), Int(max_doublings))
end

function _coordinate_slice_step!(rng, state::CoordinateState, k::Int, w::Float64, p::Int)
    checkbounds(state.position, k)
    x0 = state.position[k]
    f = z -> coordinate_logdensity(state, k, z)
    fx0 = Float64(f(x0))
    isfinite(fx0) || throw(ArgumentError(
        "coordinate $k has a non-finite conditional log density at the current point"))
    x1, evaluations = _slice_doubling(rng, f, x0, fx0, w, p)
    set_coordinate!(state, k, x1)
    return evaluations + 1
end

"""
    coordinate_slice_sweep!(rng, state; width=10.0, max_doublings=20)

One [`coordinate_slice_step!`](@ref) of every packed coordinate, in order.
`width` is one width for all coordinates or a vector with one per coordinate.
Returns the number of conditional density evaluations.
"""
function coordinate_slice_sweep!(rng, state::CoordinateState;
        width::Union{Real,AbstractVector{<:Real}}=10.0, max_doublings::Integer=20)
    _check_slice_settings(width, max_doublings)
    width isa AbstractVector && length(width) != length(state.position) && throw(DimensionMismatch(
        "one slice width per coordinate is needed"))
    evaluations = 0
    for k in eachindex(state.position)
        evaluations += _coordinate_slice_step!(rng, state, k, _slice_width(width, k), Int(max_doublings))
    end
    return evaluations
end
