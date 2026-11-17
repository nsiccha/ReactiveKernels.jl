# Conditionals of one packed coordinate. Changing one coefficient `w[m]` moves
# a product `A * w` by `(w[m] - previous) .* A[:, m]`, so the conditional reads
# each such product from a cached value instead of recomputing it. The planner
# applies no algebraic identities; this file states that one for the emitted
# program. Everything downstream of the product is the emitted program.
#
# A coordinate of a vector parameter also needs only the density terms that
# read that vector, and of a sum over the vector's cells only its own cell.

"Prepared coordinate conditionals returned by [`prepare_coordinate_conditional`](@ref)."
struct CoordinateConditional{L,J,C,U,R,B<:Tuple}
    layout::L
    joint::J
    conditional::C
    update::U
    refresh::R
    products::Vector{Symbol}
    blocks::B
end

# The conditional of the coordinates in `range`: the terms that read them,
# with each plate in `cells` read at the moved coordinate's cell only.
struct CoordinateBlock{K,N}
    range::UnitRange{Int}
    kernel::K
    cells::NTuple{N,Symbol}
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
# range (a view, or a plate over a view) or at literal positions. `recognized`
# is false when some statement reads them in another way.
function _packed_readers(statements)
    ranges = Pair{Tuple{Any,Any},Vector{Symbol}}[]
    views, plates, positions = Dict{Symbol,Tuple{Any,Any}}(), Set{Symbol}(), Int[]
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
            _packed_scalar_reads!(positions, rhs) || (recognized = false)
            continue
        end
        at = findfirst(entry -> entry.first == range, ranges)
        at === nothing ? push!(ranges, range => Symbol[name]) : push!(ranges[at].second, name)
    end
    return (; ranges, views, plates, positions, recognized)
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

# The plate summed by `term`, when a coordinate of the block moves exactly one
# of its cells: the plate runs over the block's cells, reads the block only
# through operands aligned with it, and captures nothing that depends on it.
function _coordinate_cell_plate(statements, term, roots, plates, dependent)
    term isa Symbol || return nothing
    total = _named_statement(statements, term)
    total === nothing && return nothing
    rhs = total.args[2]
    Meta.isexpr(rhs, :call, 2) && rhs.args[1] === :sum && rhs.args[2] isa Symbol || return nothing
    summed = rhs.args[2]
    summed in roots && return summed in plates ? summed : nothing
    producer = _named_statement(statements, summed)
    producer === nothing && return nothing
    source = producer.args[2]
    _is_plate(source) || return nothing
    aligned = false
    for argument in source.args[1].args[2:end]
        if argument isa Symbol && argument in roots
            aligned = true
        elseif _gibbs_reads(argument, dependent)
            return nothing
        end
    end
    aligned && !_gibbs_reads(source.args[2], dependent) || return nothing
    return summed
end

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
  vector. A term that sums a plate over the vector's cells, such as its prior
  or Jacobian, evaluates the moved coordinate's cell alone.
- Every other coordinate evaluates all terms that read any coordinate.

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

    # The emitted program with the values this preparation reads once: the
    # cached products at a point, the packed ranges and candidate plates.
    bounds = Expr(:tuple, (x for entry in readers.ranges for x in entry.first)...)
    whole_spec = _eval_kernel_def(Expr(:(=), def.args[1], Expr(:block, statements...,
        (isempty(nodes) ? () : (:(_ppl_cg_products = $products),))...,
        (isempty(readers.ranges) ? () : (:(_ppl_cg_ranges = $bounds),))...,
        :(return posterior))))
    _register_packed_check!(whole_spec, layout)
    whole(want) = Base.invokelatest(_prepare_cached, plan, whole_spec;
        have=_query_have(plan), want, bound, exemplars=(u,), on_error)

    # Vector blocks: packed ranges no other read can reach, measured on this
    # binding. Anything unrecognized keeps every coordinate on the general
    # conditional below.
    blocks = NamedTuple[]
    if readers.recognized && !isempty(readers.ranges)
        limits = whole(:_ppl_cg_ranges)(u)
        spans = [Int(limits[2i - 1]):Int(limits[2i]) for i in eachindex(readers.ranges)]
        for (i, entry) in enumerate(readers.ranges)
            span = spans[i]
            isempty(span) && continue
            any(in(span), readers.positions) && continue
            all(j -> j == i || isempty(intersect(span, spans[j])), eachindex(spans)) || continue
            roots = Set{Symbol}(entry.second)
            reached = _gibbs_dependents(statements, roots)
            cells, rest = Symbol[], Any[]
            for term in filter(t -> _gibbs_reads(t, reached), terms)
                summed = _coordinate_cell_plate(statements, term, roots, readers.plates, reached)
                if summed !== nothing
                    value = whole(summed)(u)
                    value isa AbstractVector && length(value) == length(span) || (summed = nothing)
                end
                summed === nothing ? push!(rest, term) : push!(cells, summed)
            end
            push!(blocks, (; span, bounds=entry.first, cells, rest,
                partial=Symbol(:_ppl_cg_partial_, i)))
        end
    end

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
        blockspec = _eval_kernel_def(Expr(:(=), signature, Expr(:block, own...,
            :($(block.partial)::Float64 = $(_sum_terms(block.rest))),
            :(return $(block.partial)))))
        _register_packed_check!(blockspec, layout)
        cellspec, names, indices = blockspec, Symbol[], Symbol[]
        for (c, summed) in enumerate(block.cells)
            push!(names, Symbol(block.partial, :_cell_, c))
            push!(indices, Symbol(block.partial, :_index_, c))
            cellspec = ReactiveKernels.plate_cell(cellspec, summed;
                index=last(indices), name=last(names))
        end
        kernel = Base.invokelatest(_prepare_cached, plan, cellspec;
            have=(have..., indices...), want=(block.partial, names...), bound,
            exemplars=(exemplars..., (1 for _ in indices)...), on_error)
        CoordinateBlock(block.span, kernel, Tuple(block.cells))
    end
    joint = prepare_query(built, plan, :sampler; exemplars=(u,), on_error)
    return CoordinateConditional(layout, joint, conditional, update, refresh,
        Symbol[node.name for node in nodes], Tuple(prepared))
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

@inline _coordinate_density(::Tuple{}, problem, u, products, k, previous) =
    problem.conditional(u, products..., k, previous)
@inline function _coordinate_density(blocks::Tuple, problem, u, products, k, previous)
    block = first(blocks)
    k in block.range && return _block_density(block, u, products, k, previous)
    return _coordinate_density(Base.tail(blocks), problem, u, products, k, previous)
end
_block_density(block::CoordinateBlock{K,N}, u, products, k, previous) where {K,N} =
    sum(block.kernel(u, products..., k, previous,
        ntuple(_ -> k - first(block.range) + 1, Val(N))...))

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
